"""Local restore-plan controls; no AWS or production database I/O."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import aws_subnet_restore as restore
from scripts import aws_subnet_snapshot as snapshot


def empty_evidence():
    roles = [dict(name=n, super=False, inherit=n not in restore.NOINHERIT,
                  createrole=False, createdb=False, login=False, replication=False,
                  bypassrls=False, config=None, can_set=True, can_inherit=True) for n in restore.ROLES]
    return dict(format=1, host=snapshot.TARGET_HOST,
                control=dict(role="arena_admin", database="postgres", version=170009, recovery=False, other_sessions=0, unexpected_schemas=[]),
                metadata={key: [] for key in restore.OBJECT_KINDS}, roles=roles, memberships=[],
                available_extensions=[dict(name=n) for n in snapshot.EXTENSIONS])


class RestorePlanTests(unittest.TestCase):
    def test_target_guard_rejects_identity_existing_objects_sessions_and_unsafe_roles(self):
        evidence = empty_evidence()
        self.assertEqual(restore.target_blockers(evidence), [])
        evidence["memberships"] = [dict(name=snapshot.SOURCE_ROLE, member="arena_admin", admin=True, inherit=False, set=False)]
        self.assertEqual(restore.target_blockers(evidence), [])
        for flag, unsafe in (("admin", False), ("inherit", True), ("set", True)):
            changed = copy.deepcopy(evidence)
            changed["memberships"][0][flag] = unsafe
            self.assertIn("temporary-role-membership", restore.target_blockers(changed))
        duplicated = copy.deepcopy(evidence)
        duplicated["memberships"] *= 2
        self.assertIn("temporary-role-membership", restore.target_blockers(duplicated))
        for key, value in (("role", "postgres"), ("database", "other"), ("version", 160000), ("recovery", True)):
            wrong = copy.deepcopy(evidence)
            wrong["control"][key] = value
            with self.assertRaises(snapshot.Failure):
                restore.target_blockers(wrong)
        evidence["metadata"]["relations"] = [{"name": "existing_data"}]
        evidence["control"]["other_sessions"] = 1
        evidence["control"]["unexpected_schemas"] = ["unrelated"]
        evidence["roles"][-1]["login"] = True
        evidence["memberships"] = [dict(name=snapshot.SOURCE_ROLE, member="arena_admin")]
        evidence["available_extensions"] = []
        blockers = restore.target_blockers(evidence)
        for expected in ("target-has-user-objects-or-default-acls", "target-has-other-sessions",
                         "target-has-unexpected-user-schemas",
                         "unsafe-or-unreviewed-role:" + snapshot.SOURCE_ROLE,
                         "temporary-role-membership", "unavailable-extension:http"):
            self.assertIn(expected, blockers)

    def test_toc_exclusions_require_exact_id_hash_and_safe_type(self):
        lines = ["1; 3079 100 EXTENSION - http ", "2; 0 0 COMMENT - EXTENSION http ",
                 "3; 0 100 ACL public TABLE t postgres", "4; 0 100 TABLE DATA source_add_history history postgres"]
        toc = ("; header\n" + "\n".join(lines) + "\n").encode()
        exclusion = dict(id="1", sha256=hashlib.sha256(lines[0].encode()).hexdigest(), reason="reviewed unavailable extension")
        selected, entries = restore.select_toc(toc, [exclusion])
        self.assertIn(b"; excluded 1;", selected)
        self.assertIn(lines[2].encode(), selected)
        self.assertIn(lines[3].encode(), selected)
        for item in ({**exclusion, "sha256": "0" * 64}, {**exclusion, "reason": ""},
                     dict(id="3", sha256=hashlib.sha256(lines[2].encode()).hexdigest(), reason="unsafe ACL skip"),
                     dict(id="4", sha256=hashlib.sha256(lines[3].encode()).hexdigest(), reason="unsafe history skip")):
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc, [item])
        self.assertEqual(len(entries), 4)

    def test_native_candidate_preserves_owners_acls_and_uses_one_transaction(self):
        command = restore.native_restore_command(Path("reviewed.list"), Path("archive"))
        self.assertIn("--single-transaction", command)
        self.assertIn("--exit-on-error", command)
        for unsafe in ("--no-owner", "--no-acl", "--no-privileges", "--clean", "--jobs"):
            self.assertFalse(any(arg.startswith(unsafe) for arg in command))

    def test_metadata_difference_reports_only_category(self):
        expected = {key: [] for key in restore.OBJECT_KINDS}
        actual = copy.deepcopy(expected)
        actual["routines"] = [dict(definition="private routine SQL")]
        self.assertEqual(restore.compare_metadata(expected, actual), ["routines"])

    def test_no_execute_mode_and_errors_are_sanitized(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit):
            restore.main(["execute"])
        self.assertIn("invalid choice", output.getvalue())
        previous_umask = os.umask(0o077)
        try:
            with patch.object(restore, "inventory", side_effect=ValueError("private SQL and row")), contextlib.redirect_stderr(output):
                code = restore.main(["inventory", "--target-secret-arn", "unused", "--output", "unused"])
        finally:
            os.umask(previous_umask)
        self.assertEqual(code, 1)
        self.assertNotIn("private SQL and row", output.getvalue())


PG17 = Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_RESTORE_LOCAL_PG17") == "1" and PG17.is_dir(),
                     "opt-in two disposable local PG17 clusters")
class NativeRestoreTests(unittest.TestCase):
    def test_native_owner_acl_rls_views_constraints_metadata_and_archive_sequences(self):
        with tempfile.TemporaryDirectory(prefix="lp-restore-", dir="/tmp") as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir(mode=0o700)
            base = {**snapshot.pg_environment(), "PATH": str(PG17) + os.pathsep + os.environ["PATH"]}
            def run(args, data=None, env=base):
                return subprocess.run(args, input=data, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
            @contextlib.contextmanager
            def cluster(name, user):
                cluster_root = root / name
                cluster_root.mkdir(mode=0o700)
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                bootstrap_user = "local_bootstrap" if user == "arena_admin" else user
                run([str(PG17 / "initdb"), "-D", str(cluster_root / "db"), "-U", bootstrap_user, "--auth=trust", "--no-locale", "--encoding=UTF8"])
                run([str(PG17 / "pg_ctl"), "-D", str(cluster_root / "db"), "-l", str(cluster_root / "server.log"), "-o", f"-F -h 127.0.0.1 -p {port} -k {cluster_root}", "-w", "start"])
                admin = [str(PG17 / "psql"), "-XqAt", "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1", "-p", str(port), "-U", user, "-d", "postgres"]
                try:
                    if user == "arena_admin":
                        bootstrap = admin.copy()
                        bootstrap[bootstrap.index("-U") + 1] = bootstrap_user
                        run(bootstrap, data=b"CREATE ROLE arena_admin LOGIN NOSUPERUSER CREATEROLE CREATEDB INHERIT NOBYPASSRLS; ALTER DATABASE postgres OWNER TO arena_admin;")
                    role_sql = "\n".join("CREATE ROLE " + snapshot.ident(n) + " NOLOGIN " + ("NOINHERIT" if n in restore.NOINHERIT else "INHERIT") + " NOBYPASSRLS;" for n in restore.ROLES if n != "pg_database_owner")
                    run(admin, data=role_sql.encode())
                    if user == "arena_admin":
                        run(admin, data=b"GRANT postgres,lab_arena_owner TO arena_admin WITH INHERIT TRUE, SET TRUE; GRANT CREATE ON DATABASE postgres TO postgres,lab_arena_owner;")
                        self.assertEqual(run(admin, data=b"SELECT rolsuper,rolcreaterole,rolcreatedb FROM pg_roles WHERE rolname=current_user;").strip(), b"f|t|t")
                    service = cluster_root / "service"
                    snapshot.private_write(service, (f"[migration]\nhost=127.0.0.1\nport={port}\nuser={user}\ndbname=postgres\nsslmode=disable\noptions={snapshot.OPTIONS}\n").encode())
                    read = {**base, "PGSERVICEFILE": str(service)}
                    write_service = cluster_root / "write-service"
                    snapshot.private_write(write_service, service.read_bytes().replace(("options=" + snapshot.OPTIONS + "\n").encode(), b""))
                    yield admin, read, {**base, "PGSERVICEFILE": str(write_service)}
                finally:
                    run([str(PG17 / "pg_ctl"), "-D", str(cluster_root / "db"), "-m", "fast", "-w", "stop"])

            with cluster("source", "local_operator") as (source_admin, source_env, _), cluster("target", "arena_admin") as (target_admin, target_env, target_write):
                sql = "ALTER SCHEMA public OWNER TO postgres; CREATE SCHEMA source_add_history AUTHORIZATION lab_arena_owner; CREATE SCHEMA extensions AUTHORIZATION postgres;\n"
                sql += "CREATE SEQUENCE public.seq_a; CREATE SEQUENCE public.seq_b; CREATE SEQUENCE source_add_history.seq_c; ALTER SEQUENCE public.seq_a OWNER TO postgres; ALTER SEQUENCE public.seq_b OWNER TO postgres; ALTER SEQUENCE source_add_history.seq_c OWNER TO lab_arena_owner;\n"
                for i in range(48):
                    schema, name, owner = ("source_add_history", "history", "lab_arena_owner") if i == 47 else ("public", f"t_{i}", "postgres")
                    sql += f"CREATE TABLE {schema}.{name}(id bigint PRIMARY KEY, val text NOT NULL DEFAULT 'allowed' CHECK(length(val)>0)); ALTER TABLE {schema}.{name} OWNER TO {owner}; INSERT INTO {schema}.{name} VALUES(1,'SOURCE_ADD');\n"
                sql += "ALTER TABLE source_add_history.history ADD CONSTRAINT history_fk FOREIGN KEY(id) REFERENCES public.t_0(id); CREATE INDEX history_val_idx ON source_add_history.history(val); ALTER TABLE public.t_0 ENABLE ROW LEVEL SECURITY; ALTER TABLE public.t_0 FORCE ROW LEVEL SECURITY; CREATE POLICY service_read ON public.t_0 FOR SELECT TO lab_arena_service USING(val='SOURCE_ADD');\n"
                sql += "ALTER SEQUENCE public.seq_a OWNED BY public.t_0.id; ALTER TABLE public.t_0 ALTER COLUMN id SET DEFAULT nextval('public.seq_a');\n"
                sql += "CREATE VIEW public.v_a AS SELECT id FROM public.t_0; CREATE VIEW public.v_b AS SELECT val FROM public.t_1; CREATE VIEW source_add_history.v_c AS SELECT * FROM source_add_history.history; ALTER VIEW public.v_a OWNER TO postgres; ALTER VIEW public.v_b OWNER TO postgres; ALTER VIEW source_add_history.v_c OWNER TO lab_arena_owner;\n"
                sql += "CREATE FUNCTION public.f_0() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$; ALTER FUNCTION public.f_0() OWNER TO postgres; CREATE TRIGGER t_0_audit BEFORE INSERT ON public.t_0 FOR EACH ROW EXECUTE FUNCTION public.f_0();\n"
                for i in range(1, 377):
                    sql += f"CREATE FUNCTION public.f_{i}() RETURNS integer LANGUAGE sql {'SECURITY DEFINER SET search_path=pg_catalog' if i == 1 else ''} AS $$SELECT {i}$$; ALTER FUNCTION public.f_{i}() OWNER TO postgres;\n"
                exact_tables = ",".join([f"public.t_{i}" for i in range(47)] + ["source_add_history.history"])
                sql += "GRANT USAGE ON SCHEMA public,source_add_history,extensions TO " + snapshot.SOURCE_ROLE + "; GRANT SELECT ON TABLE " + exact_tables + " TO " + snapshot.SOURCE_ROLE + "; GRANT SELECT ON ALL SEQUENCES IN SCHEMA public,source_add_history TO " + snapshot.SOURCE_ROLE + "; GRANT SELECT ON public.t_0 TO anon,authenticated,lab_arena_service; GRANT EXECUTE ON FUNCTION public.f_1() TO lab_arena_service; ALTER DEFAULT PRIVILEGES FOR ROLE lab_arena_owner IN SCHEMA source_add_history GRANT SELECT ON TABLES TO lab_arena_service; SELECT setval('public.seq_a',100,true); SELECT setval('public.seq_b',50,false); SELECT setval('source_add_history.seq_c',9,true);"
                run(source_admin, data=sql.encode())
                with snapshot.Keeper(source_env, logs) as keeper:
                    before = snapshot.capture_metadata(keeper)
                    tables = keeper.one_json(snapshot.CATALOG)
                    snapshot.validate_tables(tables)
                    for table in tables:
                        table["count"] = keeper.one_json("SELECT count(*) FROM ONLY " + snapshot.qualified(table) + ";")
                    seqs = [dict(schema="public", name="seq_a"), dict(schema="public", name="seq_b"), dict(schema="source_add_history", name="seq_c")]
                    manifest = dict(tables=tables, sequences=seqs, schema_metadata=before)
                run([str(PG17 / "pg_dump"), "--dbname=service=migration", "-Fd", "--compress=gzip:1", *["--schema=" + s for s in snapshot.SCHEMAS], "--file=" + str(root / "dump")], env=source_env)
                toc = snapshot.command(["pg_restore", "--list", str(root / "dump")], source_env, logs, "local-toc")
                public_entry = next(line for line in toc.decode().splitlines()
                                    if " SCHEMA - public postgres" in line)
                public_exclusion = dict(id=public_entry.split(";", 1)[0], sha256=hashlib.sha256(public_entry.encode()).hexdigest(),
                                        reason="retain initialized public ACLs; explicitly prepare its source owner")
                listing, _ = restore.select_toc(toc, [public_exclusion])
                snapshot.private_write(root / "restore.list", listing)
                plan_dir = root / "plan"
                plan_dir.mkdir(mode=0o700)
                states = restore.sequence_states(toc, root / "dump", source_env, logs, plan_dir)
                # Sequence proof must follow the archive, not this later source state.
                run(source_admin, data=b"SELECT setval('public.seq_a',9999,true);")
                with snapshot.Keeper(target_env, logs) as keeper:
                    empty = restore.target_evidence(keeper)
                self.assertNotIn("target-has-user-objects-or-default-acls", restore.target_blockers(empty))
                self.assertNotIn("temporary-role-membership", restore.target_blockers(empty))
                self.assertTrue(any(m["name"] == snapshot.SOURCE_ROLE and m["member"] == "arena_admin"
                                    and m["admin"] and not m["inherit"] and not m["set"] for m in empty["memberships"]))
                # Exercise the complete local planning path with sealed synthetic
                # export and target evidence. It must remain review-required.
                manifest.update(format=1, source_host=snapshot.SOURCE_HOST, source_role=snapshot.SOURCE_ROLE, database="postgres")
                manifest["archive_files"] = snapshot.validate_archive(root / "dump", manifest, source_env, logs)
                snapshot.private_json(root / "manifest.json", manifest)
                snapshot.private_write(root / "manifest.sha256", (snapshot.sha256_file(root / "manifest.json") + "\n").encode())
                evidence_dir = root / "evidence"
                evidence_dir.mkdir(mode=0o700)
                snapshot.private_json(evidence_dir / "target.json", empty)
                snapshot.private_write(evidence_dir / "target.sha256", (snapshot.sha256_file(evidence_dir / "target.json") + "\n").encode())
                snapshot.private_json(root / "exclusions.json", [public_exclusion])
                args = SimpleNamespace(export_directory=str(root), target_inventory=str(evidence_dir),
                                       output=str(root / "prepared"), exclusions=str(root / "exclusions.json"))
                with patch.dict(os.environ, {"PATH": base["PATH"]}), contextlib.redirect_stdout(io.StringIO()):
                    restore.plan(args)
                prepared = json.loads((root / "prepared" / "plan.json").read_text())
                self.assertFalse(prepared["execution_supported"])
                self.assertEqual(prepared["toc_counts"]["FUNCTION"], 377)
                self.assertEqual(prepared["archive_sequences"], states)
                for path in (root / "prepared").iterdir():
                    if path.is_file():
                        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                # Keep initialized PUBLIC USAGE ACLs and prepare the source owner.
                # Production prerequisite authority still requires live RDS proof.
                run(target_admin, data=b"ALTER SCHEMA public OWNER TO postgres;")
                snapshot.command(restore.native_restore_command(root / "restore.list", root / "dump"), target_write, logs, "local-restore")
                run(target_admin, data=b"REVOKE CREATE ON DATABASE postgres FROM postgres,lab_arena_owner;")
                with snapshot.Keeper(target_env, logs) as keeper:
                    after = snapshot.capture_metadata(keeper)
                    actual = restore.target_evidence(keeper)
                self.assertEqual(before["schemas"], after["schemas"])
                self.assertEqual(restore.compare_metadata(before, after), [])
                self.assertEqual(sorted(actual["sequences"], key=lambda x: (x["schema"], x["name"])), states)
                self.assertIn("target-has-user-objects-or-default-acls", restore.target_blockers(actual))
                self.assertEqual(len(after["routines"]), 377)
                self.assertEqual(len(after["views"]), 3)
                self.assertTrue(any(r["rls"] and r["force_rls"] for r in after["relations"]))
                self.assertEqual(run(target_admin, data=b"SELECT val FROM source_add_history.history;").strip(), b"SOURCE_ADD")
                # Guard fails before any revokes when the temporary role can log in.
                cleanup = restore.export_role_cleanup(manifest)
                run(target_admin, data=("ALTER ROLE " + snapshot.SOURCE_ROLE + " LOGIN;").encode())
                with self.assertRaises(subprocess.CalledProcessError):
                    run(target_admin, data=cleanup.encode())
                self.assertEqual(run(target_admin, data=("SELECT has_table_privilege('" + snapshot.SOURCE_ROLE + "','public.t_0','SELECT');").encode()).strip(), b"t")
                run(target_admin, data=("ALTER ROLE " + snapshot.SOURCE_ROLE + " NOLOGIN;").encode())
                run(target_admin, data=cleanup.encode())
                self.assertEqual(run(target_admin, data=("SELECT count(*) FROM pg_roles WHERE rolname='" + snapshot.SOURCE_ROLE + "';").encode()).strip(), b"0")
                self.assertEqual(run(target_admin, data=b"SELECT has_table_privilege('lab_arena_service','public.t_0','SELECT');").strip(), b"t")
                self.assertEqual(run(target_admin, data=b"SELECT has_function_privilege('lab_arena_service','public.f_1()','EXECUTE');").strip(), b"t")
                with snapshot.Keeper(target_env, logs) as keeper:
                    cleaned = snapshot.capture_metadata(keeper)
                approved = copy.deepcopy(before)
                for items in approved.values():
                    for item in items:
                        if isinstance(item.get("acl"), dict):
                            item["acl"]["items"] = [grant for grant in item["acl"]["items"]
                                                   if not grant.startswith(snapshot.SOURCE_ROLE + "=")]
                self.assertEqual(restore.compare_metadata(approved, cleaned), [])


if __name__ == "__main__":
    unittest.main()
