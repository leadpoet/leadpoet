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


AUTH_ROLE_SUPPORT = {'dependencies': [{'identity': 'auth',
                   'kind': 'n',
                   'name': 'auth',
                   'schema': None,
                   'type': 'schema'}],
 'function': {'acl': '{=X/supabase_auth_admin,supabase_auth_admin=X/supabase_auth_admin,dashboard_user=X/supabase_auth_admin}',
              'arguments': '',
              'config': None,
              'definition': 'CREATE OR REPLACE FUNCTION auth.role()\n'
                            ' RETURNS text\n'
                            ' LANGUAGE sql\n'
                            ' STABLE\n'
                            'AS $function$\n'
                            '  select \n'
                            '  coalesce(\n'
                            "    nullif(current_setting('request.jwt.claim.role', true), ''),\n"
                            "    (nullif(current_setting('request.jwt.claims', true), '')::jsonb "
                            "->> 'role')\n"
                            '  )::text\n'
                            '$function$\n',
              'language': 'sql',
              'leakproof': False,
              'name': 'role',
              'owner': 'supabase_auth_admin',
              'parallel': 'u',
              'result': 'text',
              'schema': 'auth',
              'security_definer': False,
              'sha256': 'c96a9358b02089b03f3962fd85c5f83c9d64f32240481915c0efe74398dc761b',
              'strict': False,
              'volatility': 's'},
 'schema': {'acl': '{supabase_admin=UC/supabase_admin,anon=U/supabase_admin,authenticated=U/supabase_admin,service_role=U/supabase_admin,supabase_auth_admin=UC/supabase_admin,dashboard_user=UC/supabase_admin,postgres=U/supabase_admin}',
            'name': 'auth',
            'owner': 'supabase_admin'}}


def auth_schema_sql():
    return ("CREATE SCHEMA auth AUTHORIZATION supabase_admin; "
            "GRANT USAGE ON SCHEMA auth TO anon,authenticated,service_role,postgres; "
            "GRANT USAGE,CREATE ON SCHEMA auth TO supabase_auth_admin,dashboard_user;")


def auth_function_sql():
    return (AUTH_ROLE_SUPPORT["function"]["definition"]
            + ";\nALTER FUNCTION auth.role() OWNER TO supabase_auth_admin; "
            "SET ROLE supabase_auth_admin; "
            "GRANT EXECUTE ON FUNCTION auth.role() TO PUBLIC,supabase_auth_admin,dashboard_user; RESET ROLE;")


def empty_evidence():
    roles = [dict(name=n, super=False, inherit=n not in restore.NOINHERIT,
                  createrole=False, createdb=False, login=False, replication=False,
                  bypassrls=False, config=None, can_set=True, can_inherit=True) for n in restore.ROLES]
    return dict(format=1, host=snapshot.TARGET_HOST,
                control=dict(role="arena_admin", database="postgres", version=170009, recovery=False, other_sessions=0, unexpected_schemas=[]),
                metadata={key: [] for key in restore.OBJECT_KINDS}, roles=roles, memberships=[],
                available_extensions=[dict(name=n) for n in snapshot.EXTENSIONS])


def http_acl_fixture():
    members, lines, fragments = [], ["1; 3079 100 EXTENSION - http "], []
    for i in range(19):
        name, key = "http_member_" + str(i), str(100 + i)
        members.append(dict(**{"class": "pg_proc"}, kind="function", schema="public",
                            name=name, identity_arguments="data bytea", owner="supabase_admin"))
        lines.append(key + "; 0 0 ACL public FUNCTION " + name + "(data bytea) supabase_admin")
        fragments.append("-- Name: FUNCTION " + name + "(data bytea); Type: ACL; Schema: public; Owner: supabase_admin\n--\n\n"
                         + "".join("GRANT ALL ON FUNCTION public." + name + "(data bytea) TO " + role + ";\n"
                                   for role in ("postgres", "anon", "authenticated", "service_role")) + "\n\n")
    lines += ["200; 0 0 ACL public FUNCTION ordinary() postgres", "201; 0 0 ACL public TYPE http_request supabase_admin"]
    toc = ("\n".join(lines) + "\n").encode()
    proof = dict(capture=dict(sealed_manifest_sha256="f" * 64, sealed_manifest_unchanged=True),
                 evidence=dict(evidence_kind="supplemental-current-source-catalog-not-sealed-snapshot",
                               transaction_read_only="on", search_path="pg_catalog", project="qplwoislplkcegvdmbim",
                               extension=dict(name="http", schema="public", owner="supabase_admin", version="1.6"), members=members))
    review = dict(source_proof=json.dumps(proof).encode(), acl_sql="".join(fragments).encode(), manifest_sha256="f" * 64)
    seals = dict(toc=hashlib.sha256(toc).hexdigest(), manifest="f" * 64,
                 **{k: hashlib.sha256(review[k]).hexdigest() for k in ("source_proof", "acl_sql")})
    exclusions = [dict(id=line.split(";", 1)[0], sha256=hashlib.sha256(line.encode()).hexdigest(), reason="reviewed fixture")
                  for line in lines[:20]]
    return toc, dict(routines=[dict(schema="public", name="ordinary")]), review, seals, exclusions


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

    def test_http_acl_exceptions_are_complete_exact_and_default_denied(self):
        toc, metadata, review, seals, exclusions = http_acl_fixture()
        with self.assertRaises(snapshot.Failure):
            restore.select_toc(toc, exclusions, metadata)
        with patch.dict(restore.HTTP_ACL_REVIEW_SHA256, seals, clear=True):
            selected, _ = restore.select_toc(toc, exclusions, metadata, review)
            self.assertEqual(selected.count(b"; excluded "), 20)
            self.assertIn(b"\n200; 0 0 ACL public FUNCTION ordinary() postgres\n", selected)
            self.assertIn(b"\n201; 0 0 ACL public TYPE http_request supabase_admin\n", selected)
            records = restore.reviewed_http_acls(toc, metadata, review)
            self.assertEqual(len(records), 19)
            self.assertTrue(all(len(r["acl_sql_sha256"]) == 64 for r in records.values()))
            for omitted in (exclusions[1:], exclusions[:-1]):
                with self.assertRaises(snapshot.Failure):
                    restore.select_toc(toc, omitted, metadata, review)
            for key in ("200", "201"):
                line = restore.toc_entries(toc)[key]
                extra = dict(id=key, sha256=hashlib.sha256(line.encode()).hexdigest(), reason="unapproved")
                with self.assertRaises(snapshot.Failure):
                    restore.select_toc(toc, exclusions + [extra], metadata, review)
            for key in ("source_proof", "acl_sql", "manifest_sha256"):
                changed = dict(review)
                changed[key] += b" " if isinstance(changed[key], bytes) else "0"
                with self.assertRaises(snapshot.Failure):
                    restore.select_toc(toc, exclusions, metadata, changed)
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc + b"202; 0 0 ACL public FUNCTION different() postgres\n", exclusions, metadata, review)
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc, [{**exclusions[0], "sha256": "0" * 64}] + exclusions[1:], metadata, review)
            collision = dict(routines=metadata["routines"] + [dict(schema="public", name="http_member_0", arguments="other text")])
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc, exclusions, collision, review)

    def test_http_acl_review_semantics_reject_wrong_identity_extension_and_mixed_sql(self):
        toc, metadata, review, seals, exclusions = http_acl_fixture()
        proof = json.loads(review["source_proof"])
        variants = []
        for key, value in (("version", "1.7"), ("name", "pgcrypto"), ("schema", "extensions")):
            changed = copy.deepcopy(proof)
            changed["evidence"]["extension"][key] = value
            variants.append(dict(review, source_proof=json.dumps(changed).encode()))
        for key, value in (("identity_arguments", "data text"), ("owner", "postgres"), ("kind", "type")):
            changed = copy.deepcopy(proof)
            changed["evidence"]["members"][0][key] = value
            variants.append(dict(review, source_proof=json.dumps(changed).encode()))
        variants.append(dict(review, acl_sql=review["acl_sql"].replace(b"TO postgres;", b"TO postgres;\nGRANT ALL ON FUNCTION public.ordinary() TO anon;", 1)))
        for changed in variants:
            # Re-pin synthetic fixture bytes to exercise the semantic guards too.
            changed_seals = {**seals, **{k: hashlib.sha256(changed[k]).hexdigest() for k in ("source_proof", "acl_sql")}}
            with self.subTest(changed=variants.index(changed)), patch.dict(restore.HTTP_ACL_REVIEW_SHA256, changed_seals, clear=True):
                with self.assertRaises(snapshot.Failure):
                    restore.select_toc(toc, exclusions, metadata, changed)

    def test_public_schema_exclusion_is_bound_to_exact_manifest_owner(self):
        for owner in ("postgres", "pg_database_owner"):
            with self.subTest(owner=owner):
                line = "5; 2615 2200 SCHEMA - public " + owner
                acl = "6; 0 0 ACL - SCHEMA public " + owner
                toc = (line + "\n" + acl + "\n").encode()
                exclusion = dict(id="5", sha256=hashlib.sha256(line.encode()).hexdigest(),
                                 reason="preserve matching initialized schema and native ACL entry")
                metadata = dict(schemas=[dict(schema="public", name="public", owner=owner)])
                selected, _ = restore.select_toc(toc, [exclusion], metadata)
                self.assertIn(("; excluded " + line).encode(), selected)
                self.assertIn(("\n" + acl + "\n").encode(), selected)
                wrong = "pg_database_owner" if owner == "postgres" else "postgres"
                for invalid in (None, dict(schemas=[]),
                                dict(schemas=[dict(schema="public", name="public", owner=wrong)]),
                                dict(schemas=metadata["schemas"] * 2)):
                    with self.assertRaises(snapshot.Failure):
                        restore.select_toc(toc, [exclusion], invalid)
                bad_acl = dict(id="6", sha256=hashlib.sha256(acl.encode()).hexdigest(), reason="unsafe ACL exclusion")
                with self.assertRaises(snapshot.Failure):
                    restore.select_toc(toc, [bad_acl], metadata)
        for owner in ("arena_admin", "unreviewed_owner", "postgres extra"):
            line = "5; 2615 2200 SCHEMA - public " + owner
            exclusion = dict(id="5", sha256=hashlib.sha256(line.encode()).hexdigest(), reason="unreviewed owner")
            with self.assertRaises(snapshot.Failure):
                restore.select_toc((line + "\n").encode(), [exclusion],
                                   dict(schemas=[dict(schema="public", name="public", owner=owner)]))

    def test_extensions_schema_exclusion_preserves_acl_and_requires_exact_owner(self):
        line = "7; 2615 19000 SCHEMA - extensions postgres"
        acl = "8; 0 0 ACL - SCHEMA extensions postgres"
        helper = "9; 1255 19001 FUNCTION extensions pgrst_ddl_watch() supabase_admin"
        toc = (line + "\n" + acl + "\n" + helper + "\n").encode()
        exclusion = dict(id="7", sha256=hashlib.sha256(line.encode()).hexdigest(),
                         reason="precreated exact schema for helper owner CREATE prerequisite")
        record = dict(schema="extensions", name="extensions", owner="postgres")
        selected, _ = restore.select_toc(toc, [exclusion], dict(schemas=[record]))
        self.assertIn(("; excluded " + line).encode(), selected)
        self.assertIn(("\n" + acl + "\n").encode(), selected)
        self.assertIn(("\n" + helper + "\n").encode(), selected)
        for bad in (None, dict(schemas=[]), dict(schemas=[record, record]),
                    dict(schemas=[{**record, "owner": "supabase_admin"}]),
                    dict(schemas=[{**record, "name": "public"}])):
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc, [exclusion], bad)
        for key, text in (("8", acl), ("9", helper)):
            with self.assertRaises(snapshot.Failure):
                restore.select_toc(toc, [dict(id=key, sha256=hashlib.sha256(text.encode()).hexdigest(),
                                             reason="must retain")], dict(schemas=[record]))

    def test_toc_inventory_matches_all_manifest_routines_and_views(self):
        routines = [dict(schema="public", name=f"f_{i}", arguments="", kind="f") for i in range(377)]
        routines += [dict(schema="extensions", name=name, arguments="", kind="f") for name in (
            "grant_pg_cron_access", "grant_pg_graphql_access", "grant_pg_net_access",
            "pgrst_ddl_watch", "pgrst_drop_watch", "set_graphql_placeholder")]
        views = [dict(schema="public", name=f"v_{i}") for i in range(3)]
        metadata = dict(routines=routines, views=views)
        tails = [f"FUNCTION {r['schema']} {r['name']}() postgres" for r in routines]
        tails += [f"VIEW {v['schema']} {v['name']} postgres" for v in views]
        tails += [f"TABLE DATA public t_{i} postgres" for i in range(48)]
        tails += [f"SEQUENCE SET public seq_{i} postgres" for i in range(3)]
        toc = ("\n".join(f"{i + 1}; 0 0 {tail}" for i, tail in enumerate(tails)) + "\n").encode()
        entries = restore.toc_entries(toc)
        counts = restore.validate_toc_inventory(entries, metadata)
        self.assertEqual(counts["FUNCTION"], 383)
        self.assertEqual(counts["VIEW"], 3)
        for mismatch in ({**metadata, "routines": routines[:377]},
                         {**metadata, "views": views[:-1]},
                         {**metadata, "routines": [{**routines[0], "name": "missing"}, *routines[1:]]}):
            with self.assertRaises(snapshot.Failure):
                restore.validate_toc_inventory(entries, mismatch)
        omitted = {key: line for key, line in entries.items() if "FUNCTION extensions " not in line}
        with self.assertRaises(snapshot.Failure):
            restore.validate_toc_inventory(omitted, metadata)

    def test_event_trigger_entries_require_separate_review_not_exclusion(self):
        for tail in ("EVENT TRIGGER - issue_pg_net_access postgres",
                     "COMMENT - EVENT TRIGGER issue_pg_net_access postgres",
                     "SECURITY LABEL - EVENT TRIGGER issue_pg_net_access postgres"):
            line = "1; 3466 100 " + tail
            toc = (line + "\n").encode()
            exclusion = dict(id="1", sha256=hashlib.sha256(line.encode()).hexdigest(),
                             reason="must not silently omit an event trigger")
            with self.assertRaisesRegex(snapshot.Failure, "event-trigger-review-required"):
                restore.select_toc(toc, [])
            with self.assertRaisesRegex(snapshot.Failure, "event-trigger-review-required"):
                restore.select_toc(toc, [exclusion])

    def test_supporting_auth_role_requires_exact_getter_identity_and_definition(self):
        self.assertEqual(restore.validate_supporting_auth_role(copy.deepcopy(AUTH_ROLE_SUPPORT)), AUTH_ROLE_SUPPORT)
        invalid = [None, {}, {**AUTH_ROLE_SUPPORT, "extra": "unreviewed"}]
        for field, bad in (("schema", "public"), ("name", "jwt"), ("arguments", "p_input text"),
                           ("result", "jsonb"), ("owner", "postgres"), ("security_definer", True),
                           ("language", "plpgsql"), ("volatility", "v"), ("config", ["search_path=public"]),
                           ("definition", AUTH_ROLE_SUPPORT["function"]["definition"] + "-- altered"),
                           ("sha256", "0" * 64), ("acl", [])):
            value = copy.deepcopy(AUTH_ROLE_SUPPORT)
            value["function"][field] = bad
            invalid.append(value)
        value = copy.deepcopy(AUTH_ROLE_SUPPORT)
        value["schema"]["owner"] = "postgres"
        invalid.append(value)
        value = copy.deepcopy(AUTH_ROLE_SUPPORT)
        value["dependencies"].append(dict(identity="auth.users", kind="n", name="users", schema="auth", type="table"))
        invalid.append(value)
        for value in invalid:
            with self.assertRaises(snapshot.Failure):
                restore.validate_supporting_auth_role(value)

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
        self.native_restore_control("pg_database_owner")

    def test_native_historical_postgres_public_owner(self):
        self.native_restore_control("postgres")

    def native_restore_control(self, public_owner):
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
                        run(admin, data=b"GRANT postgres,lab_arena_owner,supabase_admin,supabase_auth_admin TO arena_admin WITH INHERIT TRUE, SET TRUE; GRANT CREATE ON DATABASE postgres TO postgres,lab_arena_owner,supabase_admin,supabase_auth_admin;")
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
                if public_owner == "pg_database_owner":
                    # Native ACL-only records survive omitted extension pre-data.
                    # Model that boundary without installing the unsupported HTTP
                    # extension: omit these fixture definitions from the list.
                    setup = "CREATE FUNCTION public.ordinary() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$; ALTER FUNCTION public.ordinary() OWNER TO postgres; REVOKE ALL ON FUNCTION public.ordinary() FROM PUBLIC; GRANT EXECUTE ON FUNCTION public.ordinary() TO lab_arena_service;"
                    for i in range(19):
                        identity = "public.http_member_" + str(i) + "(data bytea)"
                        setup += ("CREATE FUNCTION " + identity + " RETURNS text LANGUAGE sql AS $$ SELECT encode(data,'hex') $$; ALTER FUNCTION "
                                  + identity + " OWNER TO supabase_admin; GRANT ALL ON FUNCTION " + identity
                                  + " TO postgres,anon,authenticated,service_role;")
                    run(source_admin, data=setup.encode())
                    archive = root / "http-acl-dump"
                    run([str(PG17 / "pg_dump"), "--dbname=service=migration", "-Fd", "--schema=public", "--file=" + str(archive)], env=source_env)
                    original = run([str(PG17 / "pg_restore"), "--list", str(archive)])
                    acl_list = root / "native-acl-only.list"
                    snapshot.private_write(acl_list, ("\n".join(line for line in original.decode().splitlines() if " ACL " in line) + "\n").encode())
                    acl_sql = run([str(PG17 / "pg_restore"), "--use-list=" + str(acl_list), "--file=-", str(archive)])
                    omitted = [line for line in original.decode().splitlines()
                               if " FUNCTION public http_member_" not in line and " SCHEMA - public " not in line]
                    omitted.append("999999; 3079 100 EXTENSION - http ")
                    toc = ("\n".join(omitted) + "\n").encode()
                    _, metadata, review, seals, _ = http_acl_fixture()
                    review["acl_sql"] = acl_sql
                    seals.update(toc=hashlib.sha256(toc).hexdigest(), acl_sql=hashlib.sha256(acl_sql).hexdigest())
                    run(target_admin, data=b"GRANT CREATE ON SCHEMA public TO postgres;")
                    bad_list = root / "http-bad.list"
                    snapshot.private_write(bad_list, toc.replace(b"999999;", b"; excluded 999999;"))
                    command = restore.native_restore_command(bad_list, archive)
                    with self.assertRaises(snapshot.Failure):
                        snapshot.command(command, target_write, logs, "local-http-missing")
                    self.assertTrue(any("function public.http_member_" in p.read_text() and "does not exist" in p.read_text()
                                        for p in logs.glob("local-http-missing-*.stderr")))
                    self.assertEqual(run(target_admin, data=b"SELECT to_regprocedure('public.ordinary()') IS NULL;").strip(), b"t")
                    with patch.dict(restore.HTTP_ACL_REVIEW_SHA256, seals, clear=True):
                        records = restore.reviewed_http_acls(toc, metadata, review)
                        exclusions = [dict(record, reason="fixture exact unsupported member ACL") for record in records.values()]
                        extension_line = toc.decode().splitlines()[-1]
                        exclusions.append(dict(id="999999", sha256=hashlib.sha256(extension_line.encode()).hexdigest(), reason="fixture unsupported extension"))
                        selected, _ = restore.select_toc(toc, exclusions, metadata, review)
                    good_list = root / "http-good.list"
                    snapshot.private_write(good_list, selected)
                    snapshot.command(restore.native_restore_command(good_list, archive), target_write, logs, "local-http-filtered")
                    self.assertEqual(run(target_admin, data=b"SELECT pg_get_userbyid(proowner),has_function_privilege('lab_arena_service',oid,'EXECUTE'),proacl::text LIKE '%lab_arena_service=X/postgres%' FROM pg_proc WHERE oid='public.ordinary()'::regprocedure;").strip(), b"postgres|t|t")
                    self.assertEqual(run(target_admin, data=b"SELECT count(*) FROM pg_proc WHERE proname LIKE 'http_member_%';").strip(), b"0")
                    run(target_admin, data=b"DROP FUNCTION public.ordinary(); REVOKE CREATE ON SCHEMA public FROM postgres;")
                    run(source_admin, data=("DROP FUNCTION public.ordinary();" + "".join("DROP FUNCTION public.http_member_" + str(i) + "(bytea);" for i in range(19))).encode())
                sql = "CREATE SCHEMA source_add_history AUTHORIZATION lab_arena_owner; CREATE SCHEMA extensions AUTHORIZATION postgres;\n"
                if public_owner == "postgres":
                    sql = "ALTER SCHEMA public OWNER TO postgres; " + sql
                else:
                    self.assertEqual(run(source_admin, data=b"SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname='public';").strip(), b"pg_database_owner")
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
                run(source_admin, data=(auth_schema_sql() + auth_function_sql()).encode())
                sql += "CREATE POLICY source_auth_claim ON public.t_0 USING (auth.role()='service_role') WITH CHECK (auth.role()='service_role');"
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
                                    if " SCHEMA - public " + public_owner in line)
                public_exclusion = dict(id=public_entry.split(";", 1)[0], sha256=hashlib.sha256(public_entry.encode()).hexdigest(),
                                        reason="retain initialized public schema and its manifest-bound source owner")
                listing, _ = restore.select_toc(toc, [public_exclusion], manifest["schema_metadata"])
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
                manifest.update(format=1, source_host=snapshot.SOURCE_HOST, source_role=snapshot.SOURCE_ROLE,
                                database="postgres", supporting_auth_role=copy.deepcopy(AUTH_ROLE_SUPPORT))
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
                self.assertEqual(prepared["source_public_schema_owner"], public_owner)
                self.assertEqual(prepared["supporting_auth_role_sha256"], restore.AUTH_ROLE_SHA256)
                self.assertEqual(json.loads((root / "prepared" / "supporting-auth-role.json").read_text()), AUTH_ROLE_SUPPORT)
                self.assertEqual(prepared["archive_sequences"], states)
                for path in (root / "prepared").iterdir():
                    if path.is_file():
                        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                # Preserve an already-matching initialized public schema.
                # Only the historical postgres variant needs a reviewed owner change.
                target_public = restore.source_public_schema(empty["metadata"])
                self.assertEqual(target_public["owner"], "pg_database_owner")
                self.assertEqual(target_public["acl"]["items"],
                                 ["=U/pg_database_owner", "pg_database_owner=UC/pg_database_owner"])
                if target_public["owner"] != public_owner:
                    self.assertEqual(public_owner, "postgres")
                    run(target_admin, data=b"ALTER SCHEMA public OWNER TO postgres;")
                else:
                    # The initialized schema's owner is already correct. Inert
                    # object owners still need CREATE to accept native ownership.
                    run(target_admin, data=b"GRANT CREATE ON SCHEMA public TO postgres,lab_arena_owner;")
                run(target_admin, data=auth_schema_sql().encode())
                with self.assertRaises(snapshot.Failure):
                    snapshot.command(restore.native_restore_command(root / "restore.list", root / "dump"), target_write, logs, "local-missing-auth")
                missing_auth = "\n".join(path.read_text() for path in logs.glob("local-missing-auth-*.stderr"))
                self.assertIn("function auth.role() does not exist", missing_auth)
                self.assertEqual(run(target_admin, data=b"SELECT to_regclass('public.t_0') IS NULL;").strip(), b"t")
                run(target_admin, data=auth_function_sql().encode())
                definition = run(target_admin, data=b"SELECT pg_get_functiondef('auth.role()'::regprocedure);").decode().rstrip("\n") + "\n"
                self.assertEqual(hashlib.sha256(definition.encode()).hexdigest(), restore.AUTH_ROLE_SHA256)
                self.assertEqual(run(target_admin, data=b"SELECT to_regclass('auth.users') IS NULL;").strip(), b"t")
                self.assertEqual(run(target_admin, data=b"SELECT set_config('request.jwt.claim.role','service_role',false); SELECT auth.role();").strip(), b"service_role\nservice_role")
                try:
                    snapshot.command(restore.native_restore_command(root / "restore.list", root / "dump"), target_write, logs, "local-restore")
                except snapshot.Failure:
                    # This disposable fixture contains no credentials or live rows.
                    # Surface only the local server's first error, not SQL output.
                    error = next((line.split("ERROR:", 1)[1].strip()
                                  for path in logs.glob("local-restore-*.stderr")
                                  for line in path.read_text().splitlines() if "ERROR:" in line), "unknown")
                    self.fail("local restore control: " + error)
                if public_owner == "pg_database_owner":
                    # Neither fixture source ACL nor target baseline grants
                    # CREATE to these roles. Remove only those temporary grants.
                    run(target_admin, data=b"REVOKE CREATE ON SCHEMA public FROM postgres,lab_arena_owner;")
                run(target_admin, data=b"REVOKE CREATE ON DATABASE postgres FROM postgres,lab_arena_owner,supabase_admin,supabase_auth_admin;")
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
