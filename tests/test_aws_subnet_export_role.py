"""Local-only controls for the one temporary export identity; all I/O is mocked."""
import base64
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import importlib.util
import io
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import aws_subnet_export_role as tool

HEAD = "a" * 40
OP = "e9d6c4ba-3f18-4cb2-9bda-31e8163ad05e"
SECRET_ARN = "arn:aws:secretsmanager:us-east-1:493765492819:secret:" + tool.SECRET_NAME + "-Ab1234"


def objects():
    tables = [dict(schema="public", name="table_" + str(i), kind="r") for i in range(47)]
    tables += [dict(schema="source_add_history", name="history", kind="r")]
    return tool.inventory(tables + [dict(schema="public", name="seq_" + str(i), kind="S") for i in range(3)])


def plan():
    return dict(format=1, project=tool.PROJECT, role=tool.ROLE, secret_name=tool.SECRET_NAME,
                head=HEAD, operation_id=OP, objects=objects(), creator_membership=copy.deepcopy(tool.CREATOR_MEMBERSHIP))


def receipt():
    return {**plan(), "secret_arn": SECRET_ARN, "status": "created"}


def source_state(role=None, sessions=0):
    return dict(current_user="postgres", database="postgres", schemas=sorted(tool.SCHEMAS),
                objects=objects(), role=role, sessions=sessions)


def role(created=True):
    return dict(comment=tool.COMMENT_PREFIX + OP, login=created, bypassrls=created,
                super=False, createdb=False, createrole=False, replication=False,
                connections=5, config=["default_transaction_read_only=on"], memberships=copy.deepcopy(tool.CREATOR_MEMBERSHIP),
                valid_until=(datetime.now(timezone.utc) + timedelta(hours=23)).isoformat(),
                grants=tool.expected_grants(objects()) if created else [])


class ExportRoleTests(unittest.TestCase):
    def test_import_does_not_perform_io(self):
        spec = importlib.util.spec_from_file_location("export_role_import_control", tool.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.object(subprocess, "run", side_effect=AssertionError("unexpected process")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            spec.loader.exec_module(module)

    def test_identifiers_and_literals_fail_closed(self):
        self.assertEqual(tool.ident("source_add_history"), '"source_add_history"')
        for invalid in ("a;DROP ROLE postgres", 'a"', "x.y", "é", "A", "", None, "a" * 64):
            with self.assertRaisesRegex(tool.Failure, "identifier"):
                tool.ident(invalid)
        self.assertEqual(tool.literal("x'\\y"), "E'x''\\\\y'")
        with self.assertRaises(tool.Failure):
            tool.literal("x\0y")
        with self.assertRaises(tool.Failure):
            tool.operation("'; drop role postgres;--")

    def test_inventory_accepts_only_exact_base_tables_and_sequences(self):
        self.assertEqual(len(tool.inventory(objects())), 51)
        for change in (lambda x: x.pop(), lambda x: x.append(x[0]),
                       lambda x: x[0].update(kind="v"), lambda x: x[0].update(kind="p"),
                       lambda x: x[0].update(kind="f"), lambda x: x[0].update(name="bad;sql"),
                       lambda x: x[0].update(schema="auth")):
            value = copy.deepcopy(objects())
            change(value)
            with self.assertRaises(tool.Failure):
                tool.inventory(value)

    def test_scram_verifier_matches_independent_stdlib_recomputation(self):
        password, salt = "safe_ASCII-password-0123456789", bytes(range(16))
        verifier = tool.scram_verifier(password, salt)
        mechanism, iteration_salt, keys = verifier.split("$")
        iterations, encoded_salt = iteration_salt.split(":")
        self.assertEqual(mechanism, "SCRAM-SHA-256")
        self.assertEqual(int(iterations), 4096)
        self.assertEqual(base64.b64decode(encoded_salt), salt)
        salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 4096)
        client = hmac.digest(salted, b"Client Key", "sha256")
        expected = (hashlib.sha256(client).digest(), hmac.digest(salted, b"Server Key", "sha256"))
        self.assertEqual(tuple(base64.b64decode(k) for k in keys.split(":")), expected)
        self.assertIsNotNone(tool.VERIFIER.fullmatch(verifier))
        for invalid in ("", "non-ascii-é"):
            with self.assertRaises(tool.Failure):
                tool.scram_verifier(invalid)

    def test_create_sql_is_one_transaction_with_exact_readonly_grants(self):
        password = "never-send-the-raw-password"
        verifier = tool.scram_verifier(password, bytes(range(16)))
        sql = tool.create_sql(plan(), verifier, "2026-10-29T00:00:00+00:00")
        self.assertTrue(sql.startswith("BEGIN;\n"))
        self.assertTrue(sql.endswith("COMMIT;"))
        self.assertEqual(sql.count("BEGIN;"), 1)
        self.assertEqual(sql.count("COMMIT;"), 1)
        self.assertNotIn(password, sql)
        for text in ("LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS",
                     "CONNECTION LIMIT 5", "VALID UNTIL", "default_transaction_read_only=on",
                     tool.COMMENT_PREFIX + OP, "current_user <> 'postgres'", "source_add_history"):
            self.assertIn(text, sql)
        grants = [line for line in sql.splitlines() if line.startswith("GRANT ")]
        self.assertEqual(len(grants), 3)
        self.assertTrue(grants[0].startswith("GRANT USAGE ON SCHEMA "))
        self.assertTrue(grants[1].startswith("GRANT SELECT ON TABLE "))
        self.assertTrue(grants[2].startswith("GRANT SELECT ON SEQUENCE "))
        for name in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "ALL TABLES", "DEFAULT PRIVILEGES",
                     "GRANT EXECUTE", " TO PUBLIC", "DROP ", "PARITY"):
            self.assertNotIn(name, sql)
        self.assertEqual(grants[1].count('"public".'), 47)
        self.assertEqual(grants[1].count('"source_add_history".'), 1)
        self.assertEqual(grants[2].count('"public".'), 3)

    def test_keychain_encoding_and_errors_are_private(self):
        value = b"private-access-token"
        result = subprocess.CompletedProcess([], 0, b"go-keyring-base64:" + base64.b64encode(value), b"")
        with patch.object(tool.platform, "system", return_value="Darwin"), patch.object(tool, "run", return_value=result) as run:
            self.assertEqual(tool.access_token(), value.decode())
            self.assertEqual(run.call_args.args[0], ["security", "find-generic-password", "-s", "Supabase CLI", "-w"])
        result.stdout = b"go-keyring-base64:!!private-token!!"
        with patch.object(tool.platform, "system", return_value="Darwin"), patch.object(tool, "run", return_value=result):
            with self.assertRaisesRegex(tool.Failure, "^keychain-format$"):
                tool.access_token()

    def test_source_request_exact_project_readonly_and_no_raw_error(self):
        response = io.BytesIO(b'[{"ok": true}]')
        with patch.object(tool, "access_token", return_value="private-token"), patch.object(tool, "urlopen", return_value=response) as opened:
            self.assertEqual(tool.query("SELECT true"), [{"ok": True}])
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.supabase.com/v1/projects/" + tool.PROJECT + "/database/query")
            self.assertEqual(json.loads(request.data), {"query": "SELECT true", "read_only": True})
            self.assertEqual(request.headers["Authorization"], "Bearer private-token")
        with patch.object(tool, "access_token", return_value="private-token"), patch.object(tool, "urlopen", side_effect=RuntimeError("private-token secret-verifier")):
            with self.assertRaisesRegex(tool.Failure, "^source-request$"):
                tool.query("private-SQL", read_only=False)

    def test_metadata_uses_admin_route_inside_explicit_readonly_transaction(self):
        response = io.BytesIO(json.dumps([{"state": source_state(role())}]).encode())
        with patch.object(tool, "access_token", return_value="local-mocked-token"), \
             patch.object(tool, "urlopen", return_value=response) as opened:
            observed = tool.state()
        payload = json.loads(opened.call_args.args[0].data)
        self.assertIs(payload["read_only"], False)
        self.assertEqual(payload["query"], "BEGIN READ ONLY;\n" + tool.STATE_SQL + "\nROLLBACK;")
        self.assertNotIn("COMMIT;", payload["query"])
        self.assertEqual(observed["current_user"], "postgres")
        self.assertEqual(observed["role"]["memberships"], tool.CREATOR_MEMBERSHIP)

    def test_preflight_rejects_wrong_identity_existing_role_and_secret(self):
        with patch.object(tool, "aws_json", return_value={"Account": "wrong"}):
            with self.assertRaisesRegex(tool.Failure, "aws-account"):
                tool.check_aws()
        for field, bad in (("current_user", "other"), ("database", "other"), ("schemas", ["public"])):
            value = source_state()
            value[field] = bad
            with patch.object(tool, "query", return_value=[{"state": value}]):
                with self.assertRaises(tool.Failure):
                    tool.state()
        with patch.object(tool, "check_aws"), patch.object(tool, "state", return_value=source_state(role())):
            with self.assertRaisesRegex(tool.Failure, "role-already-exists"):
                tool.preflight()
        for result in (subprocess.CompletedProcess([], 0, b"{}", b""),
                       subprocess.CompletedProcess([], 1, b"", b"AccessDeniedException private-token")):
            with patch.object(tool, "aws", return_value=result):
                with self.assertRaises(tool.Failure):
                    tool.secret_absent()
        with patch.object(tool, "aws", return_value=subprocess.CompletedProcess([], 1, b"", b"(ResourceNotFoundException)")):
            tool.secret_absent()

    def test_head_and_committed_tool_bytes_are_bound(self):
        good = subprocess.CompletedProcess([], 0, (HEAD + "\n").encode(), b"")
        body = subprocess.CompletedProcess([], 0, Path(tool.__file__).read_bytes(), b"")
        with patch.object(tool, "run", side_effect=[good, body]):
            self.assertEqual(tool.verified_head(Path.cwd(), HEAD), HEAD)
        with patch.object(tool, "run", return_value=subprocess.CompletedProcess([], 0, b"b" * 40, b"")):
            with self.assertRaisesRegex(tool.Failure, "repo-head"):
                tool.verified_head(Path.cwd(), HEAD)
        with patch.object(tool, "run", side_effect=[good, subprocess.CompletedProcess([], 0, b"other", b"")]):
            with self.assertRaisesRegex(tool.Failure, "repo-tool-bytes"):
                tool.verified_head(Path.cwd(), HEAD)

    def test_aws_cli_stdin_never_contains_secret_in_arguments(self):
        with patch.object(tool, "run", return_value=subprocess.CompletedProcess([], 0, b"{}", b"")) as run:
            tool.aws("secretsmanager", "create-secret", payload={"SecretString": "private-password"})
            args, data = run.call_args.args
            self.assertEqual(args[0], tool.AWS)
            self.assertIn(tool.PROFILE, args)
            self.assertIn("file:///dev/stdin", args)
            self.assertIn("private-password", data.decode())
            self.assertNotIn("private-password", " ".join(args))
            self.assertNotIn("--secret-string", args)

    def test_create_persists_secret_receipt_before_ddl_and_verifies_grants(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            p, r = Path(directory) / "plan", Path(directory) / "receipt"
            tool.write_json(p, plan())
            def fake_aws(*args, **kwargs):
                events.append("secret")
                payload = kwargs["payload"]
                credential = json.loads(payload["SecretString"])
                self.assertEqual(set(credential), {"host", "port", "dbname", "role", "user", "password", "operation_id"})
                self.assertEqual(credential["password"], "private-password")
                self.assertEqual(payload["ClientRequestToken"], OP)
                self.assertNotIn("KmsKeyId", payload)
                return {"ARN": SECRET_ARN, "Name": tool.SECRET_NAME}
            def fake_query(sql, **kwargs):
                events.append("ddl")
                self.assertEqual(json.loads(r.read_text())["secret_arn"], SECRET_ARN)
                self.assertEqual(kwargs, {"read_only": False})
                self.assertNotIn("private-password", sql)
                return []
            with patch.object(tool, "preflight", return_value=source_state()), \
                 patch.object(tool.secrets, "token_urlsafe", return_value="private-password"), \
                 patch.object(tool, "aws_json", side_effect=fake_aws), \
                 patch.object(tool, "query", side_effect=fake_query), \
                 patch.object(tool, "state", return_value=source_state(role())):
                result = tool.execute_create(p, r, HEAD)
            self.assertEqual(events, ["secret", "ddl"])
            self.assertEqual(result["status"], "created")
            self.assertEqual(r.stat().st_mode & 0o777, 0o600)
            content = r.read_text()
            for secret in ("private-password", "SCRAM-SHA-256", "SecretString"):
                self.assertNotIn(secret, content)
            self.assertEqual(json.loads(content)["objects"], objects())
            with patch.object(tool, "preflight", return_value=source_state(role())) as inspect, patch.object(tool, "aws_json") as aws, patch.object(tool, "query") as query:
                with self.assertRaisesRegex(tool.Failure, "existing-receipt-inspected-review-required"):
                    tool.execute_create(p, r, HEAD)
                inspect.assert_called_once()
                aws.assert_not_called()
                query.assert_not_called()

    def test_unknown_source_outcome_retains_metadata_and_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            p, r = Path(directory) / "plan", Path(directory) / "receipt"
            tool.write_json(p, plan())
            with patch.object(tool, "preflight", return_value=source_state()), \
                 patch.object(tool, "aws_json", return_value={"ARN": SECRET_ARN, "Name": tool.SECRET_NAME}), \
                 patch.object(tool, "query", side_effect=RuntimeError("raw private-password SQL verifier")):
                with self.assertRaisesRegex(tool.Failure, "^source-outcome-unknown-receipt-retained$"):
                    tool.execute_create(p, r, HEAD)
            saved = json.loads(r.read_text())
            self.assertEqual(saved["secret_arn"], SECRET_ARN)
            self.assertIn("unknown", saved["status"])
            self.assertNotIn("private-password", r.read_text())

    def test_disable_requires_matching_comment_and_zero_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            r = Path(directory) / "receipt"
            tool.write_json(r, receipt())
            bad_role = {**role(), "comment": tool.COMMENT_PREFIX + "wrong"}
            for observed, phase in ((source_state(bad_role), "role-operation-binding"),
                                    (source_state(role(), 1), "role-has-active-sessions"),
                                    (source_state({**role(), "memberships": []}), "role-memberships")):
                with patch.object(tool, "check_aws"), patch.object(tool, "state", return_value=observed), patch.object(tool, "query") as query:
                    with self.assertRaisesRegex(tool.Failure, phase):
                        tool.disable(r, HEAD, True)
                    query.assert_not_called()
            with patch.object(tool, "check_aws"), patch.object(tool, "state", side_effect=[source_state(role()), source_state(role(False))]), patch.object(tool, "query", return_value=[]) as query:
                output = tool.disable(r, HEAD, True)
                sql = query.call_args.args[0]
                self.assertIn("NOLOGIN NOBYPASSRLS", sql)
                self.assertIn("pg_stat_activity", sql)
                self.assertIn(tool.COMMENT_PREFIX + OP, sql)
                self.assertEqual(sql.count("REVOKE "), 3)
                for forbidden in ("DROP ", "DELETE ", "TERMINATE", "PUBLIC;", "CASCADE"):
                    self.assertNotIn(forbidden, sql)
                self.assertEqual(output["status"], "disabled-secret-cleanup-remains")

    def test_only_exact_creator_admin_membership_is_allowed(self):
        tool.bound_role(source_state(role()), receipt())
        invalid = [[], [{**tool.CREATOR_MEMBERSHIP[0], "admin": False}],
                   [{**tool.CREATOR_MEMBERSHIP[0], "inherit": True}],
                   [{**tool.CREATOR_MEMBERSHIP[0], "set": True}],
                   [{**tool.CREATOR_MEMBERSHIP[0], "member": "another_role"}],
                   [*tool.CREATOR_MEMBERSHIP, dict(role="other_role", member=tool.ROLE,
                                                  admin=False, inherit=False, set=False)]]
        for membership in invalid:
            with self.assertRaisesRegex(tool.Failure, "role-memberships"):
                tool.bound_role(source_state({**role(), "memberships": membership}), receipt())
        sql = tool.disable_sql(receipt())
        self.assertIn("m.inherit_option", sql)
        self.assertIn("m.set_option", sql)
        self.assertNotIn("REVOKE ADMIN", sql)

    def test_main_errors_never_print_raw_values(self):
        output = io.StringIO()
        with patch.object(tool, "verified_head", return_value=HEAD), \
             patch.object(tool, "preflight", side_effect=RuntimeError("private-token private-password SCRAM-SHA-256 verifier SQL")), \
             contextlib.redirect_stderr(output):
            self.assertEqual(tool.main(["preflight", "--expected-head", HEAD]), 1)
        self.assertEqual(output.getvalue(), "ERROR phase=operator-utility\n")


PG17 = Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_EXPORT_ROLE_LOCAL_PG17") == "1" and PG17.is_dir(),
                     "opt-in disposable local PG17 control")
class NativePostgresTests(unittest.TestCase):
    """Only an empty local cluster, no AWS/source hosts or credentials."""

    def test_exact_sql_scram_login_rls_and_disable(self):
        with tempfile.TemporaryDirectory(prefix="lp-role-", dir="/tmp") as directory:
            root = Path(directory)
            env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
            def run(args, data=None, success=True, extra_env=None):
                result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env={**env, **(extra_env or {})}, check=False)
                self.assertEqual(result.returncode == 0, success,
                                 "local PostgreSQL command did not have the expected result")
                return result.stdout
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            run([str(PG17 / "initdb"), "-D", str(root / "db"), "-U", "local_root",
                 "--auth-local=trust", "--auth-host=scram-sha-256", "--no-locale", "--encoding=UTF8"])
            run([str(PG17 / "pg_ctl"), "-D", str(root / "db"), "-l", str(root / "postgres.log"),
                 "-o", f"-F -h 127.0.0.1 -p {port} -k {root}", "-w", "start"])
            try:
                admin = [str(PG17 / "psql"), "-XqAt", "-v", "ON_ERROR_STOP=1", "-h", str(root),
                         "-p", str(port), "-U", "local_root", "-d", "postgres"]
                run(admin, b"CREATE ROLE postgres LOGIN NOSUPERUSER CREATEROLE BYPASSRLS; ALTER DATABASE postgres OWNER TO postgres;")
                admin[admin.index("-U") + 1] = "postgres"
                seed = "CREATE SCHEMA source_add_history; CREATE SCHEMA extensions;\n"
                for obj in objects():
                    name = tool.ident(obj["schema"]) + "." + tool.ident(obj["name"])
                    seed += ("CREATE TABLE " + name + "(id integer);\n") if obj["kind"] == "r" else ("CREATE SEQUENCE " + name + ";\n")
                seed += ("INSERT INTO public.table_0 VALUES (1),(2); ALTER TABLE public.table_0 ENABLE ROW LEVEL SECURITY; "
                         "ALTER TABLE public.table_0 FORCE ROW LEVEL SECURITY; CREATE POLICY deny_all ON public.table_0 USING (false); "
                         "CREATE VIEW public.not_exported AS SELECT * FROM public.table_0;")
                run(admin, seed.encode())
                observed = json.loads(run(admin, tool.STATE_SQL.encode()))
                self.assertEqual(observed["objects"], objects())
                password = "local-test-only-password"
                expiry = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
                run(admin, tool.create_sql(plan(), tool.scram_verifier(password), expiry).encode())
                observed = json.loads(run(admin, tool.STATE_SQL.encode()))
                self.assertEqual(observed["role"]["grants"], tool.expected_grants(objects()))
                self.assertEqual(observed["role"]["memberships"], tool.CREATOR_MEMBERSHIP)
                password_file = root / "pass"
                fd = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as out:
                    out.write(f"127.0.0.1:{port}:postgres:{tool.ROLE}:{password}\n")
                reader = [str(PG17 / "psql"), "-XqAt", "-w", "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1",
                          "-p", str(port), "-U", tool.ROLE, "-d", "postgres"]
                read_env = {"PGPASSFILE": str(password_file)}
                self.assertEqual(run(reader, b"SHOW transaction_read_only; SELECT count(*) FROM public.table_0;", extra_env=read_env), b"on\n2\n")
                run(reader, b"INSERT INTO public.table_0 VALUES(3);", success=False, extra_env=read_env)
                run(reader, b"BEGIN READ WRITE; INSERT INTO public.table_0 VALUES(3);", success=False, extra_env=read_env)
                run(reader, b"SELECT * FROM public.not_exported;", success=False, extra_env=read_env)
                run(admin, tool.disable_sql(receipt()).encode())
                observed = json.loads(run(admin, tool.STATE_SQL.encode()))
                self.assertFalse(observed["role"]["login"])
                self.assertFalse(observed["role"]["bypassrls"])
                self.assertEqual(observed["role"]["grants"], [])
                self.assertEqual(run(admin, b"SELECT count(*) FROM public.table_0;"), b"2\n")
                run(reader, b"SELECT 1;", success=False, extra_env=read_env)
            finally:
                run([str(PG17 / "pg_ctl"), "-D", str(root / "db"), "-m", "immediate", "-w", "stop"])


if __name__ == "__main__":
    unittest.main()
