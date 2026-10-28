"""Private, read-only snapshot operator controls; no AWS or live database I/O."""

import contextlib
import copy
import gzip
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import aws_subnet_snapshot as snapshot


class SnapshotTests(unittest.TestCase):
    def test_manifest_metadata_is_captured_by_the_exact_snapshot_keeper(self):
        tables = [dict(schema="public", name=f"t_{i}", kind="r", filtered_rls=False, inherited=False,
                       columns=[dict(name="id", type="integer", generated="")]) for i in range(48)]
        sequences = [dict(schema="public", name=f"seq_{i}") for i in range(3)]
        class FakeKeeper:
            initial_settings = dict(timezone="UTC", bytea_output="hex")
            def query(self, sql):
                self.lock = sql
            def one_json(self, sql):
                if sql == snapshot.CATALOG:
                    return tables
                if sql.startswith("SELECT count(*)"):
                    return 2
                if "c.relkind='S'" in sql:
                    return sequences
                if "'last_value'" in sql:
                    return dict(last_value="1", is_called=False)
                if "FROM pg_catalog.pg_extension" in sql:
                    return [dict(name=e) for e in snapshot.EXTENSIONS]
                raise AssertionError("unexpected catalog query")
        keeper = FakeKeeper()
        metadata = {key: [] for key in snapshot.OBJECT_KINDS}
        with patch.object(snapshot, "capture_metadata", return_value=metadata) as capture:
            manifest = snapshot.capture_manifest(keeper, "00000003-0000001B-1")
        capture.assert_called_once_with(keeper)
        self.assertIs(manifest["schema_metadata"], metadata)
        self.assertEqual(manifest["snapshot"], "00000003-0000001B-1")
        self.assertEqual(len(manifest["tables"]), 48)
        self.assertTrue(keeper.lock.startswith("LOCK TABLE"))

    def test_dump_defaults_fail_closed_without_changing_source_role(self):
        snapshot.check_dump_defaults(SimpleNamespace(initial_settings=dict(timezone="UTC", bytea_output="hex")))
        for settings in (dict(timezone="America/New_York", bytea_output="hex"),
                         dict(timezone="UTC", bytea_output="escape")):
            with self.assertRaisesRegex(snapshot.Failure, "dump-session-defaults"):
                snapshot.check_dump_defaults(SimpleNamespace(initial_settings=settings))

    def test_multiset_is_order_independent_and_preserves_duplicates(self):
        digest = snapshot.stream_digest
        self.assertEqual(digest(io.BytesIO(b"a\nb\na\n")), digest(io.BytesIO(b"b\na\na\n")))
        self.assertNotEqual(digest(io.BytesIO(b"a\na\n")), digest(io.BytesIO(b"")))
        self.assertNotEqual(digest(io.BytesIO(b"a\nb\n")), digest(io.BytesIO(b"a\nc\n")))
        self.assertEqual(digest(io.BytesIO(b""))["count"], 0)

    def test_archive_terminator_and_escaped_newlines(self):
        rows = b"a\\nb\t\\N\n\\\\.\n"
        self.assertEqual(snapshot.stream_digest(io.BytesIO(rows)),
                         snapshot.stream_digest(io.BytesIO(rows + b"\\.\n\n\n"), archive=True))
        self.assertEqual(snapshot.stream_digest(io.BytesIO(rows))["count"], 2)
        for content in (b"row", b"row\n", b"row\n\\.\ntrailing\n"):
            with self.assertRaises(snapshot.Failure):
                snapshot.stream_digest(io.BytesIO(content), archive=True)
        with self.assertRaises(snapshot.Failure):
            snapshot.stream_digest(io.BytesIO(b"\\.\n"))

    def test_digest_handles_chunk_boundaries_and_large_rows(self):
        class SmallChunks(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 2))
        rows = b"x" * (1024 * 1024 + 5) + b"\ny\n"
        self.assertEqual(snapshot.stream_digest(io.BytesIO(rows))["count"], 2)
        self.assertEqual(snapshot.stream_digest(SmallChunks(b"abc\n\\.\n\n"), archive=True),
                         snapshot.stream_digest(io.BytesIO(b"abc\n")))

    def source_secret(self):
        return dict(host=snapshot.SOURCE_HOST, port=5432, dbname="postgres",
                    role=snapshot.SOURCE_ROLE, user=snapshot.SOURCE_USER, password="private:pass\\word")

    def test_exact_source_and_target_identity_validation(self):
        value = self.source_secret()
        snapshot.connection_config("source", value, snapshot.SOURCE_HOST)
        for key, bad in (("host", "evil.example"), ("dbname", "other"), ("role", "postgres"),
                         ("user", "other.project"), ("port", 6543), ("password", "bad\nvalue")):
            with self.assertRaises(snapshot.Failure):
                snapshot.connection_config("source", {**value, key: bad}, snapshot.SOURCE_HOST)
        with self.assertRaises(snapshot.Failure):
            snapshot.connection_config("source", value, "127.0.0.1")
        with self.assertRaises(snapshot.Failure):
            snapshot.connection_config("target", dict(username="arena_admin", password="x"), "evil.example")
        with self.assertRaises(snapshot.Failure):
            snapshot.connection_config("target", dict(username="postgres", password="x"), snapshot.TARGET_HOST)

    def test_private_service_password_lifecycle_and_no_credentials_in_arguments(self):
        config = snapshot.connection_config("source", self.source_secret(), snapshot.SOURCE_HOST)
        with patch.dict(os.environ, {"PGPASSWORD": "inherited-secret", "PGOPTIONS": "-c row_security=on"}):
            with snapshot.service(config) as env:
                service = Path(env["PGSERVICEFILE"])
                password = Path(env["PGPASSFILE"])
                root = service.parent
                self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                self.assertEqual(service.stat().st_mode & 0o777, 0o600)
                self.assertEqual(password.stat().st_mode & 0o777, 0o600)
                self.assertIn("sslmode=verify-full", service.read_text())
                self.assertIn("sslrootcert=/etc/leadpoet-migration/supabase-prod-ca-2021.crt", service.read_text())
                self.assertIn("default_transaction_read_only=on", service.read_text())
                for option in ("keepalives_idle=30", "keepalives_interval=10", "keepalives_count=3"):
                    self.assertIn(option, service.read_text())
                self.assertIn("private\\:pass\\\\word", password.read_text())
                self.assertNotIn("PGPASSWORD", env)
                self.assertNotIn("PGOPTIONS", env)
                self.assertNotIn(config["password"], service.read_text())
                self.assertNotIn(config["password"], " ".join(snapshot.psql_command()))
        self.assertFalse(root.exists())

    def test_export_closes_source_keeper_before_local_archive_validation(self):
        events = []
        @contextlib.contextmanager
        def keeper(*_):
            events.append("enter")
            try:
                yield SimpleNamespace(snapshot=lambda: "00000003-0000001B-1", initial_settings=dict(timezone="UTC", bytea_output="hex"))
            finally:
                events.append("exit")
        def run(*_, **__):
            events.append("dump")
        def validate(*_):
            self.assertEqual(events, ["enter", "dump", "exit"])
            events.append("validate")
            return {}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(directory=str(Path(directory) / "export"), source_secret_arn="unused", source_host=snapshot.SOURCE_HOST)
            with patch.object(snapshot, "secret", return_value=self.source_secret()), \
                 patch.object(snapshot, "check_tools"), patch.object(snapshot, "check_identity"), \
                 patch.object(snapshot, "Keeper", side_effect=keeper), \
                 patch.object(snapshot, "capture_manifest", return_value={"tables": []}), \
                 patch.object(snapshot, "command", side_effect=run), \
                 patch.object(snapshot, "validate_archive", side_effect=validate), contextlib.redirect_stdout(io.StringIO()):
                snapshot.export(args)
            self.assertTrue((Path(args.directory) / "manifest.sha256").is_file())
        self.assertEqual(events, ["enter", "dump", "exit", "validate"])

    def test_sigterm_cleans_credentials_exits_keeper_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        events, paths = [], []
        @contextlib.contextmanager
        def keeper():
            try:
                yield
            finally:
                events.append("keeper-exited")
        def terminate(_):
            config = snapshot.connection_config("source", self.source_secret(), snapshot.SOURCE_HOST)
            with snapshot.service(config) as env, keeper():
                paths.append(Path(env["PGPASSFILE"]).parent)
                # Invoke the actual installed POSIX handler without signalling
                # unrelated threads or subprocesses in the test runner.
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        output = io.StringIO()
        old_umask = os.umask(0o077)
        try:
            with patch.object(snapshot, "export", side_effect=terminate), contextlib.redirect_stderr(output):
                code = snapshot.main(["export", "--directory", "unused", "--source-secret-arn", "unused"])
        finally:
            os.umask(old_umask)
        self.assertEqual(code, 1)
        self.assertEqual(events, ["keeper-exited"])
        self.assertFalse(paths[0].exists())
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(output.getvalue(), "ERROR phase=interrupted\n")

    def test_failed_secret_and_unexpected_error_never_print_private_output(self):
        result = subprocess.CompletedProcess([], 1, stdout=b"secret-value", stderr=b"private-row")
        with patch.object(snapshot.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(snapshot.Failure, "^secret-fetch$"):
                snapshot.secret("arn:aws:secretsmanager:us-east-1:493765492819:secret:rds!db-test-Ab1234")
        output = io.StringIO()
        old_umask = os.umask(0o077)
        try:
            with patch.object(snapshot, "export", side_effect=ValueError("secret-value private-row")), contextlib.redirect_stderr(output):
                code = snapshot.main(["export", "--directory", "unused", "--source-secret-arn", "unused"])
        finally:
            os.umask(old_umask)
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "ERROR phase=operator-utility\n")

    def test_commands_keep_source_readonly_and_dump_complete(self):
        args = snapshot.dump_command("00000003-0000001B-1", Path("private/dump"))
        for option in ("-Fd", "-j2", "--strict-names", "--no-publications", "--no-subscriptions", "--encoding=UTF8", "--extra-float-digits=3",
                       "--compress=gzip:1", "--lock-wait-timeout=5s"):
            self.assertIn(option, args)
        self.assertEqual({a.split("=", 1)[1] for a in args if a.startswith("--schema=")}, set(snapshot.SCHEMAS))
        self.assertEqual({a.split("=", 1)[1] for a in args if a.startswith("--extension=")}, set(snapshot.EXTENSIONS))
        self.assertNotIn("--enable-row-security", args)
        self.assertIn("READ ONLY", snapshot.BEGIN)
        self.assertIn("--no-psqlrc", snapshot.psql_command())
        self.assertIn("--set=ON_ERROR_STOP=1", snapshot.psql_command())
        for bad in ("'; DROP TABLE public.a; --", "bad-snapshot", "1-2-3"):
            with self.assertRaises(snapshot.Failure):
                snapshot.dump_command(bad, Path("dump"))

    def test_toc_rejects_duplicates_injection_and_missing_tables(self):
        toc = "\n".join(f"{i+100}; 0 {i+200} TABLE DATA public t_{i} owner" for i in range(48)).encode()
        self.assertEqual(snapshot.table_data_entries(toc)["public", "t_0"], "100.dat.gz")
        for bad in (toc + b"\n100; 0 200 TABLE DATA public t_0 owner", toc.replace(b"public t_0", b"public evil;sql"), b""):
            with self.assertRaises(snapshot.Failure):
                snapshot.table_data_entries(bad)

    def test_table_validation_rejects_filtered_rls_and_generated_columns(self):
        tables = [dict(schema="public", name=f"t_{i}", kind="r", filtered_rls=False, inherited=False,
                       columns=[dict(name="id", type="integer", generated="")]) for i in range(48)]
        snapshot.validate_tables(tables)
        for key, bad in (("filtered_rls", True), ("inherited", True), ("kind", "p"), ("name", "bad;sql")):
            changed = copy.deepcopy(tables)
            changed[0][key] = bad
            with self.assertRaises(snapshot.Failure):
                snapshot.validate_tables(changed)
        tables[0]["columns"][0]["generated"] = "s"
        with self.assertRaises(snapshot.Failure):
            snapshot.validate_tables(tables)


PG17 = Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_SNAPSHOT_LOCAL_PG17") == "1" and PG17.is_dir(),
                     "opt-in disposable local PG17 control")
class NativePostgresTests(unittest.TestCase):
    """Native dump/restore proof with no AWS, no production hosts, and no TLS.

    Supabase-only extensions are deliberately omitted from this local fixture.
    The production CLI has no host/TLS/role bypass flag.
    """

    def test_native_snapshot_directory_archive_restore_and_full_row_copy(self):
        with tempfile.TemporaryDirectory(prefix="lp-snapshot-", dir="/tmp") as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir(mode=0o700)
            base = {**snapshot.pg_environment(), "PATH": str(PG17) + os.pathsep + os.environ["PATH"]}
            def run(args, env=base, data=None):
                return subprocess.run(args, env=env, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            run([str(PG17 / "initdb"), "-D", str(root / "db"), "-U", "local_operator", "--auth=trust", "--no-locale", "--encoding=UTF8"])
            run([str(PG17 / "pg_ctl"), "-D", str(root / "db"), "-l", str(root / "postgres.log"),
                 "-o", f"-F -h 127.0.0.1 -p {port} -k {root}", "-w", "start"])
            try:
                admin = [str(PG17 / "psql"), "-XqAt", "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1", "-p", str(port), "-U", "local_operator"]
                run(admin + ["-d", "postgres"], data=b"CREATE DATABASE source; CREATE DATABASE target;")
                seed = "CREATE SCHEMA source_add_history; CREATE SCHEMA extensions;\n"
                for i in range(48):
                    seed += f"CREATE TABLE public.t_{i}(id integer, txt text, f double precision, b bytea, d timestamptz, j jsonb, n text);\n"
                    seed += f"INSERT INTO public.t_{i} VALUES (1,$$line\nsecond\t\\tail$$,0.1,'\\x00ff','2026-10-28T00:00:00Z','{{\"a\":1}}',NULL),(2,$$\\.$$,'Infinity','\\x','2026-10-28T01:02:03Z','null',''),(1,$$line\nsecond\t\\tail$$,0.1,'\\x00ff','2026-10-28T00:00:00Z','{{\"a\":1}}',NULL);\n"
                seed += "CREATE SEQUENCE public.seq_a; CREATE SEQUENCE public.seq_b; CREATE SEQUENCE source_add_history.seq_c;"
                run(admin + ["-d", "source"], data=seed.encode())
                # Simulate a session pooler that ignores libpq startup options.
                # Local disposable source defaults satisfy pg_dump's inherited
                # timezone/bytea requirement; all other defaults conflict.
                run(admin + ["-d", "postgres"], data=b"ALTER DATABASE source SET TimeZone='UTC'; ALTER DATABASE source SET bytea_output='hex'; ALTER DATABASE source SET DateStyle='German, DMY'; ALTER DATABASE source SET IntervalStyle='sql_standard'; ALTER DATABASE source SET extra_float_digits=-1; ALTER DATABASE source SET row_security=on; ALTER DATABASE source SET statement_timeout='10s'; ALTER DATABASE source SET lock_timeout='100ms'; ALTER DATABASE source SET transaction_timeout='10s'; ALTER DATABASE target SET TimeZone='Asia/Kolkata'; ALTER DATABASE target SET bytea_output='escape'; ALTER DATABASE target SET extra_float_digits=-1; ALTER DATABASE target SET row_security=on; ALTER DATABASE target SET DateStyle='German, DMY'; ALTER DATABASE target SET IntervalStyle='sql_standard';")

                def env_for(db, readonly=True):
                    path = root / (db + ("-read" if readonly else "-write") + ".service")
                    snapshot.private_write(path, (f"[migration]\nhost=127.0.0.1\nport={port}\nuser=local_operator\ndbname={db}\nsslmode=disable\n").encode())
                    return {**base, "PGSERVICEFILE": str(path)}
                source_env, target_env, restore_env = env_for("source"), env_for("target"), env_for("target", False)
                with snapshot.Keeper(source_env, logs) as keeper:
                    snapshot.check_dump_defaults(keeper)
                    settings = keeper.one_json("""SELECT json_build_object(
                      'readonly',current_setting('transaction_read_only'),'rls',current_setting('row_security'),
                      'timezone',current_setting('TimeZone'),'bytea',current_setting('bytea_output'),
                      'dates',current_setting('DateStyle'),'intervals',current_setting('IntervalStyle'),
                      'floats',current_setting('extra_float_digits'),'lock',current_setting('lock_timeout'),
                      'statement',current_setting('statement_timeout'),'transaction',current_setting('transaction_timeout'),
                      'encoding',current_setting('client_encoding'),'path',current_setting('search_path'));
                    """)
                    self.assertEqual({k:v for k,v in settings.items() if k != 'dates'}, dict(readonly="on",rls="off",timezone="UTC",bytea="hex",intervals="postgres",floats="3",lock="5s",statement="0",transaction="0",encoding="UTF8",path="pg_catalog"))
                    self.assertTrue(settings["dates"].startswith("ISO"))
                    exported = keeper.snapshot()
                    tables = keeper.one_json(snapshot.CATALOG)
                    snapshot.validate_tables(tables)
                    schema_metadata = snapshot.capture_metadata(keeper)
                    for table in tables:
                        table["count"] = keeper.one_json("SELECT count(*) FROM ONLY " + snapshot.qualified(table) + ";")
                    # A committed later source row must not enter the exported snapshot.
                    run(admin + ["-d", "source"], data=b"INSERT INTO public.t_0(id,txt) VALUES(999,'later');")
                    args = [a for a in snapshot.dump_command(exported, root / "dump") if not a.startswith("--extension=")]
                    snapshot.command(args, source_env, logs, "local-dump")
                    manifest = dict(tables=tables, schema_metadata=schema_metadata, sequences=[dict(schema="public", name="seq_a"),
                                    dict(schema="public", name="seq_b"), dict(schema="source_add_history", name="seq_c")])
                files = snapshot.validate_archive(root / "dump", manifest, source_env, logs)
                self.assertEqual(len(files), 49)
                self.assertEqual(set(manifest["schema_metadata"]), set(snapshot.OBJECT_KINDS))
                self.assertEqual(len(manifest["schema_metadata"]["relations"]), 51)
                self.assertEqual(sum(t["digest"]["count"] for t in tables), 144)
                # Only the disposable target: native archive contains CREATE SCHEMA public.
                run(admin + ["-d", "target"], data=b"DROP SCHEMA public;")
                snapshot.command(["pg_restore", "--exit-on-error", "--no-owner", "--no-privileges", "--dbname=service=migration", str(root / "dump")], restore_env, logs, "local-restore")
                # Change physical order while preserving duplicate rows and contents.
                run(admin + ["-d", "target"], data=b"CREATE TEMP TABLE reordered AS SELECT * FROM public.t_0 ORDER BY id DESC; TRUNCATE public.t_0; INSERT INTO public.t_0 SELECT * FROM reordered;")
                with snapshot.Keeper(target_env, logs) as keeper:
                    target_snapshot = keeper.snapshot()
                    for table in tables:
                        self.assertEqual(snapshot.copy_digest(table, target_snapshot, target_env, logs), table["digest"])
                run(admin + ["-d", "target"], data=b"UPDATE public.t_0 SET txt='changed' WHERE id=2;")
                with snapshot.Keeper(target_env, logs) as keeper:
                    self.assertNotEqual(snapshot.copy_digest(tables[0], keeper.snapshot(), target_env, logs), tables[0]["digest"])
                with snapshot.Keeper(source_env, logs) as keeper:
                    with self.assertRaises(snapshot.Failure):
                        keeper.query("INSERT INTO public.t_0(id) VALUES(1000);")
                run(admin + ["-d", "source"], data=b"CREATE ROLE fixture_reader LOGIN; GRANT USAGE ON SCHEMA public,source_add_history,extensions TO fixture_reader; GRANT SELECT ON ALL TABLES IN SCHEMA public TO fixture_reader; ALTER TABLE public.t_1 ENABLE ROW LEVEL SECURITY;")
                reader_service = root / "reader.service"
                snapshot.private_write(reader_service, Path(source_env["PGSERVICEFILE"]).read_bytes().replace(b"user=local_operator", b"user=fixture_reader"))
                reader_env = {**source_env, "PGSERVICEFILE": str(reader_service)}
                with snapshot.Keeper(reader_env, logs) as keeper:
                    with self.assertRaises(snapshot.Failure):
                        snapshot.validate_tables(keeper.one_json(snapshot.CATALOG))
            finally:
                run([str(PG17 / "pg_ctl"), "-D", str(root / "db"), "-m", "fast", "-w", "stop"])


if __name__ == "__main__":
    unittest.main()
