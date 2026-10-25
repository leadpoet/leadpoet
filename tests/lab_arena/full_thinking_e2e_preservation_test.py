"""The paid E2E keeps a real database and object checkpoint across export interruption."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from scripts import verify_arena_full_thinking_e2e as e2e
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


class _Evidence:
    def __init__(self):
        self.events = []

    def add(self, event, **fields):
        self.events.append((event, fields))


def test_checkpoint_is_restorable_after_export_keyboard_interrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_ARENA_PG_LOCAL", "1")
    database = database_with_lab_arena_migration(())
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("CREATE TABLE checkpoint_probe (value integer NOT NULL)")
            cursor.execute("INSERT INTO checkpoint_probe VALUES (37)")
        objects_root = tmp_path / "working-objects"
        objects_root.mkdir()
        (objects_root / "receipt.json").write_text('{"run":"accepted"}', encoding="utf-8")
        harness = SimpleNamespace(round_id="round-local", objects_root=objects_root)
        report = tmp_path / "evidence-private-ledger.json"
        evidence = _Evidence()

        checkpoint = e2e._preserve_checkpoint(
            harness, report, dsn, evidence, e2e._local_pg_dump_tools())
        monkeypatch.setattr(e2e.training_reader, "export_run", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            e2e.training_reader.export_run(None, None, "run-local")

        dump = Path(checkpoint["database_path"])
        assert dump.is_file() and dump.stat().st_size
        assert contracts.hash_bytes(dump.read_bytes()) == checkpoint["database_sha256"]
        restore_dsn = {**dsn, "dbname": "checkpoint_restore"}
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("CREATE DATABASE checkpoint_restore")
        finally:
            connection.close()
        subprocess.run(
            [str(e2e._local_pg_dump_tools()[1]), "-h", str(dsn["host"]),
             "-p", str(dsn["port"]), "-U", str(dsn["user"]),
             "-d", restore_dsn["dbname"], str(dump)],
            capture_output=True, text=True, check=True)
        with psycopg2.connect(**restore_dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT value FROM checkpoint_probe")
            assert cursor.fetchall() == [(37,)]
        objects = Path(checkpoint["object_path"])
        manifest = json.loads((objects / "_manifest.json").read_text())
        assert manifest == [{"ref": "receipt.json", "sha256": contracts.hash_bytes(
            b'{"run":"accepted"}'), "bytes": len(b'{"run":"accepted"}')}]
        assert (objects / "receipt.json").read_bytes() == b'{"run":"accepted"}'
        assert any(event == "private_resume_checkpoint" for event, _ in evidence.events)
    finally:
        database.close()
