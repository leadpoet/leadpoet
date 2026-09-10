from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from leadpoet_canonical.production_parity import ProductionParityError
from scripts import production_parity_snapshot as snapshot


def _stub_restore_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    migration_delta: list[dict[str, object]],
) -> dict[str, int]:
    shape = {
        "relation_count": 1,
        "total_relation_bytes": 8192,
        "largest_relation_bytes": 8192,
    }
    monkeypatch.setattr(
        snapshot,
        "verify_snapshot",
        lambda **_kwargs: {"migration_delta": migration_delta},
    )
    monkeypatch.setattr(
        snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {
            "capture_mode": "schema-only",
            "database": shape,
        },
    )
    monkeypatch.setattr(snapshot, "validate_snapshot_manifest", lambda value: value)
    monkeypatch.setattr(
        snapshot,
        "_database_relation_shape",
        lambda *_args, **_kwargs: dict(shape),
    )
    return shape


def _restore_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "root": tmp_path,
        "contract_path": tmp_path / "contract.json",
        "manifest_path": tmp_path / "manifest.json",
        "archive_path": tmp_path / "snapshot.dump",
        "target_dsn": (
            "postgresql://postgres:fixture@127.0.0.1:32768/"
            "leadpoet_parity_statistics"
        ),
        "production_host": "db.production.example",
        "timeout_seconds": 73,
    }


def test_restore_analyzes_after_unchanged_candidate_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    migration = tmp_path / "999-statistics-fixture.sql"
    migration.write_bytes(b"SELECT 1;\n")
    migration_bytes = migration.read_bytes()
    delta = [{"path": migration.name, "sha256": snapshot.file_sha256(migration)}]
    _stub_restore_contract(monkeypatch, tmp_path, migration_delta=delta)
    calls: list[dict[str, object]] = []

    def run_postgres(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(snapshot, "_run_postgres", run_postgres)

    result = snapshot.restore_snapshot(**_restore_kwargs(tmp_path))

    assert [call["command"][0] for call in calls] == [
        "pg_restore",
        "psql",
        "psql",
    ]
    assert calls[1]["command"][-1] == str(migration)
    assert calls[2]["command"] == [
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "ANALYZE",
    ]
    assert calls[2]["timeout"] == 73
    assert calls[2].get("mounts", ()) == ()
    assert calls[2]["env"]["PGSSLMODE"] == "disable"
    assert result["database_statistics_analyzed"] is True
    assert migration.read_bytes() == migration_bytes


def test_analyze_failure_prevents_restore_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_restore_contract(monkeypatch, tmp_path, migration_delta=[])
    commands: list[list[str]] = []

    def run_postgres(command, **_kwargs):
        normalized = list(command)
        commands.append(normalized)
        return subprocess.CompletedProcess(
            command,
            9 if normalized[-1] == "ANALYZE" else 0,
            stdout=b"",
            stderr=(
                b"statistics fixture failure"
                if normalized[-1] == "ANALYZE"
                else b""
            ),
        )

    monkeypatch.setattr(snapshot, "_run_postgres", run_postgres)

    with pytest.raises(
        ProductionParityError,
        match=(
            "restored database statistics analysis failed: "
            "statistics fixture failure"
        ),
    ):
        snapshot.restore_snapshot(**_restore_kwargs(tmp_path))

    assert commands[-1][-1] == "ANALYZE"
