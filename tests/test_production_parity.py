from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
import yaml

from gateway.tee import supabase_schema_preflight_v2 as schema_preflight

from leadpoet_canonical.production_parity import (
    SNAPSHOT_SCHEMA_VERSION,
    ProductionParityError,
    StageLedger,
    safe_database_target,
    sha256_json,
    validate_ledger,
    validate_snapshot_manifest,
)
from scripts.materialize_production_parity_secrets import (
    build_gateway_environment,
    is_process_control_environment_key,
    production_parity_scoring_cache_dir,
    production_parity_trace_prefixes,
)
from scripts import production_parity_snapshot as parity_snapshot
from scripts import run_local_restart_rehearsal as restart_rehearsal
from scripts import run_production_parity_fast as fast_parity
from scripts import run_production_parity_full_host as full_host
from scripts.production_parity_snapshot import (
    DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS,
    DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS,
    FULL_SNAPSHOT_DISK_RESERVE_BYTES,
    MAX_SNAPSHOT_IO_TIMEOUT_SECONDS,
    _require_full_snapshot_disk_headroom,
    _snapshot_io_timeout_seconds,
    capture_snapshot,
    restore_snapshot,
)
from scripts.run_production_parity_full_host import (
    FullParityError,
    _arena_provider_keys,
    _builtwith_key_from_secret,
    _clone_arena_service_role_key,
    _clone_service_role_key,
    _current_epoch_from_readiness,
    _dsn_from_secret,
    _parse_gateway_environment_file,
    _required_secret_from_environment,
    _verify_builtwith_credential_live,
)
from scripts.run_production_parity_fast import (
    _ProductionReadOnlySupabaseProvider,
)
from scripts.setup_production_parity_staging import (
    DEFAULT_REPOSITORY,
    _controller_policy,
    _runner_policy,
    _verify_readonly_dsn,
)
from scripts.resolve_production_parity_controller_requirements import (
    resolve_controller_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
HASH = "sha256:" + "b" * 64
ORIGIN = "https://d111111abcdef8.cloudfront.net"
PARITY_GATEWAY_PUBLIC_KEY = "c" * 64


def test_setup_targets_the_authoritative_github_repository():
    assert DEFAULT_REPOSITORY == "leadpoet/leadpoet"


def test_setup_keeps_readonly_database_password_out_of_process_arguments(
    monkeypatch,
):
    observed = {}
    credential = "c" * 64

    class TrustedPsqlFixture:
        def resolve(self, *, strict):
            assert strict is True
            return self

        def stat(self):
            return Path(sys.executable).stat()

        def __fspath__(self):
            return sys.executable

        def __str__(self):
            return "/opt/homebrew/Cellar/libpq/test/bin/psql"

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        observed["input"] = kwargs["input"]
        observed["start_new_session"] = kwargs["start_new_session"]
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "role_name": "leadpoet_parity_reader",
                    "read_only": True,
                    "superuser": False,
                    "bypass_rls": True,
                    "createdb": False,
                    "createrole": False,
                    "inherit": False,
                    "replication": False,
                    "connection_limit": 2,
                    "table_write_capable": False,
                    "sequence_write_capable": False,
                    "schema_create_capable": False,
                    "membership_count": 0,
                    "public_relation_count": 1,
                }
            ),
        )

    monkeypatch.setattr(
        "scripts.setup_production_parity_staging.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.setup_production_parity_staging.LOCAL_PSQL",
        TrustedPsqlFixture(),
    )
    _verify_readonly_dsn(
        "postgresql://leadpoet_parity_reader.qplwoislplkcegvdmbim:"
        + credential
        + "@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
        "?sslmode=require"
    )

    assert all(credential not in item for item in observed["argv"])
    assert observed["argv"][0] == "/opt/homebrew/Cellar/libpq/test/bin/psql"
    assert credential not in json.dumps(observed["env"])
    assert "PGPASSWORD" not in observed["env"]
    assert "PGPASSFILE" not in observed["env"]
    assert observed["input"] == credential + "\n"
    assert observed["start_new_session"] is True
    assert observed["env"]["PGSSLMODE"] == "require"
    assert observed["env"]["PGCONNECT_TIMEOUT"] == "15"


def _snapshot(*, bypass_rls: bool = True, capture_mode: str = "full") -> dict:
    captured = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    body = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_environment": "production-read-only",
        "source_host_hash": HASH,
        "capture_sha": SHA,
        "capture_contract_hash": HASH,
        "source_sha": "c" * 40,
        "captured_at": captured.isoformat(),
        "expires_at": (captured + timedelta(hours=2)).isoformat(),
        "capture_transaction_read_only": True,
        "capture_mode": capture_mode,
        "archive": {
            "format": (
                "postgres-custom"
                if capture_mode == "full"
                else "postgres-schema-custom"
            ),
            "storage": "ephemeral-encrypted-volume",
            "persisted": False,
            "sha256": HASH,
            "size_bytes": 40 * 1024 * 1024,
        },
        "database": {
            "server_version_num": "150010",
            "relation_count": 312,
            "total_relation_bytes": 9_000_000_000,
            "largest_relation_bytes": 2_000_000_000,
            "capture_utc_date": "2026-08-15",
            "target_rebenchmark_date": "2026-08-16",
            "latest_completed_benchmark_date": "2026-08-14",
            "current_day_rebenchmark_run_count": 1,
            "current_day_benchmark_bundle_count": 1,
            "source_role": {
                "role_hash": HASH,
                "transaction_read_only": True,
                "superuser": False,
                "bypass_rls": bypass_rls,
                "replication": False,
                "table_write_capable": False,
            },
            "weight_history_scope": {
                "netuid": 71,
                "start_epoch": 24000,
                "end_epoch": 24570,
                "expected_rows": 571,
            },
        },
        "migrations": [
            {
                "path": "scripts/001-example.sql",
                "sequence": 1,
                "sha256": HASH,
                "transaction_mode": "candidate-file",
            }
        ],
        "data_classification": "production-confidential-ephemeral",
    }
    return {**body, "manifest_hash": sha256_json(body)}


def test_snapshot_binds_real_scale_future_day_and_readonly_role():
    value = validate_snapshot_manifest(
        _snapshot(), now=datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc)
    )
    assert value["archive"]["size_bytes"] == 40 * 1024 * 1024
    assert value["archive"]["persisted"] is False
    assert value["database"]["target_rebenchmark_date"] == "2026-08-16"
    assert value["database"]["source_role"]["bypass_rls"] is True
    assert value["database"]["source_role"]["table_write_capable"] is False
    assert value["database"]["weight_history_scope"]["expected_rows"] == 571


def test_schema_only_snapshot_is_explicit_and_cannot_claim_full_data():
    value = validate_snapshot_manifest(
        _snapshot(capture_mode="schema-only"),
        now=datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc),
    )
    assert value["capture_mode"] == "schema-only"
    assert value["archive"]["format"] == "postgres-schema-custom"
    mismatched = _snapshot(capture_mode="schema-only")
    mismatched["archive"]["format"] = "postgres-custom"
    body = {key: item for key, item in mismatched.items() if key != "manifest_hash"}
    mismatched["manifest_hash"] = sha256_json(body)
    with pytest.raises(ProductionParityError, match="archive format"):
        validate_snapshot_manifest(
            mismatched,
            now=datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc),
        )


def test_snapshot_io_defaults_stay_short_and_explicit():
    assert DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS == 900
    assert (
        inspect.signature(capture_snapshot).parameters["timeout_seconds"].default == 900
    )
    assert (
        inspect.signature(restore_snapshot).parameters["timeout_seconds"].default == 900
    )
    assert (
        _snapshot_io_timeout_seconds(MAX_SNAPSHOT_IO_TIMEOUT_SECONDS)
        == MAX_SNAPSHOT_IO_TIMEOUT_SECONDS
    )
    for invalid in (True, 0, MAX_SNAPSHOT_IO_TIMEOUT_SECONDS + 1):
        with pytest.raises(ProductionParityError, match="timeout is invalid"):
            _snapshot_io_timeout_seconds(invalid)


def test_pinned_postgres_client_is_confined_and_redacts_environment(
    monkeypatch,
    tmp_path: Path,
):
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    archive.write_bytes(b"snapshot")
    image = "postgres@sha256:" + "c" * 64
    pg_env = {
        "PGHOST": "database-host-sentinel",
        "PGPORT": "6543",
        "PGDATABASE": "database-name-sentinel",
        "PGUSER": "database-user-sentinel",
        "PGPASSWORD": "database-password-sentinel",
        "PGSSLMODE": "require",
        "PGOPTIONS": "options-sentinel",
        "AWS_SECRET_ACCESS_KEY": "ambient-aws-secret",
        "LEADPOET_SENTRY_API_TOKEN": "ambient-sentry-secret",
    }
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = kwargs
        diagnostic = "|".join(
            str(pg_env[key]) for key in parity_snapshot._POSTGRES_ENVIRONMENT_KEYS
        ).encode()
        return subprocess.CompletedProcess(
            command,
            7,
            stdout=b"",
            stderr=diagnostic,
        )

    monkeypatch.setattr(parity_snapshot.subprocess, "run", fake_run)
    result = parity_snapshot._run_postgres(
        ["pg_restore", "--list", parity_snapshot._POSTGRES_ARCHIVE_TARGET],
        env=pg_env,
        timeout=120,
        postgres_image=image,
        mounts=(
            parity_snapshot._PostgresClientMount(
                source=archive,
                target=parity_snapshot._POSTGRES_ARCHIVE_TARGET,
                read_only=True,
            ),
        ),
    )

    command = observed["command"]
    kwargs = observed["kwargs"]
    assert command[:4] == ["docker", "run", "--rm", "--name"]
    assert command[command.index("--network") + 1] == "host"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--entrypoint") + 1] == "pg_restore"
    assert command[command.index("--entrypoint") + 2] == image
    mount_values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--mount"
    ]
    assert mount_values == [
        (
            f"type=bind,src={archive.resolve()},"
            f"dst={parity_snapshot._POSTGRES_ARCHIVE_TARGET},readonly"
        )
    ]
    for key in parity_snapshot._POSTGRES_ENVIRONMENT_KEYS:
        index = command.index(key)
        assert command[index - 1] == "--env"
        assert str(pg_env[key]) not in command
        assert str(pg_env[key]).encode() not in (result.stderr or b"")
    assert kwargs["env"] == {
        "PATH": os.environ.get("PATH") or os.defpath,
        **{
            key: pg_env[key]
            for key in parity_snapshot._POSTGRES_ENVIRONMENT_KEYS
        },
    }
    assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
    assert "LEADPOET_SENTRY_API_TOKEN" not in kwargs["env"]
    mount_sources = {
        field.removeprefix("src=")
        for value in mount_values
        for field in value.split(",")
        if field.startswith("src=")
    }
    assert mount_sources == {str(archive.resolve())}
    assert str(archive.parent.resolve()) not in mount_sources


@pytest.mark.parametrize(
    ("image", "command", "error"),
    (
        ("postgres:15", ["psql"], "digest-pinned"),
        ("postgres@sha256:" + "c" * 64, ["bash"], "not allowed"),
    ),
)
def test_pinned_postgres_client_rejects_untrusted_execution_before_subprocess(
    monkeypatch,
    image: str,
    command: list[str],
    error: str,
):
    monkeypatch.setattr(
        parity_snapshot.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )
    with pytest.raises(ProductionParityError, match=error):
        parity_snapshot._run_postgres(
            command,
            env={},
            timeout=30,
            postgres_image=image,
        )


def test_postgres_client_host_mode_delegates_byte_for_byte(monkeypatch):
    command = ["psql", "-X", "-c", "SELECT 1"]
    env = {"PGHOST": "127.0.0.1", "PGPASSWORD": "host-mode-sentinel"}
    result = subprocess.CompletedProcess(command, 0, stdout=b"1\n", stderr=b"")
    observed: dict[str, object] = {}

    def fake_run(received, **kwargs):
        observed["command"] = received
        observed.update(kwargs)
        return result

    monkeypatch.setattr(parity_snapshot, "_run", fake_run)
    assert parity_snapshot._run_postgres(
        command,
        env=env,
        timeout=37,
        stdin=b"input",
        postgres_image=None,
    ) is result
    assert observed == {
        "command": command,
        "env": env,
        "timeout": 37,
        "stdin": b"input",
    }


@pytest.mark.parametrize(
    "interruption",
    (
        subprocess.TimeoutExpired(cmd=["docker"], timeout=1),
        KeyboardInterrupt(),
    ),
)
def test_pinned_postgres_client_interrupt_removes_exact_container(
    monkeypatch,
    interruption: BaseException,
):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        if len(calls) == 1:
            raise interruption
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(parity_snapshot.subprocess, "run", fake_run)
    with pytest.raises(type(interruption)):
        parity_snapshot._run_postgres(
            ["psql", "-c", "SELECT 1"],
            env={"PGPASSWORD": "cleanup-password-sentinel"},
            timeout=1,
            postgres_image="postgres@sha256:" + "c" * 64,
        )

    container_name = calls[0][0][calls[0][0].index("--name") + 1]
    assert calls[1][0] == ["docker", "rm", "-f", container_name]
    assert calls[1][1]["env"] == {
        "PATH": os.environ.get("PATH") or os.defpath
    }
    assert calls[2][0] == [
        "docker",
        "container",
        "ls",
        "--all",
        "--quiet",
        "--filter",
        f"name=^/{container_name}$",
    ]
    assert calls[2][1]["env"] == {
        "PATH": os.environ.get("PATH") or os.defpath
    }
    assert "cleanup-password-sentinel" not in calls[1][0]
    assert "cleanup-password-sentinel" not in calls[2][0]


@pytest.mark.parametrize(
    ("remove_result", "probe_result", "probe_error", "cleanup_proven"),
    (
        (1, 0, None, True),
        (0, 1, None, False),
        (0, 0, b"still-present\n", False),
        (0, None, OSError("docker unavailable"), False),
    ),
)
def test_pinned_postgres_client_interrupt_requires_proven_absence(
    monkeypatch,
    remove_result: int,
    probe_result: int | None,
    probe_error: BaseException | bytes | None,
    cleanup_proven: bool,
):
    interruption = subprocess.TimeoutExpired(cmd=["docker"], timeout=1)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            raise interruption
        if len(calls) == 2:
            return subprocess.CompletedProcess(
                command,
                remove_result,
                stdout=b"",
                stderr=b"remove failed" if remove_result else b"",
            )
        if isinstance(probe_error, BaseException):
            raise probe_error
        return subprocess.CompletedProcess(
            command,
            int(probe_result or 0),
            stdout=probe_error if isinstance(probe_error, bytes) else b"",
            stderr=b"probe failed" if probe_result else b"",
        )

    monkeypatch.setattr(parity_snapshot.subprocess, "run", fake_run)
    expected = subprocess.TimeoutExpired if cleanup_proven else ProductionParityError
    with pytest.raises(expected):
        parity_snapshot._run_postgres(
            ["psql", "-c", "SELECT 1"],
            env={"PGPASSWORD": "cleanup-password-sentinel"},
            timeout=1,
            postgres_image="postgres@sha256:" + "c" * 64,
        )

    assert len(calls) == 3
    assert calls[1][:3] == ["docker", "rm", "-f"]
    assert calls[2][:5] == ["docker", "container", "ls", "--all", "--quiet"]


def test_capture_snapshot_routes_every_postgres_call_through_pinned_image(
    monkeypatch,
    tmp_path: Path,
):
    image = "postgres@sha256:" + "c" * 64
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    (archive.parent / "runtime-config.json").write_text(
        "secret-neighbor", encoding="utf-8"
    )
    contract = {
        "candidate_sha": SHA,
        "contract_hash": HASH,
    }
    stats = {
        "server_version_num": "150008",
        "relation_count": 163,
        "total_relation_bytes": 176_022_568_960,
        "largest_relation_bytes": 48_022_609_920,
        "capture_utc_timestamp": "2026-08-18T12:00:00+00:00",
        "capture_utc_date": "2026-08-18",
        "latest_completed_benchmark_date": "2026-08-17",
        "current_day_rebenchmark_run_count": 0,
        "current_day_benchmark_bundle_count": 0,
        "weight_history_scope": {"netuid": 71},
        "source_role": {
            "role_name": "readonly",
            "transaction_read_only": True,
            "superuser": False,
            "bypass_rls": False,
            "replication": False,
            "table_write_capable": False,
        },
    }
    calls: list[dict[str, object]] = []

    def fake_run_postgres(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        if command[0] == "pg_dump":
            mounts = kwargs["mounts"]
            assert len(mounts) == 1
            Path(mounts[0].source).write_bytes(b"production-shaped-dump")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[-1] == "SHOW transaction_read_only":
            stdout = b"on\n"
        else:
            stdout = json.dumps(stats).encode()
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)
    monkeypatch.setattr(parity_snapshot, "_load_json", lambda *_a, **_k: contract)
    monkeypatch.setattr(parity_snapshot, "validate_contract", lambda value: value)
    monkeypatch.setattr(
        parity_snapshot,
        "validate_snapshot_manifest",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(parity_snapshot, "_source_migrations", lambda **_kwargs: [])
    monkeypatch.setattr(
        parity_snapshot,
        "_require_full_snapshot_disk_headroom",
        lambda *_args, **_kwargs: {},
    )

    capture_snapshot(
        contract_path=tmp_path / "contract.json",
        archive_path=archive,
        manifest_path=tmp_path / "snapshot-manifest.json",
        dsn="postgresql://reader:secret@db.production.example/postgres",
        expected_production_host="db.production.example",
        ttl_hours=24,
        source_sha="c" * 40,
        postgres_image=image,
    )

    assert [call["command"][0] for call in calls] == [
        "psql",
        "psql",
        "pg_dump",
        "psql",
    ]
    assert [call["timeout"] for call in calls] == [600, 60, 900, 60]
    assert all(call["postgres_image"] == image for call in calls)
    dump = calls[2]
    assert dump["command"][-1] == parity_snapshot._POSTGRES_ARCHIVE_TARGET
    assert len(dump["mounts"]) == 1
    assert dump["mounts"][0] == parity_snapshot._PostgresClientMount(
        source=archive.resolve(),
        target=parity_snapshot._POSTGRES_ARCHIVE_TARGET,
        read_only=False,
    )
    assert all(not call.get("mounts") for call in (calls[0], calls[1], calls[3]))
    assert archive.stat().st_mode & 0o777 == 0o600


def test_database_stats_does_not_require_candidate_arena_schema(monkeypatch):
    observed = {}
    value = {
        "latest_completed_benchmark_date": None,
        "current_day_rebenchmark_run_count": 0,
        "current_day_benchmark_bundle_count": 0,
        "source_role": {
            "role_name": "readonly",
            "transaction_read_only": True,
            "superuser": False,
            "bypass_rls": False,
            "replication": False,
            "table_write_capable": False,
        },
    }

    def fake_run_postgres(command, **kwargs):
        observed["sql"] = command[-1]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(value).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    stats = parity_snapshot._database_stats({})

    assert "lab_arena_rounds" not in observed["sql"]
    assert stats["latest_completed_benchmark_date"] is None
    assert stats["current_day_rebenchmark_run_count"] == 0
    assert stats["current_day_benchmark_bundle_count"] == 0


def test_isolated_snapshot_restore_disables_ssl_after_target_validation(
    monkeypatch,
    tmp_path: Path,
):
    observed: dict[str, object] = {}
    original_safe_database_target = parity_snapshot.safe_database_target

    def record_safe_database_target(dsn: str, *, production_host: str) -> None:
        original_safe_database_target(dsn, production_host=production_host)
        observed["target_validated"] = True

    def fake_run(command, *, env, timeout, stdin=None):
        assert observed.get("target_validated") is True
        observed["command"] = list(command)
        observed["env"] = dict(env)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        parity_snapshot,
        "verify_snapshot",
        lambda **_kwargs: {"migration_delta": []},
    )
    monkeypatch.setattr(
        parity_snapshot,
        "safe_database_target",
        record_safe_database_target,
    )
    monkeypatch.setattr(
        parity_snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {"capture_mode": "schema-only"},
    )
    monkeypatch.setattr(
        parity_snapshot,
        "validate_snapshot_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(parity_snapshot, "_run", fake_run)
    monkeypatch.setenv("PGSSLMODE", "verify-full")

    production_env, _ = parity_snapshot._postgres_env(
        "postgresql://reader:x@db.production.example/postgres",
        read_only=True,
    )
    restore_snapshot(
        root=tmp_path,
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=tmp_path / "snapshot.dump",
        target_dsn=("postgresql://postgres:x@127.0.0.1:32768/leadpoet_parity_test"),
        production_host="db.production.example",
    )

    assert production_env["PGSSLMODE"] == "require"
    assert observed["command"][0] == "pg_restore"
    assert observed["env"]["PGSSLMODE"] == "disable"
    assert observed["env"]["PGOPTIONS"] == "-c check_function_bodies=off"


def test_pinned_snapshot_verify_lists_only_the_archive_file(
    monkeypatch,
    tmp_path: Path,
):
    image = "postgres@sha256:" + "c" * 64
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    archive.write_bytes(b"snapshot")
    contract = {
        "base_sha": "c" * 40,
        "candidate_sha": SHA,
        "contract_hash": HASH,
        "migrations": [],
    }
    manifest = {
        "source_sha": contract["base_sha"],
        "capture_sha": contract["candidate_sha"],
        "capture_contract_hash": contract["contract_hash"],
        "source_host_hash": "sha256:" + "d" * 64,
        "manifest_hash": "sha256:" + "e" * 64,
        "archive": {"sha256": "sha256:" + "f" * 64},
        "migrations": [],
    }
    observed: dict[str, object] = {}

    def fake_load(_path, *, description):
        return contract if description == "parity contract" else manifest

    def fake_run_postgres(command, **kwargs):
        observed["command"] = list(command)
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"one\ntwo\n",
            stderr=b"",
        )

    monkeypatch.setattr(parity_snapshot, "_load_json", fake_load)
    monkeypatch.setattr(parity_snapshot, "validate_contract", lambda value: value)
    monkeypatch.setattr(
        parity_snapshot, "validate_snapshot_manifest", lambda value: value
    )
    monkeypatch.setattr(parity_snapshot, "validate_archive", lambda *_args: None)
    monkeypatch.setattr(parity_snapshot, "migration_delta", lambda **_kwargs: [])
    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    evidence = parity_snapshot.verify_snapshot(
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=archive,
        postgres_image=image,
    )

    assert observed["command"] == [
        "pg_restore",
        "--list",
        parity_snapshot._POSTGRES_ARCHIVE_TARGET,
    ]
    assert observed["postgres_image"] == image
    assert observed["mounts"] == (
        parity_snapshot._PostgresClientMount(
            source=archive,
            target=parity_snapshot._POSTGRES_ARCHIVE_TARGET,
            read_only=True,
        ),
    )
    assert evidence["archive_entries"] == 2


def test_pinned_snapshot_restore_mounts_only_archive_and_exact_migration(
    monkeypatch,
    tmp_path: Path,
):
    image = "postgres@sha256:" + "c" * 64
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    archive.write_bytes(b"snapshot")
    migration = tmp_path / "migrations" / "999-test.sql"
    migration.parent.mkdir()
    migration.write_text("SELECT 1;\n", encoding="utf-8")
    migration_hash = parity_snapshot.file_sha256(migration)
    calls: list[dict[str, object]] = []

    def fake_verify_snapshot(**kwargs):
        assert kwargs["postgres_image"] == image
        return {
            "migration_delta": [
                {"path": "migrations/999-test.sql", "sha256": migration_hash}
            ]
        }

    def fake_run_postgres(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(parity_snapshot, "verify_snapshot", fake_verify_snapshot)
    monkeypatch.setattr(
        parity_snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {"capture_mode": "schema-only"},
    )
    monkeypatch.setattr(
        parity_snapshot, "validate_snapshot_manifest", lambda value: value
    )
    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    restore_snapshot(
        root=tmp_path,
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=archive,
        target_dsn=(
            "postgresql://postgres:secret@127.0.0.1:32768/"
            "leadpoet_parity_test"
        ),
        production_host="db.production.example",
        postgres_image=image,
    )

    assert [call["command"][0] for call in calls] == ["pg_restore", "psql"]
    assert all(call["postgres_image"] == image for call in calls)
    assert "--dbname=" in calls[0]["command"]
    assert "leadpoet_parity_test" not in calls[0]["command"]
    assert "secret" not in calls[0]["command"]
    assert calls[0]["env"]["PGSSLMODE"] == "disable"
    assert calls[0]["mounts"] == (
        parity_snapshot._PostgresClientMount(
            source=archive,
            target=parity_snapshot._POSTGRES_ARCHIVE_TARGET,
            read_only=True,
        ),
    )
    assert calls[1]["command"][-1] == parity_snapshot._POSTGRES_MIGRATION_TARGET
    assert calls[1]["mounts"] == (
        parity_snapshot._PostgresClientMount(
            source=migration,
            target=parity_snapshot._POSTGRES_MIGRATION_TARGET,
            read_only=True,
        ),
    )


def _schema_only_source_add_maintenance_readback() -> dict[str, object]:
    return {
        "schema_version": (
            parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_SCHEMA_VERSION
        ),
        "initial_paused": False,
        "pause_rpc": "research_lab_source_add_set_paused",
        "control_rows": 1,
        "work_rows": 0,
        "paused": True,
        "guard_active": False,
        "guard_generation": 0,
        "reason_bound": True,
        "actor_bound": True,
    }


def _copy_cutover_migration(
    tmp_path: Path, migration_identity: dict[str, object]
) -> Path:
    migration = tmp_path / str(migration_identity["path"])
    migration.parent.mkdir(parents=True, exist_ok=True)
    migration.write_bytes((ROOT / str(migration_identity["path"])).read_bytes())
    assert parity_snapshot.file_sha256(migration) == migration_identity["sha256"]
    return migration


@pytest.mark.parametrize(
    "migration_identity",
    parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS,
    ids=lambda migration: f"migration-{migration['sequence']}",
)
def test_schema_only_restore_stages_exact_source_add_migration_precondition(
    monkeypatch,
    tmp_path: Path,
    migration_identity: dict[str, object],
):
    image = "postgres@sha256:" + "c" * 64
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    archive.write_bytes(b"snapshot")
    migration_identity = dict(migration_identity)
    migration = _copy_cutover_migration(tmp_path, migration_identity)
    maintenance = _schema_only_source_add_maintenance_readback()
    calls: list[dict[str, object]] = []

    def fake_run_postgres(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        stdout = b""
        if command[0] == "psql" and "-f" not in command:
            stdout = (json.dumps(maintenance, sort_keys=True) + "\n").encode()
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(
        parity_snapshot,
        "verify_snapshot",
        lambda **_kwargs: {"migration_delta": [migration_identity]},
    )
    monkeypatch.setattr(
        parity_snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {"capture_mode": "schema-only"},
    )
    monkeypatch.setattr(
        parity_snapshot, "validate_snapshot_manifest", lambda value: value
    )
    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    evidence = restore_snapshot(
        root=tmp_path,
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=archive,
        target_dsn=(
            "postgresql://postgres:secret@127.0.0.1:32768/"
            "leadpoet_parity_test"
        ),
        production_host="db.production.example",
        postgres_image=image,
    )

    assert [call["command"][0] for call in calls] == [
        "pg_restore",
        "psql",
        "psql",
    ]
    staging = calls[1]
    assert staging.get("mounts", ()) == ()
    assert "-f" not in staging["command"]
    staging_sql = staging["stdin"].decode("utf-8")
    assert "schema-only SOURCE_ADD control state is not empty" in staging_sql
    assert "schema-only SOURCE_ADD work state is not empty" in staging_sql
    assert "IN ACCESS EXCLUSIVE MODE NOWAIT" in staging_sql
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in staging_sql
    assert "research_lab_source_add_set_paused(" in staging_sql
    assert "FALSE," in staging_sql
    assert parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_REASON in staging_sql
    assert parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_MAINTENANCE_ACTOR in staging_sql
    applied = calls[2]
    assert applied["command"][-1] == parity_snapshot._POSTGRES_MIGRATION_TARGET
    assert applied["mounts"] == (
        parity_snapshot._PostgresClientMount(
            source=migration,
            target=parity_snapshot._POSTGRES_MIGRATION_TARGET,
            read_only=True,
        ),
    )
    assert evidence["clone_migration_preconditions"] == [
        {
            **maintenance,
            "migration_path": migration_identity["path"],
            "migration_sha256": migration_identity["sha256"],
        }
    ]


def test_schema_only_restore_stages_source_add_cutover_only_once(
    monkeypatch,
    tmp_path: Path,
):
    image = "postgres@sha256:" + "c" * 64
    archive = tmp_path / "runtime" / "production.dump"
    archive.parent.mkdir()
    archive.write_bytes(b"snapshot")
    migrations = [
        dict(migration)
        for migration in parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS
    ]
    migration_paths = [
        _copy_cutover_migration(tmp_path, migration) for migration in migrations
    ]
    maintenance = _schema_only_source_add_maintenance_readback()
    calls: list[dict[str, object]] = []

    def fake_run_postgres(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        stdout = b""
        if command[0] == "psql" and "-f" not in command:
            stdout = (json.dumps(maintenance, sort_keys=True) + "\n").encode()
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(
        parity_snapshot,
        "verify_snapshot",
        lambda **_kwargs: {"migration_delta": migrations},
    )
    monkeypatch.setattr(
        parity_snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {"capture_mode": "schema-only"},
    )
    monkeypatch.setattr(
        parity_snapshot, "validate_snapshot_manifest", lambda value: value
    )
    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    evidence = restore_snapshot(
        root=tmp_path,
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=archive,
        target_dsn=(
            "postgresql://postgres:secret@127.0.0.1:32768/"
            "leadpoet_parity_test"
        ),
        production_host="db.production.example",
        postgres_image=image,
    )

    assert [call["command"][0] for call in calls] == [
        "pg_restore",
        "psql",
        *(["psql"] * len(migrations)),
    ]
    assert "-f" not in calls[1]["command"]
    assert [
        calls[index]["mounts"][0].source
        for index in range(2, 2 + len(migrations))
    ] == migration_paths
    assert evidence["clone_migration_preconditions"] == [
        {
            **maintenance,
            "migration_path": migrations[0]["path"],
            "migration_sha256": migrations[0]["sha256"],
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sequence", 174),
        ("sha256", "sha256:" + "0" * 64),
        ("transaction_mode", "autocommit"),
    ),
)
def test_schema_only_source_add_cutover_rejects_malformed_identity(field, value):
    migration = dict(
        parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS[-1]
    )
    migration[field] = value
    with pytest.raises(
        ProductionParityError,
        match="schema-only SOURCE_ADD maintenance migration identity differs",
    ):
        parity_snapshot._schema_only_source_add_maintenance_sql(migration)


def test_schema_only_source_add_acl_is_exact_migration_bound():
    migrations = parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
    sql = parity_snapshot._schema_only_source_add_acl_sql(migrations).decode(
        "utf-8"
    )

    duplicate_privacy_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/171-research-lab-source-add-duplicate-privacy.sql"
        )
    )
    assert duplicate_privacy_migration["sha256"] in sql
    provenance_leg1_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/175-research-lab-source-add-provenance-leg1.sql"
        )
    )
    assert provenance_leg1_migration["sha256"] in sql
    provenance_origin_repair_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
        )
    )
    assert provenance_origin_repair_migration["sha256"] in sql
    provenance_authority_acl_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/177-research-lab-source-add-provenance-authority-acl.sql"
        )
    )
    assert provenance_authority_acl_migration["sha256"] in sql
    miner_status_migration = parity_snapshot._schema_only_source_add_acl_migration(
        "scripts/178-research-lab-source-add-miner-status.sql"
    )
    assert miner_status_migration["sha256"] in sql
    assert (
        "public.research_lab_source_add_admit_v3"
        "(jsonb,text,text,text,text,text,integer,integer,integer,integer)"
    ) in sql
    assert "public.research_lab_source_add_finish_work" in sql
    assert "public.research_lab_source_add_begin_provider_execution" in sql
    assert "public.research_lab_source_add_requeue_provenance_v2" in sql
    assert "public.enforce_research_lab_source_add_leg1_obligation_v2()" in sql
    assert "public.research_lab_source_add_finalize_leg1_v4" in sql
    assert "public.research_lab_source_add_post_accept_leg1_contract_v4()" in sql
    assert (
        "public.research_lab_source_add_miner_status_page_v1"
        "(text,text,integer)"
    ) in sql
    assert "public.research_lab_source_add_miner_status_contract_v1()" in sql
    assert (
        "REVOKE ALL ON TABLE public.research_lab_source_add_miner_status_v1"
        in sql
    )
    assert (
        "GRANT SELECT ON TABLE public.research_lab_source_add_miner_status_v1"
        in sql
    )
    assert "'security_invoker=true', 'security_barrier=true'" in sql
    assert "miner_status_view_acl_bound" in sql
    assert "provenance_leg1_trigger_authority_bound" in sql
    assert "provenance_leg1_view_authority_bound" in sql
    assert "provenance_origin_repair_authority_bound" in sql
    assert "provenance_leg1_policy_bound" in sql
    assert "schema-only SOURCE_ADD ACL function inventory differs" in sql
    assert "schema-only SOURCE_ADD ACL readback differs" in sql
    assert "pg_catalog.aclexplode" in sql
    assert "FROM PUBLIC, anon, authenticated, service_role" in sql
    assert "TO PUBLIC" in sql
    assert len(parity_snapshot._schema_only_source_add_acl_expectations()) == 79

    rewritten = [dict(item) for item in migrations]
    next(
        item
        for item in rewritten
        if item["path"]
        == "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql"
    )["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(
        ProductionParityError,
        match="schema-only SOURCE_ADD ACL migration identity differs",
    ):
        parity_snapshot._schema_only_source_add_acl_sql(rewritten)

    extended = [
        *migrations,
        {
            **migrations[-1],
            "path": "scripts/179-next.sql",
            "sequence": 179,
        },
    ]
    parity_snapshot._schema_only_source_add_acl_sql(extended)


def test_database_shape_capture_does_not_require_candidate_arena_tables(monkeypatch):
    observed = {}

    def fake_run_postgres(command, **kwargs):
        observed["sql"] = command[-1]
        value = {
            "server_version_num": "150010",
            "relation_count": 1,
            "total_relation_bytes": 1,
            "largest_relation_bytes": 1,
            "capture_utc_timestamp": "2026-09-04T00:00:00+00:00",
            "capture_utc_date": "2026-09-04",
            "latest_completed_benchmark_date": None,
            "current_day_rebenchmark_run_count": 0,
            "current_day_benchmark_bundle_count": 0,
            "weight_history_scope": None,
            "source_role": {
                "role_name": "readonly",
                "transaction_read_only": True,
                "superuser": False,
                "bypass_rls": False,
                "replication": False,
                "table_write_capable": False,
            },
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(value).encode(), stderr=b""
        )

    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)
    result = parity_snapshot._database_stats(
        {"PGHOST": "db.production.example"},
        postgres_image="postgres@sha256:" + "a" * 64,
    )

    assert "FROM public.lab_arena_rounds" not in observed["sql"]
    assert result["latest_completed_benchmark_date"] is None
    assert result["current_day_rebenchmark_run_count"] == 0
    assert result["current_day_benchmark_bundle_count"] == 0


def test_schema_only_source_add_acl_readback_is_exhaustive_and_compact(
    monkeypatch,
):
    expectations = parity_snapshot._schema_only_source_add_acl_expectations()
    duplicate_privacy_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/171-research-lab-source-add-duplicate-privacy.sql"
        )
    )
    provenance_leg1_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/175-research-lab-source-add-provenance-leg1.sql"
        )
    )
    provenance_origin_repair_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
        )
    )
    provenance_authority_acl_migration = (
        parity_snapshot._schema_only_source_add_acl_migration(
            "scripts/177-research-lab-source-add-provenance-authority-acl.sql"
        )
    )
    miner_status_migration = parity_snapshot._schema_only_source_add_acl_migration(
        "scripts/178-research-lab-source-add-miner-status.sql"
    )
    readback = {
        "schema_version": parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_SCHEMA_VERSION,
        "migration_count": len(
            parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
        ),
        "migration_171_sha256": duplicate_privacy_migration["sha256"],
        "migration_175_sha256": provenance_leg1_migration["sha256"],
        "migration_176_sha256": provenance_origin_repair_migration["sha256"],
        "migration_177_sha256": provenance_authority_acl_migration["sha256"],
        "migration_178_sha256": miner_status_migration["sha256"],
        "function_signature_count": len(expectations),
        "service_role_function_count": sum(
            privileges["service_role_callable"]
            for privileges in expectations.values()
        ),
        "non_service_role_function_count": sum(
            not privileges["service_role_callable"]
            for privileges in expectations.values()
        ),
        "public_function_count": sum(
            privileges["public_callable"] for privileges in expectations.values()
        ),
        "anon_callable_function_count": sum(
            privileges["anon_callable"] for privileges in expectations.values()
        ),
        "authenticated_callable_function_count": sum(
            privileges["authenticated_callable"]
            for privileges in expectations.values()
        ),
        "miner_status_view_acl_bound": True,
        "function_acl_inventory": expectations,
        "duplicate_privacy_authority_bound": True,
        "duplicate_privacy_permissions_bound": True,
        "post_accept_leg1_authority_bound": True,
        "provenance_leg1_trigger_authority_bound": True,
        "provenance_leg1_view_authority_bound": True,
        "provenance_origin_repair_authority_bound": True,
        "provenance_leg1_policy_bound": True,
        "post_accept_leg1_permissions_bound": True,
        "claim_control_authority_bound": True,
        "claim_control_permissions_bound": True,
    }

    monkeypatch.setattr(parity_snapshot, "safe_database_target", lambda *_a, **_k: None)
    monkeypatch.setattr(
        parity_snapshot,
        "_postgres_env",
        lambda *_a, **_k: ({}, "127.0.0.1"),
    )

    def postgres_readback(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            (), 0, stdout=json.dumps(readback).encode("utf-8"), stderr=b""
        )

    monkeypatch.setattr(parity_snapshot, "_run_postgres", postgres_readback)
    evidence = parity_snapshot.restore_schema_only_source_add_acl_contract(
        target_dsn="postgresql://postgres:x@127.0.0.1/leadpoet_parity_test",
        production_host="db.production.example",
        candidate_migrations=parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS,
    )
    assert "function_acl_inventory" not in evidence
    assert evidence["function_acl_inventory_sha256"] == sha256_json(
        {"functions": expectations}
    )

    readback["function_acl_inventory"] = {
        **expectations,
        "public.research_lab_source_add_finish_work"
        "(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,"
        "timestamp with time zone,boolean)": {
            **expectations[
                "public.research_lab_source_add_finish_work"
                "(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,"
                "timestamp with time zone,boolean)"
            ],
            "anon_callable": True,
        },
    }
    with pytest.raises(
        ProductionParityError,
        match="schema-only SOURCE_ADD ACL readback differs",
    ):
        parity_snapshot.restore_schema_only_source_add_acl_contract(
            target_dsn="postgresql://postgres:x@127.0.0.1/leadpoet_parity_test",
            production_host="db.production.example",
            candidate_migrations=(
                parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_ACL_MIGRATIONS
            ),
        )


def test_full_restore_does_not_stage_schema_only_source_add_state(
    monkeypatch,
    tmp_path: Path,
):
    archive = tmp_path / "production.dump"
    archive.write_bytes(b"snapshot")
    migration_identity = dict(
        parity_snapshot._SCHEMA_ONLY_SOURCE_ADD_CUTOVER_MIGRATIONS[-1]
    )
    migration = _copy_cutover_migration(tmp_path, migration_identity)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        parity_snapshot,
        "verify_snapshot",
        lambda **_kwargs: {"migration_delta": [migration_identity]},
    )
    monkeypatch.setattr(
        parity_snapshot,
        "_load_json",
        lambda *_args, **_kwargs: {
            "capture_mode": "full",
            "database": {"total_relation_bytes": 1},
        },
    )
    monkeypatch.setattr(
        parity_snapshot, "validate_snapshot_manifest", lambda value: value
    )
    monkeypatch.setattr(
        parity_snapshot,
        "_require_full_snapshot_disk_headroom",
        lambda *_args, **_kwargs: {},
    )

    def fake_run_postgres(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(parity_snapshot, "_run_postgres", fake_run_postgres)

    evidence = restore_snapshot(
        root=tmp_path,
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=archive,
        target_dsn=(
            "postgresql://postgres:secret@127.0.0.1:32768/"
            "leadpoet_parity_test"
        ),
        production_host="db.production.example",
    )

    assert [command[0] for command in calls] == ["pg_restore", "psql"]
    assert "-f" in calls[1]
    assert "clone_migration_preconditions" not in evidence


def test_disposable_clone_bootstraps_exact_supabase_restore_prerequisites(
    monkeypatch,
):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )
    expected = {
        "anon_role",
        "authenticated_role",
        "service_role",
        "auth_schema",
        "extensions_schema",
        "pgcrypto_extension",
        "auth_role_function",
        "auth_jwt_function",
    }
    observed: list[str] = []

    def fake_psql(sql: str, *, timeout: int = 120) -> str:
        assert timeout == 120
        observed.append(sql)
        return "" if len(observed) == 1 else json.dumps({key: True for key in expected})

    monkeypatch.setattr(database, "_psql", fake_psql)

    assert database.prepare_snapshot_restore() == {
        key: True for key in sorted(expected)
    }
    bootstrap = observed[0]
    for role in ("anon", "authenticated", "service_role"):
        assert f"CREATE ROLE {role} NOLOGIN INHERIT" in bootstrap
    assert "CREATE SCHEMA IF NOT EXISTS auth" in bootstrap
    assert "CREATE SCHEMA IF NOT EXISTS extensions" in bootstrap
    assert "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions" in bootstrap
    for function in ("auth.role()", "auth.jwt()"):
        assert f"FUNCTION {function}" in bootstrap
    assert "auth.users" not in bootstrap
    assert "auth.uid" not in bootstrap
    assert "net.http" not in bootstrap


def test_disposable_clone_waits_for_final_tcp_postmaster(monkeypatch):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )
    postmaster_states = iter(("bootstrap-ready", "bootstrap-shutdown", "final-ready"))
    observed_states: list[str] = []
    readiness_commands: list[list[str]] = []
    docker_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        docker_commands.append(command)
        if command[:4] == ["docker", "exec", database.postgres, "pg_isready"]:
            state = next(postmaster_states)
            observed_states.append(state)
            readiness_commands.append(command)
            # The image bootstrap postmaster is socket-only. An old socket probe
            # would accept its first ready state; TCP becomes ready only after the
            # bootstrap shutdown and the final postmaster exec.
            tcp_probe = (
                "-h" in command and command[command.index("-h") + 1] == "127.0.0.1"
            )
            return SimpleNamespace(
                returncode=0 if tcp_probe and state == "final-ready" else 1,
                stdout="",
                stderr="",
            )
        if command[:3] == ["docker", "port", database.postgres]:
            return SimpleNamespace(
                returncode=0,
                stdout="127.0.0.1:32768\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fast_parity, "_run", fake_run)
    monkeypatch.setattr(fast_parity.time, "sleep", lambda _seconds: None)

    database.start()

    assert observed_states == [
        "bootstrap-ready",
        "bootstrap-shutdown",
        "final-ready",
    ]
    assert all("-h" in command for command in readiness_commands)
    assert all(
        command[command.index("-d") + 1] == database.database
        for command in readiness_commands
    )
    assert database.target_dsn.endswith(":32768/" + database.database)
    assert ["docker", "volume", "create", database.postgres_volume] in docker_commands
    postgres_run = next(
        command for command in docker_commands if command[:3] == ["docker", "run", "-d"]
    )
    assert (
        "type=volume,source="
        f"{database.postgres_volume},target=/var/lib/postgresql/data"
    ) in postgres_run


def test_disposable_clone_proves_restored_deterministic_uuid(monkeypatch):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )
    observed: list[str] = []

    def fake_psql(sql: str, *, timeout: int = 120) -> str:
        assert timeout == 120
        observed.append(sql)
        return json.dumps({"deterministic_uuid_repeatable": True})

    monkeypatch.setattr(database, "_psql", fake_psql)

    assert database.verify_snapshot_restore() == {"deterministic_uuid_repeatable": True}
    assert "research_lab_deterministic_uuid" in observed[0]


def test_database_lane_retains_primary_failure_and_cleanup_evidence(
    monkeypatch,
    tmp_path: Path,
):
    cleanup = {
        "containers_removed": ["postgres", "postgrest"],
        "network_removed": "network",
        "volume_removed": "volume",
    }

    class Database:
        target_dsn = "postgresql://postgres:x@127.0.0.1/leadpoet_parity_test"

        @staticmethod
        def start():
            return None

        @staticmethod
        def prepare_snapshot_restore():
            return {"verified": True}

        @staticmethod
        def verify_snapshot_restore():
            return {"deterministic_uuid_repeatable": True}

        @staticmethod
        def cleanup():
            return cleanup

    monkeypatch.setattr(fast_parity, "_DockerDatabase", lambda **_kwargs: Database())
    monkeypatch.setattr(
        fast_parity,
        "_load_json",
        lambda *_args, **_kwargs: {"candidate_sha": SHA},
    )
    monkeypatch.setattr(fast_parity, "validate_contract", lambda value: value)

    def fail_restore(**_kwargs):
        raise ProductionParityError("isolated restore failed")

    monkeypatch.setattr(fast_parity, "restore_snapshot", fail_restore)
    outcome = fast_parity._run_database_lane(
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=tmp_path / "snapshot.dump",
        production_host="db.production.example",
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
        region="us-east-1",
        production_gateway_secret_id="gateway-secret",
    )

    assert outcome["result"] is None
    assert isinstance(outcome["primary_error"], ProductionParityError)
    assert outcome["cleanup"] == cleanup
    assert outcome["cleanup_error"] is None


def test_fast_ledger_records_cleanup_independently_of_database_failure(
    monkeypatch,
    tmp_path: Path,
):
    oracle: dict[str, object] = {"schema_version": "test-oracle"}
    contract = {
        "candidate_sha": SHA,
        "base_sha": "c" * 40,
        "risk": "high",
        "source_commitments": [],
        "historical_oracle_hash": sha256_json(oracle),
        "behavior_contract_hash": HASH,
        "contract_hash": HASH,
    }
    manifest = {
        "capture_mode": "schema-only",
        "manifest_hash": HASH,
        "source_sha": "c" * 40,
        "capture_sha": "d" * 40,
    }
    cleanup = {
        "containers_removed": ["postgres", "postgrest"],
        "network_removed": "network",
        "volume_removed": "volume",
    }
    events: list[str] = []
    clock = [100.0]

    monkeypatch.setattr(fast_parity, "_load_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        fast_parity,
        "verify_contract_checkout",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        fast_parity,
        "validate_snapshot_manifest",
        lambda _value: manifest,
    )
    monkeypatch.setattr(
        fast_parity,
        "validate_historical_oracle",
        lambda _value: oracle,
    )
    monkeypatch.setattr(
        fast_parity, "required_oracle_stage_ids", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        fast_parity,
        "verify_snapshot",
        lambda **_kwargs: {"snapshot": "verified", "migration_delta": []},
    )
    monkeypatch.setattr(
        fast_parity,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(fast_parity.time, "monotonic", lambda: clock[0])

    def failed_database_lane(**_kwargs):
        clock[0] += 5.0
        events.append("database-cleanup-complete")
        return {
            "result": None,
            "primary_error": ProductionParityError("isolated restore failed"),
            "cleanup": cleanup,
            "cleanup_error": None,
        }

    monkeypatch.setattr(fast_parity, "_run_database_lane", failed_database_lane)

    def fail_rehearsal(**_kwargs):
        clock[0] += 11.0
        events.append("rehearsal")
        raise ProductionParityError("rehearsal failed")

    monkeypatch.setattr(fast_parity, "_run_rehearsal", fail_rehearsal)
    result = fast_parity.run_fast_lane(
        contract_path=tmp_path / "contract.json",
        manifest_path=tmp_path / "manifest.json",
        archive_path=tmp_path / "snapshot.dump",
        ledger_path=tmp_path / "ledger.json",
        production_host="db.production.example",
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
        region="us-east-1",
        production_gateway_secret_id="gateway-secret",
    )
    stages = {item["stage_id"]: item for item in result["stages"]}

    assert stages["snapshot-restore-and-migrations"]["status"] == "failed"
    assert (
        "isolated restore failed" in stages["snapshot-restore-and-migrations"]["reason"]
    )
    assert stages["cleanup"]["status"] == "passed"
    assert stages["cleanup"]["evidence"] == cleanup
    assert stages["snapshot-restore-and-migrations"]["duration_seconds"] == 5.0
    assert stages["exact-n-minus-one-launchers"]["duration_seconds"] == 11.0
    assert events == ["database-cleanup-complete", "rehearsal"]


def test_full_deadline_accepts_twenty_hours_and_rejects_larger_budget():
    maximum = full_host.MAX_FULL_TIMEOUT_SECONDS
    assert maximum == 20 * 60 * 60
    assert full_host._full_deadline(started=10.0, timeout_seconds=maximum) == (
        10.0 + maximum
    )
    for invalid in (True, 0, maximum + 1):
        with pytest.raises(FullParityError, match="timeout is invalid"):
            full_host._full_deadline(started=10.0, timeout_seconds=invalid)


def test_full_clone_prefix_adapter_forwards_only_strict_supabase_paths():
    observed: list[dict[str, object]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            return None

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            observed.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "body": body,
                    "authorization": self.headers.get("Authorization"),
                    "request_hop": self.headers.get("X-Request-Hop"),
                }
            )
            response_body = b'{"ok":true}'
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-End-To-End", "kept")
            self.send_header("Connection", "X-Response-Hop")
            self.send_header("X-Response-Hop", "drop")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def do_GET(self):
            observed.append({"method": self.command, "path": self.path})
            self.send_response(302)
            self.send_header("Location", "https://foreign.example.invalid/")
            self.send_header("Content-Length", "0")
            self.end_headers()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    adapter = full_host._ClonePostgrestPrefixAdapter(
        upstream_origin=f"http://127.0.0.1:{upstream.server_address[1]}",
        public_origin=ORIGIN,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    try:
        adapter_evidence = adapter.start()
        port = int(adapter_evidence["listen_port"])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/rest/v1/research_lab_jobs?select=id",
            body=b'{"status":"queued"}',
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "Origin": ORIGIN,
                "Connection": "X-Request-Hop",
                "X-Request-Hop": "drop",
            },
        )
        response = connection.getresponse()
        assert response.status == 201
        assert response.read() == b'{"ok":true}'
        assert response.getheader("X-End-To-End") == "kept"
        assert response.getheader("X-Response-Hop") is None
        connection.close()
        assert observed == [
            {
                "method": "POST",
                "path": "/research_lab_jobs?select=id",
                "body": b'{"status":"queued"}',
                "authorization": "Bearer test-token",
                "request_hop": None,
            }
        ]

        for path, headers in (
            ("/rpc/foreign", {}),
            ("/rest/v1/table", {"Origin": "https://foreign.example"}),
            (
                "/rest/v1/table",
                {"Proxy-Authorization": "Basic forbidden"},
            ),
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=5
            )
            connection.request("GET", path, headers=headers)
            rejected = connection.getresponse()
            assert rejected.status == 404
            rejected.read()
            connection.close()
        assert len(observed) == 1

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.putrequest(
            "GET", "http://foreign.example/rest/v1/table", skip_host=True
        )
        connection.putheader("Host", "foreign.example")
        connection.endheaders()
        rejected = connection.getresponse()
        assert rejected.status == 404
        rejected.read()
        connection.close()
        assert len(observed) == 1

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/rest/v1/redirect")
        rejected_redirect = connection.getresponse()
        assert rejected_redirect.status == 502
        rejected_redirect.read()
        connection.close()
        assert observed[-1] == {"method": "GET", "path": "/redirect"}
    finally:
        assert adapter.cleanup() in {"removed", "already_absent"}
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
    assert adapter.cleanup() == "already_absent"
    assert not upstream_thread.is_alive()


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:3000",
        "http://127.0.0.1:99999",
        "http://localhost:54321",
        "http://user@127.0.0.1:54321",
        "http://127.0.0.1:54321/foreign",
    ],
)
def test_full_clone_prefix_adapter_rejects_foreign_upstreams(origin):
    with pytest.raises(FullParityError, match="adapter identity is invalid"):
        full_host._ClonePostgrestPrefixAdapter(
            upstream_origin=origin,
            public_origin=ORIGIN,
            listen_host="127.0.0.1",
            listen_port=0,
        )


def test_full_clone_https_readiness_probes_exact_rest_prefix(monkeypatch):
    observed = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def open(self, request, *, timeout):
            observed["url"] = request.full_url
            observed["timeout"] = timeout
            return Response()

    monkeypatch.setattr(full_host, "build_opener", lambda *_args: Opener())
    full_host._wait_https_origin(ORIGIN, timeout_seconds=1)
    assert observed == {"url": ORIGIN + "/rest/v1/", "timeout": 10}



def test_full_snapshot_disk_headroom_fits_512_gib_and_fails_closed(
    monkeypatch,
    tmp_path: Path,
):
    source_bytes = 174_400_000_000
    capture_required = source_bytes * 2 + FULL_SNAPSHOT_DISK_RESERVE_BYTES
    assert capture_required < 512 * 1024**3

    monkeypatch.setattr(
        "scripts.production_parity_snapshot.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=capture_required),
    )
    assert _require_full_snapshot_disk_headroom(
        tmp_path,
        total_relation_bytes=source_bytes,
        simultaneous_copies=2,
    ) == {
        "required_free_bytes": capture_required,
        "available_free_bytes": capture_required,
    }

    monkeypatch.setattr(
        "scripts.production_parity_snapshot.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=capture_required - 1),
    )
    with pytest.raises(ProductionParityError, match="headroom is insufficient"):
        _require_full_snapshot_disk_headroom(
            tmp_path,
            total_relation_bytes=source_bytes,
            simultaneous_copies=2,
        )


def _full_host_main_args(output: Path) -> list[str]:
    return [
        "--region",
        "us-east-1",
        "--run-id",
        "pp-test-1",
        "--base-sha",
        "a" * 40,
        "--candidate-sha",
        "b" * 40,
        "--production-gateway-secret-id",
        "gateway-secret",
        "--readonly-dsn-secret-id",
        "readonly-secret",
        "--miner-intake-secret-id",
        "miner-secret",
        "--supabase-origin",
        ORIGIN,
        "--artifact-bucket",
        "parity-artifacts",
        "--postgres-image",
        "postgres@sha256:" + "c" * 64,
        "--postgrest-image",
        "postgrest@sha256:" + "d" * 64,
        "--output",
        str(output),
    ]


def test_full_main_retains_bounded_early_failure_without_overwrite(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "evidence" / "full-evidence.json"
    secret = "must-not-escape-early-failure"
    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(FullParityError(secret)),
    )

    assert full_host.main(_full_host_main_args(output)) == 1

    encoded = output.read_text(encoding="utf-8")
    evidence = json.loads(encoded)
    assert evidence == {
        "schema_version": full_host.SCHEMA_VERSION,
        "run_id": "pp-test-1",
        "candidate_sha": "b" * 40,
        "base_sha": "a" * 40,
        "started_at": evidence["started_at"],
        "status": "failed",
        "failure_stage": "initialization",
        "error_type": "FullParityError",
        "cleanup": {},
        "duration_seconds": evidence["duration_seconds"],
        "finished_at": evidence["finished_at"],
    }
    assert 0 <= evidence["duration_seconds"] < 10
    assert len(encoded.encode("utf-8")) < 1_024
    assert secret not in encoded
    assert capsys.readouterr().err.strip() == "ERROR: full parity failed closed"

    retained = encoded
    assert full_host.main(_full_host_main_args(output)) == 1
    assert output.read_text(encoding="utf-8") == retained


def test_full_main_concurrent_authoritative_evidence_wins_atomically(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "full-evidence.json"
    authoritative = '{"status":"passed"}\n'
    real_open = full_host.os.open

    def race_open(path, flags, mode=0o777):
        if Path(path) == output:
            output.write_text(authoritative, encoding="utf-8")
        return real_open(path, flags, mode)

    monkeypatch.setattr(full_host.os, "open", race_open)
    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(FullParityError("redacted")),
    )

    assert full_host.main(_full_host_main_args(output)) == 1
    assert output.read_text(encoding="utf-8") == authoritative


def test_full_main_never_follows_preexisting_evidence_symlink(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "full-evidence.json"
    authoritative = tmp_path / "authoritative.json"
    retained = '{"status":"passed"}\n'
    authoritative.write_text(retained, encoding="utf-8")
    output.symlink_to(authoritative)
    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(FullParityError("redacted")),
    )

    assert full_host.main(_full_host_main_args(output)) == 1
    assert output.is_symlink()
    assert authoritative.read_text(encoding="utf-8") == retained


def test_full_main_completes_short_writes_and_closes_descriptor(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "full-evidence.json"
    real_open = full_host.os.open
    real_write = full_host.os.write
    descriptors: list[int] = []
    writes = 0

    def capture_open(path, flags, mode=0o777):
        descriptor = real_open(path, flags, mode)
        descriptors.append(descriptor)
        return descriptor

    def short_write(descriptor, value):
        nonlocal writes
        writes += 1
        return real_write(descriptor, value[:7])

    monkeypatch.setattr(full_host.os, "open", capture_open)
    monkeypatch.setattr(full_host.os, "write", short_write)
    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(FullParityError("redacted")),
    )

    assert full_host.main(_full_host_main_args(output)) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert writes > 1
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_full_main_preserves_keyboard_interrupt(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "full-evidence.json"
    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        full_host.main(_full_host_main_args(output))
    assert not output.exists()


def test_full_main_redacts_unexpected_early_exception(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "full-evidence.json"
    secret = "must-not-escape-unexpected-failure"

    class SecretFailure(Exception):
        pass

    monkeypatch.setattr(
        full_host,
        "run_full",
        lambda **_kwargs: (_ for _ in ()).throw(SecretFailure(secret)),
    )

    assert full_host.main(_full_host_main_args(output)) == 1

    encoded = output.read_text(encoding="utf-8")
    evidence = json.loads(encoded)
    assert evidence["failure_stage"] == "initialization"
    assert evidence["error_type"] == "UnexpectedError"
    assert secret not in encoded
    assert secret not in capsys.readouterr().err


@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_full_runner_forwards_remaining_clone_budget_and_cleans_runtime(
    monkeypatch,
    tmp_path: Path,
    cleanup_raises: bool,
):
    marker = tmp_path / "early-boot-isolated"
    marker.write_text("isolated\n", encoding="utf-8")
    work_root = tmp_path / "encrypted-root-volume"
    output = tmp_path / "evidence" / "full.json"
    observed: dict[str, object] = {}
    monotonic_values = iter((100.0, 110.2, 250.1, 251.0))

    class Database:
        target_dsn = "postgresql://postgres:x@127.0.0.1/leadpoet_parity_test"

        @staticmethod
        def start():
            return None

        @staticmethod
        def prepare_snapshot_restore():
            return {"verified": True}

        @staticmethod
        def cleanup():
            if cleanup_raises:
                raise OSError("injected cleanup failure")
            return {"status": "removed"}

    def fake_capture_snapshot(**kwargs):
        observed["capture"] = kwargs["timeout_seconds"]
        observed["capture_image"] = kwargs["postgres_image"]
        return {
            "capture_mode": "full",
            "database": {"target_rebenchmark_date": "2026-08-19"},
        }

    def fake_restore_snapshot(**kwargs):
        observed["restore"] = kwargs["timeout_seconds"]
        observed["restore_image"] = kwargs["postgres_image"]
        raise FullParityError("stop after restore")

    def database_factory(**kwargs):
        observed["postgres_publish"] = kwargs["postgres_publish"]
        observed["postgrest_publish"] = kwargs["postgrest_publish"]
        return Database()

    monkeypatch.setattr(full_host, "EARLY_BOOT_MARKER", marker)
    monkeypatch.setattr(full_host, "FULL_WORK_ROOT", work_root)
    monkeypatch.setattr(
        full_host,
        "_materialize_run_owned_runtime_identity",
        lambda _path: {
            "gateway_private_key_path": "/run/parity/gateway-private.pem",
            "gateway_public_key": PARITY_GATEWAY_PUBLIC_KEY,
            "gateway_public_key_hash": "sha256:" + "1" * 64,
            "arweave_keyfile_path": "/run/parity/arweave.json",
            "arweave_address_hash": "sha256:" + "2" * 64,
        },
    )
    monkeypatch.setattr(full_host, "_checkout_identity", lambda _sha: None)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(full_host.boto3, "client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(full_host, "_DockerDatabase", database_factory)
    monkeypatch.setattr(full_host, "capture", lambda **_kwargs: None)
    monkeypatch.setattr(
        full_host,
        "build_contract",
        lambda **_kwargs: {"candidate_sha": "b" * 40, "contract_hash": HASH},
    )
    monkeypatch.setattr(
        full_host,
        "_secret_value",
        lambda *_args, **_kwargs: "postgresql://reader:x@db.example/postgres",
    )
    monkeypatch.setattr(full_host, "capture_snapshot", fake_capture_snapshot)
    monkeypatch.setattr(full_host, "restore_snapshot", fake_restore_snapshot)

    with pytest.raises(FullParityError, match="stop after restore"):
        full_host.run_full(
            region="us-east-1",
            run_id="pp-test-1",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            production_gateway_secret_id="gateway-secret",
            readonly_dsn_secret_id="readonly-secret",
            miner_intake_secret_id="miner-secret",
            supabase_origin=ORIGIN,
            artifact_bucket="parity-artifacts",
            postgres_image="postgres@sha256:" + "c" * 64,
            postgrest_image="postgrest@sha256:" + "d" * 64,
            output=output,
            timeout_seconds=1_000,
        )

    assert observed == {
        "postgres_publish": "127.0.0.1::5432",
        "postgrest_publish": "127.0.0.1::3000",
        "capture": 990,
        "capture_image": "postgres@sha256:" + "c" * 64,
        "restore": 850,
        "restore_image": "postgres@sha256:" + "c" * 64,
    }
    assert not (work_root / "pp-test-1" / "runtime").exists()
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["failure_stage"] == "snapshot-restore"
    assert evidence["error_type"] == "FullParityError"
    cleanup = evidence["cleanup"]
    assert cleanup["work"] == "removed"
    if cleanup_raises:
        assert cleanup["database_error"] == "OSError"
    else:
        assert cleanup["database"] == {"status": "removed"}


def test_full_runner_retains_exact_bounded_initialization_stage(
    monkeypatch,
    tmp_path: Path,
):
    marker = tmp_path / "early-boot-isolated"
    marker.write_text("isolated\n", encoding="utf-8")
    work_root = tmp_path / "encrypted-root-volume"
    output = tmp_path / "evidence" / "full.json"
    secret = "must-not-escape-initialization-failure"

    monkeypatch.setattr(full_host, "EARLY_BOOT_MARKER", marker)
    monkeypatch.setattr(full_host, "FULL_WORK_ROOT", work_root)
    monkeypatch.setattr(full_host, "_checkout_identity", lambda _sha: None)
    monkeypatch.setattr(
        full_host,
        "_materialize_run_owned_runtime_identity",
        lambda _path: (_ for _ in ()).throw(FullParityError(secret)),
    )

    with pytest.raises(FullParityError, match=secret):
        full_host.run_full(
            region="us-east-1",
            run_id="pp-test-1",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            production_gateway_secret_id="gateway-secret",
            readonly_dsn_secret_id="readonly-secret",
            miner_intake_secret_id="miner-secret",
            supabase_origin=ORIGIN,
            artifact_bucket="parity-artifacts",
            postgres_image="postgres@sha256:" + "c" * 64,
            postgrest_image="postgrest@sha256:" + "d" * 64,
            output=output,
            timeout_seconds=1_000,
        )

    encoded = output.read_text(encoding="utf-8")
    evidence = json.loads(encoded)
    assert evidence["status"] == "failed"
    assert evidence["failure_stage"] == "runtime-identity"
    assert evidence["error_type"] == "FullParityError"
    assert evidence["cleanup"] == {"work": "removed"}
    assert secret not in encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("archive", "persisted"), True),
        (("database", "target_rebenchmark_date"), "2026-08-15"),
        (("database", "source_role", "superuser"), True),
        (("database", "source_role", "replication"), True),
        (("database", "source_role", "table_write_capable"), True),
    ],
)
def test_snapshot_rejects_persistent_or_mutable_production_input(field, value):
    document = _snapshot()
    target = document
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    body = {key: item for key, item in document.items() if key != "manifest_hash"}
    document["manifest_hash"] = sha256_json(body)
    with pytest.raises(ProductionParityError):
        validate_snapshot_manifest(
            document,
            now=datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("manifest_field", "replacement", "error"),
    (
        ("source_sha", "d" * 40, "source commit"),
        ("capture_sha", "d" * 40, "capture commit"),
        ("capture_contract_hash", "sha256:" + "d" * 64, "capture contract"),
    ),
)
def test_snapshot_verification_requires_exact_contract_bindings(
    monkeypatch,
    tmp_path: Path,
    manifest_field: str,
    replacement: str,
    error: str,
):
    contract = {
        "base_sha": "c" * 40,
        "candidate_sha": SHA,
        "contract_hash": HASH,
    }
    manifest = {
        "source_sha": contract["base_sha"],
        "capture_sha": contract["candidate_sha"],
        "capture_contract_hash": contract["contract_hash"],
    }
    manifest[manifest_field] = replacement

    def fake_load(_path, *, description):
        return contract if description == "parity contract" else manifest

    monkeypatch.setattr(parity_snapshot, "_load_json", fake_load)
    monkeypatch.setattr(parity_snapshot, "validate_contract", lambda value: value)
    monkeypatch.setattr(
        parity_snapshot,
        "validate_snapshot_manifest",
        lambda value: value,
    )

    with pytest.raises(ProductionParityError, match=error):
        parity_snapshot.verify_snapshot(
            contract_path=tmp_path / "contract.json",
            manifest_path=tmp_path / "manifest.json",
            archive_path=tmp_path / "snapshot.dump",
        )


def test_clone_target_can_never_resolve_to_production():
    with pytest.raises(ProductionParityError, match="isolated parity database"):
        safe_database_target(
            "postgresql://reader:x@db.production.example/leadpoet",
            production_host="db.production.example",
        )
    safe_database_target(
        "postgresql://postgres:x@127.0.0.1:5432/leadpoet_parity_run",
        production_host="db.production.example",
    )


def test_fast_live_boundary_rejects_every_non_get_or_foreign_request():
    provider = _ProductionReadOnlySupabaseProvider(
        origin="https://qplwoislplkcegvdmbim.supabase.co",
        service_role_key="secret-read-key",
    )
    base = {
        "provider_id": "supabase",
        "method": "GET",
        "url": (
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
            "research_lab_finalized_allocation_epochs_v2?select=epoch_id"
        ),
        "headers": {"range": "0-1"},
        "body_b64": "",
        "timeout_ms": 1000,
    }
    with pytest.raises(ProductionParityError, match="non-GET or foreign"):
        provider({**base, "method": "POST"})
    with pytest.raises(ProductionParityError, match="non-GET or foreign"):
        provider({**base, "url": "https://example.com/rest/v1/table"})
    with pytest.raises(ProductionParityError, match="non-GET or foreign"):
        provider({**base, "body_b64": "eA=="})


def test_fast_live_boundary_disables_environment_proxy_routing(monkeypatch):
    observed_handlers: list[object] = []

    def fake_build_opener(*handlers):
        observed_handlers.extend(handlers)
        return object()

    monkeypatch.setattr(fast_parity, "build_opener", fake_build_opener)
    _ProductionReadOnlySupabaseProvider(
        origin="https://qplwoislplkcegvdmbim.supabase.co",
        service_role_key="secret-read-key",
    )

    assert len(observed_handlers) == 2
    assert isinstance(observed_handlers[0], fast_parity.ProxyHandler)
    assert observed_handlers[0].proxies == {}
    assert observed_handlers[1] is fast_parity._NoRedirect


def test_database_cleanup_fails_closed_when_docker_absence_cannot_be_proven(
    monkeypatch,
):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )

    def unavailable(command, *, timeout, **_kwargs):
        assert timeout in {10, 30}
        if "ls" in command:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Cannot connect to the Docker daemon",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="remove failed")

    monkeypatch.setattr(fast_parity, "_run", unavailable)
    with pytest.raises(
        ProductionParityError,
        match="container cleanup verification failed",
    ):
        database.cleanup()


def test_database_cleanup_accepts_independently_proven_absence(monkeypatch):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )

    def absent(command, *, timeout, **_kwargs):
        assert timeout in {10, 30}
        if "ls" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="already absent")

    monkeypatch.setattr(fast_parity, "_run", absent)
    assert database.cleanup() == {
        "containers_removed": [database.postgres, database.postgrest],
        "network_removed": database.network,
        "volume_removed": database.postgres_volume,
    }


def test_database_cleanup_rejects_a_surviving_clone_volume(monkeypatch):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )

    def volume_survives(command, *, timeout, **_kwargs):
        assert timeout in {10, 30}
        if command[:3] == ["docker", "volume", "ls"]:
            return SimpleNamespace(
                returncode=0,
                stdout="volume-id\n",
                stderr="",
            )
        if "ls" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fast_parity, "_run", volume_survives)
    with pytest.raises(ProductionParityError, match="volume"):
        database.cleanup()


def test_standalone_postgrest_schema_opener_rewrites_exact_supabase_prefix(
    monkeypatch,
):
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        @staticmethod
        def open(request, *, timeout):
            observed["url"] = request.full_url
            observed["headers"] = dict(request.header_items())
            observed["method"] = request.get_method()
            observed["data"] = request.data
            observed["timeout"] = timeout
            return Response()

    monkeypatch.setattr(fast_parity, "build_opener", lambda *_handlers: Opener())
    opener = fast_parity._StandalonePostgrestSchemaOpener("http://127.0.0.1:32768")
    request = fast_parity.Request(
        "http://127.0.0.1:32768/rest/v1/rpc/schema_probe?value=one",
        data=b'{"probe":true}',
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer clone-token",
            "apikey": "clone-token",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with opener(request, timeout=20) as response:
        assert isinstance(response, Response)

    assert observed == {
        "url": "http://127.0.0.1:32768/rpc/schema_probe?value=one",
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer clone-token",
            "Apikey": "clone-token",
            "Content-type": "application/json",
        },
        "method": "POST",
        "data": b'{"probe":true}',
        "timeout": 20,
    }


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:32768/rest/v1/table",
        "http://localhost:32768/rest/v1/table",
        "http://127.0.0.1:32769/rest/v1/table",
        "http://user@127.0.0.1:32768/rest/v1/table",
        "http://127.0.0.1:32768/other/table",
        "http://127.0.0.1:32768/rest/v10/table",
        "http://127.0.0.1:32768/rest/v1/table#fragment",
    ),
)
def test_standalone_postgrest_schema_opener_rejects_nonclone_requests(url):
    opener = fast_parity._StandalonePostgrestSchemaOpener("http://127.0.0.1:32768")
    with pytest.raises(ProductionParityError, match="escaped the clone"):
        opener(fast_parity.Request(url), timeout=20)


@pytest.mark.parametrize(
    "origin",
    (
        "https://127.0.0.1:32768",
        "http://localhost:32768",
        "http://127.0.0.1",
        "http://user@127.0.0.1:32768",
        "http://127.0.0.1:32768/rest/v1",
    ),
)
def test_standalone_postgrest_schema_opener_rejects_nonloopback_origin(origin):
    with pytest.raises(ProductionParityError, match="not exact loopback"):
        fast_parity._StandalonePostgrestSchemaOpener(origin)


def _chain_realized_activation_row(**overrides):
    row = {
        "netuid": 71,
        "schema_version": fast_parity.CHAIN_REALIZED_ACTIVATION_SCHEMA_VERSION,
        "first_epoch_id": 24196,
        "source_bundle_hash": "sha256:" + "a" * 64,
        "source_bundle_epoch_id": 24196,
        "source_finalized_block": 8715224,
    }
    row.update(overrides)
    return row


def _activation_provider_result(payload: bytes, *, status: int = 200):
    return {
        "terminal_status": "authenticated_response",
        "http_status": status,
        "body_b64": base64.b64encode(payload).decode("ascii"),
        "transport_attempt": {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "response_hash": "sha256:"
            + fast_parity.hashlib.sha256(payload).hexdigest(),
            "request_artifact_hash": "sha256:" + "c" * 64,
            "response_artifact_hash": "sha256:"
            + fast_parity.hashlib.sha256(payload).hexdigest(),
            "adapter": "strict-read-only-production-postgrest",
        },
    }


def test_fast_reads_one_exact_live_activation_without_exposing_payload():
    observed = {}
    expected = _chain_realized_activation_row()

    def provider(request):
        observed.update(request)
        return _activation_provider_result(json.dumps([expected]).encode())

    row, evidence = fast_parity._read_production_chain_realized_activation(
        provider=provider,
        netuid=71,
    )

    query = fast_parity.urlencode(
        {
            "select": ",".join(fast_parity.CHAIN_REALIZED_ACTIVATION_COLUMNS),
            "netuid": "eq.71",
            "limit": "2",
        }
    )
    assert observed == {
        "provider_id": "supabase",
        "method": "GET",
        "url": (
            f"{fast_parity.SUPABASE_WEIGHT_SOURCE_ORIGIN}/rest/v1/"
            f"{fast_parity.CHAIN_REALIZED_ACTIVATION_TABLE}?{query}"
        ),
        "headers": {"Accept": "application/json"},
        "body_b64": "",
        "timeout_ms": 20_000,
        "logical_operation_id": ("production-parity-chain-realized-activation-v1"),
    }
    assert row == expected
    assert evidence["netuid"] == 71
    assert evidence["first_epoch_id"] == expected["first_epoch_id"]
    assert evidence["source_bundle_epoch_id"] == expected["source_bundle_epoch_id"]
    assert evidence["source_finalized_block"] == expected["source_finalized_block"]
    assert all(
        type(value) is int or re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        for value in evidence.values()
    )
    assert fast_parity.CHAIN_REALIZED_ACTIVATION_SCHEMA_VERSION not in json.dumps(
        evidence
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"[]", "missing or ambiguous"),
        (
            json.dumps(
                [
                    _chain_realized_activation_row(),
                    _chain_realized_activation_row(),
                ]
            ).encode(),
            "missing or ambiguous",
        ),
        (
            json.dumps(
                [
                    _chain_realized_activation_row(
                        schema_version="unexpected.activation.v1"
                    )
                ]
            ).encode(),
            "row is invalid",
        ),
        (
            json.dumps(
                [_chain_realized_activation_row(first_epoch_id="24196")]
            ).encode(),
            "row is invalid",
        ),
        (
            json.dumps(
                [_chain_realized_activation_row(source_bundle_epoch_id=24197)]
            ).encode(),
            "row is invalid",
        ),
        (
            json.dumps(
                [_chain_realized_activation_row(source_bundle_hash="not-a-hash")]
            ).encode(),
            "row is invalid",
        ),
        (
            json.dumps(
                [_chain_realized_activation_row(unexpected_field="value")]
            ).encode(),
            "row is invalid",
        ),
    ),
)
def test_fast_rejects_missing_ambiguous_or_malformed_live_activation(
    payload,
    message,
):
    with pytest.raises(ProductionParityError, match=message):
        fast_parity._read_production_chain_realized_activation(
            provider=lambda _request: _activation_provider_result(payload),
            netuid=71,
        )


def test_fast_rejects_duplicate_live_activation_fields():
    payload = (
        b'[{"netuid":71,"netuid":71,"schema_version":"'
        + fast_parity.CHAIN_REALIZED_ACTIVATION_SCHEMA_VERSION.encode()
        + b'","first_epoch_id":24196,"source_bundle_hash":"sha256:'
        + b"a" * 64
        + b'","source_bundle_epoch_id":24196,"source_finalized_block":8715224}]'
    )
    with pytest.raises(ProductionParityError, match="response is invalid"):
        fast_parity._read_production_chain_realized_activation(
            provider=lambda _request: _activation_provider_result(payload),
            netuid=71,
        )


def test_weight_scale_evidence_excludes_prior_activation_read(monkeypatch):
    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image="postgres@sha256:" + "c" * 64,
        postgrest_image="postgrest@sha256:" + "d" * 64,
    )

    class Provider:
        adapter_name = "strict-read-only-production-postgrest"

        def __init__(self):
            self.pages = [
                {
                    "response_bytes": 400,
                    "response_hash": "sha256:" + "1" * 64,
                }
            ]

    provider = Provider()

    class Reader:
        def __init__(self, *, execute_provider, **_kwargs):
            self.provider = execute_provider

        def read(self, *, record_transport, record_artifact, **_kwargs):
            self.provider.pages.append(
                {
                    "response_bytes": 200,
                    "response_hash": "sha256:" + "2" * 64,
                }
            )
            record_transport({"terminal_status": "authenticated_response"})
            record_artifact("sha256:" + "3" * 64)
            return [{"epoch_id": 24570}]

    monkeypatch.setattr(fast_parity, "SupabaseSourceReaderV2", Reader)
    evidence = database.weight_input_scale_evidence(
        service_role_key="clone-service-role",
        scope={
            "netuid": 71,
            "start_epoch": 24570,
            "end_epoch": 24570,
            "expected_rows": 1,
        },
        provider=provider,
    )

    assert evidence["page_count"] == 1
    assert evidence["total_response_bytes"] == 200
    assert evidence["response_hashes"] == ["sha256:" + "2" * 64]


def test_disposable_postgrest_progresses_through_activation_openapi_and_rpcs():
    postgres_image = os.environ.get("LEADPOET_PARITY_POSTGRES_IMAGE") or os.environ.get(
        "POSTGRES_IMAGE"
    )
    postgrest_image = os.environ.get(
        "LEADPOET_PARITY_POSTGREST_IMAGE"
    ) or os.environ.get("POSTGREST_IMAGE")
    if shutil.which("docker") is None or not postgres_image or not postgrest_image:
        pytest.skip("pinned disposable parity images are unavailable")

    database = fast_parity._DockerDatabase(
        candidate_sha=SHA,
        postgres_image=postgres_image,
        postgrest_image=postgrest_image,
    )
    try:
        database.start()
        database.prepare_snapshot_restore()
        tables: dict[str, set[str]] = {}
        for _migration, table, columns in schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA:
            tables.setdefault(table, set()).update(columns)
        statements = []
        for table, columns in sorted(tables.items()):
            assert re.fullmatch(r"[a-z_][a-z0-9_]*", table)
            assert all(re.fullmatch(r"[a-z_][a-z0-9_]*", column) for column in columns)
            if table == fast_parity.CHAIN_REALIZED_ACTIVATION_TABLE:
                statements.append(
                    """
CREATE TABLE public.research_lab_chain_realized_settlement_activation_v1 (
  netuid integer PRIMARY KEY CHECK (netuid > 0),
  schema_version text NOT NULL,
  first_epoch_id integer NOT NULL CHECK (first_epoch_id >= 0),
  source_bundle_hash text NOT NULL,
  source_bundle_epoch_id integer NOT NULL,
  source_finalized_block bigint NOT NULL CHECK (source_finalized_block >= 0),
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.research_lab_chain_realized_settlement_activation_v1
  ENABLE ROW LEVEL SECURITY;
"""
                )
                continue
            definitions = ", ".join(f'"{column}" text' for column in sorted(columns))
            statements.append(f'CREATE TABLE public."{table}" ({definitions});')

        clauses = []
        for role, purposes in schema_preflight.ROLE_PURPOSES.items():
            encoded_purposes = ", ".join(
                f"'{purpose}'::text" for purpose in sorted(purposes)
            )
            clauses.append(
                f"((role = '{role}'::text) AND "
                f"(purpose = ANY (ARRAY[{encoded_purposes}])))"
            )
        purpose_definition = "CHECK (" + " OR ".join(clauses) + ")"
        compact_contract = {
            "schema_version": (
                "leadpoet.research_lab_compact_weight_settlement_contract.v1"
            ),
            "max_authority_bytes": 8_388_608,
            "size_constraint_valid": True,
            "append_only_trigger_enabled": True,
            "identity_unique_constraint_enabled": True,
            "row_level_security_enabled": True,
            "finalized_stage_supported": True,
        }
        purpose_contract = {
            "schema_version": (
                "leadpoet.research_lab_candidate_hybrid_purpose_contract.v1"
            ),
            "constraint_name": (
                "research_lab_attested_execution_receipts_v2_role_purpose_check"
            ),
            "constraint_valid": True,
            "constraint_definition": purpose_definition,
        }
        source_add_origin_contract = {
            "schema_version": "leadpoet.source_add_provider_origin_contract.v1",
            "identity_version": "v1",
            "identity_scope": "normalized_exact_host",
            "admission_rpc": "research_lab_source_add_admit_v2",
            "recheck_rpc": "research_lab_source_add_requeue_provenance_v2",
            "owner_count": 0,
            "reserved_count": 0,
            "coverage_complete": True,
            "collision_free": True,
            "submission_trigger_enabled": True,
            "catalog_trigger_enabled": True,
            "provision_trigger_enabled": True,
            "terminal_release_trigger_enabled": True,
            "append_only_trigger_enabled": True,
            "row_level_security_enabled": True,
            "service_role_policy_enabled": True,
        }
        source_add_duplicate_privacy_contract = {
            "schema_version": (
                "leadpoet.source_add_duplicate_privacy_contract.v1"
            ),
            "admission_rpc": "research_lab_source_add_admit_v3",
            "admission_signature": (
                "jsonb,text,text,text,text,text,integer,integer,integer,integer"
            ),
            "compatibility_rpc": "research_lab_source_add_admit_v2",
            "compatibility_signature": (
                "jsonb,text,text,text,text,text,integer,integer,integer"
            ),
            "compatibility_cooldown_seconds": 20,
            "cooldown_parameter_min_seconds": 1,
            "cooldown_parameter_max_seconds": 3600,
            "cooldown_clock": "clock_timestamp_after_advisory_locks",
            "cooldown_source": "durable_miner_provenance_work",
            "duplicate_precedes_cooldown": True,
            "lock_order": [
                "provider_origin_or_identity",
                "hotkey",
                "submission_or_work",
            ],
            "function_authority_sha256": (
                schema_preflight.SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256
            ),
            "functions": {
                "admit_v1": True,
                "admit_v2_compatibility": True,
                "admit_v3": True,
                "provider_origin_hash_v1": True,
                "provider_origin_host_v1": True,
            },
            "permissions": {
                "service_role_exists": True,
                "v3_service_role_callable": True,
                "v2_service_role_callable": True,
                "contract_service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        }
        source_add_provenance_leg1_contract = {
            "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v4",
            "required_migration": (
                "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
            ),
            "daily_cap": 50,
            "leg1_alpha_percent": 0.2,
            "leg1_reward_epochs": 20,
            "approval_boundary": "provenance_precheck_passed",
            "backfill_policy": (
                "earliest_exact_attested_provenance_per_provider_origin"
            ),
            "provider_origin_scope": "normalized_exact_host",
            "provider_origin_winner_order": [
                "provenance_created_at",
                "submission_id",
            ],
            "cancelled_intents_are_authority": False,
            "public_trigger_fields": [
                "precheck_status",
                "provenance_artifact_hash",
                "provenance_precheck_passed",
                "provenance_receipt_hash",
                "provenance_result_hash",
                "submission_id",
            ],
            "authority_view": (
                "research_lab_source_add_provenance_leg1_authority_v1"
            ),
            "function_authority_sha256": (
                schema_preflight.SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
            ),
            "trigger_authority_sha256": (
                schema_preflight.SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
            ),
            "view_authority_sha256": (
                schema_preflight.SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256
            ),
            "repair_function_authority_sha256": (
                schema_preflight.SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256
            ),
            "functions": {
                "configure_probe_v3": True,
                "enqueue_leg1_after_provenance_v1": True,
                "enqueue_provision_smoke_v2": True,
                "finalize_leg1_v4": True,
                "finalize_provision_smoke_v3": True,
                "finalize_provision_v3": True,
                "reject_current_builtin_v3": True,
                "reconcile_provenance_leg1_v1": True,
                "reserve_leg1_slot_v4": True,
            },
            "triggers": {
                "automatic_enqueue": True,
                "eligible_v2": True,
                "eligible_v3": True,
                "leg1_initial_event_v3": True,
                "leg1_obligation_v3": True,
                "leg1_slot_v3": True,
                "leg1_work_v3": True,
            },
            "columns": {
                "intent_approval_kind": True,
                "intent_provenance_artifact_hash": True,
                "intent_provenance_receipt_hash": True,
                "slot_approval_kind": True,
            },
            "permissions": {
                "service_role_exists": True,
                "candidate_callable": True,
                "internal_not_callable": True,
                "rollback_v2_callable": True,
            },
        }
        contract_functions = {
            "research_lab_compact_weight_settlement_contract_v1": compact_contract,
            "research_lab_candidate_hybrid_purpose_contract_v1": purpose_contract,
            "research_lab_source_add_provider_origin_contract_v1": (
                source_add_origin_contract
            ),
            "research_lab_source_add_duplicate_privacy_contract_v1": (
                source_add_duplicate_privacy_contract
            ),
            "research_lab_source_add_post_accept_leg1_contract_v4": (
                source_add_provenance_leg1_contract
            ),
        }
        for _migration, function_name in schema_preflight.REQUIRED_SUPABASE_V2_RPCS:
            assert re.fullmatch(r"[a-z_][a-z0-9_]*", function_name)
            payload = json.dumps(
                contract_functions.get(function_name, {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            statements.append(
                f'CREATE FUNCTION public."{function_name}"() RETURNS jsonb '
                "LANGUAGE sql STABLE AS $function$ "
                f"SELECT $json${payload}$json$::jsonb $function$;"
            )
        database._psql("\n".join(statements))

        activation = _chain_realized_activation_row()
        assert (
            database._psql(
                "SELECT COUNT(*)::text FROM public."
                "research_lab_chain_realized_settlement_activation_v1;"
            ).strip()
            == "0"
        )
        supabase_url, service_role_key = database.start_postgrest()
        assert database._psql(
            "SELECT has_schema_privilege('service_role','extensions','USAGE')::text;"
        ).strip() == "t"
        result = schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": supabase_url,
                "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
                "BITTENSOR_NETUID": "71",
            },
            opener=fast_parity._StandalonePostgrestSchemaOpener(supabase_url),
            timeout_seconds=20,
            chain_realized_activation_authority=activation,
        )

        assert (
            database._psql(
                "SELECT COUNT(*)::text FROM public."
                "research_lab_chain_realized_settlement_activation_v1;"
            ).strip()
            == "0"
        )
        assert result["status"] == "ready"
        assert result["data_probe_count"] == 4
        assert (
            result["chain_realized_settlement_activation_http_probe_count"] == 0
        )
        assert result["chain_realized_settlement_activation_source"] == (
            "provided-authority"
        )
        assert result["chain_realized_settlement_activation"] == {
            "netuid": 71,
            "first_epoch_id": activation["first_epoch_id"],
            "source_bundle_hash": activation["source_bundle_hash"],
            "source_finalized_block": activation["source_finalized_block"],
        }
        assert result["compact_weight_settlement_contract"] == compact_contract
        assert result["candidate_hybrid_purpose_contract"]["constraint_valid"] is True
        assert (
            result["source_add_provider_origin_contract"]
            == source_add_origin_contract
        )
        assert result["rpc_probe_count"] == len(
            schema_preflight.REQUIRED_SUPABASE_V2_RPCS
        )
    finally:
        database.cleanup()


def test_rehearsal_failure_diagnostics_are_bounded_and_redacted(
    monkeypatch,
    tmp_path: Path,
):
    candidate_sha = "d" * 40
    durable_root = tmp_path / (
        f"leadpoet-rehearsal-failure-{candidate_sha[:12]}-full-path-test"
    )
    durable_root.mkdir()
    secret = "must-not-escape-diagnostic"
    summary = {
        "candidate_sha": candidate_sha,
        "status": "failed",
        "stages": [
            {
                "stage": "gateway-forward-1",
                "status": "failed",
                "error_type": "CalledProcessError",
                "returncode": 1,
                "duration_seconds": 12.3456,
                "error": secret,
                "command": ["gateway", "--credential", secret],
                "fixture_generation_diagnostic": {
                    "category": "resource",
                    "status": 137,
                },
                "evidence_normalization_diagnostics": [
                    {
                        "phase": "container",
                        "category": "permission",
                        "status": 23,
                    }
                ],
                "workflow_failure_projection": {
                    "available": True,
                    "failed_count": 1,
                    "unexercised_count": 1,
                    "emitted_count": 1,
                    "truncated": True,
                    "stages": [
                        {
                            "status": "failed",
                            "stage_kind": "behavior",
                            "stage_id_sha256": "a" * 64,
                            "error_type": "RuntimeError",
                        }
                    ],
                    "raw": secret,
                },
            },
            {
                "stage": "evidence-join-prepush",
                "status": "unexercised",
                "blocked_by": ["gateway-forward-1"],
            },
        ],
    }
    (durable_root / "failure-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    monkeypatch.setattr(fast_parity.tempfile, "gettempdir", lambda: str(tmp_path))
    result = fast_parity.subprocess.CompletedProcess(
        args=["rehearsal"],
        returncode=1,
        stdout="",
        stderr=(
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0\n"
            + ("x" * 9000)
            + "\n"
            "REHEARSAL_FAILURE_DIAGNOSTICS component=gateway status=137\n"
            "REHEARSAL_HTTP_DIAGNOSTIC endpoint=/attest status=503\n"
            "REHEARSAL_STAGE_FAILED_CONTINUING stage=gateway-forward-1 "
            "error_type=CalledProcessError duration_seconds=12.3456 "
            "error='launcher failed'\n"
            "REHEARSAL CONTRACT ERROR [docker]: unknown operation: ['docker']\n"
            "ERROR: process terminated out of memory\n"
            f"ERROR: bearer token={secret} permission denied\n"
            f"raw={secret}\nREHEARSAL_BATCH_FAILURE_EVIDENCE {durable_root}\n"
        ),
    )

    diagnostics = fast_parity._rehearsal_failure_diagnostics(
        result, candidate_sha=candidate_sha
    )
    encoded = json.dumps(diagnostics, sort_keys=True)
    assert diagnostics["failure_summary_available"] is True
    assert diagnostics["failed_stage_count"] == 1
    assert diagnostics["unexercised_stage_count"] == 1
    assert diagnostics["output_markers"] == [
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {"marker": "component_failure", "component": "gateway", "status": 137},
        {"marker": "http", "endpoint": "/attest", "status": "503"},
        {
            "marker": "stage_failure",
            "stage": "gateway-forward-1",
            "error_type": "CalledProcessError",
            "duration_seconds": 12.346,
        },
        {"marker": "contract_error", "kind": "docker"},
        {"marker": "error", "category": "resource_oom"},
    ]
    assert diagnostics["stages"] == [
        {
            "stage": "gateway-forward-1",
            "status": "failed",
            "error_type": "CalledProcessError",
            "returncode": 1,
            "duration_seconds": 12.346,
            "fixture_generation_diagnostic": {
                "category": "resource",
                "status": 137,
            },
            "evidence_normalization_diagnostics": [
                {
                    "phase": "container",
                    "category": "permission",
                    "status": 23,
                }
            ],
            "workflow_failure_projection": {
                "available": True,
                "failed_count": 1,
                "unexercised_count": 1,
                "emitted_count": 1,
                "truncated": True,
                "stages": [
                    {
                        "status": "failed",
                        "stage_kind": "behavior",
                        "stage_id_sha256": "a" * 64,
                        "error_type": "RuntimeError",
                    }
                ],
            },
        },
        {
            "stage": "evidence-join-prepush",
            "status": "unexercised",
        },
    ]
    assert secret not in encoded
    assert "command" not in encoded
    assert '"error":' not in encoded
    assert "launcher failed" not in encoded
    assert "permission denied" not in encoded
    assert "unknown operation" not in encoded
    assert "stdout" not in diagnostics
    assert "stderr" not in diagnostics
    assert "x" * 100 not in encoded
    assert len(encoded) < 4096


def test_rehearsal_failure_diagnostics_retain_marker_before_long_cleanup(
    monkeypatch,
    tmp_path: Path,
):
    candidate_sha = "e" * 40
    durable_root = tmp_path / (
        f"leadpoet-rehearsal-failure-{candidate_sha[:12]}-full-path-cleanup"
    )
    durable_root.mkdir()
    summary = {
        "candidate_sha": candidate_sha,
        "status": "failed",
        "stages": [
            {
                "stage": "validator-forward-1",
                "status": "failed",
                "error_type": "CalledProcessError",
                "returncode": 1,
            }
        ],
    }
    (durable_root / "failure-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    monkeypatch.setattr(fast_parity.tempfile, "gettempdir", lambda: str(tmp_path))
    secret = "must-not-escape-long-cleanup"
    result = fast_parity.subprocess.CompletedProcess(
        args=["rehearsal"],
        returncode=1,
        stdout="",
        stderr=(
            f"REHEARSAL_BATCH_FAILURE_EVIDENCE {durable_root}\n"
            + (f"cleanup {secret}\n" * 1024)
        ),
    )

    diagnostics = fast_parity._rehearsal_failure_diagnostics(
        result, candidate_sha=candidate_sha
    )
    encoded = json.dumps(diagnostics, sort_keys=True)
    assert diagnostics["failure_summary_available"] is True
    assert diagnostics["failed_stage_count"] == 1
    assert diagnostics["stages"] == [
        {
            "stage": "validator-forward-1",
            "status": "failed",
            "error_type": "CalledProcessError",
            "returncode": 1,
        }
    ]
    assert secret not in encoded
    assert "stdout" not in diagnostics
    assert "stderr" not in diagnostics
    assert len(encoded) < 4096


def test_rehearsal_fixed_diagnostic_markers_are_strict_and_secret_safe():
    secret = "must-not-escape-fixed-marker"
    stage_hash = "a" * 64
    output = "\n".join(
        [
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=fixture-preparation "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=fixture-preparation "
            "status=passed duration_seconds=48.123",
            "REHEARSAL_EVIDENCE_NORMALIZATION_FAILED "
            "phase=container category=permission status=23",
            "REHEARSAL_FIXTURE_GENERATION_FAILED category=resource status=137",
            "REHEARSAL_WORKFLOW_FAILURE_SUMMARY "
            "failed=1 unexercised=2 emitted=3 truncated=0",
            "REHEARSAL_WORKFLOW_STAGE_RESULT "
            f"status=failed stage_kind=behavior stage_id_sha256={stage_hash} "
            "error_type=RuntimeError",
            "REHEARSAL_WORKFLOW_DIAGNOSTIC_UNAVAILABLE "
            "category=not_found status=127",
            "REHEARSAL_EVIDENCE_NORMALIZATION_FAILED "
            f"phase=host category=permission status=1 token={secret}",
            "REHEARSAL_FIXTURE_GENERATION_FAILED category=secret status=1",
            "REHEARSAL_WORKFLOW_STAGE_RESULT "
            f"status=failed stage_kind=behavior stage_id_sha256={stage_hash} "
            f"error_type=RuntimeError bearer={secret}",
            "REHEARSAL_WORKFLOW_FAILURE_SUMMARY "
            "failed=999 unexercised=0 emitted=0 truncated=0",
            "REHEARSAL_PREPUSH_PHASE phase=unknown-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=unknown duration_seconds=1.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=failed duration_seconds=601.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            f"status=failed duration_seconds=1.0 token={secret}",
        ]
    )

    phase_diagnostics = fast_parity._prepush_phase_diagnostics(output)
    ordinary_diagnostics = fast_parity._rehearsal_output_diagnostics(output)
    diagnostics = [*phase_diagnostics, *ordinary_diagnostics]
    assert fast_parity.SAFE_PREPUSH_PHASES == restart_rehearsal._PREPUSH_PHASES
    assert diagnostics == [
        {
            "marker": "prepush_phase",
            "phase": "fixture-preparation",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "prepush_phase",
            "phase": "fixture-preparation",
            "status": "passed",
            "duration_seconds": 48.123,
        },
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "evidence_normalization_failure",
            "phase": "container",
            "category": "permission",
            "status": 23,
        },
        {
            "marker": "fixture_generation_failure",
            "category": "resource",
            "status": 137,
        },
        {
            "marker": "workflow_failure_summary",
            "failed_count": 1,
            "unexercised_count": 2,
            "emitted_count": 3,
            "truncated": False,
        },
        {
            "marker": "workflow_stage_result",
            "status": "failed",
            "stage_kind": "behavior",
            "stage_id_sha256": stage_hash,
            "error_type": "RuntimeError",
        },
        {
            "marker": "workflow_diagnostic_unavailable",
            "category": "not_found",
            "status": 127,
        },
    ]
    assert all(item["marker"] != "prepush_phase" for item in ordinary_diagnostics)
    assert secret not in json.dumps(diagnostics, sort_keys=True)


def test_prepush_phase_markers_survive_full_stream_beyond_ordinary_tail():
    candidate_sha = "e" * 40
    secret = "must-not-escape-phase-cap"
    stderr = "\n".join(
        [
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0",
            *(
                "REHEARSAL_FAILURE_DIAGNOSTICS component=workflow "
                f"status={status}"
                for status in range(40)
            ),
            "x" * 9000,
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=passed duration_seconds=321.123",
            "REHEARSAL_FAILURE_DIAGNOSTICS component=workflow status=250",
            "REHEARSAL_PREPUSH_PHASE phase=validator-runtime "
            f"status=started duration_seconds=0.0 token={secret}",
            f"raw={secret}",
        ]
    )
    result = subprocess.CompletedProcess(
        ["rehearsal"],
        1,
        stdout="",
        stderr=stderr,
    )

    diagnostics = fast_parity._rehearsal_failure_diagnostics(
        result,
        candidate_sha=candidate_sha,
    )

    assert diagnostics["output_markers"] == [
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "passed",
            "duration_seconds": 321.123,
        },
        {"marker": "component_failure", "component": "workflow", "status": 250},
    ]
    encoded = json.dumps(diagnostics, sort_keys=True)
    assert secret not in encoded
    assert "raw=" not in encoded
    assert "x" * 100 not in encoded


@pytest.mark.parametrize(
    ("spoofed_terminal", "controller_terminal"),
    (("passed", "failed"), ("failed", "passed")),
)
def test_prepush_phase_conflicting_terminal_is_omitted_fail_closed(
    spoofed_terminal: str,
    controller_terminal: str,
):
    stream = "\n".join(
        (
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            f"status={spoofed_terminal} duration_seconds=1.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            f"status={controller_terminal} duration_seconds=2.0",
        )
    )

    assert fast_parity._prepush_phase_diagnostics(stream) == [
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        }
    ]
    assert fast_parity._rehearsal_output_diagnostics(stream) == []


def test_stdout_phase_spoof_is_ignored_by_failure_composition():
    candidate_sha = "f" * 40
    stdout = (
        "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
        "status=started duration_seconds=0.0\n"
        "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
        "status=failed duration_seconds=1.0\n"
    )
    stderr = (
        "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
        "status=started duration_seconds=0.0\n"
        "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
        "status=passed duration_seconds=2.0\n"
    )

    diagnostics = fast_parity._rehearsal_failure_diagnostics(
        subprocess.CompletedProcess(
            ["rehearsal"],
            1,
            stdout=stdout,
            stderr=stderr,
        ),
        candidate_sha=candidate_sha,
    )
    assert diagnostics["output_markers"] == [
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "prepush_phase",
            "phase": "workflow-runtime",
            "status": "passed",
            "duration_seconds": 2.0,
        },
    ]


def test_duplicate_phase_start_on_controller_stream_is_omitted():
    stream = "\n".join(
        (
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
            "status=passed duration_seconds=2.0",
        )
    )

    assert fast_parity._prepush_phase_diagnostics(stream) == []


def test_high_volume_duplicate_phase_markers_remain_fail_closed_and_bounded():
    duplicate_count = 10_000
    stream = "\n".join(
        (
            *(
                "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
                "status=started duration_seconds=0.0"
                for _ in range(duplicate_count)
            ),
            *(
                "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
                "status=passed duration_seconds=1.0"
                for _ in range(duplicate_count)
            ),
            "REHEARSAL_PREPUSH_PHASE phase=validator-runtime "
            "status=started duration_seconds=0.0",
            "REHEARSAL_PREPUSH_PHASE phase=validator-runtime "
            "status=passed duration_seconds=2.0",
        )
    )

    assert fast_parity._prepush_phase_diagnostics(stream) == [
        {
            "marker": "prepush_phase",
            "phase": "validator-runtime",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "prepush_phase",
            "phase": "validator-runtime",
            "status": "passed",
            "duration_seconds": 2.0,
        },
    ]


def test_image_build_failure_projects_only_strict_buildkit_lifecycle():
    secret = "must-not-escape-image-build-diagnostic"
    stderr = "\n".join(
        (
            "#7 [3/8] RUN echo REHEARSAL_IMAGE_BUILD_PHASE "
            "phase=image-finalization status=failed",
            "#7 0.001 REHEARSAL_IMAGE_BUILD_PHASE "
            "phase=system-packages status=started",
            "#7 12.345 REHEARSAL_IMAGE_BUILD_PHASE "
            "phase=system-packages status=passed",
            "#9 0.002 REHEARSAL_IMAGE_BUILD_PHASE "
            "phase=python-dependencies status=started",
            "#9 4.567 REHEARSAL_IMAGE_BUILD_PHASE "
            "phase=python-dependencies status=failed",
            f"#9 4.568 raw bearer={secret}",
        )
    )

    diagnostics = fast_parity._image_build_failure_diagnostics(
        stderr,
        exact_image_build_failed=True,
    )

    assert diagnostics == [
        {
            "marker": "image_build_failure",
            "phase": "python-dependencies",
            "category": "build_command_failed",
        }
    ]
    assert secret not in json.dumps(diagnostics, sort_keys=True)


def test_image_build_failure_projects_post_phase_export_or_load_boundary():
    markers = []
    for index, phase in enumerate(
        fast_parity.SAFE_IMAGE_BUILD_PHASE_ORDER,
        start=1,
    ):
        markers.extend(
            (
                f"#{index} 0.001 REHEARSAL_IMAGE_BUILD_PHASE "
                f"phase={phase} status=started",
                f"#{index} 1.234 REHEARSAL_IMAGE_BUILD_PHASE "
                f"phase={phase} status=passed",
            )
        )

    assert fast_parity._image_build_failure_diagnostics(
        "\n".join(markers),
        exact_image_build_failed=True,
    ) == [
        {
            "marker": "image_build_failure",
            "phase": "image-export-load",
            "category": "build_export_or_load_failed",
        }
    ]


@pytest.mark.parametrize(
    "markers",
    (
        (
            ("system-packages", "started"),
            ("system-packages", "started"),
            ("system-packages", "failed"),
        ),
        (
            ("system-packages", "started"),
            ("system-packages", "passed"),
            ("system-packages", "failed"),
        ),
        (
            ("python-dependencies", "started"),
            ("python-dependencies", "failed"),
        ),
        (("system-packages", "failed"),),
        (("unknown-phase", "started"),),
    ),
)
def test_image_build_failure_rejects_ambiguous_lifecycle(markers):
    stderr = "\n".join(
        f"#{index} 0.001 REHEARSAL_IMAGE_BUILD_PHASE "
        f"phase={phase} status={status}"
        for index, (phase, status) in enumerate(markers, start=1)
    )

    assert fast_parity._image_build_failure_diagnostics(
        stderr,
        exact_image_build_failed=True,
    ) == [
        {
            "marker": "image_build_failure",
            "phase": "unknown",
            "category": "unlocalized",
        }
    ]


def test_cached_image_build_failure_without_markers_remains_unlocalized():
    candidate_sha = "b" * 40
    stderr = "\n".join(
        (
            "REHEARSAL_PREPUSH_PHASE phase=exact-image-build "
            "status=started duration_seconds=0.0",
            "#7 CACHED",
            "REHEARSAL_PREPUSH_PHASE phase=exact-image-build "
            "status=failed duration_seconds=156.35",
        )
    )

    diagnostics = fast_parity._rehearsal_failure_diagnostics(
        subprocess.CompletedProcess(
            ["rehearsal"],
            1,
            stdout="",
            stderr=stderr,
        ),
        candidate_sha=candidate_sha,
    )

    assert diagnostics["output_markers"] == [
        {
            "marker": "prepush_phase",
            "phase": "exact-image-build",
            "status": "started",
            "duration_seconds": 0.0,
        },
        {
            "marker": "prepush_phase",
            "phase": "exact-image-build",
            "status": "failed",
            "duration_seconds": 156.35,
        },
        {
            "marker": "image_build_failure",
            "phase": "unknown",
            "category": "unlocalized",
        },
    ]


def test_fast_rehearsal_parent_waits_for_inner_budget_failure_evidence(
    monkeypatch,
    tmp_path: Path,
):
    candidate_sha = "c" * 40
    secret = "must-not-escape-inner-timeout-evidence"
    durable_root = tmp_path / (
        f"leadpoet-rehearsal-failure-{candidate_sha[:12]}-full-path-timeout"
    )
    durable_root.mkdir()
    summary = {
        "candidate_sha": candidate_sha,
        "status": "failed",
        "stages": [
            {
                "duration_seconds": 600,
                "error": secret,
                "error_type": "RehearsalTimeBudgetExceeded",
                "stage": "time-budget",
                "status": "failed",
            },
            {
                "blocked_by": ["time-budget"],
                "stage": "evidence-join-prepush",
                "status": "unexercised",
            },
        ],
    }
    (durable_root / "failure-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    observed: dict[str, int] = {}

    def inner_budget_failure(command, *, timeout, **_kwargs):
        observed["timeout"] = timeout
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=(
                "REHEARSAL_TIME_BUDGET_EXCEEDED profile=prepush "
                "error='prepush rehearsal exceeded its 600-second wall-clock budget'\n"
                f"REHEARSAL_BATCH_FAILURE_EVIDENCE {durable_root}\n"
            ),
        )

    monkeypatch.setattr(fast_parity, "_run", inner_budget_failure)
    monkeypatch.setattr(fast_parity.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        fast_parity,
        "_rehearsal_evidence_path",
        lambda _candidate_sha: tmp_path / "joined-evidence.json",
    )

    with pytest.raises(ProductionParityError) as raised:
        fast_parity._run_rehearsal(
            base_sha="b" * 40,
            candidate_sha=candidate_sha,
        )

    message = str(raised.value)
    assert observed == {"timeout": 720}
    assert "candidate-derived N-1 rehearsal failed" in message
    assert "parent watchdog timed out" not in message
    assert '"failure_summary_available":true' in message
    assert '"stage":"time-budget"' in message
    assert '"stage":"evidence-join-prepush"' in message
    assert secret not in message


@pytest.mark.parametrize(
    ("stream_kind", "expects_safe_markers"),
    (("bytes", True), ("str", True), ("malformed", False)),
)
def test_fast_parent_timeout_projects_only_sanitized_child_evidence(
    monkeypatch,
    tmp_path: Path,
    stream_kind: str,
    expects_safe_markers: bool,
):
    candidate_sha = "d" * 40
    secret = "must-not-escape-parent-timeout-output"

    stdout_text = (
        "REHEARSAL_FAILURE_DIAGNOSTICS component=workflow status=124\n"
        f"raw={secret}\n"
    )
    stderr_text = (
        "REHEARSAL_PREPUSH_PHASE phase=workflow-runtime "
        "status=started duration_seconds=0.0\n"
        + ("x" * 9000)
        + "\n"
        "REHEARSAL_WORKFLOW_DIAGNOSTIC_UNAVAILABLE "
        "category=resource status=124\n"
        f"ERROR: bearer token={secret} permission denied\n"
    )
    if stream_kind == "bytes":
        timeout_stdout = stdout_text.encode()
        timeout_stderr = stderr_text.encode()
    elif stream_kind == "str":
        timeout_stdout = stdout_text
        timeout_stderr = stderr_text
    else:
        timeout_stdout = {"malformed": secret}
        timeout_stderr = ["malformed", secret]

    def parent_timeout(command, *, timeout, **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=timeout_stdout,
            stderr=timeout_stderr,
        )

    monkeypatch.setattr(fast_parity, "_run", parent_timeout)
    monkeypatch.setattr(
        fast_parity,
        "_rehearsal_evidence_path",
        lambda _candidate_sha: tmp_path / "joined-evidence.json",
    )

    with pytest.raises(ProductionParityError) as raised:
        fast_parity._run_rehearsal(
            base_sha="b" * 40,
            candidate_sha=candidate_sha,
        )

    message = str(raised.value)
    assert "parent watchdog timed out" in message
    assert '"parent_watchdog_timeout_seconds":720' in message
    if expects_safe_markers:
        assert '"marker":"prepush_phase"' in message
        assert '"phase":"workflow-runtime"' in message
        assert '"status":"started"' in message
        assert '"component":"workflow"' in message
        assert '"category":"resource"' in message
    else:
        assert '"component":"workflow"' not in message
        assert '"category":"resource"' not in message
    assert '"returncode":124' in message
    assert secret not in message
    assert "malformed" not in message
    assert "raw=" not in message
    assert "bearer token" not in message


def test_fast_workflow_budget_covers_sequential_database_and_rehearsal():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/production-parity-fast.yml").read_text(
            encoding="utf-8"
        )
    )
    outer_seconds = int(workflow["jobs"]["validate"]["timeout-minutes"]) * 60
    role_step = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("name") == "Configure read-only parity role"
    )
    role_duration_seconds = int(role_step["with"]["role-duration-seconds"])
    expected_minimum = (
        2 * DEFAULT_SNAPSHOT_IO_TIMEOUT_SECONDS
        + fast_parity.FAST_CANDIDATE_MIGRATION_HEADROOM_COUNT
        * DEFAULT_CANDIDATE_MIGRATION_TIMEOUT_SECONDS
        + fast_parity.FAST_REHEARSAL_TIMEOUT_SECONDS
        + fast_parity.FAST_SCHEMA_PREFLIGHT_NETWORK_PROBE_COUNT
        * fast_parity.FAST_SCHEMA_PREFLIGHT_TIMEOUT_SECONDS
        + fast_parity.FAST_DOCKER_STARTUP_AND_CLEANUP_HEADROOM_SECONDS
        + fast_parity.FAST_PRODUCTION_DATA_READ_TIMEOUT_SECONDS
        + fast_parity.FAST_JOB_SETUP_HEADROOM_SECONDS
    )
    assert fast_parity.FAST_SCHEMA_PREFLIGHT_NETWORK_PROBE_COUNT == (
        len(schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA) + 3
    )
    assert fast_parity.FAST_CANDIDATE_MIGRATION_HEADROOM_COUNT == 2
    terminate_bound = 2 * restart_rehearsal._WORKER_TERMINATE_GRACE_SECONDS
    container_bound = (
        2 * restart_rehearsal._WORKER_DOCKER_CLEANUP_SECONDS
        + restart_rehearsal._WORKER_DOCKER_CONVERGENCE_SECONDS
        + 2 * restart_rehearsal._WORKER_DOCKER_CLEANUP_SECONDS
    )
    process_cleanup_bound = 2 * terminate_bound + container_bound
    scheduler_cleanup_bound = 2 * process_cleanup_bound
    normalization_cleanup_bound = (
        restart_rehearsal._EVIDENCE_NORMALIZATION_TIMEOUT_SECONDS
        + process_cleanup_bound
    )
    bounded_external_cleanup = (
        scheduler_cleanup_bound + normalization_cleanup_bound
    )
    assert restart_rehearsal.PROFILE_LIMITS["prepush"]["target_seconds"] == 600
    assert fast_parity.FAST_REHEARSAL_INNER_TIMEOUT_SECONDS == 600
    assert bounded_external_cleanup == 96
    assert fast_parity.FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS == 120
    assert (
        fast_parity.FAST_REHEARSAL_PARENT_CLEANUP_HEADROOM_SECONDS
        > bounded_external_cleanup
    )
    assert fast_parity.FAST_REHEARSAL_TIMEOUT_SECONDS == 720
    assert fast_parity._fast_job_minimum_timeout_seconds(2) == expected_minimum
    assert fast_parity.FAST_JOB_MINIMUM_TIMEOUT_SECONDS == expected_minimum
    assert outer_seconds == fast_parity.FAST_JOB_OUTER_TIMEOUT_SECONDS
    assert outer_seconds - expected_minimum >= 10 * 60
    assert fast_parity._fast_job_minimum_timeout_seconds(7) >= outer_seconds
    assert role_duration_seconds == fast_parity.FAST_AWS_ROLE_DURATION_SECONDS
    assert role_duration_seconds - outer_seconds >= 19 * 60


def test_fast_workflow_freezes_snapshot_source_to_contract_base():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/production-parity-fast.yml").read_text(
            encoding="utf-8"
        )
    )
    step = next(
        item
        for item in workflow["jobs"]["validate"]["steps"]
        if item.get("name") == "Capture production shape and run bounded parity"
    )
    script = str(step["run"])
    assert '--source-sha "${{ steps.inputs.outputs.base }}"' in script
    assert "resolve_production_parity_deployed_sha.py" not in script


def test_fast_live_boundary_executes_get_without_disclosing_credential():
    credential = "production-read-credential"
    provider = _ProductionReadOnlySupabaseProvider(
        origin="https://qplwoislplkcegvdmbim.supabase.co",
        service_role_key=credential,
    )
    observed = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'[{"epoch_id":24570}]'

    class _Opener:
        @staticmethod
        def open(request, *, timeout):
            observed["url"] = request.full_url
            observed["headers"] = dict(request.header_items())
            observed["timeout"] = timeout
            return _Response()

    provider._opener = _Opener()
    result = provider(
        {
            "provider_id": "supabase",
            "method": "GET",
            "url": (
                "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                "research_lab_finalized_allocation_epochs_v2?select=epoch_id"
            ),
            "headers": {
                "authorization": "Bearer attacker-value",
                "apikey": "attacker-value",
                "range": "0-1",
            },
            "body_b64": "",
            "timeout_ms": 2000,
            "logical_operation_id": "weight-history-page-1",
        }
    )
    assert observed["url"].startswith(
        "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
    )
    assert observed["headers"]["Authorization"] == f"Bearer {credential}"
    assert observed["headers"]["Apikey"] == credential
    assert observed["timeout"] == 2
    assert base64.b64decode(result["body_b64"]) == b'[{"epoch_id":24570}]'
    assert credential not in json.dumps(provider.pages, sort_keys=True)
    assert "attacker-value" not in json.dumps(provider.pages, sort_keys=True)


def test_fast_live_boundary_enforces_per_request_and_aggregate_deadlines(
    monkeypatch,
):
    now = [100.0]
    monkeypatch.setattr(fast_parity.time, "monotonic", lambda: now[0])
    provider = _ProductionReadOnlySupabaseProvider(
        origin="https://qplwoislplkcegvdmbim.supabase.co",
        service_role_key="production-read-credential",
        deadline_monotonic=108.0,
    )
    observed: dict[str, float] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b"[]"

    class Opener:
        @staticmethod
        def open(_request, *, timeout):
            observed["timeout"] = timeout
            return Response()

    provider._opener = Opener()
    request = {
        "provider_id": "supabase",
        "method": "GET",
        "url": (
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
            "research_lab_finalized_allocation_epochs_v2?select=epoch_id"
        ),
        "headers": {"range": "0-1"},
        "body_b64": "",
        "timeout_ms": 45_000,
        "logical_operation_id": "bounded-page",
    }
    provider(request)
    assert observed["timeout"] == (
        fast_parity.FAST_PRODUCTION_DATA_READ_REQUEST_TIMEOUT_SECONDS
    )

    now[0] = 108.0
    with pytest.raises(ProductionParityError, match="deadline expired"):
        provider(request)


@pytest.mark.asyncio
def test_critical_stage_ledger_fails_closed_and_hashes_evidence():
    ledger = StageLedger(
        lane="full",
        candidate_sha=SHA,
        contract_hash=HASH,
        snapshot_hash=HASH,
        critical_stage_ids=("scoring", "weights"),
    )
    ledger.record(
        "scoring",
        status="passed",
        duration_seconds=1.5,
        evidence={"all_icps": 40},
    )
    failed = ledger.finalize()
    assert failed["status"] == "failed"
    assert failed["missing_critical_stage_ids"] == ["weights"]
    ledger.record(
        "weights",
        status="passed",
        duration_seconds=0.5,
        evidence={"primary_audit_equal": True},
    )
    passed = validate_ledger(ledger.finalize())
    assert passed["status"] == "passed"


def test_gateway_secret_keeps_real_reads_but_isolates_every_mutation():
    artifact_bucket = "leadpoet-parity-493765492819-" + "f" * 16
    environment = build_gateway_environment(
        {
            "OPENROUTER_API_KEY": "runtime-key",
            "EXA_API_KEY": "exa-key",
            "SCRAPINGDOG_API_KEY": "dog-key",
            "OPENROUTER_MANAGEMENT_KEY": "must-drop",
            "WALLET_PRIVATE_KEY": "must-drop",
            "SUPABASE_URL": "https://qplwoislplkcegvdmbim.supabase.co",
            "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "true",
            "RESEARCH_LAB_RAW_TRACE_S3_PREFIX": "s3://production/raw",
            "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX": "s3://production/scorer",
            "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX": (
                "s3://production/incontainer"
            ),
            "PATH": "/production/poison",
            "PYTHONPATH": "/production/import-poison",
            "HTTP_PROXY": "http://production-proxy.invalid",
            "HTTPS_PROXY": "http://production-proxy.invalid",
            "GIT_SSH_COMMAND": "ssh -i /production/key",
            "GATEWAY_RESTART_GIT_SSH_COMMAND": (
                "ssh -i /production/restart-key"
            ),
            "AWS_ACCESS_KEY_ID": "must-drop",
            "AWS_SECRET_ACCESS_KEY": "must-drop",
            "GATEWAY_ENV_FILE": "/production/gateway.env",
            "GATEWAY_PRIVATE_KEY_PATH": "/production/gateway-private-key.pem",
            "ARWEAVE_KEYFILE_PATH": "/production/arweave-keyfile.json",
            "GATEWAY_PYTHON_BIN": "/production/python",
            "GATEWAY_RESTART_LOCK_FILE": "/production/restart.lock",
            "GATEWAY_TEE_EIF_ROOT": "/production/tee",
            "GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST": "/production/corpus.json",
            "GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT": "/production/corpus",
            "GATEWAY_V2_CONFIG_DIR": "/production/v2",
            "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": "/production/offline",
            "GATEWAY_V2_RELEASE_BUCKET": "production-bucket-poison",
            "GATEWAY_V2_RELEASE_PREFIX": "production-prefix-poison",
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": "/production/docker.lock",
            "LEADPOET_GATEWAY_ENV_SECRET_ID": "production-secret-poison",
            "LEADPOET_REPO_ROOT": "/production/repo",
            "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT": "/production/validator",
            "DISABLE_BACKGROUND_TASKS": "false",
            "GATEWAY_STATEFUL_CUTOVER_CEREMONY": "1",
            "GATEWAY_TEE_TOPOLOGY_MODE": "component",
            "RESEARCH_LAB_PROVIDER_HTTP_PROXY": "provider-proxy-ref",
            "LANGFUSE_ENABLED": "true",
            "LANGFUSE_PUBLIC_KEY": "must-drop",
            "LANGFUSE_SECRET_KEY": "must-drop",
            "LANGFUSE_HOST": "https://telemetry.example.invalid",
            "LANGFUSE_BASE_URL": "https://telemetry.example.invalid",
            "MINIO_ACCESS_KEY": "must-drop",
            "MINIO_SECRET_KEY": "must-drop",
            "MINIO_ENDPOINT": "https://object-store.example.invalid",
            "MINIO_BUCKET": "production-poison",
            "AWS_S3_BUCKET": "production-leads-poison",
            "RESEARCH_LAB_CORPUS_EXPORT_ENABLED": "true",
            "RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX": "s3://production/corpus",
            "RESEARCH_LAB_EVIDENCE_PROXY_URL": "http://production-proxy:8765",
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR": (
                "/production/provider-evidence-cache"
            ),
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": (
                "/production/provider-evidence.jsonl"
            ),
            "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH": (
                "/production/provider-outcomes.jsonl"
            ),
            "RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX": (
                "s3://production/signatures"
            ),
            "RESEARCH_LAB_SCORING_CACHE_DIR": "/production/scoring-cache",
            "LAB_ARENA_MODE": "live",
            "LAB_ARENA_SUPABASE_URL": (
                "https://qplwoislplkcegvdmbim.supabase.co"
            ),
            "LAB_ARENA_SUPABASE_ANON_KEY": "production-anon-poison",
            "LAB_ARENA_SERVICE_JWT": "production.arena.poison",
            "LAB_ARENA_BUCKET": "production-arena-poison",
        },
        run_id="pp-1-1",
        candidate_sha=SHA,
        gateway_public_key=PARITY_GATEWAY_PUBLIC_KEY,
        supabase_origin=ORIGIN,
        artifact_bucket=artifact_bucket,
        benchmark_date="2026-08-16",
        jwt_secret="j" * 48,
    )
    assert environment["OPENROUTER_API_KEY"] == "runtime-key"
    assert environment["EXA_API_KEY"] == "exa-key"
    assert environment["SCRAPINGDOG_API_KEY"] == "dog-key"
    assert "OPENROUTER_MANAGEMENT_KEY" not in environment
    assert "WALLET_PRIVATE_KEY" not in environment
    assert environment["SUPABASE_URL"] == ORIGIN
    assert (
        environment["RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET"]
        == artifact_bucket
    )
    assert {
        name: environment[name]
        for name in production_parity_trace_prefixes(
            artifact_bucket=artifact_bucket,
            run_id="pp-1-1",
        )
    } == production_parity_trace_prefixes(
        artifact_bucket=artifact_bucket,
        run_id="pp-1-1",
    )
    for key in (
        "PATH",
        "PYTHONPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "GIT_SSH_COMMAND",
        "GATEWAY_RESTART_GIT_SSH_COMMAND",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GATEWAY_ENV_FILE",
        "GATEWAY_PRIVATE_KEY_PATH",
        "ARWEAVE_KEYFILE_PATH",
        "GATEWAY_PYTHON_BIN",
        "GATEWAY_RESTART_LOCK_FILE",
        "GATEWAY_TEE_EIF_ROOT",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT",
        "GATEWAY_V2_CONFIG_DIR",
        "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
        "GATEWAY_V2_RELEASE_BUCKET",
        "GATEWAY_V2_RELEASE_PREFIX",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        "LEADPOET_GATEWAY_ENV_SECRET_ID",
        "LEADPOET_REPO_ROOT",
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT",
    ):
        assert key not in environment
        assert is_process_control_environment_key(key)
    assert (
        environment["RESEARCH_LAB_PROVIDER_HTTP_PROXY"]
        == "provider-proxy-ref"
    )
    assert environment["ENABLE_FULFILLMENT"] == "false"
    assert environment["BITTENSOR_NETWORK"] == "finney"
    assert environment["BITTENSOR_NETUID"] == "71"
    assert environment["RESEARCH_LAB_SCORING_WORKER_ENABLED"] == "true"
    assert environment["RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_PAID_LOOPS_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_HOSTED_RUNS_ENABLED"] == "true"
    assert environment["RESEARCH_LAB_HOSTED_WORKER_ENABLED"] == "true"
    assert environment["RESEARCH_LAB_HOSTED_WORKER_DRY_RUN"] == "true"
    assert environment["RESEARCH_LAB_HOSTED_WORKER_MAX_RUNS"] == "0"
    assert environment["RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_AUTO_PROMOTION_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_AUTO_COMMIT_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED"] == "false"
    assert environment["DISABLE_BACKGROUND_TASKS"] == "true"
    assert environment["GATEWAY_STATEFUL_CUTOVER_CEREMONY"] == "0"
    assert environment["GATEWAY_TEE_TOPOLOGY_MODE"] == "full"
    assert environment["LANGFUSE_ENABLED"] == "false"
    assert environment["LAB_ARENA_MODE"] == "off"
    assert environment["LAB_ARENA_SUPABASE_URL"] == ORIGIN
    assert environment["LAB_ARENA_SUPABASE_ANON_KEY"].count(".") == 2
    assert environment["LAB_ARENA_SERVICE_JWT"].count(".") == 2
    assert environment["LAB_ARENA_BUCKET"] == artifact_bucket
    assert environment["AWS_S3_BUCKET"] == artifact_bucket
    assert environment["RESEARCH_LAB_CORPUS_EXPORT_ENABLED"] == "false"
    assert environment["RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX"] == ""
    assert environment["RESEARCH_LAB_EVIDENCE_PROXY_URL"] == ""
    assert environment["RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR"] == ""
    assert environment["RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH"] == ""
    assert environment["RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH"] == ""
    assert environment["RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX"] == ""
    assert environment["RESEARCH_LAB_SCORING_CACHE_DIR"] == (
        production_parity_scoring_cache_dir(run_id="pp-1-1")
    )
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_ENDPOINT",
        "MINIO_BUCKET",
    ):
        assert key not in environment


def test_full_clone_environment_rejects_tampered_trace_destination(
    monkeypatch, tmp_path: Path
):
    run_id = "pp-1-1"
    artifact_bucket = "leadpoet-parity-493765492819-" + "f" * 16
    environment = build_gateway_environment(
        {},
        run_id=run_id,
        candidate_sha=SHA,
        gateway_public_key=PARITY_GATEWAY_PUBLIC_KEY,
        supabase_origin=ORIGIN,
        artifact_bucket=artifact_bucket,
        benchmark_date="2026-08-19",
        jwt_secret="j" * 48,
    )
    monkeypatch.setattr(full_host, "FULL_WORK_ROOT", tmp_path)
    gateway_env_file = tmp_path / run_id / "runtime" / "gateway.env"
    gateway_env_file.parent.mkdir(parents=True)

    def write_environment(values):
        gateway_env_file.write_text(
            "".join(f"{key}={value}\n" for key, value in sorted(values.items())),
            encoding="utf-8",
        )
        gateway_env_file.chmod(0o600)

    write_environment(environment)
    assert full_host._validated_clone_environment(
        gateway_env_file,
        candidate_sha=SHA,
        run_id=run_id,
        supabase_origin=ORIGIN,
        artifact_bucket=artifact_bucket,
    )["RESEARCH_LAB_RAW_TRACE_S3_PREFIX"].endswith("/traces/raw")

    write_environment(
        {
            **environment,
            "RESEARCH_LAB_RAW_TRACE_S3_PREFIX": "s3://production-private/raw",
        }
    )
    with pytest.raises(FullParityError, match="clone gateway boundary identity"):
        full_host._validated_clone_environment(
            gateway_env_file,
            candidate_sha=SHA,
            run_id=run_id,
            supabase_origin=ORIGIN,
            artifact_bucket=artifact_bucket,
        )

    substituted_bucket = "leadpoet-parity-493765492819-" + "e" * 16
    write_environment(
        {
            **environment,
            "AWS_S3_BUCKET": substituted_bucket,
            "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET": substituted_bucket,
            **production_parity_trace_prefixes(
                artifact_bucket=substituted_bucket,
                run_id=run_id,
            ),
        }
    )
    with pytest.raises(FullParityError, match="clone gateway boundary identity"):
        full_host._validated_clone_environment(
            gateway_env_file,
            candidate_sha=SHA,
            run_id=run_id,
            supabase_origin=ORIGIN,
            artifact_bucket=artifact_bucket,
        )

    for poison in (
        {"LANGFUSE_ENABLED": "true"},
        {"LANGFUSE_PUBLIC_KEY": "telemetry-poison"},
        {"MINIO_ENDPOINT": "https://object-store.example.invalid"},
        {"AWS_S3_BUCKET": "production-leads-poison"},
        {"RESEARCH_LAB_CORPUS_EXPORT_ENABLED": "true"},
        {"RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX": "s3://production/corpus"},
        {"RESEARCH_LAB_EVIDENCE_PROXY_URL": "http://production-proxy:8765"},
        {
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR": (
                "/production/provider-evidence-cache"
            )
        },
        {
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": (
                "/production/provider-evidence.jsonl"
            )
        },
        {
            "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH": (
                "/production/provider-outcomes.jsonl"
            )
        },
        {
            "RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX": (
                "s3://production/signatures"
            )
        },
        {"RESEARCH_LAB_SCORING_CACHE_DIR": "/production/scoring-cache"},
    ):
        write_environment({**environment, **poison})
        with pytest.raises(
            FullParityError, match="clone gateway boundary identity"
        ):
            full_host._validated_clone_environment(
                gateway_env_file,
                candidate_sha=SHA,
                run_id=run_id,
                supabase_origin=ORIGIN,
                artifact_bucket=artifact_bucket,
            )


def test_full_runtime_identity_is_run_owned_and_non_production(tmp_path: Path):
    identity_dir = tmp_path / "runtime-identity"
    identity = full_host._materialize_run_owned_runtime_identity(identity_dir)

    assert set(identity) == {
        "gateway_private_key_path",
        "gateway_public_key",
        "gateway_public_key_hash",
        "arweave_keyfile_path",
        "arweave_address_hash",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", identity["gateway_public_key"])
    for field in ("gateway_public_key_hash", "arweave_address_hash"):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", identity[field])
    for field in ("gateway_private_key_path", "arweave_keyfile_path"):
        path = Path(identity[field])
        metadata = path.lstat()
        assert path.parent == identity_dir
        assert path.is_file()
        assert not path.is_symlink()
        assert metadata.st_size > 0
        assert metadata.st_uid == os.getuid()
        assert metadata.st_gid == os.getgid()
        assert metadata.st_mode & 0o777 == 0o600


def test_full_runtime_identity_normalizes_inherited_setgid_directory(
    monkeypatch,
    tmp_path: Path,
):
    identity_dir = tmp_path / "runtime-identity"
    original_mkdir = Path.mkdir

    def mkdir_with_inherited_setgid(path: Path, *args, **kwargs) -> None:
        original_mkdir(path, *args, **kwargs)
        if path == identity_dir:
            os.chmod(path, 0o2700)

    monkeypatch.setattr(Path, "mkdir", mkdir_with_inherited_setgid)

    identity = full_host._materialize_run_owned_runtime_identity(identity_dir)

    metadata = identity_dir.lstat()
    assert metadata.st_uid == os.getuid()
    assert metadata.st_gid == os.getgid()
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert Path(identity["gateway_private_key_path"]).is_file()
    assert Path(identity["arweave_keyfile_path"]).is_file()


def test_full_runtime_identity_rejects_existing_directory(tmp_path: Path):
    identity_dir = tmp_path / "runtime-identity"
    identity_dir.mkdir(mode=0o700)
    with pytest.raises(
        FullParityError,
        match="runtime identity directory is unavailable",
    ):
        full_host._materialize_run_owned_runtime_identity(identity_dir)


def test_full_gateway_restart_reasserts_run_owned_path_authority():
    source = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    authority_keys = {
        "GATEWAY_ENV_FILE",
        "GATEWAY_PRIVATE_KEY_PATH",
        "ARWEAVE_KEYFILE_PATH",
        "GATEWAY_RESTART_GIT_SSH_COMMAND",
        "LEADPOET_GATEWAY_ENV_SECRET_ID",
        "GATEWAY_RESTART_CONTROLLER_ROOT",
        "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        "GATEWAY_V2_CONFIG_DIR",
        "GATEWAY_V2_RELEASE_MANIFEST",
        "GATEWAY_V2_RELEASE_LINEAGE",
        "GATEWAY_V2_ARTIFACT_POLICY",
        "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT",
        "GATEWAY_V2_RELEASE_PREFIX",
        "GATEWAY_V2_RELEASE_BUCKET",
        "GATEWAY_V2_KMS_KEY_ID",
    }
    for key in authority_keys:
        assert source.count(f'"{key}"') >= 3
        assert f"  {key}\n" in source
        assert is_process_control_environment_key(key)
    assert source.count(
        'set -a\n. "$ENV_CLONE"\nset +a\n'
        "restore_gateway_restart_path_authority"
    ) == 2
    assert source.count('"GATEWAY_PYTHON_BIN"') >= 3
    assert (
        "printf 'export GATEWAY_PYTHON_BIN=%q\\n' \"$GATEWAY_PYTHON_BIN\" "
        '>> "$ENV_CLONE"'
    ) in source
    assert (
        "printf 'export GATEWAY_PRIVATE_KEY_PATH=%q\\n' "
        '"$GATEWAY_PRIVATE_KEY_PATH" >> "$ENV_CLONE"'
    ) in source
    assert 'GATEWAY_PRIVATE_KEY_PATH="$GATEWAY_PRIVATE_KEY_PATH" \\' in source
    assert (
        "printf 'export ARWEAVE_KEYFILE_PATH=%q\\n' "
        '"$ARWEAVE_KEYFILE_PATH" >> "$ENV_CLONE"'
    ) in source
    assert 'ARWEAVE_KEYFILE_PATH="$ARWEAVE_KEYFILE_PATH" \\' in source
    assert (
        'GATEWAY_RESTART_GIT_SSH_COMMAND="$GATEWAY_RESTART_GIT_SSH_COMMAND" \\'
        in source
    )
    assert is_process_control_environment_key("GATEWAY_PYTHON_BIN")
    full_source = inspect.getsource(full_host.run_full)
    assert '"GATEWAY_V2_RELEASE_BUCKET": ATTESTED_V2_RELEASE_BUCKET' in full_source
    assert '"GATEWAY_V2_KMS_KEY_ID": ATTESTED_V2_KMS_KEY_ID' in full_source
    assert '"GATEWAY_PRIVATE_KEY_PATH": gateway_private_key_path' in full_source
    assert '"ARWEAVE_KEYFILE_PATH": arweave_keyfile_path' in full_source
    assert (
        '"GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT": "full-parity"'
        in full_source
    )
    assert '"GATEWAY_RESTART_GIT_SSH_COMMAND":' not in full_source
    assert "research_lab_private_model_deploy" not in full_source


def test_controller_dependency_closure_includes_runtime_identity_and_database_clients():
    selected = resolve_controller_requirements(ROOT / "requirements.txt")
    names = {re.split(r"[<>=!~ ]", item, maxsplit=1)[0].lower() for item in selected}
    assert names == {
        "arweave-python-client",
        "bittensor",
        "boto3",
        "cryptography",
        "fastapi",
        "httpx",
        "supabase",
        "uvicorn[standard]",
    }
    assert "arweave-python-client>=1.0.19" in selected
    assert 'bittensor==10.5.0; python_version >= "3.10"' in selected


def test_arena_rebenchmark_is_live_complete_and_precedes_weights():
    source = inspect.getsource(full_host.run_full)
    gateway = source.index('failure_stage = "gateway-health"')
    arena = source.index("_run_arena_rebenchmark_path(")
    weights = source.index('failure_stage = "weight-readiness"')
    assert gateway < arena < weights
    assert '"arena_rebenchmark": arena_rebenchmark' in source


def test_arena_rebenchmark_accepts_existing_gateway_provider_key_names():
    assert _arena_provider_keys(
        {
            "OPENROUTER_KEY": "openrouter-test",
            "QUALIFICATION_SCRAPINGDOG_API_KEY": "scrapingdog-test",
            "RESEARCH_LAB_V2_DEEPLINE_API_KEY": "deepline-test",
        }
    ) == {
        "openrouter": "openrouter-test",
        "scrapingdog": "scrapingdog-test",
        "deepline": "deepline-test",
    }


def test_arena_rebenchmark_counts_only_public_https_and_successful_openrouter():
    assert full_host._arena_https_evidence_urls(
        {
            "fit_evidence_urls": [
                "https://evidence.example.com/news",
                "http://evidence.example.com/plaintext",
                "https://localhost/private",
                "https://user@evidence.example.com/credential",
            ],
            "intent_signals": [{"url": "https://127.0.0.1/private"}],
        }
    ) == {"https://evidence.example.com/news"}
    assert full_host._successful_openrouter_settlement_count(
        [
            {
                "entry_kind": "settlement",
                "provider": "openrouter",
                "terminal_response": {"status": 200},
            },
            {
                "entry_kind": "settlement",
                "provider": "openrouter",
                "terminal_response": {"status": 429},
            },
            {
                "entry_kind": "settlement",
                "provider": "deepline",
                "terminal_response": {"status": 200},
            },
            {
                "entry_kind": "dispatch",
                "provider": "openrouter",
                "terminal_response": {"status": 200},
            },
        ]
    ) == 1


def test_clone_postgrest_can_assume_the_candidate_arena_service_role():
    source = inspect.getsource(fast_parity._DockerDatabase.start_postgrest)
    assert "rolname = 'lab_arena_service'" in source
    assert "GRANT lab_arena_service TO authenticator" in source


def test_arena_rebenchmark_evidence_requires_every_icp_and_live_evidence():
    bucket = "leadpoet-parity-493765492819-" + "f" * 16
    evidence = {
        "schema_version": (
            "leadpoet.production_parity_arena_rebenchmark_evidence.v1"
        ),
        "candidate_sha": SHA,
        "run_id": "pp-1-1",
        "artifact_bucket": bucket,
        "status": "passed",
        "mode": "shadow",
        "round_id": "arena-2026-09-04-abcdef123456",
        "evaluation_date": "2026-09-04",
        "daily_icp_set_id": 20260904,
        "baseline_source_url": (
            "https://github.com/leadpoet/pydantic-harness/"
            "archive/refs/heads/main.tar.gz"
        ),
        "baseline_final_score": 72.5,
        "icp_results": [
            {
                "icp_position": position,
                "execute_accepted": True,
                "score_accepted": True,
                "company_count": 1,
                "valid_company_with_https_evidence_count": 1,
                "https_evidence_url_count": 1,
                "successful_openrouter_execute_call_count": 1,
                "successful_openrouter_score_settlement_count": 1,
            }
            for position in range(20)
        ],
        "counts": {
            "configured_icp_count": 20,
            "stage_1_icp_count": 10,
            "stage_2_icp_count": 10,
            "accepted_execute_runs": 20,
            "accepted_score_runs": 20,
            "scored_icp_count": 20,
            "unique_icp_positions": 20,
            "company_count": 20,
            "evidence_url_count": 20,
        },
        "providers": {
            "transport": "live-httpx",
            "names": ["deepline", "openrouter", "scrapingdog"],
            "settled_provider_call_count": 87,
            "execute_settled_provider_call_count": 42,
            "score_settled_provider_call_count": 45,
            "successful_openrouter_execute_call_count": 20,
            "successful_openrouter_score_settlement_count": 20,
        },
        "runtime": {
            "runner": "lab_arena.runner.Runner",
            "sandbox": "gvisor-runsc",
            "api": "lab_arena.api.loopback-http",
            "object_store": "s3",
            "judge_image_materialization": "exact-candidate-local-docker",
        },
        "restart_recovery": {
            "service_restarted": True,
            "runner_restarted": True,
            "resumed_round_status": "stage1",
            "persisted_execute_runs": 10,
        },
        "publication_visible": True,
        "public_benchmark_visible": True,
        "public_results_visible": True,
        "production_database_mutated": False,
        "production_chain_mutated": False,
    }
    assert full_host._validate_arena_rebenchmark_evidence(
        evidence,
        candidate_sha=SHA,
        run_id="pp-1-1",
        artifact_bucket=bucket,
    )["status"] == "passed"

    broken_values = []
    for section, key, value in (
        ("counts", "stage_1_icp_count", 9),
        ("counts", "accepted_execute_runs", 19),
        ("counts", "accepted_score_runs", 19),
        ("counts", "company_count", 0),
        ("counts", "evidence_url_count", 0),
        ("providers", "execute_settled_provider_call_count", 0),
        ("providers", "score_settled_provider_call_count", 0),
        ("providers", "successful_openrouter_execute_call_count", 19),
        ("providers", "successful_openrouter_score_settlement_count", 19),
        ("restart_recovery", "service_restarted", False),
    ):
        broken = json.loads(json.dumps(evidence))
        broken[section][key] = value
        broken_values.append(broken)
    for broken in broken_values:
        with pytest.raises(FullParityError, match="evidence is incomplete"):
            full_host._validate_arena_rebenchmark_evidence(
                broken,
                candidate_sha=SHA,
                run_id="pp-1-1",
                artifact_bucket=bucket,
            )

    per_icp_broken_values = []
    missing_position = json.loads(json.dumps(evidence))
    missing_position["icp_results"].pop()
    per_icp_broken_values.append(missing_position)
    duplicate_position = json.loads(json.dumps(evidence))
    duplicate_position["icp_results"][-1]["icp_position"] = 18
    per_icp_broken_values.append(duplicate_position)
    for key, value in (
        ("execute_accepted", False),
        ("score_accepted", False),
        ("company_count", 0),
        ("valid_company_with_https_evidence_count", 0),
        ("https_evidence_url_count", 0),
        ("successful_openrouter_execute_call_count", 0),
        ("successful_openrouter_score_settlement_count", 0),
    ):
        broken = json.loads(json.dumps(evidence))
        broken["icp_results"][7][key] = value
        per_icp_broken_values.append(broken)
    for broken in per_icp_broken_values:
        with pytest.raises(FullParityError, match="evidence is incomplete"):
            full_host._validate_arena_rebenchmark_evidence(
                broken,
                candidate_sha=SHA,
                run_id="pp-1-1",
                artifact_bucket=bucket,
            )


def test_full_runner_accepts_json_secret_and_strict_env_file(tmp_path: Path):
    dsn = "postgresql://reader:password@db.example.com:5432/postgres"
    assert _dsn_from_secret(json.dumps({"readonly_dsn": dsn})) == dsn
    env_file = tmp_path / "gateway.env"
    env_file.write_text(
        "export SIMPLE=value\nMULTI=a value with spaces\n"
        "UNMATCHED='raw shell junk\nEMPTY=\nSIMPLE=value\n",
        encoding="utf-8",
    )
    assert _parse_gateway_environment_file(env_file) == {
        "SIMPLE": "value",
        "MULTI": "a value with spaces",
        "UNMATCHED": "'raw shell junk",
        "EMPTY": "",
    }
    env_file.write_text("DUP=one\nDUP=two\n", encoding="utf-8")
    with pytest.raises(FullParityError, match="conflicting duplicate"):
        _parse_gateway_environment_file(env_file)
    env_file.write_bytes(b"BAD=value\x00poison\n")
    with pytest.raises(FullParityError, match="invalid row"):
        _parse_gateway_environment_file(env_file)


def test_full_parent_import_never_loads_gateway_database_globals():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import scripts.run_production_parity_full_host  # noqa: F401
for name in ('gateway.config', 'gateway.db', 'gateway.db.client',
             'gateway.research_lab.maintenance', 'gateway.research_lab.store'):
    if name in sys.modules:
        raise SystemExit(name)
"""
    environment = {
        **os.environ,
        "SUPABASE_URL": "https://production.example.invalid",
        "SUPABASE_SERVICE_ROLE_KEY": "production-poison",
        "PYTHONPATH": "/production/import-poison",
    }
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_full_child_environments_ignore_parent_and_clone_process_poison(
    monkeypatch,
    tmp_path: Path,
):
    poison = {
        "PATH": "/production/path-poison",
        "PYTHONPATH": "/production/import-poison",
        "HTTP_PROXY": "http://production-proxy.invalid",
        "HTTPS_PROXY": "http://production-proxy.invalid",
        "GIT_SSH_COMMAND": "ssh -i /production/key",
        "AWS_ACCESS_KEY_ID": "production-access-key",
        "AWS_SECRET_ACCESS_KEY": "production-secret-key",
    }
    for key, value in poison.items():
        monkeypatch.setenv(key, value)

    child = full_host._clone_child_environment(region="us-east-1")
    assert child["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert child["AWS_REGION"] == "us-east-1"
    assert child["AWS_DEFAULT_REGION"] == "us-east-1"
    for key in poison:
        if key != "PATH":
            assert key not in child

    runtime = full_host._clone_runtime_environment(
        {
            **poison,
            "RESEARCH_LAB_PROVIDER_HTTP_PROXY": "provider-proxy-ref",
            "SUPABASE_URL": ORIGIN,
        },
        gateway_env_file=tmp_path / "gateway.env",
        region="us-east-1",
    )
    assert runtime["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert runtime["SUPABASE_URL"] == ORIGIN
    assert runtime["RESEARCH_LAB_PROVIDER_HTTP_PROXY"] == "provider-proxy-ref"
    for key in poison:
        if key != "PATH":
            assert key not in runtime


def test_miner_intake_subprocess_starts_before_clone_environment_is_applied(
    monkeypatch,
    tmp_path: Path,
):
    observed: dict[str, object] = {}
    poisoned_clone = {
        "PATH": "/clone/path-poison",
        "PYTHONPATH": "/clone/import-poison",
        "HTTP_PROXY": "http://clone-proxy.invalid",
        "HTTPS_PROXY": "http://clone-proxy.invalid",
        "GIT_SSH_COMMAND": "ssh -i /clone/key",
        "AWS_ACCESS_KEY_ID": "clone-access-key",
        "AWS_SECRET_ACCESS_KEY": "clone-secret-key",
        "RESEARCH_LAB_PROVIDER_HTTP_PROXY": "provider-proxy-ref",
    }

    monkeypatch.setattr(
        full_host,
        "_validated_clone_environment",
        lambda *_args, **_kwargs: poisoned_clone,
    )

    def fake_run(_command, **kwargs):
        observed["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "schema_version": (
                        "leadpoet.production_parity_miner_intake_evidence.v1"
                    ),
                    "candidate_sha": SHA,
                    "run_id": "pp-1-1",
                    "artifact_bucket": (
                        "leadpoet-parity-493765492819-" + "f" * 16
                    ),
                    "status": "passed",
                    "production_database_mutated": False,
                    "production_chain_mutated": False,
                    "chain_registration_boundary": "strict-ephemeral-hotkey",
                    "source_add": {
                        "admitted": True,
                        "global_miner_submissions_enabled": False,
                        "source_add_paused": False,
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(full_host, "_run", fake_run)
    full_host._run_miner_intake_path(
        region="us-east-1",
        candidate_sha=SHA,
        run_id="pp-1-1",
        supabase_origin=ORIGIN,
        gateway_env_file=tmp_path / "gateway.env",
        artifact_bucket="leadpoet-parity-493765492819-" + "f" * 16,
        miner_intake_secret=json.dumps(
            {"builtwith_api_key": "builtwith-credential"}
        ),
    )
    child_env = observed["env"]
    assert isinstance(child_env, dict)
    assert child_env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert child_env["AWS_REGION"] == "us-east-1"
    assert child_env["AWS_DEFAULT_REGION"] == "us-east-1"
    assert child_env["RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED"] == "false"
    assert child_env["RESEARCH_LAB_SOURCE_ADD_ENABLED"] == "true"
    for key in poisoned_clone:
        if key != "PATH":
            assert key not in child_env


def test_miner_intake_secret_resolution_is_strict_and_value_opaque():
    assert (
        _builtwith_key_from_secret(
            json.dumps({"builtwith_api_key": "provider-value-for-test"})
        )
        == "provider-value-for-test"
    )
    with pytest.raises(FullParityError, match="miner-intake secret is invalid"):
        _builtwith_key_from_secret(json.dumps({"builtwith_api_key": "bad value"}))
    assert (
        _required_secret_from_environment(
            {"FIRST": "", "SECOND": "credential-value"},
            ("FIRST", "SECOND"),
            field="credential",
        )
        == "credential-value"
    )
    with pytest.raises(FullParityError, match="credential is unavailable"):
        _required_secret_from_environment({}, ("FIRST",), field="credential")


@pytest.mark.asyncio
async def test_miner_intake_restores_controls_changed_for_source_only_state():
    observed: list[tuple[str, object]] = []

    async def call_rpc(name, payload):
        observed.append((name, dict(payload)))

    await full_host._restore_miner_intake_controls(
        {"source_add_paused": True},
        call_rpc=call_rpc,
    )

    assert observed == [
        (
            "research_lab_source_add_set_paused",
            {
                "p_paused": True,
                "p_reason": "production_parity_miner_intake_complete",
                "p_actor_ref": "system:production-parity",
            },
        )
    ]


@pytest.mark.asyncio
async def test_miner_intake_does_not_rewrite_already_active_source_add():
    observed: list[tuple[str, object]] = []

    async def call_rpc(name, payload):
        observed.append((name, dict(payload)))

    await full_host._restore_miner_intake_controls(
        {"source_add_paused": False},
        call_rpc=call_rpc,
    )

    assert observed == []


def test_full_clone_final_evidence_uses_run_scoped_gateway_token():
    jwt_secret = "j" * 48
    environment = build_gateway_environment(
        {},
        run_id="parity-20260815",
        candidate_sha=SHA,
        gateway_public_key=PARITY_GATEWAY_PUBLIC_KEY,
        supabase_origin=ORIGIN,
        artifact_bucket="leadpoet-parity-artifacts-example",
        benchmark_date="2026-08-16",
        jwt_secret=jwt_secret,
    )
    token = _clone_service_role_key(
        environment,
        candidate_sha=SHA,
        run_id="parity-20260815",
        supabase_origin=ORIGIN,
        jwt_secret=jwt_secret,
    )
    assert token == environment["SUPABASE_SERVICE_ROLE_KEY"]
    arena_token = _clone_arena_service_role_key(
        environment,
        candidate_sha=SHA,
        run_id="parity-20260815",
        supabase_origin=ORIGIN,
        jwt_secret=jwt_secret,
    )
    assert arena_token == environment["LAB_ARENA_SERVICE_JWT"]
    with pytest.raises(FullParityError, match="credential is unavailable"):
        _clone_service_role_key(
            {**environment, "SUPABASE_SERVICE_ROLE_KEY": ""},
            candidate_sha=SHA,
            run_id="parity-20260815",
            supabase_origin=ORIGIN,
            jwt_secret=jwt_secret,
        )
    with pytest.raises(FullParityError, match="run-scoped clone service role"):
        _clone_service_role_key(
            {**environment, "LEADPOET_PARITY_CANDIDATE_SHA": "b" * 40},
            candidate_sha=SHA,
            run_id="parity-20260815",
            supabase_origin=ORIGIN,
            jwt_secret=jwt_secret,
        )
    with pytest.raises(FullParityError, match="credential identity differs"):
        _clone_service_role_key(
            environment,
            candidate_sha=SHA,
            run_id="parity-20260815",
            supabase_origin=ORIGIN,
            jwt_secret="x" * 48,
        )
    with pytest.raises(FullParityError, match="credential identity differs"):
        _clone_arena_service_role_key(
            {
                **environment,
                "LAB_ARENA_SERVICE_JWT": environment[
                    "SUPABASE_SERVICE_ROLE_KEY"
                ],
            },
            candidate_sha=SHA,
            run_id="parity-20260815",
            supabase_origin=ORIGIN,
            jwt_secret=jwt_secret,
        )


def test_builtwith_live_probe_keeps_credential_out_of_url(monkeypatch):
    observed = {}
    payload = b'{"Results":[{"Lookup":"builtwith.com"}]}'

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return payload

    def fake_urlopen(request, *, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr("scripts.run_production_parity_full_host.urlopen", fake_urlopen)
    credential = "provider-credential-for-test"
    assert _verify_builtwith_credential_live(credential) == {
        "http_status": 200,
        "json_verified": True,
        "response_bytes": len(payload),
    }
    assert credential not in observed["url"]
    assert "LOOKUP=builtwith.com" in observed["url"]
    assert "NOPII=yes" in observed["url"]
    assert observed["authorization"] == f"API {credential}"
    assert observed["timeout"] == 45


def test_weight_readiness_epoch_parser_uses_real_reported_epoch():
    output = 'noise\n{"status":"ready","effective_epoch":24561}\n'
    assert _current_epoch_from_readiness(output) == 24561
    with pytest.raises(FullParityError, match="effective epoch"):
        _current_epoch_from_readiness("no json")


def test_runner_iam_policy_is_nonforwarding_and_write_scoped():
    controller = _controller_policy(
        account_id="493765492819",
        region="us-east-1",
        production_secret_id="leadpoet/prod/gateway/env",
        readonly_secret_id="leadpoet/staging/production-parity/readonly-dsn",
        miner_intake_secret_id=("leadpoet/staging/production-parity-miner-intake"),
        runner_arn="arn:aws:iam::493765492819:role/runner",
    )
    runner = _runner_policy(
        account_id="493765492819",
        region="us-east-1",
        production_secret_id="leadpoet/prod/gateway/env",
        readonly_secret_id="leadpoet/staging/production-parity/readonly-dsn",
        miner_intake_secret_id=("leadpoet/staging/production-parity-miner-intake"),
    )
    encoded = json.dumps({"controller": controller, "runner": runner})
    assert "iam:PassRole" in encoded
    assert "leadpoet-parity-493765492819-*" in encoded
    assert "leadpoet/staging/production-parity-miner-intake" in encoded
    assert "ssm:DescribeInstanceInformation" in encoded
    assert "secretsmanager:ListSecrets" in encoded
    assert "s3:GetBucketObjectLockConfiguration" in encoded
    assert "s3:PutBucketObjectLockConfiguration" in encoded
    assert "s3:GetObjectLockConfiguration" not in encoded
    assert "s3:PutObjectLockConfiguration" not in encoded
    assert '"Effect": "Deny"' in encoded
    assert "kms:ScheduleKeyDeletion" in encoded
    assert "testnet" not in encoded
    version_statements = [
        statement
        for statement in runner["Statement"]
        if statement.get("Action") == ["s3:GetObjectVersion"]
    ]
    assert version_statements == [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObjectVersion"],
            "Resource": "arn:aws:s3:::leadpoet-attested-v2-artifacts-*/*",
        }
    ]
    retention_statements = [
        statement
        for statement in runner["Statement"]
        if statement.get("Action") == ["s3:GetObjectRetention"]
    ]
    assert retention_statements == [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObjectRetention"],
            "Resource": [
                "arn:aws:s3:::leadpoet-parity-493765492819-*/*",
                "arn:aws:s3:::leadpoet-attested-v2-artifacts-*/*",
            ],
        }
    ]
    get_object = next(
        statement
        for statement in runner["Statement"]
        if statement.get("Action") == ["s3:GetObject"]
    )
    assert "s3:GetObjectVersion" not in get_object["Action"]
    write_statement = next(
        statement
        for statement in runner["Statement"]
        if "secretsmanager:PutSecretValue"
        in (
            statement["Action"]
            if isinstance(statement["Action"], list)
            else [statement["Action"]]
        )
    )
    assert write_statement["Resource"].endswith(
        "production-parity/runs/pp-*/gateway-??????"
    )
    assert "production-parity-miner-intake" not in write_statement["Resource"]


def test_agent_guides_are_identical_and_define_both_parity_lanes():
    agents = (ROOT / "AGENTS.md").read_bytes()
    assert agents == (ROOT / "CLAUDE.md").read_bytes()
    source = agents.decode("utf-8")
    assert "asynchronous post-push diagnostic" in source
    assert "candidate product/trust failure" in source
    assert "Production Parity Full" in source
    assert "strict non-forwarding chain boundary" in source
    assert "No permanent staging fleet" in source
