from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from gateway.research_lab import daily_baseline_readiness
from gateway.research_lab import scoring_worker
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
)
from scripts import production_parity_snapshot as parity_snapshot
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
    _builtwith_key_from_secret,
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

    def fake_run(command, **_kwargs):
        command = list(command)
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


def test_full_runner_forwards_remaining_clone_budget_and_cleans_runtime(
    monkeypatch,
    tmp_path: Path,
):
    marker = tmp_path / "early-boot-isolated"
    marker.write_text("isolated\n", encoding="utf-8")
    work_root = tmp_path / "encrypted-root-volume"
    output = tmp_path / "evidence" / "full.json"
    observed: dict[str, int] = {}
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
            return {"status": "removed"}

    def fake_capture_snapshot(**kwargs):
        observed["capture"] = kwargs["timeout_seconds"]
        return {
            "capture_mode": "full",
            "database": {"target_rebenchmark_date": "2026-08-19"},
        }

    def fake_restore_snapshot(**kwargs):
        observed["restore"] = kwargs["timeout_seconds"]
        raise FullParityError("stop after restore")

    def database_factory(**kwargs):
        observed["postgres_publish"] = kwargs["postgres_publish"]
        observed["postgrest_publish"] = kwargs["postgrest_publish"]
        return Database()

    monkeypatch.setattr(full_host, "EARLY_BOOT_MARKER", marker)
    monkeypatch.setattr(full_host, "FULL_WORK_ROOT", work_root)
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
            supabase_origin="https://parity.example",
            artifact_bucket="parity-artifacts",
            postgres_image="postgres@sha256:" + "c" * 64,
            postgrest_image="postgrest@sha256:" + "d" * 64,
            output=output,
            timeout_seconds=1_000,
        )

    assert observed == {
        "postgres_publish": "127.0.0.1::5432",
        "postgrest_publish": "0.0.0.0:3000:3000",
        "capture": 990,
        "restore": 850,
    }
    assert not (work_root / "pp-test-1" / "runtime").exists()
    cleanup = json.loads(output.read_text(encoding="utf-8"))["cleanup"]
    assert cleanup["work"] == "removed"


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
        contract_functions = {
            "research_lab_compact_weight_settlement_contract_v1": compact_contract,
            "research_lab_candidate_hybrid_purpose_contract_v1": purpose_contract,
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
        assert result["data_probe_count"] == 3
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
    assert "permission" not in encoded
    assert "unknown operation" not in encoded
    assert len(encoded) < 4096


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
        + fast_parity.FAST_PRODUCTION_DATA_READ_HEADROOM_SECONDS
        + fast_parity.FAST_JOB_SETUP_HEADROOM_SECONDS
    )
    assert fast_parity.FAST_SCHEMA_PREFLIGHT_NETWORK_PROBE_COUNT == (
        len(schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA) + 3
    )
    assert fast_parity.FAST_CANDIDATE_MIGRATION_HEADROOM_COUNT == 2
    assert fast_parity._fast_job_minimum_timeout_seconds(2) == expected_minimum
    assert fast_parity.FAST_JOB_MINIMUM_TIMEOUT_SECONDS == expected_minimum
    assert outer_seconds == fast_parity.FAST_JOB_OUTER_TIMEOUT_SECONDS
    assert outer_seconds - expected_minimum >= 10 * 60
    assert fast_parity._fast_job_minimum_timeout_seconds(5) >= outer_seconds
    assert role_duration_seconds == fast_parity.FAST_AWS_ROLE_DURATION_SECONDS
    assert role_duration_seconds - outer_seconds >= 20 * 60


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


@pytest.mark.asyncio
async def test_parity_date_controls_baseline_rollover_and_readiness(monkeypatch):
    target_date = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    parity_environment = {
        "LEADPOET_PRODUCTION_PARITY_MODE": "enabled",
        "LEADPOET_PRODUCTION_PARITY_RUN_ID": "parity-date-check",
        "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN": (
            "https://d111111abcdef8.cloudfront.net"
        ),
        "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE": target_date,
    }
    for name, value in parity_environment.items():
        monkeypatch.setenv(name, value)

    async def maintenance_state():
        return {"paused": False, "reason": ""}

    async def unavailable_active_model(*_args, **_kwargs):
        raise RuntimeError("bounded readiness probe")

    monkeypatch.setattr(
        scoring_worker,
        "get_scoring_maintenance_state",
        maintenance_state,
    )
    monkeypatch.setattr(
        daily_baseline_readiness,
        "load_active_private_model",
        unavailable_active_model,
    )
    await scoring_worker._enforce_baseline_wave_maintenance_boundary(
        completed_icps=12,
        total_icps=40,
        benchmark_date=target_date,
    )
    readiness = await daily_baseline_readiness.autoresearch_daily_baseline_readiness(
        SimpleNamespace(private_baseline_rebenchmark_enabled=True)
    )
    assert readiness == {
        "available": False,
        "reason": "daily_baseline_gate_unavailable",
        "benchmark_date": target_date,
    }


def test_critical_stage_ledger_fails_closed_and_hashes_evidence():
    ledger = StageLedger(
        lane="full",
        candidate_sha=SHA,
        contract_hash=HASH,
        snapshot_hash=HASH,
        critical_stage_ids=("rebenchmark", "weights"),
    )
    ledger.record(
        "rebenchmark",
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
    environment = build_gateway_environment(
        {
            "OPENROUTER_API_KEY": "runtime-key",
            "EXA_API_KEY": "exa-key",
            "SCRAPINGDOG_API_KEY": "dog-key",
            "OPENROUTER_MANAGEMENT_KEY": "must-drop",
            "WALLET_PRIVATE_KEY": "must-drop",
            "SUPABASE_URL": "https://qplwoislplkcegvdmbim.supabase.co",
            "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "true",
        },
        run_id="parity-20260815",
        candidate_sha=SHA,
        supabase_origin=ORIGIN,
        artifact_bucket="leadpoet-parity-artifacts-example",
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
        == "leadpoet-parity-artifacts-example"
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


def test_controller_dependency_closure_includes_gateway_database_client():
    selected = resolve_controller_requirements(ROOT / "requirements.txt")
    names = {re.split(r"[<>=!~ ]", item, maxsplit=1)[0].lower() for item in selected}
    assert names == {"boto3", "cryptography", "httpx", "supabase"}


def test_full_runner_accepts_json_secret_and_strict_env_file(tmp_path: Path):
    dsn = "postgresql://reader:password@db.example.com:5432/postgres"
    assert _dsn_from_secret(json.dumps({"readonly_dsn": dsn})) == dsn
    env_file = tmp_path / "gateway.env"
    env_file.write_text(
        "export SIMPLE=value\nQUOTED='a value'\nEMPTY=\n",
        encoding="utf-8",
    )
    assert _parse_gateway_environment_file(env_file) == {
        "SIMPLE": "value",
        "QUOTED": "a value",
        "EMPTY": "",
    }
    env_file.write_text("BAD=a b\n", encoding="utf-8")
    with pytest.raises(FullParityError, match="multi-token"):
        _parse_gateway_environment_file(env_file)


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


def test_full_clone_final_evidence_uses_run_scoped_gateway_token():
    jwt_secret = "j" * 48
    environment = build_gateway_environment(
        {},
        run_id="parity-20260815",
        candidate_sha=SHA,
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


def test_agent_guides_are_identical_and_require_both_parity_lanes():
    agents = (ROOT / "AGENTS.md").read_bytes()
    assert agents == (ROOT / "CLAUDE.md").read_bytes()
    source = agents.decode("utf-8")
    assert "mandatory 5-10-minute post-push" in source
    assert "Production Parity Full" in source
    assert "strict non-forwarding chain boundary" in source
    assert "No permanent staging fleet" in source
