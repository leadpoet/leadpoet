from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from gateway.research_lab import daily_baseline_readiness
from gateway.research_lab import scoring_worker

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
from scripts import run_production_parity_full_host as full_host
from scripts.production_parity_snapshot import (
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
        inspect.signature(capture_snapshot)
        .parameters["timeout_seconds"]
        .default
        == 900
    )
    assert (
        inspect.signature(restore_snapshot)
        .parameters["timeout_seconds"]
        .default
        == 900
    )
    assert (
        _snapshot_io_timeout_seconds(MAX_SNAPSHOT_IO_TIMEOUT_SECONDS)
        == MAX_SNAPSHOT_IO_TIMEOUT_SECONDS
    )
    for invalid in (True, 0, MAX_SNAPSHOT_IO_TIMEOUT_SECONDS + 1):
        with pytest.raises(ProductionParityError, match="timeout is invalid"):
            _snapshot_io_timeout_seconds(invalid)


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

    monkeypatch.setattr(full_host, "EARLY_BOOT_MARKER", marker)
    monkeypatch.setattr(full_host, "FULL_WORK_ROOT", work_root)
    monkeypatch.setattr(full_host, "_checkout_identity", lambda _sha: None)
    monkeypatch.setattr(full_host.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(full_host.boto3, "client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(full_host, "_DockerDatabase", lambda **_kwargs: Database())
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

    assert observed == {"capture": 990, "restore": 850}
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
    target_date = (
        datetime.now(timezone.utc).date() + timedelta(days=1)
    ).isoformat()
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
    assert _builtwith_key_from_secret(
        json.dumps({"builtwith_api_key": "provider-value-for-test"})
    ) == "provider-value-for-test"
    with pytest.raises(FullParityError, match="miner-intake secret is invalid"):
        _builtwith_key_from_secret(json.dumps({"builtwith_api_key": "bad value"}))
    assert _required_secret_from_environment(
        {"FIRST": "", "SECOND": "credential-value"},
        ("FIRST", "SECOND"),
        field="credential",
    ) == "credential-value"
    with pytest.raises(FullParityError, match="credential is unavailable"):
        _required_secret_from_environment({}, ("FIRST",), field="credential")


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

    monkeypatch.setattr(
        "scripts.run_production_parity_full_host.urlopen", fake_urlopen
    )
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
        miner_intake_secret_id=(
            "leadpoet/staging/production-parity-miner-intake"
        ),
        runner_arn="arn:aws:iam::493765492819:role/runner",
    )
    runner = _runner_policy(
        account_id="493765492819",
        region="us-east-1",
        production_secret_id="leadpoet/prod/gateway/env",
        readonly_secret_id="leadpoet/staging/production-parity/readonly-dsn",
        miner_intake_secret_id=(
            "leadpoet/staging/production-parity-miner-intake"
        ),
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
