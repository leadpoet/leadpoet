from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
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
from scripts.run_production_parity_full_host import (
    FullParityError,
    _current_epoch_from_readiness,
    _dsn_from_secret,
    _parse_gateway_environment_file,
)
from scripts.run_production_parity_fast import (
    _ProductionReadOnlySupabaseProvider,
)
from scripts.setup_production_parity_staging import (
    _controller_policy,
    _runner_policy,
)
from scripts.resolve_production_parity_controller_requirements import (
    resolve_controller_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
HASH = "sha256:" + "b" * 64
ORIGIN = "https://d111111abcdef8.cloudfront.net"


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
        runner_arn="arn:aws:iam::493765492819:role/runner",
    )
    runner = _runner_policy(
        account_id="493765492819",
        region="us-east-1",
        production_secret_id="leadpoet/prod/gateway/env",
        readonly_secret_id="leadpoet/staging/production-parity/readonly-dsn",
    )
    encoded = json.dumps({"controller": controller, "runner": runner})
    assert "iam:PassRole" in encoded
    assert "leadpoet-parity-493765492819-*" in encoded
    assert "ssm:DescribeInstanceInformation" in encoded
    assert "secretsmanager:ListSecrets" in encoded
    assert "s3:PutObjectLockConfiguration" in encoded
    assert '"Effect": "Deny"' in encoded
    assert "kms:ScheduleKeyDeletion" in encoded
    assert "testnet" not in encoded


def test_agent_guides_are_identical_and_require_both_parity_lanes():
    agents = (ROOT / "AGENTS.md").read_bytes()
    assert agents == (ROOT / "CLAUDE.md").read_bytes()
    source = agents.decode("utf-8")
    assert "mandatory 5-10-minute post-push" in source
    assert "Production Parity Full" in source
    assert "strict non-forwarding chain boundary" in source
    assert "No permanent staging fleet" in source
