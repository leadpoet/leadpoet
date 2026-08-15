from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

from botocore.exceptions import ClientError
import pytest

from leadpoet_canonical.production_parity import (
    LEDGER_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    ProductionParityError,
    StageLedger,
    migration_delta,
    production_database_host_hash,
    safe_database_target,
    sha256_json,
    validate_ledger,
    validate_snapshot_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _migration(sequence: int, name: str, digest: str) -> dict:
    return {
        "path": f"scripts/{sequence}-{name}.sql",
        "sequence": sequence,
        "sha256": "sha256:" + digest * 64,
        "transaction_mode": "candidate-file",
    }


def _snapshot() -> dict:
    now = datetime.now(timezone.utc)
    body = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_environment": "production-read-only",
        "source_host_hash": production_database_host_hash(
            "db.production.example"
        ),
        "capture_sha": "a" * 40,
        "capture_contract_hash": "sha256:" + "b" * 64,
        "source_sha": "9" * 40,
        "captured_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "capture_transaction_read_only": True,
        "archive": {
            "format": "postgres-custom",
            "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/example",
            "s3_uri": "s3://leadpoet-parity/snapshots/example.dump",
            "sha256": "sha256:" + "c" * 64,
            "size_bytes": 1024,
        },
        "database": {
            "server_version_num": "160001",
            "relation_count": 42,
            "total_relation_bytes": 40 * 1024 * 1024,
            "largest_relation_bytes": 12 * 1024 * 1024,
            "capture_utc_date": now.date().isoformat(),
            "target_rebenchmark_date": now.date().isoformat(),
            "latest_completed_benchmark_date": (
                now.date() - timedelta(days=1)
            ).isoformat(),
            "current_day_rebenchmark_run_count": 0,
            "current_day_benchmark_bundle_count": 0,
        },
        "migrations": [_migration(1, "base", "d")],
        "data_classification": "production-confidential-kms-encrypted",
    }
    return {**body, "manifest_hash": sha256_json(body)}


def test_snapshot_manifest_binds_production_scale_and_daily_frontier() -> None:
    snapshot = validate_snapshot_manifest(_snapshot())

    assert snapshot["database"]["total_relation_bytes"] == 40 * 1024 * 1024
    assert snapshot["database"]["current_day_rebenchmark_run_count"] == 0

    tampered = _snapshot()
    tampered["database"]["current_day_rebenchmark_run_count"] = 1
    with pytest.raises(ProductionParityError, match="hash differs"):
        validate_snapshot_manifest(tampered)


def test_snapshot_manifest_binds_the_configured_production_database_host() -> None:
    snapshot = validate_snapshot_manifest(_snapshot())

    assert snapshot["source_host_hash"] == production_database_host_hash(
        "DB.PRODUCTION.EXAMPLE."
    )
    assert snapshot["source_host_hash"] != production_database_host_hash(
        "other.production.example"
    )
    with pytest.raises(ProductionParityError, match="host is invalid"):
        production_database_host_hash("https://db.production.example")


def test_rebenchmark_readiness_evidence_is_complete_and_redacted() -> None:
    module = _load_script("check_production_parity_rebenchmark")
    candidate = "a" * 40
    value = {
        "available": True,
        "reason": "daily_baseline_published",
        "benchmark_date": "2026-08-15",
        "report_id": "report-1",
        "benchmark_bundle_id": "bundle-1",
        "rolling_window_hash": "sha256:" + "1" * 64,
        "completion_commitments": {
            "all_icp_count": 40,
            "per_icp_summaries_hash": "sha256:" + "2" * 64,
            "category_assignment_hash": "sha256:" + "3" * 64,
            "conditional_policy_hash": "sha256:" + "4" * 64,
            "category_counts": {
                "public": 10,
                "private": 10,
                "conditional": 20,
            },
            "category_strength_counts": {
                "public": {"weak": 7, "strong": 3},
                "private": {"weak": 3, "strong": 7},
                "conditional": {"center": 20},
            },
            "minimum_icp_score": 0.0,
            "maximum_icp_score": 87.5,
        },
    }

    result = module._sanitize(value, candidate_sha=candidate)

    assert result["available"] is True
    assert result["completion_commitments"]["all_icp_count"] == 40
    assert set(result["completion_commitments"]) == {
        "all_icp_count",
        "per_icp_summaries_hash",
        "category_assignment_hash",
        "conditional_policy_hash",
        "category_counts",
        "category_strength_counts",
        "minimum_icp_score",
        "maximum_icp_score",
    }
    assert module._sanitize(
        {**value, "completion_commitments": {}},
        candidate_sha=candidate,
    )["available"] is False


def test_candidate_migrations_cannot_rewrite_production_history() -> None:
    applied = [_migration(1, "base", "a")]
    candidate = [*applied, _migration(2, "next", "b")]

    assert migration_delta(
        snapshot_migrations=applied, candidate_migrations=candidate
    ) == [_migration(2, "next", "b")]

    rewritten = [_migration(1, "base", "c"), _migration(2, "next", "b")]
    with pytest.raises(ProductionParityError, match="rewrote applied migration"):
        migration_delta(
            snapshot_migrations=applied, candidate_migrations=rewritten
        )


def test_ledger_hash_and_independent_stage_outcomes_are_fail_closed() -> None:
    ledger = StageLedger(
        lane="full",
        candidate_sha="a" * 40,
        contract_hash="sha256:" + "b" * 64,
        snapshot_hash="sha256:" + "c" * 64,
        critical_stage_ids=("rebenchmark", "weights"),
    )
    ledger.record(
        "rebenchmark", status="failed", duration_seconds=1.5, reason="timeout"
    )
    ledger.record(
        "weights", status="passed", duration_seconds=2.5, evidence={"epochs": 3}
    )
    document = ledger.finalize()

    assert document["schema_version"] == LEDGER_SCHEMA_VERSION
    assert validate_ledger(document)["status"] == "failed"
    document["stages"][1]["evidence"]["epochs"] = 2
    with pytest.raises(ProductionParityError, match="evidence differs"):
        validate_ledger(document)


def test_snapshot_restore_target_can_never_be_production() -> None:
    safe_database_target(
        "postgresql://postgres:test@127.0.0.1:5432/leadpoet_parity_candidate",
        production_host="db.production.example",
    )
    with pytest.raises(ProductionParityError, match="not an isolated"):
        safe_database_target(
            "postgresql://postgres:test@db.production.example/leadpoet",
            production_host="db.production.example",
        )


class _FakeS3:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], bytes] = {}
        self.put_requests: list[dict] = []
        self.heads: dict[tuple[str, str], dict] = {}

    def get_object(self, *, Bucket: str, Key: str):
        value = self.values.get((Bucket, Key))
        if value is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": io.BytesIO(value)}

    def put_object(self, **request):
        key = (request["Bucket"], request["Key"])
        if key in self.values:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "exists"}},
                "PutObject",
            )
        body = request["Body"]
        value = body if isinstance(body, bytes) else body.read()
        self.put_requests.append(request)
        self.values[key] = value
        if "ChecksumSHA256" in request:
            self.heads[key] = {
                "ChecksumSHA256": request["ChecksumSHA256"],
                "ContentLength": request["ContentLength"],
                "Metadata": request["Metadata"],
                "ServerSideEncryption": request["ServerSideEncryption"],
                "SSEKMSKeyId": request["SSEKMSKeyId"],
                "VersionId": "version-1",
                "ObjectLockMode": request.get("ObjectLockMode"),
                "ObjectLockRetainUntilDate": request.get(
                    "ObjectLockRetainUntilDate"
                ),
            }
        return {"VersionId": "version-1"}

    def head_object(
        self,
        *,
        Bucket: str,
        Key: str,
        ChecksumMode: str,
        VersionId: str = "",
    ):
        assert ChecksumMode == "ENABLED"
        assert VersionId in {"", "version-1"}
        value = self.heads.get((Bucket, Key))
        if value is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "HeadObject",
            )
        return value


def test_parity_evidence_is_immutable_and_kms_bound() -> None:
    module = _load_script("publish_production_parity_evidence")
    client = _FakeS3()
    kwargs = {
        "client": client,
        "bucket": "leadpoet-parity-evidence",
        "key": "production-parity/contracts/" + "a" * 40 + "/ledger.json",
        "payload": b'{"status":"passed"}\n',
        "kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/example",
        "object_lock_days": 365,
    }

    assert module.publish_exact(**kwargs)["created"] is True
    assert module.publish_exact(**kwargs)["created"] is False
    assert client.put_requests[0]["IfNoneMatch"] == "*"
    assert client.put_requests[0]["ObjectLockMode"] == "COMPLIANCE"
    with pytest.raises(module.EvidencePublicationError, match="different bytes"):
        module.publish_exact(**{**kwargs, "payload": b'{"status":"failed"}\n'})


def test_large_snapshot_publication_is_create_only_and_checksum_bound(
    tmp_path: Path,
) -> None:
    module = _load_script("publish_production_parity_evidence")
    client = _FakeS3()
    archive = tmp_path / "snapshot.dump"
    archive.write_bytes(b"production-shaped-snapshot" * 1024)
    kwargs = {
        "client": client,
        "bucket": "leadpoet-parity-evidence",
        "key": "production-parity/snapshots/run/production.dump",
        "path": archive,
        "kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/example",
        "content_type": "application/octet-stream",
        "max_bytes": 5 * 1024 * 1024 * 1024,
        "object_lock_days": 7,
    }

    created = module.publish_file_exact(**kwargs)
    repeated = module.publish_file_exact(**kwargs)

    assert created["created"] is True
    assert repeated["created"] is False
    assert created["version_id"] == "version-1"
    assert repeated["version_id"] == "version-1"
    assert created["size_bytes"] == archive.stat().st_size
    assert client.put_requests[0]["IfNoneMatch"] == "*"
    assert client.put_requests[0]["ChecksumAlgorithm"] == "SHA256"
    assert client.put_requests[0]["ObjectLockMode"] == "COMPLIANCE"
    archive.write_bytes(b"different")
    with pytest.raises(module.EvidencePublicationError, match="different bytes"):
        module.publish_file_exact(**kwargs)


def test_snapshot_target_may_be_the_next_utc_day() -> None:
    now = datetime.now(timezone.utc)
    snapshot = _snapshot()
    snapshot.pop("manifest_hash")
    snapshot["database"]["capture_utc_date"] = (
        now.date() - timedelta(days=1)
    ).isoformat()
    snapshot["database"]["target_rebenchmark_date"] = now.date().isoformat()
    snapshot["database"]["latest_completed_benchmark_date"] = (
        now.date() - timedelta(days=1)
    ).isoformat()
    snapshot["manifest_hash"] = sha256_json(snapshot)

    assert (
        validate_snapshot_manifest(snapshot)["database"]["target_rebenchmark_date"]
        == now.date().isoformat()
    )


def test_snapshot_frontier_rejects_contaminated_current_day() -> None:
    module = _load_script("production_parity_snapshot")
    clean = {
        "capture_utc_timestamp": "2026-08-15T00:05:00+00:00",
        "capture_utc_date": "2026-08-15",
        "latest_completed_benchmark_date": "2026-08-14",
        "current_day_rebenchmark_run_count": 0,
        "current_day_benchmark_bundle_count": 0,
    }
    assert module._target_rebenchmark_date(clean).isoformat() == "2026-08-15"
    with pytest.raises(ProductionParityError, match="clean target-day"):
        module._target_rebenchmark_date(
            {**clean, "current_day_rebenchmark_run_count": 1}
        )

    before_rollover = {
        **clean,
        "capture_utc_timestamp": "2026-08-15T23:45:00+00:00",
    }
    assert module._target_rebenchmark_date(before_rollover).isoformat() == "2026-08-16"


def test_snapshot_migration_frontier_comes_from_deployed_source() -> None:
    module = _load_script("production_parity_snapshot")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    migrations = module._source_migrations(
        root=ROOT,
        source_sha=source,
        candidate_sha=source,
    )

    assert migrations
    assert migrations == sorted(
        migrations, key=lambda item: (item["sequence"], item["path"])
    )
    assert all(item["path"].startswith("scripts/") for item in migrations)
    assert all(item["sha256"].startswith("sha256:") for item in migrations)


def test_secret_materializer_rejects_unclassified_production_writes() -> None:
    module = _load_script("materialize_production_parity_secrets")

    with pytest.raises(
        module.SecretMaterializationError,
        match="unclassified production boundaries",
    ):
        module._build_runtime(
            source={"OPENROUTER_API_KEY": "provider", "REDIS_URL": "prod"},
            overlay={},
            generated={"SUPABASE_URL": "https://staging.example"},
            allowed_production_boundaries=set(),
            required_overlay_keys=set(),
            field="gateway",
        )


def test_secret_materializer_requires_staging_provider_control_credentials() -> None:
    module = _load_script("materialize_production_parity_secrets")
    source = {
        "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY": "production-control",
    }

    with pytest.raises(
        module.SecretMaterializationError,
        match="unclassified production boundaries",
    ):
        module._build_runtime(
            source=source,
            overlay={},
            generated={},
            allowed_production_boundaries=set(source),
            required_overlay_keys=set(),
            field="gateway",
        )

    runtime = module._build_runtime(
        source=source,
        overlay={
            "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY": "staging-control",
        },
        generated={},
        allowed_production_boundaries=set(),
        required_overlay_keys={"RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY"},
        field="gateway",
    )
    assert runtime["RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY"] == "staging-control"


def test_secret_materializer_replaces_control_internal_and_telemetry_boundaries() -> None:
    module = _load_script("materialize_production_parity_secrets")
    source = {
        "GITHUB_TOKEN": "production-github",
        "LEADPOET_INTERNAL_SECRET": "production-internal",
        "RESEARCH_LAB_INTERNAL_API_KEY": "production-research-lab",
        "GATEWAY_OTEL_ENABLED": "1",
        "GATEWAY_OTEL_ENDPOINT": "https://production-otel.invalid",
        "GATEWAY_OTEL_METRICS_ENDPOINT": "https://production-metrics.invalid",
        "GATEWAY_OTEL_TOKEN": "production-otel-token",
    }
    runtime = module._build_runtime(
        source=source,
        overlay={"GITHUB_TOKEN": "staging-read-only-github"},
        generated={
            "LEADPOET_INTERNAL_SECRET": "run-scoped-internal",
            "RESEARCH_LAB_INTERNAL_API_KEY": "run-scoped-research-lab",
            "GATEWAY_OTEL_ENABLED": "0",
            "GATEWAY_OTEL_ENDPOINT": "",
            "GATEWAY_OTEL_METRICS_ENDPOINT": "",
            "GATEWAY_OTEL_TOKEN": "",
        },
        allowed_production_boundaries=set(source),
        required_overlay_keys={"GITHUB_TOKEN"},
        field="gateway",
    )

    assert runtime["GITHUB_TOKEN"] == "staging-read-only-github"
    assert runtime["LEADPOET_INTERNAL_SECRET"] == "run-scoped-internal"
    assert runtime["RESEARCH_LAB_INTERNAL_API_KEY"] == "run-scoped-research-lab"
    assert runtime["GATEWAY_OTEL_ENABLED"] == "0"
    assert runtime["GATEWAY_OTEL_ENDPOINT"] == ""
    assert runtime["GATEWAY_OTEL_METRICS_ENDPOINT"] == ""
    assert runtime["GATEWAY_OTEL_TOKEN"] == ""
    assert not set(source.values()).intersection(runtime.values())


def test_runtime_config_capture_keeps_policy_and_drops_secrets() -> None:
    module = _load_script("capture_production_parity_runtime_config")
    document = module.canonical_runtime_config(
        {
            "RESEARCH_LAB_PUBLIC_ICP_TOTAL": "10",
            "RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE": "enforce",
            "RESEARCH_LAB_OPENROUTER_API_KEY": "must-not-leak",
            "LEADPOET_SENTRY_DSN": "https://must-not-leak.invalid",
            "SUPABASE_SERVICE_ROLE_KEY": "must-not-leak",
        }
    )

    execution = document["execution_config"]
    assert execution["schema_version"] == "leadpoet.research_lab_execution_config.v7"
    assert (
        execution["behavior_environment"][
            "RESEARCH_LAB_CONDITIONAL_VALIDATION_MODE"
        ]
        == "enforce"
    )
    assert execution["fields"]["public_benchmark_public_total_icps"] == 10
    assert "must-not-leak" not in json.dumps(document, sort_keys=True)


class _FakeSecrets:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.created: dict[str, dict[str, str]] = {}

    def get_secret_value(self, *, SecretId: str):
        return {"SecretString": self.values[SecretId]}

    def create_secret(self, **request):
        self.created[request["Name"]] = json.loads(request["SecretString"])

    def delete_secret(self, **request):
        self.created.pop(request["SecretId"], None)


def test_secret_materializer_creates_six_isolated_testnet_roles() -> None:
    module = _load_script("materialize_production_parity_secrets")
    gateway_source = json.dumps(
        {
            "OPENROUTER_API_KEY": "provider-value",
            "SUPABASE_URL": "https://production.invalid",
            "BITTENSOR_NETWORK": "finney",
            "GITHUB_TOKEN": "production-github",
            "LEADPOET_INTERNAL_SECRET": "production-internal",
            "RESEARCH_LAB_INTERNAL_API_KEY": "production-research-lab",
            "GATEWAY_OTEL_ENABLED": "1",
            "GATEWAY_OTEL_ENDPOINT": "https://production-otel.invalid",
            "GATEWAY_OTEL_METRICS_ENDPOINT": "https://production-metrics.invalid",
            "GATEWAY_OTEL_TOKEN": "production-otel-token",
            "LEADPOET_SENTRY_API_TOKEN": "must-not-enter-staging",
            "LEADPOET_SENTRY_DSN": "https://must-not-enter-staging.invalid/1",
            "LEADPOET_SENTRY_ENABLED": "1",
        }
    )
    validator_source = json.dumps(
        {
            "SUPABASE_URL": "https://production.invalid",
            "BITTENSOR_NETWORK": "finney",
        }
    )
    auditor = {
        "BT_WALLET_NAME": "parity",
        "BT_WALLET_HOTKEY": "auditor",
        "BT_WALLET_PATH": "/run/parity-wallet",
    }
    overlay = json.dumps(
        {
            "gateway": {"GITHUB_TOKEN": "staging-read-only-github"},
            "validator": {},
            "dashboard": {},
            "auditors": [auditor, {**auditor, "BT_WALLET_HOTKEY": "auditor-b"}],
        }
    )
    client = _FakeSecrets(
        {"prod-gateway": gateway_source, "prod-validator": validator_source, "overlay": overlay}
    )
    config = {
        "region": "us-east-1",
        "runtime_metadata": {
            "network": "test",
            "netuid": 1,
            "testnet_chain_endpoint": "wss://test.finney.opentensor.ai:443",
        },
        "wallet_artifacts": {
            "primary_validator": {
                "schema_version": "leadpoet.production_parity_wallet_spec.v1",
                "s3_uri": "s3://leadpoet-parity/wallets/primary.tar",
                "version_id": "primary-v1",
                "sha256": "sha256:" + "1" * 64,
                "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/example",
                "wallet_name": "primary",
                "wallet_hotkey": "default",
                "expected_hotkey": "5" * 48,
            },
            "audit_validators": [
                {
                    "schema_version": "leadpoet.production_parity_wallet_spec.v1",
                    "s3_uri": "s3://leadpoet-parity/wallets/auditor-a.tar",
                    "version_id": "auditor-a-v1",
                    "sha256": "sha256:" + "2" * 64,
                    "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/example",
                    "wallet_name": "auditor-a",
                    "wallet_hotkey": "default",
                    "expected_hotkey": "6" * 48,
                },
                {
                    "schema_version": "leadpoet.production_parity_wallet_spec.v1",
                    "s3_uri": "s3://leadpoet-parity/wallets/auditor-b.tar",
                    "version_id": "auditor-b-v1",
                    "sha256": "sha256:" + "3" * 64,
                    "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/example",
                    "wallet_name": "auditor-b",
                    "wallet_hotkey": "default",
                    "expected_hotkey": "7" * 48,
                },
            ],
        },
        "epoch_authority_artifact": {
            "schema_version": "leadpoet.production_parity_epoch_authority_spec.v1",
            "network": "test",
            "netuid": 1,
            "s3_uri": "s3://leadpoet-parity/authority/testnet.tar",
            "version_id": "authority-v1",
            "sha256": "sha256:" + "4" * 64,
            "kms_key_arn": "arn:aws:kms:us-east-1:123456789012:key/example",
            "mapping_hash": "sha256:" + "8" * 64,
            "network_genesis_hash": "0x" + "9" * 64,
        },
        "secret_materialization": {
            "production_gateway_secret_id": "prod-gateway",
            "production_validator_secret_id": "prod-validator",
            "staging_overlay_secret_id": "overlay",
            "staging_kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/example",
            "allowed_production_boundary_keys": [],
            "required_gateway_overlay_keys": ["GITHUB_TOKEN"],
            "required_validator_overlay_keys": [],
            "required_auditor_overlay_keys": [
                "BT_WALLET_NAME",
                "BT_WALLET_HOTKEY",
                "BT_WALLET_PATH",
            ],
        },
    }
    state = {
        "run_id": "run-123456",
        "candidate_sha": "a" * 40,
        "outputs": {
            "GatewayDomain": "gateway-run.example.test",
            "DatabaseDomain": "database-run-123456.example.test",
            "DashboardDomain": "dashboard-run.example.test",
        },
    }

    result = module.materialize(
        config=config,
        state=state,
        candidate_sha="a" * 40,
        client=client,
    )

    assert len(client.created) == 6
    assert set(result["secret_ids"]) == {
        "gateway",
        "validator",
        "database",
        "auditor-a",
        "auditor-b",
        "dashboard",
    }
    gateway = client.created[result["secret_ids"]["gateway"]]
    auditor_a = client.created[result["secret_ids"]["auditor-a"]]
    assert gateway["SUPABASE_URL"] == "https://database-run-123456.example.test"
    assert gateway["LEADPOET_PRODUCTION_PARITY_MODE"] == "enabled"
    assert gateway["LEADPOET_PRODUCTION_PARITY_RUN_ID"] == "run-123456"
    assert gateway["LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN"] == (
        "https://database-run-123456.example.test"
    )
    assert gateway["BITTENSOR_NETWORK"] == "test"
    assert gateway["LEADPOET_SENTRY_API_TOKEN"] == ""
    assert gateway["LEADPOET_SENTRY_DSN"] == ""
    assert gateway["LEADPOET_SENTRY_ENABLED"] == "0"
    assert gateway["GITHUB_TOKEN"] == "staging-read-only-github"
    assert gateway["LEADPOET_INTERNAL_SECRET"]
    assert gateway["LEADPOET_INTERNAL_SECRET"] != "production-internal"
    assert gateway["RESEARCH_LAB_INTERNAL_API_KEY"]
    assert gateway["RESEARCH_LAB_INTERNAL_API_KEY"] != "production-research-lab"
    assert gateway["GATEWAY_OTEL_ENABLED"] == "0"
    assert gateway["GATEWAY_OTEL_ENDPOINT"] == ""
    assert gateway["GATEWAY_OTEL_METRICS_ENDPOINT"] == ""
    assert gateway["GATEWAY_OTEL_TOKEN"] == ""
    assert auditor_a["AUDITOR_WEIGHT_PROTOCOL"] == "authoritative_v2"
    assert gateway["LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"].endswith(
        "/run-123456/epoch-authority/stateful-epoch-cutover.json"
    )
    assert result["epoch_authority_identity"]["mapping_hash"] == (
        "sha256:" + "8" * 64
    )
    assert "production.invalid" not in json.dumps(client.created)


def test_secret_cleanup_is_idempotent_when_resources_are_already_absent() -> None:
    module = _load_script("materialize_production_parity_secrets")

    class MissingSecrets:
        def delete_secret(self, **request):
            raise ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": request["SecretId"],
                    }
                },
                "DeleteSecret",
            )

    state = module._cleanup_state(
        run_id="run-123456",
        candidate_sha="a" * 40,
        region="us-east-1",
    )
    result = module.cleanup(state=state, client=MissingSecrets())

    assert result == {
        "run_id": "run-123456",
        "cleaned_secret_count": 6,
        "deleted_secret_count": 0,
        "already_absent_secret_count": 6,
    }
    tampered = json.loads(json.dumps(state))
    tampered["secret_ids"]["gateway"] = (
        "leadpoet/staging/production-parity/run-123456/not-gateway"
    )
    with pytest.raises(module.SecretMaterializationError, match="scope differs"):
        module.cleanup(state=tampered, client=MissingSecrets())


def test_secret_create_persists_cleanup_scope_before_first_aws_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("materialize_production_parity_secrets")
    config_path = tmp_path / "config.json"
    stack_path = tmp_path / "stack.json"
    state_path = tmp_path / "secret-state.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "leadpoet.production_parity_infra.v1",
                "region": "us-east-1",
            }
        ),
        encoding="utf-8",
    )
    stack_path.write_text(
        json.dumps({"run_id": "run-123456", "candidate_sha": "a" * 40}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module,
        "materialize",
        lambda **kwargs: (_ for _ in ()).throw(
            module.SecretMaterializationError("first write failed")
        ),
    )

    assert module.main(
        [
            "create",
            "--config",
            str(config_path),
            "--stack-state",
            str(stack_path),
            "--candidate-sha",
            "a" * 40,
            "--state",
            str(state_path),
        ]
    ) == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "creating"
    assert len(state["secret_ids"]) == 6


def test_database_ingress_is_private_from_initial_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("bootstrap_production_parity_staging")
    commands: list[list[str]] = []
    tls_expected = [
        {
            "FromPort": 443,
            "GroupIds": ["sg-22222222", "sg-33333333"],
            "IpProtocol": "tcp",
            "IpRanges": [],
            "Ipv6Ranges": [],
            "PrefixListIds": [],
            "ToPort": 443,
        }
    ]
    service_expected = [
        {
            "FromPort": 3000,
            "GroupIds": ["sg-11111111"],
            "IpProtocol": "tcp",
            "IpRanges": [],
            "Ipv6Ranges": [],
            "PrefixListIds": [],
            "ToPort": 3000,
        }
    ]

    def fake_run(command, *, timeout):
        commands.append(list(command))
        group = command[command.index("--group-ids") + 1]
        output = json.dumps(
            tls_expected if group == "sg-11111111" else service_expected
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module, "_run", fake_run)

    evidence = module._verify_database_ingress(
        region="us-east-1",
        database_tls_security_group="sg-11111111",
        database_service_security_group="sg-44444444",
        gateway_security_group="sg-22222222",
        dashboard_security_group="sg-33333333",
    )

    assert evidence["public_ingress_absent"] is True
    assert len(commands) == 2
    assert all("describe-security-groups" in item for item in commands)


def test_private_database_tls_readback_rejects_nonprivate_dns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("bootstrap_production_parity_staging")
    key = tmp_path / "controller.pem"
    key.write_text("test", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_ssh",
        lambda *args, **kwargs: json.dumps(
            {"addresses": ["10.0.4.12"], "http_status": 200}
        ),
    )

    value = module._wait_private_database_tls(
        host="ec2-user@203.0.113.10",
        key=key,
        database_domain="database-run.example.test",
    )
    assert value["private_addresses"] == ["10.0.4.12"]

    monkeypatch.setattr(
        module,
        "_ssh",
        lambda *args, **kwargs: json.dumps(
            {"addresses": ["203.0.113.20"], "http_status": 200}
        ),
    )
    with pytest.raises(module.BootstrapError, match="TLS evidence"):
        module._wait_private_database_tls(
            host="ec2-user@203.0.113.10",
            key=key,
            database_domain="database-run.example.test",
        )


def test_database_data_plane_uses_internal_acm_tls_only() -> None:
    template = (ROOT / "infra" / "production-parity-staging.yml").read_text(
        encoding="utf-8"
    )
    database_tls = template.split("  DatabaseWebSecurityGroup:", 1)[1].split(
        "  DatabaseServiceSecurityGroup:", 1
    )[0]
    database_service = template.split(
        "  DatabaseServiceSecurityGroup:", 1
    )[1].split("  DashboardWebSecurityGroup:", 1)[0]
    load_balancer = template.split("  DatabaseLoadBalancer:", 1)[1].split(
        "  GatewayRecord:", 1
    )[0]

    assert "CidrIp: 0.0.0.0/0" not in database_tls
    assert "FromPort: 443" in database_tls
    assert "SourceSecurityGroupId: !Ref GatewayWebSecurityGroup" in database_tls
    assert "SourceSecurityGroupId: !Ref DashboardWebSecurityGroup" in database_tls
    assert "FromPort: 3000" in database_service
    assert "SourceSecurityGroupId: !Ref DatabaseWebSecurityGroup" in database_service
    assert "Type: network" in load_balancer
    assert "Scheme: internal" in load_balancer
    assert "Protocol: TLS" in load_balancer
    assert "CertificateArn: !Ref DatabaseCertificateArn" in load_balancer


def test_stale_parity_cleanup_requires_age_name_and_exact_tags() -> None:
    module = _load_script("cleanup_production_parity_staging")
    now = datetime(2026, 8, 15, 12, tzinfo=timezone.utc)
    run_id = "run-123456"
    candidate = "a" * 40
    tags = [
        {"Key": "leadpoet:parity-run", "Value": run_id},
        {"Key": "leadpoet:candidate-sha", "Value": candidate},
    ]

    class Paginator:
        def __init__(self, values, key):
            self.values = values
            self.key = key

        def paginate(self, **kwargs):
            return [{self.key: self.values}]

    class CloudFormation:
        def __init__(self):
            self.deleted = []

        def get_paginator(self, operation):
            assert operation == "list_stacks"
            return Paginator(
                [
                    {
                        "StackName": "leadpoet-parity-" + run_id,
                        "CreationTime": now - timedelta(hours=13),
                    },
                    {
                        "StackName": "leadpoet-parity-run-fresh",
                        "CreationTime": now - timedelta(hours=1),
                    },
                ],
                "StackSummaries",
            )

        def describe_stacks(self, *, StackName):
            return {"Stacks": [{"Tags": tags}]}

        def delete_stack(self, *, StackName):
            self.deleted.append(StackName)

    class Secrets:
        def __init__(self):
            self.deleted = []

        def get_paginator(self, operation):
            assert operation == "list_secrets"
            return Paginator(
                [
                    {
                        "Name": (
                            "leadpoet/staging/production-parity/"
                            + run_id
                            + "/gateway"
                        ),
                        "CreatedDate": now - timedelta(hours=13),
                        "Tags": tags,
                    },
                    {
                        "Name": "leadpoet/prod/gateway/env",
                        "CreatedDate": now - timedelta(days=30),
                        "Tags": tags,
                    },
                ],
                "SecretList",
            )

        def delete_secret(self, **request):
            self.deleted.append(request["SecretId"])

    class EC2:
        def __init__(self):
            self.deleted = []

        def describe_key_pairs(self, **kwargs):
            return {
                "KeyPairs": [
                    {
                        "KeyName": "leadpoet-parity-" + run_id,
                        "CreateTime": now - timedelta(hours=13),
                        "Tags": tags,
                    }
                ]
            }

        def delete_key_pair(self, **request):
            self.deleted.append(request["KeyName"])

    cloudformation = CloudFormation()
    secrets = Secrets()
    ec2 = EC2()
    result = module.cleanup_stale(
        cloudformation=cloudformation,
        secretsmanager=secrets,
        ec2=ec2,
        now=now,
        max_age_hours=12,
        apply=True,
    )

    assert result["stack_count"] == 1
    assert result["secret_count"] == 1
    assert result["key_pair_count"] == 1
    assert cloudformation.deleted == ["leadpoet-parity-" + run_id]
    assert secrets.deleted == [
        "leadpoet/staging/production-parity/" + run_id + "/gateway"
    ]
    assert ec2.deleted == ["leadpoet-parity-" + run_id]

    cleanup_workflow = (
        ROOT / ".github" / "workflows" / "production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")
    assert 'cron: "17 * * * *"' in cleanup_workflow
    assert "if: vars.LEADPOET_PARITY_INFRA_READY == 'true'" in cleanup_workflow
    assert "--max-age-hours 12" in cleanup_workflow
    assert "--apply" in cleanup_workflow


def _testnet_cutover() -> dict:
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    return SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=1,
        cutover_block=100,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=20,
        last_legacy_epoch_id=19,
    ).to_dict()


def test_epoch_authority_builder_is_stable_and_self_validating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("build_production_parity_epoch_authority")
    authority = __import__(
        "leadpoet_canonical.production_parity_epoch_authority",
        fromlist=["REQUIRED_TABLES"],
    )
    cutover = _testnet_cutover()
    cutover_path = tmp_path / "cutover.json"
    cutover_path.write_text(
        json.dumps(cutover, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    tables = sorted(authority.REQUIRED_TABLES)
    counts = {table: 1 for table in tables}
    state = {
        "lifecycle_state": "stateful_active",
        "mapping_hash": cutover["mapping_hash"],
        "network_genesis_hash": cutover["network_genesis_hash"],
        "netuid": cutover["netuid"],
        "last_legacy_epoch_id": cutover["last_legacy_epoch_id"],
        "first_settlement_epoch_id": cutover["first_settlement_epoch_id"],
    }
    fingerprint = "sha256:" + "3" * 64
    monkeypatch.setattr(module, "_database_env", lambda value: {})
    monkeypatch.setattr(module, "_authority_tables", lambda **kwargs: tables)
    monkeypatch.setattr(
        module,
        "_fingerprint",
        lambda **kwargs: (counts, state, fingerprint),
    )
    monkeypatch.setattr(
        module,
        "_dump_tables",
        lambda path, **kwargs: path.write_bytes(b"custom-postgres-dump"),
    )
    output = tmp_path / "authority.tar"

    result = module.build(
        cutover_path=cutover_path,
        output=output,
        dsn_env="TEST_DSN",
    )
    files, manifest = authority.validate_archive(
        output.read_bytes(),
        {
            "netuid": 1,
            "mapping_hash": cutover["mapping_hash"],
            "network_genesis_hash": cutover["network_genesis_hash"],
        },
    )

    assert result["database_fingerprint_hash"] == fingerprint
    assert set(files) == authority.ARCHIVE_PATHS
    assert manifest["ceremony_evidence"]["authority_dump_hash"] == (
        authority.sha256_bytes(b"custom-postgres-dump")
    )


def test_epoch_authority_builder_rejects_a_changing_ceremony(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("build_production_parity_epoch_authority")
    authority = __import__(
        "leadpoet_canonical.production_parity_epoch_authority",
        fromlist=["REQUIRED_TABLES"],
    )
    cutover = _testnet_cutover()
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(cutover), encoding="ascii")
    tables = sorted(authority.REQUIRED_TABLES)
    counts = {table: 1 for table in tables}
    state = {
        "lifecycle_state": "stateful_active",
        "mapping_hash": cutover["mapping_hash"],
        "network_genesis_hash": cutover["network_genesis_hash"],
        "netuid": 1,
        "last_legacy_epoch_id": 19,
        "first_settlement_epoch_id": 20,
    }
    fingerprints = iter(
        [
            (counts, state, "sha256:" + "3" * 64),
            (counts, state, "sha256:" + "4" * 64),
        ]
    )
    monkeypatch.setattr(module, "_database_env", lambda value: {})
    monkeypatch.setattr(module, "_authority_tables", lambda **kwargs: tables)
    monkeypatch.setattr(module, "_fingerprint", lambda **kwargs: next(fingerprints))
    monkeypatch.setattr(
        module,
        "_dump_tables",
        lambda target, **kwargs: target.write_bytes(b"dump"),
    )

    with pytest.raises(
        authority.ProductionParityEpochAuthorityError,
        match="changed during capture",
    ):
        module.build(
            cutover_path=path,
            output=tmp_path / "authority.tar",
            dsn_env="TEST_DSN",
        )


def test_epoch_authority_rejects_unsafe_archive_members() -> None:
    authority = __import__(
        "leadpoet_canonical.production_parity_epoch_authority",
        fromlist=["validate_archive"],
    )
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        member = tarfile.TarInfo("../authority.dump")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"dump"))

    with pytest.raises(
        authority.ProductionParityEpochAuthorityError,
        match="unsafe member",
    ):
        authority.validate_archive(
            payload.getvalue(),
            {
                "netuid": 1,
                "mapping_hash": "sha256:" + "1" * 64,
                "network_genesis_hash": "0x" + "2" * 64,
            },
        )


def test_bootstrap_verifies_every_ephemeral_instance_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("bootstrap_production_parity_staging")
    fields = {
        "GatewayInstanceId": ("i-11111111", "gateway"),
        "ValidatorInstanceId": ("i-22222222", "primary-validator"),
        "AuditorAInstanceId": ("i-33333333", "audit-validator-a"),
        "AuditorBInstanceId": ("i-44444444", "audit-validator-b"),
        "DatabaseInstanceId": ("i-55555555", "database"),
        "DashboardInstanceId": ("i-66666666", "dashboard"),
    }
    run_id = "run-123456"
    candidate = "a" * 40
    instances = [
        {
            "InstanceId": instance_id,
            "State": {"Name": "running"},
            "MetadataOptions": {
                "HttpTokens": "required",
                "InstanceMetadataTags": "enabled",
            },
            "Tags": [
                {"Key": "leadpoet:parity-run", "Value": run_id},
                {"Key": "leadpoet:parity-role", "Value": role},
                {"Key": "leadpoet:candidate-sha", "Value": candidate},
            ],
        }
        for instance_id, role in fields.values()
    ]
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, timeout: subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"Reservations": [{"Instances": instances}]}),
            "",
        ),
    )

    evidence = module._verify_ephemeral_instances(
        region="us-east-1",
        outputs={field: value[0] for field, value in fields.items()},
        run_id=run_id,
        candidate_sha=candidate,
    )

    assert set(evidence) == {value[1] for value in fields.values()}
    assert all(value["imds_v2_required"] for value in evidence.values())


def test_provisioning_guard_cleans_a_late_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("provision_production_parity_staging")
    cleaned = []

    def fail_late(**kwargs):
        kwargs["output_dir"].mkdir()
        (kwargs["output_dir"] / "controller.pem").write_text("private")
        (kwargs["output_dir"] / "controller.pem.pub").write_text("public")
        raise module.ProvisioningError("late failure")

    monkeypatch.setattr(module, "_provision", fail_late)
    monkeypatch.setattr(
        module,
        "_cleanup_failed_stack",
        lambda **kwargs: cleaned.append(kwargs),
    )

    with pytest.raises(module.ProvisioningError, match="late failure"):
        module.provision(
            config={"region": "us-east-1"},
            candidate_sha="a" * 40,
            run_id="run-123456",
            output_dir=tmp_path / "run",
        )

    assert len(cleaned) == 1
    assert cleaned[0]["stack_name"] == "leadpoet-parity-run-123456"
    assert cleaned[0]["public_path"].name == "controller.pem.pub"


def test_database_authority_toc_requires_exact_declared_tables() -> None:
    module = _load_script("production_parity_database_host")
    listing = "\n".join(
        [
            "; archive header",
            "100; 0 1 TABLE DATA public research_lab_one postgres",
            "101; 0 2 TABLE DATA public research_lab_two postgres",
        ]
    )

    selected = module._authority_toc_entries(
        listing,
        tables=["research_lab_one", "research_lab_two"],
    )

    assert len(selected) == 2
    with pytest.raises(module.DatabaseHostError, match="missing table data"):
        module._authority_toc_entries(
            listing,
            tables=["research_lab_one", "research_lab_missing"],
        )


def test_epoch_authority_builder_rejects_reserved_dsn_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("build_production_parity_epoch_authority")
    monkeypatch.setenv("PGDATABASE", "postgresql://example.invalid/test")

    with pytest.raises(
        module.ProductionParityEpochAuthorityError,
        match="environment name is reserved",
    ):
        module._database_env("PGDATABASE")


def test_production_parity_workflows_remain_candidate_bound_and_ephemeral() -> None:
    fast = (ROOT / ".github/workflows/production-parity-fast.yml").read_text(
        encoding="utf-8"
    )
    full = (ROOT / ".github/workflows/physical-v2-staging.yml").read_text(
        encoding="utf-8"
    )
    release = (ROOT / ".github/workflows/attested-v2-release.yml").read_text(
        encoding="utf-8"
    )

    assert "timeout-minutes: 10" in fast
    assert "if: vars.LEADPOET_PARITY_INFRA_READY == 'true'" in fast
    assert "./.github/actions/setup-production-parity-controller" in fast
    assert "branches:\n      - main" in fast
    assert "run_production_parity_fast.py" in fast
    assert "resolve_production_parity_deployed_sha.py" in fast
    assert "LEADPOET_PARITY_PRODUCTION_GATEWAY_URL" in fast
    assert 'base="$(git rev-parse HEAD^)"' not in fast
    assert 'test "$base" != "$candidate"' in fast
    assert fast.count("--object-lock-days 30") == 3
    assert fast.count("--version-id") >= 4
    assert "archive_version_id" in fast
    assert "manifest_version_id" in fast
    assert "production.dump" in fast
    assert "rm -f --" in fast
    assert "LEADPOET_PARITY_ENFORCEMENT_ENABLED == 'true'" in full
    assert "vars.LEADPOET_PARITY_INFRA_READY == 'true' &&" in full
    assert "github.event_name == 'workflow_dispatch' ||" in full
    assert "RELEASE_SOURCE_PREFIX:" in full
    assert '--release-prefix "$RELEASE_SOURCE_PREFIX"' in full
    assert '--production-db-host "$PRODUCTION_DB_HOST"' in full
    assert "./.github/actions/setup-production-parity-controller" in full
    assert "provision_production_parity_staging.py up" in full
    assert "provision_production_parity_staging.py down" in full
    assert "materialize_production_parity_secrets.py delete" in full
    assert "run_physical_v2_staging.py" in full
    assert "contract_version_id" in full
    assert "ledger_version_id" in full
    assert "--snapshot-archive-version-id" in full
    assert "if: success() && vars.LEADPOET_PARITY_ENFORCEMENT_ENABLED == 'true'" in full
    assert "--prefix attested-v2/releases" in full
    assert "attested-v2/candidates" in release

    snapshot = (
        ROOT / ".github/workflows/production-parity-snapshot.yml"
    ).read_text(encoding="utf-8")
    assert "resolve_production_parity_deployed_sha.py" in snapshot
    assert "if: vars.LEADPOET_PARITY_INFRA_READY == 'true'" in snapshot
    assert '--source-sha "$source_sha"' in snapshot
    assert "./.github/actions/setup-production-parity-controller" in snapshot
    assert "archive-publication.json" in snapshot
    assert "manifest-publication.json" in snapshot

    cleanup = (
        ROOT / ".github/workflows/production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")
    assert "./.github/actions/setup-production-parity-controller" in cleanup


def test_controller_requirements_are_resolved_from_candidate_file(
    tmp_path: Path,
) -> None:
    module = _load_script("resolve_production_parity_controller_requirements")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "boto3==1.40.1\ncryptography>=41.0.7\nhttpx>=0.28.1  # client\n",
        encoding="utf-8",
    )

    assert module.resolve_controller_requirements(requirements) == (
        "boto3==1.40.1",
        "cryptography>=41.0.7",
        "httpx>=0.28.1",
    )


def test_controller_requirements_fail_closed_on_missing_or_ambiguous_input(
    tmp_path: Path,
) -> None:
    module = _load_script("resolve_production_parity_controller_requirements")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("boto3>=1\ncryptography>=41\n", encoding="utf-8")
    with pytest.raises(module.ControllerRequirementsError, match="omit"):
        module.resolve_controller_requirements(requirements)

    requirements.write_text(
        "boto3>=1\nboto3<2\ncryptography>=41\nhttpx>=0.28\n",
        encoding="utf-8",
    )
    with pytest.raises(module.ControllerRequirementsError, match="more than once"):
        module.resolve_controller_requirements(requirements)


def test_fast_clone_adapter_preserves_measured_request_but_targets_raw_postgrest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_production_parity_fast")
    observed: dict[str, Any] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    adapter = module._CloneSupabaseProvider(
        clone_url="http://127.0.0.1:3100",
        service_role_key="test-service-role",
    )
    result = adapter(
        {
            "provider_id": "supabase",
            "method": "GET",
            "url": (
                module.SUPABASE_WEIGHT_SOURCE_ORIGIN
                + "/rest/v1/example?select=id&order=id.asc"
            ),
            "headers": {"range": "0-999"},
            "timeout_ms": 45_000,
            "logical_operation_id": "example-read",
        }
    )

    assert observed == {
        "url": "http://127.0.0.1:3100/example?select=id&order=id.asc",
        "timeout": 45,
    }
    assert result["http_status"] == 200
    assert adapter.pages[0]["request_artifact_hash"].startswith("sha256:")


def test_agent_guides_are_identical_and_require_fast_post_push_parity() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()

    assert agents == claude
    text = agents.decode("utf-8").lower()
    assert "production parity fast" in text
    assert "after every push" in text
    assert "while attestation builds" in text
