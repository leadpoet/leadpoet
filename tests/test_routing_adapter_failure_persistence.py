from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from gateway.research_lab.routing_experiment_store import (
    RoutingExperimentExecutionClaim,
    RoutingExperimentStoreError,
    SupabaseRoutingProviderReceiptRepository,
    routing_claim_fence_hash_v3,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import (
    ProviderOutcome,
    ProviderReceipt,
    provider_receipt_key,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "160-research-lab-routing-adapter-failures.sql"
BEHAVIOR = ROOT / "tests" / "sql" / "test_routing_adapter_failures_v2.sql"


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _claim() -> RoutingExperimentExecutionClaim:
    return RoutingExperimentExecutionClaim(
        experiment_hash=_hash("a"),
        claim_key=_hash("b"),
        claim_generation=3,
        claim_fence_hash=routing_claim_fence_hash_v3(
            experiment_hash=_hash("a"),
            claim_key=_hash("b"),
            claim_generation=3,
        ),
    )


def _failure(*, evidence: str = "d") -> ProviderReceipt:
    identity = {
        "binding_id": "binding",
        "tool_id": "intent.source_add.bloomberry_jobs",
        "binding_version": "adapter-v1",
        "source_lineage_id": "lineage",
        "unit_ref": "unit-1",
        "request_fingerprint": _hash("c"),
        "outcome": ProviderOutcome.ADAPTER_FAILURE.value,
        "evidence_hash": _hash(evidence),
        "credit_microunits": 0,
        "latency_ms": 0,
        "execution_mode": "measured_lab",
    }
    return ProviderReceipt(
        receipt_ref="provider_receipt:" + sha256_json(identity).split(":", 1)[1][:16],
        **identity,
    )


def test_adapter_failure_migration_is_separate_and_fail_closed():
    sql = MIGRATION.read_text(encoding="utf-8")
    start = sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_append_adapter_failure_v3"
    )
    end = sql.index("ALTER TABLE public.research_lab_routing_adapter_failures_v2 ENABLE", start)
    function = sql[start:end]
    assert "research_lab_routing_adapter_failures_v2" in sql
    assert "failure_key TEXT PRIMARY KEY" in sql
    assert "provider_receipt_ref TEXT NOT NULL UNIQUE" in sql
    assert "credit_microunits = 0" in sql
    assert "provider_dispatch_started" in function
    assert "research_lab_routing_adapter_failure_v3_dispatch_started" in function
    assert "research_lab_routing_append_provider_attempt_v3" not in function
    assert "terminal_proof" in function
    assert "protected_release_receipt" in function
    assert "provider_attempts_v2" in function
    assert "attempt.provider_receipt_ref = p_provider_receipt_ref" in function
    # This migration does not add a provider-attempt or promotion root.
    assert "research_lab_routing_promote_v3" not in function

    authority_sql = (ROOT / "scripts" / "157-research-lab-routing-experiment-authority.sql").read_text(
        encoding="utf-8"
    )
    provider_start = authority_sql.index(
        "CREATE OR REPLACE FUNCTION public.research_lab_routing_append_provider_attempt_v3"
    )
    provider_end = authority_sql.index("$append_attempt_v3$;", provider_start)
    provider_function = authority_sql[provider_start:provider_end]
    for source in (function, provider_function):
        assert "hashtextextended(p_provider_receipt_ref, 0)" in source
        assert "pg_advisory_xact_lock" in source
        assert "<= " in source
        assert source.index("pg_advisory_xact_lock") < source.index(
            "SELECT * INTO existing"
        )


@dataclass
class _AuthorityStore:
    failure_rows: dict[str, dict] = field(default_factory=dict)
    provider_rows: dict[str, dict] = field(default_factory=dict)
    append_calls: list[tuple[str, dict]] = field(default_factory=list)

    def provider_attempt_row(self, key: str):
        return self.provider_rows.get(key)

    def provider_attempt_keys(self, _experiment_hash: str):
        return tuple(self.provider_rows)

    def adapter_failure_row(self, key: str):
        return self.failure_rows.get(key)

    def adapter_failure_keys(self, _experiment_hash: str):
        return tuple(self.failure_rows)

    def append_adapter_failure(self, **kwargs):
        self.append_calls.append(("adapter_failure", kwargs))
        receipt = kwargs["receipt"]
        key = kwargs["key"]
        self.failure_rows[key] = {
            "experiment_hash": kwargs["experiment_hash"],
            "failure_doc": {
                "schema_version": "leadpoet.research_lab.routing_adapter_failure.v3",
                "failure_key": key,
                "experiment_hash": kwargs["experiment_hash"],
                "binding_id": receipt.binding_id,
                "tool_id": receipt.tool_id,
                "variant_id": kwargs["variant_id"],
                "unit_ref": receipt.unit_ref,
                "claim_key": kwargs["claim"].claim_key,
                "claim_generation": kwargs["claim"].claim_generation,
                "request_fingerprint": receipt.request_fingerprint,
                "outcome": receipt.outcome,
                "credit_microunits": 0,
                "latency_ms": receipt.latency_ms,
                "execution_mode": receipt.execution_mode,
                "pre_dispatch": True,
                "provider_receipt": receipt.to_dict(),
            },
        }
        return {"failure_key": key, "idempotent": False}

    def append_provider_attempt(self, **kwargs):
        self.append_calls.append(("provider_attempt", kwargs))
        return {"attempt_key": kwargs["key"]}


def test_pre_dispatch_failure_uses_dedicated_zero_cost_path_without_provider_proof():
    store = _AuthorityStore()
    claim = _claim()
    repository = SupabaseRoutingProviderReceiptRepository(
        store=store,
        experiment_hash=claim.experiment_hash,
        claim=claim,
    )
    receipt = _failure()
    key = provider_receipt_key(
        tool_id=receipt.tool_id,
        binding_version=receipt.binding_version,
        request_fingerprint=receipt.request_fingerprint,
    )

    assert repository.append_with_context(
        key,
        receipt,
        {"variant_id": "candidate", "unit_ref": receipt.unit_ref},
    ) == receipt
    assert [name for name, _ in store.append_calls] == ["adapter_failure"]
    document = store.failure_rows[key]["failure_doc"]
    assert document["pre_dispatch"] is True
    assert document["provider_receipt"]["credit_microunits"] == 0
    assert not any(
        field in document
        for field in (
            "terminal_proof",
            "terminal_result",
            "terminal_execution_receipt",
            "protected_release_receipt",
            "call_grant_receipt",
            "admission_bundle",
        )
    )
    assert repository.get(key) == receipt
    assert tuple(repository.keys()) == (key,)


def test_pre_dispatch_failure_replay_is_idempotent_and_identity_conflict_is_blocked():
    store = _AuthorityStore()
    claim = _claim()
    repository = SupabaseRoutingProviderReceiptRepository(
        store=store,
        experiment_hash=claim.experiment_hash,
        claim=claim,
    )
    receipt = _failure()
    key = provider_receipt_key(
        tool_id=receipt.tool_id,
        binding_version=receipt.binding_version,
        request_fingerprint=receipt.request_fingerprint,
    )
    repository.append_with_context(
        key,
        receipt,
        {"variant_id": "candidate", "unit_ref": receipt.unit_ref},
    )
    # A second exact write is read-idempotent and does not invoke the durable
    # append path a second time.
    repository.append_with_context(
        key,
        receipt,
        {"variant_id": "candidate", "unit_ref": receipt.unit_ref},
    )
    assert len(store.append_calls) == 1

    conflicting = _failure(evidence="e")
    with pytest.raises(RoutingExperimentStoreError, match="key collision"):
        repository.append_with_context(
            key,
            conflicting,
            {"variant_id": "candidate", "unit_ref": conflicting.unit_ref},
        )


def test_non_failure_receipt_does_not_use_pre_dispatch_failure_path():
    store = _AuthorityStore()
    claim = _claim()
    repository = SupabaseRoutingProviderReceiptRepository(
        store=store,
        experiment_hash=claim.experiment_hash,
        claim=claim,
    )
    receipt = _failure()
    non_failure = ProviderReceipt(
        receipt_ref="provider_receipt:" + "f" * 16,
        binding_id=receipt.binding_id,
        tool_id=receipt.tool_id,
        binding_version=receipt.binding_version,
        source_lineage_id=receipt.source_lineage_id,
        unit_ref=receipt.unit_ref,
        request_fingerprint=receipt.request_fingerprint,
        outcome=ProviderOutcome.SOURCE_MISS.value,
        evidence_hash=receipt.evidence_hash,
        credit_microunits=0,
        latency_ms=0,
        execution_mode=receipt.execution_mode,
    )
    key = provider_receipt_key(
        tool_id=non_failure.tool_id,
        binding_version=non_failure.binding_version,
        request_fingerprint=non_failure.request_fingerprint,
    )
    repository.append_with_context(
        key,
        non_failure,
        {"variant_id": "candidate", "unit_ref": non_failure.unit_ref},
    )
    assert [name for name, _ in store.append_calls] == ["provider_attempt"]


@pytest.mark.skipif(
    not os.getenv("ROUTING_EXPERIMENT_TEST_PG_DSN"),
    reason="set ROUTING_EXPERIMENT_TEST_PG_DSN for disposable PostgreSQL behavior test",
)
def test_adapter_failure_disposable_postgres_behavior():
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is unavailable")
    dsn = os.environ["ROUTING_EXPERIMENT_TEST_PG_DSN"]
    migration = subprocess.run(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert migration.returncode == 0, migration.stderr
    behavior = subprocess.run(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-f", str(BEHAVIOR)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert behavior.returncode == 0, behavior.stderr
