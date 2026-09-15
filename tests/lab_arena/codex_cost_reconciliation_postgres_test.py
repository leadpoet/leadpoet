"""Responses uses the existing exact-generation recovery and service ACLs."""

from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.openrouter_delayed_cost_reconciliation_postgres_test import (
    resources, _round_and_run, _uncertain_call, _settlements,
    test_reconciliation_functions_are_service_only,
)

MIGRATION = "258-lab-arena-codex-cost-reconciliation.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (MIGRATION,))


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_exact_responses_generation_recovery_is_idempotent_and_owner_bound(resources, operation):
    store, connect = resources
    label = operation.rsplit(".", 1)[1]
    round_id, _, run, token_hash = _round_and_run(store, "codex" + label)
    fingerprint = "sha256:" + "a" * 64
    generation = "gen-codex-" + label
    identity = _uncertain_call(store, run, token_hash, label=label, sequence=0,
                               generation_id=generation, credential_fingerprint=fingerprint,
                               operation_id=operation)
    candidates = store.list_openrouter_cost_reconciliations(round_id)
    assert len(candidates) == 1
    args = dict(round_id=round_id, run_id=run["run_id"], call_identity=identity,
                uncertain_entry_id=candidates[0]["uncertain_entry_id"], generation_id=generation,
                credential_fingerprint=fingerprint, actual_microusd=123, cost_units="0.000123")
    assert store.reconcile_openrouter_cost(**{**args, "credential_fingerprint": "sha256:" + "b" * 64})["status"] == "stale"
    assert store.reconcile_openrouter_cost(**args)["status"] == "settled"
    assert store.reconcile_openrouter_cost(**args)["idempotent"] is True
    settlements = _settlements(store, identity)
    assert len(settlements) == 1 and settlements[0]["amount_microusd"] == 123
    assert settlements[0]["terminal_response"]["provider_cost"]["operation"] == operation
    assert store.list_openrouter_cost_reconciliations(round_id) == []


def test_reapplication_preserves_recovery_and_permissions(resources):
    _, connect = resources
    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text())
    test_reconciliation_functions_are_service_only(resources)
