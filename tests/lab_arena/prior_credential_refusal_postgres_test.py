"""Durable classification of retries blocked by a proven credential refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    open_round,
    sha,
)


MIGRATION = "214-lab-arena-prior-credential-refusal.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS + (MIGRATION,)
    )


def test_only_matching_unknown_cost_credential_refusal_marks_later_budget_refusal(
    database,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2026-09-11-credproof"
    runners, _participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix="credproof",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=50_000_000,
    )
    leases = [
        claim(store, round_id, runners[0], parallelism=3, ceiling=3)[:2]
        for _ in range(3)
    ]
    dynamic_doc = {"reserve_remaining_budget": True, "tool": "exa_search"}
    first, first_token = leases[0]
    first_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=first["assignment_id"],
        icp_position=first["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha("credential-refused"),
    )
    assert store.reserve_call(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc=dynamic_doc,
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        call_doc={
            "reason": "missing_provider_cost",
            "provider_status": 401,
        },
    )["status"] == "uncertain"

    retry, retry_token = leases[1]
    retry_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=retry["assignment_id"],
        icp_position=retry["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha("credential-retry"),
    )
    refused = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=retry_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc=dynamic_doc,
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "money_cap"
    assert refused["prior_miner_credential_refusal"] is True
    replay = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=retry_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc=dynamic_doc,
    )
    assert replay["status"] == "refused" and replay["idempotent"] is True
    assert replay["prior_miner_credential_refusal"] is True

    other, other_token = leases[2]
    other_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=other["assignment_id"],
        icp_position=other["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.google",
        request_hash=sha("unrelated-provider"),
    )
    unrelated = store.reserve_call(
        run_id=other["run_id"],
        lease_token_hash=hash_lease_token(other_token),
        call_identity=other_identity,
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        funding_source="miner_key",
        amount_microusd=1_000_000,
        call_doc={},
    )
    assert unrelated["status"] == "refused"
    assert unrelated["prior_miner_credential_refusal"] is False

    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text(encoding="utf-8"))
