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
    complete,
    open_round,
    sha,
)


PRIOR_CREDENTIAL_MIGRATION = "214-lab-arena-prior-credential-refusal.sql"
MIGRATION = "218-lab-arena-cross-provider-credential-refusal.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS + (PRIOR_CREDENTIAL_MIGRATION, MIGRATION)
    )


def test_proven_credential_refusal_marks_later_shared_budget_refusal(
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
    assert unrelated["reason"] == "money_cap"
    assert unrelated["prior_miner_credential_refusal"] is True

    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text(encoding="utf-8"))


def test_cross_provider_retry_keeps_proven_miner_credential_failure(database):
    psycopg2, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    round_id = "arena-2026-09-11-xretry"
    runners, _participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix="cross-provider-retry",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=50_000_000,
    )
    first, first_token, _, _ = claim(
        store, round_id, runners[0], parallelism=1, ceiling=1
    )
    deepline_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=first["assignment_id"],
        icp_position=first["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha("deepline-credential-refusal"),
    )
    first_lease_hash = hash_lease_token(first_token)
    assert store.reserve_call(
        run_id=first["run_id"],
        lease_token_hash=first_lease_hash,
        call_identity=deepline_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True, "tool": "exa_search"},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=first["run_id"],
        lease_token_hash=first_lease_hash,
        call_identity=deepline_identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=first["run_id"],
        lease_token_hash=first_lease_hash,
        call_identity=deepline_identity,
        call_doc={"reason": "missing_provider_cost", "provider_status": 402},
    )["status"] == "uncertain"
    failed = complete(store, first["run_id"], first_lease_hash, "provider_error")
    assert failed["confirmation_attempt"] == 2

    retry, retry_token, _, _ = claim(
        store, round_id, runners[0], parallelism=1, ceiling=1
    )
    assert retry["assignment_id"] == first["assignment_id"]
    assert retry["attempt"] == 2
    openrouter_identity = contracts.provider_call_identity(
        attempt=2,
        assignment_id=retry["assignment_id"],
        icp_position=retry["icp_position"],
        action_sequence=0,
        operation_id="openrouter.chat",
        request_hash=sha("openrouter-shared-budget-refusal"),
    )
    refused = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=openrouter_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=1,
        call_doc={},
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "money_cap"
    assert refused["prior_miner_credential_refusal"] is True


def test_unknown_provider_cost_is_distinct_from_a_settled_cap(database):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2026-09-11-costproof"
    runners, _participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix="costproof",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=50_000_000,
    )
    leases = [
        claim(store, round_id, runners[0], parallelism=3, ceiling=3)[:2]
        for _ in range(3)
    ]
    first, first_token = leases[0]
    first_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=first["assignment_id"],
        icp_position=first["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha("unknown-503"),
    )
    dynamic_doc = {"reserve_remaining_budget": True, "tool": "exa_search"}
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
        call_doc={"reason": "missing_provider_cost", "provider_status": 503},
    )["status"] == "uncertain"

    retry, retry_token = leases[1]
    retry_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=retry["assignment_id"],
        icp_position=retry["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha("unknown-503-retry"),
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
    assert refused["reason"] == "provider_cost_uncertain"
    assert refused["prior_miner_credential_refusal"] is False
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
    assert replay["reason"] == "provider_cost_uncertain"

    cross_provider, cross_provider_token = leases[2]
    cross_provider_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=cross_provider["assignment_id"],
        icp_position=cross_provider["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.google",
        request_hash=sha("unknown-503-cross-provider"),
    )
    cross_provider_refusal = store.reserve_call(
        run_id=cross_provider["run_id"],
        lease_token_hash=hash_lease_token(cross_provider_token),
        call_identity=cross_provider_identity,
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        funding_source="miner_key",
        amount_microusd=1_000_000,
        call_doc={},
    )
    assert cross_provider_refusal["status"] == "refused"
    assert cross_provider_refusal["reason"] == "provider_cost_uncertain"
    assert cross_provider_refusal["prior_miner_credential_refusal"] is False


def test_settled_spend_that_consumes_the_cap_remains_money_cap(database):
    psycopg2, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    round_id = "arena-2026-09-11-settledcap"
    runners, _participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix="settledcap",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=10_000_000,
    )
    leases = [
        claim(store, round_id, runners[0], parallelism=2, ceiling=2)[:2]
        for _ in range(2)
    ]
    first, first_token = leases[0]
    first_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=first["assignment_id"],
        icp_position=first["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.google",
        request_hash=sha("settled-cap"),
    )
    assert store.reserve_call(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        funding_source="miner_key",
        amount_microusd=10_000_000,
        call_doc={},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        actual_microusd=10_000_000,
        terminal_response={"status": 200, "headers": {}, "body_b64": ""},
    )["status"] == "settled"

    retry, retry_token = leases[1]
    retry_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=retry["assignment_id"],
        icp_position=retry["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.google",
        request_hash=sha("settled-cap-retry"),
    )
    refused = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=retry_identity,
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        funding_source="miner_key",
        amount_microusd=1_000_000,
        call_doc={},
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "money_cap"
    assert refused["prior_miner_credential_refusal"] is False
