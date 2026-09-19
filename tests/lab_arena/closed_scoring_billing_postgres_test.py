"""Exact late judge billing preserves the already-published result."""
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _open_scoring, _store,
)
from tests.lab_arena.deepline_interrupted_cost_reconciliation_postgres_test import _settle
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha

MIGRATION = Path(__file__).resolve().parents[2] / "scripts/311-lab-arena-per-icp-closed-billing-reconciliation.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION.name,)
    )


def _closed_call(database, label, *, dynamic=True, policy="successful_calls_v1"):
    psycopg2, dsn = database
    store = _store(database)
    round_id = "arena-2026-09-14-cb" + label
    runners, _ = _open_scoring(store, round_id, participants=2, runners=2)
    run, token, _, _ = claim(store, round_id, runners[0], parallelism=8, ceiling=8)
    identity = contracts.provider_call_identity(
        attempt=run["attempt"], assignment_id=run["assignment_id"],
        icp_position=run["icp_position"], action_sequence=0,
        operation_id="scrapingdog.scrape", request_hash=sha(label),
    )
    request_id = "ctx-tool-" + identity[7:39]
    fingerprint = "sha256:" + "a" * 64
    token_hash = hash_lease_token(token)
    assert store.reserve_call(
        run_id=run["run_id"], lease_token_hash=token_hash, call_identity=identity,
        operation_id="scrapingdog.scrape", provider="deepline", funding_source="miner_key",
        amount_microusd=0 if dynamic else 10_000,
        call_doc={"request_hash": sha(label), "tool": "firecrawl_scrape",
                  "deepline_request_id": request_id, "credential_fingerprint": fingerprint,
                  **({"reserve_remaining_budget": True} if dynamic else {})},
    )["status"] == "reserved"
    assert store.mark_dispatched(run_id=run["run_id"], lease_token_hash=token_hash,
                                 call_identity=identity)["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=run["run_id"], lease_token_hash=token_hash, call_identity=identity,
        call_doc={"reason": "transport_failure", "call_succeeded": False,
                  "deepline_request_id": request_id, "deepline_operation": "firecrawl_scrape",
                  "credential_fingerprint": fingerprint},
    )["status"] == "uncertain"
    complete(store, run["run_id"], token_hash, "judge_error")
    assert store.cancel_round(round_id, "test_cancel")["status"] == "cancelled"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc || "
            "jsonb_build_object('mode','live','network_name','finney','netuid',71,"
            "'sourcing_cost_eligibility_policy',%s::text) "
            "WHERE round_id=%s", (policy, round_id),
        )
    candidate = store.list_deepline_cost_reconciliations(round_id)[0]
    return store, round_id, candidate


def _next(store, round_id, **kwargs):
    return store.next_closed_deepline_reconciliation(
        mode="live", network_name="finney", netuid=71,
        round_id=round_id, **kwargs,
    )


@pytest.mark.parametrize("policy", ["successful_calls_v1", "successful_calls_per_icp_v1"])
def test_published_late_bill_is_exact_idempotent_and_does_not_rewrite_result(database, policy):
    psycopg2, dsn = database
    store, round_id, candidate = _closed_call(database, "positive" + str("per_icp" in policy).lower(), policy=policy)
    try:
        first = _next(store, round_id)
        assert first["uncertain_entry_id"] == candidate["uncertain_entry_id"]
        assert _next(store, round_id, after_entry_id=first["uncertain_entry_id"]) == first
        assert store.next_closed_deepline_reconciliation(
            mode="live", network_name="test", netuid=71, round_id=round_id,
        ) == {"status": "none"}
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published',"
                "publication_doc='{\"immutable_snapshot\":true}'::jsonb WHERE round_id=%s",
                (round_id,),
            )
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (round_id,))
            before = cursor.fetchone()[0]
        old_ledger = store.list_ledger(call_identity=candidate["call_identity"])
        assert store.list_deepline_cost_reconciliations(round_id) == [candidate]
        wrong = dict(candidate, request_id="ctx-tool-" + "f" * 32)
        with pytest.raises(ArenaStoreError):
            _settle(store, wrong)
        assert store.list_ledger(call_identity=candidate["call_identity"]) == old_ledger
        result = _settle(store, candidate)
        assert result["status"] == "settled" and result["idempotent"] is False
        assert _settle(store, candidate) == dict(result, idempotent=True)
        assert _next(store, round_id) == {"status": "none"}
        new_ledger = store.list_ledger(call_identity=candidate["call_identity"])
        assert new_ledger[:-1] == old_ledger
        assert new_ledger[-1]["amount_microusd"] == 2_000
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (round_id,))
            assert cursor.fetchone()[0] == before
    finally:
        store.close()


@pytest.mark.parametrize("policy", ["successful_calls_v1", "successful_calls_per_icp_v1"])
def test_fixed_unknown_is_not_admitted_to_published_mutation_path(database, policy):
    psycopg2, dsn = database
    store, round_id, candidate = _closed_call(database, "fixed" + str("per_icp" in policy).lower(), dynamic=False, policy=policy)
    try:
        assert _next(store, round_id) == {"status": "none"}
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (round_id,))
        assert store.list_deepline_cost_reconciliations(round_id) == []
        assert _settle(store, candidate) == {"status": "stale"}
    finally:
        store.close()


def test_closed_billing_migration_is_idempotent_and_service_only(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            signature = "public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)"
            for role, expected in (("lab_arena_service", True), ("anon", False),
                                   ("authenticated", False), ("service_role", False)):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                assert cursor.fetchone() == (expected,)
    finally:
        connection.close()
