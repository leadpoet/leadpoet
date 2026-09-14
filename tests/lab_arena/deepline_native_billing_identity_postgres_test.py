"""Real SQL binding and append-only settlement of native Deepline receipts."""

from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _open_scoring, _store,
)
from tests.lab_arena.deepline_interrupted_cost_reconciliation_postgres_test import (
    _reserve, _settle,
)
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import NATIVE_REQUEST_ID
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_native_migration_is_repeatable_and_private(database):
    psycopg2, dsn = database
    migration = Path(__file__).resolve().parents[2] / "scripts/247-lab-arena-deepline-native-billing-identities.sql"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(migration.read_text())
            cursor.execute(migration.read_text())
            cursor.execute("SELECT public.lab_arena_deepline_cost_reconciliation_schema_v1()->>'version'")
            assert cursor.fetchone() == ("247",)
            for signature in (
                "public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)",
                "public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)",
            ):
                cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
                assert cursor.fetchone()[0].count("lab_arena_deepline_native_billing_identity") == 1
                for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                    cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                    assert cursor.fetchone() == (role == "lab_arena_service",)
    finally:
        connection.close()


@pytest.mark.parametrize("native", [True, False])
def test_exact_retained_receipt_settles_once_preserves_history_and_releases_retry(database, native):
    store = _store(database)
    label = "nativeid" if native else "callerid"
    round_id = "arena-2026-09-14-" + label
    try:
        runners, _ = _open_scoring(store, round_id, participants=1, runners=3)
        run, token, _, _ = claim(store, round_id, runners[0], parallelism=8, ceiling=8,
                                  excluded=[runners[0]])
        identity, caller_id, fingerprint = _reserve(store, run, token, label)
        call_doc = {
            "reason": "transport_failure", "call_succeeded": False,
            "deepline_request_id": caller_id, "deepline_operation": "firecrawl_scrape",
            "credential_fingerprint": fingerprint,
            **({"deepline_job_id": NATIVE_REQUEST_ID} if native else {}),
        }
        assert store.mark_uncertain(run_id=run["run_id"], lease_token_hash=hash_lease_token(token),
                                    call_identity=identity, call_doc=call_doc)["status"] == "uncertain"
        assert complete(store, run["run_id"], hash_lease_token(token), "judge_error")["status"] == "failed"
        original = store.list_ledger(call_identity=identity)
        psycopg2, dsn = database
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT md5(string_agg(to_jsonb(r)::text,'|' ORDER BY run_id)) "
                           "FROM public.lab_arena_runs r WHERE round_id=%s AND status='accepted'", (round_id,))
            accepted_before = cursor.fetchone()
        candidates = store.list_deepline_cost_reconciliations(round_id)
        assert len(candidates) == 1
        candidate = candidates[0]
        expected = NATIVE_REQUEST_ID if native else caller_id
        assert candidate["request_id"] == expected
        # Neither a nearby provider job nor the pre-dispatch local id may
        # substitute for an immutable native receipt.
        wrong = ["iad1::other-1789361973860-8634524a6f3a"]
        if native:
            wrong.append(caller_id)
        for request_id in wrong:
            assert _settle(store, {**candidate, "request_id": request_id}) == {"status": "stale"}
        assert store.list_ledger(call_identity=identity) == original
        blocked, _, _, _ = claim(store, round_id, runners[1], parallelism=8, ceiling=8,
                                 excluded=[runners[1]])
        assert blocked["status"] == "no_pending"
        result = _settle(store, candidate)
        assert result == {"status": "settled", "idempotent": False,
                          "actual_microusd": 2_000, "released_microusd": 8_000,
                          "variance_microusd": -8_000}
        assert _settle(store, candidate) == {**result, "idempotent": True}
        assert _settle(store, {**candidate, "request_id": wrong[0]}) == {"status": "conflict"}
        after = store.list_ledger(call_identity=identity)
        assert after[:-1] == original
        assert after[-1]["terminal_response"]["provider_cost"]["request_id"] == expected
        assert store.list_deepline_cost_reconciliations(round_id) == []
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT md5(string_agg(to_jsonb(r)::text,'|' ORDER BY run_id)) "
                           "FROM public.lab_arena_runs r WHERE round_id=%s AND status='accepted'", (round_id,))
            assert cursor.fetchone() == accepted_before
        released, _, _, _ = claim(store, round_id, runners[1], parallelism=8, ceiling=8,
                                  excluded=[runners[1]])
        assert released["status"] == "leased"
    finally:
        store.close()
