"""Budget admission for closed, failed dynamic Deepline scoring calls."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _open_scoring,
    _store,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/248-lab-arena-closed-scoring-reservation-admission.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _dynamic_call(store, run, token, label, *, fingerprint="sha256:" + "a" * 64):
    identity = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.scrape",
        request_hash=sha(label),
    )
    request_id = "ctx-tool-" + identity.removeprefix("sha256:")[:32]
    token_hash = hash_lease_token(token)
    reserved = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="scrapingdog.scrape",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc={
            "request_hash": sha(label),
            "base_call_identity": identity,
            "provider_attempt": 1,
            "action_sequence": 0,
            "tool": "firecrawl_scrape",
            "deepline_request_id": request_id,
            "credential_fingerprint": fingerprint,
            "reserve_remaining_budget": True,
        },
    )
    assert reserved["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    return identity, request_id, reserved


def _uncertain(store, run, token, identity, request_id, *, fingerprint):
    assert store.mark_uncertain(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        call_doc={
            "reason": "transport_failure",
            "call_succeeded": False,
            "deepline_request_id": request_id,
            "deepline_operation": "firecrawl_scrape",
            "credential_fingerprint": fingerprint,
        },
    )["status"] == "uncertain"


def _admission(database, submission_id):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__submission_kind_admission_spend_v1(%s,'score',false)",
            (submission_id,),
        )
        return cursor.fetchone()[0]


def test_migration_is_repeatable_and_keeps_internal_helpers_private(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'"
                "::regprocedure),pg_get_functiondef("
                "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'"
                "::regprocedure),public.lab_arena_deepline_cost_reconciliation_schema_v1()"
            )
            reserve_definition, claim_definition, capability = cursor.fetchone()
            assert reserve_definition.count(
                "lab_arena_closed_scoring_reservation_admission"
            ) == 1
            assert claim_definition.count(
                "lab_arena_closed_scoring_reservation_claim"
            ) == 1
            assert "lab_arena_deepline_reconciliation_retry_deferral" not in claim_definition
            assert capability["version"] == 248
            for signature in (
                "public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)",
                "public.lab_arena__submission_kind_admission_spend_v1(text,text,boolean)",
                "public.lab_arena__submission_kind_has_admission_uncertainty_v1(text,text)",
            ):
                cursor.execute(
                    "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')",
                    (signature,),
                )
                assert cursor.fetchone() == (False,)
    finally:
        connection.close()


def test_closed_dynamic_uncertainty_is_reported_but_does_not_hold_admission(database):
    store = _store(database)
    round_id = "arena-2026-09-14-cbdynamic"
    try:
        runners, _ = _open_scoring(store, round_id, participants=1, runners=2)
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, request_id, reservation = _dynamic_call(
            store, first, token, "closed-dynamic"
        )
        assert reservation["amount_microusd"] == 50_000_000
        _uncertain(
            store, first, token, identity, request_id,
            fingerprint="sha256:" + "a" * 64,
        )
        assert complete(
            store, first["run_id"], hash_lease_token(token), "judge_error"
        )["status"] == "failed"

        uncertain_entry = store.list_ledger(call_identity=identity)[-1]
        assert uncertain_entry["amount_microusd"] == 50_000_000
        assert _admission(database, first["submission_id"]) == 0
        costs = store.submission_costs(first["submission_id"])
        assert sum(
            row["reserved_or_uncertain_microusd"] for row in costs["providers"]
        ) == 50_000_000

        retry, retry_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert retry["submission_id"] == first["submission_id"]
        second_identity, _, second_reservation = _dynamic_call(
            store, retry, retry_token, "closed-dynamic-retry"
        )
        assert second_reservation["amount_microusd"] == 50_000_000
        assert store.settle_call(
            run_id=retry["run_id"],
            lease_token_hash=hash_lease_token(retry_token),
            call_identity=second_identity,
            actual_microusd=50_000_000,
            terminal_response={"status": 200, "call_succeeded": True},
        )["status"] == "settled"
        refused = store.reserve_call(
            run_id=retry["run_id"],
            lease_token_hash=hash_lease_token(retry_token),
            call_identity=sha("real-cap-refusal"),
            operation_id="scrapingdog.scrape",
            provider="deepline",
            funding_source="miner_key",
            amount_microusd=1,
            call_doc={"request_hash": sha("real-cap-refusal")},
        )
        assert refused["status"] == "refused"
        assert refused["reason"] == "money_cap"
        assert complete(
            store, retry["run_id"], hash_lease_token(retry_token), "judge_error"
        )["status"] == "failed"
        assert store.cancel_round(round_id, "test_reconciliation")["status"] == "cancelled"
        candidate = next(
            row for row in store.list_deepline_cost_reconciliations(round_id)
            if row["call_identity"] == identity
        )
        assert store.reconcile_deepline_cost(
            round_id=round_id,
            run_id=first["run_id"],
            call_identity=identity,
            uncertain_entry_id=candidate["uncertain_entry_id"],
            request_id=request_id,
            operation="firecrawl_scrape",
            credential_fingerprint="sha256:" + "a" * 64,
            actual_microusd=2_000,
            cost_units="0.02",
        )["status"] == "settled"
        assert _admission(database, first["submission_id"]) == 50_002_000
    finally:
        store.close()


@pytest.mark.parametrize("case", ["active", "fixed", "false_binding"])
def test_nonretired_uncertainty_keeps_its_existing_admission_hold(database, case):
    store = _store(database)
    round_id = "arena-2026-09-14-adm" + case.replace("_", "")
    try:
        runners, _ = _open_scoring(store, round_id, participants=1, runners=2)
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, request_id, reservation = _dynamic_call(store, first, token, case)
        expected = 50_000_000
        if case == "fixed":
            # Replace only this fixture's immutable reservation shape. The
            # accounting amount remains the same, but it is no longer dynamic.
            psycopg2, dsn = database
            with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'reserve_remaining_budget' "
                    "WHERE call_identity=%s AND entry_kind='reservation'",
                    (identity,),
                )
        if case != "active":
            _uncertain(
                store, first, token, identity, request_id,
                fingerprint=("sha256:" + ("b" if case == "false_binding" else "a") * 64),
            )
            assert complete(
                store, first["run_id"], hash_lease_token(token), "judge_error"
            )["status"] == "failed"
        assert reservation["amount_microusd"] == expected
        assert _admission(database, first["submission_id"]) == expected

        if case != "active":
            retry, retry_token, _, _ = claim(
                store, round_id, runners[1], parallelism=8, ceiling=8,
                excluded=[runners[1]],
            )
            assert retry["status"] == "leased"
            refused = store.reserve_call(
                run_id=retry["run_id"],
                lease_token_hash=hash_lease_token(retry_token),
                call_identity=sha("held-" + case),
                operation_id="scrapingdog.scrape",
                provider="deepline",
                funding_source="miner_key",
                amount_microusd=0,
                call_doc={
                    "request_hash": sha("held-" + case),
                    "reserve_remaining_budget": True,
                },
            )
            assert refused["status"] == "refused"
            assert refused["reason"] == "provider_cost_uncertain"
    finally:
        store.close()
