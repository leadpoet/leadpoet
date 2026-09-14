"""Exact Deepline cost recovery after database-authored call termination."""

from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _open_run,
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


def _reserve(
    store,
    run,
    token,
    label,
    *,
    dispatch=True,
    request_id=None,
    include_binding=True,
):
    identity = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.scrape",
        request_hash=sha(label),
    )
    expected_request_id = "ctx-tool-" + identity.removeprefix("sha256:")[:32]
    request_id = expected_request_id if request_id is None else request_id
    fingerprint = "sha256:" + "a" * 64
    call_doc = {"request_hash": sha(label)}
    if include_binding:
        call_doc.update(
            {
                "tool": "firecrawl_scrape",
                "deepline_request_id": request_id,
                "credential_fingerprint": fingerprint,
            }
        )
    token_hash = hash_lease_token(token)
    assert store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="scrapingdog.scrape",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=10_000,
        call_doc=call_doc,
    )["status"] == "reserved"
    if dispatch:
        assert store.mark_dispatched(
            run_id=run["run_id"],
            lease_token_hash=token_hash,
            call_identity=identity,
        )["status"] == "dispatched"
    return identity, request_id, fingerprint


def _settle(store, candidate, *, amount=2_000, units="0.02"):
    return store.reconcile_deepline_cost(
        round_id=candidate["round_id"],
        run_id=candidate["run_id"],
        call_identity=candidate["call_identity"],
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=candidate["request_id"],
        operation=candidate["operation"],
        credential_fingerprint=candidate["credential_fingerprint"],
        actual_microusd=amount,
        cost_units=units,
    )


def test_schema_capability_is_service_only(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_deepline_cost_reconciliation_schema_v1(),"
            "has_function_privilege('lab_arena_service',"
            "'public.lab_arena_deepline_cost_reconciliation_schema_v1()',"
            "'EXECUTE'),has_function_privilege('anon',"
            "'public.lab_arena_deepline_cost_reconciliation_schema_v1()',"
            "'EXECUTE'),has_function_privilege('authenticated',"
            "'public.lab_arena_deepline_cost_reconciliation_schema_v1()',"
            "'EXECUTE'),has_function_privilege('service_role',"
            "'public.lab_arena_deepline_cost_reconciliation_schema_v1()',"
            "'EXECUTE'),has_function_privilege('lab_arena_service',"
            "'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',"
            "'EXECUTE')"
        )
        capability, service, anon, authenticated, service_role, helper = (
            cursor.fetchone()
        )
    assert capability == {
        "schema_version":
            "leadpoet.lab_arena.deepline_cost_reconciliation_schema.v1",
        "version": 248,
    }
    assert (service, anon, authenticated, service_role, helper) == (
        True, False, False, False, False
    )
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,"
                "bigint,text,text,text,bigint,text)'::regprocedure),"
                "pg_get_functiondef("
                "'public.lab_arena_claim_assignment(text,text,integer,integer,"
                "text[],text,text,text,integer)'::regprocedure)"
            )
            settlement, claim_definition = cursor.fetchone()
        assert settlement.count(
            "lab_arena_deepline_interrupted_cost_reconciliation"
        ) == 1
        assert "lab_arena_deepline_interrupted_retry_deferral" not in claim_definition
        assert claim_definition.count(
            "lab_arena_closed_scoring_reservation_claim"
        ) == 1
    finally:
        connection.close()


def test_lease_expiry_reconciles_exact_cost_without_stalling_its_retry(database):
    store = _store(database)
    psycopg2, dsn = database
    round_id = "arena-2026-09-14-hardint"
    try:
        runners, _participants = _open_scoring(
            store, round_id, participants=2, runners=3
        )
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, request_id, fingerprint = _reserve(
            store, first, token, "hard-interruption"
        )
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at="
                "clock_timestamp()-interval '1 second' WHERE run_id=%s",
                (first["run_id"],),
            )
        expired = store.expire_leases(round_id)
        assert expired["expired"] == 1 and expired["retried"] == 1
        retry_id = first["assignment_id"] + ":2"
        rows = store.list_ledger(call_identity=identity)
        assert [row["entry_kind"] for row in rows] == [
            "reservation", "dispatch", "uncertain"
        ]
        assert rows[-1]["entry_doc"] == {
            "reason": "lease_expired", "call": {"call_succeeded": False}
        }
        assert rows[-1]["amount_microusd"] == 10_000
        candidates = store.list_deepline_cost_reconciliations(round_id)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert (candidate["request_id"], candidate["credential_fingerprint"]) == (
            request_id, fingerprint
        )

        # Keep only the retry for this submission. Other-submission work must
        # remain claimable while the exact billing row has not arrived.
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='stage_closed' WHERE round_id=%s AND kind='score' "
                "AND submission_id=%s AND status='pending' AND run_id<>%s",
                (round_id, first["submission_id"], retry_id),
            )
        unrelated, unrelated_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert unrelated["status"] == "leased"
        assert unrelated["submission_id"] != first["submission_id"]
        assert complete(
            store, unrelated["run_id"], hash_lease_token(unrelated_token),
            "accepted", output_ref="arena/test/unrelated-hard-interruption.json",
        )["status"] == "accepted"
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='stage_closed' WHERE round_id=%s AND kind='score' "
                "AND submission_id<>%s AND status='pending'",
                (round_id, first["submission_id"]),
            )
        retry, retry_token, _, _ = claim(
            store, round_id, runners[2], parallelism=8, ceiling=8,
            excluded=[runners[2]],
        )
        assert retry["status"] == "leased"
        assert retry["run_id"] == retry_id
        assert complete(
            store, retry["run_id"], hash_lease_token(retry_token),
            "accepted", output_ref="arena/test/retry-hard-interruption.json",
        )["status"] == "accepted"

        first_settlement = _settle(store, candidate)
        assert first_settlement == {
            "status": "settled", "idempotent": False,
            "actual_microusd": 2_000, "released_microusd": 8_000,
            "variance_microusd": -8_000,
        }
        assert _settle(store, candidate) == dict(
            first_settlement, idempotent=True
        )
        settlement = store.list_ledger(call_identity=identity)[-1]
        assert settlement["entry_doc"]["reconciled_uncertainty_reason"] == (
            "lease_expired"
        )
        exhausted, _, _, _ = claim(
            store, round_id, runners[2], parallelism=8, ceiling=8,
            excluded=[runners[2]],
        )
        assert exhausted["status"] == "no_pending"
    finally:
        store.close()


@pytest.mark.parametrize("reason", ["stage_closed", "round_cancelled"])
def test_other_database_termination_reasons_can_settle_exact_cost(
    database, reason
):
    store = _store(database)
    label = {"stage_closed": "dbstage", "round_cancelled": "dbcancel"}[reason]
    try:
        round_id, _submission, run, token = _open_run(store, label)
        _reserve(store, run, token, label)
        if reason == "round_cancelled":
            assert store.cancel_round(round_id, "test_cancel")["status"] == "cancelled"
        else:
            store.close_stage(round_id, 1)
        candidate = store.list_deepline_cost_reconciliations(round_id)[0]
        assert _settle(store, candidate)["status"] == "settled"
        assert store.list_ledger(
            call_identity=candidate["call_identity"]
        )[-1]["entry_doc"]["reconciled_uncertainty_reason"] == reason
    finally:
        store.close()


@pytest.mark.parametrize(
    "case", ["unknown_reason", "legacy", "bad_request_id", "missing_dispatch"]
)
def test_untrusted_or_incomplete_database_termination_is_not_reconciled(
    database, case
):
    store = _store(database)
    psycopg2, dsn = database
    label = {
        "unknown_reason": "reju",
        "legacy": "rejl",
        "bad_request_id": "rejb",
        "missing_dispatch": "rejmd",
    }[case]
    try:
        round_id, _submission, run, token = _open_run(store, label)
        identity, request_id, fingerprint = _reserve(
            store,
            run,
            token,
            label,
            dispatch=case != "missing_dispatch",
            include_binding=case != "legacy",
            request_id=("ctx-tool-" + "f" * 32) if case == "bad_request_id" else None,
        )
        store.cancel_round(round_id, "test_cancel")
        if case == "unknown_reason":
            with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_ledger SET entry_doc="
                    "jsonb_set(entry_doc,'{reason}',to_jsonb('unknown'::text)) "
                    "WHERE call_identity=%s AND entry_kind='uncertain'",
                    (identity,),
                )
        assert store.list_deepline_cost_reconciliations(round_id) == []
        rows = store.list_ledger(call_identity=identity)
        if case == "missing_dispatch":
            assert rows[-1]["entry_kind"] == "recovery"
        else:
            uncertainty = rows[-1]
            assert uncertainty["entry_kind"] == "uncertain"
            arguments = dict(
                round_id=round_id,
                run_id=run["run_id"],
                call_identity=identity,
                uncertain_entry_id=uncertainty["entry_id"],
                request_id=request_id,
                operation="firecrawl_scrape",
                credential_fingerprint=fingerprint,
                actual_microusd=2_000,
                cost_units="0.02",
            )
            if case == "bad_request_id":
                with pytest.raises(
                    ArenaStoreError,
                    match="lab_arena_deepline_reconciliation_input_invalid",
                ):
                    store.reconcile_deepline_cost(**arguments)
            else:
                assert store.reconcile_deepline_cost(**arguments)["status"] == (
                    "stale"
                )
    finally:
        store.close()
