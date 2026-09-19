"""Confirmed-cost-only admission for execute provider calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _uncertain_call,
)
from tests.lab_arena.deepline_interrupted_cost_reconciliation_postgres_test import (
    _settle as reconcile_deepline,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.score_submission_serialization_postgres_test import (
    _open_scoring,
    _reserve_dynamic as reserve_score_dynamic,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    complete,
    sha,
)
from tests.lab_arena.test_lab_arena_service_round import Harness


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "321-lab-arena-confirmed-cost-admission.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "264-lab-arena-codex-cost-reconciliation.sql",
            "289-lab-arena-per-icp-cost-policy.sql",
            "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            MIGRATION.name,
        )
    )


def _harness(database, tmp_path, label):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = Harness(connect, tmp_path, challengers=[], runners=[label])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    round_id = {
        "old-hold": "arena-2099-03-10",
        "cross-attempt": "arena-2099-03-11",
        "parallel-zero": "arena-2099-03-12",
        "score-separation": "arena-2099-03-13",
        "openrouter-success": "arena-2099-03-15",
        "openrouter-failure": "arena-2099-03-16",
    }[label]
    _start_parallel_round(harness, round_id, slot_ceiling=2)
    lease, token = claim(
        harness.service.store,
        harness.round_id,
        harness.runner_keys[0],
        parallelism=2,
        ceiling=2,
    )[:2]
    return harness, lease, token


def _reserve(store, lease, token, label, amount, *, dynamic=False):
    operation = "deepline.execute" if dynamic else "scrapingdog.scrape"
    provider = "deepline" if dynamic else "scrapingdog"
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id=operation,
        request_hash=sha(label),
    )
    call_doc = {"request_hash": sha(label)}
    if dynamic:
        call_doc.update(
            {"reserve_remaining_budget": True, "tool": "exa_search"}
        )
    result = store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id=operation,
        provider=provider,
        funding_source=store.provider_funding(
            lease["run_id"], provider
        )["funding_source"],
        amount_microusd=amount,
        call_doc=call_doc,
    )
    return identity, result


def _dispatch(store, lease, token, identity):
    return store.mark_dispatched(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
    )


def _settle(store, lease, token, identity, amount, *, succeeded=True):
    assert _dispatch(store, lease, token, identity)["status"] == "dispatched"
    return store.settle_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        actual_microusd=amount,
        terminal_response={"status": 200, "call_succeeded": succeeded},
    )


def _insert_historical_hold(connect, lease, identity, amount, *, succeeded=True):
    with connect() as connection, connection.cursor() as cursor:
        for kind in ("reservation", "dispatch", "uncertain"):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc) SELECT "
                "%s,miner_hotkey,round_id,submission_id,run_id,stage,%s,"
                "'openrouter','openrouter.responses','miner_key',%s,"
                "%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",
                (
                    kind,
                    identity,
                    amount,
                    '{"reason":"historical_unknown",'
                    f'"call":{{"call_succeeded":{str(succeeded).lower()}}}}}',
                    lease["run_id"],
                ),
            )


def _uncertain_openrouter(store, lease, token, label, *, succeeded):
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id="openrouter.responses",
        request_hash=sha(label),
    )
    reserved = store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id="openrouter.responses",
        provider="openrouter",
        funding_source=store.provider_funding(
            lease["run_id"], "openrouter"
        )["funding_source"],
        amount_microusd=3_500_000,
        call_doc={
            "model": "openrouter/fixture",
            "reserve_remaining_budget": True,
        },
    )
    assert (reserved["status"], reserved["amount_microusd"]) == (
        "reserved",
        0,
    )
    assert _dispatch(store, lease, token, identity)["status"] == "dispatched"
    generation_id = "gen-" + label
    credential_fingerprint = "sha256:" + ("a" if succeeded else "b") * 64
    assert store.mark_uncertain(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        call_doc={
            "reason": "missing_provider_cost",
            "provider_status": 200 if succeeded else 429,
            "openrouter_generation_id": generation_id,
            "credential_fingerprint": credential_fingerprint,
            "call_succeeded": succeeded,
        },
    )["status"] == "uncertain"
    return identity, generation_id, credential_fingerprint


def test_old_hold_is_retained_but_does_not_block_new_or_reconciled_cost(
    database, tmp_path
):
    harness, lease, token = _harness(database, tmp_path, "old-hold")
    store = harness.service.store
    connect = harness.connect
    old_identity = sha("historical-positive-hold")
    _insert_historical_hold(connect, lease, old_identity, 1_100_000)
    old_rows = store.list_ledger(call_identity=old_identity)

    first_identity, first = _reserve(store, lease, token, "real-sixty", 900_000)
    assert (first["status"], first["amount_microusd"]) == ("reserved", 0)
    assert _settle(store, lease, token, first_identity, 60_000)["status"] == "settled"

    unknown_identity, *_ = _uncertain_call(
        store,
        lease,
        token,
        label="zero-hold-unknown",
        amount=1_000_000,
        call_succeeded=True,
        reason="missing_provider_cost",
        funding_source=store.provider_funding(
            lease["run_id"], "deepline"
        )["funding_source"],
    )
    unknown_rows = store.list_ledger(call_identity=unknown_identity)
    assert [row["amount_microusd"] for row in unknown_rows] == [0, 0, 0]
    replay = store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=unknown_identity,
        operation_id="scrapingdog.scrape",
        provider="deepline",
        funding_source=store.provider_funding(
            lease["run_id"], "deepline"
        )["funding_source"],
        amount_microusd=1_000_000,
        call_doc={"request_hash": sha("zero-hold-unknown")},
    )
    assert (replay["status"], replay["idempotent"]) == ("uncertain", True)
    assert store.list_ledger(call_identity=unknown_identity) == unknown_rows
    unresolved = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=lease["submission_id"],
        icp_position=lease["icp_position"],
        qualified_company_count=1,
    )
    assert unresolved["eligibility_reason"] == "provider_cost_uncertain"
    assert unresolved["competition_sourcing_microusd"] == 60_000
    assert unresolved["execution"]["success_unresolved_calls"] == 2
    assert unresolved["execution"]["success_unresolved_microusd"] == 1_100_000

    later_identity, later = _reserve(
        store, lease, token, "after-old-and-new-unknown", 2_000_000
    )
    assert (later["status"], later["amount_microusd"]) == ("reserved", 0)
    assert _settle(store, lease, token, later_identity, 10_000)["status"] == "settled"
    assert store.list_ledger(call_identity=old_identity) == old_rows

    candidates = store.list_deepline_cost_reconciliations(harness.round_id)
    candidate = next(
        row for row in candidates if row["call_identity"] == unknown_identity
    )
    reconciled = reconcile_deepline(
        store, candidate, amount=60_000, units="0.6"
    )
    assert (reconciled["status"], reconciled["idempotent"]) == (
        "settled",
        False,
    )
    assert reconcile_deepline(
        store, candidate, amount=60_000, units="0.6"
    )["idempotent"] is True

    state = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=lease["submission_id"],
        icp_position=lease["icp_position"],
        qualified_company_count=1,
    )
    assert state["eligibility_reason"] == "provider_cost_uncertain"
    assert state["competition_sourcing_microusd"] == 130_000
    assert state["execution"]["settled_microusd"] == 130_000
    assert state["execution"]["reserved_or_uncertain_microusd"] == 1_100_000


def test_confirmed_billed_failure_reaches_cap_across_attempts(database, tmp_path):
    harness, first, first_token = _harness(database, tmp_path, "cross-attempt")
    store = harness.service.store
    first_identity, reserved = _reserve(
        store, first, first_token, "failed-3940", 3_940_000
    )
    assert reserved["amount_microusd"] == 0
    assert _settle(
        store,
        first,
        first_token,
        first_identity,
        3_940_000,
        succeeded=False,
    )["status"] == "settled"
    assert complete(
        store,
        first["run_id"],
        hash_lease_token(first_token),
        "model_error",
    )["status"] == "failed"

    retry, retry_token = claim(
        store,
        harness.round_id,
        harness.runner_keys[0],
        parallelism=2,
        ceiling=2,
    )[:2]
    assert (retry["assignment_id"], retry["attempt"]) == (
        first["assignment_id"],
        2,
    )
    second_identity, second = _reserve(
        store, retry, retry_token, "failed-60", 1_000_000
    )
    assert (second["status"], second["amount_microusd"]) == ("reserved", 0)
    settled = _settle(
        store, retry, retry_token, second_identity, 60_000, succeeded=False
    )
    assert settled["status"] == "settled"
    replay = store.settle_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=second_identity,
        actual_microusd=60_000,
        terminal_response={"status": 200, "call_succeeded": False},
    )
    assert (replay["status"], replay["idempotent"]) == ("settled", True)

    _paid_identity, refused = _reserve(
        store, retry, retry_token, "paid-after-cap", 1
    )
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")
    free_identity, free = _reserve(
        store, retry, retry_token, "free-after-cap", 0
    )
    assert (free["status"], free["amount_microusd"]) == ("reserved", 0)
    assert _settle(
        store, retry, retry_token, free_identity, 0
    )["status"] == "settled"


def test_parallel_execute_calls_both_reserve_zero_and_can_dispatch(
    database, tmp_path
):
    harness, lease, token = _harness(database, tmp_path, "parallel-zero")
    store = harness.service.store
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = list(
            pool.map(
                lambda label: _reserve(store, lease, token, label, 3_000_000),
                ("parallel-a", "parallel-b"),
            )
        )
    assert [row[1]["status"] for row in calls] == ["reserved", "reserved"]
    assert [row[1]["amount_microusd"] for row in calls] == [0, 0]
    for identity, _reserved in calls:
        assert _dispatch(store, lease, token, identity)["status"] == "dispatched"
        assert [
            row["amount_microusd"]
            for row in store.list_ledger(call_identity=identity)
        ] == [0, 0]


@pytest.mark.parametrize(
    ("succeeded", "actual_microusd", "expected_reason", "expected_spend"),
    [
        (True, 900_000, "cost_per_company_exceeded", 900_000),
        (False, 4_000_000, "eligible", 0),
    ],
)
def test_delayed_openrouter_settlement_keeps_success_and_failure_semantics(
    database,
    tmp_path,
    succeeded,
    actual_microusd,
    expected_reason,
    expected_spend,
):
    label = "openrouter-success" if succeeded else "openrouter-failure"
    harness, lease, token = _harness(database, tmp_path, label)
    store = harness.service.store
    identity, generation_id, fingerprint = _uncertain_openrouter(
        store, lease, token, label, succeeded=succeeded
    )

    pending = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=lease["submission_id"],
        icp_position=lease["icp_position"],
        qualified_company_count=1,
    )
    assert pending["competition_sourcing_microusd"] == 0
    assert pending["execution"]["uncertain_calls"] == 1
    assert pending["eligibility_reason"] == (
        "provider_cost_uncertain" if succeeded else "eligible"
    )

    later_identity, later = _reserve(
        store, lease, token, label + "-during-reconciliation", 2_000_000
    )
    assert (later["status"], later["amount_microusd"]) == ("reserved", 0)
    assert _settle(store, lease, token, later_identity, 0)["status"] == "settled"

    candidates = store.list_openrouter_cost_reconciliations(harness.round_id)
    candidate = next(row for row in candidates if row["call_identity"] == identity)
    arguments = {
        "round_id": harness.round_id,
        "run_id": lease["run_id"],
        "call_identity": identity,
        "uncertain_entry_id": candidate["uncertain_entry_id"],
        "generation_id": generation_id,
        "credential_fingerprint": fingerprint,
        "actual_microusd": actual_microusd,
        "cost_units": str(actual_microusd / 1_000_000),
    }
    reconciled = store.reconcile_openrouter_cost(**arguments)
    assert (reconciled["status"], reconciled["idempotent"]) == (
        "settled",
        False,
    )
    assert store.reconcile_openrouter_cost(**arguments)["idempotent"] is True

    final = store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=lease["submission_id"],
        icp_position=lease["icp_position"],
        qualified_company_count=1,
    )
    assert final["eligibility_reason"] == expected_reason
    assert final["competition_sourcing_microusd"] == expected_spend
    assert final["execution"]["success_unresolved_calls"] == 0
    ledger = store.list_ledger(call_identity=identity)
    settlement = ledger[-1]
    assert settlement["entry_kind"] == "settlement"
    assert settlement["terminal_response"]["provider_cost"] == {
        "basis": "openrouter_generation_cost",
        "operation": "openrouter.responses",
        "request_id": generation_id,
        "unit_name": "usd",
        "units": str(actual_microusd / 1_000_000),
    }
    uncertainty = next(row for row in ledger if row["entry_kind"] == "uncertain")
    assert uncertainty["entry_doc"]["call"]["call_succeeded"] is succeeded

    if not succeeded:
        _next_identity, refused = _reserve(
            store, lease, token, label + "-after-confirmed-cap", 1
        )
        assert (refused["status"], refused["reason"]) == (
            "refused",
            "money_cap",
        )


def test_score_admission_and_execute_score_cost_separation_are_unchanged(
    database, tmp_path
):
    psycopg2, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    runners, _participants = _open_scoring(
        store,
        "arena-2099-03-14",
        participants=1,
        runners=2,
    )
    first, first_token = claim(
        store,
        "arena-2099-03-14",
        runners[0],
        parallelism=8,
        ceiling=8,
        excluded=[runners[0]],
    )[:2]
    assert first["kind"] == "score"
    _first_identity, first_reservation = reserve_score_dynamic(
        store, first, first_token, "score-first"
    )
    assert first_reservation["status"] == "reserved"
    assert first_reservation["amount_microusd"] > 0
    _second_identity, second_reservation = reserve_score_dynamic(
        store, first, first_token, "score-second"
    )
    assert second_reservation["status"] == "budget_busy"

    harness, lease, token = _harness(database, tmp_path, "score-separation")
    score_run_id = harness.round_id + ":historical-score"
    with harness.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,stage_generation,kind,scored_run_id) "
            "VALUES (%s,%s,%s,%s,%s,1,%s,1,'accepted',1,'score',%s)",
            (
                score_run_id,
                score_run_id,
                harness.round_id,
                lease["submission_id"],
                lease["miner_hotkey"],
                lease["icp_position"],
                lease["run_id"],
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,"
            "amount_microusd,terminal_response,entry_doc) VALUES "
            "('settlement',%s,%s,%s,%s,1,%s,'openrouter',"
            "'openrouter.responses','miner_key',50000000,"
            "'{\"status\":200,\"call_succeeded\":true}'::jsonb,'{}'::jsonb)",
            (
                lease["miner_hotkey"],
                harness.round_id,
                lease["submission_id"],
                score_run_id,
                sha("judge-only-cost"),
            ),
        )
    execute_identity, execute = _reserve(
        harness.service.store, lease, token, "execute-after-score", 1
    )
    assert (execute["status"], execute["amount_microusd"]) == ("reserved", 0)
    assert _settle(
        harness.service.store, lease, token, execute_identity, 100_000
    )["status"] == "settled"
    execution = harness.service.store.icp_cost_eligibility(
        round_id=harness.round_id,
        submission_id=lease["submission_id"],
        icp_position=lease["icp_position"],
        qualified_company_count=1,
    )["execution"]
    assert execution["settled_microusd"] == 100_000


def test_migration_is_exactly_replayable_and_keeps_scoped_markers(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_reserve_call(text,text,text,text,text,text,"
                "bigint,jsonb,integer)'::pg_catalog.regprocedure)"
            )
            before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'"
                "::pg_catalog.regprocedure), pg_catalog.pg_get_functiondef("
                "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
                "::pg_catalog.regprocedure)"
            )
            eligibility_before, publication_before = cursor.fetchone()
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_reserve_call(text,text,text,text,text,text,"
                "bigint,jsonb,integer)'::pg_catalog.regprocedure)"
            )
            after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'"
                "::pg_catalog.regprocedure), pg_catalog.pg_get_functiondef("
                "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
                "::pg_catalog.regprocedure)"
            )
            eligibility_after, publication_after = cursor.fetchone()
    assert after == before
    assert eligibility_after == eligibility_before
    assert publication_after == publication_before
    assert after.count("lab_arena_confirmed_cost_admission") == 1
    assert eligibility_after.count("lab_arena_confirmed_cost_eligibility") == 1
    assert "success_unresolved_microusd')::BIGINT" not in eligibility_after
    assert "lab_arena_closed_scoring_reservation_admission" in after
    assert "AND (v_dynamic OR p_amount_microusd > 0)" in after
    assert "IF v_per_icp_policy THEN\n    p_amount_microusd := 0;" in after
