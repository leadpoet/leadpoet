"""Exact-ID OpenRouter billing reconciliation races and ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.participation_original_judgments_postgres_test import (
    _has_recent_participation,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, open_round, sha


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


@pytest.fixture()
def resources(database):
    psycopg2, dsn = database
    connections = []

    def connect():
        connection = psycopg2.connect(**dsn)
        connections.append(connection)
        return connection

    store = ArenaStore(PsycopgTransport(connect), lease_ttl_seconds=120)
    yield store, connect
    store.close()
    for connection in connections:
        if not connection.closed:
            connection.close()


def _round_and_run(store: ArenaStore, label: str):
    round_id = "arena-2026-09-12-%s" % label
    runners, participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix=label,
        execution_cap_microusd=10_000_000,
    )
    run, token, _, _ = claim(store, round_id, runners[0])
    return round_id, participants[0]["submission_id"], run, hash_lease_token(token)


def _uncertain_call(
    store: ArenaStore,
    run,
    token_hash: str,
    *,
    label: str,
    sequence: int,
    amount: int = 500_000,
    generation_id: str | None = None,
    credential_fingerprint: str | None = None,
):
    identity = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=sequence,
        operation_id="openrouter.chat",
        request_hash=sha(label),
    )
    assert store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=amount,
        call_doc={"model": "fixture"},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    call_doc = {
        "reason": "missing_provider_cost",
        "provider_status": 502,
    }
    if generation_id is not None:
        call_doc["openrouter_generation_id"] = generation_id
    if credential_fingerprint is not None:
        call_doc["credential_fingerprint"] = credential_fingerprint
    assert store.mark_uncertain(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        call_doc=call_doc,
    )["status"] == "uncertain"
    return identity


def _settlements(store: ArenaStore, identity: str):
    return [
        row
        for row in store.list_ledger(call_identity=identity)
        if row["entry_kind"] == "settlement"
    ]


def test_reconciliation_functions_are_service_only(resources):
    _store, connect = resources
    signatures = (
        "public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer)",
        "public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text)",
    )
    with connect() as connection, connection.cursor() as cursor:
        for signature in signatures:
            cursor.execute(
                "SELECT has_function_privilege(%s, %s, 'EXECUTE')",
                ("lab_arena_service", signature),
            )
            assert cursor.fetchone()[0] is True
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute(
                    "SELECT has_function_privilege(%s, %s, 'EXECUTE')",
                    (role, signature),
                )
                assert cursor.fetchone()[0] is False


@pytest.mark.parametrize(
    ("cost_units", "actual_microusd"),
    (("0", 0), ("0.000123", 123), ("1.25", 1_250_000)),
)
def test_exact_zero_positive_and_overestimate_costs_settle(
    resources, cost_units, actual_microusd
):
    store, _connect = resources
    label = "cost%s" % actual_microusd
    round_id, _submission_id, run, token_hash = _round_and_run(store, label)
    fingerprint = "sha256:" + "a" * 64
    generation_id = "gen-%s" % label
    identity = _uncertain_call(
        store,
        run,
        token_hash,
        label=label,
        sequence=0,
        amount=500_000,
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
    )
    candidate = store.list_openrouter_cost_reconciliations(round_id)[0]

    result = store.reconcile_openrouter_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
        actual_microusd=actual_microusd,
        cost_units=cost_units,
    )

    assert result["status"] == "settled" and result["idempotent"] is False
    assert result["variance_microusd"] == actual_microusd - 500_000
    settlement = _settlements(store, identity)[0]
    assert settlement["amount_microusd"] == actual_microusd
    assert settlement["terminal_response"]["provider_cost"] == {
        "basis": "openrouter_generation_cost",
        "units": cost_units,
        "unit_name": "usd",
        "operation": "openrouter.chat",
        "request_id": generation_id,
    }
    assert store.list_openrouter_cost_reconciliations(round_id) == []


def test_reconciliation_is_atomic_idempotent_and_owner_bound(resources):
    store, _connect = resources
    round_id, _submission_id, run, token_hash = _round_and_run(store, "race")
    other_round, _, _, _ = _round_and_run(store, "otherowner")
    fingerprint = "sha256:" + "b" * 64
    generation_id = "gen-race"
    identity = _uncertain_call(
        store,
        run,
        token_hash,
        label="race",
        sequence=0,
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
    )
    candidate = store.list_openrouter_cost_reconciliations(round_id)[0]
    arguments = {
        "round_id": round_id,
        "run_id": run["run_id"],
        "call_identity": identity,
        "uncertain_entry_id": candidate["uncertain_entry_id"],
        "generation_id": generation_id,
        "credential_fingerprint": fingerprint,
        "actual_microusd": 321,
        "cost_units": "0.000321",
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: store.reconcile_openrouter_cost(**arguments), range(2)))

    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert len(_settlements(store, identity)) == 1
    wrong_owner = store.reconcile_openrouter_cost(
        **dict(arguments, round_id=other_round)
    )
    assert wrong_owner["status"] == "stale"
    conflict = store.reconcile_openrouter_cost(
        **dict(arguments, actual_microusd=322, cost_units="0.000322")
    )
    assert conflict["status"] == "conflict"
    assert len(_settlements(store, identity)) == 1


def test_reconciliation_after_n_minus_one_acceptance_preserves_participation(
    resources, database
):
    store, _connect = resources
    round_id, _submission_id, run, token_hash = _round_and_run(
        store, "accepted"
    )
    runner = store.get_run(run["run_id"])["runner_hotkey"]
    fingerprint = "sha256:" + "f" * 64
    generation_id = "gen-accepted"
    identity = _uncertain_call(
        store,
        run,
        token_hash,
        label="accepted",
        sequence=0,
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
    )

    # An N-1 service could complete after the uncertainty became terminal.
    # Exercise that SQL RPC directly; the current service normally reconciles
    # first and waits while provider cost remains unavailable.
    completed = store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/%s/accepted.json" % round_id,
    )
    assert completed["status"] == "accepted"
    accepted_before = store.get_run(run["run_id"])
    accepted_at = accepted_before["participation_accepted_at"]
    assert accepted_at is not None
    assert accepted_before["status"] == "accepted"
    assert accepted_before["runner_hotkey"] == runner
    assert _has_recent_participation(database, runner)

    candidates = store.list_openrouter_cost_reconciliations(
        round_id, run_id=run["run_id"]
    )
    assert len(candidates) == 1
    assert candidates[0]["run_status"] == "accepted"
    arguments = {
        "round_id": round_id,
        "run_id": run["run_id"],
        "call_identity": identity,
        "uncertain_entry_id": candidates[0]["uncertain_entry_id"],
        "generation_id": generation_id,
        "credential_fingerprint": fingerprint,
        "actual_microusd": 123,
        "cost_units": "0.000123",
    }
    settled = store.reconcile_openrouter_cost(**arguments)
    assert settled["status"] == "settled" and settled["idempotent"] is False
    assert store.get_run(run["run_id"]) == accepted_before
    assert _has_recent_participation(database, runner)

    replay = store.reconcile_openrouter_cost(**arguments)
    assert replay["status"] == "settled" and replay["idempotent"] is True
    accepted_after = store.get_run(run["run_id"])
    assert accepted_after == accepted_before
    assert accepted_after["participation_accepted_at"] == accepted_at
    assert accepted_after["runner_hotkey"] == runner
    assert len(_settlements(store, identity)) == 1


def test_list_cursor_skips_unavailable_first_item_and_wraps(resources):
    store, _connect = resources
    round_id = "arena-2026-09-12-cursor"
    runners, _participants = open_round(
        store,
        round_id,
        participants=2,
        runners=1,
        prefix="cursor",
        execution_cap_microusd=10_000_000,
    )
    first_run, first_token, _, _ = claim(
        store, round_id, runners[0], parallelism=2
    )
    second_run, second_token, _, _ = claim(
        store,
        round_id,
        runners[0],
        parallelism=2,
        excluded=(first_run["miner_hotkey"],),
    )
    fingerprint = "sha256:" + "c" * 64
    first_identity = _uncertain_call(
        store,
        first_run,
        hash_lease_token(first_token),
        label="cursor-first",
        sequence=0,
        generation_id="gen-cursor-first",
        credential_fingerprint=fingerprint,
    )
    second_identity = _uncertain_call(
        store,
        second_run,
        hash_lease_token(second_token),
        label="cursor-second",
        sequence=1,
        generation_id="gen-cursor-second",
        credential_fingerprint=fingerprint,
    )

    first = store.list_openrouter_cost_reconciliations(round_id, limit=1)[0]
    second = store.list_openrouter_cost_reconciliations(
        round_id, after_entry_id=first["uncertain_entry_id"], limit=1
    )[0]
    wrapped = store.list_openrouter_cost_reconciliations(
        round_id, after_entry_id=second["uncertain_entry_id"], limit=1
    )[0]

    assert first["call_identity"] == first_identity
    assert second["call_identity"] == second_identity
    assert wrapped["call_identity"] == first_identity


def test_malformed_missing_identity_and_terminal_round_stay_unchanged(resources):
    store, _connect = resources
    round_id, _submission_id, run, token_hash = _round_and_run(store, "closed")
    malformed = _uncertain_call(
        store,
        run,
        token_hash,
        label="missing-id",
        sequence=0,
    )
    fingerprint = "sha256:" + "d" * 64
    valid = _uncertain_call(
        store,
        run,
        token_hash,
        label="closed-valid",
        sequence=1,
        generation_id="gen-closed",
        credential_fingerprint=fingerprint,
    )
    candidates = store.list_openrouter_cost_reconciliations(round_id, limit=20)
    assert [row["call_identity"] for row in candidates] == [valid]
    valid_candidate = candidates[0]
    assert store.cancel_round(round_id, "test_cancel")["status"] == "cancelled"

    result = store.reconcile_openrouter_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=valid,
        uncertain_entry_id=valid_candidate["uncertain_entry_id"],
        generation_id="gen-closed",
        credential_fingerprint=fingerprint,
        actual_microusd=10,
        cost_units="0.00001",
    )

    assert result["status"] == "stale"
    assert _settlements(store, valid) == []
    assert _settlements(store, malformed) == []
    assert store.list_openrouter_cost_reconciliations(round_id, limit=20) == []


def test_settlement_before_completion_preserves_the_normal_retry_claim(
    resources, database
):
    store, _connect = resources
    round_id = "arena-2026-09-12-retry"
    runners, _participants = open_round(
        store,
        round_id,
        participants=1,
        runners=2,
        prefix="retry",
        execution_cap_microusd=10_000_000,
    )
    run, token, _, _ = claim(store, round_id, runners[0])
    token_hash = hash_lease_token(token)
    assert not _has_recent_participation(database, runners[0])
    fingerprint = "sha256:" + "e" * 64
    generation_id = "gen-retry"
    identity = _uncertain_call(
        store,
        run,
        token_hash,
        label="retry",
        sequence=0,
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
    )
    candidate = store.list_openrouter_cost_reconciliations(
        round_id, run_id=run["run_id"]
    )[0]
    assert store.reconcile_openrouter_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        generation_id=generation_id,
        credential_fingerprint=fingerprint,
        actual_microusd=45,
        cost_units="0.000045",
    )["status"] == "settled"

    completed = store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        result={"terminal_status": "provider_error"},
        terminal_cause="provider_error",
        output_ref="",
    )
    assert completed["status"] == "failed"
    assert completed["confirmation_attempt"] == 2
    assert store.get_run(run["run_id"])["participation_accepted_at"] is None
    assert not _has_recent_participation(database, runners[0])
    retried, retry_token, _request_id, _request_hash = claim(
        store, round_id, runners[1]
    )
    assert retried["assignment_id"] == run["assignment_id"]
    assert retried["attempt"] == 2
    retried_run = store.get_run(retried["run_id"])
    assert retried_run["runner_hotkey"] == runners[1]
    assert retried_run["participation_accepted_at"] is None
    assert not _has_recent_participation(database, runners[1])
    accepted = store.complete_attempt(
        run_id=retried["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/%s/retry-accepted.json" % round_id,
    )
    assert accepted["status"] == "accepted"
    accepted_run = store.get_run(retried["run_id"])
    assert accepted_run["runner_hotkey"] == runners[1]
    assert accepted_run["participation_accepted_at"] is not None
    assert _has_recent_participation(database, runners[1])
    assert not _has_recent_participation(database, runners[0])
