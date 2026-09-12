"""PostgreSQL races and security for cancellation-time call settlement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, open_round, sha


MIGRATION = "223-lab-arena-cancelled-call-late-settlement.sql"


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


def _terminal(status: int = 200, *, marker: str = "ok", provider_cost=None):
    document = {"status": status, "headers": {}, "body_b64": marker}
    if provider_cost is not None:
        document["provider_cost"] = provider_cost
    return document


def _dispatched_call(store: ArenaStore, label: str, *, amount=1_000_000, sequence=0):
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
    token_hash = hash_lease_token(token)
    identity = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=sequence,
        operation_id="openrouter.chat",
        request_hash=sha(label),
    )
    reserved = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=amount,
        call_doc={"model": "fixture"},
    )
    assert reserved["status"] == "reserved", reserved
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    return {
        "round_id": round_id,
        "submission_id": participants[0]["submission_id"],
        "run": run,
        "token_hash": token_hash,
        "identity": identity,
        "amount": amount,
    }


def _spend(connect, submission_id: str) -> int:
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__submission_kind_spend(%s, 'execute')",
            (submission_id,),
        )
        return int(cursor.fetchone()[0])


def _kinds(store: ArenaStore, identity: str):
    return [row["entry_kind"] for row in store.list_ledger(call_identity=identity)]


def test_settlement_that_locks_first_remains_the_only_terminal(resources):
    store, connect = resources
    call = _dispatched_call(store, "settlefirst", amount=900_000)
    terminal = _terminal(marker="c2V0dGxlLWZpcnN0")
    settled = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=123_456,
        terminal_response=terminal,
    )
    assert settled["status"] == "settled"
    assert "late_reconciliation" not in settled
    assert store.cancel_round(call["round_id"], "test_cancel")["status"] == "cancelled"
    assert _kinds(store, call["identity"]) == ["reservation", "dispatch", "settlement"]
    assert _spend(connect, call["submission_id"]) == 123_456


def test_cancellation_that_locks_first_accepts_exact_late_reply_once(resources):
    store, connect = resources
    call = _dispatched_call(store, "cancelfirst", amount=800_000)
    assert store.cancel_round(call["round_id"], "test_cancel")["status"] == "cancelled"
    assert _kinds(store, call["identity"]) == ["reservation", "dispatch", "uncertain"]
    assert _spend(connect, call["submission_id"]) == 800_000

    terminal = _terminal(marker="Y2FuY2VsLWZpcnN0")
    settled = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=234_567,
        terminal_response=terminal,
    )
    assert settled["status"] == "settled"
    assert settled["late_reconciliation"] is True
    assert settled["released_microusd"] == 565_433
    assert _kinds(store, call["identity"]) == [
        "reservation",
        "dispatch",
        "uncertain",
        "settlement",
    ]
    assert _spend(connect, call["submission_id"]) == 234_567
    costs = store.submission_costs(call["submission_id"])
    assert sum(row["settled_microusd"] for row in costs["providers"]) == 234_567
    assert sum(row["reserved_or_uncertain_microusd"] for row in costs["providers"]) == 0
    assert sum(row["uncertain_calls"] for row in costs["providers"]) == 0

    replay = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=234_567,
        terminal_response=terminal,
    )
    assert replay["status"] == "settled"
    assert replay["idempotent"] is True
    assert replay["late_reconciliation"] is True
    conflict = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=234_568,
        terminal_response=terminal,
    )
    assert conflict == {"status": "conflict", "call_identity": call["identity"]}
    assert _kinds(store, call["identity"]).count("settlement") == 1
    assert _spend(connect, call["submission_id"]) == 234_567


def test_known_cost_error_can_settle_but_unpriced_error_cannot(resources):
    store, connect = resources
    call = _dispatched_call(store, "knownerror", amount=700_000)
    second_identity = contracts.provider_call_identity(
        attempt=call["run"]["attempt"],
        assignment_id=call["run"]["assignment_id"],
        icp_position=call["run"]["icp_position"],
        action_sequence=1,
        operation_id="openrouter.chat",
        request_hash=sha("unknown-error"),
    )
    assert store.reserve_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=second_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=600_000,
        call_doc={"model": "fixture"},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=second_identity,
    )["status"] == "dispatched"
    assert store.cancel_round(call["round_id"], "test_cancel")["status"] == "cancelled"

    cost = {
        "basis": "openrouter_usage_cost",
        "units": "0.000321",
        "unit_name": "usd",
        "operation": "openrouter.chat",
        "request_id": "generation-fixture",
    }
    known = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=321,
        terminal_response=_terminal(503, marker="ZXJyb3I=", provider_cost=cost),
    )
    assert known["status"] == "settled" and known["late_reconciliation"] is True
    unknown = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=second_identity,
        actual_microusd=0,
        terminal_response=_terminal(503, marker="dW5rbm93bg=="),
    )
    assert unknown["status"] == "stale"
    assert _kinds(store, second_identity)[-1] == "uncertain"
    assert _spend(connect, call["submission_id"]) == 600_321


def test_wrong_identity_token_and_worker_uncertainty_stay_closed(resources):
    store, connect = resources
    call = _dispatched_call(store, "security", amount=500_000)
    worker_identity = contracts.provider_call_identity(
        attempt=call["run"]["attempt"],
        assignment_id=call["run"]["assignment_id"],
        icp_position=call["run"]["icp_position"],
        action_sequence=1,
        operation_id="openrouter.chat",
        request_hash=sha("worker-uncertain"),
    )
    assert store.reserve_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=worker_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=400_000,
        call_doc={},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=worker_identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=worker_identity,
        call_doc={"reason": "settle_failure", "provider_status": 200},
    )["status"] == "uncertain"
    assert store.cancel_round(call["round_id"], "test_cancel")["status"] == "cancelled"

    wrong_token = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=hash_lease_token("wrong-token"),
        call_identity=call["identity"],
        actual_microusd=1,
        terminal_response=_terminal(),
    )
    assert wrong_token["status"] == "stale"
    wrong_identity = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity="sha256:" + "f" * 64,
        actual_microusd=1,
        terminal_response=_terminal(),
    )
    assert wrong_identity["status"] == "stale"
    worker = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=worker_identity,
        actual_microusd=1,
        terminal_response=_terminal(),
    )
    assert worker["status"] == "stale"
    assert _kinds(store, worker_identity)[-1] == "uncertain"
    assert _spend(connect, call["submission_id"]) == 900_000


def test_recovery_generation_change_rejects_old_reply_and_acl_is_service_only(resources):
    store, connect = resources
    call = _dispatched_call(store, "restart", amount=300_000)
    assert store.cancel_round(call["round_id"], "test_cancel")["status"] == "cancelled"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal")
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET stage_generation = stage_generation - 1
            WHERE run_id = %s
            """,
            (call["run"]["run_id"],),
        )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal")
        connection.commit()
    stale = store.settle_call(
        run_id=call["run"]["run_id"],
        lease_token_hash=call["token_hash"],
        call_identity=call["identity"],
        actual_microusd=2,
        terminal_response=_terminal(),
    )
    assert stale["status"] == "stale"
    assert _kinds(store, call["identity"])[-1] == "uncertain"

    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              has_function_privilege(
                'lab_arena_service',
                'public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)',
                'EXECUTE'
              ),
              has_function_privilege(
                'authenticated',
                'public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)',
                'EXECUTE'
              ),
              has_table_privilege('lab_arena_service', 'public.lab_arena_ledger', 'INSERT'),
              (SELECT tgenabled = 'O' FROM pg_catalog.pg_trigger
               WHERE tgrelid = 'public.lab_arena_runs'::REGCLASS
                 AND tgname = 'lab_arena_runs_terminal')
            """
        )
        assert cursor.fetchone() == (True, False, False, True)


def test_cancel_and_settle_concurrency_has_no_unaccounted_outcome(resources):
    store, connect = resources
    call = _dispatched_call(store, "concurrent", amount=200_000)
    terminal = _terminal(marker="cmFjZQ==")

    def settle():
        local = ArenaStore(PsycopgTransport(connect), lease_ttl_seconds=120)
        try:
            return local.settle_call(
                run_id=call["run"]["run_id"],
                lease_token_hash=call["token_hash"],
                call_identity=call["identity"],
                actual_microusd=12_345,
                terminal_response=terminal,
            )
        finally:
            local.close()

    def cancel():
        local = ArenaStore(PsycopgTransport(connect), lease_ttl_seconds=120)
        try:
            return local.cancel_round(call["round_id"], "concurrent_cancel")
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        settle_future = pool.submit(settle)
        cancel_future = pool.submit(cancel)
        settled = settle_future.result(timeout=10)
        cancelled = cancel_future.result(timeout=10)
    assert settled["status"] == "settled"
    assert cancelled["status"] == "cancelled"
    assert _kinds(store, call["identity"]) in (
        ["reservation", "dispatch", "settlement"],
        ["reservation", "dispatch", "uncertain", "settlement"],
    )
    assert _spend(connect, call["submission_id"]) == 12_345


def test_migration_indexes_and_replay_are_stable(resources):
    _store, connect = resources
    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text(encoding="utf-8"))
        cursor.execute(
            """
            SELECT
              pg_catalog.to_regclass('public.lab_arena_ledger_terminal_uq') IS NULL,
              pg_catalog.to_regclass('public.lab_arena_ledger_settlement_uq') IS NOT NULL,
              pg_catalog.to_regclass(
                'public.lab_arena_ledger_nonsettlement_terminal_uq'
              ) IS NOT NULL
            """
        )
        assert cursor.fetchone() == (True, True, True)
