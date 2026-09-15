"""Real-ledger proof for ambiguous reservation response recovery."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import broker as br
from lab_arena.store import (
    ArenaStore,
    ArenaStoreUnavailable,
    PsycopgTransport,
    hash_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_broker import (
    FakeTransport,
    HOST_KEYS,
    price_table,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    commit_round,
    frozen_participants,
    hotkey,
    round_config,
    stage_positions,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _store(database) -> ArenaStore:
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _open_round(store, round_id: str, *, prefix: str, stage_1_icps: int, cap=5_000_000):
    runners = [hotkey(prefix + "-runner")]
    config = round_config(
        round_id,
        runners,
        stage_1_icps=stage_1_icps,
        execution_cap_microusd=cap,
    )
    config["schedule"]["submission_cutoff"] = (
        datetime.now(timezone.utc) + timedelta(minutes=30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert store.create_round(round_id, config)["status"] == "created"
    participants = frozen_participants(store, round_id, 1, prefix=prefix)
    commit_round(store, round_id, participants)
    assert store.open_stage(round_id, 1, participants, stage_positions(1))[
        "status"
    ] == "ok"
    return runners


def _context(run, token: str, round_id: str) -> br.RunContext:
    return br.RunContext(
        run_id=run["run_id"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        lease_token_hash=hash_lease_token(token),
        miner_hotkey=run["miner_hotkey"],
        submission_id=run["submission_id"],
        stage=run["stage"],
        kind=run["kind"],
        attempt=run["attempt"],
        round_id=round_id,
    )


def _broker(store, transport) -> br.Broker:
    return br.Broker(
        store=store,
        key_for=lambda provider: HOST_KEYS[provider],
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        price_table=price_table(),
        transport=transport,
    )


def _execute(broker: br.Broker, context: br.RunContext, action: int, query: str):
    return broker.execute(
        context,
        operation_id="deepline.execute",
        parameters={"tool": "exa_search", "payload": {"query": query}},
        action_sequence=action,
        timeout_ms=5_000,
    )


class _ResponseLossStore:
    def __init__(self, inner: ArenaStore, *, commit_first: bool):
        self.inner = inner
        self.commit_first = commit_first
        self.reserve_attempts = 0
        self.committed_reservation = None

    def reserve_call(self, **kwargs):
        self.reserve_attempts += 1
        if self.reserve_attempts == 1:
            if self.commit_first:
                result = self.inner.reserve_call(**kwargs)
                rows = self.inner.list_ledger(
                    call_identity=kwargs["call_identity"], limit=64
                )
                self.committed_reservation = rows[0]
            raise ArenaStoreUnavailable("synthetic reserve response loss")
        return self.inner.reserve_call(**kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_committed_dynamic_reservation_is_recovered_settled_and_budget_reused(
    database,
):
    store = _store(database)
    round_id = "arena-2099-09-15-resploss"
    runners = _open_round(
        store,
        round_id,
        prefix="reserve-response-loss",
        stage_1_icps=2,
        cap=50_000_000,
    )
    leases = [
        claim(store, round_id, runners[0], parallelism=2, ceiling=2)[:2]
        for _ in range(2)
    ]
    wrapped = _ResponseLossStore(store, commit_first=True)
    transport = FakeTransport([(200, {"results": []}), (200, {"results": []})])
    broker = _broker(wrapped, transport)

    first = _execute(broker, _context(*leases[0], round_id), 0, "first")
    assert first.status == 200 and first.call["outcome"] == "settled"
    assert first.call["reserved_microusd"] == 50_000_000
    assert first.call["actual_microusd"] == 0
    assert wrapped.reserve_attempts == 2
    assert wrapped.committed_reservation["amount_microusd"] == 50_000_000
    first_rows = store.list_ledger(call_identity=first.call["call_identity"])
    assert [row["entry_kind"] for row in first_rows] == [
        "reservation", "dispatch", "settlement",
    ]
    assert first_rows[-1]["amount_microusd"] == 0

    second = _execute(broker, _context(*leases[1], round_id), 0, "second")
    assert second.status == 200 and second.call["outcome"] == "settled"
    assert second.call["reserved_microusd"] == 50_000_000
    assert leases[1][0]["run_id"] != leases[0][0]["run_id"]
    assert leases[1][0]["icp_position"] != leases[0][0]["icp_position"]
    assert second.call["call_identity"] != first.call["call_identity"]
    assert len(transport.sent) == 2
    store.close()


def test_absent_first_reservation_creates_exactly_one_reservation(database):
    store = _store(database)
    round_id = "arena-2099-09-15-resprollback"
    runners = _open_round(
        store, round_id, prefix="reserve-rollback", stage_1_icps=1,
    )
    run, token, _, _ = claim(store, round_id, runners[0])
    wrapped = _ResponseLossStore(store, commit_first=False)
    transport = FakeTransport([(200, {"results": []})])

    result = _execute(_broker(wrapped, transport), _context(run, token, round_id), 0, "once")
    assert result.status == 200 and wrapped.reserve_attempts == 2
    rows = store.list_ledger(call_identity=result.call["call_identity"])
    assert [row["entry_kind"] for row in rows] == [
        "reservation", "dispatch", "settlement",
    ]
    assert len(transport.sent) == 1
    store.close()


def test_concurrent_recovery_has_one_atomic_dispatch_and_one_provider_post(database):
    setup = _store(database)
    round_id = "arena-2099-09-15-resprace"
    runners = _open_round(
        setup, round_id, prefix="reserve-race", stage_1_icps=1,
    )
    run, token, _, _ = claim(setup, round_id, runners[0])
    context = _context(run, token, round_id)
    committed = threading.Event()
    reservations_ready = threading.Barrier(2)
    dispatches_ready = threading.Barrier(2)
    first_lock = threading.Lock()
    first = True

    class RacingStore:
        def __init__(self, inner, *, waits_for_commit):
            self.inner = inner
            self.waits_for_commit = waits_for_commit

        def reserve_call(self, **kwargs):
            nonlocal first
            if self.waits_for_commit:
                assert committed.wait(timeout=5)
            result = self.inner.reserve_call(**kwargs)
            with first_lock:
                lose_response = first
                if first:
                    first = False
            if lose_response:
                committed.set()
                raise ArenaStoreUnavailable("synthetic reserve response loss")
            reservations_ready.wait(timeout=5)
            return result

        def mark_dispatched(self, **kwargs):
            result = self.inner.mark_dispatched(**kwargs)
            dispatches_ready.wait(timeout=5)
            return result

        def __getattr__(self, name):
            return getattr(self.inner, name)

    stores = [
        RacingStore(_store(database), waits_for_commit=False),
        RacingStore(_store(database), waits_for_commit=True),
    ]
    transports = [FakeTransport([(200, {"results": []})]) for _ in stores]
    results = []
    failures = []

    def run_one(index):
        try:
            results.append(_execute(
                _broker(stores[index], transports[index]), context, 0, "same"
            ))
        except Exception as exc:  # assertion below retains the exact failure
            failures.append(exc)

    threads = [threading.Thread(target=run_one, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures and all(not thread.is_alive() for thread in threads)
    assert sorted(result.status for result in results) == [200, 409]
    assert sum(len(transport.sent) for transport in transports) == 1
    identity = results[0].call["call_identity"]
    rows = setup.list_ledger(call_identity=identity)
    assert [row["entry_kind"] for row in rows] == [
        "reservation", "dispatch", "settlement",
    ]
    for store in stores:
        store.inner.close()
    setup.close()
