"""Concurrent provider-call cycles across runs, submissions, and claims on
disposable PostgreSQL: no deadlock may go unrecovered (labarena.md 18.2, 18.3)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import LAB_ARENA_MIGRATION, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import claim, make_event, open_round, sha


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration((LAB_ARENA_MIGRATION,))


def test_concurrent_call_cycles_and_claims_complete_without_unrecovered_deadlock(database):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    setup = ArenaStore(PsycopgTransport(connect, pool_size=4))
    round_id = "arena-2026-09-02-st"
    runners, parts = open_round(setup, round_id, participants=4, runners=3, prefix="st", per_icp_cap=5_000_000, ceiling_1=200_000_000)
    stores = [ArenaStore(PsycopgTransport(connect, pool_size=4)) for _ in range(3)]
    errors = []
    lock = threading.Lock()
    completed = []

    def worker(index: int) -> None:
        store = stores[index % 3]
        runner = runners[index % 3]
        while True:
            response, token, _, _ = claim(store, round_id, runner, parallelism=8, ceiling=8)
            if response["status"] != "leased":
                return
            lease_hash = hash_lease_token(token)
            run_id = response["run_id"]
            try:
                cursor, head = int(response["event_cursor"]), str(response["event_head_hash"])
                for sequence in range(3):
                    identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=sequence, operation_id="exa.search", request_hash=sha("%s-%d" % (run_id, sequence)))
                    reserved = store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="exa.search", provider="exa", funding_source="tao", amount_microusd=5000, call_doc={})
                    assert reserved["status"] == "reserved", reserved
                    assert store.mark_dispatched(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity)["status"] == "dispatched"
                    event = make_event(response, cursor, head, event_type="provider_call")
                    settled = store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, actual_microusd=5000, terminal_response={"status": 200}, event=event)
                    assert settled["status"] == "settled", settled
                    cursor, head = int(settled["event_cursor"]), str(settled["event_head_hash"])
                    appended = store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[make_event(response, cursor, head, event_type="stdout")])
                    cursor, head = int(appended["event_cursor"]), str(appended["event_head_hash"])
                done = store.complete_attempt(run_id=run_id, lease_token_hash=lease_hash, receipt={"receipt_hash": sha("rc" + run_id)}, receipt_hash=sha("rc" + run_id), terminal_cause="accepted", output_hash=sha("o" + run_id), output_ref="ref", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
                assert done["status"] == "accepted", done
                with lock:
                    completed.append(run_id)
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                with lock:
                    errors.append("%s: %s" % (type(exc).__name__, str(exc)[:3000]))
                return

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))
    retries = sum(store._transport.deadlock_retries for store in stores)
    details = [store._transport.last_deadlock_detail for store in stores if store._transport.last_deadlock_detail]
    for error in errors[:2]:
        print("STRESS-ERROR:", error)
    assert errors == [], (len(errors), retries)
    assert len(completed) == 80
    runs = setup.list_runs(round_id, stage=1)
    assert all(run["status"] == "accepted" for run in runs) and len(runs) == 80
    ledger = setup.list_ledger()
    heads = {}
    for entry in ledger:
        if entry.get("round_id") == round_id and entry.get("call_identity"):
            heads[entry["call_identity"]] = entry["entry_kind"]
    assert len(heads) == 240 and set(heads.values()) == {"settlement"}
    for part in parts:
        account = setup.get_account(part["miner_hotkey"])
        assert account["balance_microusd"] == 5_000_000 - 60 * 5000
    print("deadlock retries:", retries, details[:1])
    for store in stores + [setup]:
        store.close()
