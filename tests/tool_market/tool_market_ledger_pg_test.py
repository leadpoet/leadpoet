"""Exercise authoritative reservations, settlements and balance reads on PostgreSQL."""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena.store import ArenaStoreError
from lab_arena.tool_market import market
from tests.tool_market.tool_market_pg_test import (
    PROFILE, USD, database, hotkey, store, world,
)


def reserve(world, token, key):
    call_id = str(uuid.uuid4())
    status = world.store.reserve_call(
        call_id=call_id, token_hash=market.token_hash(token), tool="harvestapi",
        operation="linkedin_profile", key_id=key, ask_microusd=10_000,
        owed_cap_microusd=10_000)
    return call_id, status


def test_reserve_rechecks_key_health_after_book_read(world):
    miner, key = world.miner_key("stale-health", 10_000)
    _, token = world.buyer()
    assert world.market.order_book("harvestapi")["asks"]
    world.store.record_key_event(key_id=key, status="invalid", reason="unauthorized")
    assert reserve(world, token, key)[1] == "ask_changed"
    assert world.market.statement(miner)["keys"][0]["calls"] == 0


def test_exact_settlement_replay_and_miner_cap(world):
    miner, key = world.miner_key("settlement-cap", 10_000)
    _, token = world.buyer()
    call_id, status = reserve(world, token, key)
    assert status == "reserved"
    answer = dict(call_id=call_id, status="settled", http_status=200, units=1,
                  owed_microusd=10_000, charge_microusd=11_000, epoch=7)
    # The buyer's charge cap is larger than the miner's oweable maximum.
    with pytest.raises(ArenaStoreError):
        world.store.settle_call(**dict(answer, owed_microusd=11_000))
    assert world.store.settle_call(**answer) == "settled"
    assert world.store.settle_call(**answer) == "exists"
    with pytest.raises(ArenaStoreError):
        world.store.settle_call(**dict(answer, epoch=8))
    assert world.market.statement(miner)["keys"][0]["owed_microusd"] == 10_000


def test_conflicting_finalized_read_never_adds_credit(world):
    hk = hotkey()
    same = dict(hotkey=hk, block_number=8, block_hash="0x" + "a0" * 32,
                spent_units=100_000, available=True)
    assert world.store.record_gamma_reading(**same)["balance_microusd"] == 100
    for conflict in ({"spent_units": 200_000}, {"block_hash": "0x" + "a1" * 32}):
        assert world.store.record_gamma_reading(**dict(same, **conflict))["status"] == "reading_conflict"
    assert world.store.record_gamma_reading(**same)["balance_microusd"] == 100


def test_simultaneous_calls_cannot_exceed_tab_balance(world):
    miner, key = world.miner_key("parallel-reserve", 10_000)
    buyer, token = world.buyer(budget=11_000)
    barrier = threading.Barrier(2)

    def invoke(_):
        barrier.wait(timeout=5)
        try:
            return world.market.call(token, PROFILE)["charge_microusd"]
        except market.ToolMarketError as e:
            return e.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, range(2)))
    assert sorted(results, key=str) == [11_000, "tab_budget"]
    assert len(world.vendor.sent) == 1
    world.market.close_tab(token)
    assert world.market.balance(buyer)["balance_microusd"] == USD - 11_000
    assert world.market.statement(miner)["keys"][0]["owed_microusd"] == 10_000


def test_accrual_report_binds_call_to_one_supplier_and_gross_fee(world, database):
    from pathlib import Path

    miner, _ = world.miner_key("accrual-report", 10_000)
    _, token = world.buyer()
    call = world.market.call(token, PROFILE)
    psycopg2, dsn = database
    sql = (Path(__file__).parents[2] / "scripts" / "tool_market_accruals.sql").read_text()
    with psycopg2.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET LOCAL ROLE lab_arena_owner")
        cur.execute(sql.replace(":'epoch'", "%s"), (7,))
        matches = [r for r in cur.fetchall() if str(r[0]) == call["call_id"]]
    assert len(matches) == 1
    assert matches[0][1:10] == (7, miner, "harvestapi", "linkedin_profile", 10_000, 1, 11_000, 10_000, 1_000)


def test_registered_miner_lists_and_serves_on_its_signed_attestation(world):
    miner, key = world.miner_key("self-service", 5_000)
    _, token = world.buyer()
    assert world.market.call(token, PROFILE)["charge_microusd"] == 5500
    assert world.market.statement(miner)["keys"][0]["owed_microusd"] == 5000
    world.market.revoke_key(world.who(miner), key)
    assert world.market.order_book("harvestapi")["asks"] == []
