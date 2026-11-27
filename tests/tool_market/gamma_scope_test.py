"""Deployment identity is independent of the buyer and their cumulative spend."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from lab_arena.store import ArenaStoreError, PsycopgTransport
from lab_arena.tool_market.store import ToolMarketStore
from tests.tool_market.tool_market_pg_test import MIGRATIONS, hotkey
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


@pytest.fixture
def pg():
    yield from database_with_lab_arena_migration(MIGRATIONS)


def test_only_one_concurrent_gamma_identity_can_own_the_ledger(pg):
    psycopg2, dsn = pg
    store = ToolMarketStore(PsycopgTransport(lambda: psycopg2.connect(**dsn), role="lab_arena_service"))
    identities = [dict(genesis="0x" + "10" * 32, netuid=71, service=hotkey(), registered_at=740) for _ in range(2)]
    barrier = threading.Barrier(2)

    def bind(scope):
        barrier.wait(timeout=5)
        try:
            store.bind_gamma_source(**scope)
            return scope
        except ArenaStoreError:
            return None

    with ThreadPoolExecutor(2) as pool:
        winners = [s for s in pool.map(bind, identities) if s is not None]
    assert len(winners) == 1
    winner = winners[0]
    store.bind_gamma_source(**winner)
    reading = dict(hotkey=hotkey(), block_number=900, block_hash="0x" + "ab" * 32, spent_units=10**9, available=True)
    assert store.record_gamma_reading(**reading)["credit_microusd"] == 1_000_000
    for change in (dict(genesis="0x" + "11" * 32), dict(netuid=72), dict(service=hotkey()), dict(registered_at=800)):
        with pytest.raises(ArenaStoreError):
            store.bind_gamma_source(**dict(winner, **change))
    with pytest.raises(ArenaStoreError):
        store._rpc("tool_market_record_gamma_reading_v1", p_hotkey=reading["hotkey"], p_block_number=901,
                   p_block_hash="0x" + "cd" * 32, p_units=str(2 * 10**9), p_available=True,
                   p_genesis=winner["genesis"], p_netuid=71, p_service=hotkey(), p_registered_at=740)
    assert store.record_gamma_reading(**reading)["credit_microusd"] == 1_000_000
    store._t.close()


def test_old_unbound_credit_must_be_reconciled(pg):
    psycopg2, dsn = pg
    hk = hotkey()
    with psycopg2.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO public.tool_market_gamma_readings(hotkey,block_number,block_hash,spent_units,available)
                       VALUES(%s,901,%s,1000000000,true)""", (hk, "0x"+"ab"*32))
    store = ToolMarketStore(PsycopgTransport(lambda: psycopg2.connect(**dsn), role="lab_arena_service"))
    with pytest.raises(ArenaStoreError):
        store.bind_gamma_source(genesis="0x"+"11"*32, netuid=71, service=hotkey(), registered_at=740)
    store._t.close()


def test_read_and_purchase_check_the_same_finalized_subnet_incarnation(monkeypatch):
    from types import SimpleNamespace
    from lab_arena import chain as chain_module
    from lab_arena.tool_market.wiring import MarketGammaReader

    service = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    genesis = "0x" + "01" * 32
    head = chain_module.BlockRef(900, "0x" + "ab" * 32)
    calls, bindings = [], []
    state = {"NetworkRegisteredAt": 700, "GammaSpent": 10**9}

    def query(module, item, args, *, block_hash):
        calls.append((module, item, args, block_hash))
        return state[item]

    chain = SimpleNamespace(config=SimpleNamespace(netuid=71), client=SimpleNamespace(query=query),
                            finalized_head=lambda: head, finalized_block_hash=lambda _, head: genesis)
    cutover = SimpleNamespace(netuid=71, network_genesis_hash=genesis)
    store = SimpleNamespace(bind_gamma_source=lambda **k: bindings.append(k))
    reader = MarketGammaReader(chain, service, cutover, store)
    assert bindings == [dict(genesis=genesis, service=service, netuid=71, registered_at=700)]
    assert reader.read(service).spent_units == 10**9
    assert calls[-2:] == [("SubtensorModule", "NetworkRegisteredAt", [71], head.hash),
                          ("SubtensorModule", "GammaSpent", [71, service, service], head.hash)]
    monkeypatch.setattr(chain_module, "finalized_epoch_snapshot", lambda _: SimpleNamespace(
        current_block=head.number, block_hash=head.hash, settlement_epoch_id=lambda _: 7))
    assert reader.current_epoch() == 7

    state["NetworkRegisteredAt"] = 830
    with pytest.raises(chain_module.ArenaChainError, match="changed"):
        reader.read(service)
    with pytest.raises(chain_module.ArenaChainError, match="changed"):
        reader.current_epoch()
    assert [c[1] for c in calls].count("GammaSpent") == 1

    chain.finalized_block_hash = lambda _, head: "0x" + "99" * 32
    with pytest.raises(chain_module.ArenaChainError, match="differs"):
        MarketGammaReader(chain, service, cutover, store)
    chain.finalized_block_hash = lambda _, head: genesis
    del state["NetworkRegisteredAt"]
    with pytest.raises(chain_module.ArenaChainError, match="unavailable"):
        MarketGammaReader(chain, service, cutover, store)
    assert len(bindings) == 1
