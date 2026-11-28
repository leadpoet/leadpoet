"""Exercise signed operation offers and the transactional price check."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import pytest

from lab_arena.tool_market import market
from tests.tool_market.tool_market_pg_test import PROFILE, database, store, world


def test_new_sku_needs_its_own_offer_and_withdrawal_sticks(world):
    miner, key = world.miner_key("sku-opt-in", 10_000)
    _, token = world.buyer()
    job = dict(tool="harvestapi", operation="get_job", params={"jobId": "3547788307"})
    with pytest.raises(market.ToolMarketError, match="no_supply"):
        world.market.call(token, job)
    assert not world.vendor.sent
    assert world.market.order_book("harvestapi", "get_job")["asks"] == []

    def price(op, ask):
        world.market.post_ask(world.who(miner), dict(key_id=key, operation=op, ask_microusd=ask))

    price("get_job", 500)
    assert world.market.call(token, job)["charge_microusd"] == 550
    assert world.market.call(token, PROFILE)["charge_microusd"] == 11_000
    price("linkedin_profile", 12_000)
    assert world.market.call(token, PROFILE)["charge_microusd"] == 13_200
    price("linkedin_profile", None)
    assert world.market.order_book("harvestapi", "linkedin_profile")["asks"] == []
    with pytest.raises(market.ToolMarketError, match="no_supply"):
        world.market.call(token, PROFILE)
    assert world.market.order_book("harvestapi", "linkedin_company")["asks"][0]["ask_microusd"] == 10_000
    price("get_job", None)
    with pytest.raises(market.ToolMarketError, match="no_supply"):
        world.market.call(token, job)
    assert len(world.vendor.sent) == 3


def test_reserve_sees_concurrent_sku_reprice_before_vendor_dispatch(world, database):
    pg, dsn = database
    miner, key = world.miner_key("sku-price-race", 20_000)
    _, token = world.buyer()
    world.market.post_ask(world.who(miner), dict(key_id=key, operation="get_job", ask_microusd=500))
    quote = world.market.order_book("harvestapi", "get_job")["asks"][0]
    started = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        with pg.connect(**dsn) as tx, tx.cursor() as q:
            q.execute("SET LOCAL ROLE lab_arena_service")
            q.execute("SELECT public.tool_market_post_ask_v1(%s,%s,1000,'get_job')", (key, miner))

            def reserve_old_price():
                started.set()
                return world.store.reserve_call(call_id=str(uuid.uuid4()), token_hash=market.token_hash(token),
                    tool="harvestapi", operation="get_job", key_id=key, ask_microusd=quote["ask_microusd"],
                    owed_cap_microusd=quote["ask_microusd"])

            future = pool.submit(reserve_old_price)
            assert started.wait(2)
            stop = time.monotonic() + 4
            while time.monotonic() < stop:
                q.execute("SELECT EXISTS (SELECT FROM pg_locks WHERE NOT granted AND locktype='advisory' "
                          "AND pid<>pg_backend_pid())")
                if q.fetchone()[0]:
                    break
                time.sleep(.01)
            else:
                pytest.fail("reservation never reached the locked key")
        assert future.result(timeout=3) == "ask_changed"
    assert not world.vendor.sent and world.market.tab(token)["calls"] == 0



def test_general_prices_cover_only_the_original_market_operations(world):
    from lab_arena.tool_market import book, catalog

    import dataclasses
    world.market._config = dataclasses.replace(world.market._config, tools=frozenset(catalog.TOOLS))
    for tool, legacy in book.GENERAL_ASK_OPERATIONS.items():
        _, key_id = world.miner_key("legacy-scope-" + tool, 30_000, tool=tool)
        for t, name in catalog.OPERATIONS:
            if t == tool:
                rows = world.store.book(tool=t, operation=name, terms_sha256=world.terms)
                assert any(r["key_id"] == key_id for r in rows) == (name in legacy), (t, name)
    assert world.market.order_book("harvestapi", "get_job")["operations"] == ["get_job"]
