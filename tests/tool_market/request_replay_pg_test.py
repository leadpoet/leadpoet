"""A client retry across transports must not send a paid request twice."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import pytest

from lab_arena.tool_market.market import ToolMarketError
from tests.tool_market.tool_market_pg_test import PROFILE, database, store, world


def test_concurrent_and_lost_http_reply_share_one_purchase(world):
    world.miner_key("fixture-supplier", 5_000)
    buyer, token = world.buyer()
    body = dict(PROFILE, request_id=str(uuid.uuid4()))
    start = threading.Barrier(2)
    def submit():
        start.wait(5)
        try:
            return world.market.call(token, body)
        except ToolMarketError as err:
            assert err.code == "request_already_used" and err.status == 409
            return err.detail
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    completed, = [r for r in results if "call_id" in r]
    duplicate, = [r for r in results if "possible_call_ids" in r]
    assert completed["call_id"] in duplicate["possible_call_ids"]
    assert len(world.vendor.sent) == 1
    with pytest.raises(ToolMarketError, match="request_already_used"):
        world.market.call("", body, world.who(buyer))
    assert len(world.vendor.sent) == 1
    _, other = world.buyer()
    assert world.market.call(other, body)["call_id"] != completed["call_id"]
    assert len(world.vendor.sent) == 2


def test_a_purchase_refused_before_any_reserve_keeps_its_request_id(world):
    world.miner_key("replay-budget-supplier", 5_000)
    buyer, token = world.buyer(budget=1)
    body = dict(PROFILE, request_id=str(uuid.uuid4()))
    with pytest.raises(ToolMarketError, match="tab_budget"):
        world.market.call(token, body)
    bigger = world.market.open_tab(world.who(buyer), {"budget_microusd": 100_000, "seconds": 600})["token"]
    assert world.market.call(bigger, body)["status"] == 200  # the same purchase on a funded tab
    assert len(world.vendor.sent) == 1
