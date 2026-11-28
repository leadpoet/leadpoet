"""A client retry across transports must not send a paid request twice."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import pytest

from lab_arena.lead_orders.vault_client import VaultTransportError
from lab_arena.tool_market import book, catalog
from lab_arena.tool_market.market import ToolMarketError
from tests.tool_market.tool_market_pg_test import PROFILE, database, ok, store, world


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


def test_a_duplicate_of_a_purchase_that_spent_its_tab_still_gets_its_call_ids(world):
    world.miner_key("replay-spent-supplier", 5_000)
    op = catalog.operation("harvestapi", "linkedin_profile")
    buyer, token = world.buyer(budget=book.charge_microusd(book.owed_microusd(op, op.max_units, 5_000)))
    body = dict(PROFILE, request_id=str(uuid.uuid4()))
    first = world.market.call(token, body)
    with pytest.raises(ToolMarketError) as duplicate:  # the tab has nothing left for a second reserve
        world.market.call(token, body)
    assert (duplicate.value.status, duplicate.value.code) == (409, "request_already_used")
    assert first["call_id"] in duplicate.value.detail["possible_call_ids"]


def test_an_error_after_a_purchase_is_claimed_says_so(world):
    world.miner_key("replay-limited-supplier", 5_000)
    world.vendor.script["replay-limited-supplier"] = [ok({}, status=429)]
    _, token = world.buyer()
    body = dict(PROFILE, request_id=str(uuid.uuid4()))
    with pytest.raises(ToolMarketError) as none_left:
        world.market.call(token, body)
    assert (none_left.value.status, none_left.value.code) == (503, "no_supply")
    assert none_left.value.detail["request_id_used"] is True
    world.vendor.script["replay-limited-supplier"] = [VaultTransportError("vault_unreachable", forwarded=True)]
    with pytest.raises(ToolMarketError) as lost:  # it may have reached the vendor
        world.market.call(token, dict(PROFILE, request_id=str(uuid.uuid4())))
    assert lost.value.status == 502 and lost.value.detail["request_id_used"] is True
