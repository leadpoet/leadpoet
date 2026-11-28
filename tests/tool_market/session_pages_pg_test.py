"""Paid pagination must retain the supplier that owns the vendor session."""
from uuid import uuid4
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.tool_market.tool_market_pg_test import database, store, world, ok
from lab_arena.tool_market.market import ToolMarketError


def test_lead_search_session_stays_on_first_supplier_after_price_change(world):
    a, key_a = world.miner_key("session-supplier-A", 10)
    b, key_b = world.miner_key("session-supplier-B", 12)
    def price(miner, key, ask):
        world.market.post_ask(world.who(miner), dict(key_id=key, operation="search_leads", ask_microusd=ask))
    price(a, key_a, 5000)
    price(b, key_b, 6000)
    for secret in ("session-supplier-A", "session-supplier-B"):
        world.vendor.script[secret] = [ok({"status":200,"elements":[]})]
    _, tab = world.buyer()
    session = str(uuid4())
    def page(n, token=tab):
        return world.market.call(token, {"tool":"harvestapi","operation":"search_leads",
              "params":{"currentJobTitles":"Owner","sessionId":session,"page":n}})
    with pytest.raises(ToolMarketError, match="session_page_one_required"):
        page(2)
    page(1)
    price(b, key_b, 4000)
    page(2)
    assert [x[0] for x in world.vendor.sent] == ["session-supplier-A"] * 2
    price(a, key_a, None)
    with pytest.raises(ToolMarketError, match="no_supply"):
        page(3)
    _, other = world.buyer()
    page(1, other)
    upstream_ids = [parse_qs(urlsplit(x[2]).query)["sessionId"][0] for x in world.vendor.sent]
    assert upstream_ids[0] == upstream_ids[1] != upstream_ids[2]
    assert session not in upstream_ids


def test_page_one_fails_over_and_the_session_pins_to_the_key_that_served_it(world):
    for secret, ask in (("session-limited", 5000), ("session-serving", 6000)):
        miner, key = world.miner_key(secret, 10)
        world.market.post_ask(world.who(miner), dict(key_id=key, operation="search_leads", ask_microusd=ask))
    world.vendor.script["session-limited"] = [ok({}, status=429)]
    world.vendor.script["session-serving"] = [ok({"status": 200, "elements": []})]
    _, tab = world.buyer()
    session = str(uuid4())
    def page(n):
        return world.market.call(tab, {"tool": "harvestapi", "operation": "search_leads",
              "params": {"currentJobTitles": "Owner", "sessionId": session, "page": n}})
    page(1)  # the cheaper key is rate-limited; the next one serves page one
    world.vendor.script["session-limited"] = [ok({"status": 200, "elements": []})]
    page(2)
    assert [x[0] for x in world.vendor.sent] == ["session-limited", "session-serving", "session-serving"]
