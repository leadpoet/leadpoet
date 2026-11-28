"""Price and consent for both higher-cost Harvest operations on the normal market path."""

from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market.market import ToolMarketError
from tests.tool_market.tool_market_pg_test import database, ok, store, world


@pytest.mark.parametrize(
    "op,params,ask,document,path,upstream",
    [
        (
            "search_leads",
            {"currentCompanies": "fixture", "seniorityLevelIds": "220,300", "page": 1},
            42_000,
            {"status": 200, "elements": [], "pagination": {"totalPages": 0}},
            "/linkedin/lead-search",
            {"currentCompanies": ["fixture"], "seniorityLevelIds": ["220,300"], "page": ["1"]},
        ),
        (
            "linkedin_profile_email",
            {"url": "https://www.linkedin.com/in/example", "main": "true"},
            8_000,
            {"status": 200, "element": {"id": "fixture", "emails": []}},
            "/linkedin/profile",
            {"query": ["https://www.linkedin.com/in/example"], "findEmail": ["true"], "main": ["true"]},
        ),
    ],
)
def test_supplier_must_offer_this_exact_premium_lookup(world, op, params, ask, document, path, upstream):
    credential = "premium-fixture-" + op
    miner, key = world.miner_key(credential, 1_000)
    _, tab = world.buyer()
    request = {"tool": "harvestapi", "operation": op, "params": params, "max_ask_microusd": ask}
    with pytest.raises(ToolMarketError, match="no_supply"):
        world.market.call(tab, request)
    world.market.post_ask(world.who(miner), {"key_id": key, "operation": op, "ask_microusd": ask})
    with pytest.raises(ToolMarketError, match="no_supply"):
        world.market.call(tab, dict(request, max_ask_microusd=ask - 1))
    assert not world.vendor.sent
    world.vendor.script[credential] = [ok(document)]
    result = world.market.call(tab, request)
    assert result["body"] == document and result["charge_microusd"] == ask * 11 // 10
    assert len(world.vendor.sent) == 1
    secret, method, url, body = world.vendor.sent[0]
    parts = urlsplit(url)
    assert (secret, method, parts.hostname, parts.path, body) == (
        credential,
        "GET",
        "api.harvestapi.io",
        path,
        b"",
    )
    assert parse_qs(parts.query) == upstream
    own = world.market.statement(miner)["keys"][0]
    assert own["owed_microusd"] == ask and own["calls"] == 1
