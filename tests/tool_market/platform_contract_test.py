"""Lead Search raw-v2 contracts from the actual platform call shapes."""

from urllib.parse import parse_qs, urlsplit

import json

import pytest

from lab_arena.tool_market import catalog, platform
from lab_arena.tool_market.market import ToolMarket, ToolMarketError
from lab_arena.tool_market.config import ToolMarketConfig


@pytest.mark.parametrize(
    "payload,query,price_key",
    [
        (
            {"url": "https://www.linkedin.com/in/ada", "main": "true"},
            {"query": ["https://www.linkedin.com/in/ada"], "main": ["true"]},
            "harvestapi_get_profile.main",
        ),
        (
            {"url": "https://www.linkedin.com/in/ada", "main": False},
            {"query": ["https://www.linkedin.com/in/ada"]},
            "harvestapi_get_profile",
        ),
    ],
)
def test_harvest_profile_version_remains_exact(payload, query, price_key):
    tool, name, params, key = platform.platform_request("harvestapi_get_profile", payload)
    assert key == price_key
    assert parse_qs(urlsplit(catalog.operation(tool, name).build(params)[0]).query) == query


@pytest.mark.parametrize(
    "op,payload",
    [
        ("harvestapi_get_profile", {"url": "person", "findEmail": "maybe"}),
        ("harvestapi_get_company", {"universalName": "acme", "url": "different"}),
        ("harvestapi_get_company", {"search": "Acme Corp"}),
        ("prospeo_enrich_person", {"linkedin_url": "person", "enrich_mobile": True}),
        ("findymail_find_from_name", {"name": "A", "domain": "example.org", "webhook_url": "callback"}),
    ],
)
def test_unsupported_platform_variants_cannot_spend(op, payload):
    with pytest.raises(catalog.CatalogError):
        tool, name, params, _ = platform.platform_request(op, payload)
        catalog.operation(tool, name).build(params)


def test_private_funded_company_lookup_serves_site_parsers_and_its_actual_charge():
    market = object.__new__(ToolMarket)
    market._config = ToolMarketConfig(platform_max_asks={"harvestapi_get_company": 1800})
    seen = []
    raw = {"status": 200, "element": {"universalName": "acme", "id": "1"}}

    def call(token, body, arrived=None):
        seen.append((token, body))
        return {"status": 200, "call_id": "id", "body": raw, "charge_microusd": 1980}

    market.call = call
    result = market.platform_call("token", "harvestapi_get_company", {"payload": {"universalName": "acme"}})
    assert result["status"] == "completed" and result["billing"]["cost_usd"] == 0.00198
    assert result["toolResponse"]["rawV2"] == raw
    assert "result" not in result
    assert seen == [
        (
            "token",
            {
                "tool": "harvestapi",
                "operation": "linkedin_company",
                "params": {"url": "acme"},
                "private": True,
                "max_ask_microusd": 1800,
            },
        )
    ]
    with pytest.raises(ToolMarketError) as unavailable:
        market.platform_call("token", "harvestapi_get_profile", {"payload": {"url": "ada"}})
    assert unavailable.value.status == 404 and len(seen) == 1


@pytest.mark.parametrize(
    "alias,payload,expected",
    [
        (
            "findymail_find_from_name",
            {"name": "Ada", "domain": "example.test"},
            {"name": "Ada", "domain": "example.test"},
        ),
        (
            "findymail_find_from_business_profile",
            {"linkedin_url": "https://linkedin.com/in/ada"},
            {"linkedin_url": "https://linkedin.com/in/ada"},
        ),
        (
            "prospeo_enrich_person",
            {"full_name": "Ada", "company_website": "example.test", "enrich_mobile": False},
            {"data": {"full_name": "Ada", "company_website": "example.test"}},
        ),
    ],
)
def test_email_discovery_wire_matches_platform_and_vendor(alias, payload, expected):
    tool, name, params, price_key = platform.platform_request(alias, payload)
    assert price_key == alias
    _, _, body = catalog.operation(tool, name).build(params)
    assert json.loads(body) == expected


def test_default_result_view_and_raw_v2_keep_provider_errors_distinct():
    market = object.__new__(ToolMarket)
    market._config = ToolMarketConfig(platform_max_asks={"prospeo_enrich_person": 1000})
    request = {"payload": {"linkedin_url": "example"}}
    missed = {"error": True, "error_code": "NO_MATCH"}

    def call(token, body, arrived=None):
        return {"status": 400, "call_id": "1", "body": missed, "charge_microusd": 0}

    market.call = call
    assert market.platform_call("tab", "prospeo_enrich_person", request, raw=False)["result"] == missed
    assert market.platform_call("tab", "prospeo_enrich_person", request)["status"] == "no_result"
    missed["error_code"] = "INTERNAL_ERROR"
    error = market.platform_call("tab", "prospeo_enrich_person", request)
    assert error["status"] == "failed" and not error["toolResponse"] and "result" not in error


def test_platform_ceiling_must_specify_known_approved_operation_and_integer_ask():
    from lab_arena.tool_market.config import PLATFORM_MAX_ASKS_ENV, load_config, ToolMarketConfigError

    env = {
        "TOOL_MARKET_MODE": "live",
        "TOOL_MARKET_TERMS_VERSION": "tool-market-terms-v1",
        "TOOL_MARKET_PUBLIC_BASE_URL": "https://market.example.test",
        "TOOL_MARKET_GAMMA_SERVICE": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    }
    assert not load_config(env).platform_max_asks
    alias = platform.SHORT_PROFILE
    assert load_config(dict(env, **{PLATFORM_MAX_ASKS_ENV: json.dumps({alias: 1200})})).platform_max_asks == {
        alias: 1200
    }
    for bad in (
        "[1]",
        '{"typo":200}',
        '{"zerobounce_validate":200}',  # its ceiling has its own variable
        "{",
        json.dumps({alias: 0}),
        json.dumps({alias: True}),
        json.dumps({alias: 2**53}),
    ):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(env, **{PLATFORM_MAX_ASKS_ENV: bad}))
