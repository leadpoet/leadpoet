"""Tool-market money, catalog and config rules that need no database."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market import book, catalog
from lab_arena.tool_market.config import ToolMarketConfigError, load_config
from lead_order_vault import vault

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SCRAPE = catalog.operation("scrapecreators", "linkedin_profile")
SERP = catalog.operation("dataforseo", "google_organic_serp")
EXA = catalog.operation("exa", "search")


def test_a_one_cent_call_costs_the_buyer_one_point_one_cents():
    owed = book.owed_microusd(SCRAPE, 1, 10_000)
    assert owed == 10_000 and book.charge_microusd(owed) == 11_000
    assert book.reserve_microusd(SCRAPE, 10_000) == 11_000


def test_a_vendor_usd_ask_is_a_share_of_the_reported_cost():
    # $0.002 reported, sold at 90% of the vendor's price: owed $0.0018, buyer $0.00198.
    assert book.owed_microusd(SERP, 2_000, 900_000) == 1_800
    assert book.charge_microusd(1_800) == 1_980
    # The reserve covers the operation's maximum: $0.05 x 90% x 1.1.
    assert book.reserve_microusd(SERP, 900_000) == 49_500
    # Usage beyond the maximum is not counted, so a charge never passes the reserve.
    assert book.owed_microusd(SERP, 10**9, 900_000) == 45_000


def test_markup_rounds_up_and_never_charges_less_than_owed():
    for owed in (0, 1, 9, 10, 11, 999_999, 10**12):
        assert book.charge_microusd(owed) == -(-owed * 11 // 10) >= owed


def test_cheapest_ask_first_and_earlier_wins_a_tie():
    asks = [book.Ask("c", ALICE, "e", 30, 1), book.Ask("a", ALICE, "e", 10, 3), book.Ask("b", ALICE, "e", 10, 2)]
    assert [a.key_id for a in book.rank(asks)] == ["b", "a", "c"]


def test_ask_bounds():
    assert book.valid_ask(1) and book.valid_ask(book.MAX_ASK_MICROUSD)
    assert not any(book.valid_ask(v) for v in (0, -1, book.MAX_ASK_MICROUSD + 1, True, 1.5, "10"))


def test_key_failures_and_rate_limits():
    assert book.key_failure(402) == "no_credit" and book.key_failure(401) and book.key_failure(403)
    assert book.key_failure(400) is None and book.key_failure(429) is None
    assert book.rate_limited(429, {}) and not book.rate_limited(429, {"Retry-After": "3"})


def test_get_operations_send_only_listed_params():
    url, headers, body = catalog.operation("scrapecreators", "google_search").build(
        {"query": "payroll software", "page": 2})
    parts = urlsplit(url)
    assert parts.path == "/v1/google/search" and parse_qs(parts.query) == {"page": ["2"], "query": ["payroll software"]}
    assert body == b"" and "authorization" not in headers
    for params in ({}, {"url": "x", "extra": 1}, {"url": 5}, {"url": ""}, {"url": "x" * 301}, "x"):
        with pytest.raises(catalog.CatalogError):
            SCRAPE.build(params)
    with pytest.raises(catalog.CatalogError):
        catalog.operation("scrapecreators", "google_search").build({"query": "q", "page": 12})


def test_dataforseo_gets_exactly_one_task():
    _, headers, body = SERP.build({"keyword": "payroll", "location_code": 2840})
    assert json.loads(body) == [{"keyword": "payroll", "location_code": 2840}]
    assert headers["content-type"] == "application/json"


def test_a_2xx_body_error_is_read_as_its_real_status():
    assert SERP.status(200, {"status_code": 40200}) == 402 and SERP.status(200, {"status_code": 20000}) == 200
    assert SERP.status(500, {"status_code": 40200}) == 500
    deepline = catalog.operation("deepline", "zerobounce_validate")
    assert deepline.status(200, {"billing": {"kind": "insufficient_credits"}}) == 402
    assert deepline.units(200, {"billing": {"credits_charged": 0.28}}) == 28_000
    assert deepline.units(200, {"result": {}}) is None
    _, headers, _ = deepline.build({"email": "ada@acme.com"})
    assert headers["x-deepline-execute-response-intent"] == "raw"


def test_units_come_from_the_vendor_answer():
    assert SCRAPE.units(200, None) == 1
    assert SCRAPE.units(404, None) == 0 and SCRAPE.units(500, None) is None
    assert SERP.units(200, {"cost": 0.0021}) == 2_100
    assert SERP.units(200, {"cost": "0.0000001"}) == 1  # rounded up to a micro-USD
    assert EXA.units(200, {"costDollars": {"total": 0.005}}) == 5_000
    assert EXA.units(200, {"costDollars": {"total": 1000}}) == EXA.max_units
    for answer in (None, {}, {"cost": None}, {"cost": -1}, {"cost": "NaN"}, {"cost": True}, [1]):
        assert SERP.units(200, answer) is None  # unknown usage: nobody pays


def test_every_operation_fits_its_vault_route():
    for op in catalog.OPERATIONS.values():
        url, headers, body = op.build({name: _sample(p) for name, p in op.params.items() if p.required})
        parts = urlsplit(url)
        assert vault.HOSTS[parts.hostname] == op.tool
        assert vault.TOOLS[op.tool].route(op.method, parts.path, parts.query), (op.tool, op.name)
    assert set(catalog.TOOLS) <= set(vault.TOOLS)


def _sample(param):
    return {"str": "https://www.linkedin.com/in/someone", "int": param.low, "bool": True,
            "str_list": ["a"] * max(param.low, 1), "enum": param.choices[:1] and param.choices[0]}[param.kind]


def test_config_is_off_by_default_and_strict_when_live():
    assert not load_config({}).enabled
    live = {"LEADPOET_TOOL_MARKET_MODE": "live", "LEADPOET_TOOL_MARKET_GAMMA_SERVICE": ALICE,
            "LEADPOET_TOOL_MARKET_TERMS_VERSION": "v1", "LEADPOET_TOOL_MARKET_TERMS_SHA256": "a" * 64,
            "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "https://gw.example/"}
    config = load_config(live)
    # Live means every catalog tool trades; one switch turns single tools off.
    assert config.enabled and config.tools == set(catalog.TOOLS) and config.public_host == "gw.example"
    assert "deepline" in config.tools
    assert load_config(dict(live, LEADPOET_TOOL_MARKET_TOOLS_OFF="exa, deepline")).tools \
        == set(catalog.TOOLS) - {"exa", "deepline"}
    for bad in ({"LEADPOET_TOOL_MARKET_TOOLS_OFF": "openrouter"},
                {"LEADPOET_LEAD_ORDERS_GAMMA_SERVICE": ALICE}, {"LEADPOET_TOOL_MARKET_TERMS_SHA256": "x"},
                {"LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "http://gw.example"},
                {"LEADPOET_TOOL_MARKET_GAMMA_SERVICE": "71"}, {"LEADPOET_TOOL_MARKET_BUYER_HOTKEYS": "nope"},
                {"LEADPOET_TOOL_MARKET_MODE": "staff"}):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(live, **bad))
