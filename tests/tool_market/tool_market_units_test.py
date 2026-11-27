"""Tool-market money, catalog and config rules that need no database."""

from __future__ import annotations

import hashlib
import json
import pathlib
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market import book, catalog
from lab_arena.tool_market.config import PUBLISHED_TERMS, ToolMarketConfigError, load_config
from lead_order_vault import tools

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
ZB = catalog.operation("zerobounce", "validate")
PROSPEO = catalog.operation("prospeo", "enrich_person")
HUNTER = catalog.operation("hunter", "email_finder")
FINDYMAIL = catalog.operation("findymail", "find_from_name")
EXA = catalog.operation("exa", "search")


def test_a_one_cent_call_costs_the_buyer_one_point_one_cents():
    owed = book.owed_microusd(ZB, 1, 10_000)
    assert owed == 10_000 and book.charge_microusd(owed) == 11_000
    assert book.reserve_microusd(ZB, 10_000) == 11_000


def test_a_vendor_usd_ask_is_a_share_of_the_reported_cost():
    # $0.002 reported, sold at 90% of the vendor's price: owed $0.0018, buyer $0.00198.
    assert book.owed_microusd(EXA, 2_000, 900_000) == 1_800
    assert book.charge_microusd(1_800) == 1_980
    # The reserve covers the operation's maximum: $0.05 x 90% x 1.1.
    assert book.reserve_microusd(EXA, 900_000) == 49_500
    # Usage beyond the maximum is not counted, so a charge never passes the reserve.
    assert book.owed_microusd(EXA, 10**9, 900_000) == 45_000


def test_markup_rounds_up_and_never_charges_less_than_owed():
    for owed in (0, 1, 9, 10, 11, 999_999, 10**12):
        assert book.charge_microusd(owed) == -(-owed * 11 // 10) >= owed


def test_cheapest_ask_first_and_earlier_wins_a_tie():
    asks = [book.Ask("c", ALICE, "e", 30, 1, 99), book.Ask("a", ALICE, "e", 10, 3, 99),
            book.Ask("b", ALICE, "e", 10, 2, 99)]
    assert [a.key_id for a in book.rank(asks)] == ["b", "a", "c"]


def test_ask_bounds():
    assert book.valid_ask(1, 10) and book.valid_ask(10, 10)
    assert not any(book.valid_ask(v, 10) for v in (0, -1, 11, True, 1.5, "10"))


def test_key_failures_and_rate_limits():
    assert book.key_failure(402) == "no_credit" and book.key_failure(401) and book.key_failure(403)
    assert book.key_failure(400) is None and book.key_failure(429) is None


def test_get_operations_send_only_listed_params():
    url, headers, body = ZB.build({"email": "ada@acme.com", "ip_address": "1.2.3.4"})
    parts = urlsplit(url)
    assert parts.path == "/v2/validate" and parse_qs(parts.query) == {"email": ["ada@acme.com"], "ip_address": ["1.2.3.4"]}
    assert body == b"" and headers == {"accept": "application/json"}  # the vault adds the key
    for params in ({}, {"email": "x", "api_key": "k"}, {"email": 5}, {"email": ""}, {"email": "x" * 321}, "x",
                   {"email": "\U0001d51e" * 320}):  # 320 characters, but past the vault's 2000 once encoded
        with pytest.raises(catalog.CatalogError):
            ZB.build(params)
    with pytest.raises(catalog.CatalogError):
        HUNTER.build({"domain": "acme.com", "max_duration": 21})


def test_prospeo_gets_its_person_inside_data():
    _, headers, body = PROSPEO.build({"full_name": "Ada Lovelace", "company_website": "acme.com",
                                      "only_verified_email": True})
    assert json.loads(body) == {"data": {"full_name": "Ada Lovelace", "company_website": "acme.com"},
                                "only_verified_email": True}
    assert headers["content-type"] == "application/json"
    assert json.loads(PROSPEO.build({"linkedin_url": "https://www.linkedin.com/in/ada"})[2]) \
        == {"data": {"linkedin_url": "https://www.linkedin.com/in/ada"}}


# Answers as the vendors document them.
ZB_VALID = {"address": "ada@acme.com", "status": "valid", "sub_status": "", "free_email": False}
ZB_DEAD_KEY = {"error": "Invalid API Key or your account ran out of credits"}
PROSPEO_MATCH = {"error": False, "free_enrichment": False, "person": {"email": {"email": "ada@acme.com"}}}


def test_each_vendor_status_is_read_as_what_it_means():
    # ZeroBounce answers 200 for a dead key or an empty account.
    assert ZB.status(200, ZB_DEAD_KEY) == 402 and ZB.status(200, ZB_VALID) == 200
    # A bad request is the buyer's, so it fails no key.
    assert ZB.status(200, {"error": "Invalid IP address"}) == 400 and ZB.status(200, {}) == 200
    assert PROSPEO.status(400, {"error": True, "error_code": "INSUFFICIENT_CREDITS"}) == 402
    assert PROSPEO.status(400, {"error": True, "error_code": "INVALID_API_KEY"}) == 401
    assert PROSPEO.status(400, {"error": True, "error_code": "NO_MATCH"}) == 400
    # Hunter's 403 is a rate limit and its 429 an empty account.
    assert HUNTER.status(403, {}) == 429 and HUNTER.status(429, {}) == 402 and HUNTER.status(451, {}) == 451
    # A paused Findymail account is a bad key: no free check sees it.
    assert FINDYMAIL.status(423, {}) == 401 and FINDYMAIL.status(402, {}) == 402


def test_units_count_only_what_the_vendor_charges():
    assert ZB.units(200, ZB_VALID) == 1 and ZB.units(200, dict(ZB_VALID, status="catch-all")) == 1
    assert ZB.units(200, dict(ZB_VALID, status="unknown")) == 0  # ZeroBounce does not charge unknowns
    assert ZB.units(200, {}) is None and ZB.units(500, None) is None and ZB.units(400, None) == 0
    assert PROSPEO.units(200, PROSPEO_MATCH) == 1
    assert PROSPEO.units(200, dict(PROSPEO_MATCH, free_enrichment=True)) == 0
    assert PROSPEO.units(400, {"error": True, "error_code": "NO_MATCH"}) == 0
    assert HUNTER.units(200, {"data": {"email": "ada@acme.com", "score": 97}}) == 1
    assert HUNTER.units(200, {"data": {"email": None}}) == 0
    assert FINDYMAIL.units(200, {"contact": {"name": "Ada", "email": "ada@acme.com"}}) == 1
    assert FINDYMAIL.units(200, {"contact": None}) == 0
    assert EXA.units(200, {"costDollars": {"total": 0.005}}) == 5_000
    assert EXA.units(200, {"costDollars": {"total": "0.0000001"}}) == 1  # rounded up to a micro-USD
    assert EXA.units(200, {"costDollars": {"total": 1000}}) == EXA.max_units
    for answer in (None, {}, {"costDollars": None}, {"costDollars": {"total": -1}}, {"costDollars": {"total": "NaN"}},
                   {"costDollars": {"total": True}}, {"costDollars": {"total": "1e999999"}}, [1]):
        assert EXA.units(200, answer) is None  # unknown usage: nobody pays


def test_every_operation_fits_its_vault_route():
    for op in catalog.OPERATIONS.values():
        url, headers, body = op.build({name: _sample(p) for name, p in op.params.items() if p.required})
        parts = urlsplit(url)
        assert tools.HOSTS[parts.hostname] == op.tool
        assert tools.TOOLS[op.tool].route(op.method, parts.path, parts.query), (op.tool, op.name)
        assert not tools.TOOLS[op.tool].caller_sets_key(parts.query)
    assert set(catalog.TOOLS) <= set(tools.TOOLS)
    # The vault allows only the routes the catalog prices.
    assert not tools.TOOLS["harvestapi"].route("GET", "/linkedin/profile-search", "search=a")


def _sample(param):
    return {"str": "https://www.linkedin.com/in/someone", "int": param.low, "bool": True,
            "str_list": ["a"] * max(param.low, 1), "enum": param.choices[:1] and param.choices[0]}[param.kind]


def test_config_is_off_by_default_and_strict_when_live():
    assert not load_config({}).enabled
    live = {"LEADPOET_TOOL_MARKET_MODE": "live", "LEADPOET_TOOL_MARKET_GAMMA_SERVICE": ALICE,
            "LEADPOET_TOOL_MARKET_TERMS_VERSION": "tool-market-terms-v1",
            "LEADPOET_TOOL_MARKET_TERMS_SHA256": PUBLISHED_TERMS["tool-market-terms-v1"],
            "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "https://gw.example/"}
    config = load_config(live)
    # Live means every catalog tool trades; one switch turns single tools off.
    assert config.enabled and config.tools == set(catalog.TOOLS) and config.public_host == "gw.example"
    assert "zerobounce" in config.tools
    assert load_config(dict(live, LEADPOET_TOOL_MARKET_TOOLS_OFF="exa, hunter")).tools \
        == set(catalog.TOOLS) - {"exa", "hunter"}
    for bad in ({"LEADPOET_TOOL_MARKET_TOOLS_OFF": "deepline"},
                {"LEADPOET_LEAD_ORDERS_GAMMA_SERVICE": ALICE}, {"LEADPOET_TOOL_MARKET_TERMS_SHA256": "a" * 64},
                {"LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "http://gw.example"},
                {"LEADPOET_TOOL_MARKET_GAMMA_SERVICE": "71"}, {"LEADPOET_TOOL_MARKET_BUYER_HOTKEYS": "nope"},
                {"LEADPOET_TOOL_MARKET_MODE": "staff"}):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(live, **bad))


def test_the_published_terms_hash_is_the_terms_file():
    # The hash both sides sign must be the published file's exact bytes.
    path = pathlib.Path(__file__).resolve().parents[2] / "docs" / "tool-market-terms.md"
    text = path.read_bytes()
    assert hashlib.sha256(text).hexdigest() == PUBLISHED_TERMS["tool-market-terms-v1"]
    assert b"(tool-market-terms-v1)" in text


def test_every_operation_says_what_it_does():
    # Agents choose from the catalog by task; an operation without a line is unfindable.
    assert all(op.does for op in catalog.OPERATIONS.values())
