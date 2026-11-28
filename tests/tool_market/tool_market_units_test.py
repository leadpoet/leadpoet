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


def test_a_vendor_usd_ask_is_a_share_of_the_reported_cost():
    # $0.002 reported, sold at 90% of the vendor's price: owed $0.0018, buyer $0.00198.
    assert book.owed_microusd(EXA, 2_000, 900_000) == 1_800
    assert book.charge_microusd(1_800) == 1_980
    # The reserve covers the operation's maximum: $0.05 x 90% x 1.1.
    assert book.charge_microusd(book.owed_microusd(EXA, EXA.max_units, 900_000)) == 49_500
    # Usage beyond the maximum is not counted, so a charge never passes the reserve.
    assert EXA.units(200, {"costDollars": {"total": 10**6}}) == EXA.max_units
    # A 2xx the gateway cannot send back is unknown, not free or one credit.
    assert ZB.units(200, None) is None and EXA.units(200, None) is None


def test_markup_rounds_up_and_never_charges_less_than_owed():
    for owed in (0, 1, 9, 10, 11, 999_999, 10**12):
        assert book.charge_microusd(owed) == -(-owed * 11 // 10) >= owed


def test_cheapest_ask_first_and_earlier_wins_a_tie():
    asks = [book.Ask("c", ALICE, "e", 30, 1, 99), book.Ask("a", ALICE, "e", 10, 3, 99),
            book.Ask("b", ALICE, "e", 10, 2, 99)]
    assert [a.key_id for a in book.rank(asks)] == ["b", "a", "c"]


def test_get_operations_send_only_listed_params():
    url, headers, body = ZB.build({"email": "ada@acme.com", "ip_address": "1.2.3.4"})
    parts = urlsplit(url)
    assert parts.path == "/v2/validate" and parse_qs(parts.query) == {"email": ["ada@acme.com"], "ip_address": ["1.2.3.4"]}
    assert body == b"" and headers == {"accept": "application/json"}  # the vault adds the key
    for params in ({}, {"email": "x", "api_key": "k"}, {"email": 5}, {"email": ""}, {"email": "x" * 321}, "x",
                   {"email": "\U0001d51e" * 320}, {"email": "a\ud800@x.com"}):  # 320 characters, but past the vault's 2000 once encoded
        with pytest.raises(catalog.CatalogError):
            ZB.build(params)
    with pytest.raises(catalog.CatalogError):  # each value fits, but the URL would pass the vault's 4096
        HUNTER.build({name: "\u20ac" * 200 for name in ("first_name", "last_name", "full_name")})
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
    for answer in ({}, {"costDollars": None}, {"costDollars": {"total": -1}}, {"costDollars": {"total": "NaN"}},
                   {"costDollars": {"total": True}}, {"costDollars": {"total": "1e999999"}}, [1]):
        assert EXA.units(200, answer) is None  # unknown usage: nobody pays


def test_vendor_answer_readers_never_raise():
    # A vendor answer is any JSON; a reader that raises leaves the call unsettled.
    odd = [None, True, 1, "x", [], [1], {}, {"error_code": ["X"]}, {"error_code": {"a": 1}}, {"error": [1]},
           {"status": ["valid"]}, {"data": [1]}, {"data": {"email": ["a"]}}, {"contact": "x"},
           {"costDollars": [1]}, {"costDollars": {"total": [1]}}, {"error": False, "free_enrichment": [1]}]
    for op in catalog.OPERATIONS.values():
        for status in (200, 400, 402, 500):
            for document in odd:
                op.units(op.status(status, document), document)


def test_every_operation_fits_its_vault_route():
    for op in catalog.OPERATIONS.values():
        # The required parameters, then every parameter at its lowest and highest
        # (skipping a combination the operation itself refuses, such as Exa's).
        for required_only, chosen in ((True, {n: _sample(p) for n, p in op.params.items() if p.required}),
                                      (False, {n: _sample(p) for n, p in op.params.items()}),
                                      (False, {n: _sample(p, high=True) for n, p in op.params.items()})):
            if required_only and op.name == "search_leads":
                chosen["currentJobTitles"] = "Director"  # at least one explicit targeting filter
            try:
                url, headers, body = op.build(chosen)
            except catalog.CatalogError:
                assert not required_only, (op.tool, op.name)
                continue
            parts = urlsplit(url)
            assert tools.HOSTS[parts.hostname] == op.tool
            assert tools.TOOLS[op.tool].route(op.method, parts.path, parts.query), (op.tool, op.name, chosen)
            assert not tools.TOOLS[op.tool].caller_sets_key(parts.query)
    assert set(catalog.TOOLS) <= set(tools.TOOLS)
    # The vault allows only the routes the catalog prices.
    assert not tools.TOOLS["harvestapi"].route("GET", "/linkedin/profile-search", "search=a")


def test_the_spec_bounds_each_parameter_as_call_does():
    import jsonschema

    for op in catalog.OPERATIONS.values():
        for name, param in op.params.items():
            longest = "x" * param.max_length
            values = [_sample(param), _sample(param, True), longest, longest + "x", "", None, True, 0,
                      param.low - 1, param.high + 1, [], ["a"] * (param.high + 1), [longest + "x"], "nope"]
            for value in values:
                try:
                    param.read(value)
                    read = True
                except catalog.CatalogError:
                    read = False
                valid = jsonschema.Draft4Validator(param.schema()).is_valid(value)
                assert valid == read, (op.tool, op.name, name, value)


def _sample(param, high=False):
    count = param.high if high else param.low
    return {"str": "https://www.linkedin.com/in/someone", "int": count, "bool": True,
            "str_list": ["a"] * count, "enum": param.choices[-1 if high else 0] if param.choices else None}[param.kind]


def test_config_is_off_by_default_and_strict_when_live():
    assert not load_config({}).enabled
    live = {"LEADPOET_TOOL_MARKET_MODE": "live", "LEADPOET_TOOL_MARKET_GAMMA_SERVICE": ALICE,
            "LEADPOET_TOOL_MARKET_TERMS_VERSION": "tool-market-terms-v1",
            "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "https://gw.example/"}
    config = load_config(live)
    assert config.terms_sha256 == PUBLISHED_TERMS["tool-market-terms-v1"]
    # Live means every catalog tool trades; one switch turns single tools off.
    assert config.enabled and config.tools == set(catalog.TOOLS) and config.public_host == "gw.example"
    assert load_config(dict(live, LEADPOET_TOOL_MARKET_TOOLS_OFF="exa, hunter")).tools \
        == set(catalog.TOOLS) - {"exa", "hunter"}
    for bad in ({"LEADPOET_TOOL_MARKET_TOOLS_OFF": "deepline"},
                {"LEADPOET_LEAD_ORDERS_GAMMA_SERVICE": ALICE}, {"LEADPOET_TOOL_MARKET_TERMS_VERSION": "tool-market-terms-v0"},
                {"LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "http://gw.example"},
                {"LEADPOET_TOOL_MARKET_GAMMA_SERVICE": "71"}, {"LEADPOET_TOOL_MARKET_BUYER_HOTKEYS": "nope"},
                {"LEADPOET_TOOL_MARKET_MODE": "staff"}, {"LAB_ARENA_NETUID": "72"}):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(live, **bad))
    # Resale permission is per vendor and hotkey.
    bob = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    assert not config.may_supply("zerobounce", bob)
    supplied = load_config(dict(live, LEADPOET_TOOL_MARKET_APPROVED_SUPPLIERS="zerobounce:" + bob))
    assert supplied.may_supply("zerobounce", bob)
    assert not supplied.may_supply("harvestapi", bob) and not supplied.may_supply("zerobounce", ALICE)
    for invalid in ("unknown:" + bob, "zerobounce:", bob):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(live, LEADPOET_TOOL_MARKET_APPROVED_SUPPLIERS=invalid))


def test_the_published_terms_hash_is_the_terms_file():
    # The hash both sides sign must be the published file's exact bytes.
    path = pathlib.Path(__file__).resolve().parents[2] / "docs" / "tool-market-terms.md"
    text = path.read_bytes()
    assert hashlib.sha256(text).hexdigest() == PUBLISHED_TERMS["tool-market-terms-v1"]
    assert b"(tool-market-terms-v1)" in text


def test_btauth_signs_the_standards_eight_lines_and_refuses_each_bad_request():
    from bittensor_wallet import Keypair

    from lab_arena.chain import wallet_signature_verifier
    from lab_arena.lead_orders.auth import AuthError
    from lab_arena.tool_market import btauth

    # The signed-requests guide's sr25519 test vector.
    alice, bob = Keypair.create_from_uri("//Alice"), "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    assert btauth.payload("post", "/generate?stream=false", b'{"prompt": "hello"}', "1752076800000000000",
                          alice.ss58_address, bob) == "\n".join((
        "btauth/1", "sr25519", "POST", "/generate?stream=false",
        "341c57448e531310fbbe83f44cea2a5e838bd9e8a6b82b269f01d0dbbc23c3cc", "1752076800000000000",
        alice.ss58_address, bob))
    now = 1752076800 * 10**9
    used = set()

    def once(hotkey, nonce_hash, signed_at):
        if (hotkey, nonce_hash) in used:
            return False
        used.add((hotkey, nonce_hash))
        return True

    def check(headers, *, body=b"{}", receiver=bob, at=now):
        return btauth.verify({k.lower(): v for k, v in headers.items()}, body, method="POST", target="/x",
                             receiver=receiver, verifier=wallet_signature_verifier, use_nonce=once,
                             now_ns=lambda: at)

    def fresh():
        return btauth.sign(alice, method="POST", target="/x", body=b"{}", receiver=bob, now_ns=lambda: now)

    headers = fresh()
    assert check(headers).hotkey == alice.ss58_address
    refusals = {
        "btauth_replayed_nonce": lambda: check(headers),
        "btauth_unknown_version": lambda: check(dict(fresh(), **{"X-Bittensor-Version": "2"})),
        "btauth_missing_header": lambda: check({k: v for k, v in fresh().items() if k != "X-Bittensor-Nonce"}),
        "btauth_unsupported_scheme": lambda: check(dict(fresh(), **{"X-Bittensor-Crypto": "ed25519"})),
        "btauth_wrong_receiver": lambda: check(fresh(), receiver=alice.ss58_address),
        "btauth_stale_nonce": lambda: check(fresh(), at=now + 11 * 10**9),
        "btauth_bad_signature": lambda: check(fresh(), body=b'{"a": 1}'),
    }
    for code, attempt in refusals.items():
        with pytest.raises(AuthError) as refused:
            attempt()
        assert (refused.value.status, refused.value.code) == (401, code)
