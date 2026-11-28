"""Gateway tool-market routes: off by default, signed miners and buyers, tab tokens."""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import tool_market as routes
from lab_arena.chain import wallet_signature_verifier
from lab_arena.store import ArenaStoreError
from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.market import ToolMarket, ToolMarketError

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # the market's Gamma service hotkey


class FakeMarket:
    platform_call = ToolMarket.platform_call
    def __init__(self):
        self.used = set()
        self.seen = []

    def use_nonce(self, hotkey, signature_hash, signed_at):
        if (hotkey, signature_hash) in self.used:
            return False
        self.used.add((hotkey, signature_hash))
        return True

    def post_ask(self, identity, body):
        self.seen.append(("ask", identity.hotkey, body))
        return {"status": "posted", "ask_sequence": 1}

    def call(self, token, body, identity=None, arrived=None):
        if identity is not None:
            self.seen.append(("signed call", identity.hotkey, body))
            return {"call_id": "c", "status": 200, "body": {}, "charge_microusd": 11_000}
        self.seen.append(("call", token, body))
        if token != "tmt_good":
            raise ToolMarketError(402, "tab_budget")
        return {"call_id": "c", "status": 200, "body": {"ok": True}, "charge_microusd": 11_000}

    def tool_catalog(self):
        return {"tools": []}

    def usage(self, identity, after, limit):
        self.seen.append(("usage", identity.hotkey, after, limit))
        return {"calls": [], "watermark": after}

    def order_book(self, tool, operation=None):
        return {"tool": tool, "asks": []}

    def gamma_balance(self, hotkey):
        return {"hotkey": hotkey, "credited_usd": "0.000000", "used_usd": "0.000000", "remaining_usd": "0.000000"}

    def services(self):
        return {"publishers": [{"netuid": 71, "hotkey": BOB}]}

    def openapi(self):
        return {"openapi": "3.0.3"}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as c:
        yield c
    routes.configure()


def enable():
    config = ToolMarketConfig(mode="live", tools=frozenset({"exa"}), gamma_service=BOB, terms_version="v1",
                              terms_sha256="a" * 64, public_base_url="https://gw.example")
    market = FakeMarket()
    routes.configure(builder=lambda: (config, market), verifier=wallet_signature_verifier)
    return market


def signed(path, body, *, method="POST"):
    from bittensor_wallet import Keypair

    timestamp = str(int(time.time()))
    message = "%s\n%s\n%s\n%s\n%s" % (method, "gw.example", path, timestamp, hashlib.sha256(body).hexdigest())
    signature = "0x" + Keypair.create_from_uri("//Alice").sign(message.encode()).hex()
    return {"x-hotkey": ALICE, "x-timestamp": timestamp, "x-signature": signature}


def test_every_route_is_404_when_the_mode_is_off(client, monkeypatch):
    monkeypatch.delenv("LEADPOET_TOOL_MARKET_MODE", raising=False)
    routes.configure()
    for method, path in (("GET", "/tool-market/books/exa"), ("GET", "/tool-market/catalog"), ("POST", "/tool-market/call"),
                         ("GET", "/tool-market/tab"), ("POST", "/tool-market/tab/close"),
                         ("POST", "/tool-market/terms"), ("GET", "/tool-market/gamma/balance"),
                         ("GET", "/tool-market/services.json"), ("GET", "/tool-market/openapi.json"),
                         ("POST", "/tool-market/tabs"), ("POST", "/tool-market/keys"),
                         ("POST", "/tool-market/keys/k/check"), ("POST", "/tool-market/keys/k/revoke"),
                         ("POST", "/tool-market/asks"), ("GET", "/tool-market/statement"),
                         ("GET", "/tool-market/usage")):
        response = client.request(method, path)
        assert response.status_code == 404 and response.json() == {"detail": "Not Found"}
    assert routes._state is None


def test_misconfigured_live_mode_answers_503(client, monkeypatch):
    monkeypatch.setenv("LEADPOET_TOOL_MARKET_MODE", "live")
    routes.configure()
    response = client.get("/tool-market/books/exa")
    assert response.status_code == 503 and response.json() == {"error": "tool_market_misconfigured"}


def test_a_failed_build_is_not_retried_on_every_request(client, monkeypatch):
    builds = []

    def chain_down():
        builds.append(1)
        raise RuntimeError("chain unavailable")

    routes.configure(builder=chain_down)
    for _ in range(3):
        assert client.get("/tool-market/catalog").json() == {"error": "tool_market_unavailable"}
    assert len(builds) == 1
    monkeypatch.setattr(routes, "BUILD_RETRY_SECONDS", 0)
    client.get("/tool-market/catalog")
    assert len(builds) == 2


def test_a_slow_build_does_not_hold_other_requests(client):
    started, release = threading.Event(), threading.Event()

    def chain_hangs():
        started.set()
        release.wait(10)
        raise RuntimeError("chain timed out")

    routes.configure(builder=chain_hangs)
    building = threading.Thread(target=lambda: client.get("/tool-market/catalog"))
    building.start()
    try:
        assert started.wait(10)
        began = time.monotonic()
        assert client.get("/tool-market/books/exa").json() == {"error": "tool_market_unavailable"}
        assert time.monotonic() - began < 5  # answered at once, not after the build
    finally:
        release.set()
        building.join(10)


def test_a_store_outage_answers_503(client):
    market = enable()

    def store_down(token, body, identity=None, arrived=None):
        raise ArenaStoreError("store unavailable")

    market.call = store_down
    answer = client.post("/tool-market/call", json={"tool": "exa", "operation": "search"},
                         headers={"authorization": "Bearer tmt_good"})
    assert answer.status_code == 503 and answer.json() == {"error": "store_unavailable"}


def test_signed_routes_take_one_use_of_a_signature(client):
    market = enable()
    body = json.dumps({"key_id": "k", "ask_microusd": 10_000}).encode()
    headers = signed("/tool-market/asks", body)
    assert client.post("/tool-market/asks", content=body, headers=headers).status_code == 200
    assert market.seen == [("ask", ALICE, {"key_id": "k", "ask_microusd": 10_000})]
    again = client.post("/tool-market/asks", content=body, headers=headers)
    assert again.status_code == 401 and again.json() == {"error": "signature_replayed"}
    assert client.post("/tool-market/asks", content=body).status_code == 401
    assert client.post("/tool-market/asks", content=b"[]", headers=signed("/tool-market/asks", b"[]")).json() \
        == {"error": "invalid_json"}


def test_calls_take_the_tab_token_and_return_the_market_answer(client):
    market = enable()
    body = {"tool": "exa", "operation": "search", "params": {"query": "q"}}
    answer = client.post("/tool-market/call", json=body, headers={"authorization": "Bearer tmt_good"})
    assert answer.status_code == 200 and answer.json()["charge_microusd"] == 11_000
    refused = client.post("/tool-market/call", json=body, headers={"authorization": "Basic tmt_good"})
    assert refused.status_code == 402 and refused.json() == {"error": "tab_budget"}
    assert market.seen[-1][1] == ""  # a non-bearer header passes no token


def test_usage_is_signed_and_pages_by_a_strict_watermark(client):
    market = enable()
    answer = client.get("/tool-market/usage?after=7&limit=50", headers=signed("/tool-market/usage", b"", method="GET"))
    assert answer.status_code == 200 and market.seen[-1] == ("usage", ALICE, 7, 50)
    client.get("/tool-market/usage?after=-1", headers=signed("/tool-market/usage", b"", method="GET"))
    assert market.seen[-1] == ("usage", ALICE, None, 100)  # not digits: the market refuses it
    assert client.get("/tool-market/usage").status_code == 401


def test_lead_search_raw_v2_verification_preserves_vendor_verdict_and_actual_charge(client):
    market = enable()
    market._config = ToolMarketConfig(platform_max_asks={"zerobounce_validate": 5899})
    calls = []
    def verify(token, body, arrived=None):
        calls.append((token, body))
        return {"call_id": "call1", "status": 200, "body": {"status": "invalid"}, "charge_microusd": 5100}
    market.call = verify
    url = "/tool-market/api/v2/integrations/zerobounce_validate/execute"
    response = client.post(url, headers={"Authorization": "Bearer tab"}, json={
        "payload": {"email": "fixture@example.test"}, "forceToolRefresh": True})
    assert response.status_code == 200
    assert response.json() == {"status": "completed", "result": {"status": "invalid"}, "toolResponse": {"raw": {"status": "invalid"}},
                               "billing": {"cost_usd": 0.0051}, "call_id": "call1"}
    assert calls == [("tab", {"tool": "zerobounce", "operation": "validate", "private": True, "max_ask_microusd": 5899,
                               "params": {"email": "fixture@example.test"}})]
    assert client.post(url, json={"payload": {}, "forceToolRefresh": "yes"}).status_code == 400
    assert len(calls) == 1
    market.call = lambda *_, **__: {"call_id": "call2", "status": 200, "body": {}, "charge_microusd": 0}
    assert client.post(url, json={"payload": {}}).json()["status"] == "failed"
    market.call = lambda *_, **__: {"call_id": "call3", "status": 500, "body": {}, "charge_microusd": 0}
    assert client.post(url, json={"payload": {}}).json()["status"] == "failed"


def btauth_headers(path, body, *, receiver=BOB):
    from bittensor_wallet import Keypair

    from lab_arena.tool_market import btauth

    return btauth.sign(Keypair.create_from_uri("//Alice"), method="POST", target=path, body=body,
                       receiver=receiver)


def test_a_btauth_call_is_charged_to_the_signers_balance_with_no_key(client):
    market = enable()
    body = json.dumps({"tool": "exa", "operation": "search", "params": {"query": "q"}}).encode()
    headers = btauth_headers("/tool-market/call", body)
    answer = client.post("/tool-market/call", content=body, headers=headers)
    assert answer.status_code == 200 and market.seen[-1] == ("signed call", ALICE, json.loads(body))
    replayed = client.post("/tool-market/call", content=body, headers=headers)
    assert replayed.status_code == 401 and replayed.json() == {"error": "btauth_replayed_nonce"}
    assert replayed.headers["www-authenticate"] == "btauth/1"
    wrong = client.post("/tool-market/call", content=body,
                        headers=btauth_headers("/tool-market/call", body, receiver=ALICE))
    assert wrong.json() == {"error": "btauth_wrong_receiver"}
    both = client.post("/tool-market/call", content=body, headers=dict(
        btauth_headers("/tool-market/call", body), authorization="Bearer tmt_good"))
    assert both.status_code == 400 and both.json() == {"error": "btauth_ambiguous_credentials"}
    # The path signed must be the path sent, query and all.
    moved = client.post("/tool-market/call?x=1", content=body, headers=btauth_headers("/tool-market/call", body))
    assert moved.json() == {"error": "btauth_bad_signature"}


def test_the_terms_take_only_the_scheme_whose_signature_is_stored(client):
    enable()
    body = b"{}"
    answer = client.post("/tool-market/terms", content=body, headers=btauth_headers("/tool-market/terms", body))
    assert answer.status_code == 401 and answer.json() == {"error": "hotkey_invalid"}


def test_discovery_and_balance_are_public_and_readable_from_any_page(client):
    enable()
    for path in ("/tool-market/services.json", "/tool-market/openapi.json", "/tool-market/catalog",
                 "/tool-market/gamma/balance?hotkey=" + ALICE):
        answer = client.get(path)
        assert answer.status_code == 200 and answer.headers["access-control-allow-origin"] == "*"
    assert client.get("/tool-market/gamma/balance?hotkey=" + ALICE).json()["remaining_usd"] == "0.000000"
