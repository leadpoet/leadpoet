"""Gateway tool-market routes: off by default, signed miners and buyers, tab tokens."""

from __future__ import annotations

import hashlib
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import tool_market as routes
from lab_arena.chain import wallet_signature_verifier
from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.market import ToolMarketError

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


class FakeMarket:
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

    def call(self, token, body):
        self.seen.append(("call", token, body))
        if token != "tmt_good":
            raise ToolMarketError(402, "tab_budget")
        return {"call_id": "c", "status": 200, "body": {"ok": True}, "charge_microusd": 11_000}

    def order_book(self, tool):
        return {"tool": tool, "asks": []}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as c:
        yield c
    routes.configure()


def enable():
    config = ToolMarketConfig(mode="live", tools=frozenset({"exa"}), gamma_service=ALICE, terms_version="v1",
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
    for method, path in (("GET", "/tool-market/books/exa"), ("POST", "/tool-market/call"),
                         ("GET", "/tool-market/tab"), ("POST", "/tool-market/tab/close"),
                         ("POST", "/tool-market/terms"), ("GET", "/tool-market/balance"),
                         ("POST", "/tool-market/tabs"), ("POST", "/tool-market/keys"),
                         ("POST", "/tool-market/keys/k/check"), ("POST", "/tool-market/keys/k/revoke"),
                         ("POST", "/tool-market/asks"), ("GET", "/tool-market/statement")):
        response = client.request(method, path)
        assert response.status_code == 404 and response.json() == {"detail": "Not Found"}
    assert routes._state is None


def test_misconfigured_live_mode_answers_503(client, monkeypatch):
    monkeypatch.setenv("LEADPOET_TOOL_MARKET_MODE", "live")
    routes.configure()
    response = client.get("/tool-market/books/exa")
    assert response.status_code == 503 and response.json() == {"error": "tool_market_misconfigured"}


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
    assert client.get("/tool-market/books/exa").json() == {"tool": "exa", "asks": []}
