"""Gateway lead-order routes: off by default, signed buyers, bearer runner."""

from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import lead_orders as routes
from lab_arena.broker import ProviderResponse
from lab_arena.chain import wallet_signature_verifier
from lab_arena.lead_orders.config import LeadOrdersConfig
from lab_arena.lead_orders.service import LeadOrderServiceError

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SECRET = "r" * 40


class FakeStore:
    def __init__(self):
        self.used = set()

    def use_nonce(self, *, hotkey, signature_hash, signed_at):
        if (hotkey, signature_hash) in self.used:
            return False
        self.used.add((hotkey, signature_hash))
        return True


class FakeService:
    def __init__(self):
        self._store = FakeStore()
        self.placed = []

    def place_order(self, hotkey, payload):
        self.placed.append((hotkey, payload))
        return {"order_id": "o1", "status": "queued"}

    def order_view(self, hotkey, order_id):
        raise LeadOrderServiceError(404, "order_unknown")

    def claim(self):
        return None

    def heartbeat(self, order_id):
        return {"continue": False, "spent_usd": "0.00", "deadline": None}

    def finish(self, order_id, payload):
        self.finished = (order_id, payload)
        return {"status": payload["status"]}

    def accept_miner_terms(self, identity, payload):
        self.terms = (identity, payload)
        return {"status": "accepted", "terms_version": payload["terms_version"], "accepted_at": "t"}

    def miner_terms(self):
        return {"terms_version": "v1", "terms_sha256": "a" * 64, "text": "terms"}

    def push_suppressions(self, payload):
        self.pushed = payload
        return {"received": 1, "added": 1}

    def read_suppressions(self, hotkey, since):
        self.read = (hotkey, since)
        return {"suppressions": [], "next_since": 0, "more": False}

    def public_receipt(self, order_id):
        return {"receipt": {"order_id": order_id}, "calls": []}

    def statement(self, hotkey, since):
        self.statement_read = (hotkey, since)
        return {"calls": [], "next_since": 0, "more": False}

    def public_policy(self, version):
        if version != "v1":
            raise LeadOrderServiceError(404, "policy_unknown")
        return {"policy": {"version": "v1"}}


class FakeBroker:
    def __init__(self):
        self.calls = []

    def call(self, token, request):
        self.calls.append((token, request))
        return ProviderResponse(402, {"content-type": "application/json"}, b'{"error":"spend_ceiling"}')


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as c:
        yield c
    routes.configure()


def enable(mode="staff"):
    config = LeadOrdersConfig(mode=mode, staff_hotkeys=frozenset({ALICE}), runner_secret=SECRET,
                              policy_path="/x", terms_version="t", public_base_url="https://gw.example")
    service, broker = FakeService(), FakeBroker()
    routes.configure(builder=lambda: (config, service, broker), verifier=wallet_signature_verifier)
    return service, broker


def signed(path, body, *, timestamp=None, method="POST", host="gw.example"):
    from bittensor_wallet import Keypair

    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    message = "%s\n%s\n%s\n%s\n%s" % (method, host, path, timestamp,
                                       __import__("hashlib").sha256(body).hexdigest())
    signature = "0x" + Keypair.create_from_uri("//Alice").sign(message.encode()).hex()
    return {"x-hotkey": ALICE, "x-timestamp": timestamp, "x-signature": signature, "content-type": "application/json"}


ORDER = {"seller": {"product": "p"}, "search": {"signal": "hiring"}, "leads": 10}


def test_every_route_is_404_when_the_mode_is_off(client, monkeypatch):
    monkeypatch.delenv("LEADPOET_LEAD_ORDERS_MODE", raising=False)
    routes.configure()
    for method, path in (("POST", "/lead-orders"), ("GET", "/lead-orders/balance"),
                         ("GET", "/lead-orders/" + "0" * 8),
                         ("POST", "/internal/lead-orders/claim"), ("POST", "/internal/lead-orders/x/heartbeat"),
                         ("POST", "/internal/lead-orders/x/finish"), ("GET", "/lead-orders/receipts/x"),
                         ("GET", "/lead-orders/buyers/confirm?token=t"), ("GET", "/lead-orders/epochs/abc"),
                         ("GET", "/lead-orders/epochs/12"), ("POST", "/internal/lead-orders/suppressions"),
                         ("GET", "/lead-orders/suppressions?since=0"), ("POST", "/lead-orders/miner-terms"),
                         ("GET", "/lead-orders/miner-terms"),
                         ("POST", "/lead-orders/broker/openrouter/api/v1/chat/completions")):
        response = client.request(method, path)
        assert response.status_code == 404 and response.json() == {"detail": "Not Found"}
    assert routes._state is None  # nothing was built or cached


def test_misconfigured_mode_answers_503_without_detail(client, monkeypatch):
    monkeypatch.setenv("LEADPOET_LEAD_ORDERS_MODE", "live")
    routes.configure()
    response = client.post("/internal/lead-orders/claim")
    assert response.status_code == 503 and response.json() == {"error": "lead_orders_misconfigured"}


def test_signed_buyer_request_once_then_replay_refused(client):
    service, _ = enable()
    body = json.dumps(ORDER).encode()
    headers = signed("/lead-orders", body)
    first = client.post("/lead-orders", content=body, headers=headers)
    assert first.status_code == 200 and service.placed == [(ALICE, ORDER)]
    again = client.post("/lead-orders", content=body, headers=headers)
    assert again.status_code == 401 and again.json() == {"error": "signature_replayed"}
    stale = client.post("/lead-orders", content=body,
                        headers=signed("/lead-orders", body, timestamp=int(time.time()) - 120))
    assert stale.json() == {"error": "timestamp_outside_window"}
    unsigned = client.post("/lead-orders", content=body)
    assert unsigned.status_code == 401
    # A request signed for another gateway host cannot be replayed here.
    elsewhere = client.post("/lead-orders", content=body, headers=signed("/lead-orders", body, host="arena.example"))
    assert elsewhere.status_code == 401 and elsewhere.json() == {"error": "signature_invalid"}
    missing = client.get("/lead-orders/abc", headers=signed("/lead-orders/abc", b"", method="GET"))
    assert missing.status_code == 404 and missing.json() == {"error": "order_unknown"}
    # A signature for one path does not authorize another.
    other = client.get("/lead-orders/balance", headers=signed("/lead-orders/abc", b"", method="GET"))
    assert other.status_code == 401


def test_runner_routes_need_the_bearer_secret(client):
    service, _ = enable()
    assert client.post("/internal/lead-orders/claim").status_code == 401
    assert client.post("/internal/lead-orders/claim", headers={"authorization": "Bearer " + "r" * 39}).status_code == 401
    runner = {"authorization": "Bearer " + SECRET}
    empty = client.post("/internal/lead-orders/claim", headers=runner)
    assert empty.status_code == 204 and empty.content == b""
    beat = client.post("/internal/lead-orders/o1/heartbeat", headers=runner, content=b"{}")
    assert beat.json() == {"continue": False, "spent_usd": "0.00", "deadline": None}
    assert client.post("/internal/lead-orders/o1/finish", content=b'{"status":"failed"}').status_code == 401
    done = client.post("/internal/lead-orders/o1/finish", headers=runner, content=b'{"status":"failed"}')
    assert done.json() == {"status": "failed"} and service.finished == ("o1", {"status": "failed"})


def test_broker_takes_the_order_token_as_the_api_key(client):
    _, broker = enable()
    path = "/lead-orders/broker/openrouter/api/v1/chat/completions"
    proxied = client.post(path, headers={"authorization": "Bearer tok"}, content=b"{}")
    assert proxied.status_code == 402 and proxied.json() == {"error": "spend_ceiling"}
    token, request = broker.calls[0]
    assert token == "tok" and request.provider == "openrouter" and request.path == "api/v1/chat/completions"
    client.get("/lead-orders/broker/deepline/api/v2/jobs/1?page=2", headers={"authorization": "Basic tok"})
    token, request = broker.calls[1]
    assert token == "" and request.query == {"page": "2"}  # a non-bearer header passes no token


def test_public_receipt_needs_no_auth(client):
    enable()
    assert client.get("/lead-orders/receipts/o9").json()["receipt"] == {"order_id": "o9"}


def test_broker_passes_only_the_deepline_response_shape_headers(client):
    _, broker = enable()
    client.post("/lead-orders/broker/deepline/api/v2/integrations/icypeas_read_results/execute",
                headers={"authorization": "Bearer tok", "x-deepline-execute-response-contract": "raw-v2",
                         "x-deepline-execute-response-intent": "raw", "x-deepline-tool-error-schema": "1",
                         "x-other": "kept-out", "cookie": "c=1"}, content=b"{}")
    _, request = broker.calls[0]
    assert request.headers == {"x-deepline-execute-response-contract": "raw-v2",
                               "x-deepline-execute-response-intent": "raw", "x-deepline-tool-error-schema": "1"}


def test_suppression_push_needs_the_runner_secret_and_the_feed_a_signed_hotkey(client):
    service, _ = enable()
    body = b'{"suppressions":[{"email_sha256":"%s","added_at":"2026-11-19T10:00:00Z"}]}' % (b"0" * 64)
    assert client.post("/internal/lead-orders/suppressions", content=body).status_code == 401
    pushed = client.post("/internal/lead-orders/suppressions", content=body,
                         headers={"authorization": "Bearer " + SECRET})
    assert pushed.json() == {"received": 1, "added": 1} and service.pushed == json.loads(body)
    assert client.get("/lead-orders/suppressions?since=5").status_code == 401
    feed = client.get("/lead-orders/suppressions?since=5",
                      headers=signed("/lead-orders/suppressions", b"", method="GET"))
    assert feed.status_code == 200 and service.read == (ALICE, "5")
    served = client.get("/lead-orders/miner-terms")  # public: a miner reads before signing
    assert served.status_code == 200 and served.json()["terms_sha256"] == "a" * 64
    terms = json.dumps({"terms_sha256": "a" * 64, "terms_version": "v1"}, separators=(",", ":")).encode()
    accepted = client.post("/lead-orders/miner-terms", content=terms, headers=signed("/lead-orders/miner-terms", terms))
    identity, payload = service.terms
    assert accepted.status_code == 200 and payload == {"terms_sha256": "a" * 64, "terms_version": "v1"} and identity.hotkey == ALICE
    assert identity.body_sha256 == __import__("hashlib").sha256(terms).hexdigest() and len(identity.signature) == 130


def test_statement_needs_the_miners_signature_and_policies_are_public(client):
    service, _ = enable()
    assert client.get("/lead-orders/statement?since=3").status_code == 401
    path = "/lead-orders/statement"
    read = client.get(path + "?since=3", headers=signed(path, b"", method="GET"))
    assert read.status_code == 200 and service.statement_read == (ALICE, "3")
    assert client.get("/lead-orders/policies/v1").json() == {"policy": {"version": "v1"}}
    assert client.get("/lead-orders/policies/v9").status_code == 404
