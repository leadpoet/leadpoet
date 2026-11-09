"""Oversize bodies remain rejected; original dispatch receipts can price them."""
import json

import httpx
import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    NATIVE_REQUEST_ID, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import (
    DL_KEY, FakeLedgerStore, _async_client_factory, make_broker,
)


def _send(url, headers, *, method="POST", body=b"oversize", limit=3):
    transport = br.HttpxProviderTransport(client_factory=_async_client_factory(
        lambda _: httpx.Response(200, headers=headers, content=body)
    ))
    try:
        return transport.send(method=method, url=url, headers={}, body=b"{}",
                              timeout_seconds=1, max_response_bytes=limit)
    finally:
        transport.close()


@pytest.mark.parametrize("url, headers, expected", [
    ("https://code.deepline.com/api/v2/execute",
     {"x-deepline-request-id": NATIVE_REQUEST_ID},
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/integrations/firecrawl_scrape/execute",
     {"x-vercel-id": "iad1::" + NATIVE_REQUEST_ID},
     {"x-vercel-id": NATIVE_REQUEST_ID}),
    ("https://openrouter.ai/api/v1/chat/completions",
     {"x-generation-id": "gen-original"}, {"x-generation-id": "gen-original"}),
    ("https://openrouter.ai/api/v1/responses",
     {"x-generation-id": "gen-original"}, {"x-generation-id": "gen-original"}),
])
def test_original_dispatch_receipt_survives_rejected_body(url, headers, expected):
    response = _send(url, {**headers, "set-cookie": "private", "x-unrelated": "discard"})
    assert response.status == 502
    assert response.body == operations.GENERIC_UNAVAILABLE_BODY
    assert response.internal_provenance == "response_too_large"
    assert response.headers == {"content-type": "application/json", **expected}


@pytest.mark.parametrize("url, method, headers", [
    ("https://code.deepline.com/api/v2/executions/by-key/key", "GET",
     {"x-vercel-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/executions/by-key/key", "GET",
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/executions/by-key/key", "POST",
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/execute", "GET",
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://openrouter.ai/api/v1/responses?lookup=true", "POST",
     {"x-generation-id": "gen-not-dispatch"}),
    ("https://openrouter.ai/api/v1/generation?id=gen-original", "GET",
     {"x-generation-id": "gen-lookup"}),
    ("https://api.scrapingdog.com/scrape", "POST",
     {"x-deepline-request-id": NATIVE_REQUEST_ID, "x-generation-id": "gen-foreign"}),
    ("https://code.deepline.com.evil.example/api/v2/execute", "POST",
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com:444/api/v2/execute", "POST",
     {"x-deepline-request-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/execute", "POST",
     {"x-deepline-request-id": "bad/id", "x-vercel-id": NATIVE_REQUEST_ID}),
    ("https://code.deepline.com/api/v2/execute", "POST",
     {"x-deepline-request-id": "ctx-tool-" + "a" * 32}),
    ("https://code.deepline.com/api/v2/execute", "POST",
     [("x-deepline-request-id", NATIVE_REQUEST_ID),
      ("x-deepline-request-id", "conflicting-native")]),
    ("https://openrouter.ai/api/v1/responses", "POST",
     {"x-generation-id": "bad/id"}),
    ("https://openrouter.ai/api/v1/responses", "POST",
     [("x-generation-id", "gen-one"), ("x-generation-id", "gen-two")]),
])
def test_non_dispatch_or_invalid_receipt_cannot_survive(url, method, headers):
    response = _send(url, headers, method=method)
    assert response.headers == {"content-type": "application/json"}
    assert response.status == 502 and response.internal_provenance == "response_too_large"


def test_normal_body_and_exact_cap_remain_unchanged():
    headers = {"x-deepline-request-id": NATIVE_REQUEST_ID, "x-unrelated": "normal"}
    response = _send("https://code.deepline.com/api/v2/execute", headers, body=b"123", limit=3)
    assert response.status == 200 and response.body == b"123"
    assert response.internal_provenance is None
    assert response.headers["x-unrelated"] == "normal"


@pytest.mark.parametrize("priced", [True, False])
def test_oversize_uses_existing_exact_receipt_without_source_lookup(priced):
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, headers={"x-deepline-request-id": NATIVE_REQUEST_ID},
                                  content=b"oversize source body")
        assert str(request.url) == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE_REQUEST_ID, safe="")
        return httpx.Response(200, json={"recent": {"request_id": NATIVE_REQUEST_ID,
            "entries": [{"id": "charge", "request_id": NATIVE_REQUEST_ID,
                "provider": "firecrawl", "operation": "firecrawl_scrape",
                "credits": 0.02, "delta": -0.02, "charge_state": "posted",
                "charge_finality": "final", "metadata": {}}] if priced else "invalid"}})
    class SmallDispatch(br.HttpxProviderTransport):
        def send(self, **kwargs):
            if kwargs["method"] == "POST":
                kwargs["max_response_bytes"] = 3
            return super().send(**kwargs)
    transport = SmallDispatch(client_factory=_async_client_factory(respond))
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    try:
        result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
            parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=30_000)
    finally:
        transport.close()
    assert [request.method for request in requests] == ["POST", "GET"]
    assert result.status == 502
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert NATIVE_REQUEST_ID not in json.dumps(result.to_document())
    retained = store.calls[result.call["call_identity"]]
    if priced:
        assert result.call["outcome"] == "settled" and retained["actual"] == 2_000
        assert retained["terminal"]["provider_cost"]["request_id"] == NATIVE_REQUEST_ID
    else:
        assert result.call["outcome"] == "uncertain"
        assert "actual_microusd" not in result.call
        assert retained["uncertain_doc"]["deepline_job_id"] == NATIVE_REQUEST_ID
        assert retained["uncertain_doc"]["response_provenance"] == "response_too_large"


@pytest.mark.parametrize("provider", ["deepline", "openrouter"])
def test_echoed_receipt_still_fails_existing_credential_guard(provider):
    from tests.lab_arena.test_lab_arena_broker import CHAT, CONTEXT, HOST_KEYS
    secret = HOST_KEYS[provider]
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            receipt = "x-deepline-request-id" if provider == "deepline" else "x-generation-id"
            return httpx.Response(200, headers={receipt: secret}, content=b"oversize")
        return httpx.Response(200, headers={"x-deepline-idempotency-supported": "true"}, json={})
    class SmallDispatch(br.HttpxProviderTransport):
        def send(self, **kwargs):
            if kwargs["method"] == "POST":
                kwargs["max_response_bytes"] = 3
            return super().send(**kwargs)
    transport = SmallDispatch(client_factory=_async_client_factory(respond))
    broker, store, _ = make_broker(transport=transport,
        credential_for=lambda _context, name: HOST_KEYS[name],
        funding_source_for=lambda _context: "miner_key")
    try:
        result = broker.execute(
            _score_context() if provider == "deepline" else CONTEXT,
            operation_id="scrapingdog.scrape" if provider == "deepline" else "openrouter.chat",
            parameters={"url": "https://example.com/about"} if provider == "deepline" else CHAT,
            action_sequence=0, timeout_ms=30_000,
        )
    finally:
        transport.close()
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    retained = store.calls[result.call["call_identity"]]
    assert retained["uncertain_doc"]["response_provenance"] == "credential_echo"
    assert secret not in json.dumps(retained)
    assert secret not in json.dumps(result.to_document())
    assert sum(request.method == "POST" for request in requests) == 1


@pytest.mark.parametrize("priced", [True, False])
def test_openrouter_oversize_uses_existing_generation_cost_or_remains_unknown(priced, monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda _seconds: None)
    from tests.lab_arena.test_lab_arena_broker import CHAT, CONTEXT
    generation = "gen-original"
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, headers={"x-generation-id": generation}, content=b"oversize")
        assert str(request.url) == br.OPENROUTER_GENERATION_URL + generation
        return httpx.Response(200, json={"data": {
            "id": generation, "total_cost": "0.001234" if priced else "invalid",
        }})
    class SmallDispatch(br.HttpxProviderTransport):
        def send(self, **kwargs):
            if kwargs["method"] == "POST":
                kwargs["max_response_bytes"] = 3
            return super().send(**kwargs)
    transport = SmallDispatch(client_factory=_async_client_factory(respond))
    broker, store, _ = make_broker(transport=transport)
    try:
        result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                                action_sequence=0, timeout_ms=30_000)
    finally:
        transport.close()
    assert requests[0].method == "POST"
    assert all(request.method == "GET" for request in requests[1:])
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    retained = store.calls[result.call["call_identity"]]
    if priced:
        assert result.call["outcome"] == "settled" and retained["actual"] == 1234
        assert retained["terminal"]["provider_cost"]["request_id"] == generation
    else:
        assert result.call["outcome"] == "uncertain"
        assert "actual_microusd" not in result.call
        assert retained["uncertain_doc"]["openrouter_generation_id"] == generation
        assert retained["uncertain_doc"]["response_provenance"] == "response_too_large"
