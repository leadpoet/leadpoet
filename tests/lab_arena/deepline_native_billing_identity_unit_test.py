"""Retain a native billing receipt without accepting a different call's cost."""

import json

import httpx
import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    NATIVE_REQUEST_ID, ReconciliationStore, SECRET, _broker, _candidate,
    _ledger_entry, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import (
    DL_KEY, FakeLedgerStore, _ReadTimeoutAfterHeaders, _TrickleUntilCancelled,
    _async_client_factory, make_broker,
)


@pytest.mark.parametrize("headers, expected", [
    ({}, None),
    ({"x-vercel-id": NATIVE_REQUEST_ID}, NATIVE_REQUEST_ID),
    ({"x-vercel-id": "iad1::" + NATIVE_REQUEST_ID}, NATIVE_REQUEST_ID),
    ({"x-deepline-request-id": NATIVE_REQUEST_ID}, NATIVE_REQUEST_ID),
    ({"x-vercel-id": "bad/edge::" + NATIVE_REQUEST_ID}, None),
    ({"x-vercel-id": "arbitrary"}, None),
    ({"x-deepline-request-id": "invalid", "x-vercel-id": NATIVE_REQUEST_ID}, None),
    ({"x-deepline-request-id": "ctx-tool-" + "a" * 32,
      "x-vercel-id": NATIVE_REQUEST_ID}, None),
    ([("x-vercel-id", NATIVE_REQUEST_ID),
      ("x-vercel-id", "iad1::other-1789361973860-8634524a6f3a")], None),
])
def test_broken_body_retains_only_one_valid_native_header(headers, expected):
    transport = br.HttpxProviderTransport(client_factory=_async_client_factory(
        lambda _: httpx.Response(
            200, headers=headers, stream=_ReadTimeoutAfterHeaders(),
        )
    ))
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(method="POST", url="https://code.deepline.com/api/v2/execute",
                           headers={}, body=b"{}", timeout_seconds=1)
        assert str(raised.value) == "ReadTimeout"
        assert raised.value.deepline_job_id == expected
    finally:
        transport.close()


def test_foreign_provider_header_is_not_a_deepline_receipt():
    transport = br.HttpxProviderTransport(client_factory=_async_client_factory(
        lambda _: httpx.Response(
            200, headers={"x-vercel-id": NATIVE_REQUEST_ID},
            stream=_ReadTimeoutAfterHeaders(),
        )
    ))
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(method="POST", url="https://api.scrapingdog.com/scrape",
                           headers={}, body=b"{}", timeout_seconds=1)
        assert raised.value.deepline_job_id is None
    finally:
        transport.close()


def test_absolute_deadline_retains_deepline_header_and_closes_stream():
    stream = _TrickleUntilCancelled()
    transport = br.HttpxProviderTransport(client_factory=_async_client_factory(
        lambda _: httpx.Response(
            200,
            headers={"x-vercel-id": "iad1::" + NATIVE_REQUEST_ID},
            stream=stream,
        )
    ))
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(
                method="POST",
                url="https://code.deepline.com/api/v2/execute",
                headers={},
                body=b"{}",
                timeout_seconds=0.06,
            )
    finally:
        transport.close()
    assert str(raised.value) == "ReadTimeout"
    assert raised.value.deepline_job_id == NATIVE_REQUEST_ID
    assert stream.closed.is_set()


@pytest.mark.parametrize("known_cost", [True, False])
def test_lost_body_uses_native_receipt_and_never_repeats_paid_call(known_cost):
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, headers={"x-vercel-id": "iad1::" + NATIVE_REQUEST_ID},
                                  stream=_ReadTimeoutAfterHeaders())
        return httpx.Response(200, json={
            "entries": [_ledger_entry(credits=0.02, request_id=NATIVE_REQUEST_ID)]
            if known_cost else "invalid", "has_more": False,
        })
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(respond)
    )
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    try:
        result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                                parameters={"url": "https://example.com/about"},
                                action_sequence=0, timeout_ms=30_000)
    finally:
        transport.close()
    assert [r.method for r in requests] == ["POST", "GET"]
    retained = store.calls[result.call["call_identity"]]
    assert result.call["outcome"] == ("settled" if known_cost else "uncertain")
    if known_cost:
        assert retained["actual"] == 2_000
        assert store.openrouter_capacity == 49_943_650
        assert retained["terminal"]["provider_cost"]["request_id"] == NATIVE_REQUEST_ID
    else:
        assert retained["uncertain_doc"]["deepline_job_id"] == NATIVE_REQUEST_ID
        assert store.openrouter_capacity == 0
    assert NATIVE_REQUEST_ID not in json.dumps(result.to_document())


def test_credential_shaped_like_native_id_is_never_retained():
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, headers={"x-deepline-request-id": NATIVE_REQUEST_ID},
                                  stream=_ReadTimeoutAfterHeaders())
        return httpx.Response(200, json={"entries": "invalid"})
    transport = br.HttpxProviderTransport(
        client_factory=_async_client_factory(respond)
    )
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: NATIVE_REQUEST_ID,
        funding_source_for=lambda _context: "miner_key")
    try:
        result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                                parameters={"url": "https://example.com/about"},
                                action_sequence=0, timeout_ms=30_000)
    finally:
        transport.close()
    assert result.call["outcome"] == "uncertain"
    assert NATIVE_REQUEST_ID not in json.dumps(store.calls)
    assert NATIVE_REQUEST_ID not in json.dumps(result.to_document())


def test_delayed_native_id_settles_exact_authenticated_cost():
    class NativeLedger:
        def __init__(self):
            self.sent = []
        def send(self, **kwargs):
            self.sent.append(kwargs)
            assert kwargs["method"] == "GET"
            assert kwargs["headers"]["authorization"] == "Bearer " + SECRET
            return br.ProviderResponse(200, {}, json.dumps({
                "entries": [_ledger_entry(request_id=NATIVE_REQUEST_ID)], "has_more": False,
            }).encode())
    store, transport = ReconciliationStore(), NativeLedger()
    result = _broker(store, transport).reconcile_deepline_cost(_candidate(request_id=NATIVE_REQUEST_ID))
    assert result["status"] == "settled"
    assert store.calls[0]["request_id"] == NATIVE_REQUEST_ID
    assert len(transport.sent) == 1
