"""Large Firecrawl replies still need an exact, bounded billing readback."""

import json
from urllib.parse import unquote

import httpx
import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    NATIVE_REQUEST_ID, ROUND_ID, ReconciliationStore, SECRET, _broker, _candidate,
    _score_context,
)
from tests.lab_arena.deepline_execution_recovery_test import KEY, exact_bill
from tests.lab_arena.test_lab_arena_broker import (
    DL_KEY, FakeLedgerStore, _async_client_factory, make_broker,
)


def _original_response():
    # Larger than the ordinary 4 MiB readback, within the existing 16 MiB
    # Firecrawl execution envelope. All identities and content are synthetic.
    return {
        "status": "completed",
        "result": {"data": {
            "rawHtml": "<html>" + "x" * (4 * 1024 * 1024) + "</html>",
            "metadata": {
                "url": "https://example.com/about",
                "sourceURL": "https://example.com/about",
                "statusCode": 200,
            },
        }},
    }


def _lookup(key=KEY, **patch):
    document = {
        "requestId": NATIVE_REQUEST_ID,
        "toolId": "firecrawl_scrape",
        "executionRecovery": {"idempotencyKey": key, "state": "completed"},
        "responseStatus": 200,
        "response": _original_response(),
    }
    document.update(patch)
    assert len(json.dumps(document).encode()) > 4 * 1024 * 1024
    return document


def _transport(respond):
    return br.HttpxProviderTransport(client_factory=_async_client_factory(respond))


def test_large_completed_firecrawl_cost_is_read_with_exact_bill_during_execution():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=_original_response())
        if "/executions/by-key/" in str(request.url):
            key = unquote(str(request.url).rsplit("/", 1)[-1])
            return httpx.Response(200,
                headers={"x-deepline-idempotency-supported": "true"},
                json=_lookup(key))
        assert str(request.url).startswith(br.DEEPLINE_EXACT_BILLING_URL)
        return httpx.Response(200, json=exact_bill(status="completed"))

    transport = _transport(respond)
    store = FakeLedgerStore(openrouter_capacity=10_000_000)
    broker, _, _ = make_broker(
        store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "host",
        host_shadow_score_compat_round_id=ROUND_ID,
        host_shadow_score_compat_source_round_id=ROUND_ID,
    )
    try:
        result = broker.execute(
            _score_context(), operation_id="scrapingdog.scrape",
            parameters={"url": "https://example.com/about"},
            action_sequence=0, timeout_ms=30_000,
        )
    finally:
        transport.close()
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 3_000
    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert store.calls[result.call["call_identity"]]["actual"] == 3_000


def test_delayed_large_firecrawl_lookup_keeps_default_cap_and_settles_exactly_once():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        if "/executions/by-key/" in str(request.url):
            return httpx.Response(200,
                headers={"x-deepline-idempotency-supported": "true"},
                json=_lookup())
        assert str(request.url) == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE_REQUEST_ID, safe="")
        return httpx.Response(200, json=exact_bill())

    transport = _transport(respond)
    store = ReconciliationStore()
    try:
        native, cost = br._deepline_exact_readback(
            transport=transport, secret=SECRET, request_id=None,
            execution_key=KEY, operation="firecrawl_scrape", provider="firecrawl",
            reconciliation_deadline=br.time.monotonic() + 5, poll=False,
        )
        assert native is None and cost is None
        assert len(requests) == 1  # The unchanged default rejects the large lookup.
        result = _broker(store, transport).reconcile_deepline_cost(
            _candidate(execution_key=KEY, billing_provider="firecrawl")
        )
    finally:
        transport.close()
    assert result["status"] == "settled"
    assert [request.method for request in requests] == ["GET", "GET", "GET"]
    assert len(store.calls) == 1
    assert store.calls[0]["recovered_request_id"] == NATIVE_REQUEST_ID
    assert store.calls[0]["actual_microusd"] == 3_000


@pytest.mark.parametrize("change", [
    "foreign_key", "foreign_tool", "synthetic_id", "foreign_bill",
    "credential_echo", "missing_support_header",
])
def test_large_lookup_cannot_settle_from_forged_identity_or_charge(change):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        if "/executions/by-key/" in str(request.url):
            patch = {}
            if change == "foreign_key":
                patch["executionRecovery"] = {
                    "idempotencyKey": "arena:" + "c" * 64, "state": "completed",
                }
            elif change == "foreign_tool":
                patch["toolId"] = "another_tool"
            elif change == "synthetic_id":
                patch["requestId"] = "ctx-tool-" + "a" * 32
            elif change == "credential_echo":
                patch["provider_metadata"] = SECRET
            return httpx.Response(200,
                headers=({} if change == "missing_support_header" else
                         {"x-deepline-idempotency-supported": "true"}),
                json=_lookup(**patch))
        return httpx.Response(200, json=exact_bill(
            request_id="another-native-id" if change == "foreign_bill" else NATIVE_REQUEST_ID))

    transport = _transport(respond)
    store = ReconciliationStore()
    try:
        result = _broker(store, transport).reconcile_deepline_cost(
            _candidate(execution_key=KEY, billing_provider="firecrawl")
        )
    finally:
        transport.close()
    assert result == {"status": "pending"}
    assert store.calls == []
    assert all(request.method == "GET" for request in requests)
    assert len(requests) == (2 if change == "foreign_bill" else 1)


def test_firecrawl_lookup_remains_bounded_above_existing_envelope_allowance():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert "/executions/by-key/" in str(request.url)
        return httpx.Response(200,
            headers={"x-deepline-idempotency-supported": "true"},
            content=b"x" * (br._DEEPLINE_FIRECRAWL_ENVELOPE_MAX_BYTES + 16_385))

    transport = _transport(respond)
    store = ReconciliationStore()
    try:
        result = _broker(store, transport).reconcile_deepline_cost(
            _candidate(execution_key=KEY, billing_provider="firecrawl")
        )
    finally:
        transport.close()
    assert result == {"status": "pending"}
    assert len(requests) == 1 and store.calls == []


def test_other_deepline_tools_keep_ordinary_lookup_limit():
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert "/executions/by-key/" in str(request.url)
        return httpx.Response(200,
            headers={"x-deepline-idempotency-supported": "true"},
            json=_lookup(toolId="exa_search"))

    transport = _transport(respond)
    store = ReconciliationStore()
    try:
        result = _broker(store, transport).reconcile_deepline_cost(
            _candidate(operation="exa_search", execution_key=KEY,
                       billing_provider="exa")
        )
    finally:
        transport.close()
    assert result == {"status": "pending"}
    assert len(requests) == 1 and store.calls == []


@pytest.mark.parametrize("invalid_bound", [True, 0, 4 * 1024 * 1024, 32 * 1024 * 1024])
def test_large_lookup_bound_is_fixed_and_rejects_other_values(invalid_bound):
    class NoRequests:
        def send(self, **_request):
            raise AssertionError("invalid bound must not dispatch")

    assert br._deepline_exact_readback(
        transport=NoRequests(), secret=SECRET, request_id=None,
        execution_key=KEY, operation="firecrawl_scrape",
        reconciliation_deadline=br.time.monotonic() + 5, poll=False,
        lookup_max_response_bytes=invalid_bound,
    ) == (None, None)
