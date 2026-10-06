"""Lost responses recover by saved key and exact ID without a paid retry."""

import json
from urllib.parse import unquote

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    CALL_IDENTITY, NATIVE_REQUEST_ID, ReconciliationStore, SECRET,
    _broker, _candidate, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeLedgerStore, make_broker


KEY = "arena:" + CALL_IDENTITY[7:]


def exact_bill(request_id=NATIVE_REQUEST_ID, **patch):
    row = {
        "id": "exact-usage-row", "request_id": request_id,
        "provider": "firecrawl", "operation": "firecrawl_scrape",
        "status": "error", "charge_state": "posted", "charge_finality": "final",
        "billing_mode": "deduct_on_settle", "credits": "0.03", "delta": "-0.03",
        "batch_count": None, "metadata": {},
    }
    row.update(patch)
    return {"org_id": "organization-1", "recent": {"request_id": request_id, "entries": [row]}}


class RecoveryTransport:
    def __init__(self, *, store=None, lost=True, record_patch=None, bill_patch=None, supported=True):
        self.store, self.lost = store, lost
        self.record_patch, self.bill_patch = record_patch or {}, bill_patch or {}
        self.supported = supported
        self.sent = []
        self.execution_key = KEY

    def send(self, **request):
        self.sent.append(request)
        if request["method"] == "POST":
            self.execution_key = request["headers"]["idempotency-key"]
            assert self.store.log == ["reserve", "dispatch"]
            saved = next(iter(self.store.calls.values()))["call_doc"]
            assert saved["deepline_execution_key"] == self.execution_key
            if self.lost:
                raise br.ProviderTransportError("ReadTimeout")
            return br.ProviderResponse(200, {}, json.dumps({
                "job_id": NATIVE_REQUEST_ID, "status": "completed",
                "toolResponse": {"success": True, "data": {"markdown": "Company text"}},
            }).encode())
        assert request["method"] == "GET"
        assert "/billing/ledger" not in request["url"]
        assert "recent_limit" not in request["url"]
        if "/executions/by-key/" in request["url"]:
            assert unquote(request["url"].split("/by-key/", 1)[1]) == self.execution_key
            record = {"requestId": NATIVE_REQUEST_ID, "toolId": "firecrawl_scrape",
                      "executionRecovery": {"idempotencyKey": self.execution_key, "state": "completed"}}
            record.update(self.record_patch)
            return br.ProviderResponse(200,
                {"X-Deepline-Idempotency-Supported": "true"} if self.supported else {},
                json.dumps(record).encode())
        assert request["url"] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE_REQUEST_ID, safe="")
        return br.ProviderResponse(200, {}, json.dumps(exact_bill(**self.bill_patch)).encode())


def execute(broker):
    return broker.execute(_score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=30_000)


def test_timeout_key_is_durable_before_dispatch_and_cost_is_exact_once():
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    transport = RecoveryTransport(store=store)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    first = execute(broker)
    assert first.call["outcome"] == "settled"
    assert first.status == 502  # A lost result remains a failed call even with a known charge.
    assert first.call["actual_microusd"] == 3_000
    assert [request["method"] for request in transport.sent] == ["POST", "GET", "GET", "GET"]
    call = store.calls[first.call["call_identity"]]
    assert call["terminal"]["provider_cost"]["request_id"] == NATIVE_REQUEST_ID
    assert transport.execution_key not in json.dumps(first.to_document())
    second = execute(broker)
    assert second.call["idempotent"] is True
    assert len(transport.sent) == 4
    assert store.openrouter_capacity == 49_942_650


def test_completed_response_without_bill_uses_only_its_exact_id():
    store = FakeLedgerStore()
    transport = RecoveryTransport(store=store, lost=False)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    result = execute(broker)
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 3_000
    assert len(transport.sent) == 2
    assert "/usage?request_id=" in transport.sent[1]["url"]


@pytest.mark.parametrize("options", [
    {"supported": False},
    {"record_patch": {"executionRecovery": {"idempotencyKey": "other-key"}}},
    {"record_patch": {"toolId": "firecrawl_search"}},
    {"record_patch": {"requestId": "bad/id"}},
    {"bill_patch": {"request_id": "another-request"}},
    {"bill_patch": {"provider": "exa"}},
    {"bill_patch": {"operation": "firecrawl_search"}},
])
def test_unknown_or_unbound_key_and_bill_do_not_settle_or_retry(options):
    store = FakeLedgerStore()
    transport = RecoveryTransport(store=store, **options)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    result = execute(broker)
    assert result.call["outcome"] == "uncertain"
    assert len([r for r in transport.sent if r["method"] == "POST"]) == 1
    uncertain = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert uncertain["deepline_execution_key"] == transport.execution_key
    before = len(transport.sent)
    assert execute(broker).call["outcome"] == "uncertain"
    assert len(transport.sent) == before


def test_pending_async_zero_is_not_final_charge(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = RecoveryTransport(bill_patch={"billing_mode": "async_hold", "credits": 0, "delta": 0})
    request_id, cost = br._deepline_exact_readback(transport=transport, secret=SECRET,
        request_id=None, execution_key=KEY, operation="firecrawl_scrape",
        reconciliation_deadline=br.time.monotonic() + 1)
    assert request_id == NATIVE_REQUEST_ID and cost is None
    assert len(transport.sent) == 2


def test_restart_recovery_uses_saved_key_and_keeps_immutable_sql_identity():
    store, transport = ReconciliationStore(), RecoveryTransport()
    result = _broker(store, transport).reconcile_deepline_cost(_candidate(execution_key=KEY))
    assert result["status"] == "settled"
    assert all(request["method"] == "GET" for request in transport.sent)
    assert store.calls[0]["request_id"] == _candidate()["request_id"]
    assert store.calls[0]["execution_key"] == KEY
    assert store.calls[0]["recovered_request_id"] == NATIVE_REQUEST_ID


def test_unkeyed_batch_recovers_only_the_saved_native_billing_identity():
    class ExactTransport:
        def __init__(self):
            self.sent = []

        def send(self, **request):
            self.sent.append(request)
            assert request["method"] == "GET"
            assert request["url"] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE_REQUEST_ID, safe="")
            return br.ProviderResponse(200, {}, json.dumps(exact_bill(
                operation="firecrawl_batch_scrape",
            )).encode())

    store, transport = ReconciliationStore(), ExactTransport()
    candidate = _candidate(request_id=NATIVE_REQUEST_ID, operation="firecrawl_batch_scrape",
        execution_key=None, billing_provider="firecrawl", operation_aliases=[])
    result = _broker(store, transport).reconcile_deepline_cost(candidate)
    assert result["status"] == "settled"
    assert len(transport.sent) == 1
    assert store.calls[0]["request_id"] == NATIVE_REQUEST_ID
    assert store.calls[0]["execution_key"] is None
    assert store.calls[0]["recovered_request_id"] == NATIVE_REQUEST_ID


def test_unkeyed_lost_batch_without_native_receipt_is_not_dispatched_again():
    transport = RecoveryTransport()
    request_id, cost = br._deepline_exact_readback(
        transport=transport, secret=SECRET, request_id=_candidate()["request_id"],
        execution_key=None, operation="firecrawl_batch_scrape", provider="firecrawl",
        reconciliation_deadline=br.time.monotonic() + 1,
    )
    assert request_id is None and cost is None
    assert transport.sent == []


@pytest.mark.parametrize("patch", [
    {"execution_key": "arena:" + "a" * 64},
    {"credential_fingerprint": br._credential_fingerprint("different-credential")},
])
def test_restart_wrong_binding_permits_no_provider_lookup(patch):
    store, transport = ReconciliationStore(), RecoveryTransport()
    candidate = _candidate(**dict({"execution_key": KEY}, **patch))
    result = _broker(store, transport).reconcile_deepline_cost(candidate)
    assert result["status"] in ("invalid", "credential_mismatch")
    assert transport.sent == [] and store.calls == []
