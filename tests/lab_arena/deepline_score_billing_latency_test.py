"""Completed judge evidence must not wait for Deepline billing finality."""

from dataclasses import replace
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_completed_response_recovery_test import (
    NATIVE, RecoveryTransport, catalog, execute, record,
)
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import _candidate
from tests.lab_arena.deepline_late_response_recovery_test import ResponseStore
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, FakeLedgerStore, make_broker


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(br.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(br.time, "sleep", fake.sleep)
    return fake


class CompletedTransport(RecoveryTransport):
    def __init__(self, *, inline=False, missing_id=False, conflicting_header=False):
        super().__init__(bill_final=False)
        self.conflicting_header = conflicting_header
        self.response = record()["response"]
        if not inline:
            self.response.pop("billing")
        if missing_id:
            self.response.pop("job_id")

    def send(self, **request):
        if request["method"] == "POST":
            self.requests.append(request)
            wire = json.loads(request["body"])
            self.operation, self.provider = wire["operation"], wire["provider"]
            self.document["toolId"] = self.operation
            self.document["executionRecovery"]["idempotencyKey"] = request["headers"]["idempotency-key"]
            headers = {"x-deepline-request-id": "different-request"} if self.conflicting_header else {}
            return br.ProviderResponse(200, headers, json.dumps(self.response).encode())
        return super().send(**request)


class CostStore(ResponseStore):
    def reserve_call(self, **kwargs):
        return FakeLedgerStore.reserve_call(self, **kwargs)

    def reconcile_deepline_cost(self, **kwargs):
        call = self.calls[kwargs["call_identity"]]
        assert call["kind"] == "uncertain"
        assert kwargs["execution_key"] == call["call_doc"]["deepline_execution_key"]
        assert kwargs["credential_fingerprint"] == call["call_doc"]["credential_fingerprint"]
        assert kwargs["recovered_request_id"] == NATIVE
        self.log.append("reconcile")
        self.openrouter_capacity += call["amount"] - kwargs["actual_microusd"]
        call.update(kind="settlement", actual=kwargs["actual_microusd"],
                    terminal=self.responses[kwargs["call_identity"]])
        return {"status": "settled", "actual_microusd": kwargs["actual_microusd"]}


def setup(transport, *, kind="score"):
    broker, store, _ = make_broker(store=CostStore(), transport=transport)
    return broker, store, replace(CONTEXT, kind=kind, deepline_catalog=catalog())


def test_completed_score_returns_after_one_short_exact_read_and_settles_later(clock):
    transport = CompletedTransport()
    broker, store, context = setup(transport)
    initial_capacity = store.openrouter_capacity
    result = execute(broker, context)

    assert result.status == 200
    assert json.loads(result.body)["result"] == record()["response"]["result"]
    assert [request["method"] for request in transport.requests] == ["POST", "GET"]
    assert clock.sleeps == [] and clock.now == 0
    assert 0 < transport.requests[-1]["timeout_seconds"] <= 5
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    identity = result.call["call_identity"]
    saved = store.calls[identity]
    assert saved["kind"] == "uncertain" and saved["amount"] == 20_000
    assert result.call["reserved_microusd"] == 20_000
    assert store.openrouter_capacity == initial_capacity - 20_000
    assert saved["uncertain_doc"]["call_succeeded"] is True
    assert saved["uncertain_doc"]["reason"] == "missing_provider_cost"
    assert saved["uncertain_doc"]["deepline_job_id"] == NATIVE
    assert br._decode_terminal(store.responses[identity])[2] == result.body

    # A replay returns saved evidence and cannot repeat the paid request.
    before = len(transport.requests)
    replay = execute(broker, context)
    assert replay.body == result.body and len(transport.requests) == before
    assert "actual_microusd" not in replay.call
    assert saved["amount"] == 20_000 and store.openrouter_capacity == initial_capacity - 20_000

    # The existing delayed settlement path gets the authenticated exact amount.
    transport.bill_final = True
    reservation = saved["call_doc"]
    candidate = _candidate(
        run_id=context.run_id, round_id=context.round_id,
        assignment_id=context.assignment_id, submission_id=context.submission_id,
        miner_hotkey=context.miner_hotkey, call_identity=identity,
        request_id=NATIVE, operation="parallel_search", billing_provider="parallel",
        credential_fingerprint=reservation["credential_fingerprint"],
        execution_key=reservation["deepline_execution_key"],
    )
    assert broker.reconcile_deepline_cost(candidate) == {"status": "settled", "actual_microusd": 2000}
    settled = execute(broker, context)
    assert settled.body == result.body and settled.call["actual_microusd"] == 2000
    assert store.calls[identity]["amount"] == 20_000
    assert store.openrouter_capacity == initial_capacity - 2000
    assert [request["method"] for request in transport.requests] == ["POST", "GET", "GET"]
    assert store.log.count("dispatch") == 1 and store.log.count("reconcile") == 1


def test_completed_score_without_native_id_keeps_normal_recovery(clock):
    transport = CompletedTransport(missing_id=True)
    broker, store, context = setup(transport)
    result = execute(broker, context)
    # A missing receipt is not itself terminal-success proof, so retain normal
    # recovery rather than make an early completed-response assumption.
    assert result.call["outcome"] == "uncertain"
    assert clock.sleeps and clock.now == 30
    assert not store.responses


def test_completed_score_with_conflicting_receipt_has_one_bounded_key_read_chain(clock):
    transport = CompletedTransport(conflicting_header=True)
    broker, store, context = setup(transport)
    result = execute(broker, context)
    assert result.status == 200 and result.call["outcome"] == "uncertain"
    assert [request["method"] for request in transport.requests] == ["POST", "GET", "GET"]
    assert "/executions/by-key/" in transport.requests[1]["url"]
    assert clock.sleeps == []
    assert all(0 < request["timeout_seconds"] <= 5 for request in transport.requests[1:])
    saved = store.calls[result.call["call_identity"]]
    assert saved["uncertain_doc"]["deepline_job_id"] == NATIVE
    assert "actual_microusd" not in result.call


def test_inline_native_bill_needs_no_readback(clock):
    transport = CompletedTransport(inline=True)
    broker, store, context = setup(transport)
    result = execute(broker, context)
    assert result.status == 200 and result.call["actual_microusd"] == 2000
    assert result.call["outcome"] == "settled"
    assert [request["method"] for request in transport.requests] == ["POST"]
    assert clock.sleeps == [] and store.calls[result.call["call_identity"]]["kind"] == "settlement"


def test_non_score_completed_response_keeps_existing_billing_poll(clock):
    transport = CompletedTransport()
    broker, _, context = setup(transport, kind="execute")
    result = execute(broker, context)
    assert result.status == 200 and result.call["outcome"] == "uncertain"
    assert clock.now == 30 and clock.sleeps
    assert sum(request["method"] == "GET" for request in transport.requests) > 1
    assert sum(request["method"] == "POST" for request in transport.requests) == 1


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b'"scalar"'])
def test_malformed_score_body_stays_fail_closed(clock, body, monkeypatch):
    transport = CompletedTransport()
    original_send = transport.send

    def send(**request):
        response = original_send(**request)
        return br.ProviderResponse(200, {}, body) if request["method"] == "POST" else response

    monkeypatch.setattr(transport, "send", send)
    broker, store, context = setup(transport)
    result = execute(broker, context)
    assert result.status == 502 and not store.responses
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    assert clock.sleeps and clock.now == 30
    assert sum(request["method"] == "POST" for request in transport.requests) == 1


@pytest.mark.parametrize("patch", [
    {"status": "running"}, {"error": {"code": "UPSTREAM_FAILURE"}},
    {"tool_error": {"code": "UPSTREAM_FAILURE"}}, {"result": None},
    {"result": {"data": "not usable judge evidence"}},
])
def test_nonterminal_error_or_unusable_result_keeps_existing_recovery(clock, patch):
    transport = CompletedTransport()
    transport.response.update(patch)
    broker, _, context = setup(transport)
    result = execute(broker, context)
    assert clock.now == 30 and clock.sleeps
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    assert sum(request["method"] == "POST" for request in transport.requests) == 1


@pytest.mark.parametrize("patch", [
    None, {"status": "running"}, {"error": {"code": "UPSTREAM_FAILURE"}},
    {"result": None}, {"result": {"data": "invalid"}},
])
def test_routed_judge_scrape_returns_only_usable_completed_evidence(clock, patch):
    transport = CompletedTransport()
    transport.response["result"] = {"data": {
        "rawHtml": "<html>Official company evidence</html>",
        "metadata": {"url": "https://example.com/", "sourceURL": "https://example.com/",
                     "statusCode": 200},
    }}
    if patch:
        transport.response.update(patch)
    broker, store, _ = make_broker(store=CostStore(), transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key")
    result = broker.execute(replace(CONTEXT, kind="score", round_id="arena-2026-10-06"),
        operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/"},
        action_sequence=0, timeout_ms=60_000)

    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    assert sum(request["method"] == "POST" for request in transport.requests) == 1
    if patch is None:
        assert result.status == 200 and b"Official company evidence" in result.body
        assert clock.sleeps == []
        assert [request["method"] for request in transport.requests] == ["POST", "GET"]
        assert br._decode_terminal(store.responses[result.call["call_identity"]])[2] == result.body
    else:
        assert result.status == 502 and not store.responses
        assert clock.now == 30 and clock.sleeps
