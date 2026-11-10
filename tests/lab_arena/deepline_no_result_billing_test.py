"""A final-priced negative lookup must not depend on a second billing read."""

import base64
from dataclasses import replace
import json

from fastapi.testclient import TestClient
import pytest

from lab_arena import broker as br, contracts
from lab_arena.api import create_app
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, ZeroReservationLedgerStore, make_broker


NATIVE_ID = "public-execute:11111111-2222-4333-8444-555555555555"
TOOL = "contextdev_post_news_search"
PARAMETERS = {"tool": TOOL, "payload": {"query": "example"}}


def negative_result():
    # The retained production shape, with private identifiers replaced.
    return {
        "job_id": NATIVE_ID, "status": "no_result",
        "result": {"data": {"message": "No company found for domain example.com",
                             "error_code": "NOT_FOUND", "key_metadata": {},
                             "request_id": "source-request"}},
        "billing": {"credits_charged": 0.007, "cost_usd": 0.0007,
                    "pricing_status": "final", "settlement_status": "queued"},
        "executionRecovery": {"idempotencyKey": None, "state": "completed",
                              "replayed": False},
    }


class NegativeTransport:
    def __init__(self, patch=None, *, billing_available=False, status=200, headers=None, provenance=None):
        self.patch = patch or (lambda document: None)
        self.billing_available, self.status = billing_available, status
        self.headers, self.provenance = headers or {}, provenance
        self.sent = []

    def send(self, **request):
        self.sent.append(request)
        if request["method"] == "POST":
            self.document = negative_result()
            self.document["executionRecovery"]["idempotencyKey"] = request["headers"]["idempotency-key"]
            self.patch(self.document)
            self.body = json.dumps(self.document).encode()
            return br.ProviderResponse(self.status, {"content-type": "application/json", **self.headers},
                                       self.body, self.provenance)
        assert request["method"] == "GET"
        if not self.billing_available:
            raise br.ProviderTransportError("ReadTimeout")
        return br.ProviderResponse(200, {}, json.dumps({"recent": {"request_id": NATIVE_ID,
            "entries": [{"id": "charge-1", "request_id": NATIVE_ID, "provider": "contextdev",
                         "operation": TOOL, "status": "completed", "charge_state": "posted",
                         "charge_finality": "final", "credits": "0.007", "delta": "-0.007"}]}}).encode())


def setup(transport, *, baseline=False, kind="execute", funding_source="host", store=None):
    context = replace(CONTEXT, kind=kind,
                      submission_id="baseline-2026-11-10" if baseline else "sub-miner",
                      deepline_catalog=frozen(row(TOOL, provider="contextdev")))
    broker, store, _ = make_broker(
        store=store or ZeroReservationLedgerStore(), transport=transport,
        credential_for=lambda *_: DL_KEY,
        provider_funding_source_for=lambda *_: funding_source,
    )

    class Service:
        def handle_provider(self, run_id, lease, frame, cancel_requested=None):
            assert run_id == context.run_id and lease == "a" * 64
            return broker.execute(context, **frame, cancel_requested=cancel_requested).to_document()

    client = TestClient(create_app(Service()))
    return client, broker, store, context


def send(client, context):
    response = client.post(f"/arena/v1/runs/{context.run_id}/provider",
                           headers={"x-lab-arena-lease": "a" * 64},
                           json={"operation_id": "deepline.execute", "parameters": PARAMETERS,
                                 "action_sequence": 0, "timeout_ms": 30_000})
    assert response.status_code == 200
    document = response.json()
    return document, base64.b64decode(document["body_b64"])


@pytest.fixture(autouse=True)
def bounded_failed_readback(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)


@pytest.mark.parametrize("baseline", [False, True], ids=["miner", "baseline"])
@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_final_negative_is_delivered_with_exact_charge_without_billing_read(baseline, kind, funding_source):
    transport = NegativeTransport()
    client, _, store, context = setup(transport, baseline=baseline, kind=kind, funding_source=funding_source)
    with client:
        result, body = send(client, context)
        replay, replay_body = send(client, context)
    assert result["status"] == replay["status"] == 200
    assert body == replay_body == transport.body
    assert result["call"]["actual_microusd"] == 700
    assert result["call"]["funding_source"] == funding_source
    assert result["call"]["outcome"] == "settled"
    terminal = store.calls[result["call"]["call_identity"]]["terminal"]
    assert terminal["call_succeeded"] is False
    assert base64.b64decode(terminal["body_b64"]) == transport.body
    assert [request["method"] for request in transport.sent] == ["POST"]
    assert store.log == ["reserve", "dispatch", "settle", "reserve"]


@pytest.mark.parametrize("patch", [
    lambda d: d.pop("billing"),
    lambda d: d["billing"].pop("pricing_status"),
    lambda d: d["billing"].update(pricing_status="pending"),
    lambda d: d["billing"].pop("cost_usd"),
    lambda d: d["billing"].update(cost_usd=9),
    lambda d: d["billing"].update(credits_charged=True),
    lambda d: d["billing"].update(credits_charged=-1),
    lambda d: d["billing"].update(credits_charged="NaN"),
    lambda d: d["billing"].update(credits=9),
    lambda d: d["billing"].update(billing_mode="async_hold"),
    lambda d: d["executionRecovery"].update(idempotencyKey="arena:" + "f" * 64),
    lambda d: d["executionRecovery"].update(state="pending"),
    lambda d: d.pop("executionRecovery"),
    lambda d: d.update(request_id="conflicting-request"),
    lambda d: d.update(error={"code": "UPSTREAM_ERROR"}),
    lambda d: d.update(tool_error={"code": "UPSTREAM_ERROR"}),
    lambda d: d.update(result={"data": "malformed"}),
    lambda d: d.update(operation="different-operation"),
    lambda d: d.update(status="failed"),
])
def test_unproved_negative_keeps_cost_uncertain_and_does_not_replay_paid_request(patch):
    transport = NegativeTransport(patch)
    client, _, store, context = setup(transport)
    with client:
        result, _ = send(client, context)
        before = len(transport.sent)
        replay, _ = send(client, context)
    assert result["status"] == 502 and result["call"]["outcome"] == "uncertain"
    assert "actual_microusd" not in result["call"]
    assert replay["status"] == 409 and replay["call"]["error_code"] == "call_uncertain"
    assert all(request["method"] == "GET" for request in transport.sent[before:])
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
    saved = store.calls[result["call"]["call_identity"]]["uncertain_doc"]
    receipt = saved["upstream_response_receipt"]
    assert base64.b64decode(receipt["body_b64"]) == transport.body
    assert receipt["body_sha256"] == contracts.hash_bytes(transport.body)


def test_exact_readback_can_still_deliver_negative_without_trusting_inline_price():
    transport = NegativeTransport(lambda d: d["billing"].pop("pricing_status"), billing_available=True)
    client, _, store, context = setup(transport)
    with client:
        result, body = send(client, context)
    assert result["status"] == 200 and body == transport.body
    assert result["call"]["actual_microusd"] == 700
    assert result["call"]["cost_basis"] == "deepline_exact_request_credits_x_0.10_usd"
    assert store.calls[result["call"]["call_identity"]]["terminal"]["call_succeeded"] is False


@pytest.mark.parametrize("data", [[], {}])
@pytest.mark.parametrize("credits,usd,amount", [(0, 0, 0), ("0.0000001", "0.00000001", 1)])
def test_explicit_final_zero_or_fractional_price_and_empty_search_data(data, credits, usd, amount):
    def patch(d):
        d["result"]["data"] = data
        d["billing"].update(credits_charged=credits, cost_usd=usd)
    transport = NegativeTransport(patch)
    client, _, store, context = setup(transport)
    with client:
        result, body = send(client, context)
    assert result["status"] == 200 and body == transport.body
    assert result["call"]["actual_microusd"] == amount
    assert store.calls[result["call"]["call_identity"]]["terminal"]["call_succeeded"] is False
    assert len(transport.sent) == 1


def test_completed_positive_keeps_existing_inline_billing_and_success_flag():
    def patch(d):
        d.update(status="completed", result={"data": [{"title": "News", "url": "https://example.com"}]})
    transport = NegativeTransport(patch)
    client, _, store, context = setup(transport)
    with client:
        result, body = send(client, context)
    assert result["status"] == 200 and body == transport.body
    assert result["call"]["actual_microusd"] == 700
    assert store.calls[result["call"]["call_identity"]]["terminal"]["call_succeeded"] is True
    assert len(transport.sent) == 1


@pytest.mark.parametrize("status", [429, 502])
def test_real_provider_failure_with_no_bill_keeps_failure_and_uncertainty(status):
    def patch(d):
        d.clear()
        d.update(request_id=NATIVE_ID, error={"code": "RATE_LIMIT" if status == 429 else "UPSTREAM_ERROR"})
    transport = NegativeTransport(patch, status=status)
    client, _, store, context = setup(transport)
    with client:
        result, _ = send(client, context)
    assert result["status"] == 502 and result["call"]["outcome"] == "uncertain"
    assert "actual_microusd" not in result["call"]
    assert store.calls[result["call"]["call_identity"]]["uncertain_doc"]["call_succeeded"] is False


@pytest.mark.parametrize("options", [{"headers": {"x-deepline-request-id": "wrong-id"}},
                                      {"provenance": "response_too_large"}])
def test_conflicting_or_synthetic_negative_is_not_inline_bill_proof(options):
    transport = NegativeTransport(**options)
    client, _, _, context = setup(transport)
    with client:
        result, _ = send(client, context)
    assert result["status"] == 502 and result["call"]["outcome"] == "uncertain"
    assert "actual_microusd" not in result["call"]


def test_credential_echo_remains_private():
    transport = NegativeTransport(lambda d: d["result"]["data"].update(message=DL_KEY))
    client, _, _, context = setup(transport)
    with client:
        result, body = send(client, context)
    assert result["status"] == 502 and DL_KEY.encode() not in body
    assert DL_KEY not in json.dumps(result)
