"""A completed native failure does not need a second worker response lookup."""
from __future__ import annotations

import base64
import copy
import json

import pytest

from lab_arena import broker as br, contracts, operations, training_observations as obs
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import _candidate
from tests.lab_arena.deepline_terminal_error_test import (
    ErrorTransport, REQUEST_ID, RetainedErrorStore, billing, execute,
)
from tests.lab_arena.deepline_worker_same_action_recovery_test import (
    Api, PARAMETERS, document, worker,
)
from tests.lab_arena.test_lab_arena_broker import DL_KEY


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(br.time, "monotonic", value.monotonic)
    monkeypatch.setattr(br.time, "sleep", value.sleep)
    return value


class CompleteErrorTransport(ErrorTransport):
    def __init__(self, status=429, *, mutate=None, clock=None, provenance=None):
        super().__init__(status)
        self.mutate, self.clock, self.provenance = mutate, clock, provenance

    def send(self, **request):
        if request["method"] == "POST":
            self.sent.append(request)
            self.key = request["headers"]["idempotency-key"]
            self.native = {
                "request_id": REQUEST_ID,
                "error": "Native request failed",
                "executionRecovery": {
                    "idempotencyKey": self.key, "state": "completed", "replayed": False,
                },
            }
            if self.mutate:
                self.mutate(self.native)
            if self.clock:
                self.clock.now += 0.663
            return br.ProviderResponse(self.status, {"content-type": "application/json"},
                json.dumps(self.native).encode(), self.provenance)
        if "/executions/by-key/" in request["url"]:
            self.sent.append(request)
            return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"},
                json.dumps({
                    "requestId": REQUEST_ID, "toolId": "exa_search",
                    "executionRecovery": {"idempotencyKey": self.key, "state": "completed"},
                    "responseStatus": self.status, "response": self.native,
                }).encode())
        return super().send(**request)


class SettlingStore(RetainedErrorStore):
    def reconcile_deepline_cost(self, **kwargs):
        self.reconciled = kwargs
        self.calls[kwargs["call_identity"]].update(kind="settlement",
            actual=kwargs["actual_microusd"], terminal=br._terminal_response_document(
                502, {"content-type": "application/json"},
                operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False))
        return {"status": "settled", "actual_microusd": kwargs["actual_microusd"]}


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("status", [400, 404, 422, 429, 500, 502, 503])
def test_bound_complete_native_failure_is_a_delivery_fact_not_a_bill(clock, kind, status):
    transport = CompleteErrorTransport(status, clock=clock)
    result, _, store, _, _ = execute(transport, kind=kind)
    assert result.call.get("deepline_terminal_failure") is True
    assert result.call["provider_status"] == status
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert saved["call_succeeded"] is False
    assert "deepline_terminal_failure" not in saved
    upstream = saved["upstream_response_receipt"]
    assert upstream["wire_body_sha256"] == contracts.hash_bytes(json.dumps(transport.native).encode())
    # Existing request errors get a short billing read; throttles/outages keep
    # their full initial polling window, independently of the delivery fact.
    assert clock.now == pytest.approx(0.663 if status in (400, 404, 422) else 30.663)


@pytest.mark.parametrize("change", [
    "missing", "wrong_key", "running", "malformed", "missing_identity",
    "conflicting_identity", "missing_error", "result", "pending_body", "spoof_fact",
    "wrong_operation", "wrong_tool",
])
def test_unproved_or_contradictory_response_does_not_create_fact(clock, change):
    def mutate(body):
        if change == "missing": body.pop("executionRecovery")
        elif change == "wrong_key": body["executionRecovery"]["idempotencyKey"] = "arena:" + "f" * 64
        elif change == "running": body["executionRecovery"]["state"] = "running"
        elif change == "malformed": body["executionRecovery"] = []
        elif change == "missing_identity": body.pop("request_id")
        elif change == "conflicting_identity": body["requestId"] = "other-request"
        elif change == "missing_error": body.pop("error")
        elif change == "result": body["result"] = {"data": {"value": "success-shaped"}}
        elif change == "pending_body": body["status"] = "running"
        elif change == "wrong_operation": body["operation"] = "another_operation"
        elif change == "wrong_tool": body["toolId"] = "another_tool"
        elif change == "spoof_fact":
            body.pop("executionRecovery")
            body["deepline_terminal_failure"] = True
    result, _, _, _, _ = execute(CompleteErrorTransport(mutate=mutate, clock=clock))
    assert "deepline_terminal_failure" not in result.call


def test_conflicting_native_header_does_not_create_fact(clock):
    class ConflictingHeader(CompleteErrorTransport):
        def send(self, **request):
            response = super().send(**request)
            if request["method"] == "POST":
                return br.ProviderResponse(response.status,
                    {**response.headers, "x-deepline-request-id": "other-request"}, response.body)
            return response
    result, _, _, _, _ = execute(ConflictingHeader(clock=clock))
    assert "deepline_terminal_failure" not in result.call


@pytest.mark.parametrize("billed", [False, True])
def test_error_headers_then_body_loss_or_billed_missing_reply_stays_recoverable(clock, billed):
    class LostBody(CompleteErrorTransport):
        def send(self, **request):
            if request["method"] == "POST":
                super().send(**request)
                raise br.ProviderTransportError("ReadTimeout", observed_status=429)
            if "/executions/by-key/" in request["url"]:
                self.sent.append(request)
                return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"},
                    json.dumps({"requestId": REQUEST_ID, "toolId": "exa_search",
                        "executionRecovery": {"idempotencyKey": self.key, "state": "running"}}).encode())
            return super().send(**request)
    transport = LostBody(clock=clock)
    if billed: transport.bill = billing(credits="0.5")
    result, _, _, _, _ = execute(transport)
    assert "deepline_terminal_failure" not in result.call
    assert result.call["transport_error_class"] == "ReadTimeout"
    if billed:
        assert result.call["deepline_response_missing"] is True
        assert result.call["actual_microusd"] == 50_000
    else:
        assert result.call["outcome"] == "uncertain"
        assert "actual_microusd" not in result.call


def test_real_credential_echo_does_not_create_fact(clock):
    transport = CompleteErrorTransport(clock=clock, mutate=lambda body: body.update(error=DL_KEY))
    result, _, _, _, _ = execute(transport)
    assert "deepline_terminal_failure" not in result.call
    assert DL_KEY.encode() not in result.body


@pytest.mark.parametrize("provenance", ["credential_echo", "response_too_large", "redirect_rejected"])
def test_synthetic_provider_error_does_not_create_fact(clock, provenance):
    result, _, _, _, _ = execute(CompleteErrorTransport(clock=clock, provenance=provenance))
    assert "deepline_terminal_failure" not in result.call


@pytest.mark.parametrize("status", [True, "429", 200, 399, 600])
def test_completion_helper_rejects_non_failure_status_types(status):
    response = br.ProviderResponse(status, {}, b"{}")
    assert not br._deepline_terminal_failure(response, {}, execution_key="arena:" + "a" * 64)


def test_live_shaped_broker_worker_transition_returns_before_sixty_seconds(clock, tmp_path):
    transport = CompleteErrorTransport(clock=clock)
    initial, _, store, _, arguments = execute(transport)
    # Measured broker time was33.393s; retain the measured local/admission
    # remainder while the real broker above executes its unchanged30s polling.
    clock.now = 0.0

    class BrokerApi:
        def __init__(self): self.frames = []
        def provider(self, _run, _token, frame):
            assert frame["parameters"] == arguments["parameters"]
            assert frame["action_sequence"] == arguments["action_sequence"]
            self.frames.append(copy.deepcopy(frame))
            if len(self.frames) == 1:
                clock.now += 33.393
                return initial.to_document()
            # A mistaken recovery consumes the measured second API window.
            clock.now += 32.458
            recovered = initial.to_document()
            recovered["call"]["transport_recovery"] = "deepline_execution_lookup"
            return recovered

    api = BrokerApi()
    host = worker(tmp_path, api, monotonic=clock.monotonic)
    error, result = host._dispatch_once("deepline.execute", arguments["parameters"], 240_000)
    assert error is None and result["status"] == initial.status == 502
    assert base64.b64decode(result["body_b64"]) == initial.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result["call"]["outcome"] == "uncertain" and "actual_microusd" not in result["call"]
    assert len(api.frames) == 1 and clock.now == 33.393
    assert store.log.count("dispatch") == 1
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
    assert "deepline_terminal_failure" not in store.calls[initial.call["call_identity"]]["uncertain_doc"]


@pytest.mark.parametrize("written", [False, True])
def test_full_failure_transition_preserves_http_snapshot_late_bill_and_replay(clock, tmp_path, written):
    transport = CompleteErrorTransport(clock=clock)
    initial, broker, store, context, arguments = execute(transport, store=SettlingStore())
    saved = copy.deepcopy(store.calls[initial.call["call_identity"]]["uncertain_doc"])
    parsed = br.parse_broker_document(initial.to_document())
    assert parsed.body == initial.body and parsed.call["deepline_terminal_failure"] is True
    api = Api(parsed.to_document())
    host = worker(tmp_path, api, monotonic=clock.monotonic)
    report = {}
    status, headers, body = host.handle_http("POST",
        "https://code.deepline.com/api/v2/integrations/exa_search/execute",
        contracts.canonical_json({"payload": arguments["parameters"]["payload"]}).encode(),
        {"content-type": "application/json"}, delivery_report=report)
    assert status == initial.status and body == initial.body
    assert "deepline_terminal_failure" not in headers and len(api.frames) == 1
    host._mark_socket_delivery(report, written=written)
    payload = host._state.worker_observations.document()
    assert "deepline_terminal_failure" not in json.dumps(payload)
    receipt = payload["response_receipts"][0]
    assert receipt["body_sha256"] == contracts.hash_bytes(body)
    assert receipt["write_state"] == ("socket_written" if written else "socket_write_failed")
    assert payload["complete"] is written
    run_id = host._state.lease["run_id"]
    claim_hash = "sha256:" + "c" * 64
    snapshot, compressed = obs.build_snapshot(run_id, "execute", payload,
        lease_generation=1, claim_request_hash=claim_hash)

    class Objects:
        def get_bounded(self, ref, limit):
            assert ref == obs.object_ref(run_id, 1, claim_hash) and len(compressed) <= limit
            return compressed

    assert obs.read_snapshot(Objects(), run_id, "execute", 1, claim_hash) == snapshot
    transport.bill = billing(credits="0.5")
    candidate = _candidate(run_id=context.run_id, round_id=context.round_id,
        assignment_id=context.assignment_id, submission_id=context.submission_id,
        miner_hotkey=context.miner_hotkey, call_identity=initial.call["call_identity"],
        request_id=saved["deepline_job_id"], execution_key=saved["deepline_execution_key"],
        credential_fingerprint=saved["credential_fingerprint"], operation="exa_search",
        billing_provider="exa", kind=context.kind)
    assert broker.reconcile_deepline_cost(candidate) == {"status": "settled", "actual_microusd": 50_000}
    replay = broker.execute(context, **arguments)
    assert replay.status == initial.status and replay.body == initial.body
    assert store.calls[initial.call["call_identity"]]["actual"] == 50_000
    assert store.reconciled["call_identity"] == initial.call["call_identity"]
    assert store.calls[initial.call["call_identity"]]["uncertain_doc"] == saved
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
    assert store.log.count("dispatch") == 1
    # Later accounting cannot rewrite the earlier successful/failed delivery.
    assert obs.read_snapshot(Objects(), run_id, "execute", 1, claim_hash) == snapshot


@pytest.mark.parametrize("fact", [None, False, 1, "true"])
def test_old_or_malformed_fact_keeps_status_only_failure_recoverable(tmp_path, fact):
    initial = document()
    initial["call"]["provider_status"] = 429
    if fact is not None: initial["call"]["deepline_terminal_failure"] = fact
    api = Api(initial, document(200, recovered=True))
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 240_000)[1]["status"] == 200
    assert len(api.frames) == 2


@pytest.mark.parametrize("loss", [
    {"transport_error_class": "ReadTimeout"}, {"deepline_response_missing": True},
])
def test_loss_evidence_keeps_recovery_even_with_conflicting_failure_fact(tmp_path, loss):
    initial = document()
    initial["call"].update(provider_status=429, deepline_terminal_failure=True, **loss)
    api = Api(initial, document(200, recovered=True))
    host = worker(tmp_path, api)
    assert host._dispatch_once("deepline.execute", PARAMETERS, 240_000)[1]["status"] == 200
    assert len(api.frames) == 2


@pytest.mark.parametrize("status", [429, 500])
def test_validated_completed_error_lookup_preserves_internal_proof(clock, status):
    transport = CompleteErrorTransport(status, clock=clock)
    first, _, _, _, _ = execute(transport)
    response = br._deepline_execution_response_readback(transport=transport,
        secret=DL_KEY, execution_key=transport.key, operation="exa_search",
        operation_aliases=(), max_response_bytes=4096)
    assert response is not None and response.status == status
    assert response.deepline_completed_execution is True
    assert "executionRecovery" not in json.loads(response.body)
    assert br._deepline_terminal_failure(response, json.loads(response.body), execution_key=transport.key)
    assert first.call["outcome"] == "uncertain"


def test_existing_uncertain_identity_uses_validated_lookup_without_another_post(clock):
    transport = CompleteErrorTransport(clock=clock)
    first, broker, store, context, arguments = execute(transport)
    original = copy.deepcopy(store.calls[first.call["call_identity"]]["uncertain_doc"])
    replay = broker.execute(context, **arguments)
    assert replay.status == first.status and replay.body == first.body
    assert replay.call["deepline_terminal_failure"] is True
    assert replay.call["transport_recovery"] == "deepline_execution_lookup"
    assert replay.call["outcome"] == "uncertain" and "actual_microusd" not in replay.call
    assert store.calls[first.call["call_identity"]]["uncertain_doc"] == original
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
    assert store.log.count("dispatch") == 1


def test_compatibility_route_has_the_same_completed_failure_fact(clock):
    transport = CompleteErrorTransport(clock=clock)
    _, broker, _, context, _ = execute(transport)
    result = broker.execute(context, operation_id="exa.search", parameters={"query": "example"},
                            action_sequence=1, timeout_ms=30_000)
    assert result.call["provider"] == "deepline"
    assert result.call["deepline_terminal_failure"] is True
    assert result.call["outcome"] == "uncertain"


def test_later_billing_does_not_change_the_failure_or_repeat_post(clock):
    class CostStore(RetainedErrorStore):
        def reconcile_deepline_cost(self, **kwargs):
            self.reconciled = kwargs
            return {"status": "settled", "actual_microusd": kwargs["actual_microusd"]}

    transport = CompleteErrorTransport(clock=clock)
    first, broker, store, context, _ = execute(transport, store=CostStore())
    saved = copy.deepcopy(store.calls[first.call["call_identity"]]["uncertain_doc"])
    transport.bill = billing(credits="0.5")
    candidate = _candidate(run_id=context.run_id, round_id=context.round_id,
        assignment_id=context.assignment_id, submission_id=context.submission_id,
        miner_hotkey=context.miner_hotkey, call_identity=first.call["call_identity"],
        request_id=saved["deepline_job_id"], execution_key=saved["deepline_execution_key"],
        credential_fingerprint=saved["credential_fingerprint"], operation="exa_search",
        billing_provider="exa", kind=context.kind)
    assert broker.reconcile_deepline_cost(candidate) == {"status": "settled", "actual_microusd": 50_000}
    assert store.reconciled["call_identity"] == first.call["call_identity"]
    assert store.calls[first.call["call_identity"]]["uncertain_doc"] == saved
    assert sum(r["method"] == "POST" for r in transport.sent) == 1
