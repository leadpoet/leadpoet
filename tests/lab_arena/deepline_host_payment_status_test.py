"""Host judge payment refusals retain status independently of charge proof."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, provider_costs
from tests.lab_arena.deepline_terminal_error_test import (
    ErrorTransport, RetainedErrorStore, billing,
)
from tests.lab_arena.deepline_worker_same_action_recovery_test import Api, worker
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeLedgerStore, make_broker


REFUSAL = {
    "code": "INSUFFICIENT_CREDITS", "error": "Insufficient credits",
    "billing": {"kind": "insufficient_credits", "required_credits": 0.02,
                "balance_credits": -4.993, "needed_credits": 0.02},
}
PARAMETERS = {"tool": "exa_search", "payload": {"query": "example"}}


@pytest.fixture(autouse=True)
def one_billing_attempt(monkeypatch):
    # Synthetic/unrecognized responses retain their existing recovery path;
    # do not spend its production polling window in these classification tests.
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)


class HostPaymentStore(RetainedErrorStore):
    def reserve_call(self, **kwargs):
        return FakeLedgerStore.reserve_call(self, **kwargs)


def setup(*, document=None, provenance=None, status=402, routed=False, kind="score"):
    transport = ErrorTransport(status, body=json.dumps(
        REFUSAL if document is None else document).encode())
    original_send = transport.send
    if provenance:
        def send(**request):
            response = original_send(**request)
            return br.ProviderResponse(response.status, response.headers, response.body,
                                       provenance) if request["method"] == "POST" else response
        transport.send = send
    context = replace(CONTEXT, kind=kind, round_id="arena-2026-10-06",
                      deepline_catalog=frozen(row("exa_search", provider="exa")))
    options = {"host_shadow_score_compat_round_id": context.round_id} if routed else {}
    broker, store, _ = make_broker(store=HostPaymentStore(), transport=transport, **options)
    arguments = dict(operation_id="exa.search" if routed else "deepline.execute",
        parameters={"query": "example"} if routed else PARAMETERS,
        action_sequence=0, timeout_ms=30_000)
    return broker, store, transport, context, arguments


@pytest.mark.parametrize("routed", [False, True])
def test_host_score_refusal_without_native_id_preserves_402_unknown_cost_and_replay(
    monkeypatch, tmp_path, routed,
):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("payment refusal must not poll"))
    assert provider_costs.deepline_payment_refusal_cost(402, REFUSAL) is None
    broker, store, transport, context, arguments = setup(routed=routed)
    before_capacity = store.openrouter_capacity
    result = broker.execute(context, **arguments)
    assert result.status == 402
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert result.call["funding_source"] == "host"
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["provider_status"] == 402 and result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call and "credit_failure_proof" not in result.call
    saved = store.calls[result.call["call_identity"]]
    reserved = saved["amount"]
    assert reserved > 0 and store.openrouter_capacity == before_capacity - reserved
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert "credit_failure_proof" not in saved["uncertain_doc"]
    assert "account_failure_evidence" not in saved["uncertain_doc"]
    assert "miner_credentials_unavailable" not in json.dumps(result.to_document())
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == 402 and replay.body == result.body and replay.call["idempotent"]
    assert replay.call["outcome"] == "uncertain" and "actual_microusd" not in replay.call
    assert len(transport.sent) == before and saved["amount"] == reserved
    assert store.openrouter_capacity == before_capacity - reserved
    assert sum(request["method"] == "POST" for request in transport.sent) == 1

    # Renderer socket gets the safe HTTP status; no same-action recovery request.
    api = Api(result.to_document())
    host = worker(tmp_path, api)
    error, returned = host._dispatch_once(arguments["operation_id"], arguments["parameters"], 30_000)
    assert error is None and returned["status"] == 402
    assert len(api.frames) == 1
    assert host._state.refusals == 0  # Host failure cannot become miner refusal.


@pytest.mark.parametrize("change", [
    {"code": "OTHER_PAYMENT_ERROR"}, {"error": ""}, {"billing": None},
    {"billing": {"kind": "unrelated"}},
    {"billing": {"kind": "insufficient_credits", "required_credits": "NaN", "balance_credits": 0, "needed_credits": 1}},
    {"billing": {"kind": "insufficient_credits", "required_credits": -1, "balance_credits": 0, "needed_credits": 1}},
    {"billing": {"kind": "insufficient_credits", "required_credits": "1e" + "9" * 70, "balance_credits": 0, "needed_credits": 1}},
    {"job_id": "one-native", "request_id": "different-native"},
    {"job_id": "invalid/id"},
])
def test_malformed_unrelated_or_conflicting_refusal_keeps_normal_host_failure(change):
    document = deepcopy(REFUSAL)
    document.update(change)
    broker, _, _, context, arguments = setup(document=document)
    result = broker.execute(context, **arguments)
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call


@pytest.mark.parametrize("provenance", ["redirect_rejected", "credential_echo", "response_too_large"])
def test_synthetic_402_cannot_preserve_host_payment_refusal(provenance):
    broker, _, _, context, arguments = setup(provenance=provenance)
    result = broker.execute(context, **arguments)
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    assert "credit_failure_proof" not in result.call


def test_execution_host_payment_behavior_remains_generic():
    broker, _, _, context, arguments = setup(kind="execute")
    result = broker.execute(context, **arguments)
    assert result.status == 502 and result.call["outcome"] == "uncertain"


def test_positive_final_charge_is_not_erased_by_payment_status():
    broker, store, transport, context, arguments = setup()
    transport.bill = billing(credits="0.5")
    result = broker.execute(context, **arguments)
    assert result.status == 402 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 50_000
    assert "credit_failure_proof" not in result.call
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == 402 and replay.call["actual_microusd"] == 50_000
    assert replay.call["error_code"] == "provider_unavailable" and replay.call["provider_status"] == 402
    assert replay.call["funding_source"] == "host" and len(transport.sent) == before


def test_body_and_header_identity_conflict_cannot_preserve_402():
    document = {**REFUSAL, "job_id": "body-native-request"}
    broker, _, transport, context, arguments = setup(document=document)
    original_send = transport.send

    def send(**request):
        response = original_send(**request)
        return br.ProviderResponse(402, {"x-deepline-request-id": "header-native-request"},
                                   response.body) if request["method"] == "POST" else response

    transport.send = send
    result = broker.execute(context, **arguments)
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call


def test_refusal_replay_requires_bound_original_reservation():
    broker, store, transport, context, arguments = setup()
    first = broker.execute(context, **arguments)
    assert first.status == 402
    saved = store.calls[first.call["call_identity"]]
    saved["call_doc"]["request_hash"] = "sha256:" + "f" * 64
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == 503 and replay.call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == before


def test_payment_refusal_replay_after_delayed_settlement_keeps_exact_cost():
    broker, store, transport, context, arguments = setup()
    first = broker.execute(context, **arguments)
    saved = store.calls[first.call["call_identity"]]
    terminal = saved["uncertain_doc"]["deepline_terminal_response"]
    # The delayed RPC records exact billing independently of the saved refusal.
    saved.update(kind="settlement", actual=2000, terminal=br._terminal_response_document(
        502, {}, br.operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False))
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == 402 and replay.body == br._decode_terminal(terminal)[2]
    assert replay.call["outcome"] == "settled" and replay.call["actual_microusd"] == 2000
    assert replay.call["funding_source"] == "host" and replay.call["error_code"] == "provider_unavailable"
    assert "credit_failure_proof" not in replay.call and len(transport.sent) == before
