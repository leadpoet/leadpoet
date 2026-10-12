"""Credit diagnostics must not change credential refusal or scoring policy."""
from copy import deepcopy
from dataclasses import replace
import json
import os

import pytest

from lab_arena import broker, contracts, provider_costs, public_dashboard, runner, runtime, scoring, shim, trajectory
from tests.lab_arena.deepline_host_payment_status_test import REFUSAL, NATIVE, EDGE_TRACE
from tests.lab_arena.deepline_terminal_error_test import RetainedErrorStore
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, HOST_KEYS, make_broker,
)
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, RefusedJudgeRuntime, RefusingApi, lease, make_config, scoring_lease, valid_company,
)
from tests.lab_arena.public_evaluation_progress_test import NOW, run


PROOF = {
    "schema_version": "leadpoet.lab_arena.credit_failure_proof.v1",
    "provider": "deepline", "reason": "out_of_credit",
    "provider_status": 402, "actual_microusd": 0,
}
CALL = {
    "provider": "deepline", "funding_source": "miner_key",
    "outcome": "settled", "error_code": "miner_credentials_unavailable",
    "provider_status": 402, "actual_microusd": 0,
    "credit_failure_proof": PROOF,
}


def test_broker_direct_and_idempotent_credit_replay_keep_operational_error():
    document = {
        "code": "INSUFFICIENT_CREDITS", "error": "Insufficient credits",
        "billing": {"kind": "insufficient_credits", "required_credits": 1,
                    "balance_credits": 0, "needed_credits": 1},
    }
    transport = FakeTransport([(402, document)])
    instance, _, _ = make_broker(
        transport=transport, funding_source_for=lambda _: "miner_key",
        credential_for=lambda _, provider: HOST_KEYS[provider],
    )
    arguments = dict(operation_id="deepline.execute",
                     parameters={"tool": "exa_search", "payload": {"query": "example"}},
                     action_sequence=0, timeout_ms=5000)
    first = instance.execute(CONTEXT, **arguments)
    replay = instance.execute(CONTEXT, **arguments)
    for response in (first, replay):
        assert response.status == 402
        assert json.loads(response.body)["error"]["code"] == "miner_credentials_unavailable"
        assert response.call["credit_failure_proof"] == PROOF
        assert broker.provider_credit_failure_reason(response.call) == "provider_credit_exhausted"
    assert replay.call["idempotent"] is True
    assert sum(item["method"] == "POST" for item in transport.sent) == 1


class DiagnosticReplayStore(RetainedErrorStore):
    def _view(self, call):
        result = super()._view(call)
        # The real call-state view exposes this from an uncertain head only.
        # A later billing-only settlement requires the retained history read.
        if call["kind"] == "settlement":
            result.pop("account_failure_evidence", None)
        return result


def _unknown_fee_refusal(*, kind="execute", document=None, status=402,
                         headers=None, funding="miner_key", missing_key=False):
    transport = FakeTransport([
        (status, deepcopy(REFUSAL if document is None else document), headers or {}),
        (200, {"recent": {"entries": []}}),
    ])
    instance, store, _ = make_broker(
        store=DiagnosticReplayStore(), transport=transport,
        funding_source_for=lambda _: funding,
        credential_for=lambda _, provider: "" if missing_key else HOST_KEYS[provider],
    )
    context = replace(CONTEXT, kind=kind,
                      deepline_catalog=frozen(row("exa_search", provider="exa")))
    arguments = dict(operation_id="deepline.execute",
                     parameters={"tool": "exa_search", "payload": {"query": "example"}},
                     action_sequence=0, timeout_ms=5000)
    result = instance.execute(context, **arguments)
    return result, instance, store, transport, context, arguments


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("native", [False, True])
def test_typed_credit_refusal_unknown_fee_is_diagnostic_only(monkeypatch, kind, native):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    assert provider_costs.deepline_payment_refusal_cost(402, REFUSAL) is None
    document = {**deepcopy(REFUSAL), **({"job_id": NATIVE} if native else {})}
    first, instance, store, transport, context, arguments = _unknown_fee_refusal(
        kind=kind, document=document, headers={"x-vercel-id": EDGE_TRACE},
    )
    assert first.status == 402
    assert json.loads(first.body) == {"error": {"code": "miner_credentials_unavailable"}}
    assert first.call["outcome"] == "uncertain"
    assert "actual_microusd" not in first.call and "credit_failure_proof" not in first.call
    evidence = first.call["account_failure_evidence"]
    assert evidence["reason"] == "out_of_credit"
    saved = store.calls[first.call["call_identity"]]
    assert saved["uncertain_doc"]["account_failure_evidence"] == evidence
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert saved["kind"] == "uncertain" and saved["amount"] == 0
    assert broker.provider_credit_failure_reason(first.call) == "provider_credit_exhausted"
    projected = trajectory.provider_response_content(first.to_document(), elapsed_ms=10)
    assert projected["call"]["account_failure_evidence"]["reason"] == "out_of_credit"
    assert json.loads(json.dumps(first.to_document())) == first.to_document()
    before = len(transport.sent)
    replay = instance.execute(context, **arguments)
    # Preserve the existing uncertain replay reply and paid-request behavior.
    assert replay.status == 409 and replay.call["error_code"] == "call_uncertain"
    assert replay.call["outcome"] == "uncertain"
    assert broker.provider_credit_failure_reason(replay.call) == "provider_credit_exhausted"
    assert "actual_microusd" not in replay.call and "credit_failure_proof" not in replay.call
    assert len(transport.sent) == before
    assert sum(item["method"] == "POST" for item in transport.sent) == 1


@pytest.mark.parametrize("actual", [0, 2000])
def test_unknown_fee_credit_diagnostic_survives_billing_only_settlement(monkeypatch, actual):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    first, instance, store, transport, context, arguments = _unknown_fee_refusal()
    saved = store.calls[first.call["call_identity"]]
    saved.update(kind="settlement", actual=actual, terminal=broker._terminal_response_document(
        502, {}, broker.operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False))
    # Avoid inventing a replacement execution receipt in this billing-only test.
    transport.responses = [(404, {})]
    replay = instance.execute(context, **arguments)
    assert replay.status == 502 and replay.call["outcome"] == "settled"
    assert replay.call["actual_microusd"] == actual
    assert "credit_failure_proof" not in replay.call
    assert broker.provider_credit_failure_reason(replay.call) == "provider_credit_exhausted"
    assert saved["actual"] == actual
    assert sum(item["method"] == "POST" for item in transport.sent) == 1


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_unknown_fee_broker_refusal_reaches_existing_runner_diagnostic(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    response, _, _, transport, _, _ = _unknown_fee_refusal(kind=kind)

    class Api(RefusingApi):
        events = None

        def provider(self, *args):
            super().provider(*args)
            return response.to_document()

        def trajectory(self, _run_id, _lease_token, events):
            self.events = (self.events or []) + events
            return {"accepted": len(events)}

    scoring_run = kind == "score"
    api = Api([scoring_lease() if scoring_run else lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    failure = scoring.build_scoring_failure("r1", "judge_error", detail="private-detail")
    sandbox = RefusedJudgeRuntime(output=failure)
    assert runner.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "credential_error"
    assert result["failure_diagnostic"]["reason"] == "provider_credit_exhausted"
    assert result["resource_summary"]["provider_refusal_count"] == 1
    assert len(api.provider_frames) == 1
    assert sum(item["method"] == "POST" for item in transport.sent) == 1


@pytest.mark.parametrize("status,change,headers,funding,missing_key", [
    (401, {}, {}, "miner_key", False),
    (403, {}, {}, "miner_key", False),
    (402, {"code": "OTHER_PAYMENT_ERROR"}, {}, "miner_key", False),
    (402, {"billing": None}, {}, "miner_key", False),
    (402, {"error": ""}, {}, "miner_key", False),
    (402, {"billing": {"kind": "insufficient_credits", "required_credits": "NaN",
                       "balance_credits": 0, "needed_credits": 1}}, {}, "miner_key", False),
    (402, {"job_id": "invalid/id"}, {}, "miner_key", False),
    (402, {"job_id": NATIVE, "request_id": "different-native"}, {}, "miner_key", False),
    (402, {"job_id": NATIVE}, {"x-deepline-request-id": "different-native"}, "miner_key", False),
    (402, {"job_id": NATIVE}, {"x-deepline-request-id": "invalid/id"}, "miner_key", False),
    (402, {}, {}, "host", False),
    (402, {}, {}, "miner_key", True),
])
def test_unknown_fee_credit_diagnostic_rejects_unbound_or_unrecognized_refusals(
    monkeypatch, status, change, headers, funding, missing_key,
):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    document = {**deepcopy(REFUSAL), **change}
    response, _, _, _, _, _ = _unknown_fee_refusal(
        status=status, document=document, headers=headers,
        funding=funding, missing_key=missing_key,
    )
    assert broker.provider_credit_failure_reason(response.call) is None
    assert "reason" not in response.call.get("account_failure_evidence", {})


@pytest.mark.parametrize("change", [
    {"base_call_identity": "sha256:" + "f" * 64},
    {"action_sequence": 1}, {"provider_attempt": 2},
    {"reason": "unrecognized"}, {"reason": True},
    {"provider_status": 401}, {"extra": "not_allowed"},
])
def test_unknown_fee_credit_replay_rejects_changed_evidence_binding(monkeypatch, change):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    first, instance, store, transport, context, arguments = _unknown_fee_refusal()
    evidence = store.calls[first.call["call_identity"]]["uncertain_doc"]["account_failure_evidence"]
    evidence.update(change)
    before = len(transport.sent)
    replay = instance.execute(context, **arguments)
    assert replay.status == 503 and replay.call["error_code"] == "broker_unavailable"
    assert broker.provider_credit_failure_reason(replay.call) is None
    assert len(transport.sent) == before


def test_old_unknown_fee_account_evidence_cannot_be_inferred_as_credit(monkeypatch):
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    first, instance, store, transport, context, arguments = _unknown_fee_refusal()
    evidence = store.calls[first.call["call_identity"]]["uncertain_doc"]["account_failure_evidence"]
    evidence.pop("reason")
    before = len(transport.sent)
    replay = instance.execute(context, **arguments)
    assert replay.status == 409 and replay.call["error_code"] == "call_uncertain"
    assert broker.provider_credit_failure_reason(replay.call) is None
    assert len(transport.sent) == before


@pytest.mark.parametrize("change", [
    {"credit_failure_proof": None}, {"credit_failure_proof": {"reason": "out_of_credit"}},
    {"credit_failure_proof": dict(PROOF, actual_microusd=False)},
    {"credit_failure_proof": dict(PROOF, provider_status=True)},
    {"credit_failure_proof": dict(PROOF, reason="private-detail")},
    {"provider": "openrouter"}, {"provider_status": 401},
    {"actual_microusd": 1}, {"actual_microusd": False},
    {"outcome": "uncertain"}, {"funding_source": "host"},
    {"error_code": "provider_unavailable"},
])
def test_missing_invalid_unpriced_or_other_account_failure_has_no_credit_diagnostic(change):
    assert broker.provider_credit_failure_reason({**deepcopy(CALL), **change}) is None


@pytest.mark.parametrize("scoring_run", [False, True])
@pytest.mark.parametrize("proved", [False, True])
def test_worker_completion_and_runtime_diagnostic_preserve_credential_terminal(
    tmp_path, scoring_run, proved,
):
    class Api(RefusingApi):
        events = None

        def trajectory(self, _run_id, _lease_token, events):
            if self.events is None:
                self.events = []
            self.events.extend(events)
            return {"accepted": len(events)}

        def provider(self, *args):
            response = super().provider(*args)
            response["call"].update(deepcopy(CALL))
            if not proved:
                response["call"].pop("credit_failure_proof")
            return response

    api = Api([scoring_lease() if scoring_run else lease()], code="miner_credentials_unavailable")
    failure = scoring.build_scoring_failure("r1", "judge_error", detail="private-detail")
    (tmp_path / "work").mkdir()
    sandbox = RefusedJudgeRuntime(output=failure)
    instance = runner.Runner(make_config(tmp_path, api, sandbox))
    assert instance.run_once() == 1 and instance.abandoned == 0
    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(json.loads(json.dumps(completion["result"])))
    assert result["terminal_status"] == "credential_error"
    assert result["resource_summary"]["provider_refusal_count"] == 1
    assert len(api.provider_frames) == 1
    assert completion["output"] is None
    if proved:
        assert result["failure_diagnostic"] == {
            "stage": "provider_call", "error_class": "provider_unavailable",
            "reason": "provider_credit_exhausted",
        }
    else:
        assert "failure_diagnostic" not in result
    assert "private-detail" not in json.dumps(result)
    finished = next(event for event in api.events if event["kind"] == "runtime.finished")
    diagnostic = finished["content"]["failure_diagnostic"] or {}
    assert diagnostic.get("reason") == ("provider_credit_exhausted" if proved else None)
    assert finished["content"]["status"] == "credential_error"


@pytest.mark.parametrize("recovered", [False, True])
def test_partial_output_and_later_same_tool_recovery_do_not_get_failure_diagnostic(tmp_path, recovered):
    class Api(RefusingApi):
        def provider(self, *args):
            response = super().provider(*args)
            response["call"].update(deepcopy(CALL))
            if recovered and len(self.provider_frames) > 1:
                response["status"] = 200
                response["call"].update(error_code=None, provider_status=200)
                response["call"].pop("credit_failure_proof")
            return response

    class Sandbox(BridgingRuntime):
        def run_icp(self, spec, **_):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for _ in range(2 if recovered else 1):
                    shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "example"}}, 5000)
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            return runtime.fake_result(output_bytes=json.dumps({
                "companies": [] if recovered else [valid_company(1)]
            }).encode())

    api = Api([lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, Sandbox())).run_once() == 1
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "accepted"
    assert "failure_diagnostic" not in result


@pytest.mark.parametrize("cause,reason,expected", [
    ("credential_error", "provider_credit_exhausted", "provider_credit_exhausted"),
    ("credential_error", None, "provider_credentials"),
    ("credential_error", "private-detail", "provider_credentials"),
    ("judge_error", "provider_credit_exhausted", "review"),
])
def test_public_projection_uses_only_fixed_diagnostic_without_changing_assignment(cause, reason, expected):
    failed = run(status="failed", terminal_cause=cause, failure_reason=reason)
    result = public_dashboard.evaluation_progress([failed], NOW)
    assert result["failure_reasons"] == [expected]
    assert result["counts"]["failed"] == 1
    assert "private-detail" not in json.dumps(result)
    for successor in [run(attempt=2, status="pending"), run(attempt=2, status="accepted")]:
        result = public_dashboard.evaluation_progress([failed, successor], NOW)
        assert "failure_reasons" not in result


@pytest.mark.parametrize("kind,expected", [("execute", "execution_window"), ("score", "scoring_window")])
def test_unclaimed_stage_closed_kind_keeps_completed_execution_and_scores(kind, expected):
    completed = run(kind="execute", assignment_id="execution", status="accepted")
    closed = run(kind=kind, assignment_id="closed", status="failed", terminal_cause="stage_closed",
                 runner_hotkey=None, lease_generation=0)
    result = public_dashboard.evaluation_progress([completed, closed], NOW)
    assert result["failure_reasons"] == [expected]
    assert result["counts"]["completed"] == result["counts"]["failed"] == 1
    assert result["validators"] == []
