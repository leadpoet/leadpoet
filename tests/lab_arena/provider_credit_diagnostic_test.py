"""Credit diagnostics must not change credential refusal or scoring policy."""
from copy import deepcopy
import json
import os

import pytest

from lab_arena import broker, contracts, public_dashboard, runner, runtime, scoring, shim
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
