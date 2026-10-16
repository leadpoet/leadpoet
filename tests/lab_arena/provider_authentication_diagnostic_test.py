"""Bound submitted-account authentication diagnostics do not infer key expiry."""
from copy import deepcopy
from dataclasses import replace
import json
import os

import pytest

from lab_arena import broker, contracts, public_dashboard, runner, runtime, scoring, shim
from tests.lab_arena.deepline_managed_account_test import managed_error
from tests.lab_arena.public_evaluation_progress_test import NOW, run
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, HOST_KEYS, deepline_generic_http_denial, make_broker,
)
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, RefusedJudgeRuntime, RefusingApi, lease, make_config, scoring_lease,
)


def authentication_refusal():
    transport = FakeTransport([(401, {"error": {"code": 401, "message": "private auth detail"}})])
    instance, store, _ = make_broker(
        transport=transport, funding_source_for=lambda _: "miner_key",
        credential_for=lambda _, provider: HOST_KEYS[provider],
    )
    arguments = dict(operation_id="openrouter.chat", parameters={
        "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]},
        action_sequence=0, timeout_ms=5000)
    first = instance.execute(CONTEXT, **arguments)
    return first, instance, store, transport, arguments


def test_broker_authentication_proof_survives_replay_without_redispatch_or_charge_change():
    first, instance, store, transport, arguments = authentication_refusal()
    before = deepcopy(store.calls)
    replay = instance.execute(CONTEXT, **arguments)
    for result in (first, replay):
        assert result.status == 402
        assert result.call["error_code"] == "miner_credentials_unavailable"
        assert result.call["actual_microusd"] == 0
        assert broker.provider_authentication_failure_reason(result.call) == "provider_authentication_failed"
        assert broker.provider_credit_failure_reason(result.call) is None
        assert "private auth detail" not in json.dumps(result.to_document())
    assert replay.call["idempotent"] is True
    assert store.calls == before
    assert len(transport.sent) == 1


@pytest.mark.parametrize("field,value", [
    ("funding_source", "host"), ("provider_status", 403), ("provider_status", 429),
    ("provider_status", None), ("error_code", "miner_provider_not_configured"),
    ("error_code", "provider_unavailable"), ("error_code", "provider_request_refused"),
    ("outcome", "refused"), ("base_call_identity", "sha256:" + "f" * 64),
    ("provider_attempt", 2), ("action_sequence", 1),
    ("account_failure_evidence", None),
])
def test_authentication_diagnostic_requires_bound_account_proof(field, value):
    first, *_ = authentication_refusal()
    call = {**first.call, field: value}
    assert broker.provider_authentication_failure_reason(call) is None


@pytest.mark.parametrize("case", ["managed401", "target401", "policy403", "rate429", "missing"])
def test_native_non_account_failures_do_not_get_authentication_diagnostic(case):
    tool, payload = "exa_search", {"query": "example"}
    if case == "managed401":
        document = managed_error("exa", tool)
        document["upstream_status"] = document["tool_error"]["statusCode"] = 401
        status = 401
    elif case == "target401":
        tool, payload = "generic_http_request", {"url": "https://example.com/"}
        document = deepline_generic_http_denial()
        document.update(credential_owner="none", error_category="provider_auth", upstream_status=401)
        document["tool_error"]["statusCode"] = 401
        status = 401
    else:
        status = 429 if case == "rate429" else 403
        document = {"error": {"code": status, "metadata": {"error_type": "content_policy_violation"}}}
    transport = FakeTransport([(status, document)])
    instance, _, _ = make_broker(
        transport=transport, funding_source_for=lambda _: "miner_key",
        credential_for=lambda _, provider: "" if case == "missing" else HOST_KEYS[provider],
    )
    is_openrouter = case in ("policy403", "rate429")
    context = CONTEXT if is_openrouter or case == "target401" else replace(CONTEXT, deepline_catalog=frozen(row(tool, provider="exa")))
    result = instance.execute(context,
        operation_id="openrouter.chat" if is_openrouter else "deepline.execute",
        parameters={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]} if is_openrouter else {"tool": tool, "payload": payload},
        action_sequence=0, timeout_ms=5000)
    assert broker.provider_authentication_failure_reason(result.call) is None
    assert "account_failure_evidence" not in result.call
    assert sum(x["method"] == "POST" for x in transport.sent) == (0 if case == "missing" else 1)


@pytest.mark.parametrize("scoring_run", [False, True])
def test_broker_worker_runner_completion_and_public_projection_share_fixed_reason(tmp_path, scoring_run):
    response, *_ = authentication_refusal()
    class Api(RefusingApi):
        def provider(self, *args):
            super().provider(*args)
            return response.to_document()
    api = Api([scoring_lease() if scoring_run else lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    sandbox = RefusedJudgeRuntime(output=scoring.build_scoring_failure("r1", "judge_error", detail="private"))
    assert runner.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    result = contracts.validate_run_result(json.loads(json.dumps(api.completions[0]["body"]["result"])))
    assert result["terminal_status"] == "credential_error"
    assert result["failure_diagnostic"] == {"stage": "provider_call", "error_class": "provider_unavailable", "reason": "provider_authentication_failed"}
    projected = public_dashboard.evaluation_progress([run(status="failed", terminal_cause=result["terminal_status"], failure_reason=result["failure_diagnostic"]["reason"])], NOW)
    assert projected["failure_reasons"] == ["provider_authentication_failed"]
    assert len(api.provider_frames) == 1
    assert "private" not in json.dumps(projected)


@pytest.mark.parametrize("recovered", [False, True])
def test_accepted_partial_or_same_operation_recovery_does_not_get_authentication_diagnostic(tmp_path, recovered):
    response, *_ = authentication_refusal()
    class Api(RefusingApi):
        def provider(self, *args):
            super().provider(*args)
            document = response.to_document()
            if recovered and len(self.provider_frames) > 1:
                document["status"] = 200
                document["call"].update(error_code=None, provider_status=200)
                document["call"].pop("account_failure_evidence")
            return document
    class Sandbox(BridgingRuntime):
        def run_icp(self, spec, **_):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for _ in range(2 if recovered else 1):
                    shim.dispatch("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]}, 5000)
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            from tests.lab_arena.test_lab_arena_runner import valid_company
            return runtime.fake_result(output_bytes=json.dumps({"companies": [] if recovered else [valid_company(1)]}).encode())
    api = Api([lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, Sandbox())).run_once() == 1
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "accepted"
    assert "failure_diagnostic" not in result
    assert len(api.provider_frames) == (2 if recovered else 1)


@pytest.mark.parametrize("change", [
    {"provider_status": 403}, {"provider_status": True},
    {"base_call_identity": "sha256:" + "f" * 64},
    {"provider_attempt": 2}, {"action_sequence": 1},
    {"error_class": "provider_auth"}, {"reason": "expired_key"},
    {"extra": "private detail"},
])
def test_changed_account_evidence_cannot_create_authentication_diagnostic(change):
    first, *_ = authentication_refusal()
    call = deepcopy(first.call)
    call["account_failure_evidence"].update(change)
    assert broker.provider_authentication_failure_reason(call) is None


def test_credit_diagnostic_keeps_priority_over_another_authentication_failure(tmp_path, monkeypatch):
    from tests.lab_arena.provider_credit_diagnostic_test import _unknown_fee_refusal
    monkeypatch.setattr(broker, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    authentication, *_ = authentication_refusal()
    _, credit_broker, _, credit_transport, context, arguments = _unknown_fee_refusal()
    from tests.lab_arena.deepline_host_payment_status_test import REFUSAL
    credit_transport.responses.extend([(402, deepcopy(REFUSAL)), (200, {"recent": {"entries": []}})])
    credit = credit_broker.execute(context, **{**arguments, "action_sequence": 1})
    class Api(RefusingApi):
        def provider(self, *args):
            super().provider(*args)
            return (authentication if len(self.provider_frames) == 1 else credit).to_document()
    class Sandbox(BridgingRuntime):
        def run_icp(self, spec, **_):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                shim.dispatch("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]}, 5000)
                shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "example"}}, 5000)
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            return runtime.fake_result(output_bytes=json.dumps(scoring.build_scoring_failure("r1", "judge_error", detail="private")).encode())
    api = Api([lease()], code="miner_credentials_unavailable")
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, Sandbox())).run_once() == 1
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "credential_error"
    assert result["failure_diagnostic"]["reason"] == "provider_credit_exhausted"
