"""A bounded credential refusal must not turn an unresearched run into success."""

from __future__ import annotations

import base64
import json
import os

from lab_arena import contracts, runner as rn, runtime, shim
from test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config, valid_company


class CreditRefusalApi(FakeApi):
    credit_available = False

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        failed = frame["parameters"].get("tool") == "exa_search" and not self.credit_available
        error = "miner_credentials_unavailable" if failed else None
        call = {
            "call_identity": contracts.document_hash(["credit-call", frame["action_sequence"]]),
            "operation_id": frame["operation_id"],
            "action_sequence": frame["action_sequence"],
            "provider": "deepline",
            "funding_source": "miner_key",
            "provider_status": 402 if failed else 200,
            "outcome": "uncertain" if failed else "settled",
            "error_code": error,
            "actual_microusd": None if failed else 0,
        }
        body = (
            {"error": {"code": "payment_required"}}
            if failed else {"results": [{"url": "https://co1.example.com"}]}
        )
        return {
            "status": 402 if failed else 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(json.dumps(body).encode()).decode(),
            "call": call,
        }


class ExhaustedCreditRuntime(BridgingRuntime):
    def __init__(self, *, productive_other_tool: bool):
        super().__init__(output={"companies": [valid_company(1)]} if productive_other_tool else {"companies": []}, calls=0)
        self.productive_other_tool = productive_other_tool
        self.local_refusals = 0

    def run_icp(self, spec, **_):
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            if self.productive_other_tool:
                status, _headers, _body = shim.dispatch(
                    "deepline.execute",
                    {"tool": "exa_contents", "payload": {"ids": ["https://co1.example.com"]}},
                    5000,
                )
                assert status == 200
            for _ in range(rn.MAX_REFUSED_FRAMES + 1):
                try:
                    status, _headers, _body = shim.dispatch(
                        "deepline.execute",
                        {"tool": "exa_search", "payload": {"query": "fintech"}},
                        5000,
                    )
                    assert status == 402
                except shim.ShimError as exc:
                    assert "miner_credentials_unavailable" in str(exc)
                    self.local_refusals += 1
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        spec.output_path.write_bytes(json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


def test_repeated_credit_402_without_productive_call_cannot_finish_ok(tmp_path):
    api = CreditRefusalApi([lease()])
    sandbox = ExhaustedCreditRuntime(productive_other_tool=False)
    (tmp_path / "work").mkdir()
    runner = rn.Runner(make_config(tmp_path, api, sandbox))

    assert runner.run_once() == 1
    assert sandbox.local_refusals == 1
    assert len(api.provider_frames) == rn.MAX_REFUSED_FRAMES
    completion = api.completions[0]["body"]
    summary = completion["result"]["resource_summary"]
    assert summary["provider_call_count"] == rn.MAX_REFUSED_FRAMES
    assert summary["provider_refusal_count"] == rn.MAX_REFUSED_FRAMES
    assert summary["provider_local_refusal_count"] == 1
    assert summary["provider_unresolved_failure_count"] == 1
    assert completion["result"]["terminal_status"] == "credential_error"
    assert completion["output"] is None


def test_repeated_credit_402_preserves_partial_valid_output(tmp_path):
    api = CreditRefusalApi([lease()])
    sandbox = ExhaustedCreditRuntime(productive_other_tool=True)
    (tmp_path / "work").mkdir()
    runner = rn.Runner(make_config(tmp_path, api, sandbox))

    assert runner.run_once() == 1
    assert sandbox.local_refusals == 1
    assert len(api.provider_frames) == rn.MAX_REFUSED_FRAMES + 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    summary = completion["result"]["resource_summary"]
    assert summary["provider_call_count"] == rn.MAX_REFUSED_FRAMES + 1
    assert summary["provider_unresolved_failure_count"] == 1
    assert [row["company_name"] for row in completion["output"]["companies"]] == ["Co 1"]


def test_success_clears_prior_credential_refusal_count_for_the_same_tool(tmp_path):
    api = CreditRefusalApi([])
    state = rn.RunState(lease=lease(), lease_token="token")
    worker = rn.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    parameters = {"tool": "exa_search", "payload": {"query": "fintech"}}

    for _ in range(rn.MAX_REFUSED_FRAMES - 1):
        error, document = worker._dispatch_once("deepline.execute", parameters, 5000)
        assert error is None and document["status"] == 402
    api.credit_available = True
    error, document = worker._dispatch_once("deepline.execute", parameters, 5000)
    assert error is None and document["status"] == 200

    api.credit_available = False
    for _ in range(rn.MAX_REFUSED_FRAMES - 1):
        error, document = worker._dispatch_once("deepline.execute", parameters, 5000)
        assert error is None and document["status"] == 402
    assert len(api.provider_frames) == 2 * rn.MAX_REFUSED_FRAMES - 1


def test_valid_empty_output_without_provider_failure_stays_accepted(tmp_path):
    api = FakeApi([lease()])
    sandbox = BridgingRuntime(output={"companies": []}, calls=0)
    (tmp_path / "work").mkdir()

    assert rn.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    assert completion["output"]["companies"] == []
    assert completion["result"]["resource_summary"]["provider_unresolved_failure_count"] == 0


def test_later_same_tool_success_resolves_earlier_402_for_empty_output(tmp_path):
    api = CreditRefusalApi([lease()])

    class RecoveredRuntime(BridgingRuntime):
        def __init__(self):
            super().__init__(output={"companies": []}, calls=0)

        def run_icp(self, spec, **_):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                parameters = {"tool": "exa_search", "payload": {"query": "fintech"}}
                assert shim.dispatch("deepline.execute", parameters, 5000)[0] == 402
                api.credit_available = True
                assert shim.dispatch("deepline.execute", parameters, 5000)[0] == 200
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            spec.output_path.write_bytes(json.dumps(self.output).encode())
            return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))

    (tmp_path / "work").mkdir()
    assert rn.Runner(make_config(tmp_path, api, RecoveredRuntime())).run_once() == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    assert completion["output"]["companies"] == []
    summary = completion["result"]["resource_summary"]
    assert summary["provider_refusal_count"] == 1
    assert summary["provider_unresolved_failure_count"] == 0
    assert summary["provider_call_count"] == len(api.provider_frames) == 2


class OneProviderFailureRuntime(BridgingRuntime):
    def __init__(self):
        super().__init__(output={"companies": []}, calls=0)

    def run_icp(self, spec, **_):
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            assert shim.dispatch(
                "deepline.execute",
                {"tool": "exa_search", "payload": {"query": "fintech"}},
                5000,
            )[0] == 502
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        spec.output_path.write_bytes(json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


def test_empty_output_with_unresolved_provider_failure_is_provider_error(tmp_path):
    class ProviderFailureApi(CreditRefusalApi):
        def provider(self, run_id, lease_token, frame):
            response = super().provider(run_id, lease_token, frame)
            response["status"] = 502
            response["call"].update(
                provider_status=502, error_code="provider_unavailable",
                outcome="uncertain", actual_microusd=None,
            )
            return response

    api = ProviderFailureApi([lease()])
    (tmp_path / "work").mkdir()
    assert rn.Runner(make_config(tmp_path, api, OneProviderFailureRuntime())).run_once() == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "provider_error"
    assert completion["result"]["resource_summary"]["provider_unresolved_failure_count"] == 1
    assert completion["output"] is None
    assert len(api.provider_frames) == 1


def test_later_proved_per_icp_budget_stop_preserves_valid_empty_result(tmp_path):
    documents = []
    for sequence, (error, status, outcome, reason) in enumerate((
        ("provider_unavailable", 502, "settled", None),
        ("budget_refused", 402, "refused", "per_icp_quota"),
    )):
        body = json.dumps({"error": {"code": error}}).encode()
        call = {
            "call_identity": contracts.document_hash(["budget-control", sequence]),
            "operation_id": "deepline.execute",
            "action_sequence": sequence,
            "outcome": outcome,
            "actual_microusd": 0,
            "error_code": error,
            "provider_status": 503 if sequence == 0 else None,
        }
        if reason:
            call["reason"] = reason
        documents.append({
            "status": status,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode(),
            "call": call,
        })

    class BudgetStopRuntime(BridgingRuntime):
        def __init__(self):
            super().__init__(output={"companies": []}, calls=0)

        def run_icp(self, spec, **_):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for tool, status in (("exa_search", 502), ("exa_contents", 402)):
                    assert shim.dispatch(
                        "deepline.execute", {"tool": tool, "payload": {}}, 5000,
                    )[0] == status
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            spec.output_path.write_bytes(json.dumps(self.output).encode())
            return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))

    leased = lease()
    leased["sourcing_cost_eligibility_policy"] = (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    api = FakeApi([leased], broker_documents=documents)
    (tmp_path / "work").mkdir()
    assert rn.Runner(make_config(tmp_path, api, BudgetStopRuntime())).run_once() == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    assert completion["output"]["companies"] == []
    assert completion["result"]["resource_summary"]["provider_call_count"] == 2
