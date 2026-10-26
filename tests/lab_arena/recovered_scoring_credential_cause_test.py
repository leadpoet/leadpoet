"""Recovered account refusals must not replace a later judge's actual failure."""

import base64
import json
import os

import pytest

from lab_arena import contracts, runtime, scoring, shim
from tests.lab_arena.company_judgments_test import _input, _new, _refs
from tests.lab_arena.partial_company_judgments_test import _lease
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, lease, scoring_lease
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model


def _response(sequence, status, *, error=None, funding="miner_key"):
    body = json.dumps({"error": {"code": error}} if error else {"ok": True}).encode()
    call = {
        "call_identity": contracts.document_hash(["account-recovery", sequence]),
        "operation_id": "deepline.execute",
        "provider": "deepline",
        "funding_source": funding,
        "provider_status": status,
        "actual_microusd": 2000 if status == 200 else 0,
        "outcome": "settled" if status == 200 else "uncertain",
    }
    if error:
        call["error_code"] = error
    return {"status": status, "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode(), "call": call}


class AccountRecoveryRuntime(BridgingRuntime):
    def __init__(self, tools, *, output=None, timed_out=True):
        super().__init__(output=output, timed_out=timed_out)
        self.tools = tools

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for tool in self.tools:
                shim.dispatch("deepline.execute", {"tool": tool, "payload": {}}, 5000)
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        return runtime.fake_result(
            timed_out=self.timed_out, exit_code=None if self.timed_out else 0,
            output_bytes=json.dumps(self.output).encode() if self.output else None,
        )


def _run(tmp_path, documents, tools, *, output=None, timed_out=True, leased=None):
    api = LoggingApi()
    api.leases = [leased or scoring_lease()]
    api.broker_documents = list(documents)
    run_model(tmp_path, api, AccountRecoveryRuntime(tools, output=output, timed_out=timed_out))
    body = api.completions[0]["body"]
    contracts.validate_run_result(body["result"])
    finished = next(event["content"] for batch in api.batches for event in batch["events"]
                    if event["kind"] == "runtime.finished")
    assert len(api.provider_frames) == len(documents)
    assert body["result"]["resource_summary"]["provider_call_count"] == len(documents)
    assert not api.broker_documents
    return body, finished


def test_recovered_same_tool_account_refusal_preserves_actual_judge_timeout(tmp_path):
    body, finished = _run(tmp_path, [
        _response(0, 402, error="miner_credentials_unavailable"), _response(1, 200),
    ], ["firecrawl_scrape", "firecrawl_scrape"])
    assert body["result"]["terminal_status"] == "judge_timeout"
    assert body["result"]["failure_diagnostic"] == {
        "stage": "sandbox", "error_class": "judge_timeout",
    }
    assert body["result"]["resource_summary"]["provider_unresolved_failure_count"] == 0
    assert finished["timed_out"] is True
    assert finished["status"] == "judge_timeout"


@pytest.mark.parametrize("status,second_tool", [(401, None), (402, None), (403, None),
                                                   (402, "exa_search")])
def test_unresolved_account_failure_still_owns_failed_judge(tmp_path, status, second_tool):
    documents = [_response(0, status, error="miner_credentials_unavailable")]
    tools = ["firecrawl_scrape"]
    if second_tool:
        documents.append(_response(1, 200))
        tools.append(second_tool)
    body, _ = _run(tmp_path, documents, tools)
    assert body["result"]["terminal_status"] == "credential_error"
    assert body["result"]["resource_summary"]["provider_unresolved_failure_count"] == 1


def test_later_same_tool_account_failure_is_not_cleared_by_earlier_success(tmp_path):
    body, _ = _run(tmp_path, [_response(0, 200),
        _response(1, 402, error="miner_credentials_unavailable")],
        ["firecrawl_scrape", "firecrawl_scrape"])
    assert body["result"]["terminal_status"] == "credential_error"


def test_scoring_budget_refusal_remains_challenger_specific_after_later_success(tmp_path):
    body, _ = _run(tmp_path, [_response(0, 402, error="budget_refused"), _response(1, 200)],
                   ["firecrawl_scrape", "firecrawl_scrape"])
    assert body["result"]["terminal_status"] == "credential_error"


@pytest.mark.parametrize("recovered", [False, True])
def test_managed_provider_failure_preserves_existing_infrastructure_classification(tmp_path, recovered):
    documents = [_response(0, 502, error="provider_unavailable")]
    tools = ["firecrawl_scrape"]
    if recovered:
        documents.append(_response(1, 200))
        tools.append("firecrawl_scrape")
    body, _ = _run(tmp_path, documents, tools)
    # Historical infrastructure errors retain their existing precedence.
    assert body["result"]["terminal_status"] == "judge_error"
    assert body["result"]["failure_diagnostic"] == {
        "stage": "sandbox", "error_class": "judge_timeout",
    }


def test_valid_accepted_judge_output_survives_account_refusal(tmp_path):
    output = scoring.build_scoring_output("r1", [
        {"final_score": 71.0, "failure_reason": ""},
        {"final_score": 44.5, "failure_reason": ""},
    ])
    body, _ = _run(tmp_path, [_response(0, 401, error="miner_credentials_unavailable")],
                   ["firecrawl_scrape"], output=output, timed_out=False)
    assert body["result"]["terminal_status"] == "accepted"
    assert body["output"] == output


@pytest.mark.parametrize("recovered", [False, True])
def test_partial_completed_company_hashes_survive_account_failure_and_recovery(tmp_path, recovered):
    document = _input()
    leased = scoring_lease()
    leased.update(scorer_policy=document["scorer_policy"],
                  company_quality_policy="company_quality_v1", company_judgment_cache=_lease(document),
                  scored_output={"companies": document["companies"]})
    rows = [_new(_refs(document)[0])]
    row_hash = contracts.document_hash(rows)
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)
    documents = [_response(0, 402, error="miner_credentials_unavailable")]
    tools = ["firecrawl_scrape"]
    if recovered:
        documents.append(_response(1, 200))
        tools.append("firecrawl_scrape")
    body, _ = _run(tmp_path, documents, tools, output=output, leased=leased)
    assert body["result"]["terminal_status"] == ("judge_timeout" if recovered else "credential_error")
    # The sandbox timeout owns the saved partial envelope in both cases.
    assert body["output"]["failure"] == "judge_timeout"
    assert contracts.document_hash(body["output"]["company_judgments"]) == row_hash
    assert "breakdowns" not in body["output"]


def test_execute_historical_account_failure_keeps_existing_precedence(tmp_path):
    body, _ = _run(tmp_path, [
        _response(0, 402, error="miner_credentials_unavailable"), _response(1, 200),
    ], ["firecrawl_scrape", "firecrawl_scrape"], leased=lease())
    assert body["result"]["terminal_status"] == "credential_error"
