"""Typed unavailable evidence must retain its diagnostic, not change policy."""

from copy import deepcopy
import json

import pytest

from lab_arena import contracts, runner, scorer_entrypoint, scoring, shim
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, make_config, scoring_lease,
)
from tests.lab_arena.test_lab_arena_scoring import company, make_icp


def unavailable_evidence():
    return {
        "final_score": 0.0,
        "failure_reason": "Company fit pre-check unavailable: evidence fetch failed",
        "verifier_gate_receipts": [{
            "gate": "company_fit", "decision": "unavailable",
            "required_attribute_decision": "unavailable",
            "supporting_receipts": [{
                "gate": "company_evidence_investigation",
                "completed_submitted_findings": True, "failure_reason": "",
                "usage": {"fetch_outcomes": [
                    {"ok": True, "error_class": ""},
                    {"ok": False, "error_class": "provider_diagnostic_body"},
                ]},
            }],
        }],
    }


def test_exact_typed_provider_diagnostic_is_not_unknown_or_credit():
    row = unavailable_evidence()
    before = deepcopy(row)
    assert scoring._retryable_breakdown_failure_reason(row) == "provider_error"
    assert row == before


@pytest.mark.parametrize("control", [
    "wrong_fit_gate", "fit_match", "wrong_investigation_gate", "incomplete",
    "investigation_failure", "successful_fetch", "missing_outcomes",
    "source_not_found", "other_error", "untyped_text", "malformed_outcome",
])
def test_unproved_or_unrelated_diagnostic_stays_unknown(control):
    row = unavailable_evidence()
    fit = row["verifier_gate_receipts"][0]
    investigation = fit["supporting_receipts"][0]
    outcomes = investigation["usage"]["fetch_outcomes"]
    if control == "wrong_fit_gate":
        fit["gate"] = "intent_details"
    elif control == "fit_match":
        fit["decision"] = "match"
    elif control == "wrong_investigation_gate":
        investigation["gate"] = "model_output"
    elif control == "incomplete":
        investigation["completed_submitted_findings"] = False
    elif control == "investigation_failure":
        investigation["failure_reason"] = "admission_budget_interrupted"
    elif control == "successful_fetch":
        outcomes[-1]["ok"] = True
    elif control == "missing_outcomes":
        investigation["usage"] = {}
    elif control == "source_not_found":
        outcomes[-1]["error_class"] = "source_not_found"
    elif control == "other_error":
        outcomes[-1]["error_class"] = "http_error"
    elif control == "untyped_text":
        investigation["usage"] = {"model_text": "provider_diagnostic_body"}
    elif control == "malformed_outcome":
        outcomes[-1] = "provider_diagnostic_body"
    assert scoring._retryable_breakdown_failure_reason(row) == "unknown"


@pytest.mark.parametrize("reason", sorted(scoring.FAILURE_REASON_CODES - {"unknown"}))
def test_explicit_reason_anywhere_precedes_typed_diagnostic(reason):
    row = unavailable_evidence()
    row["verifier_gate_receipts"].append({"failure_reason_code": reason})
    assert scoring._retryable_breakdown_failure_reason(row) == reason


def test_scorer_entrypoint_and_actual_runner_preserve_terminal_and_retry_contract(
    monkeypatch, tmp_path,
):
    calls = []
    row = unavailable_evidence()

    def scorer(companies, icp, reference):
        calls.append(deepcopy(companies))
        return [deepcopy(row)]

    monkeypatch.setattr(scoring, "lab_scorer", lambda *_args, **_kwargs: scorer)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "1")
    document = scoring.build_scoring_input(
        scored_run_id="r1", icp=make_icp(0), companies=[company(0)],
        policy=scoring.build_scorer_policy(), evaluation_date="2026-10-12",
    )
    failure = scorer_entrypoint.score_input(document)
    assert len(calls) == scoring.MAX_JUDGE_RETRIES
    assert all(batch == [company(0)] for batch in calls)
    assert failure["failure"] == "judge_error"
    assert failure["reason"] == "provider_error"
    assert "breakdowns" not in failure
    assert scoring.scoring_output_from_bytes(json.dumps(failure).encode()) == failure

    api = FakeApi([scoring_lease()])
    sandbox = BridgingRuntime(output=failure, calls=0)
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert worker.run_once() == 1 and worker.abandoned == 0
    finally:
        worker.close()
    completion = api.completions[0]["body"]
    result = contracts.validate_run_result(json.loads(json.dumps(completion["result"])))
    assert result["terminal_status"] == "judge_error"
    assert result["failure_diagnostic"] == {
        "stage": "scorer", "error_class": "judge_error", "reason": "provider_error",
    }
    assert completion["output"] is None
    assert len(sandbox.specs) == 1
