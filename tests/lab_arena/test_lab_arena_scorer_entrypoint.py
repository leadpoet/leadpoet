"""The judge image entrypoint: scoring input in, breakdowns or a failure document out."""

from __future__ import annotations

import json
import os

import pytest

from lab_arena import contracts, scoring, shim
from lab_arena import scorer_entrypoint as entry

@pytest.fixture(autouse=True)
def clean_scorer_environment():
    """The entrypoint binds the policy and trusted mode into the process environment; restore it afterwards."""

    saved = dict(os.environ)
    for name in [
        shim.TRUSTED_SCORER_ENV,
        *scoring.POLICY_ENV_BINDINGS,
        *scoring.CREDENTIAL_ENV_NAMES,
    ]:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


ICP = {"icp_id": "arena:x", "prompt": "fintech startups", "max_companies": 3, "employee_count": ["11-50"], "company_stage": "Seed"}
COMPANIES = [{"company_name": "Acme %d" % i, "website": "https://acme%d.example" % i, "employee_count": "11-50"} for i in range(3)]


def scoring_input(tmp_path):
    document = scoring.build_scoring_input(scored_run_id="score-run-1", icp=ICP, companies=COMPANIES, policy=scoring.build_scorer_policy(), evaluation_date="2026-09-02")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(document))
    return document, path


def test_entrypoint_scores_with_the_lab_scorer_under_trusted_mode_and_placeholder_credentials(tmp_path, monkeypatch):
    seen = {}

    def fake_lab_scorer(policy):
        seen["policy"] = policy

        def score(companies, icp, is_reference_model):
            seen["environment"] = {name: os.environ.get(name) for name in scoring.CREDENTIAL_ENV_NAMES}
            seen["trusted"] = shim.trusted_scorer_mode()
            return [{"final_score": 60.0 + index, "failure_reason": ""} for index, _ in enumerate(companies)]

        return score

    monkeypatch.setattr(scoring, "lab_scorer", fake_lab_scorer)
    monkeypatch.delenv(shim.TRUSTED_SCORER_ENV, raising=False)
    document, input_path = scoring_input(tmp_path)
    output_path = tmp_path / "output.json"
    monkeypatch.setenv("LAB_ARENA_INPUT_PATH", str(input_path))
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))
    assert entry.main() == 0
    result = json.loads(output_path.read_text())
    assert result["schema_version"] == scoring.SCORING_OUTPUT_SCHEMA_VERSION and result["scored_run_id"] == document["scored_run_id"]
    assert [b["final_score"] for b in result["breakdowns"]] == [60.0, 61.0, 62.0]
    assert seen["trusted"] is True and seen["policy"] == document["scorer_policy"]
    assert all(value.startswith("arena-placeholder-") for value in seen["environment"].values())
    assert scoring.validate_scoring_output_document(result)["breakdowns"] == result["breakdowns"]


@pytest.mark.parametrize("error, expected", [
    (shim.ShimRequestError("budget_refused"), "judge_error"),
    (shim.ShimRequestError("budget_exhausted"), "judge_error"),
    (shim.ShimRequestError("no_matching_operation"), "judge_error"),
    (scoring.ScoringError("judge unavailable"), "judge_error"),
])
def test_entrypoint_reports_failures_as_documents_never_crashes(tmp_path, monkeypatch, error, expected):
    def failing_scorer(policy):
        def score(companies, icp, is_reference_model):
            raise error

        return score

    monkeypatch.setattr(scoring, "lab_scorer", failing_scorer)
    monkeypatch.setattr(scoring, "MAX_JUDGE_RETRIES", 1)
    document, _ = scoring_input(tmp_path)
    result = entry.score_input(document)
    assert {k: v for k, v in result.items() if k != "detail"} == {"schema_version": scoring.SCORING_OUTPUT_SCHEMA_VERSION, "scored_run_id": document["scored_run_id"], "failure": expected}
    assert result["detail"] and len(result["detail"]) <= scoring.MAX_FAILURE_DETAIL_CHARS  # a bounded operator-facing reason
    assert scoring.validate_scoring_output_document(result) == result


def test_entrypoint_rejects_a_foreign_input_schema(tmp_path):
    with pytest.raises(scoring.ScoringError):
        entry.score_input({"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": ICP})
