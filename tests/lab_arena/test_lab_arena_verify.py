"""Focused tests for Arena score math, ranking, and public score rows."""

from __future__ import annotations

from typing import Any

import pytest

from leadpoet_verifier.research_evaluation import compute_evaluation_aggregates
from qualification.scoring import competition as evaluator

from lab_arena import contracts, scoring, verify
from lab_arena.contracts import ArenaContractError


POLICY = scoring.build_scorer_policy()


def _receipt(decision: str) -> dict:
    dimensions = {
        "identity": decision,
        "employee_size": decision,
        "industry": decision,
        "geography": decision,
        "stage": "match",
    }
    return {
        "gate": "company_fit",
        "contract_id": "company-fit-decision:v1",
        "contract_version": "company-fit-decision:v1",
        "decision": decision,
        "reason": "private reason",
        "company_fit_decision": decision,
        "company_fit_dimensions": dimensions,
        "company_fit_stage_required": False,
        "required_attribute_decision": "match",
        "dimension_evidence": {
            name: {
                "decision": value,
                "submitted_decision": value,
                "observed_decision": value,
                "web_evidence": {"url": "https://example.com", "quote": "private quote"},
            }
            for name, value in dimensions.items()
        },
        "identity_receipt": {"raw_response": "private"},
        "provider_observations": {"raw_response": "private"},
        "supporting_receipts": [{"raw_response": "private"}],
    }


def _signal(matched: int, score: float) -> dict:
    return {
        "raw": score,
        "after_decay": score,
        "decay": 1.0,
        "confidence": 90,
        "date_status": "verified",
        "matched_icp_signal": matched,
        "evidence_type": "HIRING",
        "quote": "private quote",
        "snippet": "private snippet",
        "judge_verdict": {
            "decision": "verified",
            "pipeline_decision": "verified",
            "stage1_status": "ok",
            "client_ready": True,
            "verification_trace": {"quote": "private"},
        },
    }


def _breakdown(score: float, *, failure_reason: Any = None, details: Any = None) -> dict:
    return {
        "icp_fit": 0.0,
        "decision_maker": 0.0,
        "intent_signal_raw": score,
        "time_decay_multiplier": 1.0,
        "intent_signal_final": score,
        "cost_penalty": 0.0,
        "time_penalty": 0.0,
        "final_score": score,
        "failure_reason": failure_reason,
        "intent_signals_detail": [_signal(0, score)] if details is None else details,
        "verifier_gate_receipts": [_receipt("match")],
        "judge_prompt": "private prompt",
        "page_content": "private page",
    }


def _junk() -> dict:
    return _breakdown(
        0.0,
        failure_reason="Company is on the ICP exclusion list: acme",
        details=[],
    )


def _icp(max_companies: int = 5) -> dict:
    return {
        "industry": "Software",
        "employee_count": "51-200",
        "employee_count_buckets": ["11-50", "51-200", "201-500"],
        "max_companies": max_companies,
        "intent_signals": ["Hiring backend engineers"],
    }


def test_per_icp_score_matches_the_shared_evaluation_math():
    icp = _icp()
    breakdowns = [
        _breakdown(80.0),
        _breakdown(60.0),
        _breakdown(40.0),
        _junk(),
        _breakdown(30.0, details=[_signal(1, 30.0)]),
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    expected = compute_evaluation_aggregates(
        [{
            "icp_ref": "current",
            "icp_company_goal": 5,
            "base_company_scores": [],
            "candidate_company_scores": [80.0, 60.0, 40.0, 0.0, 30.0],
            "candidate_fp_gate_count": gate,
            "candidate_fp_unverified_primary_count": primary,
        }],
        leads_per_icp_normalizer=5,
        fp_penalty_points=10.0,
        fp_unverified_primary_penalty_points=10.0,
        fp_penalty_icp_floor=0.0,
    )["per_icp_results"][0]["candidate_per_icp_score"]

    result = verify.per_icp_score(icp, breakdowns, POLICY)
    assert expected == 38.0
    assert result["per_icp_score"] == expected
    assert (result["fp_gate_count"], result["fp_unverified_primary_count"]) == (1, 1)
    redacted = [verify.redact_breakdown(item) for item in breakdowns]
    assert verify.per_icp_score(icp, redacted, POLICY)["per_icp_score"] == expected
    with pytest.raises(ArenaContractError):
        verify.per_icp_score(icp, ["not-an-object"], POLICY)


def test_first_n_slice_and_employee_bucket_skip_are_recomputed():
    icp = _icp(max_companies=3)
    companies = [
        {"employee_count": "51-200"},
        {"employee_count": "2-10"},
        {"employee_count": 120},
        {},
        {"employee_count": "51-200"},
        {"employee_count": "51-200"},
    ]
    assert verify.slice_first_n(companies, 3) == companies[:3]
    assert verify.bucket_skip(icp, companies) == ([0, 2, 4], [1, 3])
    assert verify.bucket_skip(icp, companies, max_scored_companies=2) == ([0, 2], [1])
    assert verify.icp_company_goal({"max_companies": 500}) == 5
    assert verify.icp_company_goal({"max_companies": 0}) == 1
    assert verify.icp_company_goal({}) == 5
    with pytest.raises(ArenaContractError):
        verify.bucket_skip(icp, ["not-a-company"])


def test_stage_score_requires_the_exact_stage_or_final_icp_count():
    assert verify.stage_score([0.1] * 10, 10) == 0.1
    assert verify.stage_score([0.1] * 20, 20) == 0.1
    assert verify.stage_score(list(range(20)), 20) == verify.stage_score(list(reversed(range(20))), 20)
    for scores, denominator in (([1.0] * 9, 10), ([1.0] * 19, 20), ([1.0] * 15, 15), ([1.0] * 29, 30)):
        with pytest.raises(ArenaContractError):
            verify.stage_score(scores, denominator)


def test_zero_rows_use_only_current_terminal_causes():
    for cause in verify.ZERO_ROW_CAUSES:
        row = verify.zero_row("sub-a", 3, cause)
        assert row["per_icp_score"] == 0.0
        assert row["breakdowns"] == []
        assert "scored_run_id" not in row
    for cause in ("accepted", "preflight_failed", "judge_key_refused", ""):
        with pytest.raises(ArenaContractError):
            verify.zero_row("sub-a", 3, cause)


def _entry(name: str, score: Any, king: bool = False) -> dict:
    return {
        "submission_id": name,
        "hotkey": "hotkey-" + name,
        "final_score": score,
        "is_king": king,
    }


def test_ranking_finalist_cut_and_king_decisions_do_not_use_image_identity():
    stage_entries = [
        {"submission_id": "sub-%02d" % index, "stage1_score": float(index), "is_king": False}
        for index in range(12)
    ] + [{"submission_id": "king", "stage1_score": 100.0, "is_king": True}]
    ranking = verify.stage1_ranking(stage_entries)
    assert len(verify.select_finalists(ranking)) == contracts.FINALIST_COUNT
    assert ranking[0]["submission_id"] == "sub-11"

    king = _entry("king", 50.0, True)
    tie = verify.king_decision([_entry("a", 50.0)], king)
    assert tie["outcome"] == "no_king" and tie["winner_submission_id"] is None
    crowned = verify.king_decision([_entry("b", 60.0), _entry("a", 60.0)], king)
    assert crowned["outcome"] == "crowned" and crowned["winner_submission_id"] == "a"
    zero_wins = verify.king_decision([_entry("a", 0.0)], _entry("king", None, True))
    assert zero_wins["outcome"] == "no_king" and zero_wins["winner_submission_id"] is None
    zero_tie = verify.king_decision([_entry("a", 0.0)], _entry("king", 0.0, True))
    assert zero_tie["outcome"] == "no_king" and zero_tie["winner_submission_id"] is None
    assert verify.king_decision([_entry("a", 100.0)], None)["outcome"] == "no_king"

    final = verify.final_ranking([
        _entry("a", 40.0),
        _entry("king", 40.0, True),
        _entry("b", None),
        _entry("c", 55.0),
    ])
    assert [row["submission_id"] for row in final] == ["c", "king", "a", "b"]
    assert [row["is_baseline"] for row in final] == [False, True, False, False]
    assert all("is_king" not in row for row in final)


def _walk_keys(value: Any) -> set:
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _walk_keys(item)
    return keys


def test_breakdown_redaction_keeps_score_inputs_and_removes_payloads():
    originals = [_breakdown(70.0), _junk()]
    redacted = [verify.redact_breakdown(item) for item in originals]
    for intent in (True, False):
        assert evaluator.count_penalizable_false_positives(
            redacted, icp_has_intent_signals=intent
        ) == evaluator.count_penalizable_false_positives(
            originals, icp_has_intent_signals=intent
        )
    forbidden = {
        "judge_prompt",
        "page_content",
        "quote",
        "snippet",
        "raw_response",
        "supporting_receipts",
        "identity_receipt",
        "provider_observations",
        "web_evidence",
    }
    for row in redacted:
        assert not (_walk_keys(row) & forbidden)
        assert verify.breakdown_is_redacted(row)
        assert verify.redact_breakdown(row) == row


def test_result_validity_requires_every_position_and_one_accepted_row():
    positions = tuple(range(contracts.STAGE_1_ICP_COUNT))
    rows = {position: verify.zero_row("sub-a", position, "model_timeout") for position in positions}
    assert verify.result_is_valid(rows, positions) is False
    rows[3] = verify.scored_row(
        "sub-a",
        3,
        "run-03",
        _icp(max_companies=1),
        [{"employee_count": "51-200"}],
        [_breakdown(10.0)],
        POLICY,
    )
    assert verify.result_is_valid(rows, positions) is True
    del rows[5]
    assert verify.result_is_valid(rows, positions) is False
