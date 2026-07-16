"""False-positive penalty scoring: junk companies subtract from the ICP score.

A company zeroed by a model-controllable gate (or carrying an unverified
primary intent) applies -X penalty points to the ICP's pre-normalization
sum, so per-ICP scores can go negative (floored) and drag the benchmark
mean: shipping junk must score worse than honestly returning fewer
companies. Both knobs default to 0 = off, preserving historical scores
exactly; the verifier recomputes with the penalty points recorded in the
bundle policy.
"""

import hashlib

import pytest

from research_lab.eval import evaluator
from leadpoet_verifier.research_evaluation import compute_evaluation_aggregates


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Penalty arithmetic (capped mode)
# ---------------------------------------------------------------------------

def test_penalty_subtracts_before_normalization(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # (80 - 50) / 3
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=3, fp_penalty_total=50.0
    ) == pytest.approx(10.0)
    # No penalty -> unchanged
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=3
    ) == pytest.approx(80.0 / 3)


def test_penalty_can_push_icp_negative_with_floor(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # (0 - 4*25)/1 = -100... with goal 1 and four junk companies
    assert evaluator.benchmark_icp_score_from_company_scores(
        [], requested_count=1, fp_penalty_total=100.0
    ) == pytest.approx(-100.0)
    # Catastrophic penalties clamp at the floor
    assert evaluator.benchmark_icp_score_from_company_scores(
        [], requested_count=1, fp_penalty_total=100000.0
    ) == pytest.approx(-100.0)


def test_negative_icp_drags_the_benchmark_mean(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    rows = [
        {"candidate_company_scores": [80.0, 80.0, 80.0], "icp_company_goal": 3},
        # 4 junk companies: (0 - 4*25)/2 = -50
        {
            "candidate_company_scores": [],
            "icp_company_goal": 2,
            "candidate_fp_gate_count": 4,
        },
    ]
    score = evaluator._benchmark_style_score(rows, "candidate_company_scores")
    assert score == pytest.approx((80.0 + -50.0) / 2)


def test_knobs_default_off_keeps_historical_scores(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.delenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", raising=False)
    monkeypatch.delenv(
        "RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", raising=False
    )
    rows = [
        {
            "candidate_company_scores": [60.0],
            "icp_company_goal": 5,
            "candidate_fp_gate_count": 7,
            "candidate_fp_unverified_primary_count": 3,
        }
    ]
    assert evaluator._benchmark_style_score(
        rows, "candidate_company_scores"
    ) == pytest.approx(12.0)


def test_side_specific_counts(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "10")
    row = {
        "base_company_scores": [50.0],
        "candidate_company_scores": [50.0],
        "icp_company_goal": 5,
        "base_fp_gate_count": 5,        # only the base side is penalized
        "candidate_fp_gate_count": 0,
    }
    base = evaluator._benchmark_style_score([row], "base_company_scores")
    cand = evaluator._benchmark_style_score([row], "candidate_company_scores")
    assert base == pytest.approx(0.0)   # (50 - 50)/5
    assert cand == pytest.approx(10.0)  # 50/5


# ---------------------------------------------------------------------------
# FP taxonomy
# ---------------------------------------------------------------------------

def _bd(reason=None, details=None):
    row = {"final_score": 0.0 if reason else 42.0, "failure_reason": reason}
    if details is not None:
        row["intent_signals_detail"] = details
    return row


def test_penalizable_gate_reasons_counted():
    breakdowns = [
        _bd("Company is on the ICP exclusion list: acme"),
        _bd("required_attribute validation did not pass"),
        _bd("Country mismatch: 'Germany' vs ICP 'United States'"),
        _bd("Duplicate company: 'Acme' already scored this evaluation"),
        _bd("Data quality issue: company_website is example/placeholder"),
        _bd("Company verification failed: stage differs"),
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 6
    assert primary == 0


def test_infra_failures_never_penalized():
    breakdowns = [
        _bd("LLM scoring error: timeout talking to provider"),
        _bd("Scorer error: HTTP 429 from provider"),
        _bd("provider timeout during verification"),
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 0 and primary == 0


def test_unverified_primary_intent_counted_separately():
    verified = [{"matched_icp_signal": 0, "after_decay": 12.0}]
    bonus_only = [{"matched_icp_signal": 1, "after_decay": 30.0}]
    breakdowns = [
        _bd(details=verified),     # primary verified -> not an FP
        _bd(details=bonus_only),   # only bonus verified -> unverified primary
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=True
    )
    assert gate == 0
    assert primary == 1
    # ICPs with no intent signals never produce primary FPs.
    gate2, primary2 = evaluator.count_penalizable_false_positives(
        breakdowns, icp_has_intent_signals=False
    )
    assert primary2 == 0


def test_verifier_infrastructure_error_fails_open():
    # A primary rejected because the three-stage verifier CRASHED (provider
    # outage) must not count as falsified intent — fail open per company.
    errored = [
        {
            "matched_icp_signal": -1,
            "after_decay": 0.0,
            "judge_verdict": {
                "decision": "rejected_verifier_error",
                "rejection_reason": "three_stage_exception",
                "error_class": "ReadTimeout",
            },
        }
    ]
    content_rejected = [
        {
            "matched_icp_signal": -1,
            "after_decay": 0.0,
            "judge_verdict": {
                "decision": "rejected_three_stage",
                "rejection_reason": "claim_not_supported_by_source",
            },
        }
    ]
    gate, primary = evaluator.count_penalizable_false_positives(
        [_bd(details=errored), _bd(details=content_rejected)],
        icp_has_intent_signals=True,
    )
    assert gate == 0
    assert primary == 1  # only the content rejection counts


def test_fp_penalty_total_helper(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", "10")
    breakdowns = [
        _bd("Country mismatch: 'Chile' vs ICP 'United States'"),
        _bd(details=[{"matched_icp_signal": 1, "after_decay": 5.0}]),
    ]
    icp = {"intent_signals": ["hiring engineers"]}
    total = evaluator.fp_penalty_total_from_breakdowns(breakdowns, icp)
    assert total == pytest.approx(25.0 + 10.0)


def test_fake_intent_inherits_main_penalty_by_default(monkeypatch):
    # Falsified intent is a false positive like any other: the ONE main knob
    # penalizes it at the same rate unless explicitly overridden.
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_PENALTY_POINTS", "25")
    monkeypatch.delenv(
        "RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", raising=False
    )
    breakdowns = [
        _bd(details=[{"matched_icp_signal": 1, "after_decay": 5.0}]),  # bonus only
    ]
    icp = {"intent_signals": ["hiring engineers"]}
    total = evaluator.fp_penalty_total_from_breakdowns(breakdowns, icp)
    assert total == pytest.approx(25.0)
    # Explicit override can weight deception harder than ordinary non-fit.
    monkeypatch.setenv("RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY", "50")
    assert evaluator.fp_penalty_total_from_breakdowns(
        breakdowns, icp
    ) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Verifier parity
# ---------------------------------------------------------------------------

def test_verifier_recomputes_with_recorded_penalties():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 2,
            "base_company_scores": [],
            "candidate_company_scores": [80.0, 80.0],
            "candidate_fp_gate_count": 2,
        },
    ]
    aggregates = compute_evaluation_aggregates(
        rows, fp_penalty_points=25.0
    )
    row = aggregates["per_icp_results"][0]
    # (160 - 2*25)/2 = 55
    assert row["candidate_per_icp_score"] == pytest.approx(55.0)
    assert row["candidate_fp_gate_count"] == 2
    assert aggregates["fp_penalty_points"] == pytest.approx(25.0)


def test_verifier_zero_knobs_matches_legacy():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 2,
            "base_company_scores": [],
            "candidate_company_scores": [80.0, 80.0],
            "candidate_fp_gate_count": 9,
        },
    ]
    aggregates = compute_evaluation_aggregates(rows)
    assert aggregates["per_icp_results"][0]["candidate_per_icp_score"] == pytest.approx(80.0)


def test_verifier_floor_applies():
    rows = [
        {
            "icp_ref": "icp:a",
            "icp_hash": _sha("1"),
            "icp_company_goal": 1,
            "base_company_scores": [],
            "candidate_company_scores": [],
            "candidate_fp_gate_count": 50,
        },
    ]
    aggregates = compute_evaluation_aggregates(rows, fp_penalty_points=100.0)
    assert aggregates["per_icp_results"][0]["candidate_per_icp_score"] == pytest.approx(-100.0)
