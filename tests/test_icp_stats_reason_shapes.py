"""build_icp_stats must never raise on non-string failure_reason shapes.

A structured (dict/list) failure_reason once made _categorize_reason call
.strip() on a non-string, raising inside the scorer wrapper's funnel stanza
and blanking the per-ICP funnel for the whole candidate — which is why loop
funnel charts showed no data.
"""
import pytest

from research_lab.eval.miner_report_stats import build_icp_stats, _categorize_reason


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("intent_fabricated", "intent_fabricated"),
        ("", "other"),
        (None, "other"),
        ({"code": "company_stage_mismatch"}, "company_stage_mismatch"),
        ({"reason": "company verification failed"}, "company_unverifiable"),
        (["employee_count_mismatch"], "employee_count_mismatch"),
        ([], "other"),
        ({}, "other"),
    ],
)
def test_categorize_reason_tolerates_non_string(reason, expected):
    assert _categorize_reason(reason) == expected


def test_build_icp_stats_funnel_with_structured_reasons():
    breakdowns = [
        {"final_score": 54.0, "failure_reason": None, "intent_signal_final": 54.0},
        {"final_score": 0.0, "failure_reason": {"code": "intent_fabricated"}},
        {"final_score": 0.0, "failure_reason": ["company_stage_mismatch"]},
        {"final_score": 0.0, "failure_reason": "company_unverifiable"},
    ]
    stats = build_icp_stats(sourced_count=6, breakdowns=breakdowns)
    funnel = stats["funnel"]
    assert funnel["sourced"] == 6
    assert funnel["scored"] == 1
    # one verify-fail passed fit; one intent-fail (dict) passed fit+verify;
    # one fit-fail (list) died early; plus the scored company.
    assert funnel["fit_pass"] >= funnel["verified"] >= funnel["intent_valid"] >= funnel["scored"]
    # the prefilter gap (sourced 6 - 4 rows) books as fit failures, never raises
    assert funnel["fit_pass"] <= funnel["sourced"]
