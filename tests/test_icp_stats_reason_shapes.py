"""build_icp_stats must never raise on non-string failure_reason shapes.

A structured (dict/list) failure_reason once made _categorize_reason call
.strip() on a non-string, raising inside the scorer wrapper's funnel stanza
and blanking the per-ICP funnel for the whole candidate — which is why loop
funnel charts showed no data.
"""
import pytest

from research_lab.eval.miner_report_stats import (
    _categorize_reason,
    build_icp_stats,
    normalize_failure_reporting_fields,
)


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("intent_fabricated", "intent_fabricated"),
        ("", "other"),
        (None, "other"),
        ({"code": "company_stage_mismatch"}, "company_stage_mismatch"),
        ({"reason": "company verification failed"}, "company_unverifiable"),
        (["employee_count_mismatch"], "employee_count_mismatch"),
        ("missing employee_count", "employee_count_missing"),
        ("missing company_stage", "company_stage_missing"),
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


def test_failure_reporting_fields_use_stable_stages_and_fill_missing_flags():
    assert normalize_failure_reporting_fields(
        {"final_score": 0, "failure_reason": "company_stage_mismatch", "stage_failed": "pre_checks"}
    ) == {
        "failure_stage": "firmographic",
        "reason_code": "company_stage_mismatch",
        "fit_passed": False,
        "attribute_passed": False,
        "intent_passed": None,
    }
    assert normalize_failure_reporting_fields(
        {"final_score": 0, "failure_reason": "intent_fabricated"}
    ) == {
        "failure_stage": "intent",
        "reason_code": "intent_fabricated",
        "fit_passed": True,
        "attribute_passed": True,
        "intent_passed": False,
    }


def test_failure_reporting_fields_preserve_explicit_flags_and_unknown_stage():
    assert normalize_failure_reporting_fields(
        {
            "final_score": 0,
            "failure_reason": "new_reason",
            "stage_failed": "custom_gate",
            "fit_passed": "true",
        }
    ) == {
        "failure_stage": "custom_gate",
        "reason_code": "other",
        "fit_passed": True,
        "attribute_passed": None,
        "intent_passed": None,
    }


def test_failure_reporting_fields_mark_accepted_rows_as_passed():
    assert normalize_failure_reporting_fields({"final_score": 12.5}) == {
        "failure_stage": None,
        "reason_code": None,
        "fit_passed": True,
        "attribute_passed": True,
        "intent_passed": True,
    }
