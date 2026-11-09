"""A completed missing-identity response is a negative, never an identity pass."""
import copy

import pytest

from qualification.scoring import lead_scorer as scorer
from tests.test_company_evidence_investigator import _company


def retained_minerva():
    # Exact three identity fields from the retained Nov9 score4 response9411270.
    company = _company().model_copy(update={
        "company_name": "Minerva Virtual Academy (MVA)",
        "company_website": "https://www.minervavirtual.com/",
        "company_linkedin": "https://www.linkedin.com/company/minerva-virtual-academy/",
    })
    verdict = {
        "observed_company_name": "Minerva Virtual Academy",
        "observed_company_website": "https://www.minervavirtual.com",
        "observed_company_linkedin": "",
    }
    investigation = {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": True, "failure_reason": "",
        "targets": ["industry", "required_attribute"],
        "claims": {}, "usage": {"fetch_outcomes": [{"ok": True}]},
    }
    return company, verdict, investigation


def terminal(verdict, receipt, investigation, *, completed=True, profile=""):
    return scorer._has_explicitly_unproven_fit_dimensions(
        verdict, ("identity",), identity_receipt=receipt,
        investigation_receipt=investigation,
        completed_identity_repair=completed, linkedin_refresh_outcome=profile,
    )


def test_retained_minerva_completed_missing_linkedin_is_terminal_unproven():
    company, verdict, investigation = retained_minerva()
    original = copy.deepcopy(verdict)
    receipt = scorer._web_identity_receipt(company, verdict)
    assert receipt["decision"] == "unavailable"
    assert receipt["reason_code"] == "identity_not_proven"
    assert receipt["submitted_name"] != receipt["observed_name"]
    assert scorer._is_explicitly_missing_web_linkedin_identity(verdict, receipt)
    assert terminal(verdict, receipt, investigation)
    assert verdict == original
    assert scorer._web_identity_receipt(company, verdict) == receipt


@pytest.mark.parametrize("fault", [
    "initial_only", "incomplete_investigation", "provider_error",
    "malformed_response", "fetch_failed", "profile_failed",
])
def test_completed_negative_does_not_hide_an_unfinished_or_failed_review(fault):
    company, verdict, investigation = retained_minerva()
    completed, profile = True, ""
    if fault == "initial_only":
        completed = False
    elif fault == "incomplete_investigation":
        investigation["completed_submitted_findings"] = False
    elif fault in {"provider_error", "malformed_response"}:
        investigation["failure_reason"] = fault
    elif fault == "fetch_failed":
        investigation["usage"]["fetch_outcomes"] = [
            {"ok": False, "error_class": "fetch_failed"},
        ]
    elif fault == "profile_failed":
        profile = "retryable_failure"
    receipt = scorer._web_identity_receipt(company, verdict)
    assert not terminal(verdict, receipt, investigation, completed=completed, profile=profile)
    assert receipt["decision"] != "match"


@pytest.mark.parametrize("field,value", [
    ("observed_company_name", ""), ("observed_company_name", None),
    ("observed_company_website", "https://other.example"),
    ("observed_company_linkedin", None), ("observed_company_linkedin", " "),
    ("observed_company_linkedin", "https://www.linkedin.com/company/other/"),
])
def test_missing_linkedin_negative_requires_complete_same_domain_observation(field, value):
    company, verdict, _investigation = retained_minerva()
    verdict[field] = value
    receipt = scorer._web_identity_receipt(company, verdict)
    assert not scorer._is_explicitly_missing_web_linkedin_identity(verdict, receipt)
    assert receipt["decision"] != "match"


def test_positive_identity_rules_remain_independent_of_negative_finality():
    company, verdict, _investigation = retained_minerva()
    verdict.update(observed_company_name=company.company_name,
                   observed_company_linkedin=company.company_linkedin)
    receipt = scorer._web_identity_receipt(company, verdict)
    assert receipt["decision"] == "match"
    assert not scorer._is_explicitly_missing_web_linkedin_identity(verdict, receipt)
