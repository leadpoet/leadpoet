"""An explicit requested-profile absence is evidence exhaustion, not credit."""
from __future__ import annotations

import asyncio
import copy

import pytest

from qualification.scoring import lead_scorer, linkedin_company_size
from tests.test_current_linkedin_company_size import (
    _company, _icp, _install_exa_bodies, _structured_company_payload, _verdict,
)
from qualification.scoring.company_fit_decision import company_fit_match


def absent():
    # Same public provider envelope as the captured target3 packet40.
    return {"job_id": "offline-execute", "status": "completed",
            "result": {"data": {"error": None, "status": 404, "element": None}}}


@pytest.mark.parametrize("body,reason", [
    (absent(), None),
    ({"error": None, "status": 404, "element": None}, None),
    ({"status": 404, "element": None}, "provider_error"),
    ({"error": None, "status": "404", "element": None}, "malformed_response"),
    ({"error": None, "status": 200, "element": None}, "malformed_response"),
    ({"error": None, "element": None}, "malformed_response"),
    ({"error": None, "status": 403, "element": None}, "provider_error"),
    ({"error": None, "status": 429, "element": None}, "provider_error"),
    ({"error": None, "status": 502, "element": None}, "provider_error"),
    ({"error": None, "status": 503, "element": None}, "provider_error"),
    (dict(absent(), status="failed"), "provider_error"),
    (dict(absent(), status=429), "provider_error"),
    (dict(absent(), error="account unavailable"), "provider_error"),
    ({"status": "completed", "result": absent()["result"],
      "data": {"status": 502, "error": "provider unavailable"}}, "provider_error"),
])
def test_exact_absence_and_provider_failure_controls(monkeypatch, body, reason):
    _install_exa_bodies(monkeypatch, body)
    monkeypatch.setenv("DEEPLINE_API_KEY", "offline-key")
    diagnostic = {}
    result = asyncio.run(linkedin_company_size.fetch_structured_linkedin_company_size(
        "acme.example.com", "https://www.linkedin.com/company/acme", diagnostic=diagnostic,
    ))
    assert result is None
    assert diagnostic == ({"failure_reason": reason} if reason else {})


def test_valid_200_profile_still_requires_exact_identity_and_projects_size(monkeypatch):
    _install_exa_bodies(monkeypatch, _structured_company_payload())
    monkeypatch.setenv("DEEPLINE_API_KEY", "offline-key")
    diagnostic = {}
    result = asyncio.run(linkedin_company_size.fetch_structured_linkedin_company_size(
        "acme.example.com", "https://www.linkedin.com/company/acme", diagnostic=diagnostic,
    ))
    assert result["employee_count"] == "11-50"
    assert diagnostic == {}


@pytest.mark.parametrize("structured_status,completed,expected_class", [
    (404, True, "insufficient_fit_evidence"), (404, False, None),
    (429, True, "employee_size_verification_failed"),
    (502, True, "employee_size_verification_failed"),
])
def test_complete_unproven_review_keeps_company_unqualified_and_transport_retryable(
    monkeypatch, structured_status, completed, expected_class,
):
    company = _company(linkedin="")
    company.company_name = "Acme.US"
    company.company_website = "https://acme.com"
    icp = _icp()
    icp.company_stage = "Series A"
    icp.required_attribute = "Provides software and is actively growing."
    verdict = {key: value.replace("acme.example.com", "acme.com") if isinstance(value, str) else value
               for key, value in _verdict().items()}
    verdict.update(observed_company_name="Acme", observed_company_website=company.company_website,
                   observed_company_linkedin="https://www.linkedin.com/company/acme",
                   observed_company_stage="", stage_matches=None, stage_evidence_url="", stage_evidence_quote="",
                   attribute_satisfied=None, required_attribute_evidence_url="", required_attribute_evidence_quote="")
    claims = {target: {"target": target, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
              for target in ("rebrand", "headcount", "stage", "required_attribute")}
    claims["industry"] = {"target": "industry", "status": "VERIFIED", "activity_role": "supplier_operator",
                           "observed_industry": "Software", "observed_subindustry": "SaaS",
                           "evidence_url": "https://acme.com/product",
                           "evidence_quote": "Acme provides workflow software for customers."}
    exa_not_found = {"results": [], "statuses": [{"id": "https://www.linkedin.com/company/acme",
                     "status": "error", "error": {"httpStatusCode": 404, "tag": "CRAWL_NOT_FOUND"}}]}
    profile = absent()
    profile["result"]["data"]["status"] = structured_status
    _install_exa_bodies(monkeypatch, exa_not_found, exa_not_found, profile)
    monkeypatch.setenv("DEEPLINE_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")

    async def homepage(*args, **kwargs):
        return company_fit_match(details={"identity": {
            "decision": "match", "evidence_source": "company_homepage",
            "submitted_name": "acmeus", "observed_name": "acmeus",
            "submitted_domain": "acme.com", "observed_domain": "acme.com",
            "submitted_linkedin_slug": "", "observed_linkedin_slug": "acme",
        }})

    async def broad(**kwargs):
        return copy.deepcopy(verdict), ""

    async def investigation(**kwargs):
        return {"claims": copy.deepcopy(claims), "_completed_submit": completed, "failure_reason": "",
                "usage": {"fetch_outcomes": [{"ok": True, "loaded_text_length": 100}] * 4}}

    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigation)
    result = asyncio.run(lead_scorer._verify_company_fit(
        company, icp, 0.0, 0.0, set(), require_https_transport=True, evidence_investigator=True,
    ))
    assert result.decision == "unavailable", (result.reason, result.details)
    assert result.details.get("failure_class") == expected_class
    assert result.details["company_fit_dimensions"]["identity"] == "unavailable"
    assert result.details["company_fit_dimensions"]["employee_size"] == "unavailable"
    assert result.details["required_attribute_decision"] == "unavailable"
    from qualification.scoring.competition import scorer_breakdown_has_retryable_infrastructure_failure
    async def completed_fit(*args, **kwargs):
        return result

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", completed_fit)
    breakdown = asyncio.run(lead_scorer.score_company_competition_intent(
        company=company, icp=icp, run_cost_usd=0.0, run_time_seconds=0.0,
        seen_companies=set(), is_reference_model=False, integrity_policy=True,
        evidence_investigator=True,
    ))
    row = breakdown.model_dump(mode="json")
    assert row["final_score"] == 0.0
    assert scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True) is (structured_status != 404 or not completed)
