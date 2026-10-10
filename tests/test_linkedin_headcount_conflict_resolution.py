"""A fetched exact-profile band can corroborate a conflicting structured range."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_MATCH, COMPANY_FIT_UNAVAILABLE
from tests.test_company_evidence_investigator import _company, _complete_verdict, _finding, _icp

PROFILE = "https://www.linkedin.com/company/br-dge"
QUOTE = "BR-DGE Company size 51-200 employees"
STRUCTURED = {
    "employee_count": "51-200", "provider": "harvestapi_get_company",
    "source_field": "employeeCountRange", "url": PROFILE, "website": "https://br-dge.to/",
}
IDENTITY = {
    "normalized_name": "BR-DGE", "registrable_dns_domain": "br-dge.to",
    "linkedin_company_slug": "br-dge",
}


def _fixture():
    company = _company(name="BR-DGE", website="https://br-dge.to/", linkedin=PROFILE)
    icp = _icp(employee_count="51-200 or 201-500 or 501-1,000 or 1,001-5,000")
    verdict = _complete_verdict(
        observed_company_name="BR-DGE", observed_company_website="https://br-dge.to/",
        observed_company_linkedin=PROFILE, observed_employee_count="11-50",
        employee_size_matches=False,
        employee_size_evidence_url="https://fundediq.co/br-dge-br-dge-to-funding/",
        employee_size_evidence_quote="Edinburgh, United Kingdom 11-50 employees",
    )
    return company, icp, verdict


def _investigation():
    return {
        "claims": {"headcount": _finding(
            "headcount", observed_value="51-200", evidence_url=PROFILE,
            evidence_quote=QUOTE,
        )},
        "_completed_submit": True, "failure_reason": "",
        investigator.PRIVATE_FETCHED_PAGES_KEY: {
            PROFILE: {"final_url": PROFILE, "text": QUOTE},
        },
    }


def _caller(company, icp, verdict):
    return lead_scorer._run_targeted_company_evidence_investigation(
        company=company, icp=icp, verdict=verdict,
        investigation_targets=("headcount",), icp_attribute="", icp_stage="",
        verified_identity=IDENTITY, verified_transport_domain="br-dge.to",
        structured_employee_size_evidence=STRUCTURED,
        structured_public_company_evidence=None,
        employee_size_conflict=True, company_quality=True,
    )


def test_brdge_existing_investigator_fetches_and_resolves_exact_profile_band(monkeypatch):
    company, icp, verdict = _fixture()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        name, args = (
            ("fetch_page", {"url": PROFILE}) if len(requests) == 1 else
            ("submit_findings", {"findings": [_investigation()["claims"]["headcount"]]})
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    fetch = AsyncMock(return_value={"ok": True, "url": PROFILE, "final_url": PROFILE, "text": QUOTE})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    projected, result, *_ = asyncio.run(_caller(company, icp, verdict))
    initial = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    prior = initial["prior_observations"]
    assert prior["observed_employee_count"] == "11-50"
    assert prior["employee_size_evidence_quote"] == verdict["employee_size_evidence_quote"]
    assert prior["untrusted_headcount_conflict_context"] == {
        "structured_profile_evidence": STRUCTURED,
        "notice": "structured_range_is_discovery_context_not_a_fetched_quote",
    }
    assert PROFILE in prior["submitted_source_urls"]
    assert verdict["employee_size_evidence_url"] in prior["submitted_source_urls"]
    fetch.assert_awaited_once()
    assert projected["observed_employee_count"] == "51-200"
    assert projected["employee_size_evidence_quote"] == QUOTE
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    receipt = result.details["employee_size_conflict_receipt"]
    assert receipt["resolution"] == "fetched_linkedin_band_corroborates_structured_profile"
    assert receipt["web_evidence"]["quote"] == verdict["employee_size_evidence_quote"]
    assert receipt["structured_evidence"] == STRUCTURED
    assert receipt["fetched_evidence"] == {"url": PROFILE, "quote": QUOTE}
    assert result.details["investigation_receipt"]["usage"]["fetch_calls"] == 1
    assert (investigator.MAX_FETCH_CALLS, investigator.MAX_SEARCH_CALLS) == (3, 2)


@pytest.mark.parametrize("unsafe", [
    "wrong_profile", "off_domain", "different_band", "unproven", "incomplete",
    "provider_failure", "no_quote", "quote_absent", "not_fetched", "wrong_final_profile",
    "member_count", "integer_estimate", "wrong_company_quote", "bare_band_quote",
])
def test_caller_keeps_unresolved_conflict_without_exact_fetched_corroboration(monkeypatch, unsafe):
    company, icp, verdict = _fixture()
    output = _investigation()
    finding = output["claims"]["headcount"]
    if unsafe in {"wrong_profile", "off_domain"}:
        url = "https://www.linkedin.com/company/other" if unsafe == "wrong_profile" else "https://directory.example/br-dge"
        finding["evidence_url"] = url
        output[investigator.PRIVATE_FETCHED_PAGES_KEY] = {url: {"final_url": url, "text": QUOTE}}
    elif unsafe == "different_band":
        finding["observed_value"] = "201-500"
        finding["evidence_quote"] = "BR-DGE Company size 201-500 employees"
        output[investigator.PRIVATE_FETCHED_PAGES_KEY][PROFILE]["text"] = finding["evidence_quote"]
    elif unsafe == "unproven":
        finding["status"] = "UNPROVEN"
    elif unsafe == "incomplete":
        output["_completed_submit"] = False
    elif unsafe == "provider_failure":
        output["failure_reason"] = "company_evidence_provider_error"
    elif unsafe == "no_quote":
        finding["evidence_quote"] = ""
    elif unsafe == "quote_absent":
        output[investigator.PRIVATE_FETCHED_PAGES_KEY][PROFILE]["text"] = "BR-DGE About us"
    elif unsafe == "not_fetched":
        output.pop(investigator.PRIVATE_FETCHED_PAGES_KEY)
    elif unsafe == "wrong_final_profile":
        output[investigator.PRIVATE_FETCHED_PAGES_KEY][PROFILE]["final_url"] = "https://www.linkedin.com/company/other"
    elif unsafe == "member_count":
        finding["evidence_quote"] = "BR-DGE has 51-200 associated members"
        output[investigator.PRIVATE_FETCHED_PAGES_KEY][PROFILE]["text"] = finding["evidence_quote"]
    elif unsafe == "integer_estimate":
        finding["observed_value"] = 99
    elif unsafe in {"wrong_company_quote", "bare_band_quote"}:
        finding["evidence_quote"] = (
            "Other company Company size 51-200 employees"
            if unsafe == "wrong_company_quote" else "Company size 51-200 employees"
        )
        output[investigator.PRIVATE_FETCHED_PAGES_KEY][PROFILE]["text"] = finding["evidence_quote"]
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", AsyncMock(return_value=output))
    projected, result, *_ = asyncio.run(_caller(company, icp, verdict))
    assert projected["observed_employee_count"] == "11-50"
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_UNAVAILABLE
    assert result.details.get("employee_size_conflict_receipt", {}).get("resolution") != (
        "fetched_linkedin_band_corroborates_structured_profile"
    )


@pytest.mark.parametrize("overrides", [
    {"url": "https://www.linkedin.com/company/other"},
    {"website": "https://other.example"},
    {"source_field": "employeeCount"},
    {"employee_count": "99"},
    {"provider": "other"},
])
def test_conflict_context_rejects_unbound_noncanonical_or_wrong_source_evidence(overrides):
    _, icp, _ = _fixture()
    assert lead_scorer._headcount_conflict_context(dict(STRUCTURED, **overrides), IDENTITY, icp) == {}


def test_projection_alone_cannot_bypass_an_existing_conflict():
    _, icp, verdict = _fixture()
    projected = lead_scorer._project_investigator_headcount(
        verdict, _investigation()["claims"]["headcount"], icp=icp, existing_conflict=True,
    )
    assert projected == verdict
