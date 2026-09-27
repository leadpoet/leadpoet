from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_evidence_investigator import _validated_findings
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    evaluate_company_identity,
)


NUMERIC_LINKEDIN = "https://www.linkedin.com/company/39624"
VANITY_LINKEDIN = "https://www.linkedin.com/company/rapid7"
PRODUCT_URL = "https://www.rapid7.com/products/insightcloudsec/"
PRODUCT_QUOTE = (
    "Key features Real-time visibility across clouds Sensitive data discovery "
    "Risk-based prioritization Cloud compliance management Cloud infrastructure "
    "entitlement management (CIEM) Agentless vulnerability management "
    "Infrastructure-as-code (IaC) security Automation remediation Kubernetes "
    "security posture management (KSPM) Cloud threat detection Use cases Monitor "
    "Assets Assess Risk Policy Management Access Management See all cloud assets "
    "in one place Enterprises are using the cloud to drive innovation and digital "
    "transformation. However, most security and operations teams lack unified "
    "visibility into the various cloud services being used by their development "
    "teams. InsightCloudSec enables continuous monitoring of all your cloud and "
    "container services in one user-friendly platform with better insights into "
    "associated risks."
)
PRODUCT_PAGE = "Rapid7 Cloud Security Packages INSIGHTCLOUDSEC\n" + PRODUCT_QUOTE
SEC_URL = (
    "https://www.sec.gov/Archives/edgar/data/1560327/"
    "000156032726000043/rp-20260807.htm"
)
SUBMITTED_STAGE_QUOTE = (
    "Rapid7, Inc. (Exact name of registrant as specified in its charter) Delaware "
    "001-37496 35-2423994 (State or other jurisdiction of incorporation) "
    "(Commission File Number) (IRS Employer Identification No.) 120 Causeway "
    "Street, Boston, Massachusetts 02114 (Address of principal executive offices), "
    "including zip code ( 617 ) 247-1717 (Registrant’s telephone number, including "
    "area code) Not Applicable (Former name, or former address, if changed since "
    "last report) Check the appropriate box below if the Form 8-K filing is intended "
    "to simultaneously satisfy the filing obligation under any of the following "
    "provisions: ☐ Written communications pursuant to Rule 425 under the Securities "
    "Act (17 CFR 230.425) ☐ Soliciting material pursuant to Rule 14a-12 under the "
    "Exchange Act (17 CFR 240.14a-12) ☐ Pre-commencement communications pursuant "
    "to Rule 14d-2(b) under the Exchange Act (17 CFR 240.14d-2(b)) ☐ "
    "Pre-commencement communications pursuant to Rule 13e-4(c) under the Exchange "
    "Act (17 CFR 240.14d-4(c)) ☐ Pre-commencement communications pursuant to Rule "
    "13e-4(c) under the Exchange Act (17 CFR 240.14d-4(c)) Securities registered "
    "pursuant to Section 12(b) of the Securities Exchange Act of 1934: Title of "
    "each class Trading symbol(s) Name of each exchange on which registered Common "
    "Stock, $0.01 par value per share RPD The Nasdaq Global Market"
)
SEC_PAGE = (
    "Rapid7, Inc. (Exact name of registrant as specified in its charter) Delaware "
    "001-37496 35-2423994 (State or other jurisdiction of incorporation) "
    "(Commission File Number) (IRS Employer Identification No.) 120 Causeway Street, "
    "Boston, Massachusetts 02114 (Address of principal executive offices), including "
    "zip code (617) 247-1717 (Registrant’s telephone number, including area code) "
    "Not Applicable (Former name, or former address, if changed since last report) "
    "Check the appropriate box below if the Form 8-K filing is intended to "
    "simultaneously satisfy the filing obligation of the registrant under any of the "
    "following provisions: Written communications pursuant to Rule 425 under the "
    "Securities Act (17 CFR 230.425) Soliciting material pursuant to Rule 14a-12 "
    "under the Exchange Act (17 CFR 240.14a-12) Pre-commencement communications "
    "pursuant to Rule 14d-2(b) under the Exchange Act (17 CFR 240.14d-2(b)) "
    "Pre-commencement communications pursuant to Rule 13e-4(c) under the Exchange "
    "Act (17 CFR 240.13e-4(c)) Securities registered pursuant to Section 12(b) of "
    "the Securities Exchange Act of 1934: Title of each class Trading symbol(s) "
    "Name of each exchange on which registered Common Stock, $0.01 par value per "
    "share RPD The Nasdaq Global Market"
)


def _company() -> CompanyOutput:
    return CompanyOutput(
        company_name="Rapid7",
        company_website="https://rapid7.com",
        company_linkedin=NUMERIC_LINKEDIN,
        industry="Cybersecurity",
        employee_count="1,001-5,000",
        country="United States",
        state="Massachusetts",
        intent_signals=[{
            "description": "Rapid7 published a current company filing.",
            "source": "company_website",
            "url": SEC_URL,
            "date": "2026-08-07",
            "snippet": "Rapid7 common stock trades under RPD.",
        }],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="numeric-alias-investigator-continuity",
        prompt="test",
        industry="Cloud security",
        sub_industry="Cloud identity and access management",
        employee_count="1,001-5,000",
        company_stage="Public",
        geography="United States",
        country="United States",
        product_service="Cloud security platform",
        required_attribute="",
    )


def _verified_homepage_identity() -> dict[str, str]:
    return {
        "normalized_name": "rapid7",
        "registrable_dns_domain": "rapid7.com",
        "linkedin_company_slug": "39624",
    }


def _structured_identity(**updates: str) -> dict[str, str]:
    evidence = {
        "name": "Rapid7, Inc.",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": VANITY_LINKEDIN,
        "website": "https://rapid7.com/",
        "company_id": "39624",
        "requested_url": NUMERIC_LINKEDIN,
    }
    evidence.update(updates)
    return evidence


def _verdict() -> dict[str, object]:
    return {
        "observed_company_name": "Rapid7, Inc.",
        "observed_company_website": "https://www.rapid7.com",
        "observed_company_linkedin": VANITY_LINKEDIN,
        "observed_employee_count": "1,001-5,000",
        "employee_size_matches": True,
        "employee_size_evidence_url": VANITY_LINKEDIN,
        "employee_size_evidence_quote": "Company size 1,001-5,000 employees",
        "observed_industry": "Cloud security",
        "observed_subindustry": "Cloud identity and access management",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": PRODUCT_URL,
        "industry_evidence_quote": PRODUCT_QUOTE,
        "observed_hq_country": "United States",
        "observed_hq_state": "Massachusetts",
        "geography_matches": True,
        "geography_evidence_url": "https://www.rapid7.com/about/",
        "geography_evidence_quote": "Rapid7 is headquartered in Boston.",
        "observed_company_stage": "Public",
        "stage_matches": True,
        "stage_evidence_url": SEC_URL,
        "stage_evidence_quote": SUBMITTED_STAGE_QUOTE,
        "attribute_satisfied": None,
        "required_attribute_evidence_url": "",
        "required_attribute_evidence_quote": "",
        "reason": "The submitted company dimensions were observed.",
    }


def _web_identity(structured_identity=None) -> dict[str, object]:
    receipt = lead_scorer._web_identity_receipt(
        _company(),
        _verdict(),
        verified_homepage_identity=_verified_homepage_identity(),
        verified_homepage_transport_domain="rapid7.com",
        verified_structured_identity=structured_identity,
        company_quality=True,
    )
    return receipt


def _industry_finding(*, role: str = "supplier_operator", url: str = PRODUCT_URL):
    return {
        "target": "industry",
        "status": "VERIFIED",
        "observed_value": "InsightCloudSec cloud security platform",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Cloud security",
        "observed_subindustry": "Cloud identity and access management",
        "activity_role": role,
        "evidence_url": url,
        "evidence_quote": PRODUCT_QUOTE,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "The product page describes a supplied cloud security platform.",
    }


def _unproven_finding() -> dict[str, object]:
    return {
        **_industry_finding(),
        "status": "UNPROVEN",
        "observed_value": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "",
        "evidence_quote": "",
        "reason": "The available evidence did not prove this target.",
    }


def _tool_response(finding: dict[str, object]) -> dict[str, object]:
    return {"choices": [{"message": {"tool_calls": [{
        "id": "call-submit",
        "type": "function",
        "function": {
            "name": "submit_findings",
            "arguments": json.dumps({"findings": [finding]}),
        },
    }]}}]}


def _search_response() -> dict[str, object]:
    return {"choices": [{"message": {"tool_calls": [{
        "id": "call-search",
        "type": "function",
        "function": {
            "name": "search_web",
            "arguments": json.dumps({
                "query": "Rapid7 cloud security InsightCloudSec"
            }),
        },
    }]}}]}


def _run_real_investigation(
    monkeypatch,
    *,
    structured_identity,
    finding,
    source_url: str = PRODUCT_URL,
    source_text: str = PRODUCT_PAGE,
):
    requests = []
    responses = [_tool_response(finding)]
    if finding["status"] == "VERIFIED":
        responses.extend([
            _tool_response(_unproven_finding()),
            _search_response(),
            _tool_response(_unproven_finding()),
        ])

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, responses.pop(0)

    search = AsyncMock(return_value={"results": []})
    fetch = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    verdict = _verdict()
    verdict["industry_evidence_url"] = source_url
    result = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(),
            icp=_icp(),
            verdict=verdict,
            investigation_targets=("industry",),
            icp_attribute="",
            icp_stage="Public",
            verified_identity=_verified_homepage_identity(),
            verified_transport_domain="rapid7.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            structured_profile_identity_evidence=structured_identity,
            review_positive_semantics=True,
            verified_homepage_pages={
                source_url: {"final_url": source_url, "text": source_text}
            },
        )
    )
    if len(requests) == 1:
        search.assert_not_awaited()
    else:
        search.assert_awaited_once()
    fetch.assert_not_awaited()
    return result, requests


def test_exact_rapid7_quote_passes_real_entry_only_after_full_alias_proof(
    monkeypatch,
):
    result, requests = _run_real_investigation(
        monkeypatch,
        structured_identity=_structured_identity(),
        finding=_industry_finding(),
    )

    claims = result[2]
    assert claims["industry"]["status"] == "VERIFIED"
    assert claims["industry"]["evidence_quote"] == PRODUCT_QUOTE
    assert len(requests) == 1
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["company_locator"]["linkedin"] == VANITY_LINKEDIN
    assert input_document["verified_homepage_identity"] == {
        "normalized_name": "rapid7",
        "registrable_dns_domain": "rapid7.com",
        "linkedin_company_slug": "rapid7",
    }


@pytest.mark.parametrize(
    "structured_identity",
    [
        pytest.param(None, id="absent-proof"),
        pytest.param(_structured_identity(company_id="99999"), id="wrong-id"),
        pytest.param(_structured_identity(name="Other Company"), id="wrong-company"),
        pytest.param(
            _structured_identity(website="https://other.example/"),
            id="wrong-domain",
        ),
    ],
)
def test_unnamed_quote_stays_unproven_without_exact_alias_chain(
    monkeypatch,
    structured_identity,
):
    result, requests = _run_real_investigation(
        monkeypatch,
        structured_identity=structured_identity,
        finding=_industry_finding(),
    )

    assert result[2]["industry"]["status"] == "UNPROVEN"
    assert result[1].decision == COMPANY_FIT_UNAVAILABLE
    assert len(requests) == 4
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["company_locator"]["linkedin"] == NUMERIC_LINKEDIN
    assert input_document["verified_homepage_identity"][
        "linkedin_company_slug"
    ] == "39624"


@pytest.mark.parametrize("role", ["customer_user", "third_party"])
def test_alias_proof_does_not_promote_customer_or_third_party_language(
    monkeypatch,
    role,
):
    result, requests = _run_real_investigation(
        monkeypatch,
        structured_identity=_structured_identity(),
        finding=_industry_finding(role=role),
    )

    assert result[2]["industry"]["status"] == "UNPROVEN"
    assert result[1].decision == COMPANY_FIT_UNAVAILABLE
    assert len(requests) == 4


def test_alias_proof_does_not_bind_a_wrong_source_domain(monkeypatch):
    other_url = "https://customer.example/rapid7-case-study"
    result, requests = _run_real_investigation(
        monkeypatch,
        structured_identity=_structured_identity(),
        finding=_industry_finding(url=other_url),
        source_url=other_url,
        source_text=PRODUCT_PAGE,
    )

    assert result[2]["industry"]["status"] == "UNPROVEN"
    assert result[1].decision == COMPANY_FIT_UNAVAILABLE
    assert len(requests) == 4


def test_numeric_alias_continuity_is_one_way_and_exact():
    verified_web = _web_identity(_structured_identity())
    identity, locator = lead_scorer._investigator_identity_context(
        NUMERIC_LINKEDIN,
        _verified_homepage_identity(),
        verified_web,
        "rapid7.com",
    )
    assert identity["linkedin_company_slug"] == "rapid7"
    assert locator == VANITY_LINKEDIN

    reverse_identity, reverse_locator = lead_scorer._investigator_identity_context(
        VANITY_LINKEDIN,
        {
            **_verified_homepage_identity(),
            "linkedin_company_slug": "rapid7",
        },
        {
            **verified_web,
            "observed_linkedin_slug": "39624",
        },
        "rapid7.com",
    )
    assert reverse_identity["linkedin_company_slug"] == "rapid7"
    assert reverse_locator == VANITY_LINKEDIN


def test_exact_observed_malformed_stage_quote_remains_rejected():
    finding = {
        **_industry_finding(),
        "target": "stage",
        "observed_value": "Public",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": SEC_URL,
        "evidence_quote": SUBMITTED_STAGE_QUOTE,
        "reason": "The filing lists Rapid7 common stock on Nasdaq.",
    }
    result = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={SEC_URL: SEC_PAGE},
        first_party_domains={"rapid7.com"},
        identity_names={"rapid7"},
        identity_anchor={
            "submitted_name": "Rapid7",
            "submitted_domain": "rapid7.com",
            "submitted_linkedin_slug": "rapid7",
            "observed_name": "Rapid7, Inc.",
            "observed_domain": "rapid7.com",
            "observed_linkedin_slug": "rapid7",
            "verified_name": "rapid7",
            "verified_domain": "rapid7.com",
            "verified_linkedin_slug": "rapid7",
        },
    )

    assert SUBMITTED_STAGE_QUOTE not in SEC_PAGE
    assert result["stage"]["status"] == "UNPROVEN"


def test_unstructured_numeric_alias_receipt_remains_unavailable():
    receipt = evaluate_company_identity(
        submitted_name="Rapid7",
        submitted_website="https://rapid7.com",
        submitted_linkedin=NUMERIC_LINKEDIN,
        observed_name="Rapid7 Inc",
        observed_website="https://rapid7.com",
        observed_linkedin=VANITY_LINKEDIN,
        evidence_source="company_web_reverification",
        company_quality=True,
    )
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert _web_identity(_structured_identity())["decision"] == COMPANY_FIT_MATCH
