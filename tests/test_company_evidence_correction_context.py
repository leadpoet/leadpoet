from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena import operations as arena_operations
from qualification.scoring import company_evidence_investigator as investigator


RAPID7_PRODUCT_URL = "https://rapid7.com/products/insightcloudsec/"
RAPID7_INDUSTRY_QUOTE = (
    "InsightCloudSec: Cloud-Native Application Protection (CNAPP) - Rapid7 "
    "Cloud Security Packages INSIGHTCLOUDSEC Cloud-Native Application "
    "Protection InsightCloudSec empowers teams to proactively manage risk, "
    "accelerate DevSecOps, and enforce compliance across multi-cloud environments."
)
RAPID7_FUNCTION_SPAN = (
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
NASDAQ_URL = (
    "https://www.nasdaq.com/press-release/rapid7-appoints-wael-mohamed-"
    "chief-executive-officer-corey-thomas-become-executive"
)
NASDAQ_QUOTE = (
    "Rapid7, Inc. (NASDAQ: RPD), a global leader in AI-powered managed "
    "cybersecurity operations, today announced a leadership transition"
)
CUSTOMER_QUOTE = (
    "Rapid7 uses Example Payroll to administer employee payroll and benefits."
)


def _finding(target: str, **overrides):
    finding = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Public",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": NASDAQ_URL,
        "evidence_quote": NASDAQ_QUOTE,
        "old_name": "Rapid7",
        "new_name": "Rapid7",
        "old_domain": "rapid7.com",
        "new_domain": "rapid7.com",
        "shared_linkedin_slug": "rapid7",
        "reason": "source supports the submitted finding",
    }
    finding.update(overrides)
    return finding


def _industry_finding(**overrides):
    values = {
        "observed_value": "Cloud security software",
        "observed_industry": "Security software",
        "observed_subindustry": "Cloud security and identity protection",
        "activity_role": "supplier_operator",
        "evidence_url": RAPID7_PRODUCT_URL,
        "evidence_quote": RAPID7_INDUSTRY_QUOTE,
    }
    values.update(overrides)
    return _finding("industry", **values)


def test_non_rejected_context_is_exact_source_bound_and_bounded():
    long_prefix = "P" * 5_000
    long_suffix = "S" * 5_000
    fetched_text = f"{long_prefix} {RAPID7_INDUSTRY_QUOTE} {long_suffix}"
    finding = _industry_finding(
        observed_value="V" * 5_000,
        observed_industry="I" * 5_000,
        observed_subindustry="S" * 5_000,
    )

    context = investigator._untrusted_non_rejected_source_context(
        targets=("stage", "industry", "industry"),
        findings={"industry": finding},
        rejected_findings=({"target": "stage", "reason": "bad quote"},),
        fetched_pages={RAPID7_PRODUCT_URL: fetched_text},
    )

    assert len(context) == 1
    item = context[0]
    assert item["target"] == "industry"
    assert item["evidence_quote"] == RAPID7_INDUSTRY_QUOTE
    assert RAPID7_INDUSTRY_QUOTE in item["source_context"]
    assert len(item["source_context"]) <= (
        investigator.REJECTED_QUOTE_CONTEXT_BEFORE_CHARACTERS
        + len(RAPID7_INDUSTRY_QUOTE)
        + investigator.REJECTED_QUOTE_CONTEXT_AFTER_CHARACTERS
    )
    assert len(item["observed_value"]) == 300
    assert len(item["observed_industry"]) == 200
    assert len(item["observed_subindustry"]) == 300


@pytest.mark.parametrize(
    "control", ["tampered_quote", "rejected_target", "semantic_followup"],
)
def test_non_rejected_context_excludes_unbound_or_rejected_findings(control):
    finding = _industry_finding()
    rejected = ()
    excluded = ()
    if control == "tampered_quote":
        finding["evidence_quote"] = "Rapid7 invented text that is absent."
    elif control == "rejected_target":
        rejected = ({"target": "industry", "reason": "bad quote"},)
    else:
        excluded = ("industry",)

    assert investigator._untrusted_non_rejected_source_context(
        targets=("industry",),
        findings={"industry": finding},
        rejected_findings=rejected,
        fetched_pages={RAPID7_PRODUCT_URL: RAPID7_INDUSTRY_QUOTE},
        excluded_targets=excluded,
    ) == []


def test_industry_semantic_followup_excludes_its_prior_contradiction(monkeypatch):
    geography_url = "https://rapid7.com/company"
    geography_quote = (
        "Rapid7 is headquartered in Boston, Massachusetts, United States."
    )
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "submit_findings", {"findings": [
                _industry_finding(
                    status="CONTRADICTED",
                    observed_value="Payroll customer",
                    observed_industry="Payroll software",
                    observed_subindustry="Payroll administration",
                    activity_role="customer_user",
                    evidence_quote=CUSTOMER_QUOTE,
                ),
                _finding(
                    "geography",
                    observed_value="Boston, Massachusetts, United States",
                    observed_country="United States",
                    observed_state="Massachusetts",
                    evidence_url=geography_url,
                    evidence_quote=geography_quote,
                ),
            ]}
        elif turn == 2:
            name, arguments = "search_web", {
                "query": "Rapid7 security software identity protection"
            }
        else:
            name, arguments = "submit_findings", {"findings": [
                _industry_finding(
                    status="UNPROVEN",
                    activity_role="unresolved",
                    evidence_url="",
                    evidence_quote="",
                ),
                _finding(
                    "geography",
                    observed_value="Boston, Massachusetts, United States",
                    observed_country="United States",
                    observed_state="Massachusetts",
                    evidence_url=geography_url,
                    evidence_quote=geography_quote,
                ),
            ]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": []}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Rapid7", "website": "https://rapid7.com"},
        targets=("industry", "geography"),
        requested_industry="Security software",
        requested_subindustry="Identity protection",
        requested_geography="Massachusetts, United States",
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "submitted_source_urls": [RAPID7_PRODUCT_URL, geography_url],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
        },
        prefetched_pages={
            RAPID7_PRODUCT_URL: {
                "final_url": RAPID7_PRODUCT_URL,
                "text": CUSTOMER_QUOTE,
            },
            geography_url: {
                "final_url": geography_url,
                "text": geography_quote,
            },
        },
    ))

    feedback = json.loads(requests[1]["messages"][-1]["content"])
    retained_targets = {
        item["target"]
        for item in feedback["untrusted_non_rejected_source_context"]
    }
    assert retained_targets == {"geography"}
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_web"},
    }
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["geography"]["status"] == "VERIFIED"


@pytest.mark.parametrize(
    ("revised_status", "expected_status"),
    [
        ("VERIFIED", "VERIFIED"),
        ("UNPROVEN", "UNPROVEN"),
        ("CONTRADICTED", "CONTRADICTED"),
    ],
)
def test_stage_quote_correction_keeps_untrusted_industry_source_context(
    monkeypatch, revised_status, expected_status,
):
    requests = []
    product_page = " ".join((
        RAPID7_INDUSTRY_QUOTE,
        RAPID7_FUNCTION_SPAN,
        CUSTOMER_QUOTE,
    ))

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        if len(requests) == 1:
            findings = [
                _finding(
                    "stage",
                    evidence_quote=(
                        "Rapid7, Inc. common stock is currently listed on Nasdaq."
                    ),
                ),
                _industry_finding(),
            ]
        else:
            if revised_status == "VERIFIED":
                revised_industry = _industry_finding()
            elif revised_status == "UNPROVEN":
                revised_industry = _industry_finding(
                    status="UNPROVEN",
                    activity_role="unresolved",
                    evidence_url="",
                    evidence_quote="",
                    reason=(
                        "The cited functions were reviewed, but the full requested "
                        "criterion remains unproven."
                    ),
                )
            else:
                revised_industry = _industry_finding(
                    status="CONTRADICTED",
                    observed_value="Payroll customer",
                    observed_industry="Payroll software",
                    observed_subindustry="Payroll administration",
                    activity_role="customer_user",
                    evidence_quote=CUSTOMER_QUOTE,
                    reason=(
                        "The exact source identifies Rapid7 as a customer of the "
                        "cited service."
                    ),
                )
            findings = [_finding("stage"), revised_industry]
        arguments = {"findings": findings}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps(arguments),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": []}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Rapid7",
            "website": "https://rapid7.com",
            "linkedin": "https://www.linkedin.com/company/rapid7",
        },
        targets=("stage", "industry"),
        requested_stage="Public",
        requested_industry="Security software and services",
        requested_subindustry="Cloud security and identity protection",
        requested_product_service="Security software and services",
        requested_attribute=(
            "Protects accounts, infrastructure, endpoints, or data"
        ),
        prior_observations={
            "observed_company_name": "Rapid7",
            "observed_company_website": "https://rapid7.com",
            "submitted_source_urls": [NASDAQ_URL, RAPID7_PRODUCT_URL],
        },
        verified_homepage_identity={
            "normalized_name": "Rapid7",
            "registrable_dns_domain": "rapid7.com",
            "linkedin_slug": "rapid7",
        },
        prefetched_pages={
            NASDAQ_URL: {"final_url": NASDAQ_URL, "text": NASDAQ_QUOTE},
            RAPID7_PRODUCT_URL: {
                "final_url": RAPID7_PRODUCT_URL,
                "text": product_page,
            },
        },
    ))

    assert len(requests) == 2
    correction = json.loads(requests[1]["messages"][-1]["content"])
    assert [item["target"] for item in correction["rejected_findings"]] == [
        "stage"
    ]
    retained = correction["untrusted_non_rejected_source_context"]
    assert len(retained) == 1
    assert retained[0]["target"] == "industry"
    assert retained[0]["prior_status"] == "VERIFIED"
    assert retained[0]["evidence_quote"] == RAPID7_INDUSTRY_QUOTE
    assert "Cloud infrastructure entitlement management (CIEM)" in (
        retained[0]["source_context"]
    )
    assert "prior statuses and observations are not authoritative" in (
        correction["instruction"]
    )
    assert "does not force acceptance" in correction["instruction"]
    assert (
        "reuse its supplied exact evidence_url and evidence_quote instead of "
        "composing a replacement span"
    ) in correction["instruction"]
    assert "Otherwise revise the finding or return UNPROVEN" in (
        correction["instruction"]
    )
    assert "perform every requested capability" in correction["instruction"]
    assert "Preserve every AND/OR condition, qualifier, and exclusion" in (
        correction["instruction"]
    )
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == expected_status
    assert result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
