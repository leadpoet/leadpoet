from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
)
from tests.test_company_evidence_investigator import _complete_verdict, _finding


@pytest.mark.parametrize(
    ("investigator_status", "expected_decision"),
    [
        ("UNPROVEN", COMPANY_FIT_UNAVAILABLE),
        ("VERIFIED", COMPANY_FIT_MATCH),
    ],
)
def test_micruity_industry_review_controls_schema_repair(
    monkeypatch,
    investigator_status,
    expected_decision,
):
    """Reproduce the frozen Sep. 27 Micruity industry-only receipt."""

    company = CompanyOutput(
        company_name="Micruity",
        company_website="https://micruity.com/",
        company_linkedin="",
        industry="Financial Services / Retirement income technology",
        employee_count="51-200",
        country="Canada",
        state="Ontario",
        intent_signals=[{
            "description": "Micruity raised a $20 million Series A.",
            "source": "company_website",
            "url": "https://micruity.com/20m-to-power-retirement-paychecks/",
            "date": "2025-12-03",
            "snippet": "Micruity has raised a $20 million Series A.",
        }],
    )
    icp = ICPPrompt(
        icp_id="icp_20260926_006",
        prompt="Lenders or investment platforms with fresh funding.",
        industry="Lending and Investments",
        sub_industry="Alternative lending and private credit platforms",
        employee_count="11-50|51-200|201-500|501-1,000",
        company_stage="Series A",
        geography="Canada",
        product_service=(
            "A lending or investment platform that helps businesses or consumers "
            "originate loans, manage capital, or access private credit products"
        ),
        required_attribute=(
            "Runs a lending or investment platform that originates, services, or "
            "facilitates credit or investment products for customers"
        ),
    )
    initial = _complete_verdict(
        observed_company_name="Micruity",
        observed_company_website="https://micruity.com",
        observed_company_linkedin="https://www.linkedin.com/company/micruity",
        observed_employee_count="11-50",
        employee_size_matches=True,
        observed_industry="Financial technology infrastructure",
        observed_subindustry="retirement income solutions",
        industry_matches=None,
        industry_activity_role="supplier_operator",
        industry_evidence_url="https://micruity.com/",
        industry_evidence_quote=(
            "END-TO-END Infrastructure powering institutional lifetime income products"
        ),
        observed_hq_country="Canada",
        observed_hq_state="Ontario",
        geography_matches=True,
        attribute_satisfied=None,
        required_attribute_evidence_url="",
        required_attribute_evidence_quote="",
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url=(
            "https://micruity.com/20m-to-power-retirement-paychecks/"
        ),
        stage_evidence_quote="Micruity has raised a $20 million Series A",
    )
    repaired = {
        **initial,
        "observed_industry": "Lending and Investments",
        "observed_subindustry": "Alternative lending and private credit platforms",
        "industry_matches": True,
        "industry_evidence_url": "https://micruity.com/platform",
        "industry_evidence_quote": (
            "Micruity originates and services private credit products for customers."
        ),
        "attribute_satisfied": True,
        "required_attribute_evidence_url": "https://micruity.com/platform",
        "required_attribute_evidence_quote": (
            "Micruity originates and services private credit products for customers."
        ),
    }
    calls = {"provider": 0, "investigator": 0}

    async def provider(**_kwargs):
        calls["provider"] += 1
        if calls["provider"] == 1:
            return initial, ""
        if investigator_status == "VERIFIED" and calls["provider"] == 2:
            return repaired, ""
        raise AssertionError("schema repair followed an unresolved industry result")

    async def bounded_investigation(**kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("industry",)
        assert kwargs["positive_semantic_review"] is True
        if investigator_status == "UNPROVEN":
            finding = _finding(
                "industry",
                status="UNPROVEN",
                observed_value="",
                observed_industry="",
                observed_subindustry="",
                activity_role="unresolved",
                evidence_url="",
                evidence_quote="",
                reason=(
                    "The source does not establish alternative lending or private "
                    "credit activity."
                ),
            )
        else:
            finding = _finding(
                "industry",
                status="VERIFIED",
                observed_value="Alternative lending and private credit platform",
                observed_industry="Lending and Investments",
                observed_subindustry=(
                    "Alternative lending and private credit platforms"
                ),
                activity_role="supplier_operator",
                evidence_url="https://micruity.com/platform",
                evidence_quote=(
                    "Micruity originates and services private credit products for "
                    "customers."
                ),
                reason="Exact requested industry and activity are supported.",
            )
        return {
            "claims": {"industry": finding},
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 0, "fetch_calls": 1},
        }

    async def keep_observation(verdict, *_args, **_kwargs):
        return verdict

    async def keep_attribute(verdict, **_kwargs):
        return verdict, {}

    homepage_identity = lead_scorer.company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "micruity",
                "observed_domain": "micruity.com",
                "observed_linkedin_slug": "micruity",
            },
            "verified_homepage_transport_domain": "micruity.com",
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "investigate_company_evidence",
        bounded_investigation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_ground_required_attribute_evidence",
        keep_attribute,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            company,
            icp,
            require_company_fit_dimensions=True,
            verified_homepage_identity=homepage_identity,
            company_quality=True,
                evidence_investigator=True,
        )
    )

    assert calls == {
        "provider": 2 if investigator_status == "VERIFIED" else 1,
        "investigator": 1,
    }
    assert result.decision == expected_decision
    assert result.details["dimension_decisions"]["industry"] == expected_decision
    receipt = result.details["investigation_receipt"]
    assert receipt["positive_semantic_review"] is True
    assert receipt["positive_semantic_review_resolved"] is (
        investigator_status == "VERIFIED"
    )
