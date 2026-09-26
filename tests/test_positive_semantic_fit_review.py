from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
)


def _company() -> CompanyOutput:
    return CompanyOutput(
        company_name="PeerConnect",
        company_website="https://peer.example",
        company_linkedin="https://www.linkedin.com/company/peerconnect",
        industry="Education",
        employee_count="11-50",
        country="United States",
        state="California",
        intent_signals=[{
            "description": "PeerConnect announced a completed funding round.",
            "source": "news",
            "url": "https://news.example/peerconnect",
            "date": "2026-09-01",
            "snippet": "PeerConnect announced a completed funding round.",
        }],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="higher-ed",
        prompt="Higher education enrollment platforms",
        industry="Education",
        sub_industry="Higher education services",
        employee_count="11-50",
        company_stage="Series A",
        geography="United States",
        product_service="Platform for student enrollment and campus operations",
        required_attribute=(
            "Platform used for enrollment, learning delivery, student "
            "communication, or campus operations"
        ),
    )


def _verdict(**overrides):
    values = {
        "observed_company_name": "PeerConnect",
        "observed_company_website": "https://peer.example",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/peerconnect"
        ),
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "observed_employee_count": "11-50",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://peer.example/about",
        "employee_size_evidence_quote": "PeerConnect has 11-50 employees.",
        "observed_industry": "Education",
        "observed_subindustry": "Education administration programs",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://peer.example/platform",
        "industry_evidence_quote": (
            "PeerConnect supplies a student enrollment platform to universities."
        ),
        "attribute_satisfied": True,
        "required_attribute_evidence_url": "https://peer.example/enrollment",
        "required_attribute_evidence_quote": (
            "Universities use PeerConnect for enrollment and student communication."
        ),
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": "https://peer.example/about",
        "geography_evidence_quote": (
            "PeerConnect is headquartered in California, United States."
        ),
        "reason": "verified",
    }
    values.update(overrides)
    return values


def _identity():
    return {
        "normalized_name": "peerconnect",
        "registrable_dns_domain": "peer.example",
        "linkedin_company_slug": "peerconnect",
    }


def _initial_result(verdict=None):
    icp = _icp()
    return lead_scorer._reverify_decision(
        verdict or _verdict(),
        icp.required_attribute,
        "series a",
        icp=icp,
        company=_company(),
        verified_homepage_identity=_identity(),
        verified_homepage_transport_domain="peer.example",
        company_quality=True,
    )


def _finding(target: str, **overrides):
    values = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Series A",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "https://peer.example/about",
        "evidence_quote": "PeerConnect announced its Series A funding round.",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "verified",
    }
    values.update(overrides)
    return values


def test_positive_review_is_added_when_recoverable_stage_is_unavailable():
    icp = _icp()
    result = _initial_result()
    targets = lead_scorer._targeted_company_investigation_dimensions(
        result,
        icp_stage="series a",
        employee_size_conflict=False,
        company=_company(),
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert targets == ("stage",)
    assert lead_scorer._positive_semantic_review_needed(result, icp, targets)

    outside_region = _initial_result(_verdict(
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url="https://peer.example/funding",
        stage_evidence_quote="PeerConnect announced its Series A.",
        observed_hq_country="Canada",
        observed_hq_state="Ontario",
        geography_matches=False,
        geography_evidence_quote="PeerConnect is headquartered in Ontario, Canada.",
    ))
    outside_targets = lead_scorer._targeted_company_investigation_dimensions(
        outside_region,
        icp_stage="series a",
        employee_size_conflict=False,
        company=_company(),
    )
    assert outside_region.decision == COMPANY_FIT_MISMATCH
    assert outside_targets == ()
    assert not lead_scorer._positive_semantic_review_needed(
        outside_region, icp, outside_targets
    )


def test_positive_review_reopens_industry_during_attribute_source_recovery():
    icp = _icp()
    result = _initial_result(_verdict(
        attribute_satisfied=None,
        observed_company_stage="Series A",
        stage_matches=True,
        stage_evidence_url="https://peer.example/funding",
        stage_evidence_quote="PeerConnect announced its Series A.",
    ))
    targets = lead_scorer._targeted_company_investigation_dimensions(
        result,
        icp_stage="series a",
        employee_size_conflict=False,
        company=_company(),
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["required_attribute_decision"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert targets == ()
    assert lead_scorer._positive_semantic_review_needed(result, icp, targets)


@pytest.mark.parametrize(
    ("industry_finding", "expected_decision", "semantic_resolved"),
    [
        (
            _finding(
                "industry",
                observed_value="Higher education enrollment platform",
                observed_industry="Software Development",
                observed_subindustry="Enterprise software",
                activity_role="supplier_operator",
                evidence_url="https://peer.example/platform",
                evidence_quote=(
                    "Universities use PeerConnect to enroll and communicate "
                    "with students."
                ),
            ),
            COMPANY_FIT_MATCH,
            True,
        ),
        (
            _finding(
                "industry",
                status="UNPROVEN",
                observed_value=None,
                activity_role="unresolved",
                evidence_url="",
                evidence_quote="",
                reason="The evidence does not establish the requested activity.",
            ),
            COMPANY_FIT_UNAVAILABLE,
            False,
        ),
        (
            _finding(
                "industry",
                status="CONTRADICTED",
                observed_value="Internal enrollment system user",
                observed_industry="Higher education",
                observed_subindustry="University",
                activity_role="customer_user",
                evidence_url="https://peer.example/platform",
                evidence_quote=(
                    "PeerConnect uses an enrollment platform for its own students."
                ),
                reason="The company is a customer, not the requested supplier.",
            ),
            COMPANY_FIT_MISMATCH,
            True,
        ),
    ],
)
def test_stage_recovery_and_positive_semantics_share_one_investigation(
    monkeypatch,
    industry_finding,
    expected_decision,
    semantic_resolved,
):
    calls = []

    async def investigate(**kwargs):
        calls.append(kwargs)
        return {
            "claims": {
                "stage": _finding("stage"),
                "industry": industry_finding,
            },
            "_validated_stage_finding": _finding("stage"),
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 1, "fetch_calls": 2},
        }

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", investigate
    )
    prior = _initial_result()
    original_attribute_quote = _verdict()["required_attribute_evidence_quote"]

    projected, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(),
            icp=_icp(),
            verdict=_verdict(),
            investigation_targets=("stage", "industry"),
            icp_attribute=_icp().required_attribute,
            icp_stage="series a",
            verified_identity=_identity(),
            verified_transport_domain="peer.example",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            prior_result=prior,
            review_positive_semantics=True,
        )
    )

    assert len(calls) == 1
    assert calls[0]["targets"] == ("stage", "industry")
    assert claims["industry"] == industry_finding
    assert result.decision == expected_decision
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH
    assert result.details["investigation_receipt"][
        "positive_semantic_review_resolved"
    ] is semantic_resolved
    assert projected["required_attribute_evidence_quote"] == original_attribute_quote
    if semantic_resolved:
        assert projected["attribute_satisfied"] is True
    else:
        assert projected["attribute_satisfied"] is None
        assert result.details["required_attribute_decision"] == (
            COMPANY_FIT_UNAVAILABLE
        )


def test_positive_review_provider_failure_is_fail_closed(monkeypatch):
    async def unavailable(**kwargs):
        kwargs["diagnostic"][lead_scorer.VERIFIER_FAILURE_REASON_KEY] = (
            lead_scorer.PROVIDER_ERROR_FAILURE_REASON
        )
        return {"claims": {}, "failure_reason": "provider HTTP 503"}

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", unavailable
    )
    _, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(),
            icp=_icp(),
            verdict=_verdict(),
            investigation_targets=("industry",),
            icp_attribute=_icp().required_attribute,
            icp_stage="series a",
            verified_identity=_identity(),
            verified_transport_domain="peer.example",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            prior_result=_initial_result(),
            review_positive_semantics=True,
        )
    )

    assert claims == {}
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details[lead_scorer.VERIFIER_FAILURE_DETAIL_KEY] == (
        lead_scorer.PROVIDER_ERROR_FAILURE_REASON
    )
