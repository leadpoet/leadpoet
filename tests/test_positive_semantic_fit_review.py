from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
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


def _saved_higher_ed_case(
    *,
    name: str,
    domain: str,
    linkedin_slug: str,
    industry_quote: str,
    attribute_quote: str,
):
    website = f"https://{domain}/"
    linkedin = f"https://www.linkedin.com/company/{linkedin_slug}"
    company = _company().model_copy(update={
        "company_name": name,
        "company_website": website,
        "company_linkedin": linkedin,
        "employee_count": "201-500",
        "country": "United Kingdom",
        "state": "",
        "intent_signals": [],
    })
    verdict = _verdict(
        observed_company_name=name,
        observed_company_website=website,
        observed_company_linkedin=linkedin,
        observed_company_stage="Series B",
        stage_matches=True,
        stage_evidence_url=f"https://{domain}/funding",
        stage_evidence_quote=f"{name} announced its Series B funding round.",
        observed_employee_count="201-500",
        employee_size_matches=True,
        employee_size_evidence_url=linkedin,
        employee_size_evidence_quote="Company size 201-500 employees",
        observed_industry="Education",
        observed_subindustry="Higher education services",
        industry_matches=True,
        industry_activity_role="supplier_operator",
        industry_evidence_url=website,
        industry_evidence_quote=industry_quote,
        attribute_satisfied=True,
        required_attribute_evidence_url=website,
        required_attribute_evidence_quote=attribute_quote,
        observed_hq_country="United Kingdom",
        observed_hq_state="",
        geography_matches=True,
        geography_evidence_url=linkedin,
        geography_evidence_quote=(
            f"{name} is headquartered in London, United Kingdom."
        ),
    )
    identity = {
        "normalized_name": name.casefold().replace(" ", ""),
        "registrable_dns_domain": domain,
        "linkedin_company_slug": linkedin_slug,
    }
    icp = _icp().model_copy(update={
        "employee_count": "201-500",
        "company_stage": "Series B",
        "geography": "United Kingdom",
    })
    prior = lead_scorer._reverify_decision(
        verdict,
        icp.required_attribute,
        "series b",
        icp=icp,
        company=company,
        verified_homepage_identity=identity,
        verified_homepage_transport_domain=domain,
        company_quality=True,
    )
    assert prior.decision == COMPANY_FIT_MATCH
    return company, icp, verdict, identity, prior


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


def test_verified_homepage_navigation_sink_reaches_reverification(monkeypatch):
    locators = [{
        "url": "https://peer.example/platform/enrollment",
        "label": "Enrollment platform",
    }]
    homepage_page = {
        "https://peer.example": {
            "final_url": "https://peer.example/",
            "text": "PeerConnect enrollment platform for universities.",
        }
    }
    captured = {}

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def homepage(*_args, **kwargs):
        kwargs["homepage_navigation_locator_sink"].extend(locators)
        kwargs["homepage_evidence_sink"].update(homepage_page)
        return company_fit_match(details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "PeerConnect",
                "observed_domain": "peer.example",
                "observed_linkedin_slug": "peerconnect",
            },
            "verified_homepage_transport_domain": "peer.example",
        })

    async def web(*_args, **kwargs):
        captured.update(kwargs)
        return company_fit_unavailable("bounded test stop")

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)

    asyncio.run(lead_scorer._verify_company_fit(
        _company(),
        _icp(),
        0.0,
        1.0,
        set(),
        require_https_transport=True,
        company_quality=True,
        evidence_investigator=True,
    ))

    assert captured["verified_homepage_navigation_locators"] == locators
    assert captured["verified_homepage_pages"] == homepage_page


@pytest.mark.parametrize("bound_web_identity", [True, False])
def test_unbound_homepage_reaches_positive_review_only_after_web_identity(
    monkeypatch, bound_web_identity,
):
    homepage_url = "https://peer.example/"
    homepage_page = {
        homepage_url: {
            "final_url": homepage_url,
            "text": "PeerConnect sells an AI enrollment platform to universities.",
        }
    }
    prior = (
        lead_scorer._reverify_decision(
            _verdict(),
            _icp().required_attribute,
            "series a",
            icp=_icp(),
            company=_company(),
            verified_homepage_identity={},
            verified_homepage_transport_domain="peer.example",
            company_quality=True,
        )
        if bound_web_identity
        else company_fit_unavailable("web identity unresolved")
    )
    calls = []

    async def investigate(**kwargs):
        calls.append(kwargs)
        return {"claims": {
            "industry": _finding(
                "industry", status="UNPROVEN", activity_role="unresolved",
                evidence_url="", evidence_quote="",
            ),
        }, "usage": {"reasoning_turns": 1}}

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=_company(),
        icp=_icp(),
        verdict=_verdict(),
        investigation_targets=("industry",),
        icp_attribute=_icp().required_attribute,
        icp_stage="series a",
        verified_identity={},
        verified_transport_domain="peer.example",
        structured_employee_size_evidence=None,
        structured_public_company_evidence=None,
        employee_size_conflict=False,
        company_quality=True,
        prior_result=prior,
        review_positive_semantics=True,
        verified_homepage_pages=homepage_page,
    ))

    assert calls[0]["prefetched_pages"] == (
        homepage_page if bound_web_identity else {}
    )


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
            homepage_navigation_locators=[{
                "url": "https://peer.example/platform/enrollment",
                "label": "Enrollment platform",
            }],
        )
    )

    assert len(calls) == 1
    assert calls[0]["targets"] == ("stage", "industry")
    assert calls[0]["positive_semantic_review"] is True
    assert calls[0]["homepage_navigation_locators"] == [{
        "url": "https://peer.example/platform/enrollment",
        "label": "Enrollment platform",
    }]
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


def test_zen_adjacent_staffing_positive_review_fails_closed(monkeypatch):
    industry_quote = (
        "Find the perfect teachers, TAs and support staff for your school - "
        "with Zen Educate, the only tech-enabled supplier on the Crown "
        "Commercial Service framework."
    )
    company, icp, verdict, identity, prior = _saved_higher_ed_case(
        name="Zen Educate",
        domain="zeneducate.com",
        linkedin_slug="zen-educate",
        industry_quote=industry_quote,
        attribute_quote="Welcome to the UK's leading digital staffing platform for educators",
    )
    calls = []

    async def investigate(**kwargs):
        calls.append(kwargs)
        return {
            "claims": {"industry": _finding(
                "industry",
                status="UNPROVEN",
                observed_value=None,
                activity_role="supplier_operator",
                evidence_url="https://zeneducate.com/",
                evidence_quote=industry_quote,
                reason=(
                    "The source proves education staffing, not the requested "
                    "higher-education enrollment, learning, communication, or "
                    "campus-operations activity."
                ),
            )},
            "failure_reason": "",
            "usage": {"reasoning_turns": 2, "search_calls": 0, "fetch_calls": 1},
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    projected, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=company,
            icp=icp,
            verdict=verdict,
            investigation_targets=("industry",),
            icp_attribute=icp.required_attribute,
            icp_stage="series b",
            verified_identity=identity,
            verified_transport_domain="zeneducate.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            prior_result=prior,
            review_positive_semantics=True,
        )
    )

    assert len(calls) == 1
    assert calls[0]["targets"] == ("industry",)
    assert calls[0]["positive_semantic_review"] is True
    assert claims["industry"]["status"] == "UNPROVEN"
    assert projected["industry_matches"] is None
    assert projected["attribute_satisfied"] is None
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["industry"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["required_attribute_decision"] == (
        COMPANY_FIT_UNAVAILABLE
    )


def test_unibuddy_direct_higher_ed_evidence_projects_as_match(monkeypatch):
    industry_quote = (
        "Our platform boosts enrollment by building trust and confidence through "
        "scalable peer-to-peer & community engagement. It generates connection—and "
        "unique, real-time insights to help higher ed understand student behavior "
        "and optimize their strategy. See why higher ed institutions love Unibuddy"
    )
    company, icp, verdict, identity, prior = _saved_higher_ed_case(
        name="Unibuddy",
        domain="unibuddy.com",
        linkedin_slug="unibuddy",
        industry_quote=industry_quote,
        attribute_quote=(
            "Unibuddy allows prospective university applicants to chat with "
            "existing students and staff."
        ),
    )

    async def investigate(**kwargs):
        assert kwargs["targets"] == ("industry",)
        assert kwargs["positive_semantic_review"] is True
        return {
            "claims": {"industry": _finding(
                "industry",
                observed_value="Higher education enrollment platform",
                observed_industry="Education",
                observed_subindustry="Higher education services",
                activity_role="supplier_operator",
                evidence_url="https://unibuddy.com/",
                evidence_quote=industry_quote,
            )},
            "failure_reason": "",
            "usage": {"reasoning_turns": 3, "search_calls": 0, "fetch_calls": 1},
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    projected, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=company,
            icp=icp,
            verdict=verdict,
            investigation_targets=("industry",),
            icp_attribute=icp.required_attribute,
            icp_stage="series b",
            verified_identity=identity,
            verified_transport_domain="unibuddy.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            prior_result=prior,
            review_positive_semantics=True,
        )
    )

    assert claims["industry"]["status"] == "VERIFIED"
    assert projected["industry_evidence_quote"] == industry_quote
    assert projected["industry_matches"] is True
    assert projected["attribute_satisfied"] is True
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["industry"] == COMPANY_FIT_MATCH
    assert result.details["required_attribute_decision"] == COMPANY_FIT_MATCH


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
