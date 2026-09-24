from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_unavailable,
    evaluate_company_identity,
)


PROFILE_URL = "https://www.linkedin.com/company/medici-brands"


def _company() -> CompanyOutput:
    return CompanyOutput(
        company_name="Medici Brands Inc",
        company_website="https://medicibrands.com",
        company_linkedin="",
        industry="Food and Beverage Manufacturing",
        employee_count="51-200",
        country="United States",
        intent_signals=[
            {
                "description": "Medici Brands announced a retail launch.",
                "source": "news",
                "url": "https://medicibrands.com/news",
                "date": "2026-09-01",
                "snippet": "Medici Brands announced a retail launch.",
            }
        ],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="medici-identity-recovery",
        prompt="test",
        industry="Food and Beverage Manufacturing",
        sub_industry="",
        employee_count="51-200",
        company_stage="",
        geography="United States",
        country="United States",
        product_service="",
    )


def _unresolved_homepage_receipt() -> dict[str, str]:
    receipt = evaluate_company_identity(
        submitted_name="Medici Brands Inc",
        submitted_website="https://medicibrands.com",
        submitted_linkedin="",
        observed_name="Medici",
        observed_website="https://medicibrands.com",
        observed_linkedin=PROFILE_URL,
        evidence_source="company_homepage",
        company_quality=True,
    )
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    return receipt


def _structured_identity(**changes: str) -> dict[str, str]:
    evidence = {
        "name": "Medici Brands Inc",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": PROFILE_URL,
        "website": "https://medicibrands.com/",
    }
    evidence.update(changes)
    return evidence


def test_server_homepage_slug_can_only_be_a_lookup_anchor():
    receipt = _unresolved_homepage_receipt()

    assert not lead_scorer._alias_unresolved_structured_profile_lookup(
        receipt,
        "medicibrands.com",
    )
    lookup = lead_scorer._alias_unresolved_structured_profile_lookup(
        receipt,
        "medicibrands.com",
        server_verified_homepage_receipt=True,
    )
    assert lookup == {
        "normalized_name": "medici",
        "registrable_dns_domain": "medicibrands.com",
        "linkedin_company_slug": "medici-brands",
    }

    resolved = lead_scorer._structured_profile_alias_identity_receipt(
        _company(),
        receipt,
        _structured_identity(),
        "medicibrands.com",
        company_quality=True,
        server_verified_homepage_receipt=True,
    )
    assert resolved["decision"] == COMPANY_FIT_MATCH
    assert resolved["evidence_source"] == "company_homepage"
    assert resolved["structured_profile_identity"] == _structured_identity()
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize(
    "structured_identity",
    [
        _structured_identity(name="Medici Holdings"),
        _structured_identity(name="David Protein"),
        _structured_identity(website="https://other.example/"),
        _structured_identity(
            url="https://www.linkedin.com/company/medici-holdings"
        ),
        None,
    ],
    ids=[
        "wrong-name",
        "parent-or-subsidiary",
        "wrong-domain",
        "wrong-slug",
        "profile-unavailable",
    ],
)
def test_homepage_lookup_rejects_nonreciprocal_structured_profiles(
    structured_identity,
):
    assert not lead_scorer._structured_profile_alias_identity_receipt(
        _company(),
        _unresolved_homepage_receipt(),
        structured_identity,
        "medicibrands.com",
        company_quality=True,
        server_verified_homepage_receipt=True,
    )


def test_medici_recovery_keeps_profile_provenance_and_one_lookup(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"structured": 0, "current": 0}
    investigator_identity = {}

    async def provider(**_kwargs):
        return {
            "observed_company_name": "Medici Brands Inc",
            "observed_company_website": "https://medicibrands.com/about",
            "observed_company_linkedin": "",
            "observed_employee_count": None,
            "employee_size_matches": None,
            "employee_size_evidence_url": "",
            "employee_size_evidence_quote": "",
            "observed_industry": "",
            "observed_subindustry": "",
            "industry_matches": None,
            "industry_activity_role": "unresolved",
            "industry_evidence_url": "",
            "industry_evidence_quote": "",
            "observed_hq_country": "United States",
            "observed_hq_state": "New York",
            "geography_matches": True,
            "geography_evidence_url": "https://medicibrands.com/contact",
            "geography_evidence_quote": "Medici Brands is based in New York.",
            "reason": "The industry observation is incomplete.",
        }, ""

    async def structured_profile(
        domain,
        url,
        *,
        diagnostic,
        public_company_evidence,
        company_identity_evidence,
    ):
        del diagnostic, public_company_evidence
        calls["structured"] += 1
        assert (domain, url) == ("medicibrands.com", PROFILE_URL)
        company_identity_evidence.update(_structured_identity())
        return {
            "employee_count": "51-200",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": PROFILE_URL,
            "website": "https://medicibrands.com/",
        }

    async def current_profile(url, **_kwargs):
        calls["current"] += 1
        assert url == PROFILE_URL
        return {"outcome": "insufficient_evidence", "url": url}

    async def investigator(**kwargs):
        investigator_identity.update(kwargs["verified_identity"])
        return (
            dict(kwargs["verdict"]),
            kwargs["prior_result"],
            {},
            {},
            {},
        )

    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_profile,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        current_profile,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_run_targeted_company_evidence_investigation",
        investigator,
    )
    homepage_result = company_fit_unavailable(
        "homepage identity is incomplete",
        details={
            "identity": _unresolved_homepage_receipt(),
            "verified_homepage_transport_domain": "medicibrands.com",
        },
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=homepage_result,
            company_quality=True,
            evidence_investigator=True,
        )
    )

    assert calls["structured"] == 1
    assert calls["current"] == 2
    assert investigator_identity == {
        "normalized_name": "Medici Brands Inc",
        "registrable_dns_domain": "medicibrands.com",
        "linkedin_company_slug": "medici-brands",
    }
    assert homepage_result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity_receipt"]["evidence_source"] == (
        "company_web_reverification"
    )
    assert result.details["identity_receipt"]["observed_linkedin_slug"] == ""
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_MATCH
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "employee_count": "51-200",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": PROFILE_URL,
        "website": "https://medicibrands.com/",
    }


def test_wrong_structured_identity_cannot_award_headcount(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"provider": 0, "structured": 0}

    async def provider(**_kwargs):
        calls["provider"] += 1
        return {
            "observed_company_name": "Medici Brands Inc",
            "observed_company_website": "https://medicibrands.com/about",
            "observed_company_linkedin": "",
            "observed_employee_count": None,
            "employee_size_matches": None,
            "employee_size_evidence_url": "",
            "employee_size_evidence_quote": "",
            "observed_industry": "Food and Beverage Manufacturing",
            "observed_subindustry": "",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://medicibrands.com/about",
            "industry_evidence_quote": "Medici Brands develops food brands.",
            "observed_hq_country": "United States",
            "observed_hq_state": "New York",
            "geography_matches": True,
            "geography_evidence_url": "https://medicibrands.com/contact",
            "geography_evidence_quote": "Medici Brands is based in New York.",
            "reason": "The profile identity is unresolved.",
        }, ""

    async def wrong_profile(
        _domain,
        _url,
        *,
        diagnostic,
        public_company_evidence,
        company_identity_evidence,
    ):
        del diagnostic, public_company_evidence
        calls["structured"] += 1
        company_identity_evidence.update(
            _structured_identity(name="David Protein")
        )
        return {
            "employee_count": "51-200",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": PROFILE_URL,
            "website": "https://medicibrands.com/",
        }

    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        wrong_profile,
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=company_fit_unavailable(
                "homepage identity is incomplete",
                details={
                    "identity": _unresolved_homepage_receipt(),
                    "verified_homepage_transport_domain": "medicibrands.com",
                },
            ),
            company_quality=True,
        )
    )

    assert calls == {"provider": 2, "structured": 1}
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "",
        "quote": "",
    }
