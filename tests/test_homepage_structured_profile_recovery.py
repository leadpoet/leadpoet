from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
    evaluate_company_identity,
)


PROFILE_URL = "https://www.linkedin.com/company/medici-brands"
RAPID7_NUMERIC_URL = "https://www.linkedin.com/company/39624"
RAPID7_VANITY_URL = "https://www.linkedin.com/company/rapid7"


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


def _rapid7_company() -> CompanyOutput:
    return _company().model_copy(update={
        "company_name": "Rapid7",
        "company_website": "https://rapid7.com",
        "company_linkedin": RAPID7_NUMERIC_URL,
        "employee_count": "10,001+",
        "state": "Massachusetts",
    })


def _rapid7_web_identity() -> dict[str, str]:
    receipt = evaluate_company_identity(
        submitted_name="Rapid7",
        submitted_website="https://rapid7.com",
        submitted_linkedin=RAPID7_NUMERIC_URL,
        observed_name="Rapid7 Inc",
        observed_website="https://rapid7.com",
        observed_linkedin=RAPID7_VANITY_URL,
        evidence_source="company_web_reverification",
        company_quality=True,
    )
    assert receipt["reason_code"] == "identity_linkedin_alias_unresolved"
    return receipt


def _rapid7_homepage_identity():
    receipt = evaluate_company_identity(
        submitted_name="Rapid7",
        submitted_website="https://rapid7.com",
        submitted_linkedin=RAPID7_NUMERIC_URL,
        observed_name="Rapid7",
        observed_website="https://www.rapid7.com",
        observed_linkedin=RAPID7_NUMERIC_URL,
        evidence_source="company_homepage",
        company_quality=True,
    )
    assert receipt["decision"] == COMPANY_FIT_MATCH
    return company_fit_match(
        "verified first-party homepage identity",
        details={
            "identity": receipt,
            "verified_homepage_transport_domain": "rapid7.com",
        },
    )


def _rapid7_structured_identity(**changes: str) -> dict[str, str]:
    evidence = {
        "name": "Rapid7, Inc.",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": RAPID7_VANITY_URL,
        "website": "https://rapid7.com/",
        "company_id": "39624",
        "requested_url": RAPID7_NUMERIC_URL,
    }
    evidence.update(changes)
    return evidence


def _rapid7_redirected_structured_identity(**changes: str) -> dict[str, str]:
    evidence = _rapid7_structured_identity()
    evidence.update({
        "provider_website_url": "https://r-7.co/3i5nlhP",
        "provider_website_requested_url": "https://r-7.co/3i5nlhP",
        "provider_website_final_url": "https://www.rapid7.com/",
    })
    evidence.update(changes)
    return evidence


def test_numeric_linkedin_alias_requires_structured_id_and_vanity_binding():
    lookup = lead_scorer._alias_unresolved_structured_profile_lookup(
        _rapid7_web_identity(),
        "rapid7.com",
    )
    assert lookup == {
        "normalized_name": "rapid7",
        "registrable_dns_domain": "rapid7.com",
        "linkedin_company_slug": "rapid7",
        "requested_profile_url": RAPID7_NUMERIC_URL,
        "observed_profile_url": RAPID7_VANITY_URL,
    }
    resolved = lead_scorer._structured_profile_alias_identity_receipt(
        _rapid7_company(),
        _rapid7_web_identity(),
        _rapid7_structured_identity(),
        "rapid7.com",
        company_quality=True,
    )
    assert resolved["decision"] == COMPANY_FIT_MATCH
    assert resolved["reason_code"] == (
        "structured_numeric_linkedin_alias_verified"
    )


def test_numeric_homepage_anchor_defers_to_structured_alias_recovery():
    homepage = lead_scorer._verified_homepage_identity_anchor(
        _rapid7_homepage_identity()
    )
    web = _rapid7_web_identity()

    assert homepage["linkedin_company_slug"] == "39624"
    assert lead_scorer._structured_profile_identity_anchor(
        homepage,
        web,
        "rapid7.com",
    ) == {}
    assert lead_scorer._alias_unresolved_structured_profile_lookup(
        web,
        "rapid7.com",
    )["requested_profile_url"] == RAPID7_NUMERIC_URL


def _rapid7_public_stage_case():
    homepage = lead_scorer._verified_homepage_identity_anchor(
        _rapid7_homepage_identity()
    )
    company = _rapid7_company().model_copy(update={
        "employee_count": "1,001-5,000",
        "company_stage": "Public",
        "industry": "Cybersecurity",
    })
    icp = _icp().model_copy(update={
        "industry": "Cybersecurity",
        "employee_count": "1,001-5,000",
        "company_stage": "Public",
        "product_service": "Cybersecurity risk and detection software.",
    })
    verdict = {
        "observed_company_name": "Rapid7, Inc.",
        "observed_company_website": "https://www.rapid7.com",
        "observed_company_linkedin": RAPID7_VANITY_URL,
        "observed_employee_count": "1,001-5,000",
        "employee_size_matches": True,
        "employee_size_evidence_url": RAPID7_VANITY_URL,
        "employee_size_evidence_quote": (
            "Company size 1,001-5,000 employees"
        ),
        "observed_industry": "Cybersecurity",
        "observed_subindustry": "",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://rapid7.com/company/about",
        "industry_evidence_quote": (
            "Rapid7 provides cybersecurity risk and detection software."
        ),
        "observed_hq_country": "United States",
        "observed_hq_state": "Massachusetts",
        "geography_matches": True,
        "geography_evidence_url": "https://rapid7.com/company/about",
        "geography_evidence_quote": "Rapid7 is based in Boston.",
        "observed_company_stage": "Public",
        "stage_matches": True,
        "stage_evidence_url": "https://rapid7.com/investors",
        "stage_evidence_quote": (
            "Rapid7 common stock is listed on Nasdaq under ticker RPD."
        ),
        "attribute_satisfied": None,
        "required_attribute_evidence_url": "",
        "required_attribute_evidence_quote": "",
        "reason": "All requested dimensions match.",
    }
    structured_identity = _rapid7_structured_identity()
    return homepage, company, icp, verdict, structured_identity


def test_numeric_homepage_structured_alias_binds_private_stage_conflict():
    homepage, company, icp, verdict, structured_identity = (
        _rapid7_public_stage_case()
    )
    identity = lead_scorer._web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=homepage,
        verified_homepage_transport_domain="rapid7.com",
        verified_structured_identity=structured_identity,
        company_quality=True,
    )
    private_evidence = {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": RAPID7_VANITY_URL,
        "website": "https://rapid7.com/",
    }

    assert identity["decision"] == COMPANY_FIT_MATCH
    assert identity["reason_code"] == (
        "structured_numeric_linkedin_alias_verified"
    )
    assert lead_scorer._structured_profile_identity_anchor(
        homepage,
        identity,
        "rapid7.com",
    ) == {
        "normalized_name": "rapid7",
        "registrable_dns_domain": "rapid7.com",
        "linkedin_company_slug": "rapid7",
    }

    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=icp,
        company=company,
        verified_homepage_identity=homepage,
        verified_homepage_transport_domain="rapid7.com",
        structured_public_company_evidence=private_evidence,
        structured_profile_identity_evidence=structured_identity,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_evidence"]["stage"] == private_evidence


def test_numeric_homepage_structured_public_type_keeps_public_stage_match():
    homepage, company, icp, verdict, structured_identity = (
        _rapid7_public_stage_case()
    )
    public_evidence = {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": RAPID7_VANITY_URL,
        "website": "https://rapid7.com/",
    }

    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=icp,
        company=company,
        verified_homepage_identity=homepage,
        verified_homepage_transport_domain="rapid7.com",
        structured_public_company_evidence=public_evidence,
        structured_profile_identity_evidence=structured_identity,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH


def test_numeric_homepage_unrelated_private_type_cannot_change_public_stage():
    homepage, company, icp, verdict, structured_identity = (
        _rapid7_public_stage_case()
    )
    unrelated_private_evidence = {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/other-company",
        "website": "https://rapid7.com/",
    }

    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=icp,
        company=company,
        verified_homepage_identity=homepage,
        verified_homepage_transport_domain="rapid7.com",
        structured_public_company_evidence=unrelated_private_evidence,
        structured_profile_identity_evidence=structured_identity,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": COMPANY_FIT_UNAVAILABLE},
        {"reason_code": "verifier_accepted"},
        {"evidence_source": "company_homepage"},
        {"submitted_domain": "other.example"},
        {"observed_domain": "other.example"},
        {"submitted_name": "other company"},
        {"observed_name": "other company"},
        {"submitted_linkedin_slug": "393624"},
        {"observed_linkedin_slug": "393624"},
        {"observed_linkedin_slug": "other-company"},
    ],
)
def test_numeric_homepage_keeps_numeric_anchor_without_exact_verified_receipt(
    changes,
):
    homepage = lead_scorer._verified_homepage_identity_anchor(
        _rapid7_homepage_identity()
    )
    resolved = lead_scorer._structured_profile_alias_identity_receipt(
        _rapid7_company(),
        _rapid7_web_identity(),
        _rapid7_structured_identity(),
        "rapid7.com",
        company_quality=True,
    )
    assert resolved["reason_code"] == (
        "structured_numeric_linkedin_alias_verified"
    )

    assert lead_scorer._structured_profile_identity_anchor(
        homepage,
        {**resolved, **changes},
        "rapid7.com",
    ) == homepage


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_code": "identity_not_proven"},
        {"observed_domain": "other.example"},
        {"observed_name": "other company"},
        {"observed_linkedin_slug": "393624"},
        {"submitted_linkedin_slug": "393624"},
    ],
)
def test_homepage_anchor_is_preserved_without_exact_numeric_alias(changes):
    homepage = lead_scorer._verified_homepage_identity_anchor(
        _rapid7_homepage_identity()
    )
    web = {**_rapid7_web_identity(), **changes}

    assert lead_scorer._structured_profile_identity_anchor(
        homepage,
        web,
        "rapid7.com",
    ) == homepage


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("company_id", "393624"),
        ("requested_url", "https://www.linkedin.com/company/393624"),
        ("url", "https://www.linkedin.com/company/other"),
        ("name", "Other Company"),
        ("website", "https://other.example/"),
    ],
)
def test_numeric_linkedin_alias_rejects_any_structured_conflict(field, value):
    assert not lead_scorer._structured_profile_alias_identity_receipt(
        _rapid7_company(),
        _rapid7_web_identity(),
        _rapid7_structured_identity(**{field: value}),
        "rapid7.com",
        company_quality=True,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_website_url": "https://other.example/"},
        {"provider_website_final_url": "https://other.example/"},
        {"provider_website_final_url": "http://rapid7.com/"},
        {"provider_website_requested_url": ""},
    ],
)
def test_numeric_linkedin_alias_rejects_redirect_receipt_conflict(changes):
    assert not lead_scorer._structured_profile_alias_identity_receipt(
        _rapid7_company(),
        _rapid7_web_identity(),
        _rapid7_redirected_structured_identity(**changes),
        "rapid7.com",
        company_quality=True,
    )


def test_numeric_linkedin_alias_rejects_incomplete_redirect_receipt():
    evidence = _rapid7_redirected_structured_identity()
    evidence.pop("provider_website_final_url")
    assert not lead_scorer._structured_profile_alias_identity_receipt(
        _rapid7_company(),
        _rapid7_web_identity(),
        evidence,
        "rapid7.com",
        company_quality=True,
    )


def test_numeric_linkedin_alias_unlocks_full_company_fit_gate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"structured": 0}

    async def provider(**_kwargs):
        return {
            "observed_company_name": "Rapid7 Inc",
            "observed_company_website": "https://rapid7.com",
            "observed_company_linkedin": RAPID7_VANITY_URL,
            "observed_employee_count": 11706,
            "employee_size_matches": True,
            "employee_size_evidence_url": (
                "https://rapid7.com/company/about"
            ),
            "employee_size_evidence_quote": "Rapid7 has 11,706 employees.",
            "observed_industry": "Food and Beverage Manufacturing",
            "observed_subindustry": "",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://rapid7.com/company/about",
            "industry_evidence_quote": "Rapid7 develops food brands.",
            "observed_hq_country": "United States",
            "observed_hq_state": "Massachusetts",
            "geography_matches": True,
            "geography_evidence_url": "https://rapid7.com/company/about",
            "geography_evidence_quote": "Rapid7 is based in Boston.",
            "reason": "All requested dimensions match.",
        }, ""

    async def structured_profile(
        domain,
        url,
        *,
        company_identity_evidence,
        company_identity_observed_profile_url,
        **_kwargs,
    ):
        calls["structured"] += 1
        assert (domain, url) == ("rapid7.com", RAPID7_NUMERIC_URL)
        assert company_identity_observed_profile_url == RAPID7_VANITY_URL
        company_identity_evidence.update(
            _rapid7_redirected_structured_identity()
        )
        return {
            "employee_count": "10,001+",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": RAPID7_VANITY_URL,
            "website": "https://rapid7.com/",
        }

    async def current_profile(url, **_kwargs):
        assert url == RAPID7_VANITY_URL
        return {"outcome": "insufficient_evidence", "url": url}

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
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _rapid7_company(),
            _icp().model_copy(update={
                "employee_count": "10,001+",
                "geography": "United States",
                "country": "United States",
            }),
            require_company_fit_dimensions=True,
            verified_homepage_identity=company_fit_unavailable(
                "homepage identity is incomplete",
                details={
                    "identity": _rapid7_web_identity(),
                    "verified_homepage_transport_domain": "rapid7.com",
                },
            ),
            company_quality=True,
        )
    )

    assert calls == {"structured": 1}
    assert result.decision == COMPANY_FIT_MATCH, result.details
    assert result.details["identity_receipt"]["reason_code"] == (
        "structured_numeric_linkedin_alias_verified"
    )


def test_numeric_homepage_identity_runs_full_structured_alias_gate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = {"structured": 0}

    async def provider(**_kwargs):
        return {
            "observed_company_name": "Rapid7, Inc.",
            "observed_company_website": "https://www.rapid7.com",
            "observed_company_linkedin": RAPID7_VANITY_URL,
            "observed_employee_count": "1,001-5,000",
            "employee_size_matches": True,
            "employee_size_evidence_url": RAPID7_VANITY_URL,
            "employee_size_evidence_quote": "Company size 1,001-5,000 employees",
            "observed_industry": "Food and Beverage Manufacturing",
            "observed_subindustry": "",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://rapid7.com/company/about",
            "industry_evidence_quote": "Rapid7 develops food brands.",
            "observed_hq_country": "United States",
            "observed_hq_state": "Massachusetts",
            "geography_matches": True,
            "geography_evidence_url": "https://rapid7.com/company/about",
            "geography_evidence_quote": "Rapid7 is based in Boston.",
            "reason": "All requested dimensions match.",
        }, ""

    async def structured_profile(
        domain,
        url,
        *,
        company_identity_evidence,
        company_identity_observed_profile_url,
        **_kwargs,
    ):
        calls["structured"] += 1
        assert (domain, url) == ("rapid7.com", RAPID7_NUMERIC_URL)
        assert company_identity_observed_profile_url == RAPID7_VANITY_URL
        company_identity_evidence.update(
            _rapid7_redirected_structured_identity()
        )
        return {
            "employee_count": "1,001-5,000",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": RAPID7_VANITY_URL,
            "website": "https://rapid7.com/",
        }

    async def current_profile(url, **_kwargs):
        assert url == RAPID7_VANITY_URL
        return {"outcome": "insufficient_evidence", "url": url}

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
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _rapid7_company().model_copy(update={
                "employee_count": "1,001-5,000",
            }),
            _icp().model_copy(update={
                "employee_count": "1,001-5,000",
                "geography": "United States",
                "country": "United States",
            }),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_rapid7_homepage_identity(),
            company_quality=True,
        )
    )

    assert calls == {"structured": 1}
    assert result.decision == COMPANY_FIT_MATCH, result.details
    assert result.details["identity_receipt"]["reason_code"] == (
        "structured_numeric_linkedin_alias_verified"
    )


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
        **_kwargs,
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
        **_kwargs,
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
