"""A requested LinkedIn URL cannot bind an unrelated fetched company body."""

from __future__ import annotations

import asyncio

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring import linkedin_company_size as size
from tests.test_current_linkedin_company_size import (
    _company, _icp, _install_exa_bodies, _verdict,
)

URL = "https://www.linkedin.com/company/novatti"
IDENTITY = {
    "normalized_name": "Novatti",
    "registrable_dns_domain": "novatti.com",
    "linkedin_company_slug": "novatti",
}
DISTRIX = (
    "Distrix Comércio e Serviços Ltda | LinkedIn\n# Distrix Comércio e Serviços Ltda\n"
    "## Sobre nós\nWebsite\nhttps://distrix.com.br/\n"
    "Tamanho da empresa\n\n 2-10 funcionários"
)


@pytest.mark.parametrize("text,expected", [
    ("# Novatti Group Ltd\n## About us\nCompany size 51-200 employees", True),
    ("Novatti | LinkedIn\n## Over ons\nBedrijfsgrootte 51-200 medewerkers", True),
    ("# Novatti Group (ASX:NOV)\n## About us\nWebsite\nhttps://www.novatti.com/\n"
     "Company size 51-200 employees", True),
    ("# Novatti Payments\n## Sobre nós\nSite\n[Novatti](https://novatti.com/)\n"
     "Tamanho da empresa 51-200 funcionários", True),
    ("# Novatti\n## About us\nWebsite: https://payments.novatti.com/\n"
     "Company size 51-200 employees", True),
    (DISTRIX, False),
    (DISTRIX.replace("Distrix Comércio e Serviços Ltda", "Novatti"), False),
    ("# Distrix\n## About us\nA customer is Novatti at https://novatti.com/.\n"
     "Company size 2-10 employees", False),
    ("## About us\nNovatti\nCompany size 51-200 employees", False),
    ("Novatti\n# Distrix\n## About us\nCompany size 2-10 employees", False),
    ("https://novatti.com/\n## About us\nCompany size 2-10 employees", False),
    ("## About us\nCompany size 2-10 employees\n## Updates\nWebsite\n"
     "https://novatti.com/", False),
    ("# Novatti\n## About us\nWebsite\nhttps://novatti.com.evil.example/\n"
     "Company size 2-10 employees", False),
    ("# Novatti\n## About us\nWebsite\nhttps://other.example/\nWebsite\n"
     "https://novatti.com/\nCompany size 2-10 employees", False),
])
def test_profile_body_uses_own_identity_fields(text, expected):
    assert size._profile_body_matches_company_identity(text, IDENTITY) is expected


def test_verified_legal_alias_and_missing_identity_controls():
    text = "# Different Legal Holdings Ltd\n## About us\nCompany size 51-200 employees"
    assert not size._profile_body_matches_company_identity(text, IDENTITY)
    assert size._profile_body_matches_company_identity(text, {
        **IDENTITY, "verified_legal_name_aliases": ["Different Legal Holdings"],
    })
    assert not size._profile_body_matches_company_identity(text, {})


@pytest.mark.parametrize("returned_url", [
    URL, "https://br.linkedin.com/company/novatti/",
])
def test_wrong_body_is_insufficient_even_when_returned_url_matches(monkeypatch, returned_url):
    calls, pending = _install_exa_bodies(monkeypatch, {
        "results": [{"url": returned_url, "text": DISTRIX}],
    })
    diagnostic, source = {}, {}
    result = asyncio.run(size.fetch_current_linkedin_company_size(
        URL, diagnostic=diagnostic, source_text_sink=source,
        expected_company_identity=IDENTITY,
    ))
    assert result == {"outcome": "insufficient_evidence", "url": returned_url}
    assert diagnostic == {}
    assert source == {"url": returned_url, "text": DISTRIX}
    assert len(calls) == 1 and pending == []


@pytest.mark.parametrize("returned_url,identity,expected", [
    ("https://br.linkedin.com/company/novatti/", IDENTITY, "51-200"),
    # Transport alias alone cannot authorize a different profile identity.
    ("https://www.linkedin.com/company/novattibr", IDENTITY, None),
    (URL, {}, None),
])
def test_body_binding_keeps_existing_profile_url_contract(monkeypatch, returned_url, identity, expected):
    calls, _ = _install_exa_bodies(monkeypatch, {"results": [{
        "url": returned_url,
        "text": "# Novatti Group Ltd\n## About us\nCompany size 51-200 employees",
    }]})
    result = asyncio.run(size.fetch_current_linkedin_company_size(
        URL, expected_company_identity=identity,
    ))
    assert (result or {}).get("employee_count") == expected
    assert len(calls) == 1


def test_wrong_body_recovery_uses_existing_bounded_fallback(monkeypatch):
    body = {"results": [{"url": URL, "text": DISTRIX}]}
    calls, pending = _install_exa_bodies(monkeypatch, body, body)
    structured_calls = []

    async def structured(domain, url, **kwargs):
        structured_calls.append((domain, url))
        return {
            "employee_count": "51-200", "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange", "url": url,
            "website": "https://novatti.com/",
        }

    monkeypatch.setattr(lead_scorer, "fetch_structured_linkedin_company_size", structured)
    company = _company().model_copy(update={
        "company_name": "Novatti", "company_website": "https://novatti.com/",
        "company_linkedin": URL,
    })
    icp = _icp().model_copy(update={"employee_count": "51-200"})
    verdict = _verdict(observed_size="2-10", employee_url=URL)
    cache = {}

    async def run():
        result = await lead_scorer._refresh_linkedin_employee_size_observation(
            verdict, company, icp, verified_homepage_identity=IDENTITY,
            invocation_cache=cache,
        )
        # Repeated schema repair must reuse both bounded provider results.
        await lead_scorer._refresh_linkedin_employee_size_observation(
            verdict, company, icp, verified_homepage_identity=IDENTITY,
            invocation_cache=cache,
        )
        return result

    result = asyncio.run(run())
    assert result["observed_employee_count"] is None
    decision = lead_scorer._reverify_decision(result, "", "", icp=icp,
        structured_employee_size_evidence=cache["structured_evidence"])
    assert decision.details["dimension_decisions"]["employee_size"] == "match"
    assert len(calls) == 2 and pending == []
    assert structured_calls == [("novatti.com", URL)]
    assert cache["refresh_outcome"] == "verified"


def test_wrong_body_without_recovery_is_unavailable_and_bounded(monkeypatch):
    body = {"results": [{"url": URL, "text": DISTRIX}]}
    calls, pending = _install_exa_bodies(monkeypatch, body, body)
    structured_calls = []

    async def no_structured(domain, url, **kwargs):
        structured_calls.append((domain, url))
        return None

    monkeypatch.setattr(lead_scorer, "fetch_structured_linkedin_company_size", no_structured)
    company = _company().model_copy(update={
        "company_name": "Novatti", "company_website": "https://novatti.com/",
        "company_linkedin": URL,
    })
    verdict = _verdict(observed_size="2-10", employee_url=URL)
    cache = {}
    for _ in range(2):
        result = asyncio.run(lead_scorer._refresh_linkedin_employee_size_observation(
            verdict, company, _icp(), verified_homepage_identity=IDENTITY,
            invocation_cache=cache,
        ))
        decision = lead_scorer._reverify_decision(result, "", "", icp=_icp())
        assert decision.details["dimension_decisions"]["employee_size"] == "unavailable"
    assert len(calls) == 2 and pending == []
    assert structured_calls == [("novatti.com", URL)]
    assert cache["refresh_outcome"] == "insufficient_evidence"
    assert "failure_reason" not in cache


def test_body_identity_requirement_does_not_mask_provider_failure(monkeypatch):
    calls, _ = _install_exa_bodies(monkeypatch, {
        "statuses": [{"status": "error", "error": {"httpStatusCode": 401}}],
        "results": [],
    })
    diagnostic = {}
    assert asyncio.run(size.fetch_current_linkedin_company_size(
        URL, diagnostic=diagnostic, expected_company_identity=IDENTITY,
    )) is None
    assert diagnostic == {"failure_reason": "provider_error"}
    assert len(calls) == 1


def test_generic_unproven_does_not_erase_valid_prior_size():
    verdict = _verdict(observed_size="11-50", size_matches=True)
    result = lead_scorer._project_investigator_headcount(
        verdict, {"status": "UNPROVEN"}, icp=_icp(), existing_conflict=False,
    )
    assert result == verdict
