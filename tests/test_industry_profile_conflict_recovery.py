"""An unfetched, model-reported profile is not stronger than an exact profile read."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_unavailable
from tests.test_fit_evidence_url_compat import _company, _complete_verdict, _icp


def _example(name="Acme", domain="acme.com", submitted="acme-systems", observed="acme"):
    company = _company(company_name=name, company_website=f"https://{domain}/",
                       company_linkedin=f"https://www.linkedin.com/company/{submitted}")
    verdict = _complete_verdict(name=name, website=f"https://{domain}/")
    verdict["observed_company_linkedin"] = f"https://www.linkedin.com/company/{observed}"
    evidence = {"name": name, "website": f"https://{domain}/",
                "url": company.company_linkedin, "provider": scorer.STRUCTURED_PROFILE_PROVIDER,
                "source_field": scorer.STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD}
    return company, verdict, evidence


@pytest.mark.parametrize("name,domain,submitted,observed", [
    ("Acme", "acme.com", "acme-systems", "acme"),
    # Exact identity discrepancy in the V40 historical provider response.
    ("EPAM Systems", "epam.com", "epam-systems", "epam"),
])
def test_exact_independent_profile_resolves_only_the_unverified_slug(
    monkeypatch, name, domain, submitted, observed,
):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example(name, domain, submitted, observed)
    if name == "EPAM Systems":
        verdict["observed_company_name"] = "EPAM Systems, Inc."
        verdict["observed_company_website"] = "https://www.epam.com"
    original = deepcopy(verdict)
    assert scorer._web_identity_receipt(company, verdict)["decision"] == "mismatch"
    receipt = scorer._web_identity_receipt(company, verdict,
        verified_structured_identity=evidence, company_quality=True)
    assert receipt["decision"] == "match"
    assert receipt["observed_linkedin_slug"] == submitted
    assert receipt["raw_observed_linkedin_slug"] == observed
    assert receipt["structured_profile_identity"] == evidence
    assert verdict == original


@pytest.mark.parametrize("change", [
    {"name": "Different Company"}, {"website": "https://other.com/"},
    {"url": "https://www.linkedin.com/company/other"},
    {"url": "https://www.linkedin.com/company/acme-systems?alias=other"},
    {"url": "http://www.linkedin.com/company/acme-systems"},
    {"provider": "model"}, {"source_field": "description"}, {"extra": True},
])
def test_unbound_or_altered_profile_cannot_override_conflict(monkeypatch, change):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    receipt = scorer._web_identity_receipt(company, verdict,
        verified_structured_identity={**evidence, **change}, company_quality=True)
    assert receipt["decision"] == "mismatch"


@pytest.mark.parametrize("fault", ["legacy_policy", "different_name", "different_domain", "fetched_homepage"])
def test_existing_identity_boundaries_remain(monkeypatch, fault):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    kwargs = {}
    if fault == "legacy_policy":
        monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    elif fault == "different_name":
        verdict["observed_company_name"] = "Different Company"
    elif fault == "different_domain":
        verdict["observed_company_website"] = "https://other.com/"
    else:
        kwargs["verified_homepage_transport_domain"] = "acme.com"
    receipt = scorer._web_identity_receipt(company, verdict,
        verified_structured_identity=evidence, company_quality=True, **kwargs)
    assert receipt["decision"] != "match"


@pytest.mark.parametrize("homepage_state", [
    "fetch_failure", "missing", "fetched_url", "identity_unproven", "mismatch",
    "legacy_policy", "wrong_profile",
])
def test_native_reverification_fetches_once_and_keeps_other_dimensions(monkeypatch, homepage_state):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    company, verdict, evidence = _example()
    homepage = company_fit_unavailable("website fetch error: Arena provider response unavailable",
        details={"failure_reason_code": "provider_error", "actual_final_url": ""})
    if homepage_state == "missing":
        homepage = None
    elif homepage_state == "fetched_url":
        homepage.details["actual_final_url"] = company.company_website
    elif homepage_state == "identity_unproven":
        homepage = company_fit_unavailable("company identity was not proven")
    elif homepage_state == "mismatch":
        homepage = scorer.company_fit_mismatch("different company")
    elif homepage_state == "legacy_policy":
        monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    elif homepage_state == "wrong_profile":
        evidence["name"] = "Different Company"
    requests = []
    async def structured(domain, profile, **kwargs):
        requests.append((domain, profile))
        kwargs["company_identity_evidence"].update(evidence)
        return None
    async def preserve(value, **kwargs):
        return value
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    monkeypatch.setattr(scorer, "_request_company_reverify_json", AsyncMock(return_value=(verdict, "")))
    monkeypatch.setattr(scorer, "_resolve_observed_linkedin_redirect_alias", AsyncMock(side_effect=lambda _c, v: v))
    monkeypatch.setattr(scorer, "_ground_employee_size_observation", preserve)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", AsyncMock(side_effect=lambda v, *_a, **_k: v))
    monkeypatch.setattr(scorer, "_positive_semantic_review_needed", lambda *_a: False)
    monkeypatch.setattr(scorer, "_targeted_company_investigation_dimensions", lambda *_a, **_k: ())
    result = asyncio.run(scorer._llm_reverify_company(company, _icp(),
        require_company_fit_dimensions=True, verified_homepage_identity=homepage,
        company_quality=True, evidence_investigator=True))
    attempted = homepage_state in {"fetch_failure", "wrong_profile"}
    assert requests == ([("acme.com", company.company_linkedin)] if attempted else [])
    if homepage_state == "fetch_failure":
        assert result.details["identity_decision"] == "match"
        assert result.details["identity_receipt"]["structured_profile_identity"] == evidence
        requests.clear()
        monkeypatch.setattr(scorer, "verify_company_exists", AsyncMock(return_value=homepage))
        monkeypatch.setattr(scorer, "run_company_zero_checks",
                            AsyncMock(return_value=scorer.company_fit_match("prechecks passed")))
        aggregate = asyncio.run(scorer._verify_company_fit(company, _icp(), 0, 0, set(),
            require_https_transport=True, company_quality=True, evidence_investigator=True))
        saved = json.loads(json.dumps(aggregate.details))
        identity = saved["dimension_evidence"]["identity"]
        assert saved["company_fit_dimensions"]["identity"] == "match"
        assert identity["homepage_identity_decision"] == "unavailable"
        assert identity["web_identity_receipt"]["structured_profile_identity"] == evidence
        assert identity["web_identity_receipt"]["raw_observed_linkedin_slug"] == "acme"
        assert requests == [("acme.com", company.company_linkedin)]
    else:
        assert result.details["identity_decision"] != "match"
    assert result.details["dimension_decisions"]["industry"] == "match"


def test_recovered_profile_survives_investigator_identity_handoff(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    anchor = {"normalized_name": evidence["name"], "registrable_dns_domain": "acme.com",
              "linkedin_company_slug": "acme-systems"}
    receipt = scorer._web_identity_receipt(company, verdict,
        verified_homepage_identity=anchor, verified_structured_identity=evidence,
        company_quality=True)
    assert receipt["decision"] == "match"
    assert receipt["structured_profile_identity"] == evidence
    conflicting_anchor = {**anchor, "linkedin_company_slug": "other-company"}
    assert scorer._web_identity_receipt(company, verdict,
        verified_homepage_identity=conflicting_anchor, verified_structured_identity=evidence,
        company_quality=True)["decision"] != "match"


def test_identity_recovery_is_not_industry_or_attribute_proof(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    verdict["industry_matches"] = None
    verdict["industry_activity_role"] = "unresolved"
    verdict["industry_evidence_url"] = ""
    verdict["industry_evidence_quote"] = ""
    verdict["attribute_satisfied"] = None
    result = scorer._reverify_decision(verdict, "Required capability", "Series A",
        icp=_icp(), company=company, company_quality=True,
        structured_profile_identity_evidence=evidence)
    assert result.details["identity_decision"] == "match"
    assert result.details["dimension_decisions"]["industry"] == "unavailable"
    assert result.details["required_attribute_decision"] == "unavailable"
    assert result.decision != "match"


def test_recovered_identity_keeps_structured_stage_conflict_guard(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    verdict.update(observed_company_stage="Public", stage_matches=True,
                   stage_evidence_url="https://acme.com/investors",
                   stage_evidence_quote="Acme (NASDAQ: ACME) is a publicly traded company.")
    private = {"company_type": "Privately Held", "provider": scorer.STRUCTURED_PROFILE_PROVIDER,
               "source_field": scorer.STRUCTURED_PROFILE_COMPANY_TYPE_SOURCE_FIELD,
               "url": evidence["url"], "website": evidence["website"]}
    icp = _icp().model_copy(update={"company_stage": "Public"})
    baseline = scorer._reverify_decision(verdict, "", "public", icp=icp,
        company=company, company_quality=True, structured_profile_identity_evidence=evidence)
    assert baseline.details["dimension_decisions"]["stage"] == "match"
    result = scorer._reverify_decision(verdict, "", "public", icp=icp,
        company=company, company_quality=True, structured_profile_identity_evidence=evidence,
        structured_public_company_evidence=private)
    assert result.details["identity_decision"] == "match"
    assert result.details["dimension_decisions"]["stage"] != "match"


@pytest.mark.parametrize("change", [
    {"name": "Different Company"}, {"website": "https://other.com/"},
    {"url": "https://www.linkedin.com/company/other"}, {"provider": "model"},
    {"source_field": "description"}, {"extra": True}, {"name": ""},
])
def test_recovered_anchor_revalidates_private_identity(monkeypatch, change):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    receipt = scorer._web_identity_receipt(company, verdict,
        verified_structured_identity=evidence, company_quality=True)
    assert scorer._structured_profile_identity_anchor({}, receipt, "")
    receipt["structured_profile_identity"].update(change)
    assert scorer._structured_profile_identity_anchor({}, receipt, "") == {}


def test_research_uses_fetched_profile_and_retains_raw_observation(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    company, verdict, evidence = _example()
    verdict.update(industry_matches=None, industry_activity_role="unresolved",
                   industry_evidence_url="", industry_evidence_quote="")
    original = deepcopy(verdict)
    anchor = {"normalized_name": evidence["name"], "registrable_dns_domain": "acme.com",
              "linkedin_company_slug": "acme-systems"}
    research = AsyncMock(return_value={"_completed_submit": True, "claims": {
        "industry": {"target": "industry", "status": "UNPROVEN", "activity_role": "unresolved",
                     "evidence_url": "", "evidence_quote": "", "reason": "No capability proof"},
    }})
    monkeypatch.setattr(scorer, "investigate_company_evidence", research)
    result = asyncio.run(scorer._run_targeted_company_evidence_investigation(
        company=company, icp=_icp(), verdict=verdict, investigation_targets=("industry",),
        icp_attribute="", icp_stage="Series A", verified_identity=anchor,
        verified_transport_domain="", structured_employee_size_evidence=None,
        structured_public_company_evidence=None, employee_size_conflict=False,
        company_quality=True, structured_profile_identity_evidence=evidence))
    kwargs = research.call_args.kwargs
    assert kwargs["prior_observations"]["observed_company_linkedin"] == evidence["url"]
    assert kwargs["verified_homepage_identity"] == anchor
    assert result[1].details["identity_receipt"]["structured_profile_identity"] == evidence
    assert result[1].details["dimension_decisions"]["industry"] == "unavailable"
    assert verdict == original
