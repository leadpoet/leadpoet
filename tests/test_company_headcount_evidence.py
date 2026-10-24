"""Headcount evidence controls from the October 23 saved company-fit failure."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from copy import deepcopy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer as scorer
from qualification.scoring import linkedin_company_size as profiles
from qualification.scoring.company_evidence_investigator import (
    PRIVATE_FETCHED_PAGES_KEY, _plain_text, _quote_supports_headcount,
)

PROFILE = "https://www.linkedin.com/company/zag-bio"
CAPTURE = json.loads((Path(__file__).parent / "fixtures/company_evidence/zag_headcount_oct23.json").read_text())
ARTICLE = CAPTURE["first_party_article"]["url"]


def company():
    return CompanyOutput(company_name="Zag Bio", company_website="https://zagbio.com/",
                         company_linkedin="", industry="Biotechnology", employee_count="11-50",
                         country="United States", intent_signals=[dict(description="Zag Bio announces financing.", source="news", url=ARTICLE, date="2026-10-01", snippet="Zag Bio announces financing.")])


def icp():
    return ICPPrompt(icp_id="headcount-proof", prompt="test", industry="Biotechnology",
                     employee_count="11-50 | 51-200", sub_industry="", product_service="", geography="United States",
                     company_stage="")


def verdict(count="0-1", quote="Zag Bio has 1-10 Employees"):
    return dict(observed_company_name="Zag Bio", observed_company_website="https://zagbio.com/",
                observed_company_linkedin="", observed_employee_count=count,
                employee_size_matches=False, employee_size_evidence_url="https://prospeo.io/c/zag-bio",
                employee_size_evidence_quote=quote, observed_industry="Biotechnology",
                observed_subindustry="Biotechnology", industry_matches=True,
                industry_activity_role="supplier_operator", industry_evidence_url="https://zagbio.com/",
                industry_evidence_quote="Zag Bio develops biotechnology medicines.",
                observed_hq_country="United States", observed_hq_state="Massachusetts",
                geography_matches=True, geography_evidence_url="https://zagbio.com/about",
                geography_evidence_quote="Zag Bio is headquartered in Massachusetts, United States.",
                reason="Independent observations")


def payload(**updates):
    return {"status": "completed", "result": {"data": {"status": 200, "element": {
        "name": "Zag Bio", "website": "https://zagbio.com/", "linkedinUrl": PROFILE,
        "employeeCountRange": {"start": 11, "end": 50}, **updates,
    }}}}


@pytest.mark.parametrize("count,quote,expected", [
    ("0-1", "Zag Bio has 1-10 Employees", "unavailable"),
    ("2-10", "Zag Bio has 1-10 Employees", "unavailable"),
    ("11-50", "Zag Bio has 11-50 employees", "match"),
    (26, "Zag Bio has 26 employees", "match"),
    (250, "Zag Bio has 250 employees", "mismatch"),
    ("11-50", "HEADCOUNT 30+ employees (growing to 100+)", "unavailable"),
    ("11-50", "Zag Bio has 111-500 employees", "unavailable"),
    (26, "Zag Bio has 26 associated members", "unavailable"),
    (26, "Zag Bio has 26 followers", "unavailable"),
    (26, "The Boston office has 26 employees", "unavailable"),
])
def test_web_fit_requires_exact_company_wide_count(count, quote, expected):
    result = scorer._reverify_decision(verdict(count, quote), "", "", icp=icp(), company=company())
    assert result.details["dimension_decisions"]["employee_size"] == expected


def test_count_range_never_matches_inside_a_larger_numeric_range():
    assert not _quote_supports_headcount("111-500 employees", "11-50")


def install_no_network(monkeypatch, observed):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    async def request(**kwargs):
        return deepcopy(observed), ""
    async def no_profile(*args, **kwargs):
        raise AssertionError("unbound company must not request a LinkedIn page")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "fetch_current_linkedin_company_size", no_profile)
    async def unavailable_page(*args, **kwargs):
        return 503, args[1], ""
    monkeypatch.setattr(scorer, "_fetch_bounded_html", unavailable_page)


@pytest.mark.parametrize("change,expected,calls", [
    ({}, "match", 1),
    ({"website": "https://other.example"}, "unavailable", 1),
    ({"linkedinUrl": "https://www.linkedin.com/company/other"}, "unavailable", 1),
    ({"employeeCountRange": {"start": 201, "end": 500}}, "mismatch", 1),
    ({"error": "unavailable"}, "unavailable", 1),
])
def test_saved_order_investigator_page_supplies_locator_only(monkeypatch, change, expected, calls):
    """Homepages have no link; a later first-party article does; LinkedIn is a wall."""
    install_no_network(monkeypatch, CAPTURE["web_observation"])
    assert len(CAPTURE["homepages"]) == 2
    assert all(PROFILE not in page["text"] for page in CAPTURE["homepages"])
    # The real 8,269-character article carries the profile at its end, after
    # the visible-link identity marker. Keep that entire bounded saved text.
    assert CAPTURE["first_party_article"]["text"].find(PROFILE) > 7500
    order = []
    async def investigation(**kwargs):
        order.append("investigator")
        assert kwargs["prior_observations"]["observed_employee_count"] is None
        return {"claims": {"headcount": {"status": "UNPROVEN"}}, "_completed_submit": True,
                PRIVATE_FETCHED_PAGES_KEY: {
                    ARTICLE: {"final_url": ARTICLE, "text": CAPTURE["first_party_article"]["text"]},
                    PROFILE: {"final_url": PROFILE, "text": "Sign Up | LinkedIn"},
                }}
    async def structured(domain, url, **kwargs):
        order.append("structured")
        reply = payload(**change)
        if "error" in change:
            return None
        identity = profiles.project_structured_linkedin_company_identity(domain, url, reply)
        if identity:
            kwargs["company_identity_evidence"].update(identity)
        return profiles.project_structured_linkedin_company_size(domain, url, reply)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigation)
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    result = asyncio.run(scorer._llm_reverify_company(company(), icp(), require_company_fit_dimensions=True,
                         evidence_investigator=True, verified_homepage_identity=scorer.company_fit_unavailable(
                             "LinkedIn company binding not found", details={"verified_homepage_transport_domain": "zagbio.com"})))
    assert result.details["dimension_decisions"]["employee_size"] == expected
    assert order == ["investigator"] + ["structured"] * calls


def test_newo_dom_boundaries_keep_separate_year_and_employee_count(monkeypatch):
    observed = verdict(76, "Founded 202376 employees")
    install_no_network(monkeypatch, observed)
    fetches = []
    async def fetch(_session, url):
        fetches.append(url)
        return 200, url, "<div>Founded 2023</div><div>76 employees</div>"
    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)
    result = asyncio.run(scorer._llm_reverify_company(company(), icp(), require_company_fit_dimensions=True))
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.details["provider_observations"]["observed_employee_count"] == 76
    assert "202376" not in result.details["dimension_evidence"]["employee_size"]["quote"]
    assert fetches == [observed["employee_size_evidence_url"]]


@pytest.mark.parametrize("text,expected", [
    ("Straße Founded 2023\n76 employees", "Straße Founded 2023\n76 employees"),
    ("Founded 2023\n76 employees / Founded 2023 76 employees", "Founded 2023\n76 employees"),
    ("Founded 2023\n76 employees / Founded 202376 employees", ""),
])
def test_count_source_boundary_mapping_handles_unicode_and_duplicate_blocks(text, expected):
    quote = "Straße Founded 202376 employees" if text.startswith("Straße") else "Founded 202376 employees"
    assert scorer._headcount_quote_with_source_boundaries(quote, text) == expected


@pytest.mark.parametrize("quote,count,expected", [
    ("Company size 11 - 50 employees", "11-50", True),
    ("Employee count 111-500", "11-50", False),
    ("The Boston office has Company size 11-50 employees", "11-50", False),
    ("The department has 26 employees", 26, False),
    ("Bedrijfsgrootte\n11 - 50 medewerkers", "11-50", True),
    ("Associated members 26", 26, False),
    ("Followers 11-50", "11-50", False),
])
def test_supported_count_fields_do_not_bypass_scope_or_number_checks(quote, count, expected):
    assert _quote_supports_headcount(quote, count) is expected


def web_identity():
    return scorer._web_identity_receipt(company(), verdict())


def page(text, url=ARTICLE, final_url=None):
    return {url: {"final_url": final_url or url, "text": text}}


@pytest.mark.parametrize("pages,submitted_slug,observed_slug", [
    (page(f"Zag Bio {PROFILE} https://www.linkedin.com/company/other"), "", ""),
    (page(f"Zag Bio {PROFILE}", "https://directory.example/zag"), "", ""),
    (page(f"Zag Bio {PROFILE}", final_url="https://directory.example/zag"), "", ""),
    (page(f"Zag Bio {PROFILE}"), "other", ""),
    (page(f"Zag Bio {PROFILE}"), "", "other"),
])
def test_ambiguous_unbound_or_conflicting_locator_never_requests_provider(monkeypatch, pages, submitted_slug, observed_slug):
    async def forbidden(*args, **kwargs):
        raise AssertionError("untrusted locator reached provider")
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", forbidden)
    candidate = company().model_copy(update={"company_linkedin": f"https://www.linkedin.com/company/{submitted_slug}" if submitted_slug else ""})
    identity = {**web_identity(), "observed_linkedin_slug": observed_slug}
    assert asyncio.run(scorer._lookup_first_party_headcount_profile(candidate, identity, "zagbio.com", pages, {})) == {}


def test_one_profile_attempt_is_shared_across_repair_even_when_provider_unavailable(monkeypatch):
    requests = []
    async def unavailable(domain, url, **kwargs):
        requests.append((domain, url))
        return None
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", unavailable)
    cache = {}
    pages = page(CAPTURE["first_party_article"]["text"])
    for _ in range(2):
        assert asyncio.run(scorer._lookup_first_party_headcount_profile(company(), web_identity(), "zagbio.com", pages, cache)) == {}
    assert requests == [("zagbio.com", PROFILE)]


@pytest.mark.parametrize("count,quote", [(26, "Zag Bio has 26 employees"), (250, "Zag Bio has 250 employees")])
def test_grounded_web_count_does_not_add_structured_lookup(monkeypatch, count, quote):
    install_no_network(monkeypatch, verdict(count, quote))
    async def forbidden(*args, **kwargs):
        raise AssertionError("already grounded headcount must not add a profile lookup")
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", forbidden)
    result = asyncio.run(scorer._llm_reverify_company(
        company(), icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=scorer.company_fit_unavailable("missing profile", details={"verified_homepage_transport_domain": "zagbio.com"}),
        verified_homepage_pages=page(CAPTURE["first_party_article"]["text"]),
    ))
    assert result.details["dimension_decisions"]["employee_size"] == ("match" if count == 26 else "mismatch")


def test_stage_only_investigation_does_not_discover_headcount_profile(monkeypatch):
    async def investigation(**kwargs):
        return {"claims": {"stage": {"status": "UNPROVEN"}}, "_completed_submit": True,
                PRIVATE_FETCHED_PAGES_KEY: page(CAPTURE["first_party_article"]["text"])}
    async def forbidden(*args, **kwargs):
        raise AssertionError("a stage-only investigation must not add headcount lookup")
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigation)
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", forbidden)
    result = asyncio.run(scorer._run_targeted_company_evidence_investigation(
        company=company(), icp=icp(), verdict=verdict(26, "Zag Bio has 26 employees"),
        investigation_targets=("stage",), icp_attribute="", icp_stage="Series A",
        verified_identity={}, verified_transport_domain="zagbio.com",
        structured_employee_size_evidence=None, structured_public_company_evidence=None,
        employee_size_conflict=False, company_quality=False, structured_profile_invocation_cache={},
    ))
    assert result[1].details["dimension_decisions"]["employee_size"] == "match"


def test_employee_source_fetch_is_bounded_and_preserves_attribute_slots(monkeypatch):
    calls = []
    async def fetch(_session, url):
        calls.append(url)
        raise TimeoutError("controlled provider timeout")
    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)
    cache = {}
    attribute_pages = {"https://zagbio.com/a": {"text": "first attribute page"},
                       "https://zagbio.com/b": {"text": "second attribute page"}}
    original = deepcopy(attribute_pages)
    for url in ("https://directory.example/a", "https://directory.example/a", "https://directory.example/b"):
        observed = verdict(76, "Founded 202376 employees")
        observed["employee_size_evidence_url"] = url
        result = asyncio.run(scorer._ground_employee_size_observation(observed, invocation_cache=cache, source_pages=attribute_pages))
        assert result["observed_employee_count"] is None
    assert calls == ["https://directory.example/a"]
    assert cache["refresh_outcome"] == "retryable_failure"
    assert cache[scorer.VERIFIER_FAILURE_DETAIL_KEY] == scorer.PROVIDER_ERROR_FAILURE_REASON
    assert attribute_pages == original


@pytest.mark.parametrize("final_url,status", [
    ("https://other.example/unrelated", "fetched"),
    ("https://prospeo.io/c/zag-bio", "source_unavailable"),
])
def test_restored_count_requires_matching_source_transport_and_success(monkeypatch, final_url, status):
    observed = verdict(76, "Founded 202376 employees")
    cached = {observed["employee_size_evidence_url"]: {
        "status": status, "final_url": final_url, "text": "Founded 2023\n76 employees",
    }}
    async def forbidden(*args, **kwargs):
        raise AssertionError("cached source must not cause another source fetch")
    monkeypatch.setattr(scorer, "_fetch_bounded_html", forbidden)
    result = asyncio.run(scorer._ground_employee_size_observation(observed, invocation_cache={}, source_pages=cached))
    assert result["observed_employee_count"] is None


@pytest.mark.parametrize("missing", ["employee_size_evidence_url", "employee_size_evidence_quote"])
def test_incomplete_web_evidence_can_use_already_fetched_first_party_locator(monkeypatch, missing):
    observed = verdict(26, "Zag Bio has 26 employees")
    observed[missing] = ""
    install_no_network(monkeypatch, observed)
    requests = []
    async def structured(domain, url, **kwargs):
        requests.append((domain, url))
        reply = payload()
        kwargs["company_identity_evidence"].update(profiles.project_structured_linkedin_company_identity(domain, url, reply))
        return profiles.project_structured_linkedin_company_size(domain, url, reply)
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    result = asyncio.run(scorer._llm_reverify_company(
        company(), icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=scorer.company_fit_unavailable("missing profile", details={"verified_homepage_transport_domain": "zagbio.com"}),
        verified_homepage_pages=page(CAPTURE["first_party_article"]["text"]),
    ))
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert requests == [("zagbio.com", PROFILE)]


def test_source_read_timeout_keeps_reverify_failure_retryable(monkeypatch):
    install_no_network(monkeypatch, verdict(76, "Founded 202376 employees"))
    calls = []
    async def timeout(_session, url):
        calls.append(url)
        raise asyncio.TimeoutError("controlled read timeout")
    monkeypatch.setattr(scorer, "_fetch_bounded_html", timeout)
    result = asyncio.run(scorer._llm_reverify_company(
        company(), icp(), require_company_fit_dimensions=True,
    ))
    assert result.decision == "unavailable"
    assert result.details["failure_class"] == scorer.EMPLOYEE_SIZE_VERIFICATION_FAILURE_CLASS
    assert result.details[scorer.VERIFIER_FAILURE_DETAIL_KEY] == scorer.PROVIDER_ERROR_FAILURE_REASON
    assert calls == ["https://prospeo.io/c/zag-bio"]
