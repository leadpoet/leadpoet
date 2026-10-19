"""Reuse only explicit, bound HQ facts already obtained by the profile fetch."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring import linkedin_company_size as profile

PROFILE = "https://www.linkedin.com/company/quaise-energy"
ANCHOR = {"normalized_name": "Quaise Energy", "registrable_dns_domain": "quaise.com",
          "linkedin_company_slug": "quaise-energy"}
CAPTURES = json.loads((Path(__file__).parent / "fixtures/quaise_structured_hq_profiles.json").read_text())
HQ_TEXT = ("Quaise Energy headquarters: Houston, Texas, United States. "
           "(harvestapi_get_company locations.headquarter=true)")


def _body():
    return deepcopy(CAPTURES[1]["fixture"])


def _project(body):
    return profile.project_structured_linkedin_company_headquarters(
        "quaise.com", PROFILE, "Quaise Energy", body,
    )


def _fetch(monkeypatch, body):
    calls = []

    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def json(self): return deepcopy(body)

    class Session:
        def __init__(self, *_args, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        def post(self, url, *, json, headers):
            assert url.endswith("/harvestapi_get_company/execute")
            assert json == {"payload": {"url": PROFILE}}
            calls.append(url)
            return Response()

    with monkeypatch.context() as local:
        local.setenv("DEEPLINE_API_KEY", "offline-placeholder")
        local.setattr(profile.aiohttp, "ClientSession", Session)
        cache = {}
        for _ in range(2):
            asyncio.run(lead_scorer._fetch_structured_linkedin_profile_once(
                ANCHOR, cache, collect_identity=True,
            ))
    assert len(calls) == 1
    return cache


@pytest.mark.parametrize("capture", CAPTURES, ids=["leader-branch-first", "zero-miner-hq-first"])
def test_captured_hq_survives_real_fetch_cache_without_another_request(monkeypatch, capture):
    cache = _fetch(monkeypatch, capture["fixture"])
    assert cache["structured_evidence"]["employee_count"] == "51-200"
    assert cache["structured_profile_headquarters_evidence"]["text"] == HQ_TEXT
    pages = lead_scorer._investigator_prefetched_pages(
        {}, [PROFILE], verified_identity=ANCHOR,
        structured_profile_description_evidence=cache["structured_profile_description_evidence"],
        structured_profile_headquarters_evidence=cache["structured_profile_headquarters_evidence"],
        include_structured_description=True, include_structured_headquarters=True,
    )
    assert pages == {PROFILE: {"final_url": PROFILE, "text": (
        capture["fixture"]["element"]["description"] + "\n\n" + HQ_TEXT
    )}}


@pytest.mark.parametrize("mutation", [
    "branch-only", "no-flag", "two-hqs", "string-true", "wrong-domain", "wrong-profile",
    "wrong-name", "country-conflict", "state-conflict", "malformed-country", "malformed-locations",
])
def test_ambiguous_unbound_or_malformed_locations_never_project_hq(mutation):
    body = _body()
    company = body["element"]
    hq, branch = company["locations"]
    if mutation == "branch-only": company["locations"] = [branch]
    elif mutation == "no-flag": hq.pop("headquarter")
    elif mutation == "two-hqs": branch["headquarter"] = True
    elif mutation == "string-true": hq["headquarter"] = "true"
    elif mutation == "wrong-domain": company["website"] = "https://other.example/"
    elif mutation == "wrong-profile": company["linkedinUrl"] = "https://www.linkedin.com/company/other"
    elif mutation == "wrong-name": company["name"] = "Other Company"
    elif mutation == "country-conflict": hq["parsed"]["countryCode"] = "DE"
    elif mutation == "state-conflict": hq["parsed"]["state"] = "Massachusetts"
    elif mutation == "malformed-country": hq["country"] = ["US"]
    elif mutation == "malformed-locations": company["locations"] = {"headquarter": True}
    assert _project(body) is None


def test_international_hq_projects_without_a_us_assumption():
    body = _body()
    body["element"]["locations"] = [{
        "headquarter": True, "country": "DE", "city": "Berlin",
        "parsed": {"country": "Germany", "countryCode": "DE", "city": "Berlin"},
    }]
    evidence = _project(body)
    assert "headquarters: Berlin, Germany." in evidence["text"]


@pytest.mark.parametrize("field,value", [
    ("name", "Other Company"), ("website", "https://other.example/"),
    ("url", "https://www.linkedin.com/company/other"),
    ("provider", "model"), ("source_field", "locations"),
])
def test_hq_prefetch_revalidates_private_source_and_identity(field, value):
    evidence = _project(_body())
    evidence[field] = value
    assert lead_scorer._investigator_prefetched_pages(
        {}, [PROFILE], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=evidence,
        include_structured_headquarters=True,
    ) == {}


def test_hq_joins_existing_page_without_losing_text_or_an_extra_page_slot():
    cache = {PROFILE: {"status": "fetched", "final_url": PROFILE,
                       "text": "Existing fetched profile text."}}
    pages = lead_scorer._investigator_prefetched_pages(
        cache, [PROFILE], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True,
    )
    assert pages[PROFILE]["text"] == "Existing fetched profile text.\n\n" + HQ_TEXT
    full = {f"https://source.example/{i}": {"status": "fetched", "final_url": f"https://source.example/{i}", "text": "Source."} for i in range(3)}
    bounded = lead_scorer._investigator_prefetched_pages(
        full, [*full, PROFILE], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True,
    )
    assert len(bounded) == 3
    assert bounded[PROFILE]["text"] == HQ_TEXT
    assert list(bounded) == [PROFILE, *list(full)[:2]]


def _company():
    return CompanyOutput(
        company_name="Quaise Energy", company_website="https://www.quaise.com/",
        company_linkedin=PROFILE, industry="Energy", sub_industry="Geothermal power",
        employee_count="51-200", company_stage="Series C+", country="United States", state="Texas",
        intent_signals=[{"description": "Quaise launched a geothermal project.",
                         "source": "company website", "url": "https://www.quaise.com/news",
                         "date": "2026-09-01", "snippet": "Quaise launched a geothermal project."}],
    )


def _verdict():
    return {
        "observed_company_name": "Quaise Energy", "observed_company_website": "https://www.quaise.com/",
        "observed_company_linkedin": PROFILE,
        "observed_employee_count": "51-200", "employee_size_matches": True,
        "employee_size_evidence_url": PROFILE, "employee_size_evidence_quote": "Company size 51-200 employees",
        "observed_industry": "Energy", "observed_subindustry": "Geothermal power",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://www.quaise.com/", "industry_evidence_quote": "Quaise Energy develops geothermal power plants.",
        "observed_hq_country": "", "observed_hq_state": "", "geography_matches": None,
        "geography_evidence_url": "", "geography_evidence_quote": "", "reason": "HQ needs evidence.",
    }


@pytest.mark.parametrize("country,state,quote,expected", [
    ("United States", "Texas", HQ_TEXT, "match"),
    ("United States", "Massachusetts", HQ_TEXT, "unavailable"),
    ("United States", "Texas", "Quaise Energy headquarters: Houston, Texas, United States. Invented sentence.", "unavailable"),
    ("Germany", "", "Quaise Energy headquarters: Berlin, Germany. (harvestapi_get_company locations.headquarter=true)", "mismatch"),
])
def test_real_targeted_scoring_consumes_hq_and_keeps_evidence_geography_gates(
    monkeypatch, country, state, quote, expected,
):
    body = _body()
    if country == "Germany":
        body["element"]["locations"] = [{"headquarter": True, "country": "Germany", "city": "Berlin"}]
    cache = _fetch(monkeypatch, body)
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        finding = {
            "target": "geography", "status": "VERIFIED", "observed_value": "",
            "observed_country": country, "observed_state": state,
            "observed_industry": "", "observed_subindustry": "", "activity_role": "unresolved",
            "evidence_url": PROFILE, "evidence_quote": quote,
            "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
            "shared_linkedin_slug": "", "reason": "The already fetched exact profile states headquarters.",
        }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit-hq", "type": "function", "function": {
                "name": "submit_findings", "arguments": json.dumps({"findings": [finding]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    fetch = AsyncMock(side_effect=AssertionError("No additional page fetch is needed"))
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    icp = ICPPrompt(icp_id="quaise-hq", industry="Energy", sub_industry="Geothermal power",
                    employee_count="51-200", company_stage="", geography="United States",
                    product_service="Geothermal power plants")
    verdict = _verdict()
    prior = lead_scorer._reverify_decision(verdict, "", "", icp=icp, company=_company(),
        verified_homepage_identity=ANCHOR, verified_homepage_transport_domain="quaise.com", company_quality=True)
    projected, result, claims, *_ = asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=_company(), icp=icp, verdict=verdict, investigation_targets=("geography",),
        icp_attribute="", icp_stage="", verified_identity=ANCHOR, verified_transport_domain="quaise.com",
        structured_employee_size_evidence=cache["structured_evidence"], structured_public_company_evidence=None,
        structured_profile_description_evidence=cache["structured_profile_description_evidence"],
        structured_profile_headquarters_evidence=cache["structured_profile_headquarters_evidence"],
        employee_size_conflict=False, company_quality=True, prior_result=prior,
        required_attribute_source_cache={},
    ))
    assert result.decision == expected, json.dumps({"details": result.details, "reason": result.reason, "claims": claims})
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["prefetched_sources"] == [{"url": PROFILE, "text": cache["structured_profile_headquarters_evidence"]["text"]}]
    assert document["investigation_limits"]["prefetched_pages"] == 1
    assert result.details["investigation_receipt"]["usage"].get("fetch_calls", 0) == 0
    assert fetch.await_count == 0
    if expected == "unavailable":
        assert claims.get("geography", {}).get("status", "UNPROVEN") == "UNPROVEN"
    else:
        assert projected["geography_matches"] is (expected == "match")


@pytest.mark.parametrize("private_source,source_update,quote", [
    (False, {}, HQ_TEXT),  # A submitted/copy-pasted profile quote is not trusted.
    (True, {"name": "Other Entity"}, HQ_TEXT),
    (True, {"website": "https://other.example/"}, HQ_TEXT),
    (True, {"url": "https://www.linkedin.com/company/other"}, HQ_TEXT),
    (True, {"source_field": "description"}, HQ_TEXT),
    (True, {}, "Quaise Energy office: Houston, Texas, United States."),
])
def test_production_findings_never_admit_generic_or_copied_linkedin_hq(
    private_source, source_update, quote,
):
    evidence = _project(_body())
    evidence.update(source_update)
    finding = {
        "target": "geography", "status": "VERIFIED", "observed_country": "United States",
        "observed_state": "Texas", "evidence_url": PROFILE, "evidence_quote": quote,
    }
    claims = investigator._validated_findings(
        {"findings": [finding]}, targets=("geography",),
        fetched_pages={PROFILE: quote}, fetched_final_urls={PROFILE: PROFILE},
        first_party_domains={"quaise.com"}, identity_names={"quaiseenergy"},
        identity_anchor={"verified_name": "Quaise Energy", "verified_domain": "quaise.com",
                         "verified_linkedin_slug": "quaise-energy"},
        structured_headquarters_evidence=evidence if private_source else None,
    )
    assert claims["geography"]["status"] == "UNPROVEN"


def test_validated_geography_boolean_cannot_be_forged_by_a_profile_quote():
    company = _company()
    icp = ICPPrompt(icp_id="hq-boolean", geography="United States", country="United States",
                    industry="Energy", sub_industry="Geothermal power", employee_count="51-200",
                    company_stage="", product_service="Geothermal power plants")
    verdict = {**_verdict(), "observed_hq_country": "United States", "observed_hq_state": "Texas",
               "geography_evidence_url": PROFILE, "geography_evidence_quote": HQ_TEXT}
    for flag in ("true", 1, None, False):
        assert lead_scorer._decision_from_observed_geography(
            {**verdict, "geography_matches": flag}, icp, company=company, company_quality=True,
        ) == "unavailable"


def test_hq_reuses_existing_trailing_slash_profile_without_a_duplicate_slot():
    submitted = PROFILE + "/"
    cache = {submitted: {"status": "fetched", "final_url": submitted,
                         "text": "Existing profile description."}}
    pages = lead_scorer._investigator_prefetched_pages(
        cache, [PROFILE, submitted], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True,
    )
    assert pages == {submitted: {"final_url": PROFILE,
                                "text": "Existing profile description.\n\n" + HQ_TEXT}}


@pytest.mark.parametrize("field,value", [
    ("name", "Other Entity"), ("website", "https://other.example/"),
    ("linkedinUrl", "https://www.linkedin.com/company/other"),
])
def test_real_profile_fetch_does_not_cache_unbound_hq(monkeypatch, field, value):
    body = _body()
    body["element"][field] = value
    assert "structured_profile_headquarters_evidence" not in _fetch(monkeypatch, body)


def test_hq_prefetch_requires_geography_scope():
    assert lead_scorer._investigator_prefetched_pages(
        {}, [PROFILE], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=_project(_body()),
    ) == {}


def test_empty_optional_location_fields_do_not_discard_explicit_hq():
    body = _body()
    hq = body["element"]["locations"][0]
    hq["geographicArea"] = ""
    hq["city"] = ""
    hq["parsed"]["countryFull"] = ""
    assert _project(body)["text"] == HQ_TEXT


def test_hq_prefetch_keeps_known_first_party_page_and_does_not_truncate_it():
    first_party = "https://www.quaise.com/contact"
    first_party_text = "Quaise Energy headquarters: Boston, Massachusetts, United States."
    pages = lead_scorer._investigator_prefetched_pages(
        {}, [first_party, PROFILE], verified_identity=ANCHOR,
        verified_homepage_pages={first_party: {"final_url": first_party, "text": first_party_text}},
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True,
    )
    assert list(pages) == [first_party, PROFILE]
    assert pages[first_party]["text"] == first_party_text
    assert pages[PROFILE]["text"] == HQ_TEXT
    assert "Use verified exact-company LinkedIn headquarters first" in investigator._SYSTEM_PROMPT


def test_structured_hq_does_not_overflow_or_truncate_an_existing_full_profile():
    text = "x" * investigator.MAX_PAGE_CHARACTERS
    pages = lead_scorer._investigator_prefetched_pages(
        {PROFILE: {"status": "fetched", "final_url": PROFILE, "text": text}},
        [PROFILE], verified_identity=ANCHOR,
        structured_profile_headquarters_evidence=_project(_body()),
        include_structured_headquarters=True,
    )
    assert pages == {PROFILE: {"final_url": PROFILE, "text": text}}


def test_current_first_party_hq_can_still_override_the_structured_profile_location():
    url = "https://www.quaise.com/contact"
    quote = "Quaise Energy is headquartered in Boston, Massachusetts, United States."
    claims = investigator._validated_findings(
        {"findings": [{"target": "geography", "status": "CONTRADICTED",
                       "observed_country": "United States", "observed_state": "Massachusetts",
                       "evidence_url": url, "evidence_quote": quote}]},
        targets=("geography",), fetched_pages={url: quote, PROFILE: HQ_TEXT},
        fetched_final_urls={url: url, PROFILE: PROFILE}, first_party_domains={"quaise.com"},
        identity_names={"quaiseenergy"},
        identity_anchor={"submitted_domain": "quaise.com", "observed_domain": "quaise.com",
                         "verified_domain": "quaise.com", "verified_name": "Quaise Energy",
                         "verified_linkedin_slug": "quaise-energy"},
        structured_headquarters_evidence=_project(_body()),
    )
    assert claims["geography"]["status"] == "CONTRADICTED"
    icp = ICPPrompt(icp_id="hq-conflict", geography="United States", industry="Energy",
                    sub_industry="Geothermal power", employee_count="51-200", company_stage="",
                    product_service="Geothermal power plants")
    projected = lead_scorer._project_investigator_geography(
        _verdict(), claims["geography"], icp=icp, company=_company(), company_quality=True,
    )
    assert lead_scorer._decision_from_observed_geography(
        projected, icp, company=_company(), company_quality=True,
        validated_geography_finding=claims["geography"],
    ) == "mismatch"
