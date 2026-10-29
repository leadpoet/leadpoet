"""LinkedIn source priority through real company scoring and schema repair."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring import linkedin_company_size as profile
from qualification.scoring.company_fit_decision import company_fit_match
from gateway.qualification.models import CompanyOutput, ICPPrompt

URL = "https://www.linkedin.com/company/quaise-energy"
ANCHOR = {"normalized_name": "Quaise Energy", "registrable_dns_domain": "quaise.com",
          "linkedin_company_slug": "quaise-energy"}
CAPTURE = json.loads((Path(__file__).parent / "fixtures/quaise_structured_hq_profiles.json").read_text())[1]["fixture"]


def company():
    return CompanyOutput(company_name="Quaise Energy", company_website="https://quaise.com/",
        company_linkedin=URL, employee_count="51-200", industry="Energy", country="United States",
        state="Texas", intent_signals=[{"description": "Quaise launched a project.", "source": "news",
        "url": "https://quaise.com/news", "date": "2026-09-01", "snippet": "Quaise launched a project."}])


def icp():
    return ICPPrompt(icp_id="linkedin-first", industry="Energy", employee_count="51-200",
        country="United States", geography="United States", company_stage="", sub_industry="", product_service="")


def verdict():
    return {"observed_company_name": "Quaise Energy", "observed_company_website": "https://quaise.com/",
        "observed_company_linkedin": URL, "observed_employee_count": "51-200", "employee_size_matches": True,
        "employee_size_evidence_url": "https://quaise.com/about", "employee_size_evidence_quote": "Quaise Energy has 51-200 employees.",
        "observed_industry": "Energy", "observed_subindustry": "", "industry_matches": True,
        "industry_activity_role": "supplier_operator", "industry_evidence_url": "https://quaise.com/",
        "industry_evidence_quote": "Quaise Energy develops geothermal power plants.",
        "observed_hq_country": "United States", "observed_hq_state": "Texas", "geography_matches": True,
        "geography_evidence_url": "https://quaise.com/contact", "geography_evidence_quote": "Quaise Energy headquarters: Houston, Texas, United States.",
        "reason": "verified"}


def install_profile(monkeypatch, body, *, failure=""):
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
            return Response()

    async def structured(domain, url, **kwargs):
        calls.append((domain, url))
        if failure:
            kwargs["diagnostic"][profile.VERIFIER_FAILURE_REASON_KEY] = failure
            return None
        return await profile.fetch_structured_linkedin_company_size(domain, url, **kwargs)

    monkeypatch.setenv("DEEPLINE_API_KEY", "offline-placeholder")
    monkeypatch.setattr(profile.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    return calls


def run(monkeypatch, first, *, second=None, raw=None, raw_size=None, homepage_slug=True, investigator=True, reviewable_hq=False):
    observations = [deepcopy(first)] + ([deepcopy(second)] if second else [])
    provider = AsyncMock(side_effect=[(value, "") for value in observations])
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", provider)
    raw_calls = []

    async def current(url, **kwargs):
        raw_calls.append(url)
        if raw:
            kwargs["source_text_sink"].update({"url": URL, "text": raw})
        if raw_size:
            return {"url": URL, "employee_count": raw_size,
                    "quote": "Company size\n" + raw_size + " employees"}
        return {"outcome": "insufficient_evidence", "url": url}

    monkeypatch.setattr(scorer, "fetch_current_linkedin_company_size", current)
    review = (AsyncMock(return_value={"claims": {"geography": {
        "target": "geography", "status": "UNPROVEN"}}, "_completed_submit": True,
        "failure_reason": ""}) if reviewable_hq else AsyncMock(
            side_effect=AssertionError("A verified LinkedIn decision must not reopen")))
    monkeypatch.setattr(scorer, "investigate_company_evidence", review)
    homepage = company_fit_match("verified homepage", details={"identity": {
        "decision": "match", "evidence_source": "company_homepage", "observed_name": "Quaise Energy",
        "observed_domain": "quaise.com", "observed_linkedin_slug": "quaise-energy" if homepage_slug else ""},
        "verified_homepage_transport_domain": "quaise.com"})
    result = asyncio.run(scorer._llm_reverify_company(company(), icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=homepage, company_quality=True, evidence_investigator=investigator,
        verified_homepage_navigation_locators=({"url": "https://quaise.com/contact", "label": "Contact"},)))
    return result, provider, raw_calls, review


@pytest.mark.parametrize("size,country,expected_size,expected_hq", [
    ("51-200", "United States", "match", "match"),
    ("11-50", "United States", "mismatch", "match"),
    ("51-200", "Germany", "match", "mismatch"),
    ("11-50", "Germany", "mismatch", "mismatch"),
])
def test_verified_linkedin_values_survive_unproven_firstparty_conflict_review(
    monkeypatch, size, country, expected_size, expected_hq,
):
    body = deepcopy(CAPTURE)
    start, end = (51, 200) if size == "51-200" else (11, 50)
    body["element"]["employeeCountRange"] = {"start": start, "end": end}
    if country == "Germany":
        body["element"]["locations"] = [{"headquarter": True, "country": "DE", "city": "Berlin"}]
    calls = install_profile(monkeypatch, body)
    observation = verdict()
    observation.update(observed_employee_count="11-50" if size == "51-200" else "51-200", employee_size_matches=False)
    if country == "United States":
        observation.update(observed_hq_country="Germany", observed_hq_state="", geography_matches=False,
            geography_evidence_quote="Quaise Energy headquarters: Berlin, Germany.")
    result, provider, raw_calls, review = run(monkeypatch, observation, reviewable_hq=country == "Germany")
    assert result.details["dimension_decisions"] == {"employee_size": expected_size, "industry": "match", "geography": expected_hq, "stage": "match"}
    assert result.details["dimension_evidence"]["employee_size"]["url"] == URL
    assert result.details["dimension_evidence"]["geography"]["url"] == URL
    assert result.details["dimension_evidence"]["industry"]["url"] == observation["industry_evidence_url"]
    assert calls == [("quaise.com", URL)]
    assert len(raw_calls) <= 2
    assert provider.await_count == 1
    if country == "Germany":
        review.assert_awaited_once()
        assert review.await_args.kwargs["targets"] == ("geography",)
    else:
        review.assert_not_awaited()


@pytest.mark.parametrize("missing", ["employeeCountRange", "locations"])
def test_size_and_hq_fallback_independently(monkeypatch, missing):
    body = deepcopy(CAPTURE)
    body["element"].pop(missing)
    calls = install_profile(monkeypatch, body)
    result, *_ = run(monkeypatch, verdict(), investigator=False)
    evidence = result.details["dimension_evidence"]
    fallback, primary = ("employee_size", "geography") if missing == "employeeCountRange" else ("geography", "employee_size")
    assert evidence[fallback]["url"] == verdict()[fallback + "_evidence_url"]
    assert evidence[primary]["url"] == URL
    assert result.decision == "match"
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [profile.PROVIDER_ERROR_FAILURE_REASON, profile.MALFORMED_RESPONSE_FAILURE_REASON, profile.SOURCE_BLOCKED_FAILURE_REASON])
def test_profile_failures_allow_complete_web_without_false_mismatch(monkeypatch, failure):
    calls = install_profile(monkeypatch, CAPTURE, failure=failure)
    result, *_ = run(monkeypatch, verdict(), investigator=False)
    assert result.decision == "match"
    assert result.details["dimension_evidence"]["employee_size"]["url"] == verdict()["employee_size_evidence_url"]
    assert result.details["dimension_evidence"]["geography"]["url"] == verdict()["geography_evidence_url"]
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("name", "Other Company"), ("website", "https://other.example"), ("linkedinUrl", "https://www.linkedin.com/company/other")])
def test_wrong_entity_profile_is_never_authority(monkeypatch, field, value):
    body = deepcopy(CAPTURE)
    body["element"][field] = value
    install_profile(monkeypatch, body)
    result, *_ = run(monkeypatch, verdict(), investigator=False)
    assert result.decision == "match"
    # Preserve the existing size identity contract: exact slug+domain can
    # survive a renamed profile; HQ additionally requires a bound name.
    if field != "name":
        assert result.details["dimension_evidence"]["employee_size"]["url"] != URL
    assert result.details["dimension_evidence"]["geography"]["url"] != URL


@pytest.mark.parametrize("size,flag,expected", [("51-200", False, "match"), ("11-50", True, "mismatch"), (80, None, "match"), (True, True, "unavailable"), ("about 80", True, "unavailable")])
def test_numerical_comparison_ignores_model_boolean(size, flag, expected):
    assert scorer._decision_from_observed_employee_size({"observed_employee_count": size, "employee_size_matches": flag}, icp()) == expected


def test_schema_repair_reuses_profile_and_preserves_linkedin_authority(monkeypatch):
    calls = install_profile(monkeypatch, CAPTURE)
    first, second = verdict(), verdict()
    first["industry_matches"] = None
    second.update(observed_employee_count="11-50", employee_size_matches=False,
        observed_hq_country="Germany", observed_hq_state="", geography_matches=False,
        geography_evidence_quote="Quaise Energy headquarters: Berlin, Germany.")
    result, provider, raw_calls, _ = run(monkeypatch, first, second=second, investigator=False)
    assert result.decision == "match"
    assert provider.await_count == 2
    assert len(calls) == 1 and len(raw_calls) == 2
    assert result.details["dimension_evidence"]["employee_size"]["url"] == URL
    assert result.details["dimension_evidence"]["geography"]["url"] == URL


def test_raw_only_size_survives_repair_with_web_derived_profile_anchor(monkeypatch):
    install_profile(monkeypatch, CAPTURE, failure=profile.SOURCE_BLOCKED_FAILURE_REASON)
    first, second = verdict(), verdict()
    first["industry_matches"] = None
    second.update(observed_employee_count="11-50", employee_size_matches=False)
    result, provider, raw_calls, _ = run(monkeypatch, first, second=second,
        raw_size="51-200", homepage_slug=False, investigator=False)
    assert provider.await_count == 2
    assert len(raw_calls) == 1
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.details["dimension_evidence"]["employee_size"]["url"] == URL


@pytest.mark.parametrize("field,expected", [
    ("## About\nHeadquarters\nHouston, Texas, United States", True),
    ("## About\nHeadquarters\nBerlin, Germany", True),
    ("## About\nLocations\nHouston, Texas, United States (HQ)", False),
    ("Houston, Texas, United States", False),
    ("## About\nHeadquarters\nContact us", False),
    ("## About\nHeadquarters\nIndianapolis", False),
    ("## About\nHeadquarters\nGeorgia", False),
    ("## About\nHeadquarters\nTbilisi, Georgia", False),
    ("## About\nHeadquarters\nPacific Northwest", False),
])
def test_raw_about_hq_uses_existing_identity_and_explicit_field_guards(field, expected):
    cache = {}
    observation = verdict()
    raw = "# Quaise Energy\nWebsite\nhttps://quaise.com/\n" + field
    projected = scorer._project_verified_linkedin_headquarters(observation, verified_identity=ANCHOR,
        invocation_cache=cache, linkedin_profile_source={"url": URL, "text": raw}, icp=icp(), company=company(), company_quality=True)
    assert bool(cache.get("linkedin_headquarters_verified")) is expected
    assert (projected["geography_evidence_url"] == URL) is expected


@pytest.mark.parametrize("name,website,url", [
    ("Other Entity", "", URL),
    ("Quaise Energy", "https://other.example/", URL),
    ("Quaise Energy", "https://quaise.com/", "https://www.linkedin.com/company/other"),
])
def test_raw_hq_never_uses_an_unbound_company(name, website, url):
    raw = f"# {name}\n## About\n" + (f"Website\n{website}\n" if website else "") + "Headquarters\nHouston, Texas, United States"
    cache = {}
    observation = verdict()
    assert scorer._project_verified_linkedin_headquarters(observation, verified_identity=ANCHOR,
        invocation_cache=cache, linkedin_profile_source={"url": url, "text": raw}, icp=icp(),
        company=company(), company_quality=True) == observation
    assert not cache.get("linkedin_headquarters_verified")


def test_missing_all_size_proof_with_provider_failure_remains_unavailable(monkeypatch):
    install_profile(monkeypatch, CAPTURE, failure=profile.PROVIDER_ERROR_FAILURE_REASON)
    initial = verdict()
    initial.update(observed_employee_count=None, employee_size_matches=None,
        employee_size_evidence_url="", employee_size_evidence_quote="")
    result, *_ = run(monkeypatch, initial, second=initial, investigator=False)
    assert result.decision == "unavailable"
    assert result.details["dimension_decisions"]["employee_size"] == "unavailable"
    assert result.details["failure_class"] == scorer.EMPLOYEE_SIZE_VERIFICATION_FAILURE_CLASS


def test_bound_hq_fact_does_not_decide_custom_locality_semantics():
    body = deepcopy(CAPTURE)
    hq = profile.project_structured_linkedin_company_headquarters("quaise.com", URL, "Quaise Energy", body)
    request = icp().model_copy(update={"geography": "San Francisco Bay Area"})
    cache = {"structured_profile_headquarters_evidence": hq}
    projected = scorer._project_verified_linkedin_headquarters(verdict(), verified_identity=ANCHOR,
        invocation_cache=cache, linkedin_profile_source={}, icp=request,
        company=company(), company_quality=True)
    assert projected["observed_hq_state"] == "Texas"
    assert projected["geography_matches"] is None
    assert scorer._decision_from_observed_geography(projected, request,
        company=company(), company_quality=True) == "unavailable"
    web_review = {"target": "geography", "status": "VERIFIED",
        "evidence_url": "https://quaise.com/contact", "evidence_quote": verdict()["geography_evidence_quote"],
        "observed_country": "United States", "observed_state": "Texas"}
    still_unresolved = scorer._project_verified_linkedin_headquarters(projected, verified_identity=ANCHOR,
        invocation_cache=cache, linkedin_profile_source={}, icp=request,
        company=company(), company_quality=True, validated_geography_finding=web_review)
    assert still_unresolved["geography_matches"] is None
    bound_review = {**web_review, "status": "CONTRADICTED", "evidence_url": URL, "evidence_quote": hq["text"]}
    resolved = scorer._project_verified_linkedin_headquarters(projected, verified_identity=ANCHOR,
        invocation_cache=cache, linkedin_profile_source={}, icp=request,
        company=company(), company_quality=True, validated_geography_finding=bound_review)
    assert scorer._decision_from_observed_geography(resolved, request, company=company(),
        company_quality=True, validated_geography_finding=bound_review) == "mismatch"


def test_semantic_review_cannot_switch_selected_structured_hq_to_conflicting_raw_hq():
    hq = profile.project_structured_linkedin_company_headquarters("quaise.com", URL, "Quaise Energy", CAPTURE)
    request = icp().model_copy(update={"geography": "San Francisco Bay Area"})
    raw = "# Quaise Energy\nWebsite\nhttps://quaise.com/\n## About\nHeadquarters\nSan Francisco, California, United States"
    review = {"target": "geography", "status": "VERIFIED", "evidence_url": URL,
        "evidence_quote": "Headquarters\nSan Francisco, California, United States",
        "observed_country": "United States", "observed_state": "California"}
    projected = scorer._project_verified_linkedin_headquarters(verdict(), verified_identity=ANCHOR,
        invocation_cache={"structured_profile_headquarters_evidence": hq},
        linkedin_profile_source={"url": URL, "text": raw}, icp=request,
        company=company(), company_quality=True, validated_geography_finding=review)
    assert projected["observed_hq_state"] == "Texas"
    assert projected["geography_matches"] is None


@pytest.mark.parametrize("web_country,expected_source", [("United States", "web"), ("Germany", "linkedin")])
def test_missing_linkedin_us_state_accepts_only_country_compatible_web_fallback(web_country, expected_source):
    body = deepcopy(CAPTURE)
    for location in body["element"]["locations"]:
        if location.get("headquarter") is True:
            location.pop("geographicArea", None)
            location.get("parsed", {}).pop("state", None)
    hq = profile.project_structured_linkedin_company_headquarters("quaise.com", URL, "Quaise Energy", body)
    assert hq["country"] == "United States" and hq["state"] == ""
    observation = verdict()
    if web_country == "Germany":
        observation.update(observed_hq_country="Germany", observed_hq_state="", geography_matches=False,
            geography_evidence_quote="Quaise Energy headquarters: Berlin, Germany.")
    projected = scorer._project_verified_linkedin_headquarters(observation, verified_identity=ANCHOR,
        invocation_cache={"structured_profile_headquarters_evidence": hq},
        linkedin_profile_source={}, icp=icp(), company=company(), company_quality=True)
    if expected_source == "web":
        assert projected == observation
        assert scorer._decision_from_observed_geography(projected, icp(), company=company(), company_quality=True) == "match"
    else:
        assert projected["geography_evidence_url"] == URL
        assert projected["observed_hq_country"] == "United States"
        assert projected["observed_hq_state"] == ""


def test_proven_foreign_linkedin_hq_mismatch_does_not_need_a_us_state():
    body = deepcopy(CAPTURE)
    body["element"]["locations"] = [{"headquarter": True, "country": "DE", "city": "Berlin"}]
    hq = profile.project_structured_linkedin_company_headquarters("quaise.com", URL, "Quaise Energy", body)
    observation = verdict()
    observation.update(observed_hq_state="California", geography_evidence_quote="Quaise Energy headquarters: San Francisco, California, United States.")
    request = icp().model_copy(update={"geography": "United States, West Coast"})
    candidate = company().model_copy(update={"state": "California"})
    projected = scorer._project_verified_linkedin_headquarters(observation, verified_identity=ANCHOR,
        invocation_cache={"structured_profile_headquarters_evidence": hq},
        linkedin_profile_source={}, icp=request, company=candidate, company_quality=True)
    assert projected["observed_hq_country"] == "Germany"
    assert projected["observed_hq_state"] == ""
    assert projected["geography_evidence_url"] == URL
    # Literal structured HQ JSON is authority through its validated private
    # finding, not through a model-authored headquarters sentence.
    bound_finding = {"target": "geography", "status": "VERIFIED",
        "evidence_url": URL, "evidence_quote": hq["text"],
        "observed_country": hq["country"], "observed_state": hq["state"]}
    assert scorer._decision_from_observed_geography(projected, request,
        company=candidate, company_quality=True,
        validated_geography_finding=bound_finding) == "mismatch"


@pytest.mark.parametrize("company_quality", [False, True])
@pytest.mark.parametrize("web_country", ["United States", "Germany"])
def test_partial_us_hq_fallback_for_state_region_is_country_compatible(company_quality, web_country):
    body = deepcopy(CAPTURE)
    for location in body["element"]["locations"]:
        if location.get("headquarter") is True:
            location.pop("geographicArea", None)
            location.get("parsed", {}).pop("state", None)
    hq = profile.project_structured_linkedin_company_headquarters("quaise.com", URL, "Quaise Energy", body)
    assert hq["country"] == "United States" and hq["state"] == ""
    request = icp().model_copy(update={"geography": "United States, West Coast"})
    candidate = company().model_copy(update={"state": "California"})
    observation = verdict()
    observation.update(observed_hq_country=web_country,
        observed_hq_state="California" if web_country == "United States" else "",
        geography_matches=web_country == "United States",
        geography_evidence_quote="Quaise Energy headquarters: San Francisco, California, United States." if web_country == "United States" else "Quaise Energy headquarters: Berlin, Germany.")
    projected = scorer._project_verified_linkedin_headquarters(observation, verified_identity=ANCHOR,
        invocation_cache={"structured_profile_headquarters_evidence": hq}, linkedin_profile_source={},
        icp=request, company=candidate, company_quality=company_quality)
    if web_country == "United States":
        assert projected == observation
    else:
        assert projected["observed_hq_country"] == "United States"
        assert projected["observed_hq_state"] == ""
        assert projected["geography_evidence_url"] == URL
        assert projected["geography_matches"] is None


def test_unknown_country_does_not_become_a_proven_foreign_hq():
    request = icp().model_copy(update={"geography": "United States, West Coast"})
    assert scorer._canonical_observed_geography_match(
        {"observed_hq_country": "Mars", "observed_hq_state": ""}, request,
        company_quality=False) is None


def test_structured_hq_cannot_use_a_hand_authored_replacement_sentence():
    hq = profile.project_structured_linkedin_company_headquarters(
        "quaise.com", URL, "Quaise Energy", CAPTURE)
    forged = {**hq, "state": "", "text": "Quaise Energy headquarters: Houston, United States."}
    cache = {"structured_profile_headquarters_evidence": forged}
    observation = verdict()
    projected = scorer._project_verified_linkedin_headquarters(
        observation, verified_identity=ANCHOR, invocation_cache=cache,
        linkedin_profile_source={}, icp=icp(), company=company(), company_quality=True)
    assert projected == observation
    assert not cache.get("linkedin_headquarters_verified")


REAL_COMPANIES = json.loads((Path(__file__).parent / "fixtures/linkedin_first_submitted_profiles.json").read_text())["companies"]


@pytest.mark.parametrize("retained", REAL_COMPANIES, ids=lambda row: row["company_name"])
def test_original_submitted_profile_facts_apply_without_replaying_history(monkeypatch, retained):
    source = retained["structured_profile"]
    name, url, website = source["name"], source["linkedinUrl"], source["website"]
    domain = scorer._registrable_domain(website)
    identity = {"normalized_name": name, "registrable_dns_domain": domain,
                "linkedin_company_slug": profile.linkedin_company_page_slug(url)}
    source_url = "https://www.linkedin.com/company/" + identity["linkedin_company_slug"]
    calls = install_profile(monkeypatch, {"status": 200, "element": source})
    raw = AsyncMock(return_value={"outcome": "insufficient_evidence", "url": source_url})
    monkeypatch.setattr(scorer, "fetch_current_linkedin_company_size", raw)
    observed = verdict()
    observed.update(observed_company_name=name, observed_company_website=website,
                    observed_company_linkedin=source_url)
    # Use retained old model observations to exercise the new source choice.
    # These are not claims that those web sources have been re-fetched today.
    selected = retained["stored_judge_selected_facts"][-1]["selected_fields"]
    observed.update({key: value for key, value in selected.items() if key.startswith(("employee_size_", "observed_employee_", "geography_", "observed_hq_"))})
    size = profile.project_structured_linkedin_company_size(domain, source_url, {"status": 200, "element": source})["employee_count"]
    if retained.get("independently_verified_web_conflict"):
        conflict = retained["independently_verified_web_conflict"]
        observed.update(observed_employee_count=conflict["web_bucket"],
            employee_size_evidence_url="https://www.akamai.com/company",
            employee_size_evidence_quote="11,500+ Employees worldwide")
        assert size != conflict["web_bucket"]
    request = icp().model_copy(update={"employee_count": size})
    company_model = company().model_copy(update={"company_name": name,
        "company_website": website, "company_linkedin": source_url})
    cache = {}
    refreshed = asyncio.run(scorer._refresh_linkedin_employee_size_observation(
        observed, company_model, request, verified_homepage_identity=identity, invocation_cache=cache))
    assert scorer._structured_employee_size_decision(cache["structured_evidence"], request) == "match"
    assert cache["structured_evidence"]["employee_count"] == size
    assert cache["linkedin_employee_size_verified"] is True
    refreshed = scorer._project_verified_linkedin_headquarters(refreshed, verified_identity=identity,
        invocation_cache=cache, linkedin_profile_source={}, icp=request,
        company=company_model, company_quality=False)
    if retained["company_name"] == "MongoDB":
        # Natural unusable HQ: retained city and parsed city differ. Existing
        # strict projection rejects it; web HQ remains independently usable.
        assert "structured_profile_headquarters_evidence" not in cache
        assert refreshed["geography_evidence_url"] == selected["geography_evidence_url"]
    else:
        assert cache["linkedin_headquarters_verified"] is True
        assert refreshed["geography_evidence_url"] == source_url
    assert calls == [(domain, source_url)]
