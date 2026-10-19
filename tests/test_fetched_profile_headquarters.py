"""Reuse exact profile HQ evidence without relaxing geography or adding calls."""

import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from lab_arena import operations
from tests.test_company_evidence_investigator import (
    _company, _complete_verdict, _finding, _icp,
)

URL = "https://www.linkedin.com/company/acme"
IDENTITY = {
    "normalized_name": "acme", "registrable_dns_domain": "acme.example",
    "linkedin_company_slug": "acme",
}
ANCHOR = {
    "verified_name": "acme", "verified_domain": "acme.example",
    "verified_linkedin_slug": "acme",
}
HQ = "Headquarters \n\n New York, NY"
BODY = (
    "Acme | LinkedIn\n## About us\nWebsite\nhttps://acme.example/\n"
    "Company size\n11-50 employees\n" + HQ +
    "\nType\nPrivately Held\n## Locations\nPrimary\nNew York, NY, US"
)


def _profile_finding(*, body=BODY, quote=HQ, url=URL, final_url=URL,
                     anchor=ANCHOR, country="United States", state="New York"):
    return investigator._validated_findings(
        {"findings": [_finding(
            "geography", observed_value=None, observed_country=country,
            observed_state=state, evidence_url=url, evidence_quote=quote,
        )]}, targets=("geography",), fetched_pages={url: body},
        fetched_final_urls={url: final_url}, first_party_domains={"acme.example"},
        identity_names={"acme"}, identity_anchor=anchor,
    )["geography"]


def test_exact_bound_profile_hq_reaches_existing_region_gate():
    finding = _profile_finding()
    assert finding["status"] == "VERIFIED"
    company = _company().model_copy(update={"state": "New York"})
    icp = _icp(geography="United States, Northeast")
    projected = lead_scorer._project_investigator_geography(
        _complete_verdict(observed_hq_country="", observed_hq_state=""),
        finding, icp=icp, company=company, company_quality=True,
    )
    assert lead_scorer._decision_from_observed_geography(
        projected, icp, company=company, company_quality=True,
        validated_geography_finding=finding,
    ) == "match"
    assert lead_scorer._decision_from_observed_geography(
        projected, _icp(geography="United States, South"), company=company,
        company_quality=True, validated_geography_finding=finding,
    ) == "mismatch"


@pytest.mark.parametrize("change", [
    {"url": "https://www.linkedin.com/company/other"},
    {"final_url": "https://www.linkedin.com/company/other"},
    {"final_url": ""},
    {"url": "https://www.linkedin.com/posts/acme"},
    {"anchor": {**ANCHOR, "verified_linkedin_slug": "other"}},
    {"anchor": {**ANCHOR, "verified_domain": "other.example"}},
    {"anchor": {**ANCHOR, "verified_name": ""}},
    {"body": BODY.replace("https://acme.example/", "https://other.example/")},
    {"body": BODY.replace("Acme | LinkedIn", "Other | LinkedIn").replace(
        "Website\nhttps://acme.example/\n", "")},
    {"body": BODY.replace("Acme | LinkedIn", "Parent | LinkedIn").replace(
        "https://acme.example/", "https://parent.example/")},
    {"body": "Sign in | LinkedIn\nEmail or phone\nPassword\n" + BODY},
    {"body": BODY.replace(HQ, "Mailing address\nNew York, NY")},
    {"body": BODY.replace(HQ, "Branch office\nNew York, NY")},
    {"body": BODY.replace(HQ, "Branch headquarters\nNew York, NY")},
    {"body": BODY.replace(HQ, "Locations\nNew York, NY")},
    {"body": BODY.replace(HQ, "") + "\n## Updates\n" + HQ},
    {"body": BODY.replace(HQ, HQ + "\nHeadquarters\nBoston, Massachusetts")},
    {"body": BODY.replace(HQ, "Headquarters\nParis"), "quote": "Headquarters\nParis"},
    {"country": ""},
    {"state": ""},
    {"state": "California"},
    {"country": "Canada"},
    {"quote": "Headquarters New York, NY, United States"},
])
def test_profile_hq_keeps_identity_source_and_location_guards(change):
    assert _profile_finding(**change)["status"] == "UNPROVEN"


def test_loaded_profile_hq_can_include_name_but_not_borrow_another_location():
    assert _profile_finding(quote=BODY[:BODY.index("\nType")])["status"] == "VERIFIED"
    body = BODY + "\n## Updates\nAcme customer headquarters in California, United States"
    assert _profile_finding(body=body, quote=body, state="California")["status"] == "UNPROVEN"


@pytest.mark.parametrize("targeted", [True, False])
def test_profile_prefetch_keeps_existing_page_cap_and_is_geography_scoped(targeted):
    urls = [URL] + [f"https://acme.example/{n}" for n in range(2)]
    pages = lead_scorer._investigator_prefetched_pages(
        {}, urls, verified_homepage_pages={
            u: {"final_url": u, "text": "Acme software."} for u in urls[1:]
        }, linkedin_profile_source={"url": URL, "text": BODY},
        verified_identity=IDENTITY, include_structured_headquarters=targeted,
    )
    assert len(pages) == (investigator.MAX_PREFETCHED_PAGES if targeted else 2)
    assert (URL in pages) is targeted
    if targeted:
        assert pages[URL] == {"final_url": URL, "text": BODY}
        assert pages[urls[1]]["text"] == "Acme software."


def test_raw_profile_does_not_displace_existing_first_party_headquarters():
    urls = ["https://acme.example/", "https://acme.example/pricing", "https://acme.example/contact"]
    hq = "Acme headquarters: 123 Main St, Raleigh, North Carolina, United States."
    existing = {u: {"final_url": u, "text": hq if u == urls[-1] else "Acme software."} for u in urls}
    assert lead_scorer._investigator_prefetched_pages(
        {}, [*urls, URL], verified_homepage_pages=existing,
        linkedin_profile_source={"url": URL, "text": BODY},
        verified_identity=IDENTITY, include_structured_headquarters=True,
    ) == existing


@pytest.mark.parametrize("gap", ["\n", "\n\n\n\n\n", "\n \n\t\n"])
def test_profile_headquarters_field_accepts_whitespace_and_markdown(gap):
    quote = "**Headquarters**" + gap + "New York, NY"
    assert _profile_finding(body=BODY.replace(HQ, quote), quote=quote)["status"] == "VERIFIED"


@pytest.mark.parametrize("profile", [
    {"url": "https://www.linkedin.com/company/other", "text": BODY},
    {"url": URL, "text": BODY.replace("acme.example", "other.example")},
    {"url": URL, "text": "Sign in | LinkedIn\nEmail or phone\nPassword\n" + BODY},
])
def test_invalid_profile_body_does_not_enter_prefetched_context(profile):
    assert lead_scorer._investigator_prefetched_pages(
        {}, [profile["url"]], linkedin_profile_source=profile,
        verified_identity=IDENTITY, include_structured_headquarters=True,
    ) == {}


def test_existing_targeted_review_receives_retained_profile_without_research(monkeypatch):
    captured = {}

    async def investigate(**kwargs):
        captured.update(kwargs)
        pages = kwargs["prefetched_pages"]
        claims = investigator._validated_findings(
            {"findings": [_finding(
                "geography", observed_value=None, observed_country="United States",
                observed_state="New York", evidence_url=URL, evidence_quote=HQ,
            )]}, targets=("geography",),
            fetched_pages={u: p["text"] for u, p in pages.items()},
            fetched_final_urls={u: p["final_url"] for u, p in pages.items()},
            first_party_domains={"acme.example"}, identity_anchor=ANCHOR,
        )
        return {"claims": claims, "_completed_submit": True, "failure_reason": ""}

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    company = _company().model_copy(update={"state": "New York"})
    projected, _, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=company, icp=_icp(geography="United States, Northeast"),
            verdict=_complete_verdict(observed_hq_country="", observed_hq_state=""),
            investigation_targets=("geography",), icp_attribute="", icp_stage="",
            verified_identity=IDENTITY, verified_transport_domain="acme.example",
            structured_employee_size_evidence=None, structured_public_company_evidence=None,
            employee_size_conflict=False, company_quality=True,
            linkedin_profile_source={"url": URL, "text": BODY},
        )
    )
    assert captured["prefetched_pages"][URL]["text"] == BODY
    assert URL in captured["prior_observations"]["submitted_source_urls"]
    assert claims["geography"]["status"] == "VERIFIED"
    assert projected["observed_hq_state"] == "New York"
    assert projected["observed_employee_count"] == "11-50"


def test_retained_profile_keeps_existing_first_party_hq_priority_fetch(monkeypatch):
    contact = "https://acme.example/contact"
    first_party_hq = "Acme headquarters: Raleigh, North Carolina, United States."
    fetched = []
    model_inputs = []

    async def fetch(_session, url):
        fetched.append(url)
        assert url == contact
        return {"ok": True, "url": url, "final_url": url, "text": first_party_hq}

    async def model(_session, _url, *, payload, **_kwargs):
        operations.validate_operation_request("openrouter.chat", payload)
        model_inputs.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit", "type": "function", "function": {
                "name": "submit_findings", "arguments": json.dumps({"findings": [_finding(
                    "geography", observed_country="United States",
                    observed_state="North Carolina", evidence_url=contact,
                    evidence_quote=first_party_hq,
                )]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", model)
    homepage = "https://acme.example/"
    stage = "https://acme.example/blog/series-a"
    prefetched = {
        homepage: {"final_url": homepage, "text": "Acme software."},
        stage: {"final_url": stage, "text": "Acme Series A."},
        URL: {"final_url": URL, "text": BODY},
    }
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": homepage, "linkedin": URL},
        targets=("geography",), requested_geography="United States, South",
        verified_homepage_identity=IDENTITY,
        homepage_navigation_locators=[{"url": contact, "label": "Contact"}],
        prior_observations={"submitted_source_urls": [*prefetched, contact]},
        prefetched_pages=prefetched,
    ))
    assert fetched == [contact]
    assert len(model_inputs) == 1
    first_content = next(m["content"] for m in model_inputs[0]["messages"] if m["role"] == "user")
    first_doc = json.loads(first_content[first_content.index("{"):])
    assert {p["url"] for p in first_doc["prefetched_sources"]} == {*prefetched, contact}
    assert result["claims"]["geography"]["status"] == "VERIFIED"
    assert result["claims"]["geography"]["observed_state"] == "North Carolina"
    assert result["usage"]["prefetched_pages"] == 3
    assert result["usage"]["fetch_calls"] == 1
    assert result["usage"]["search_calls"] == 0
