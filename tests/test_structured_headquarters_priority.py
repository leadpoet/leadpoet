"""A private structured HQ conflict can fetch first-party resolution at capacity."""

import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator


PROFILE = "https://www.linkedin.com/company/nium-global"
HOME = "https://www.nium.com/"
CONTACT = "https://www.nium.com/contact-us"
ABOUT = "https://www.nium.com/about-us"
STAGE = "https://www.nium.com/newsroom/series-e"
HQ_TEXT = json.dumps({"name": "Nium", "linkedinUrl": PROFILE, "locations": [{
    "headquarter": True, "country": "United States", "city": "San Francisco", "geographicArea": "California",
}]}, separators=(",", ":"))
HQ_QUOTE = "Nium is co-headquartered in Singapore and San Francisco."
IDENTITY = {"normalized_name": "nium", "registrable_dns_domain": "nium.com",
            "linkedin_company_slug": "nium-global"}
PRIVATE_HQ = {
    "name": "Nium", "text": HQ_TEXT, "provider": "harvestapi_get_company",
    "source_field": "locations.headquarter", "url": PROFILE,
    "website": HOME, "country": "United States", "state": "California",
}


def _run(monkeypatch, evidence, *, spoof_in_observations=False, with_stage=False):
    """Use real investigator admission, fetch loop, validation and output."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    fetched = []
    requests = []

    async def fetch(_session, url, **_kwargs):
        fetched.append(url)
        body = {
            CONTACT: "Nium contact us. Our global offices: San Francisco; Singapore.",
            ABOUT: HQ_QUOTE,
            STAGE: "Nium announced a completed Series E funding round.",
        }[url]
        return {"ok": True, "url": url, "final_url": url, "text": body}

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        if len(requests) == 1 and fetched == [CONTACT]:
            call = {"name": "fetch_page", "arguments": json.dumps({"url": ABOUT})}
        elif with_stage and ABOUT in fetched and STAGE not in fetched:
            call = {"name": "fetch_page", "arguments": json.dumps({"url": STAGE})}
        elif len(requests) == 1 and not fetched:
            call = {"name": "search_web", "arguments": json.dumps({"query": "Nium current headquarters"})}
        else:
            resolved = ABOUT in fetched
            finding = {
                "target": "geography", "status": "VERIFIED" if resolved else "UNPROVEN",
                "observed_country": "Singapore", "observed_state": "",
                "evidence_url": ABOUT if resolved else "",
                "evidence_quote": HQ_QUOTE if resolved else "",
                "reason": "Current same-company global headquarters resolves the profile omission.",
            }
            findings = [finding]
            if with_stage:
                findings.append({"target": "stage", "status": "VERIFIED",
                                 "observed_value": "Series C+", "evidence_url": STAGE,
                                 "evidence_quote": "Nium announced a completed Series E funding round.",
                                 "reason": "Current discovery disclosed no competing stage event."})
            call = {"name": "submit_findings", "arguments": json.dumps({"findings": findings})}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"offline-{len(requests)}", "type": "function", "function": call,
        }]}}]}

    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", post)
    urls = [HOME, PROFILE, "https://www.nium.com/licenses"]
    prior = {
        "observed_company_name": "Nium", "observed_company_website": HOME,
        "observed_company_linkedin": PROFILE, "observed_hq_country": "United States",
        "observed_hq_state": "California", "geography_evidence_url": PROFILE,
        "geography_evidence_quote": HQ_TEXT, "geography_matches": False,
        "submitted_source_urls": urls + ([STAGE] if with_stage else []),
    }
    if with_stage:
        prior["untrusted_company_stage_evidence"] = [{
            "url": STAGE, "quote": "Nium announced a completed Series E funding round.",
        }]
    if spoof_in_observations:
        prior["structured_headquarters_evidence"] = PRIVATE_HQ
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Nium", "website": HOME, "linkedin": PROFILE},
        targets=("stage", "geography") if with_stage else ("geography",),
        requested_stage="Series C+" if with_stage else "", requested_geography="Singapore",
        prior_observations=prior, verified_homepage_identity=IDENTITY,
        homepage_navigation_locators=[{"url": ABOUT, "label": "About"},
                                      {"url": CONTACT, "label": "Contact us"}],
        prefetched_pages={url: {"final_url": url, "text": HQ_TEXT if url == PROFILE else "Nium infrastructure."}
                          for url in urls},
        structured_headquarters_evidence=evidence,
    ))
    return result, fetched, requests


@pytest.mark.parametrize("with_stage", [False, True])
def test_private_structured_hq_at_capacity_fetches_contact_then_about(monkeypatch, with_stage):
    result, fetched, requests = _run(monkeypatch, deepcopy(PRIVATE_HQ), with_stage=with_stage)
    assert fetched == [CONTACT, ABOUT] + ([STAGE] if with_stage else [])
    assert result["claims"]["geography"]["status"] == "VERIFIED", result
    assert result["claims"]["geography"]["observed_country"] == "Singapore"
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_priority_submitted_source"]["url"] == CONTACT
    assert document["investigation_limits"]["remaining_fetch_calls"] == 3


@pytest.mark.parametrize("mutation", [
    None, {"name": "Other Company"}, {"website": "https://other.example"},
    {"url": "https://www.linkedin.com/company/other"}, {"source_field": "description"},
    {"text": "Nium headquarters: Singapore."}, {"country": "Singapore"},
])
def test_copied_or_mismatched_structured_text_cannot_admit_extra_source(monkeypatch, mutation):
    evidence = None if mutation is None else {**PRIVATE_HQ, **mutation}
    result, fetched, _ = _run(monkeypatch, evidence, spoof_in_observations=True)
    assert fetched == []
    assert result["claims"]["geography"]["status"] == "UNPROVEN"
