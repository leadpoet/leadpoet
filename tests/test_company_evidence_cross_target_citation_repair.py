"""Public-source excerpts from the Oct 23 audit; no private run payloads.

Mock replies verify correction context and deterministic gates, not LLM accuracy.
"""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv
from tests.test_company_evidence_judgment_reserve import _Clock, _finding, _response


COHN_URL = (
    "https://www.cohnreznick.com/insights/"
    "cohnreznick-secures-strategic-growth-investment-from-apax-funds"
)
OWNER_URL = "https://www.consulting.us/news/11487/apax-buys-majority-stake-in-cohnreznick"
OWNER_QUOTE = (
    "Funds advised by private equity firm Apax Partners have acquired a majority "
    "stake in CohnReznick, a top-20 accounting and consulting firm."
)
COHN_QUOTE = (
    "CohnReznick LLP (“CohnReznick” or “the Firm”), one of the United States’ "
    "leading advisory, assurance, and tax firms"
)
WAYPOINT_URL = "https://www.waypointbio.com/technology"
WAYPOINT_QUOTE = (
    "Iterative optimization Utilizing AI to rationally design better cell therapies, "
    "faster → Design CAR-T constructs & in vivo delivery vectors Test 1000s of "
    "therapies in vivo Spatial pooled screening In situ genotyping + spatial "
    "biology Rank order designs with AI In-house screens"
)
WAYPOINT_TEXT = "Waypoint Bio | Technology " + WAYPOINT_QUOTE


def finding(target, url=COHN_URL, quote=COHN_QUOTE, **changes):
    return {
        **_finding(url=url, quote=quote), "target": target,
        "observed_industry": "Accounting and advisory services",
        "observed_subindustry": "Tax and assurance",
        **changes,
    }


def validate(raw, *, pages=None, identity=None):
    return inv._validated_findings(
        {"findings": [raw]}, targets=("required_attribute",),
        fetched_pages=pages or {COHN_URL: COHN_QUOTE, OWNER_URL: OWNER_QUOTE},
        fetched_final_urls={COHN_URL: COHN_URL, OWNER_URL: OWNER_URL},
        first_party_domains={"cohnreznick.com"}, identity_names={"cohnreznick"},
        identity_anchor=identity or {},
    )["required_attribute"]


@pytest.mark.parametrize("url,quote,expected", [
    (OWNER_URL, OWNER_QUOTE.replace("private equity firm ", ""),
     "supporting quote was not present in fetched source"),
    ("https://www.consulting.us/not-loaded", OWNER_QUOTE,
     "supporting source URL was not loaded"),
    (OWNER_URL, "Another Company provides accounting services.",
     "supporting quote did not identify the investigated company"),
    (OWNER_URL, "", "supporting evidence requires a valid URL and a nonempty exact quote"),
    (OWNER_URL, "x" * 2001, "supporting quote exceeds the 2000-character limit"),
])
def test_support_rejection_reports_the_actual_failed_check(url, quote, expected):
    pages = {COHN_URL: COHN_QUOTE, OWNER_URL: OWNER_QUOTE}
    if "Another Company" in quote:
        pages[OWNER_URL] = quote
    raw = finding("required_attribute", supporting_evidence_url_1=url,
                  supporting_evidence_quote_1=quote)
    result = validate(raw, pages=pages)
    assert result["status"] == "UNPROVEN"
    assert result["reason"] == expected
    assert result["evidence_url"] == result["evidence_quote"] == ""


def test_only_the_exact_loaded_ownership_quote_repairs_cohn_support():
    raw = finding(
        "required_attribute", supporting_evidence_url_2=OWNER_URL,
        supporting_evidence_quote_2=OWNER_QUOTE.replace("private equity firm ", ""),
    )
    assert validate(raw)["status"] == "UNPROVEN"
    repaired = validate({**raw, "supporting_evidence_quote_2": OWNER_QUOTE})
    assert repaired["status"] == "VERIFIED"
    assert repaired["supporting_evidence"] == [{"url": OWNER_URL, "quote": OWNER_QUOTE}]


def run(monkeypatch, *, name, domain, sources, targets, replies, attribute,
        stage="", turns=1, **options):
    requests = []

    async def post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(copy.deepcopy(payload))
        assert len(requests) <= len(replies)
        return 200, _response("submit_findings", {"findings": replies[len(requests) - 1]},
                              call=len(requests))

    async def search(*_args, **_kwargs):
        assert stage, "Citation-only repair must not add research"
        return {"results": []}

    async def no_fetch(*_args, **_kwargs):
        raise AssertionError("Citation-only repair must not fetch")

    slug = name.lower().replace(" ", "-")
    linkedin = "https://www.linkedin.com/company/" + slug
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("EXA_API_KEY", "test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_fetch_page", no_fetch)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "MAX_REASONING_TURNS", turns)
    monkeypatch.setattr(inv, "time", _Clock())
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": name, "website": "https://" + domain,
                         "linkedin": linkedin}, targets=targets,
        requested_stage=stage, requested_industry="Company's own products or services",
        requested_attribute=attribute, positive_semantic_review=True,
        prior_observations={"observed_company_name": name,
                            "observed_company_website": "https://" + domain,
                            "observed_company_linkedin": linkedin,
                            "submitted_source_urls": list(sources)},
        verified_homepage_identity={"normalized_name": name,
                                    "registrable_dns_domain": domain,
                                    "linkedin_company_slug": slug},
        prefetched_pages={url: {"final_url": url, "text": text}
                          for url, text in sources.items()}, **options,
    ))
    return result, requests


def feedback(request):
    return next(json.loads(m["content"]) for m in reversed(request["messages"])
                if m.get("role") == "tool")


def test_cohn_support_retry_carries_failed_pair_and_exact_cross_target_fact(monkeypatch):
    stage = finding("stage", OWNER_URL, OWNER_QUOTE, observed_value="Private Equity",
                    ownership={"control_status": "current_control",
                               "controlling_owner": "Apax Partners",
                               "owner_type": "private_equity"})
    industry = finding("industry")
    attribute = finding("required_attribute", supporting_evidence_url_2=OWNER_URL,
                        supporting_evidence_quote_2=OWNER_QUOTE.replace("private equity firm ", ""))
    repaired = {**attribute, "supporting_evidence_quote_2": OWNER_QUOTE}
    result, requests = run(
        monkeypatch, name="CohnReznick", domain="cohnreznick.com",
        sources={COHN_URL: COHN_QUOTE, OWNER_URL: OWNER_QUOTE},
        targets=("stage", "industry", "required_attribute"),
        replies=[[stage, industry, attribute], [repaired]], stage="Private Equity",
        attribute="Provides accounting, tax or advisory services AND is owned or backed by private equity.",
    )
    assert len(requests) == 2
    repair = feedback(requests[1])
    rejection = repair["rejected_findings"][0]
    assert rejection["reason"] == "supporting quote was not present in fetched source"
    assert rejection["supporting_evidence_slot"] == 2
    assert rejection["source_url"] == OWNER_URL
    assert OWNER_QUOTE in rejection["source_context"]
    assert {x["target"] for x in repair["untrusted_non_rejected_source_context"]} == {"stage", "industry"}
    assert "including stage and industry" in repair["instruction"]
    assert "up to two additional exact" in repair["instruction"]
    assert all(f["status"] == "VERIFIED" for f in result["claims"].values())
    assert result["_completed_submit"] is True
    assert result["usage"]["search_calls"] == 1  # existing current-stage search
    assert result["usage"]["fetch_calls"] == 0


@pytest.mark.parametrize("final_status", ["VERIFIED", "UNPROVEN"])
def test_waypoint_retry_uses_exact_inhouse_research_facts_without_forcing_acceptance(
    monkeypatch, final_status,
):
    industry = finding("industry", WAYPOINT_URL, WAYPOINT_QUOTE,
                       observed_industry="Cell therapy research")
    bad = finding("required_attribute", OWNER_URL,
                  "Waypoint develops cell therapies.")
    revised = finding("required_attribute", WAYPOINT_URL, WAYPOINT_QUOTE,
                      status=final_status)
    if final_status == "UNPROVEN":
        revised.update(evidence_url="", evidence_quote="",
                       reason="An independently required clinical milestone is not proved.")
    result, requests = run(
        monkeypatch, name="Waypoint Bio", domain="waypointbio.com",
        sources={WAYPOINT_URL: WAYPOINT_TEXT, OWNER_URL: bad["evidence_quote"]},
        targets=("industry", "required_attribute"), replies=[[industry, bad], [revised]],
        attribute="Develops therapeutic candidates AND operates R&D"
                  + (" AND has completed Phase 3" if final_status == "UNPROVEN" else ""),
    )
    assert len(requests) == 2
    repair = feedback(requests[1])
    assert "including stage and industry" in repair["instruction"]
    assert repair["untrusted_non_rejected_source_context"][0]["evidence_quote"] == WAYPOINT_QUOTE
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == final_status
    assert result["_completed_submit"] is True
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0


def test_waypoint_industry_quote_repair_keeps_attribute_review_in_existing_turn(monkeypatch):
    industry = finding("industry", WAYPOINT_URL, WAYPOINT_QUOTE,
                       observed_industry="Cell therapy research")
    bad_industry = {**industry, "evidence_quote": "Waypoint Bio develops cell therapies in-house."}
    missing_attribute = finding("required_attribute", status="UNPROVEN",
                                evidence_url="", evidence_quote="",
                                reason="A literal R&D label was not found.")
    attribute = finding("required_attribute", WAYPOINT_URL, WAYPOINT_QUOTE)
    result, requests = run(
        monkeypatch, name="Waypoint Bio", domain="waypointbio.com",
        sources={WAYPOINT_URL: WAYPOINT_TEXT},
        targets=("industry", "required_attribute"),
        replies=[[bad_industry, missing_attribute], [industry, attribute]], turns=3,
        attribute="Develops therapeutic candidates AND operates R&D",
    )
    assert len(requests) == 2
    repair = feedback(requests[1])
    assert [f["target"] for f in repair["rejected_findings"]] == ["industry"]
    assert "including stage and industry" in repair["instruction"]
    assert WAYPOINT_QUOTE in repair["rejected_findings"][0]["source_context"]
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["_completed_submit"] is True
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0
