"""An unfinished investigation can retain source facts without a verdict."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import aiohttp
import pytest

from qualification.scoring import company_evidence_investigator as investigator, lead_scorer
from tests.test_company_evidence_late_final_submit import _investigate
from tests.test_company_evidence_judgment_reserve import (
    QUOTE, URL, _Clock, _finding, _response,
)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("fault,reason", [
    ("http", "provider_error"),
    ("timeout", "provider_error"),
    ("connection", "provider_error"),
    ("json", "malformed_response"),
    ("unexpected", "unexpected_verifier_error"),
])
def test_fresh_exact_page_survives_later_failure(monkeypatch, positive, fault, reason):
    clock = _Clock()
    fetches, requests, events = [], [], []

    async def fetch(_session, url, **_kwargs):
        fetches.append(url)
        clock.now = 111.0
        return {"ok": True, "url": url, "final_url": url, "text": QUOTE}

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if not positive and len(requests) == 1:
            return 200, _response("fetch_page", {"url": URL})
        if fault == "http":
            return 503, {"error": "offline"}
        if fault == "timeout":
            raise asyncio.TimeoutError()
        if fault == "connection":
            raise aiohttp.ClientConnectionError("offline fixture")
        if fault == "unexpected":
            raise AssertionError("offline fixture")
        return 200, _response("submit_findings", "not-json")

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(
        investigator._InvestigationToolCapture, "emit",
        lambda _self, _call, result, *_args, **_kwargs: events.append(deepcopy(result)),
    )
    result = _investigate(monkeypatch, post, clock, loaded=False, positive=positive)

    assert fetches == [URL]
    assert len(requests) == (1 if positive else 2)
    assert result["claims"] == {}
    assert result["failure_reason"] == reason
    assert result.get("_completed_submit") is not True
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        URL: {"final_url": URL, "text": QUOTE},
    }
    # The new handoff does not add source bodies to terminal tool capture.
    assert all(investigator.PRIVATE_FETCHED_PAGES_KEY not in event for event in events)
    source_cache = {URL: {"status": "source_unavailable"}}
    retained = {}
    lead_scorer._hydrate_required_attribute_source_cache(
        source_cache, result, successful_source_sink=retained,
    )
    assert source_cache[URL]["status"] == "fetched"
    assert source_cache[URL]["text"] == QUOTE
    assert retained[URL]["text"] == QUOTE
    assert result["claims"] == {}  # Source hydration does not finish judgment.
    if positive and fault in {"http", "json"}:
        _assert_caller_stays_unfinished_and_private(monkeypatch, result)


def _assert_caller_stays_unfinished_and_private(monkeypatch, investigation):
    from tests.test_company_evidence_investigator import _company, _complete_verdict, _icp

    async def investigated(**_kwargs):
        return deepcopy(investigation)

    cache = {URL: {"status": "source_unavailable"}}
    retained = {}
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigated)
    projected, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(), icp=_icp(), verdict=_complete_verdict(industry_matches=None),
            investigation_targets=("industry",), icp_attribute="", icp_stage="",
            verified_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example",
                               "linkedin_company_slug": "acme"},
            verified_transport_domain="acme.example", structured_employee_size_evidence=None,
            structured_public_company_evidence=None, employee_size_conflict=False,
            company_quality=True, required_attribute_source_cache=cache,
            successful_required_attribute_source_sink=retained,
        )
    )
    assert claims == {}
    assert projected["industry_matches"] is None
    assert result.decision == "unavailable"
    receipt = result.details["investigation_receipt"]
    assert receipt["failure_reason"] == investigation["failure_reason"]
    assert receipt["completed_submitted_findings"] is False
    assert investigator.PRIVATE_FETCHED_PAGES_KEY not in str(result.details)
    assert QUOTE not in str(result.details)
    assert cache[URL]["status"] == "fetched"
    assert retained[URL]["text"] == QUOTE


def test_failed_page_cannot_become_a_retained_success(monkeypatch):
    clock = _Clock()

    async def fetch(_session, url, **_kwargs):
        return {"ok": False, "url": url, "error": "source_not_found", "text": QUOTE}

    async def post(*_args, **_kwargs):
        return 503, {"error": "offline"}

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = _investigate(monkeypatch, post, clock, loaded=False, positive=True)
    assert result["claims"] == {}
    assert result["failure_reason"] == "provider_error"
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {}
    cache = {URL: {"status": "source_unavailable"}}
    lead_scorer._hydrate_required_attribute_source_cache(cache, result)
    assert cache == {URL: {"status": "source_unavailable"}}


@pytest.mark.parametrize("excess", ["none", "count", "text"])
def test_failure_handoff_keeps_existing_prefetch_bounds(monkeypatch, excess):
    pages = {
        f"https://acme.example/page-{index}": {
            "final_url": f"https://acme.example/page-{index}",
            "text": "x" * (investigator.MAX_PAGE_CHARACTERS + (100 if excess == "text" else 0)),
        }
        for index in range(investigator.MAX_PREFETCHED_PAGES + (excess == "count"))
    }
    expected, finals = investigator._validated_prefetched_pages(
        pages, submitted_source_urls=tuple(pages),
    )

    async def post(*_args, **_kwargs):
        return 503, {"error": "offline"}

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("required_attribute",), requested_attribute="Supplies software",
        prior_observations={"submitted_source_urls": list(pages)},
        prefetched_pages=pages,
    ))
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        url: {"final_url": finals[url], "text": text} for url, text in expected.items()
    }
    assert len(expected) <= investigator.MAX_PREFETCHED_PAGES
    assert all(len(text) <= investigator.MAX_PAGE_CHARACTERS for text in expected.values())
    assert bool(expected) is (excess == "none")


def test_failure_handoff_does_not_swallow_cancellation(monkeypatch):
    async def post(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    with pytest.raises(asyncio.CancelledError):
        _investigate(monkeypatch, post, _Clock())


def test_completed_submission_handoff_is_unchanged(monkeypatch):
    async def post(*_args, **_kwargs):
        return 200, _response("submit_findings", {"findings": [_finding()]})

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    result = _investigate(monkeypatch, post, _Clock())
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        URL: {"final_url": URL, "text": QUOTE},
    }


@pytest.mark.parametrize("deferred", [False, True])
def test_native_industry_advisory_failure_retains_loaded_pages(monkeypatch, deferred):
    reviews, searches = [], []

    async def review(*_args, **_kwargs):
        reviews.append(True)
        if not deferred or len(reviews) == 2:
            return {}, "provider_error", 1
        return {"_industry_review": {
            "decision": "unavailable", "source_urls": [],
            "missing_requirements": ["Company-bound source text for the requested activity"],
            "research_query": "enrollment software", "reason": "No loaded sources",
        }}, "", 0

    async def search(*_args, **_kwargs):
        searches.append(True)
        return {"_source_pages": {URL: QUOTE}}

    async def no_reasoning(*_args, **_kwargs):
        raise AssertionError("advisory failure must stop before reasoning")

    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "offline-test-model")
    monkeypatch.setattr(investigator, "_review_industry_sources", review)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = _investigate(monkeypatch, no_reasoning, _Clock(), loaded=not deferred)
    assert len(reviews) == (2 if deferred else 1)
    assert len(searches) == int(deferred)
    assert result["claims"] == {}
    assert result["failure_reason"] == "provider_error"
    assert result.get("_completed_submit") is not True
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        URL: {"final_url": URL, "text": QUOTE},
    }


def test_reasoning_fallthrough_retains_loaded_pages(monkeypatch):
    async def no_reasoning(*_args, **_kwargs):
        raise AssertionError("zero reasoning budget must not dispatch")

    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 0)
    result = _investigate(monkeypatch, no_reasoning, _Clock())
    assert result["claims"] == {}
    assert result["failure_reason"] == "malformed_response"
    assert result.get("_completed_submit") is not True
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == {
        URL: {"final_url": URL, "text": QUOTE},
    }


def test_missing_credentials_still_returns_before_source_initialization(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))
    assert result == {"claims": {}, "failure_reason": "provider_error"}
