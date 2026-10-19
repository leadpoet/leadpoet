"""Lexical locator ties retain the search provider's relevance order."""

from itertools import permutations
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import intent_verification_three_stage as verifier
from tests.test_intent_unavailable_same_event_source import CLAIM, ORIGINAL


# Public URL/title fields, in the exact order returned by the captured live
# search. They are locators and do not establish the partner claim or ICP fit.
CAPTURED_LOCATORS = [
    {"url": "https://echelon.e27.co/sponsors-and-exhibitors/",
     "title": "e27 - Echelon - Tech Conference - Singapore 2026"},
    {"url": "https://echelon.e27.co/speakers/",
     "title": "Speakers - Echelon Singapore 2026"},
    {"url": "https://www.linkedin.com/posts/e27-optimatic-_echelonsg-echelonsg2026-aiworkflowcompetition-activity-7459863290916093952-eUqc",
     "title": "As part of the AI Workflow Competition at Echelon Singapore, our sponsor partners will be running a series of hands-on AI workshops. These sessions are free to attend, and we’re opening them …"},
]


async def select(monkeypatch, locators):
    monkeypatch.setenv("EXA_API_KEY", "unit-test")
    search = AsyncMock(return_value={"results": list(locators)})
    monkeypatch.setattr(investigator, "_search_web", search)
    result = await verifier._find_same_event_source({
        "company": "pixverse.ai", "claim": CLAIM,
        "signal_date": "2026-05-20", "claimed_source_urls": [ORIGINAL],
    })
    assert search.await_count == 1
    return result


@pytest.mark.asyncio
async def test_exact_captured_response_retains_provider_top_locator(monkeypatch):
    result = await select(monkeypatch, CAPTURED_LOCATORS)
    assert result["selected_urls"] == [CAPTURED_LOCATORS[0]["url"]]
    assert result["reason"] == "bounded_same_event_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("locators", list(permutations(CAPTURED_LOCATORS)))
async def test_equal_overlap_retains_each_provider_order(monkeypatch, locators):
    claim_tokens = verifier._link_tokens(CLAIM) | verifier._link_tokens("pixverse.ai")
    assert {len(claim_tokens & verifier._link_tokens(item["title"] + " " + item["url"]))
            for item in locators} == {3}
    result = await select(monkeypatch, locators)
    assert result["selected_urls"] == [locators[0]["url"]]


@pytest.mark.asyncio
async def test_stronger_overlap_still_ranks_ahead_of_provider_order(monkeypatch):
    stronger = {"url": "https://organizer.example/partner-announcement", "title": CLAIM}
    result = await select(monkeypatch, [CAPTURED_LOCATORS[0], stronger])
    assert result["selected_urls"] == [stronger["url"]]


@pytest.mark.asyncio
async def test_original_and_invalid_urls_cannot_win_by_overlap(monkeypatch):
    result = await select(monkeypatch, [
        {"url": ORIGINAL, "title": CLAIM},
        {"url": "javascript:invalid", "title": CLAIM},
        {"url": "not an absolute URL", "title": CLAIM},
        CAPTURED_LOCATORS[0], CAPTURED_LOCATORS[1],
    ])
    assert result["selected_urls"] == [CAPTURED_LOCATORS[0]["url"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("locators", [[], [{"url": "https://unrelated.example/article",
                                          "title": "Unrelated news"}]])
async def test_empty_or_zero_overlap_remains_unproven(monkeypatch, locators):
    result = await select(monkeypatch, locators)
    assert result == {"status": "unproven", "reason": "no_same_event_locator"}
