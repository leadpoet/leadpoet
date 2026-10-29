"""Completed exact-source failures permit discovery, never an absence verdict."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import competition
from tests.test_intent_wayback_empty_body import SOURCE, SNAPSHOT, TIMESTAMP, ERROR_FRAME
from tests.test_intent_unavailable_same_event_source import (
    ORIGINAL, ALTERNATIVE, fetched_source, judgment, missing_source, run_verifier,
)


def install_exact_failures(monkeypatch, *, final_snapshot=SNAPSHOT, exa_documents=None, exa_http_status=200):
    """Saved native sequence: direct502, archive error body, two exact crawl errors."""
    calls = []
    documents = exa_documents or [{"results": [], "statuses": [{
        "id": SOURCE, "status": "error", "error": {
            "httpStatusCode": 500, "tag": "CRAWL_EMPTY_CONTENT",
        },
    }]}] * 2

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def get(self, url, **kwargs):
            calls.append(("GET", url))
            if url == "https://api.scrapingdog.com/scrape":
                assert kwargs["params"]["url"] == SOURCE
                return httpx.Response(502, text="source fetch failed", request=httpx.Request("GET", url))
            if url == "https://archive.org/wayback/available?url=" + SOURCE:
                return httpx.Response(200, json={"archived_snapshots": {"closest": {
                    "available": True, "status": "200", "timestamp": TIMESTAMP,
                    "url": "http" + SNAPSHOT[5:],
                }}}, request=httpx.Request("GET", url))
            assert url == SNAPSHOT
            return httpx.Response(200, text=ERROR_FRAME, request=httpx.Request("GET", final_snapshot))
        async def post(self, url, **kwargs):
            calls.append(("POST", url))
            assert kwargs["json"]["ids"] == [SOURCE]
            document = documents[sum(method == "POST" for method, _ in calls) - 1]
            return httpx.Response(exa_http_status, json=document, request=httpx.Request("POST", url))

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(intent.httpx, "AsyncClient", Client)
    return calls


def new_failure():
    result = missing_source()
    result["statuses"][0].update(
        sd_stage="all_tiers_exhausted:http_502", sd_error="http_502",
        sd_archive_failure={"requested_url": ORIGINAL,
            "archive_url": "https://web.archive.org/web/20261008185541/" + ORIGINAL,
            "stage": "wayback_error_body"},
        exa_target_crawl_failure={"id_matches_requested_url": True, "confirmed_attempts": 2,
            "observations": [{"error_tag": "CRAWL_EMPTY_CONTENT", "error_http_status": 500}] * 2},
    )
    return result


def test_native502_bad_archive_and_exact_exa_empty_reaches_discovery_gate(monkeypatch):
    calls = install_exact_failures(monkeypatch)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not fetched["results"]
    status = fetched["statuses"][0]
    assert status["sd_stage"] == "all_tiers_exhausted:http_502"
    assert status["sd_archive_failure"] == {
        "requested_url": SOURCE, "archive_url": SNAPSHOT, "stage": "wayback_error_body",
    }
    assert status["exa_target_crawl_failure"] == {
        "id_matches_requested_url": True, "confirmed_attempts": 2,
        "observations": [{"error_tag": "CRAWL_EMPTY_CONTENT", "error_http_status": 500}] * 2,
    }
    assert intent._unusable_source_allows_same_event_search(fetched, SOURCE)
    assert not intent._confirmed_source_absence(fetched, SOURCE)
    assert [method for method, _ in calls] == ["GET", "GET", "GET", "POST", "POST"]
    # No change to persisted company finality: this exact failure is not a terminal zero.
    projected = intent._project_contents_for_prompt(fetched)
    assert not competition._intent_attempts_have_exact_target_crawl_exhaustion(projected["statuses"])


@pytest.mark.parametrize("final", [
    "https://web.archive.org/web/20261008185541/https://other.example/article",
    "https://web.archive.org.evil.example/web/20261008185541/" + SOURCE,
    "https://web.archive.org/web/20261008185541/" + SOURCE + "?different=1",
])
def test_foreign_actual_archive_cannot_authorize_search(monkeypatch, final):
    install_exact_failures(monkeypatch, final_snapshot=final)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not fetched["results"]
    assert not fetched["statuses"][0].get("sd_archive_failure")
    assert not intent._unusable_source_allows_same_event_search(fetched, SOURCE)


@pytest.mark.parametrize("documents", [
    [{"results": [], "statuses": [{"id": "https://other.example", "status": "error",
        "error": {"tag": "CRAWL_EMPTY_CONTENT", "httpStatusCode": 500}}]}] * 2,
    [{"results": [], "statuses": [{"id": SOURCE, "status": "pending"}]}] * 2,
    [{"results": [], "statuses": [{"id": SOURCE, "status": "error",
        "error": {"tag": "CRAWL_EMPTY_CONTENT", "httpStatusCode": 401}}]}] * 2,
    [{"results": [], "statuses": [{"id": SOURCE, "status": "error",
        "error": {"tag": "CRAWL_EMPTY_CONTENT", "httpStatusCode": 500}}]},
     {"results": [], "statuses": []}],
])
def test_exa_foreign_unfinished_auth_or_mixed_results_do_not_authorize_search(monkeypatch, documents):
    install_exact_failures(monkeypatch, exa_documents=documents)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not fetched["statuses"][0].get("exa_target_crawl_failure")
    assert not intent._unusable_source_allows_same_event_search(fetched, SOURCE)


@pytest.mark.parametrize("http_status", [401, 429, 502])
def test_outer_exa_provider_failures_are_not_completed_target_observations(monkeypatch, http_status):
    install_exact_failures(monkeypatch, exa_http_status=http_status)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not fetched["statuses"][0].get("exa_target_crawl_failure")
    assert not intent._unusable_source_allows_same_event_search(fetched, SOURCE)


@pytest.mark.parametrize("mutation", [
    {"sd_archive_failure": None},
    {"sd_archive_failure": {"requested_url": "https://other.example", "archive_url": SNAPSHOT, "stage": "wayback_error_body"}},
    {"sd_archive_failure": {"requested_url": ORIGINAL, "archive_url": "https://foreign.example", "stage": "wayback_error_body"}},
    {"sd_archive_failure": {"requested_url": ORIGINAL, "archive_url": "https://web.archive.org/web/20261008185541/" + ORIGINAL, "stage": "wayback_exception"}},
    {"exa_target_crawl_failure": None},
    {"exa_target_crawl_failure": {"id_matches_requested_url": True, "confirmed_attempts": 1, "observations": []}},
    {"exa_stage": "exa_transient_exhausted"},
    {"exa_target_crawl_failure": {"id_matches_requested_url": False, "confirmed_attempts": 2,
        "observations": [{"error_tag": "CRAWL_EMPTY_CONTENT", "error_http_status": 500}] * 2}},
    {"exa_target_crawl_failure": {"id_matches_requested_url": True, "confirmed_attempts": 2,
        "observations": [{"error_tag": "CRAWL_EMPTY_CONTENT", "error_http_status": 500}] * 2, "extra": "untrusted"}},
    {"sd_archive_failure": {"requested_url": ORIGINAL, "archive_url": "https://web.archive.org/web/20261008185541/" + ORIGINAL,
        "stage": "wayback_error_body", "extra": "untrusted"}},
])
def test_healthy502alone_or_uncertain_retrieval_still_has_no_search(monkeypatch, mutation):
    original = new_failure()
    original["statuses"][0].update(mutation)
    result, fetch, search, judge = asyncio.run(run_verifier(monkeypatch, original=original))
    assert result["decision"] == "unavailable"
    assert fetch.await_count == 1
    search.assert_not_awaited(); judge.assert_not_awaited()


@pytest.mark.parametrize("negative", [None, "wrong_entity", "wrong_event", "wrong_date", "unfetched_quote"])
def test_new_recovery_uses_existing_one_shot_source_entity_event_date_and_quote_guards(monkeypatch, negative):
    response = judgment()
    alternative = fetched_source()
    item = response["answer"]["signal_evaluations"][0]
    if negative == "wrong_entity":
        item.update(signal_status="wrong_entity", same_entity_check="fail")
    elif negative == "wrong_event":
        item["risk_notes"] = ["source_event_date:2026-05-20"]
    elif negative == "wrong_date":
        alternative["results"][0]["text"] = alternative["results"][0]["text"].replace("May 20, 2026", "May 20, 2027")
    elif negative == "unfetched_quote":
        alternative["results"][0]["text"] = "Publisher general news and upcoming events."
    result, fetch, search, judge = asyncio.run(run_verifier(
        monkeypatch, original=new_failure(), alternative=alternative, response=response,
    ))
    assert result["client_ready"] is (negative is None)
    assert fetch.await_count == 2 and search.await_count == judge.await_count == 1
    if negative is None:
        assert result["source_resolution"]["status"] == "verified"
        assert result["verified_source_context"][0]["url"] == ALTERNATIVE


@pytest.mark.parametrize("search", [
    {"status": "unproven", "reason": "no_same_event_locator"},
    {"status": "unavailable", "reason": "search_provider_unavailable"},
])
def test_new_discovery_exhaustion_preserves_unavailable_not_absence(monkeypatch, search):
    result, fetch, discovery, judge = asyncio.run(run_verifier(monkeypatch, original=new_failure(), search=search))
    assert result["decision"] == "unavailable" and result["rejection_reason"] == "evidence_fetch_failed"
    assert fetch.await_count == discovery.await_count == 1
    judge.assert_not_awaited()
    assert not intent._confirmed_source_absence(new_failure(), ORIGINAL)


def test_new_exhaustion_keeps_existing_saved_breakdown_retry_semantics(monkeypatch):
    from tests.test_same_event_source_local_finality import _propagate_to_breakdown, _score_to_finality
    from lab_arena import scoring

    result, *_ = asyncio.run(run_verifier(monkeypatch, original=new_failure(), search={
        "status": "unproven", "reason": "no_same_event_locator",
    }))
    row = _propagate_to_breakdown(monkeypatch, result)
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=3"):
        _score_to_finality(row)
