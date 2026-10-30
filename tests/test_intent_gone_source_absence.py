"""A gone target needs the same independent absence proof as a missing target."""

import asyncio
import copy
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent


URL = "https://careers.example/jobs/123"
ABSENCE = {
    "id_matches_requested_url": True, "status": "error",
    "error_tag": "CRAWL_NOT_FOUND", "error_http_status": 404,
    "confirmed_attempts": 2,
}


def install_fetches(monkeypatch, *, direct_status=410, archive=None):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            assert url == "https://api.scrapingdog.com/scrape"
            assert kwargs["params"]["url"] == URL
            calls.append(url)
            return httpx.Response(direct_status, text="Page not found", request=httpx.Request("GET", url))

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    monkeypatch.setattr(intent.httpx, "AsyncClient", Client)
    wayback = AsyncMock(return_value=archive or {
        "ok": False, "stage": "wayback_no_snapshot", "content": "",
        "error": "no archived snapshot",
    })
    monkeypatch.setattr(intent, "_try_wayback", wayback)
    exa = AsyncMock(return_value={
        "ok": False, "stage": "exa_target_not_found", "content": "",
        "error": "target_not_found", "target_absence": ABSENCE,
    })
    monkeypatch.setattr(intent, "_scrape_exa", exa)
    return calls, wayback, exa


def test_native_gone_fetch_with_independent_absence_finishes_hiring_review(monkeypatch):
    calls, wayback, exa = install_fetches(monkeypatch)
    judge = AsyncMock(side_effect=AssertionError("absent evidence must not call a judge"))
    search = AsyncMock(side_effect=AssertionError("hiring must not discover a different job"))
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    monkeypatch.setattr(intent, "_find_same_event_source", search)
    result = asyncio.run(intent.verify_three_stage(
        None, company_name="Acme", company_website="https://acme.example",
        company_linkedin="", source_url=URL, miner_claim="Hiring engineers",
        target_signal_text="Actively hiring engineers", evidence_type="HIRING",
        integrity_policy=True, stage1_soft_reject=True,
    ))
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "evidence_not_found"
    assert result["verdict"]["signal_evaluations"][0]["signal_status"] == "unable_to_verify"
    assert result["scrape"]["statuses"][0]["sd_stage"] == "genuine_410"
    assert len(calls) == 1
    wayback.assert_awaited_once_with(URL)
    exa.assert_awaited_once_with(URL)
    judge.assert_not_awaited()
    search.assert_not_awaited()


@pytest.mark.parametrize("status", [404, 410])
def test_absent_status_requires_exact_independent_receipts(monkeypatch, status):
    install_fetches(monkeypatch, direct_status=status)
    contents = asyncio.run(intent._fetch_sd_then_exa([URL]))
    assert intent._confirmed_source_absence(contents, URL)
    assert intent._unusable_source_allows_same_event_search(contents, URL)
    for field, value in [
        ("url", URL + "-other"), ("source", "exa_fallback"),
        ("sd_stage", "all_tiers_exhausted:http_410"),
        ("sd_error", "http_502"), ("exa_stage", "exa_no_results"),
        ("exa_error", "[]"), ("exa_target_absence", {}),
    ]:
        changed = copy.deepcopy(contents)
        changed["statuses"][0][field] = value
        assert not intent._confirmed_source_absence(changed, URL), field
    for field, value in [
        ("id_matches_requested_url", False), ("confirmed_attempts", 1),
        ("status", "pending"), ("error_tag", "CRAWL_EMPTY_CONTENT"),
        ("error_http_status", 500),
    ]:
        changed = copy.deepcopy(contents)
        changed["statuses"][0]["exa_target_absence"][field] = value
        assert not intent._confirmed_source_absence(changed, URL), field


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_provider_failure_is_not_target_absence(monkeypatch, status):
    install_fetches(monkeypatch, direct_status=status)
    contents = asyncio.run(intent._fetch_sd_then_exa([URL]))
    assert not intent._confirmed_source_absence(contents, URL)


def test_readable_archive_still_wins_over_gone_live_source(monkeypatch):
    body = "Acme hiring engineers. " * 40
    _, wayback, exa = install_fetches(monkeypatch, archive={
        "ok": True, "stage": "wayback", "content": body, "error": None,
    })
    contents = asyncio.run(intent._fetch_sd_then_exa([URL]))
    assert contents["results"][0]["text"] == body
    assert not intent._confirmed_source_absence(contents, URL)
    wayback.assert_awaited_once_with(URL)
    exa.assert_not_awaited()


def test_gone_corroborating_source_keeps_existing_bounded_company_finality(monkeypatch):
    from qualification.scoring import competition
    from tests.test_same_event_source_local_finality import (
        _propagate_to_breakdown, _score_to_finality, confirmed_alternative_absence,
    )
    from tests.test_intent_unavailable_same_event_source import run_verifier

    alternative = confirmed_alternative_absence()
    alternative["statuses"][0].update(sd_stage="genuine_410", sd_error="http_410")
    result, *_ = asyncio.run(run_verifier(monkeypatch, alternative=alternative))
    assert result["decision"] == "unavailable"
    row = _propagate_to_breakdown(monkeypatch, result)
    assert competition.scorer_breakdown_has_company_local_verification_failure(row)
    terminal, calls = _score_to_finality(row)
    assert calls == 3
    assert terminal["final_score"] == 0
    assert competition.scorer_breakdown_is_terminal_company_verification_failure(terminal)
