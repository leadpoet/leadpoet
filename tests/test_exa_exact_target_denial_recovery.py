"""Authenticated exact-page access denial is a locator, never absence proof."""

from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import competition
from qualification.scoring import intent_verification_three_stage as verifier
from tests.test_intent_unavailable_same_event_source import CLAIM, ORIGINAL, QUOTE, judgment


ALTERNATIVE = "https://organizer.example/echelon-2026"


def captured_denial():
    # Sanitized exact shape from the reward-disabled live Exa contents receipt.
    return {"requestId": "unit-test", "results": [], "searchTime": 0,
            "statuses": [{"id": ORIGINAL, "status": "error", "error": {
                "httpStatusCode": 403, "tag": "SOURCE_NOT_AVAILABLE",
            }}]}


def install_transport(monkeypatch, responses):
    posts = []
    replies = iter(responses)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, **_kwargs):
            assert url == "https://api.exa.ai/contents"
            assert json["ids"] == [ORIGINAL]
            posts.append(url)
            status, body = next(replies)
            return httpx.Response(status, json=body)

    async def sd(url):
        if url == ALTERNATIVE:
            return {"ok": True, "stage": "sd:baseline", "content": QUOTE,
                    "error": None}
        assert url == ORIGINAL
        return {"ok": False, "stage": "all_tiers_exhausted:http_422",
                "error": "http_422", "content": ""}

    monkeypatch.setenv("EXA_API_KEY", "unit-test")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(verifier, "_scrape_sd_hardened", sd)
    return posts


@pytest.mark.asyncio
async def test_captured_two_observation_denial_survives_fetch_and_projection(monkeypatch):
    posts = install_transport(monkeypatch, [(200, captured_denial())] * 2)
    fetched = await verifier._fetch_sd_then_exa([ORIGINAL])
    assert len(posts) == 2
    assert fetched["results"] == []
    receipt = fetched["statuses"][0]["exa_target_crawl_failure"]
    assert receipt == {"id_matches_requested_url": True, "confirmed_attempts": 2,
                       "observations": [{"error_tag": "SOURCE_NOT_AVAILABLE",
                                         "error_http_status": 403}] * 2}
    projected = verifier._project_contents_for_prompt(fetched)
    assert projected["statuses"][0]["exa_target_crawl_failure"] == receipt
    assert competition._confirmed_exa_target_crawl_failure(receipt)
    assert verifier._unusable_source_allows_same_event_search(fetched, ORIGINAL)
    assert not verifier._confirmed_source_absence(fetched, ORIGINAL)


@pytest.mark.asyncio
async def test_captured_denial_enters_one_search_fetch_and_ordinary_judge(monkeypatch):
    posts = install_transport(monkeypatch, [(200, captured_denial())] * 2)
    search = AsyncMock(return_value={"status": "selected",
        "reason": "bounded_same_event_search", "selected_urls": [ALTERNATIVE]})
    judge = AsyncMock(return_value=judgment(url=ALTERNATIVE))
    monkeypatch.setattr(verifier, "_find_same_event_source", search)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    result = await verifier.verify_three_stage(
        object(), company_name="PixVerse", company_website="https://pixverse.ai/",
        company_linkedin="", source_url=ORIGINAL, miner_claim=CLAIM,
        target_signal_text="Publicly named as a partner at a technology event",
        miner_signal_date="2026-05-20", evidence_type="PARTNERSHIP",
        stage1_soft_reject=True, integrity_policy=True,
    )
    assert result["client_ready"]
    assert result["source_resolution"]["status"] == "verified"
    assert len(posts) == 2
    assert search.await_count == judge.await_count == 1
    assert result["verified_source_context"][0]["url"] == ALTERNATIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["api_403", "api_401", "api_402", "api_429",
    "api_503", "wrong_url", "one_off", "pending", "untyped", "wrong_tag",
    "wrong_status_code", "float_status_code"])
async def test_provider_or_unconfirmed_denial_cannot_admit_recovery(monkeypatch, case):
    denied = captured_denial()
    changed = deepcopy(denied)
    status = changed["statuses"][0]
    if case.startswith("api_"):
        replies = [(int(case[4:]), denied)] * 2
    else:
        if case == "wrong_url":
            status["id"] = "https://www.instagram.com/p/different/"
        elif case == "pending":
            status["status"] = "pending"
        elif case == "untyped":
            status.pop("error")
        elif case == "wrong_tag":
            status["error"]["tag"] = "UNAUTHORIZED"
        elif case == "wrong_status_code":
            status["error"]["httpStatusCode"] = 401
        elif case == "float_status_code":
            status["error"]["httpStatusCode"] = 403.0
        elif case == "one_off":
            status["status"] = "pending"
        replies = [(200, denied), (200, changed)] if case == "one_off" else [(200, changed)] * 2
    install_transport(monkeypatch, replies)
    search = AsyncMock()
    judge = AsyncMock()
    monkeypatch.setattr(verifier, "_find_same_event_source", search)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    result = await verifier.verify_three_stage(
        object(), company_name="PixVerse", company_website="https://pixverse.ai/",
        company_linkedin="", source_url=ORIGINAL, miner_claim=CLAIM,
        target_signal_text="Publicly named as a partner at a technology event",
        miner_signal_date="2026-05-20", evidence_type="PARTNERSHIP",
        stage1_soft_reject=True, integrity_policy=True,
    )
    assert result["decision"] == "unavailable"
    assert "source_resolution" not in result
    assert not result["scrape"]["statuses"][0].get("exa_target_crawl_failure")
    search.assert_not_awaited()
    judge.assert_not_awaited()
