"""Keep exact observed ATS fields available to the source-grounded judge."""

from copy import deepcopy

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent


JOB_ID = "1cf38233-8aa0-47f8-9d85-65ce27bc3047"
URL = f"https://jobs.ashbyhq.com/anyscale/{JOB_ID}"
# Exact retained Nov7 metadata, with a short synthetic body.
POSTING = {
    "id": JOB_ID, "jobUrl": URL, "applyUrl": URL + "/application",
    "title": "Distributed LLM Inference Engineer", "isListed": True,
    "publishedAt": "2026-05-27T23:45:11.194+00:00", "location": "San Francisco",
    "descriptionPlain": "At Anyscale, build distributed systems for machine learning inference.",
}


async def fetch(monkeypatch, posting):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers, params):
            assert url == "https://api.scrapingdog.com/scrape"
            assert params["url"] == "https://api.ashbyhq.com/posting-api/job-board/anyscale"
            calls.append(params)
            return httpx.Response(200, json={"jobs": [posting]})

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-only")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kw: Client())
    result = await intent._scrape_ashby_job(URL)
    return result, calls


@pytest.mark.asyncio
async def test_exact_observed_fields_keep_their_meaning(monkeypatch):
    result, calls = await fetch(monkeypatch, deepcopy(POSTING))
    assert result["ok"] is True and len(calls) == 1
    assert "publishedAt: 2026-05-27T23:45:11.194+00:00" in result["content"]
    assert "isListed: true" in result["content"]
    assert "applyUrl: " + URL + "/application" in result["content"]
    assert "location: San Francisco" in result["content"]
    assert "accepting applications" not in result["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [
    "https://evil.example/application", URL.replace("anyscale", "other") + "/application",
    URL.replace(JOB_ID, "12345678-1234-1234-1234-123456789abc") + "/application",
    URL + "/application?other=1", URL + "/application#other", "", None, 123,
    URL + "#fragment/application", URL.replace(".com/", ".com:444/") + "/application",
    URL.replace("https://", "http://") + "/application",
    URL.replace("https://", "https://user@") + "/application",
    URL.replace(".com/", ".com:bad/") + "/application", "\n" + URL + "/application",
])
async def test_application_locator_must_bind_exact_posting(monkeypatch, value):
    result, _ = await fetch(monkeypatch, {**POSTING, "applyUrl": value})
    assert result["ok"] is True
    assert "applyUrl:" not in result["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, None, "true", 1])
async def test_missing_or_unlisted_job_cannot_gain_listing_proof(monkeypatch, value):
    result, _ = await fetch(monkeypatch, {**POSTING, "isListed": value})
    assert result["ok"] is False and not result["content"]


@pytest.mark.asyncio
async def test_old_publication_date_cannot_be_replaced_by_update_date(monkeypatch):
    result, _ = await fetch(monkeypatch, {
        **POSTING, "publishedAt": "2025-07-21T10:00:00Z", "updatedAt": "2026-11-07T12:00:00Z",
    })
    assert "publishedAt: 2025-07-21T10:00:00Z" in result["content"]
    assert "2026-11-07" not in result["content"]


@pytest.mark.asyncio
async def test_absent_metadata_is_not_synthesized(monkeypatch):
    posting = {k: v for k, v in POSTING.items() if k not in {"applyUrl", "publishedAt"}}
    result, _ = await fetch(monkeypatch, posting)
    assert result["ok"] is True
    assert "publishedAt:" not in result["content"] and "applyUrl:" not in result["content"]
