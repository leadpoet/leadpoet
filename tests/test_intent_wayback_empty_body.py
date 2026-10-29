"""An archived challenge shell cannot suppress exact-source Exa recovery."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent


SOURCE = (
    "https://www.amdocs.com/press-release/"
    "veon-and-amdocs-enter-10-year-strategic-agreement-advance-digital-services"
)
TIMESTAMP = "20261008185541"
SNAPSHOT = "https://web.archive.org/web/" + TIMESTAMP + "/" + SOURCE
# Public HTML supplied to saved judge call 77164. Only the opaque public
# Incapsula resource query value is redacted; the empty body is unchanged.
SHELL = (Path(__file__).parent / "fixtures/amdocs_empty_wayback_shell.html").read_text()
QUOTE = "VEON and Amdocs announced a 10-year strategic agreement on October 7, 2026."
ARTICLE_TEXT = QUOTE + " " + (
    "The agreement combines operational modernization and technology "
    "contributions from both companies to advance digital services. " * 5
)
ARTICLE_HTML = (
    "<html><head><title>Agreement announcement</title></head><body>"
    "<article><h1>Agreement announcement</h1><p>" + ARTICLE_TEXT
    + "</p></article></body></html>"
)


def install_source_responses(monkeypatch, *, archive=SHELL, direct_pdf=False):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def get(self, url, **kwargs):
            calls.append((url, kwargs))
            request = httpx.Request("GET", url)
            if url == "https://api.scrapingdog.com/scrape":
                assert kwargs["params"]["url"] == SOURCE
                if direct_pdf:
                    return httpx.Response(200, content=b"%PDF-1.7\n" + b"x" * 800, request=request)
                return httpx.Response(502, text="source fetch failed", request=request)
            if url == "https://archive.org/wayback/available?url=" + SOURCE:
                return httpx.Response(200, json={"archived_snapshots": {"closest": {
                    "available": True, "status": "200", "timestamp": TIMESTAMP,
                    "url": "http" + SNAPSHOT[5:],
                }}}, request=request)
            assert url == SNAPSHOT
            return httpx.Response(200, text=archive, request=request)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-only-placeholder")
    monkeypatch.setattr(intent.httpx, "AsyncClient", Client)
    return calls


@pytest.mark.parametrize("body", [
    SHELL,
    "<html><head><title>Metadata</title></head><body><script>"
    + "hidden-challenge-state" * 40 + "</script><!-- no article --></body></html>",
], ids=["captured_incapsula_shell", "script_only_body"])
def test_archive_with_empty_visible_body_is_unusable_despite_raw_length(monkeypatch, body):
    assert len(body) >= 500 and intent._has_empty_html_body(body)
    install_source_responses(monkeypatch, archive=body)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result["ok"] is False
    assert result["stage"] == "wayback_html_empty_body"
    assert result["content"] == ""


def test_captured_shell_continues_to_exa_for_the_same_exact_source(monkeypatch):
    calls = install_source_responses(monkeypatch)
    fallback = AsyncMock(return_value={
        "ok": True, "stage": "exa_scraped", "content": ARTICLE_TEXT,
        "source_publication_date": "2026-10-07", "error": None,
    })
    monkeypatch.setattr(intent, "_scrape_exa", fallback)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert len(fetched["results"]) == 1
    assert fetched["results"][0]["url"] == SOURCE
    assert fetched["results"][0]["text"] == ARTICLE_TEXT
    assert fetched["statuses"][0]["source"] == "exa_fallback"
    assert [url for url, _ in calls] == [
        "https://api.scrapingdog.com/scrape",
        "https://archive.org/wayback/available?url=" + SOURCE,
        SNAPSHOT,
    ]
    projected = intent._project_contents_for_prompt(fetched)
    assert projected["results"][0]["url"] == SOURCE
    # Existing telemetry intentionally projects statuses to the origin. This
    # must not be mistaken for a homepage fetch or evidence substitution.
    assert projected["statuses"][0]["url"] == "https://amdocs.com"


@pytest.mark.parametrize("body", [
    ARTICLE_HTML,
    ARTICLE_HTML.replace("<body>", "").replace("</body>", ""),
    ARTICLE_TEXT,
], ids=["article_html", "implicit_html_body", "plain_text"])
def test_healthy_concise_archive_retains_original_body_and_exact_source_metadata(monkeypatch, body):
    assert 500 <= len(body) < 3000
    install_source_responses(monkeypatch, archive=body)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result["ok"] is True and result["stage"] == "wayback"
    assert result["content"] == body
    assert result["meta"]["ownership_archive_url"] == SNAPSHOT


def test_empty_archive_and_failed_exa_remain_unavailable_without_semantic_judge(monkeypatch):
    install_source_responses(monkeypatch)
    fallback = AsyncMock(return_value={"ok": False, "stage": "exa_no_results", "content": "", "error": "no usable body"})
    judge = AsyncMock(side_effect=AssertionError("empty source must not reach the judge"))
    monkeypatch.setattr(intent, "_scrape_exa", fallback)
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = asyncio.run(intent.verify_three_stage(
        object(), company_name="Amdocs", company_website="https://amdocs.com/",
        company_linkedin="", source_url=SOURCE, miner_claim=QUOTE,
        target_signal_text="Announced a strategic technology partnership",
        integrity_policy=True, stage1_soft_reject=True,
    ))
    fallback.assert_awaited_once_with(SOURCE)
    judge.assert_not_awaited()
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "evidence_fetch_failed"
    assert result["scrape"]["result_count"] == 0
    assert result["stage3"] is None


def test_recovered_exact_source_reaches_native_judge_without_archive_shell(monkeypatch):
    install_source_responses(monkeypatch)
    monkeypatch.setattr(intent, "_scrape_exa", AsyncMock(return_value={
        "ok": True, "stage": "exa_scraped", "content": ARTICLE_TEXT,
        "source_publication_date": "2026-10-07", "error": None,
    }))
    judge = AsyncMock(return_value={"model": "offline-test", "usage": {}, "answer": {
        "overall_verdict": "qualified", "overall_confidence": "high",
        "signal_evaluations": [{
            "signal_id": "signal-1", "claim": QUOTE,
            "verification_mode": "source_grounded", "signal_status": "supported",
            "same_entity_check": "pass", "confidence": "high",
            "source_urls_supplied": [SOURCE], "evidence_urls_used": [SOURCE],
            "supporting_quotes": [QUOTE], "contradicting_quotes": [],
            "unsupported_parts": [], "claim_matches_miner_date": "consistent",
            "risk_notes": ["source_event_date:2026-10-07"],
        }],
    }})
    monkeypatch.setattr(intent, "_call_openrouter", judge)
    result = asyncio.run(intent.verify_three_stage(
        object(), company_name="Amdocs", company_website="https://amdocs.com/",
        company_linkedin="", source_url=SOURCE, miner_claim=QUOTE,
        target_signal_text="Announced a strategic technology partnership",
        miner_signal_date="2026-10-07", evidence_type="PARTNERSHIP",
        integrity_policy=True, stage1_soft_reject=True,
    ))
    judge.assert_awaited_once()
    prompt = judge.await_args.args[2]
    assert SOURCE in prompt and QUOTE in prompt
    assert "_Incapsula_Resource" not in prompt and "web-static.archive.org" not in prompt
    assert result["scrape"]["result_count"] == 1
    assert result["client_ready"] is True


def test_pdf_still_bypasses_wayback_and_uses_existing_exact_url_exa_path(monkeypatch):
    calls = install_source_responses(monkeypatch, direct_pdf=True)
    fallback = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": ARTICLE_TEXT, "error": None})
    monkeypatch.setattr(intent, "_scrape_exa", fallback)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert result["results"][0]["text"] == ARTICLE_TEXT
    assert [url for url, _ in calls] == ["https://api.scrapingdog.com/scrape"]
