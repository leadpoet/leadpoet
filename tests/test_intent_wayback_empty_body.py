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
# Saved call 77458 contains the expanded archive error iframe and toolbar.
# Only opaque public Incapsula resource query values are redacted.
ERROR_FRAME = (Path(__file__).parent / "fixtures/amdocs_wayback_error_iframe.html").read_text()
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


def test_captured_nonempty_wayback_error_frame_is_not_source_content(monkeypatch):
    assert len(ERROR_FRAME)>500 and not intent._has_empty_html_body(ERROR_FRAME)
    install_source_responses(monkeypatch,archive=ERROR_FRAME)
    result=asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is False and result['content']==''
    assert result['stage']=='wayback_error_body'


def test_archive_error_frame_continues_existing_exact_url_fallback_once(monkeypatch):
    calls=install_source_responses(monkeypatch,archive=ERROR_FRAME)
    fallback=AsyncMock(return_value={'ok':True,'stage':'exa_scraped','content':ARTICLE_TEXT,'error':None})
    monkeypatch.setattr(intent,'_scrape_exa',fallback)
    result=asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert result['results'][0]['text']==ARTICLE_TEXT and result['results'][0]['url']==SOURCE
    assert len(calls)==3


@pytest.mark.parametrize('content',[
    ARTICLE_HTML.replace('</body>',ERROR_FRAME+'</body>'),
    ARTICLE_HTML.replace('</article>','<p>The Wayback Machine has not archived that URL.</p></article>'),
    ARTICLE_HTML.replace('<h1>','<div id="wm-ipp-base">Wayback Machine toolbar</div><h1>'),
    ARTICLE_HTML.replace('</article>','<div id="error"><div class="error-text">The Wayback Machine has not archived that URL.</div></div></article>'),
    ARTICLE_HTML.replace('</article>','<script>var example="error-text";</script><!-- errorBorder --></article>'),
    ARTICLE_TEXT,
    '%PDF-1.7\n'+ARTICLE_TEXT,
],ids=['article_with_broken_optional_embed','article_about_archive_errors','article_with_toolbar',
       'article_with_error_prose_markup','script_and_comment_example','plain_text','pdf_unchanged'])
def test_genuine_source_survives_archive_error_words_toolbar_or_broken_embed(monkeypatch,content):
    install_source_responses(monkeypatch,archive=content)
    result=asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is True and result['content']==content
    assert result['meta']['ownership_archive_url']==SNAPSHOT


def _standalone_archive_error():
    start=ERROR_FRAME.index('<html lang="en">')
    end=ERROR_FRAME.index('</html>',start)+len('</html>')
    return ERROR_FRAME[start:end]


def test_standalone_structural_archive_error_is_unusable(monkeypatch):
    install_source_responses(monkeypatch,archive=_standalone_archive_error())
    result=asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is False and result['stage']=='wayback_error_body'
    assert result['content']=='' and not result.get('meta')


@pytest.mark.parametrize('body',[
    _standalone_archive_error().replace('<title>Wayback Machine</title>','<title>Source report</title>'),
    _standalone_archive_error().replace('class="navia wb_embedded"','class="source-report"'),
    _standalone_archive_error().replace('action="https://web.archive.org/web/submit"','action="https://web.archive.org.evil.example/web/submit"'),
    _standalone_archive_error().replace('id="errorBorder"','id="sourceBorder"'),
    _standalone_archive_error().replace('class="error-text"','class="quoted-error-text"'),
],ids=['ordinary_title','ordinary_body','foreign_service_form','no_error_container','no_error_text_node'])
def test_partial_lookalike_archive_layout_is_not_enough_to_remove_source(body):
    assert intent._has_wayback_error_body(body) is False


@pytest.mark.parametrize('tag',['article','main','p','div','span',''])
def test_archive_service_style_error_discussion_with_independent_body_is_preserved(monkeypatch,tag):
    # Even all service markers do not erase source text outside the error UI.
    source=('<'+tag+'>'+ARTICLE_TEXT+'</'+tag+'>') if tag else ARTICLE_TEXT
    body=_standalone_archive_error().replace('<div id="errorBorder">',source+'<div id="errorBorder">')
    install_source_responses(monkeypatch,archive=body)
    result=asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is True and result['content']==body


def test_archive_error_plus_failed_exact_fallback_remains_unavailable(monkeypatch):
    install_source_responses(monkeypatch,archive=ERROR_FRAME)
    fallback=AsyncMock(return_value={'ok':False,'stage':'exa_no_results','content':'','error':'unavailable'})
    monkeypatch.setattr(intent,'_scrape_exa',fallback)
    judge=AsyncMock(side_effect=AssertionError('archive error is not judge evidence'))
    monkeypatch.setattr(intent,'_call_openrouter',judge)
    result=asyncio.run(intent.verify_three_stage(object(),company_name='Amdocs',company_website='https://amdocs.com/',
        company_linkedin='',source_url=SOURCE,miner_claim=QUOTE,target_signal_text='Announced a strategic technology partnership',
        integrity_policy=True,stage1_soft_reject=True))
    fallback.assert_awaited_once_with(SOURCE)
    judge.assert_not_awaited()
    assert result['decision']=='unavailable' and result['scrape']['result_count']==0


def test_archive_navigation_without_article_continues_exact_source_fallback(monkeypatch):
    # Production Paysend capture 7298243 had only a site title and the
    # archive toolbar. Neither is evidence for the submitted announcement.
    body = (
        '<html><head><title>Fintech News Network</title></head><body>'
        '<div id="wm-ipp-base">1 capture 10 Oct 2026 About this capture</div>'
        '<div id="wm-ipp-print">The Wayback Machine snapshot</div>'
        '<script>' + 'empty-app-state;' * 40 + '</script></body></html>'
    )
    install_source_responses(monkeypatch, archive=body)
    fallback = AsyncMock(return_value={
        'ok': True, 'stage': 'exa_scraped', 'content': ARTICLE_TEXT, 'error': None,
    })
    monkeypatch.setattr(intent, '_scrape_exa', fallback)
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    fallback.assert_awaited_once_with(SOURCE)
    assert fetched['results'][0]['text'] == ARTICLE_TEXT


@pytest.mark.parametrize('tag', ['p', 'div', 'span'])
def test_archive_navigation_does_not_hide_concise_source_body(monkeypatch, tag):
    body = (
        '<html><head><title>Announcement</title></head><body>'
        '<div id="wm-ipp-base">1 capture About this capture</div>'
        '<' + tag + '>' + QUOTE + '</' + tag + '>'
        '<script>' + 'empty-app-state;' * 40 + '</script></body></html>'
    )
    install_source_responses(monkeypatch, archive=body)
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result['ok'] is True
    assert QUOTE in result['content']
