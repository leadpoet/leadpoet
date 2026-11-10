"""Partial article retrieval supplements evidence without deleting old facts."""
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.qualification.models import ICPPrompt, IntentSignal
from lab_arena.scoring import DEFAULT_JUDGE_MODELS
from qualification.scoring import lead_scorer
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import verification_helpers as helper
from qualification.scoring.evaluation_clock import use_evaluation_date


URL = "https://cortland.com/in-the-news/cortland-finalizes-1-6-billion-acquisition-of-greater-atlanta-washington-dc-properties/"
TITLE = "Cortland finalizes $1.6 billion acquisition of greater Atlanta, Washington DC properties"
FACT = "Atlanta-based Cortland’s $1.6 billion acquisition of a 19-property portfolio closed on Nov. 12, according to a company news release."
UNITS = "Nearly 6,000 multifamily units are part of the deal, per the news release."
FULL_TEXT = TITLE + "\nNovember 13, 2025\n" + FACT + "\n" + UNITS
# Reduced from both saved Nov10 baseline/miner responses. The original CSS
# and DOM hide article prose; this change must never relax that visibility.
HTML = f'''<html><head><title>{TITLE} | In the News | Cortland</title>
<style>.wow{{visibility:hidden}} /* {'x' * 3100} */</style></head><body>
<a href="#content">Skip to main content</a><button>Menu</button><button>Find Your Home</button>
<div class="page article" role="main"><header><h1 class="wow fadeInUp">{TITLE}</h1>
<p class="wow fadeInUp">November 13, 2025</p><a href="/in-the-news/">Back to In the News</a></header>
<div class="page__content wow fadeInUp"><p>{FACT}</p><p>{UNITS}</p></div>
<a class="button" href="/in-the-news/">Back to In the News</a></div></body></html>'''
IDENTITY = {"decision": "match", "evidence_source": "company_homepage",
            "observed_name": "cortland", "observed_domain": "cortland.com",
            "observed_linkedin_slug": "cortland-"}


def sd_result():
    return {"ok": True, "stage": "sd:baseline", "content": "Original acquisition fact.",
            "partial_article": True, "source_title": "Original title",
            "source_publication_date": "2025-11-13",
            "meta": {"ownership_archive_url": "https://web.archive.org/web/20251113/" + URL}}


@pytest.mark.parametrize("body", [
    "Acme acquired Bravo.",
    '<article><h1>Acme</h1><p>Acme acquired Bravo.</p></article>',
    '<article><h1>Acme</h1><a href="/proof"><p>Acme acquired Bravo.</p></a></article>',
    '<article><a href="/proof"><h2>Acme</h2><p>Acme acquired Bravo.</p></a></article>',
    '<article><h1>Acme</h1><a href="https://[">Acme acquired Bravo.</a></article>',
    '<main><h1>Careers</h1><a href="/jobs/one">Engineer</a></main>',
])
def test_brief_linked_and_listing_shapes_do_not_request_supplement(body):
    assert helper.visible_html_article_body_partial(body) is False


def test_stored_shape_is_only_a_hint_and_excludes_hidden_source_text():
    assert helper.visible_html_article_body_partial(HTML)
    assert FACT not in helper.extract_article_body(HTML)
    assert len(helper.extract_article_body(HTML)) == 194


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", [
    {"ok": False, "content": "", "error": "provider_unavailable"},
    {"ok": True, "content": ""},
    {"ok": True, "content": HTML},
    {"ok": True, "content": "<html><head><title>Acme</title></head><body><h1>Acme</h1></body></html>"},
    {"ok": True, "content": "<article><h1>Acme</h1><p hidden>Hidden claim.</p><p>Visible overview.</p></article>"},
])
async def test_failed_empty_or_unprocessed_html_preserves_original(monkeypatch, candidate):
    original = sd_result()
    snapshot = deepcopy(original)
    fetch = AsyncMock(return_value=candidate)
    monkeypatch.setattr(verifier, "_scrape_exa", fetch)
    result, status = await verifier._supplement_partial_article(original, URL, max_chars=12000)
    assert result == snapshot and original == snapshot and status is None
    fetch.assert_awaited_once_with(URL)


@pytest.mark.asyncio
async def test_success_keeps_exact_prefix_metadata_and_provenance(monkeypatch):
    original = sd_result()
    fetch = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": FULL_TEXT,
                                   "source_publication_date": "2099-01-01", "source_title": "Changed"})
    monkeypatch.setattr(verifier, "_scrape_exa", fetch)
    result, status = await verifier._supplement_partial_article(original, URL, max_chars=12000)
    assert result["content"] == original["content"] + "\n\n" + FULL_TEXT
    assert {k: v for k, v in result.items() if k != "content"} == {k: v for k, v in original.items() if k != "content"}
    assert status == {"url": URL, "source": "exa_fallback", "stage": "exa_scraped", "sd_stage": "sd:baseline"}
    result, status = await verifier._supplement_partial_article(original, URL, max_chars=len(original["content"]))
    assert result == original and status is None
    fetch.assert_awaited_once_with(URL)  # A full original cannot spend another fetch.


@pytest.mark.asyncio
async def test_oversize_supplement_preserves_prefix_within_existing_cap(monkeypatch):
    original = sd_result()
    fetch = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": FULL_TEXT * 30})
    monkeypatch.setattr(verifier, "_scrape_exa", fetch)
    cap = len(original["content"]) + 2 + 25
    result, status = await verifier._supplement_partial_article(original, URL, max_chars=cap)
    assert result["content"] == original["content"] + "\n\n" + FULL_TEXT[:25]
    assert len(result["content"]) == cap and status is not None
    fetch.assert_awaited_once_with(URL)


@pytest.mark.asyncio
async def test_optional_failure_does_not_swallow_cancellation(monkeypatch):
    for error in (TimeoutError("offline"), RuntimeError("offline")):
        monkeypatch.setattr(verifier, "_scrape_exa", AsyncMock(side_effect=error))
        original = sd_result()
        assert await verifier._supplement_partial_article(original, URL, max_chars=12000) == (original, None)
    monkeypatch.setattr(verifier, "_scrape_exa", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await verifier._supplement_partial_article(sd_result(), URL, max_chars=12000)


@pytest.mark.asyncio
async def test_full_html_supplement_never_exposes_its_hidden_claim(monkeypatch):
    body = '<html><body><article><h1>Acme</h1><p hidden>HIDDEN ACQUISITION CLAIM</p><p>Visible company overview.</p></article></body></html>'
    monkeypatch.setattr(verifier, "_scrape_exa", AsyncMock(return_value={"ok": True, "content": body}))
    result, status = await verifier._supplement_partial_article(sd_result(), URL, max_chars=12000)
    assert status is not None and "Visible company overview." in result["content"]
    assert "HIDDEN ACQUISITION CLAIM" not in result["content"]


def transport(monkeypatch):
    requests = []
    real_client = httpx.AsyncClient
    def respond(request):
        assert request.url.host == "api.scrapingdog.com"
        requests.append(request)
        return httpx.Response(200, text=HTML, request=request)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", lambda *a, **kw: real_client(transport=httpx.MockTransport(respond)))
    return requests


@pytest.mark.asyncio
async def test_shared_native_route_uses_one_optional_helper(monkeypatch):
    requests = transport(monkeypatch)
    fetch = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": FULL_TEXT})
    monkeypatch.setattr(verifier, "_scrape_exa", fetch)
    result = await verifier._fetch_sd_then_exa([URL])
    assert len(requests) == 1  # No SD escalation or extra retry layer.
    fetch.assert_awaited_once_with(URL)
    assert result["results"][0]["text"] == helper.extract_article_body(HTML) + "\n\n" + FULL_TEXT
    assert result["statuses"][0] == {"url": URL, "source": "scrapingdog", "stage": "sd:baseline"}
    assert result["statuses"][1]["source"] == "exa_fallback"


@pytest.mark.asyncio
async def test_wayback_partial_marker_preserves_archive_source_date_and_ownership(monkeypatch):
    snapshot = "https://web.archive.org/web/20261109000101/" + URL
    raw = HTML.replace("<head>", '<head><meta property="article:published_time" content="2025-11-13">')
    raw = raw.replace("</style>", "/*" + "x" * verifier.MAX_SCRAPED_CHARS + "*/</style>")
    real_client = httpx.AsyncClient
    calls = []
    def respond(request):
        calls.append(request.url.host)
        if request.url.host == "api.scrapingdog.com":
            return httpx.Response(502, text="upstream unavailable", request=request)
        if request.url.host == "archive.org":
            return httpx.Response(200, json={"archived_snapshots": {"closest": {
                "url": snapshot, "timestamp": "20261109000101", "available": True, "status": "200"}}}, request=request)
        assert request.url.host == "web.archive.org"
        return httpx.Response(200, text=raw, request=request)
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-placeholder")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", lambda *a, **kw: real_client(transport=httpx.MockTransport(respond)))
    supplement = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": FULL_TEXT,
                                        "source_publication_date": "2099-01-01"})
    monkeypatch.setattr(verifier, "_scrape_exa", supplement)
    result = await verifier._fetch_sd_then_exa([URL])
    source = result["results"][0]
    assert source["url"] == URL and source["source_publication_date"] == "2025-11-13"
    assert source["meta"]["ownership_archive_url"] == snapshot
    assert source["text"].startswith(helper.extract_article_body(raw) + "\n\n")
    assert result["statuses"][0]["stage"] == "wayback"
    assert result["statuses"][1]["source"] == "exa_fallback"
    assert calls == ["api.scrapingdog.com", "archive.org", "web.archive.org"]
    supplement.assert_awaited_once_with(URL)


@pytest.mark.asyncio
async def test_failed_supplement_keeps_native_result_and_status(monkeypatch):
    transport(monkeypatch)
    monkeypatch.setattr(verifier, "_scrape_exa", AsyncMock(return_value={"ok": False, "content": ""}))
    result = await verifier._fetch_sd_then_exa([URL])
    assert result["results"][0]["text"] == helper.extract_article_body(HTML)
    assert result["statuses"] == [{"url": URL, "source": "scrapingdog", "stage": "sd:baseline"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_state", ["supported", "unable_to_verify", "stale", "wrong_entity"])
async def test_native_scoring_keeps_semantic_and_date_gates(monkeypatch, model_state):
    transport(monkeypatch)
    monkeypatch.setenv("ARENA_INTENT_EVIDENCE_MODEL", DEFAULT_JUDGE_MODELS["intent_fetched_evidence"])
    fetch = AsyncMock(return_value={"ok": True, "stage": "exa_scraped", "content": FULL_TEXT})
    monkeypatch.setattr(verifier, "_scrape_exa", fetch)
    status = "supported" if model_state == "stale" else model_state
    reply = {"overall_verdict": "qualified", "signal_evaluations": [{
        "signal_id": "signal-1", "signal_status": status, "confidence": "high",
        "same_entity_check": "fail" if model_state == "wrong_entity" else "pass",
        "verification_mode": "source_grounded", "evidence_urls_used": [URL],
        "supporting_quotes": [FACT], "contradicting_quotes": [], "unsupported_parts": [],
        "claim_matches_miner_date": "consistent",
        "risk_notes": ["source_event_date:" + ("2019-11-12" if model_state == "stale" else "2025-11-12")],
    }]}
    judge = AsyncMock(return_value={"answer": reply, "usage": {}, "model": DEFAULT_JUDGE_MODELS["intent_fetched_evidence"]})
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    icp = ICPPrompt(icp_id="nov10-icp8", prompt="Completed apartment acquisition within 365 days",
        industry="Real Estate", sub_industry="Multifamily Residential Owners and Operators",
        employee_count="201-500|501-1,000|1,001-5,000", company_stage="Any",
        geography="United States, South", product_service="Multifamily apartment ownership",
        intent_signals=["Completed an acquisition of an apartment community or portfolio in the last 12 months"],
        intent_signal_evidence_types=["ACQUISITION"], intent_max_age_days=365)
    signal = IntentSignal(source="company_website", description="Cortland completed its acquisition.",
                          snippet="", date="2025-11-12", url=URL, matched_icp_signal=0)
    with use_evaluation_date("2026-11-10"):
        score = await lead_scorer._score_single_intent_signal(signal, icp, None, "Cortland", "https://cortland.com",
            company_linkedin="https://www.linkedin.com/company/cortland-", stage1_soft_reject=True,
            llm_only_intent_gate=True, integrity_policy=True, verified_company_identity=IDENTITY)
    assert (score[0] > 0) is (model_state == "supported")
    assert FACT in judge.await_args.args[2]
    assert judge.await_args.args[1] == DEFAULT_JUDGE_MODELS["intent_fetched_evidence"]
