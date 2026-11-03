"""Run the native fetch, source validation and industry projection without paid I/O.

These controls prove bounded PDF text recovery, not live provider coverage or
classification accuracy. The original binary document never becomes evidence.
"""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_refusal_fallback import _install_loop, URL


QUOTE = "Coreworks supplies analysis and reporting software to customers."


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
@pytest.mark.parametrize("document", ["%PDF-1.7 binary", "JVBERi0xLjcKbase64"])
def test_broker_pdf_refusal_preserves_bill_and_native_exact_url_recovery(
    monkeypatch, funding_source, document,
):
    from tests.lab_arena.score_compat_capability_test import (
        firecrawl_catalog, routed_broker, score_context,
    )
    from tests.lab_arena.test_lab_arena_broker import FakeTransport

    transport = FakeTransport([(200, {
        "status": "completed", "billing": {"credits_charged": 0.02},
        "result": {"data": {"rawHtml": document, "metadata": {
            "url": URL, "sourceURL": URL, "statusCode": 200,
            "contentType": "application/pdf"}}},
    })])
    broker, store, _ = routed_broker(funding_source=funding_source, transport=transport)
    run, calls, _ = _install_loop(
        monkeypatch, target="industry", finding=_industry_finding(),
        contents=(200, {"results": [{"url": URL, "text": QUOTE}]}),
    )
    receipts = []

    async def native_broker_fetch(_session, _url, **kwargs):
        requested = kwargs["params"]["url"]
        result = broker.execute(
            score_context(firecrawl_catalog(parsers={"type": "array"})),
            operation_id="scrapingdog.scrape", parameters={"url": requested},
            action_sequence=1, timeout_ms=30_000,
        )
        receipts.append(result)
        assert result.status == 502
        raise c._ArenaProviderFetchError(
            "source unavailable", broker_status=result.status,
            broker_error_code=json.loads(result.body)["error"]["code"],
        )

    monkeypatch.setattr(c, "_fetch_bounded_html", native_broker_fetch)
    result = asyncio.run(run())
    assert len(transport.sent) == len(store.calls) == len(receipts) == 1
    assert receipts[0].call["actual_microusd"] > 0
    assert json.loads(transport.sent[0]["body"])["payload"]["parsers"] == []
    assert len(calls["contents"]) == 1
    assert result["usage"]["fetch_calls"] == 2
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == QUOTE



@pytest.mark.parametrize("suffix", ["/specification.pdf", "/products"])
def test_exact_source_recovery_sets_bounded_live_crawl_timeout(monkeypatch, suffix):
    """A ten-second provider default must not silently govern a longer read."""
    url = "https://www.coreworks.ai" + suffix
    calls = []
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    async def post(_session, endpoint, **kwargs):
        calls.append((endpoint, kwargs["payload"]))
        return 200, {"results": [{"url": url, "text": QUOTE}]}

    monkeypatch.setattr(c, "_post_json", post)
    result = asyncio.run(c._fetch_exa_exact_page(None, url))
    assert result["ok"] is True
    assert calls == [("https://api.exa.ai/contents", {
        "ids": [url], "text": {"maxCharacters": c.MAX_PAGE_CHARACTERS},
        "maxAgeHours": 0, "livecrawlTimeout": 20_000,
    })]


def test_pdf_locator_uses_existing_text_provider_without_html_rendering(monkeypatch):
    url = 'https://www.coreworks.ai/brochure.PDF'
    finding = _finding('industry', status='VERIFIED', evidence_url=url,
                       evidence_quote=QUOTE, activity_role='supplier_operator',
                       observed_industry='Software')
    run, calls, _ = _install_loop(monkeypatch, status=200, body='', urls=(url,),
        target='industry', finding=finding,
        contents=(200, {'results': [{'url': url, 'text': QUOTE}]}))
    result = asyncio.run(run())
    assert calls['primary'] == []
    assert len(calls['contents']) == result['usage']['fetch_calls'] == 1
    assert result['claims']['industry']['status'] == 'VERIFIED'



@pytest.mark.parametrize('case', ['recover', 'cap', 'deadline', 'denied'])
def test_pdf_suffix_does_not_erase_recoverable_html_or_bypass_limits(monkeypatch, case):
    url = 'https://www.coreworks.ai/brochure.pdf'
    urls = tuple(f'https://www.coreworks.ai/page{i}' for i in range(3)) + (url,) if case == 'cap' else (url,)
    finding = _finding('industry', status='VERIFIED', evidence_url=url,
                       evidence_quote=QUOTE, activity_role='supplier_operator',
                       observed_industry='Software')
    contents = (200, {'results': [{'url': url, 'text': ''}]})
    if case == 'denied':
        contents = (200, {'results': [], 'statuses': [{'id': url, 'status': 'error',
            'error': {'httpStatusCode': 403, 'tag': 'CRAWL_HTTP_403'}}]})
    run, calls, clock = _install_loop(monkeypatch, status=200, body='<p>' + QUOTE + '</p>',
        urls=urls, target='industry', finding=finding, contents=contents)
    original = c._post_json
    async def post(session, endpoint, **kwargs):
        result = await original(session, endpoint, **kwargs)
        if case == 'deadline' and endpoint == 'https://api.exa.ai/contents':
            clock.now = c.ADMISSION_DEADLINE_SECONDS
        return result
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run())
    assert len(calls['contents']) == 1
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    if case == 'recover':
        assert calls['primary'] == [url]
        assert result['usage']['fetch_calls'] == 2
        assert result['claims']['industry']['status'] == 'VERIFIED'
        assert result['_completed_submit'] is True
    else:
        assert url not in calls['primary']
        assert result['claims']['industry']['status'] != 'VERIFIED'


def _industry_finding(status="VERIFIED", quote=QUOTE):
    return _finding(
        "industry", status=status, evidence_url=URL, evidence_quote=quote,
        activity_role="supplier_operator", observed_industry="Software",
    )


@pytest.mark.parametrize("positive_review", [False, True])
def test_industry_pdf_recovers_exact_url_text_within_existing_fetch_budget(monkeypatch, positive_review):
    run, calls, _ = _install_loop(
        monkeypatch, status=200, body="%PDF-1.7\nbinary document",
        target="industry", finding=_industry_finding(),
        contents=(200, {"results": [{"url": URL, "text": QUOTE}]}),
    )
    original = c.investigate_company_evidence

    async def invoke(**kwargs):
        kwargs["positive_semantic_review"] = positive_review
        return await original(**kwargs)

    monkeypatch.setattr(c, "investigate_company_evidence", invoke)
    result = asyncio.run(run())
    assert calls["primary"] == [URL]
    assert len(calls["contents"]) == 1
    assert result["usage"]["fetch_calls"] == 2
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == QUOTE
    verdict = {"stage": "original", "attribute_satisfied": False}
    projected = lead_scorer._project_investigator_industry(verdict, result["claims"]["industry"])
    assert projected["industry_matches"] is True
    assert projected["industry_evidence_quote"] == QUOTE
    assert projected["stage"] == "original" and projected["attribute_satisfied"] is False


@pytest.mark.parametrize("body", ["\x89PNG\r\n", "PK\x03\x04binary archive"])
def test_other_binary_documents_do_not_start_pdf_recovery(monkeypatch, body):
    run, calls, _ = _install_loop(monkeypatch, status=200, body=body,
                                 target="industry", finding=_industry_finding("UNPROVEN"))
    asyncio.run(run())
    assert not calls["contents"]


@pytest.mark.parametrize("case,urls", [
    ("missing_key", (URL,)),
    ("deadline", (URL,)),
    ("", ("https://www.coreworks.ai/a", "https://www.coreworks.ai/b", "https://www.coreworks.ai/c", URL)),
])
def test_pdf_recovery_preserves_key_deadline_and_fetch_limits(monkeypatch, case, urls):
    run, calls, _ = _install_loop(monkeypatch, status=200, body="%PDF-1.7 binary",
        case=case, urls=urls, target="industry", finding=_industry_finding("UNPROVEN"))
    result = asyncio.run(run())
    assert not calls["contents"]
    assert result["usage"]["fetch_calls"] <= c.MAX_FETCH_CALLS


def test_pdf_does_not_expand_research_for_unrelated_targets(monkeypatch):
    run, calls, _ = _install_loop(monkeypatch, status=200, body="%PDF-1.7 binary")
    asyncio.run(run())
    assert not calls["contents"]


@pytest.mark.parametrize("contents", [
    {"results": [{"url": URL, "text": "%PDF-1.7 still binary"}]},
    {"results": [{"url": "https://other.example/", "text": QUOTE}]},
    {"results": [{"url": URL, "text": "Other Company sells something else."}]},
])
def test_pdf_recovery_cannot_supply_a_foreign_or_absent_quote(monkeypatch, contents):
    run, calls, _ = _install_loop(monkeypatch, status=200, body="%PDF-1.7 binary",
        target="industry", finding=_industry_finding(), contents=(200, contents))
    result = asyncio.run(run())
    assert len(calls["contents"]) == 1
    assert result["claims"]["industry"]["status"] != "VERIFIED"
    assert result["usage"]["fetch_calls"] == 2
