"""Original source bodies survive beyond the bounded model-visible passages."""
import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from lab_arena import operations
from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import lead_scorer, intent_details
from tests.test_company_evidence_investigator import _attribute_finding
from tests.test_matched_company_retry_source_continuity import (
    IDENTITY, PLATFORM_URL, PLATFORM_QUOTE, _fit_result,
)

URL = "https://acme.example/terms"
FEE = "Acme charges customers an annual platform subscription fee."
NEGATION = "Acme does not charge customers an annual platform subscription fee."


def _body(quote=FEE):
    return "Acme terms. " + "Standard contract condition. " * 1610 + quote


def _response(name, args, turn):
    return 200, {"choices": [{"message": {"tool_calls": [{
        "id": f"call-{turn}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}}]}


@pytest.mark.parametrize("quote,status", [(FEE, "VERIFIED"), (NEGATION, "CONTRADICTED")])
@pytest.mark.parametrize("prefetched", [False, True])
def test_late_fee_or_negation_survives_full_runtime_with_one_source_read(monkeypatch, quote, status, prefetched):
    body = c._plain_text("<html><body>" + _body(quote) + "</body></html>")
    assert body.index(quote) > c.MAX_PAGE_CHARACTERS
    requests = []
    fetch = AsyncMock(return_value={"ok": True, "url": URL, "final_url": URL, "text": body})

    async def post(_session, _url, *, headers, payload):
        operations.validate_operation_request("openrouter.chat", payload)
        requests.append(deepcopy(payload))
        shown = []
        document = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        shown.extend(document.get("prefetched_sources", []))
        shown.extend(json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool")
        for page in shown:
            if "text" in page:
                assert len(page["text"]) <= c.MAX_PAGE_CHARACTERS
                if "offset" in page:
                    assert page["text"] == body[page["offset"]:page["offset"] + len(page["text"])]
        if not shown:
            return _response("fetch_page", {"url": URL, "offset": None}, len(requests))
        last = shown[-1]
        if quote not in last["text"]:
            assert last["next_offset"] is not None
            return _response("fetch_page", {"url": URL, "offset": last["next_offset"]}, len(requests))
        return _response("submit_findings", {"findings": [
            _attribute_finding(status, url=URL, quote=quote),
        ]}, len(requests))

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("required_attribute",), requested_attribute="Charges a subscription fee",
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": body}} if prefetched else None,
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example", "linkedin_company_slug": "acme"},
    ))
    assert result.get("claims", {}).get("required_attribute", {}).get("status") == status, result
    assert result[c.PRIVATE_FETCHED_PAGES_KEY][URL]["text"] == body
    assert fetch.await_count == (0 if prefetched else 1)
    assert result["usage"]["fetch_calls"] == fetch.await_count
    assert result["usage"]["reasoning_turns"] <= c.MAX_REASONING_TURNS == 8
    assert c.MAX_FETCH_CALLS == 4 and c.MAX_SEARCH_CALLS == 2
    assert c.ADMISSION_DEADLINE_SECONDS == 110


def test_short_pages_preserve_existing_response_and_exact_quote_guards():
    page = {"ok": True, "url": URL, "text": FEE}
    assert c._page_window(page) == page
    body = _body()
    pages = {URL: body, "https://acme.example/other": "Other source"}
    assert c._quote_occurs(FEE, pages[URL])
    assert not c._quote_occurs(FEE, pages["https://acme.example/other"])
    assert not c._quote_occurs("Acme terms. " + FEE, body)
    assert not c._quote_occurs(FEE, c._plain_text("<script>" + FEE + "</script><p>Acme visible</p>"))
    assert not c._quote_occurs(FEE, c._plain_text('<div hidden>' + FEE + '</div><p>Acme visible</p>'))


@pytest.mark.parametrize("offset", [-1, True, False, 1.5, "24000", 96000, 48000])
def test_invalid_or_out_of_range_cache_offsets_fail_closed(offset):
    assert c._page_window({"url": URL, "text": "x" * 48000}, offset).get("ok") is False


def test_retained_source_cap_and_multibyte_message_offsets():
    assert len(c._plain_text("x" * (c.MAX_SOURCE_CHARACTERS + 1))) == c.MAX_SOURCE_CHARACTERS
    assert c._validated_prefetched_pages(
        {URL: {"final_url": URL, "text": "x" * (c.MAX_SOURCE_CHARACTERS + 1)}},
        submitted_source_urls=[URL],
    ) == ({}, {})
    body = "漢字😀 " * 20000
    pages = [c._page_window({"url": URL + str(i), "text": body}) for i in range(3)]
    content = c._bounded_message_json({"prefetched_sources": pages})
    operations.validate_operation_request("openrouter.chat", {
        "model": c.INVESTIGATOR_MODEL, "messages": [{"role": "user", "content": content}],
    })
    assert len(content) < operations.OPENROUTER_MAX_CONTENT_CHARS
    for page in json.loads(content)["prefetched_sources"]:
        assert page["next_offset"] == len(page["text"])
        assert 0 < page["next_offset"] < c.MAX_PAGE_CHARACTERS
        assert page["available_length"] == len(body)
        assert page["text"] == body[:page["next_offset"]]


def test_long_source_retention_grounding_and_saved_json_roundtrip():
    body = "Unibuddy " + "Earlier material. " * 2600 + PLATFORM_QUOTE
    pages = {PLATFORM_URL: {"final_url": PLATFORM_URL, "text": body}}
    investigation = {c.PRIVATE_FETCHED_PAGES_KEY: pages}
    retry = {}
    lead_scorer._retain_successful_required_attribute_source(retry, PLATFORM_URL, {
        "status": "fetched", "final_url": PLATFORM_URL, "text": body,
    })
    assert lead_scorer._validated_retry_retained_sources(json.loads(json.dumps(retry)))[PLATFORM_URL]["text"] == body
    assert lead_scorer._reviewed_required_attribute_source_cache(
        investigation, {"evidence_url": PLATFORM_URL, "evidence_quote": PLATFORM_QUOTE}, {},
    )[PLATFORM_URL]["text"] == body
    sink = {}
    fit = _fit_result()
    lead_scorer._retain_matched_investigator_source_contexts(sink, fit, investigation)
    saved = {}
    lead_scorer._retain_matched_company_retry_sources(saved, sink, fit)
    assert lead_scorer._matched_company_retry_prefetched_pages(json.loads(json.dumps(saved)), IDENTITY) == pages
    contexts = lead_scorer._matched_investigator_source_contexts(fit, sink)
    assert contexts[0]["text"] == body
    window = intent_details._continuous_source_window(body, 12000, [PLATFORM_QUOTE])
    assert PLATFORM_QUOTE in window and window in body and len(window.encode()) <= 12000
    current = {PLATFORM_URL: {"status": "fetched", "final_url": PLATFORM_URL, "text": "Fresh current source."}}
    lead_scorer._hydrate_required_attribute_source_cache(current, investigation)
    assert current[PLATFORM_URL]["text"] == "Fresh current source."


def test_exa_reports_only_actual_returned_text_with_original_upstream_cap(monkeypatch):
    async def post(_session, _url, *, headers, payload):
        assert payload["text"]["maxCharacters"] == c.MAX_PAGE_CHARACTERS
        return 200, {"results": [{"url": URL, "text": "x" * c.MAX_PAGE_CHARACTERS}]}
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    result = asyncio.run(c._fetch_exa_exact_page(object(), URL))
    window = c._page_window(result)
    assert len(window["text"]) == c.MAX_PAGE_CHARACTERS
    assert window.get("next_offset") is None
    assert "available_length" not in window


@pytest.mark.parametrize("offset", [True, -1, 96000, 48000])
def test_bad_runtime_offsets_never_admit_another_fetch(monkeypatch, offset):
    requests = []
    fetch = AsyncMock(side_effect=AssertionError("cached continuation cannot fetch"))

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        if len(requests) == 1:
            return _response("fetch_page", {"url": URL, "offset": offset}, 1)
        result = json.loads(payload["messages"][-1]["content"])
        assert result == {"ok": False, "error": "invalid_cached_offset"}
        return _response("submit_findings", {"findings": [_attribute_finding("VERIFIED", url=URL, quote=FEE)]}, 2)

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("required_attribute",), requested_attribute="Charges a subscription fee",
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": FEE + "x" * 40000}},
    ))
    assert result.get("_completed_submit"), (result, requests[-1]["messages"][-1])
    assert result["usage"]["fetch_calls"] == fetch.await_count == 0
    assert result["usage"]["reasoning_turns"] == 2


def test_continuation_uses_ordinary_turns_without_reopening_full_fetch_budget(monkeypatch):
    requests = []
    urls = [f"https://acme.example/page{i}" for i in range(5)]
    body = _body()
    fetch = AsyncMock(side_effect=lambda _session, url, **kwargs: {
        "ok": True, "url": url, "final_url": url, "text": body,
    })

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        turn = len(requests)
        if turn <= 4:
            return _response("fetch_page", {"url": urls[turn - 1], "offset": None}, turn)
        if turn == 5:
            return _response("fetch_page", {"url": urls[0], "offset": 24000}, turn)
        if turn == 6:
            result = json.loads(payload["messages"][-1]["content"])
            assert result["cache_hit"] and result["offset"] == 24000
            return _response("fetch_page", {"url": urls[4], "offset": None}, turn)
        assert json.loads(payload["messages"][-1]["content"])["error"] == "fetch_budget_exhausted"
        return _response("submit_findings", {"findings": [_attribute_finding()]}, turn)

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    result = asyncio.run(c.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("required_attribute",), requested_attribute="Charges a subscription fee",
    ))
    assert result.get("_completed_submit"), result
    assert result["usage"]["fetch_calls"] == fetch.await_count == 4
    assert result["usage"]["reasoning_turns"] == 7


def test_aggregate_trim_of_three_short_prefetches_exposes_exact_continuation():
    quote = "Decisive original clause at character fifteen thousand."
    body = "x" * 15000 + quote + "y" * (20000 - 15000 - len(quote))
    pages = [c._page_window({"url": URL + str(i), "text": body}) for i in range(3)]
    assert all("next_offset" not in page for page in pages)
    shown = json.loads(c._bounded_message_json({"prefetched_sources": pages}))["prefetched_sources"]
    for page in shown:
        assert 0 < page["next_offset"] == len(page["text"]) < 15000
        assert page["available_length"] == len(body) == 20000
        later = c._page_window({"url": page["url"], "text": body}, page["next_offset"])
        assert quote in later["text"]
        assert later["text"] == body[page["next_offset"]:]
    short = {"url": URL, "text": "Untrimmed original short page."}
    assert json.loads(c._bounded_message_json(short)) == short
