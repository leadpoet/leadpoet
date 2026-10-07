"""Cached review evidence must leave room for a fresh commercial locator."""
from __future__ import annotations

import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _finding

HOME = "https://acme.example/"
STAGE = "https://acme.example/news/series-b"
TERMS = "https://acme.example/terms-of-use"
PRICING = "https://acme.example/pricing"
HQ = "https://acme.example/contact"
STAGE_QUOTE = "Acme announced $72M in Series B funding."
TERMS_QUOTE = "Acme sells its security platform on annual renewable subscriptions."


def run_case(monkeypatch, *, stage_cached=True, geography=False, pricing=False,
             terms=True, third_cached=False, dispute=False, actions=None,
             attribute_proven=True, expire_after_fetch=False, long_text=False):
    pages = {HOME: {"final_url": HOME, "text": "Acme supplies a security platform."}}
    if stage_cached:
        pages[STAGE] = {"final_url": STAGE, "text": STAGE_QUOTE}
    if third_cached:
        pages[HOME + "saved"] = {"final_url": HOME + "saved", "text": "Saved evidence."}
    navigation = []
    if terms:
        navigation.append({"url": TERMS, "label": "Terms of use"})
    if pricing:
        navigation.append({"url": PRICING, "label": "Pricing"})
    if geography:
        navigation.append({"url": HQ, "label": "Contact"})
    prior = {"submitted_source_urls": list(pages) if stage_cached else [HOME, STAGE],
             "untrusted_company_stage_evidence": [{"url": STAGE, "quote": STAGE_QUOTE}]}
    if dispute:
        prior["stage_dispute_urls"] = [STAGE]
    targets = ["stage", "industry", "required_attribute"]
    if geography:
        targets.append("geography")
    requests, fetched, queries = [], [], []
    clock = [0.0]
    if expire_after_fetch:
        monkeypatch.setattr(investigator.time, "monotonic", lambda: clock[0])
    selected_terms = PRICING if pricing else TERMS
    findings = [_finding("stage", observed_value="Series B", evidence_url=STAGE,
                         evidence_quote=STAGE_QUOTE),
                _finding("industry", activity_role="supplier_operator", observed_value=None,
                         observed_industry="Security", evidence_url=HOME,
                         evidence_quote=pages[HOME]["text"]),
                _finding("required_attribute", activity_role="supplier_operator" if attribute_proven else "unresolved",
                         status="VERIFIED" if attribute_proven else "UNPROVEN", observed_value=None,
                         evidence_url=selected_terms if attribute_proven else "",
                         evidence_quote=TERMS_QUOTE if attribute_proven else "")]
    if geography:
        findings.append(_finding("geography", status="UNPROVEN", observed_value=None,
                                 evidence_url="", evidence_quote=""))
    actions = actions or [("submit_findings", {"findings": findings})]

    async def post(_session, _url, *, headers, payload):
        name, args = actions[len(requests)]
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    async def fetch(_session, url, **kwargs):
        fetched.append(url)
        if expire_after_fetch:
            clock[0] = investigator.ADMISSION_DEADLINE_SECONDS + 1
        text = STAGE_QUOTE if url == STAGE else TERMS_QUOTE
        if long_text and url in {TERMS, PRICING}:
            text = investigator._plain_text(
                "<html><body>" + text + (" details" * 10_000) + "</body></html>"
            )
        return {"ok": True, "url": url, "final_url": url, "text": text}

    async def search(_session, query, *, key):
        queries.append(query)
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": HOME}, targets=targets,
        requested_stage="Series B", requested_industry="Security",
        requested_product_service="Security platform", requested_attribute="Sells a subscription security platform",
        positive_semantic_review=True, prior_observations=prior,
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example", "linkedin_company_slug": "acme"},
        homepage_navigation_locators=navigation, prefetched_pages=pages))
    initial = (json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
               if requests else None)
    return result, initial, fetched, queries, requests


@pytest.mark.parametrize("pricing", [False, True])
def test_cached_series_b_and_homepage_admit_terms_with_pricing_preferred(monkeypatch, pricing):
    result, initial, fetched, queries, _ = run_case(monkeypatch, pricing=pricing)
    selected = PRICING if pricing else TERMS
    assert fetched == [selected]
    assert initial["server_priority_submitted_source"] == {
        "url": selected, "ok": True, "cache_hit": False,
        "notice": "server_selected_first_party_locator_is_untrusted_evidence"}
    assert {page["url"] for page in initial["prefetched_sources"]} == {HOME, STAGE, selected}
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == 1
    assert len(queries) == 1 and initial["server_current_stage_discovery"]["ok"] is True


def test_cached_disputed_stage_remains_loaded_and_current_search_required(monkeypatch):
    result, initial, fetched, queries, _ = run_case(monkeypatch, dispute=True)
    assert fetched == [TERMS]
    assert any(page == {"url": STAGE, "text": STAGE_QUOTE}
               for page in initial["prefetched_sources"])
    assert initial["prior_observations"]["stage_dispute_urls"] == [STAGE]
    assert len(queries) == 1
    assert result["claims"]["stage"]["status"] == "VERIFIED"


@pytest.mark.parametrize("geography,expected", [(False, STAGE), (True, HQ)])
def test_fresh_stage_or_headquarters_preserves_existing_priority(monkeypatch, geography, expected):
    _, initial, fetched, queries, _ = run_case(monkeypatch, stage_cached=False, geography=geography,
                                              attribute_proven=False)
    assert fetched == [expected]
    assert initial["server_priority_submitted_source"]["url"] == expected
    assert len(queries) == 1


@pytest.mark.parametrize("third_cached,terms", [(False, False), (True, True)])
def test_no_admissible_fresh_source_keeps_cached_stage_marker(monkeypatch, third_cached, terms):
    result, initial, fetched, queries, _ = run_case(monkeypatch, third_cached=third_cached,
                                                  terms=terms, attribute_proven=False)
    assert fetched == []
    assert initial["server_priority_submitted_source"]["url"] == STAGE
    assert initial["server_priority_submitted_source"]["cache_hit"] is True
    assert result["usage"]["fetch_calls"] == 0
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"
    assert len(queries) == 1


def test_fresh_locator_does_not_force_attribute_acceptance(monkeypatch):
    result, _, fetched, _, _ = run_case(monkeypatch, attribute_proven=False)
    assert fetched == [TERMS]
    assert result["claims"]["required_attribute"]["status"] == "UNPROVEN"


def test_server_terms_fetch_counts_against_three_call_cap(monkeypatch):
    fresh = [HOME + f"fresh-{i}" for i in range(3)]
    actions = [("fetch_page", {"url": url}) for url in fresh]
    actions.append(("submit_findings", {"findings": [
        _finding("stage", status="UNPROVEN", observed_value=None, evidence_url="", evidence_quote=""),
        _finding("industry", status="UNPROVEN", observed_value=None, evidence_url="", evidence_quote=""),
        _finding("required_attribute", status="UNPROVEN", observed_value=None, evidence_url="", evidence_quote=""),
    ]}))
    result, _, fetched, _, requests = run_case(monkeypatch, actions=actions)
    assert fetched == [TERMS, *fresh[:2]]
    assert result["usage"]["fetch_calls"] == 3
    assert json.loads(requests[-1]["messages"][-1]["content"])["error"] == "fetch_budget_exhausted"
    assert (investigator.MAX_FETCH_CALLS, investigator.MAX_PAGE_CHARACTERS,
            investigator.ADMISSION_DEADLINE_SECONDS) == (3, 24_000, 110.0)


def test_terms_fetch_does_not_extend_admission_deadline(monkeypatch):
    result, initial, fetched, queries, requests = run_case(monkeypatch, expire_after_fetch=True)
    assert fetched == [TERMS]
    assert queries == requests == []
    assert initial is None
    assert result["usage"]["reasoning_turns"] == 0
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


def test_fresh_terms_keeps_page_and_model_message_text_bounds(monkeypatch):
    result, initial, fetched, _, requests = run_case(monkeypatch, long_text=True)
    assert fetched == [TERMS]
    terms = next(page for page in initial["prefetched_sources"] if page["url"] == TERMS)
    assert len(terms["text"]) <= investigator.MAX_PAGE_CHARACTERS
    assert terms["text"].startswith(TERMS_QUOTE)
    assert len(requests[0]["messages"][1]["content"]) <= investigator.OPENROUTER_MAX_CONTENT_CHARS
    assert len(result[investigator.PRIVATE_FETCHED_PAGES_KEY][TERMS]["text"]) == 24_000
