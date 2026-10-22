"""Exact fetched page metadata survives the bounded paragraph handoff."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from qualification.scoring import intent_details, verification_helpers
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.lead_scorer import _score_single_intent_signal
from tests.test_intent_source_context_transition import (
    SOURCE_TEXT, SOURCE_URL, _icp, _provider_answer, _signal,
)


TITLE = "ThreatSpike Product Update: March 2026 - ThreatSpike"
URL = "https://www.threatspike.com/blog/threatspike-march-updates/"
QUOTE = (
    "Another SaaS integration for the books! Incidents can now be automatically "
    "generated in the portal from email addresses in ProofPoint alerts."
)
PARAGRAPH = (
    "A report dated April 2, 2026 says ThreatSpike Product Update: March 2026, "
    "noting that incidents can now integrate with ProofPoint SaaS. "
    "This update may help ThreatSpike expand its security software."
)


def _case(*, title=TITLE, text=QUOTE):
    company = SimpleNamespace(
        company_name="ThreatSpike", company_website="https://threatspike.com/",
        intent_details=PARAGRAPH,
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0, description="Submitted invented title",
            url=URL,
        )],
    )
    icp = SimpleNamespace(
        prompt="Security companies with recent product capabilities",
        product_service="Security software",
        intent_signals=["Launched a new security capability in the last year"],
    )
    signals = [{
        "after_decay": 54.0, "matched_icp_signal": 0,
        "evidence_urls": [URL],
        "judge_verdict": {
            "decision": "verified", "client_ready": True,
            "authoritative_date": "2026-04-02",
            "authoritative_date_basis": "publication_date",
            "verification_trace": {
                "verified_source_context": [{
                    "url": URL, "text": text, "source_title": title,
                    "source_publication_date": "2026-04-02",
                }],
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported", "same_entity_check": "pass",
                    "supporting_quotes": [QUOTE], "evidence_urls_used": [URL],
                }]},
            },
        },
    }]
    return company, icp, signals, {"decision": "match", "dimension_evidence": {}}


def _document(**kwargs):
    return intent_details.review_evidence(*_case(**kwargs))


def test_exact_production_paragraph_retains_title_body_and_publication_identity():
    document = _document()
    assert " ".join(unit["text"] for unit in document["intent_details_units"]) == PARAGRAPH
    source = document["admitted_evidence"][0]
    assert source["source_url"] == URL
    assert "source_title" not in source
    assert source["admitted_text"] == [QUOTE, TITLE]
    assert source["observed_dates"] == [{
        "date": "2026-04-02", "basis": "source_publication_date",
    }]
    values = intent_details._bound_evidence_sources(document)[0]
    assert all(intent_details._quote_is_bound(quote, values)
               for quote in (TITLE, QUOTE, "2026-04-02"))
    assert not intent_details._quote_is_bound("Submitted invented title", values)
    assert not intent_details._quote_is_bound(TITLE + " " + QUOTE, values)
    assert json.loads(json.dumps(document)) == document


def _title_body_date_response():
    return {
        "unit_grounding": [{
            "unit_id": 0, "contains_factual_claim": True, "status": "VERIFIED",
            "evidence": [{"source_index": 0, "quote": quote}
                         for quote in (TITLE, QUOTE, "2026-04-02")],
        }, {
            "unit_id": 1, "contains_factual_claim": False, "status": "VERIFIED",
            "evidence": [],
        }],
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in intent_details._CHECKS},
    }


def test_three_binding_limit_is_consistent_in_initial_and_repair_contracts():
    response = _title_body_date_response()
    initial = intent_details._RESPONSE_FORMAT["json_schema"]["schema"][
        "properties"
    ]["unit_grounding"]["items"]["properties"]["evidence"]
    assert initial["maxItems"] == 3
    for response_format in (intent_details._CITATION_REPAIR_RESPONSE_FORMAT,
                            intent_details._SEMANTIC_REPAIR_RESPONSE_FORMAT):
        evidence = response_format["json_schema"]["schema"]["properties"][
            "repairs"
        ]["items"]["properties"]["evidence"]
        assert evidence["maxItems"] == 3
    citation_prompt = intent_details._citation_repair_prompt(
        intent_details._SYSTEM, {0: {"too_many_quotes"}}, response,
    )
    semantic_prompt = intent_details._semantic_repair_prompt(
        intent_details._SYSTEM, {0: {"too_many_quotes"}}, response, {0},
    )
    assert "one to three evidence bindings" in intent_details._SYSTEM
    assert "at most three bindings" in citation_prompt
    assert "one to\nthree source_index/quote bindings" in semantic_prompt
    assert "at most three\nbindings" in semantic_prompt
    assert "if three bindings cannot support all clauses, do not return VERIFIED" in semantic_prompt
    for prompt in (intent_details._SYSTEM, citation_prompt, semantic_prompt):
        assert "one or two" not in prompt
        assert "at most two" not in prompt
    assert "returned evidence includes a binding to the relevant admitted publication" in (
        intent_details._SYSTEM
    )
    assert "VERIFIED must cite that date proof and substantive" in semantic_prompt


@pytest.mark.parametrize("index,invented", [
    (0, "ThreatSpike Product Update: December 2026"),
    (1, "AnotherCompany automatically generates incidents from Microsoft alerts."),
    (2, "2026-04-03"),
])
def test_each_of_three_title_body_date_bindings_still_requires_exact_evidence(index, invented):
    response = _title_body_date_response()
    response["unit_grounding"][0]["evidence"][index]["quote"] = invented
    _all_verified, _has_fact, issues, *_ = intent_details._validate_unit_grounding(
        response["unit_grounding"], _document(),
    )
    assert issues == {0: {"nonexact_quote"}}


@pytest.mark.parametrize("initial_excess,repair_excess,decision,call_count", [
    (False, False, "match", 1),
    (True, False, "match", 2),
    (True, True, "unavailable", 2),
])
def test_three_title_body_date_bindings_pass_and_four_need_one_bounded_repair(
    monkeypatch, initial_excess, repair_excess, decision, call_count,
):
    calls = []

    async def judge(_prompt, **kwargs):
        calls.append(kwargs)
        assert kwargs["max_retries"] == 0
        assert kwargs["max_tokens"] == 800
        response = _title_body_date_response()
        if (len(calls) == 1 and initial_excess) or (len(calls) == 2 and repair_excess):
            response["unit_grounding"][0]["evidence"].append({
                "source_index": 0, "quote": "ProofPoint alerts.",
            })
        if len(calls) == 1:
            assert kwargs["response_format"] == intent_details._RESPONSE_FORMAT
            return json.dumps(response)
        assert len(calls) == 2
        assert kwargs["response_format"] == intent_details._CITATION_REPAIR_RESPONSE_FORMAT
        assert "too_many_quotes" in kwargs["system_prompt"]
        return json.dumps({"repairs": [{
            "unit_id": 0, "evidence": response["unit_grounding"][0]["evidence"],
        }]})

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(*_case()))
    assert receipt["decision"] == decision
    assert len(calls) == call_count
    if decision == "unavailable":
        assert receipt["failure_class"] == "intent_details_citation_unavailable"
        assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("title", [None, [], {}, 32, "", " ", "hidden\x00title", "é" * 513])
def test_malformed_or_oversize_titles_never_enter_admitted_evidence(title):
    document = _document(title=title)
    assert document["admitted_evidence"][0]["admitted_text"] == [QUOTE]
    assert QUOTE in intent_details._bound_evidence_sources(document)[0]


@pytest.mark.parametrize("invented", [
    "ThreatSpike Product Update: December 2026",
    "AnotherCompany Product Update: March 2026",
    "2026-04-03",
    "Incidents integrate with ProofPoint and the company acquired Microsoft.",
    TITLE + " " + QUOTE,
])
def test_fabricated_title_entity_date_and_stitched_claims_fail_exact_binding(invented):
    document = _document()
    grounding = [{
        "unit_id": 0, "contains_factual_claim": True, "status": "VERIFIED",
        "evidence": [{"source_index": 0, "quote": invented}],
    }, {
        "unit_id": 1, "contains_factual_claim": False, "status": "VERIFIED",
        "evidence": [],
    }]
    _all_verified, _has_fact, issues, *_ = intent_details._validate_unit_grounding(
        grounding, document,
    )
    assert issues == {0: {"nonexact_quote"}}


def test_unrelated_page_title_does_not_cross_source_identity_boundary():
    company, icp, signals, fit = _case()
    signals[0]["judge_verdict"]["verification_trace"]["verified_source_context"][0]["url"] = (
        "https://other.example/unrelated"
    )
    document = intent_details.review_evidence(company, icp, signals, fit)
    assert not any(intent_details._quote_is_bound(TITLE, values)
                   for values in intent_details._bound_evidence_sources(document).values())


def test_generic_prompt_title_is_not_fetched_title_proof():
    projected = verifier._project_contents_for_prompt({"results": [{
        "url": URL, "text": QUOTE,
        "title": "Server-retained MATCH stage source (not event identity proof)",
    }]})
    assert "source_title" not in projected["results"][0]


def test_paragraph_window_keeps_whole_relevant_tail_within_title_and_body_budget():
    company, icp, signals, fit = _case(text="Unrelated navigation. " * 450 + QUOTE)
    # Force the existing fair split: each activity gets 6 KB, including title.
    signals.append(deepcopy(signals[0]))
    signals[1]["matched_icp_signal"] = 1
    document = intent_details.review_evidence(company, icp, signals, fit)
    sources = [item for item in document["admitted_evidence"]
               if item["evidence_kind"] == "verified_source_context"]
    original = signals[0]["judge_verdict"]["verification_trace"]["verified_source_context"][0]["text"]
    assert len(sources) == 2
    assert all(QUOTE in item["admitted_text"][0] for item in sources)
    assert all(item["admitted_text"][0] in original for item in sources)
    assert sum(len(text.encode()) for item in sources for text in item["admitted_text"]) <= (
        intent_details._MAX_SOURCE_CONTEXT_BYTES
    )
    assert len(json.dumps(document, ensure_ascii=False)) <= 48_000


@pytest.mark.parametrize("source_count", [2, 3])
@pytest.mark.parametrize("title", ["", "Short report title"])
@pytest.mark.parametrize("intent_count", [1, 2])
@pytest.mark.parametrize("company_reservation", [False, True])
def test_later_cited_page_survives_long_preceding_sources(
    source_count, title, intent_count, company_reservation,
):
    """A 12 KB first page cannot remove a later 199-byte cited page."""
    extra = "The integration also supports automatic incident routing."
    later_body = QUOTE + " " + extra
    assert len(later_body.encode()) == 199
    company, icp, signals, fit = _case(title=title)
    company.intent_details = (
        "ThreatSpike integrates with ProofPoint. " + extra
        + " This may support its security software."
    )
    trace = signals[0]["judge_verdict"]["verification_trace"]
    contexts = [{
        "url": URL if index == 0 else URL + f"?source={index}",
        "text": "é" * 6_000 if index < source_count - 1 else later_body,
        "source_title": title,
        "source_publication_date": "2026-04-02",
    } for index in range(source_count)]
    trace["verified_source_context"] = contexts
    urls = [context["url"] for context in contexts]
    trace["intent_verdict"]["signal_evaluations"][0]["evidence_urls_used"] = urls
    signals[0]["evidence_urls"] = urls
    if intent_count == 2:
        signals.append(deepcopy(signals[0]))
        signals[1]["matched_icp_signal"] = 1
    company_contexts = None
    if company_reservation:
        company_url = "https://threatspike.com/defensive-security/"
        company_quote = "ThreatSpike provides a commercial security platform."
        fit["dimension_evidence"]["industry"] = {
            "decision": "match", "web_evidence": {
                "url": company_url, "quote": company_quote,
            },
        }
        company_contexts = [{
            "dimension": "industry", "url": company_url,
            "text": company_quote + " Additional company context." * 300,
        }]
    document = intent_details.review_evidence(
        company, icp, signals, fit, company_source_contexts=company_contexts,
    )
    admitted = document["admitted_evidence"]
    intent_sources = [source for source in admitted
                      if source["evidence_kind"] == "verified_source_context"]
    intent_budget = 12_000 - (4_000 if company_reservation else 0)
    assert len(intent_sources) == source_count * intent_count
    for signal in document["verified_signals"]:
        sources = [admitted[index] for index in signal["evidence_source_indexes"]
                   if admitted[index]["evidence_kind"] == "verified_source_context"]
        assert [source["source_url"] for source in sources] == urls
        assert sources[-1]["admitted_text"][0] == later_body
        assert all(source["admitted_text"][0] in original["text"]
                   for source, original in zip(sources, contexts))
        assert sum(len(text.encode()) for source in sources
                   for text in source["admitted_text"]) <= intent_budget // intent_count
        if title:
            assert all(source["admitted_text"][1:] == [title] for source in sources)
        else:
            assert all(len(source["admitted_text"]) == 1 for source in sources)
    company_sources = [source for source in admitted
                       if source["evidence_kind"] == "verified_company_source_context"]
    assert bool(company_sources) is company_reservation
    if company_reservation:
        assert sum(len(text.encode()) for source in company_sources
                   for text in source["admitted_text"]) >= 4_000
    assert sum(len(text.encode()) for source in intent_sources + company_sources
               for text in source["admitted_text"]) <= 12_000
    assert len(json.dumps(document, ensure_ascii=False)) <= 48_000


@pytest.mark.parametrize("fetched_url, expected", [
    (URL, True), (URL + "/", True),
    (URL + "?article=different", False),
    ("https://other.example/article", False), (None, False),
])
def test_exa_title_requires_exact_contents_response_identity(monkeypatch, fetched_url, expected):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, endpoint, **kwargs):
            assert endpoint == "https://api.exa.ai/contents"
            assert kwargs["json"]["ids"] == [URL]
            return httpx.Response(200, json={"results": [{
                "url": fetched_url, "title": TITLE, "text": QUOTE * 4,
            }]})

    monkeypatch.setenv("EXA_API_KEY", "test-only")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", Client)
    result = asyncio.run(verifier._scrape_exa(URL))
    assert result["ok"]
    assert result.get("source_title", "") == (TITLE if expected else "")


def test_sd_dom_title_survives_body_extraction(monkeypatch):
    html = (
        "<!doctype html><html><head><title>" + TITLE + "</title></head>"
        "<body><article><p>ThreatSpike. " + QUOTE * 30 + "</p></article></body></html>"
    )

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _endpoint, **kwargs):
            assert kwargs["params"]["url"] == URL
            return httpx.Response(200, text=html)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-only")
    monkeypatch.setattr(verifier.httpx, "AsyncClient", Client)
    # Exercise the extraction handoff even when the article extractor omits
    # the head metadata, as happened in the saved production evidence.
    monkeypatch.setattr(
        "qualification.scoring.verification_helpers.extract_article_body",
        lambda _html: QUOTE * 30,
    )
    result = asyncio.run(verifier._scrape_sd_hardened(URL))
    assert result["ok"]
    assert result["source_title"] == TITLE
    assert TITLE not in result["content"]
    assert QUOTE in result["content"]


@pytest.mark.parametrize("provider", ["sd", "exa"])
def test_fetch_title_survives_real_score_receipt_and_review(monkeypatch, provider):
    fetched_title = "Acme names a new chief executive"
    sd = AsyncMock(return_value={"ok": provider == "sd", "content": SOURCE_TEXT,
                                 "source_title": fetched_title,
                                 "source_publication_date": "2026-09-14"})
    exa = AsyncMock(return_value={"ok": True, "content": SOURCE_TEXT,
                                  "source_title": fetched_title,
                                  "source_publication_date": "2026-09-14"})
    monkeypatch.setattr(verifier, "_scrape_sd_hardened", sd)
    monkeypatch.setattr(verifier, "_scrape_exa", exa)
    monkeypatch.setattr(verifier, "_call_openrouter", AsyncMock(
        return_value=_provider_answer("supported"),
    ))
    monkeypatch.setenv("LEADPOET_COMPETITION_EVALUATION_DATE", "2026-09-20")
    signal = _signal()
    verdicts = []
    score, _confidence, _status, _, matched = asyncio.run(_score_single_intent_signal(
        signal, _icp(), None, "Acme", "https://acme.com",
        company_linkedin="https://www.linkedin.com/company/acme",
        stage1_soft_reject=True, llm_only_intent_gate=True, integrity_policy=True,
        verdict_out=verdicts, evidence_signals=[signal],
    ))
    assert score == 54.0
    assert verdicts[-1]["verification_trace"]["verified_source_context"][0]["source_title"] == fetched_title
    company = SimpleNamespace(
        company_name="Acme", company_website="https://acme.com",
        intent_details="Acme named a new chief executive, which may support its reporting operations.",
        intent_signals=[signal],
    )
    document = intent_details.review_evidence(company, _icp(), [{
        "after_decay": score, "matched_icp_signal": matched,
        "evidence_urls": [SOURCE_URL], "judge_verdict": verdicts[-1],
    }], {"decision": "match", "dimension_evidence": {}})
    assert fetched_title in document["admitted_evidence"][0]["admitted_text"]
    assert "source_title" not in document["admitted_evidence"][0]
    assert intent_details._quote_is_bound(
        fetched_title, intent_details._bound_evidence_sources(document)[0],
    )
    assert exa.await_count == (provider == "exa")


def test_unavailable_source_cannot_supply_title_or_submitted_claim_evidence(monkeypatch):
    unavailable = {"ok": False, "content": "", "stage": "unavailable"}
    monkeypatch.setattr(verifier, "_scrape_sd_hardened", AsyncMock(return_value=unavailable))
    monkeypatch.setattr(verifier, "_scrape_exa", AsyncMock(return_value=unavailable))
    judge = AsyncMock()
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setenv("LEADPOET_COMPETITION_EVALUATION_DATE", "2026-09-20")
    signal = _signal()
    verdicts = []
    score, *_ = asyncio.run(_score_single_intent_signal(
        signal, _icp(), None, "Acme", "https://acme.com",
        company_linkedin="https://www.linkedin.com/company/acme",
        stage1_soft_reject=True, llm_only_intent_gate=True, integrity_policy=True,
        verdict_out=verdicts, evidence_signals=[signal],
    ))
    assert score == 0.0
    assert judge.await_count == 0
    assert "verified_source_context" not in verdicts[-1]["verification_trace"]
