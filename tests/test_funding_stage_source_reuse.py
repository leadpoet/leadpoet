"""Retained stage evidence informs the existing funding judge, not a new gate."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt, IntentSignal
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.evaluation_clock import use_evaluation_date


PRIMARY = "https://www.aurasell.ai/resources/aurasell-launches"
SECONDARY = "https://www.contentgrip.com/aurasell-ai-gtm-os-funding/"
QUOTE = (
    "The company has raised a $30 million Seed round from N47, Menlo Ventures, "
    "and Unusual Ventures."
)
STAGE_TEXT = "August 26, 2025. Aurasell launches. " + QUOTE
RECAP_TEXT = (
    "February 14, 2026. Aurasell launches an AI-native GTM OS. Alongside the "
    "launch, the company announced a $30 million seed round led by N47, "
    "Menlo Ventures, and Unusual Ventures."
)
IDENTITY = {
    "decision": "match", "evidence_source": "company_web_reverification",
    "observed_name": "aurasell", "observed_domain": "aurasell.ai",
    "observed_linkedin_slug": "aurasell",
}
CONTEXT = {"dimension": "stage", "url": PRIMARY, "text": STAGE_TEXT}


def _signal():
    return IntentSignal(
        source="news", description="Aurasell announced a $30 million Seed round.",
        url=SECONDARY, date="2026-02-14", snippet=RECAP_TEXT,
        matched_icp_signal=0,
    )


def _company():
    return CompanyOutput.model_construct(
        company_name="Aurasell", company_website="https://aurasell.ai/",
        company_linkedin="https://www.linkedin.com/company/aurasell",
        intent_signals=[_signal()], intent_details=None,
    )


def _icp(evidence_type="FUNDING"):
    return ICPPrompt(
        icp_id="funding-context", prompt="Find companies with a recent funding round",
        industry="Software", sub_industry="SaaS", employee_count="51-200", geography="United States",
        country="United States", product_service="Software", company_stage="Seed",
        intent_signals=["Announced funding in the past year"],
        intent_signal_evidence_types=[evidence_type], intent_max_age_days=365,
    )


def _fit():
    return company_fit_match(details={"dimension_evidence": {
        "identity": {"web_identity_receipt": IDENTITY},
        **{dimension: {
            "decision": "match", "web_evidence": {
                "url": f"https://aurasell.ai/{dimension}", "quote": dimension,
            },
        } for dimension in ("industry", "employee_size", "geography")},
        "stage": {"decision": "match", "web_evidence": {
            "url": PRIMARY, "quote": QUOTE,
        }},
    }})


def _answer(event_date, *, urls=None):
    return {"answer": {
        "signal_evaluations": [{
            "signal_id": "signal-1", "signal_status": "supported",
            "confidence": "high", "same_entity_check": "pass",
            "verification_mode": "source_grounded",
            "evidence_urls_used": urls if urls is not None else [SECONDARY, PRIMARY],
            "supporting_quotes": [QUOTE], "contradicting_quotes": [],
            "unsupported_parts": [], "claim_matches_miner_date": "consistent",
            "risk_notes": [f"source_event_date:{event_date}"] if event_date else [],
        }],
        "overall_verdict": "qualified", "overall_confidence": "high",
        "summary": "Same-event decision fixture", "missing_or_risks": [],
    }, "model": "test-model", "usage": {}}


def _run(monkeypatch, *, event_date="2025-08-26", context=None,
         body=RECAP_TEXT, evidence_type="FUNDING", urls=None, fetched=True,
         publication_date="", metadata=None, max_age_days=365, extra_sources=()):
    fetch = AsyncMock(return_value={"results": [
        {"url": SECONDARY, "text": body, "meta": metadata or {},
         "source_publication_date": publication_date}, *extra_sources,
    ] if fetched else [], "statuses": []})
    provider = AsyncMock(return_value=_answer(event_date, urls=urls))
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", provider)
    verdicts = []
    with use_evaluation_date("2026-10-19"):
        result = asyncio.run(lead_scorer._score_single_intent_signal(
            _signal(), _icp(evidence_type).model_copy(update={"intent_max_age_days": max_age_days}), None, "Aurasell", "https://aurasell.ai/",
            company_linkedin="https://www.linkedin.com/company/aurasell",
            trust_signal_date=True, stage1_soft_reject=True, llm_only_intent_gate=True,
            integrity_policy=True, company_quality=True,
            verified_company_identity=IDENTITY, evidence_signals=[_signal(), *[
                _signal().model_copy(update={"url": source["url"]}) for source in extra_sources
            ]],
            verified_funding_stage_context=context, verdict_out=verdicts,
        ))
    return result, verdicts, fetch, provider


def test_stage_source_is_selected_before_paragraph_ranking_cap(monkeypatch):
    async def fit(*args, **kwargs):
        kwargs["matched_company_source_sink"].update({
            PRIMARY: {"final_url": PRIMARY, "text": STAGE_TEXT},
            **{f"https://aurasell.ai/{d}": {
                "final_url": f"https://aurasell.ai/{d}", "text": d,
            } for d in ("industry", "employee_size", "geography")},
        })
        return _fit()

    signal_score = AsyncMock(return_value=(0, 0, 0, 0, True, []))
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signal_score)
    asyncio.run(lead_scorer.score_company_competition_intent(
        _company(), _icp(), 0, 0, set(), integrity_policy=True,
    ))
    assert signal_score.await_args.kwargs["verified_funding_stage_context"] == [CONTEXT]


@pytest.mark.parametrize("mutation", ["unrelated_url", "quote_absent", "not_match", "no_identity"])
def test_only_grounded_final_matched_same_company_stage_source_is_reused(mutation):
    fit = _fit()
    cache = {PRIMARY: {"final_url": PRIMARY, "text": STAGE_TEXT}}
    if mutation == "unrelated_url":
        cache = {"https://other.example/round": {"final_url": PRIMARY, "text": STAGE_TEXT}}
    elif mutation == "quote_absent":
        cache[PRIMARY]["text"] = "Unrelated company raised an unrelated round."
    elif mutation == "not_match":
        fit.details["dimension_evidence"]["stage"]["decision"] = "unavailable"
    else:
        fit.details["dimension_evidence"].pop("identity")
    assert lead_scorer._matched_investigator_source_contexts(fit, cache) == []


def test_aurasell_old_event_context_reaches_first_judge_and_existing_date_gate(monkeypatch):
    result, receipts, fetch, provider = _run(monkeypatch, context=[CONTEXT])
    assert result[0] == 0 and result[2] == "out_of_window"
    assert receipts[-1]["authoritative_date"] == "2025-08-26"
    assert receipts[-1]["age_days"] == 419
    assert fetch.await_count == provider.await_count == 1
    prompt = provider.await_args.args[2]
    assert STAGE_TEXT in prompt and RECAP_TEXT in prompt
    assert "Matching stage, amount or investors alone" in prompt
    assert "an older stage source must not veto a separately proved new round" in prompt
    assert "evidence_urls_used must include at least one exact URL" in prompt
    assert "SUBMITTED FUNDING SOURCE URLS: " + json.dumps([SECONDARY]) in prompt
    assert "REUSED STAGE SOURCE URLS: " + json.dumps([PRIMARY]) in prompt
    contexts = receipts[-1]["verification_trace"]["verified_source_context"]
    assert {x["url"] for x in contexts} == {PRIMARY, SECONDARY}
    stage = next(x for x in contexts if x["url"] == PRIMARY)
    assert stage["text"] == STAGE_TEXT and stage["source_role"] == "matched_company_stage"
    assert receipts[-1]["verification_trace"]["intent_verdict"]["signal_evaluations"][0]["evidence_urls_used"] == [SECONDARY, PRIMARY]


@pytest.mark.parametrize("new_event", [
    "On February 14, 2026, Aurasell closed a new Series A round.",
    "On February 14, 2026, Aurasell closed a separate $30 million Seed extension from N47, Menlo Ventures, and Unusual Ventures.",
])
def test_existing_judge_can_find_new_funding_event_despite_old_stage(monkeypatch, new_event):
    result, receipts, fetch, provider = _run(
        monkeypatch, event_date="2026-02-14", context=[CONTEXT], body=new_event,
    )
    assert result[0] > 0 and result[2] == "in_window"
    assert receipts[-1]["authoritative_date"] == "2026-02-14"
    assert new_event in provider.await_args.args[2] and STAGE_TEXT in provider.await_args.args[2]
    assert fetch.await_count == provider.await_count == 1


def test_missing_stage_date_does_not_invent_old_date_or_reject_supported_event(monkeypatch):
    context = {**CONTEXT, "text": "Aurasell is Seed funded. " + QUOTE}
    result, receipts, _, provider = _run(monkeypatch, event_date=None, context=[context])
    assert result[0] > 0 and result[2] == "uncertain"
    assert receipts[-1].get("authoritative_date") is None
    assert provider.await_count == 1


def test_newer_publication_metadata_without_judge_date_tag_stays_uncertain(monkeypatch):
    result, receipts, _, provider = _run(
        monkeypatch, event_date=None, context=[CONTEXT], publication_date="2026-02-14",
    )
    assert result[0] > 0 and result[2] == "uncertain"
    assert receipts[-1].get("authoritative_date") is None
    assert "SOURCE PAGE PUBLICATION METADATA: 2026-02-14" in provider.await_args.args[2]
    assert provider.await_count == 1


def test_long_retained_stage_date_quote_and_body_survive_prompt_and_receipt(monkeypatch):
    text = "Aurasell stage source. " + "x " * 5800 + STAGE_TEXT + " y" * 5800
    assert 23000 < len(text) < 24000
    body = RECAP_TEXT + " z" * 29000
    result, receipts, fetch, provider = _run(
        monkeypatch, context=[{**CONTEXT, "text": text}], body=body,
    )
    assert result[2] == "out_of_window"
    assert text in provider.await_args.args[2]
    stage = next(x for x in receipts[-1]["verification_trace"]["verified_source_context"]
                 if x["url"] == PRIMARY)
    assert stage["text"] == text
    assert fetch.await_count == provider.await_count == 1


@pytest.mark.parametrize("mutation", [None, "missing_role", "missing_allowlist", "wrong_fit_url"])
@pytest.mark.parametrize("original_count", [1, 3])
def test_cited_stage_evidence_stays_grounded_through_paragraph_projection(monkeypatch, mutation, original_count):
    late_text = "x " * 9500 + STAGE_TEXT
    new_quote = "On February 14, 2026, Aurasell closed a separate Seed extension."
    _, receipts, _, _ = _run(
        monkeypatch, event_date="2026-02-14", context=[{**CONTEXT, "text": late_text}],
        body="z " * 4000 + new_quote,
    )
    receipt = copy.deepcopy(receipts[-1])
    item = receipt["verification_trace"]["intent_verdict"]["signal_evaluations"][0]
    item["supporting_quotes"] = [new_quote, QUOTE]
    original_urls = [SECONDARY]
    if original_count == 3:
        extra_urls = ["https://news.example/round", "https://news.example/extension"]
        original_urls.extend(extra_urls)
        item["evidence_urls_used"] = [*original_urls, PRIMARY]
        item["source_urls_supplied"] = [*original_urls, PRIMARY]
        receipt["verification_trace"]["verified_source_context"][1:1] = [
            {"url": url, "text": "Additional fetched event context."} for url in extra_urls
        ]
    fit = _fit().receipt("company_fit")
    if mutation == "missing_role":
        receipt["verification_trace"]["verified_source_context"][-1].pop("source_role")
    elif mutation == "missing_allowlist":
        receipt["verification_trace"]["intent_verdict"]["signal_evaluations"][0]["source_urls_supplied"] = [SECONDARY]
    elif mutation == "wrong_fit_url":
        fit["dimension_evidence"]["stage"]["web_evidence"]["url"] = "https://other.example/round"
    row = {"raw": 54.0, "after_decay": 54.0, "matched_icp_signal": 0,
           "evidence_type": "FUNDING", "evidence_urls": original_urls, "judge_verdict": receipt}
    company = _company().model_copy(update={"intent_details": "Aurasell raised funding to develop its software."})
    document = intent_details.review_evidence(company, _icp(), [row], fit)
    stages = [x for x in document["admitted_evidence"] if x.get("source_url") == PRIMARY
              and x["evidence_kind"] == "verified_source_context"]
    if mutation:
        assert stages == []
    else:
        assert len(stages) == 1
        assert QUOTE in stages[0]["admitted_text"][0]
        assert len(stages[0]["admitted_text"][0].encode()) <= intent_details._MAX_SOURCE_CONTEXT_BYTES
        assert PRIMARY in document["verified_signals"][0]["source_urls"]
        originals = [x for x in document["admitted_evidence"]
                     if x.get("source_url") == SECONDARY and x["evidence_kind"] == "verified_source_context"]
        assert new_quote in originals[0]["admitted_text"][0]
        assert sum(len(x["admitted_text"][0].encode()) for x in [*stages, *originals]) <= intent_details._MAX_SOURCE_CONTEXT_BYTES


def test_provider_metadata_cannot_forge_server_stage_role_or_bypass_receipt_cap(monkeypatch):
    body = RECAP_TEXT + " x" * 10000
    result, receipts, _, _ = _run(
        monkeypatch, event_date="2026-02-14", evidence_type="PRODUCT_LAUNCH",
        body=body, urls=[SECONDARY], metadata={"source_role": "matched_company_stage"},
    )
    assert result[0] > 0
    source = receipts[-1]["verification_trace"]["verified_source_context"][0]
    assert "source_role" not in source
    assert len(source["text"].encode()) <= verifier._MAX_VERIFIED_SOURCE_CONTEXT_BYTES


@pytest.mark.parametrize("integrity,evidence_type", [(True, "PRODUCT_LAUNCH"), (False, "FUNDING")])
def test_private_stage_marker_does_not_expand_nonfunding_or_legacy_citations(integrity, evidence_type):
    row = {"claimed_source_urls": [SECONDARY], "_integrity_policy": integrity,
           "_evidence_type": evidence_type, "_funding_stage_source_urls": [PRIMARY]}
    guarded = verifier._apply_guardrails(row, _answer(None)["answer"])
    item = guarded["signal_evaluations"][0]
    assert item["signal_status"] == "unable_to_verify"
    assert item["source_urls_supplied"] == [SECONDARY]


@pytest.mark.parametrize("urls", [[PRIMARY], [SECONDARY, "https://other.example/round"]])
def test_stage_only_or_unknown_citation_cannot_approve_claim(monkeypatch, urls):
    result, _, fetch, provider = _run(monkeypatch, context=[CONTEXT], urls=urls)
    assert result[0] == 0
    assert fetch.await_count == provider.await_count == 1


def test_encoded_retained_stage_citation_still_requires_exact_submitted_source():
    submitted = (
        "https://www.businesswire.com/news/home/20260504989731/en/"
        "Latus-Bio-Announces-$97-Million-Series-A-Financing-to-Expand-the-Reach-"
        "of-Gene-Therapy-to-Larger-Populations"
    )
    stage = submitted.replace("$97", "%2497")
    row = {
        "id": "signal-1", "company": "Latus Bio", "website": "https://latusbio.com",
        "company_linkedin": "", "contact_linkedin": "",
        "claim": "Latus Bio closed a $97 million Series A financing.",
        "signal_date": "2026-05-04", "signal_type": "intent",
        "claimed_source_urls": [submitted],
        "_integrity_policy": True,
        "_evidence_type": "FUNDING",
        "_target_signal_text": "Announced a Series A financing round",
        "_funding_stage_source_urls": [stage],
    }
    prompt = verifier._build_final_judge_prompt(row, {"results": [
        {"url": submitted, "title": "Latus Bio financing", "text": row["claim"]},
        {"url": stage, "title": "Retained MATCH stage", "text": row["claim"]},
    ]})
    assert "SUBMITTED FUNDING SOURCE URLS: " + json.dumps([submitted]) in prompt
    assert "REUSED STAGE SOURCE URLS: " + json.dumps([stage]) in prompt
    assert "evidence_urls_used must include at least one exact URL" in prompt
    for urls, expected in (
        ([stage], "unable_to_verify"),
        ([submitted], "supported"),
        ([submitted, stage], "supported"),
        ([submitted, "https://www.businesswire.com/other-event"], "unable_to_verify"),
    ):
        guarded = verifier._apply_guardrails(row, copy.deepcopy(_answer(None, urls=urls)["answer"]))
        assert guarded["signal_evaluations"][0]["signal_status"] == expected


def test_retained_stage_source_cannot_rescue_missing_submitted_body(monkeypatch):
    result, _, fetch, provider = _run(monkeypatch, context=[CONTEXT], fetched=False)
    assert result[0] == 0
    assert fetch.await_count == 1 and provider.await_count == 0


def test_nonfunding_context_does_not_change_prompt_or_add_calls(monkeypatch):
    _, _, _, with_context = _run(monkeypatch, evidence_type="PRODUCT_LAUNCH", context=[CONTEXT], urls=[SECONDARY])
    first = with_context.await_args.args[2]
    _, _, _, without = _run(monkeypatch, evidence_type="PRODUCT_LAUNCH", urls=[SECONDARY])
    assert first == without.await_args.args[2]
    assert STAGE_TEXT not in first
    assert with_context.await_count == without.await_count == 1


@pytest.mark.parametrize("context", [
    [{**CONTEXT, "dimension": "industry"}],
    [{**CONTEXT, "url": "not a URL"}],
    [{**CONTEXT, "text": "x" * (verifier.MAX_SCRAPED_CHARS + 1)}],
    [CONTEXT, CONTEXT],
])
def test_malformed_or_unbounded_context_does_not_enter_judge(monkeypatch, context):
    _, _, _, provider = _run(monkeypatch, context=context, urls=[SECONDARY])
    assert STAGE_TEXT not in provider.await_args.args[2]


@pytest.mark.parametrize("evidence_type,expected_calls", [("FUNDING", 2), ("PRODUCT_LAUNCH", 1)])
def test_changed_funding_context_invalidates_positive_terminal_retry_cache(monkeypatch, evidence_type, expected_calls):
    from tests.test_arena_intent_retry_cache import _record_positive

    async def score(signal, *args, **kwargs):
        return _record_positive(signal, kwargs["evidence_signals"], kwargs["verdict_out"])

    single = AsyncMock(side_effect=score)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", single)
    cache = {}
    async def run(context):
        return await lead_scorer.score_company_competition_intent_signal(
            _company(), _icp(evidence_type), integrity_policy=True,
            verified_company_identity=IDENTITY, verified_funding_stage_context=[context],
            intent_terminal_retry_cache=cache, retry_evidence_context_key="same-execution",
        )
    with use_evaluation_date("2026-10-19"):
        asyncio.run(run(CONTEXT))
        asyncio.run(run(CONTEXT))
        asyncio.run(run({**CONTEXT, "text": STAGE_TEXT + " Updated source body."}))
    assert single.await_count == expected_calls

# Redacted public facts reproduce the source-loss shape; model verdicts below
# are stubs that test evidence plumbing, not proof of semantic judgment.
LATER = "https://press.example/aurasell-product"
LATER_CONTEXT = {"dimension": "stage", "url": LATER,
                 "text": "February 12, 2026. About Aurasell: " + QUOTE}
AUXILIARY = {"dimension": "first_party_context", "url": PRIMARY,
             "final_url": PRIMARY, "text": STAGE_TEXT}


def _six_source_shape():
    from types import SimpleNamespace
    from qualification.scoring.company_evidence_investigator import PRIVATE_FETCHED_PAGES_KEY

    company = _company().model_copy(update={
        "intent_details": "Aurasell has raised a $30 million Seed round from N47, Menlo Ventures, and Unusual Ventures.",
        "required_attribute": SimpleNamespace(
            evidence_url="https://aurasell.ai/attribute", evidence_quote="Aurasell supplies software.",
        ),
    })
    fit = _fit()
    fit.details["dimension_evidence"]["stage"]["web_evidence"]["url"] = LATER
    pages = {f"https://aurasell.ai/{d}": {
        "final_url": f"https://aurasell.ai/{d}", "text": f"Aurasell {d}",
    } for d in ("industry", "employee_size", "geography")}
    pages.update({
        LATER: {"final_url": LATER, "text": LATER_CONTEXT["text"]},
        "https://aurasell.ai/attribute": {"final_url": "https://aurasell.ai/attribute", "text": "Aurasell supplies software."},
        PRIMARY: {"final_url": PRIMARY, "text": STAGE_TEXT},
    })
    sink = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        sink, fit, {PRIVATE_FETCHED_PAGES_KEY: pages}, company=company,
    )
    return company, fit, sink


def test_four_matches_attribute_and_one_fetched_auxiliary_fit_existing_map():
    company, fit, sink = _six_source_shape()
    assert lead_scorer._MAX_RETAINED_INVESTIGATOR_CONTEXTS == 4 + 1 + 1
    assert len(sink) == 6 and sink[PRIMARY]["text"] == STAGE_TEXT
    auxiliary = lead_scorer._matched_first_party_source_context(
        fit, sink, company, paragraph=company.intent_details,
        excluded_urls=[company.required_attribute.evidence_url],
    )
    assert auxiliary == AUXILIARY


@pytest.mark.parametrize("evidence_type", ["FUNDING", "PRODUCT_LAUNCH"])
def test_whole_company_handoff_keeps_aux_before_paragraph_cap(monkeypatch, evidence_type):
    company, fit, sink = _six_source_shape()
    async def fake_fit(*args, **kwargs):
        kwargs["matched_company_source_sink"].update(sink)
        return fit
    score = AsyncMock(return_value=(0, 0, 0, 0, True, []))
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fake_fit)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", score)
    retry = {}
    asyncio.run(lead_scorer.score_company_competition_intent(
        company, _icp(evidence_type), 0, 0, set(), integrity_policy=True,
        matched_company_retry_source_cache=retry,
    ))
    context = score.await_args.kwargs.get("verified_funding_stage_context")
    if evidence_type == "FUNDING":
        assert context == [LATER_CONTEXT, AUXILIARY]
        assert list(retry["pages"])[:2] == [PRIMARY, LATER]
    else:
        assert context is None and PRIMARY not in retry["pages"]
    assert len(retry["pages"]) <= 3


@pytest.mark.parametrize("mutation", ["other_domain", "redirect", "company", "empty", "oversize", "extra_field", "http"])
def test_invalid_auxiliary_never_reaches_funding_judge(monkeypatch, mutation):
    context = copy.deepcopy(AUXILIARY)
    if mutation == "other_domain": context.update(url="https://other.example/round", final_url="https://other.example/round")
    elif mutation == "redirect": context["final_url"] = "https://other.example/login"
    elif mutation == "company": context["text"] = "An unrelated company raised a Seed round."
    elif mutation == "empty": context["text"] = ""
    elif mutation == "oversize": context["text"] = "Aurasell " + "x" * 24000
    elif mutation == "extra_field": context["proof"] = "model says verified"
    elif mutation == "http": context["url"] = context["final_url"] = "http://aurasell.ai/round"
    _, _, _, provider = _run(monkeypatch, context=[LATER_CONTEXT, context], urls=[SECONDARY])
    assert STAGE_TEXT not in provider.await_args.args[2]


@pytest.mark.parametrize("new_date,body", [
    ("2025-08-26", RECAP_TEXT),
    ("2026-02-14", "Aurasell closed a new Series A on February 14, 2026."),
    ("2026-02-14", "Aurasell closed a separate new Seed extension on February 14, 2026."),
])
def test_funding_judge_receives_full_aux_with_no_extra_fetch_or_model_call(monkeypatch, new_date, body):
    result, receipts, fetch, provider = _run(
        monkeypatch, event_date=new_date, body=body,
        context=[LATER_CONTEXT, AUXILIARY], urls=[SECONDARY, LATER, PRIMARY],
    )
    assert fetch.await_count == provider.await_count == 1
    prompt = provider.await_args.args[2]
    assert body in prompt and STAGE_TEXT in prompt and LATER_CONTEXT["text"] in prompt
    assert "not MATCH-stage or event-identity proof" in prompt
    assert (result[0] == 0) == (new_date == "2025-08-26")
    auxiliary = next(x for x in receipts[-1]["verification_trace"]["verified_source_context"] if x["url"] == PRIMARY)
    assert auxiliary == {"url": PRIMARY, "text": STAGE_TEXT, "source_publication_date": "",
                         "source_role": "first_party_context", "final_url": PRIMARY}


def test_claimed_stage_source_dedup_still_allows_distinct_auxiliary(monkeypatch):
    stage = {**LATER_CONTEXT, "url": SECONDARY, "text": RECAP_TEXT}
    _, receipts, fetch, provider = _run(
        monkeypatch, context=[stage, AUXILIARY], urls=[SECONDARY, PRIMARY],
    )
    assert fetch.await_count == provider.await_count == 1
    assert STAGE_TEXT in provider.await_args.args[2]
    retained = receipts[-1]["verification_trace"]["verified_source_context"]
    assert len([x for x in retained if x["url"] == SECONDARY]) == 1
    assert any(x["url"] == PRIMARY for x in retained)


@pytest.mark.parametrize("urls", [[PRIMARY], [LATER, PRIMARY]])
def test_auxiliary_never_replaces_submitted_evidence(monkeypatch, urls):
    result, _, _, _ = _run(monkeypatch, context=[LATER_CONTEXT, AUXILIARY], urls=urls)
    assert result[0] == 0


def test_uncited_auxiliary_is_not_projected_as_verified_context(monkeypatch):
    _, receipts, _, _ = _run(monkeypatch, context=[LATER_CONTEXT, AUXILIARY], urls=[SECONDARY, LATER])
    assert PRIMARY not in {x["url"] for x in receipts[-1]["verification_trace"]["verified_source_context"]}


def test_legacy_alias_is_not_subject_to_new_exact_name_gate():
    context = lead_scorer._validated_first_party_source_context(
        PRIMARY, {"final_url": PRIMARY, "text": "Legacy Alias supplies software."}, IDENTITY,
        company_name="Legacy Alias",
    )
    assert context is not None


def test_auxiliary_changed_body_invalidates_existing_terminal_cache_key():
    args = ({}, IDENTITY, [_signal()], "funding", "FUNDING", 365, {})
    first = lead_scorer._intent_terminal_retry_key(*args, [LATER_CONTEXT, AUXILIARY])
    changed = lead_scorer._intent_terminal_retry_key(*args, [LATER_CONTEXT, {**AUXILIARY, "text": STAGE_TEXT + " New round."}])
    assert first != changed


def test_retry_reserves_existing_three_slots_for_auxiliary_and_stage():
    company, fit, sink = _six_source_shape()
    retry = {}
    lead_scorer._retain_matched_company_retry_sources(retry, sink, fit)
    assert len(retry["pages"]) == 3 and PRIMARY not in retry["pages"]
    lead_scorer._retain_matched_company_retry_sources(
        retry, sink, fit, funding_contexts=[LATER_CONTEXT, AUXILIARY],
    )
    identity = lead_scorer._matched_company_result_retry_identity(fit)
    reused = lead_scorer._matched_company_retry_prefetched_pages(retry, identity)
    assert list(reused)[:2] == [PRIMARY, LATER] and len(reused) == 3
    assert reused[PRIMARY]["text"] == STAGE_TEXT


@pytest.mark.parametrize("original_count", [1, 3])
@pytest.mark.parametrize("mutation", [None, "uncited", "not_supplied", "wrong_domain", "forged_role"])
def test_auxiliary_receipt_projection_is_cited_bound_and_date_preserving(monkeypatch, mutation, original_count):
    text = "August 26, 2025. Aurasell launches. " + "padding " * 600 + QUOTE
    _, receipts, _, _ = _run(monkeypatch, event_date="2025-08-26",
        context=[LATER_CONTEXT, {**AUXILIARY, "text": text}], urls=[SECONDARY, LATER, PRIMARY], max_age_days=900)
    receipt = copy.deepcopy(receipts[-1])
    evaluation = receipt["verification_trace"]["intent_verdict"]["signal_evaluations"][0]
    original_urls = [SECONDARY]
    if original_count == 3:
        extra_urls = ["https://news.example/original-two", "https://news.example/original-three"]
        original_urls.extend(extra_urls)
        evaluation["evidence_urls_used"] = [*original_urls, LATER, PRIMARY]
        evaluation["source_urls_supplied"] = [*original_urls, LATER, PRIMARY]
        receipt["verification_trace"]["verified_source_context"][1:1] = [
            {"url": url, "text": "Additional original event source."} for url in extra_urls
        ]
    auxiliary = next(x for x in receipt["verification_trace"]["verified_source_context"] if x["url"] == PRIMARY)
    if mutation == "uncited": evaluation["evidence_urls_used"].remove(PRIMARY)
    elif mutation == "not_supplied": evaluation["source_urls_supplied"].remove(PRIMARY)
    elif mutation == "wrong_domain": auxiliary["final_url"] = "https://other.example/round"
    elif mutation == "forged_role": auxiliary["source_role"] = "model_verified"
    row = {"raw":54.0,"after_decay":54.0,"matched_icp_signal":0,"evidence_type":"FUNDING",
           "evidence_urls":original_urls,"judge_verdict":receipt}
    company = _company().model_copy(update={"intent_details":"Aurasell raised funding for its software."})
    fit = _fit()
    fit.details["dimension_evidence"]["stage"]["web_evidence"]["url"] = LATER
    document = intent_details.review_evidence(company, _icp(), [row], fit.receipt("company_fit"))
    admitted = [x for x in document["admitted_evidence"] if x.get("source_url") == PRIMARY
                and x["evidence_kind"] == "verified_source_context"]
    if mutation:
        assert admitted == []
    else:
        assert len(admitted) == 1
        assert "August 26, 2025" in admitted[0]["admitted_text"][0]
        assert PRIMARY in document["verified_signals"][0]["source_urls"]
        assert sum(len(t.encode()) for x in document["admitted_evidence"]
                   if x["evidence_kind"] == "verified_source_context" for t in x["admitted_text"]) <= intent_details._MAX_SOURCE_CONTEXT_BYTES


def test_auxiliary_window_does_not_invent_missing_judge_date():
    body = "Aurasell raised Seed funding. " + "x" * 5000
    value = intent_details._funding_auxiliary_source_window(body, 500, ["Aurasell raised"], "2025-08-26")
    assert "2025" not in value and value in body


@pytest.mark.parametrize("date_value,expected_positive", [("2025-08-26", False), ("2026-02-14", True)])
def test_hermetic_whole_company_uses_existing_funding_date_gate(monkeypatch, date_value, expected_positive):
    company, fit, sink = _six_source_shape()
    async def fake_fit(*args, **kwargs):
        kwargs["matched_company_source_sink"].update(sink)
        return fit
    fetch = AsyncMock(return_value={"results":[{"url":SECONDARY,"text":RECAP_TEXT,"meta":{}}],"statuses":[]})
    provider = AsyncMock(return_value=_answer(date_value, urls=[SECONDARY,LATER,PRIMARY]))
    paragraph = AsyncMock(return_value={"contract_id":"intent-details:v1","decision":"match"})
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fake_fit)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", provider)
    monkeypatch.setattr(intent_details, "review_intent_details", paragraph)
    with use_evaluation_date("2026-10-19"):
        score = asyncio.run(lead_scorer.score_company_competition_intent(
            company, _icp(), 0, 0, set(), integrity_policy=True, company_quality=True,
        ))
    assert (score.final_score > 0) == expected_positive
    assert fetch.await_count == provider.await_count == 1
    assert paragraph.await_count == int(expected_positive)
    assert STAGE_TEXT in provider.await_args.args[2]


def test_unverified_attribute_can_still_use_existing_auxiliary_fallback():
    company, fit, sink = _six_source_shape()
    # A submitted quote mismatch prevents the special attribute role, but does
    # not erase an independently fetched relevant first-party company page.
    company.required_attribute.evidence_url = PRIMARY
    company.required_attribute.evidence_quote = "An absent submitted quote."
    selected = lead_scorer._matched_company_source_contexts(
        fit, None, None, sink, paragraph=company.intent_details, company=company,
    )
    assert any(c["dimension"] == "first_party_context" and c["url"] == PRIMARY for c in selected)


@pytest.mark.parametrize("aux_size", [5700, 23900])
def test_five_source_prompt_keeps_urls_and_existing_balanced_bound(monkeypatch, aux_size):
    text = STAGE_TEXT + " x" * ((aux_size-len(STAGE_TEXT))//2)
    extra = [{"url":f"https://news.example/source-{i}","text":"Aurasell event. " + "z " * 29000,"meta":{}}
             for i in (2,3)]
    _, _, fetch, provider = _run(
        monkeypatch, event_date="2025-08-26", body=RECAP_TEXT + " y" * 29000,
        context=[LATER_CONTEXT, {**AUXILIARY,"text":text}], extra_sources=extra,
        urls=[SECONDARY, *(source["url"] for source in extra), LATER, PRIMARY],
    )
    prompt = provider.await_args.args[2]
    assert len(prompt) <= 95000
    assert all(url in prompt for url in [SECONDARY, *(source["url"] for source in extra),LATER,PRIMARY])
    assert "August 26, 2025" in prompt
    if aux_size == 5700:
        assert text in prompt
    # The existing balanced limiter may excerpt the 24KB body; do not claim
    # universal full-body survival or fabricate a missing date after clipping.
    assert fetch.await_count == provider.await_count == 1


@pytest.mark.parametrize("literal_date", [True, False])
def test_auxiliary_initial_window_keeps_late_literal_date_without_inventing_one(monkeypatch, literal_date):
    text = "Aurasell. " + QUOTE + " padding" * 2300
    if literal_date:
        text += " August 26, 2025. Aurasell financing announcement."
    assert 12000 < len(text.encode()) < 24000
    _, receipts, _, _ = _run(
        monkeypatch, event_date="2025-08-26", max_age_days=900,
        context=[LATER_CONTEXT, {**AUXILIARY, "text":text}],
        urls=[SECONDARY,LATER,PRIMARY],
    )
    fit = _fit()
    fit.details["dimension_evidence"]["stage"]["web_evidence"]["url"] = LATER
    row = {"raw":54.0,"after_decay":54.0,"matched_icp_signal":0,"evidence_type":"FUNDING",
           "evidence_urls":[SECONDARY],"judge_verdict":receipts[-1]}
    company = _company().model_copy(update={"intent_details":"Aurasell raised funding for its software."})
    document = intent_details.review_evidence(company,_icp(),[row],fit.receipt("company_fit"))
    source = next(v for v in document["admitted_evidence"] if v.get("source_url")==PRIMARY
                  and v["evidence_kind"]=="verified_source_context")
    retained = source["admitted_text"][0]
    assert ("August 26, 2025" in retained) is literal_date
    assert retained in text
    assert len(retained.encode()) <= intent_details._MAX_SOURCE_CONTEXT_BYTES
