"""Reuse already fetched first-party facts in the bounded paragraph review."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import intent_details, lead_scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match, company_fit_unavailable,
)


INVESTOR = (
    "https://greshamhouseventures.com/"
    "gresham-house-ventures-leads-20m-investment-in-uk-based-payments-platform-ryft/"
)
ABOUT = "https://ryftpay.com/about-us"
HOME = "https://ryftpay.com/"
HOME_QUOTE = (
    "Ryft is a UK-based Payment Services Provider (PSP) that offers embedded "
    "payments for marketplaces, platforms and retailers."
)
FUNDING = (
    "Gresham House Ventures has led a £20 million Series B investment in Ryft, "
    "to support the Manchester-based payments platform’s expansion across "
    "Europe and the US and accelerate its move upmarket into high-growth "
    "and enterprise accounts."
)
PRODUCT = (
    "Say hello to payment processing with Ryft. Accept payments online and "
    "in person, manage subscriptions and delays, automate split payments, "
    "and monetise transactions all from one integration."
)
REGULATION = "Ryft is regulated by the Financial Conduct Authority (FCA)."
PARAGRAPH = (
    "Ryft recently secured a £20 million Series B investment to support "
    "expansion across Europe and the US and accelerate its move into "
    "high-growth and enterprise accounts. Its own site describes an "
    "FCA-regulated payments platform that lets marketplaces and platforms "
    "accept payments, automate split payments, and monetise transactions "
    "through one integration. This expansion may create opportunities "
    "around scaling its software-led payment infrastructure for business "
    "customers."
)


def _case(*, paragraph=PARAGRAPH, complete_identity=True):
    company = SimpleNamespace(
        company_name="Ryft", company_website=HOME,
        intent_details=paragraph,
        required_attribute=SimpleNamespace(
            evidence_url=HOME, evidence_quote=HOME_QUOTE,
        ),
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0, description="Ryft secured Series B funding",
            url=INVESTOR,
        )],
    )
    icp = SimpleNamespace(
        prompt="Payment platforms with recent funding",
        product_service="payment platform", intent_signals=["Recent funding"],
    )
    fit = company_fit_match(details={
        "required_attribute_decision": "match",
        "supporting_receipts": [{
            "gate": "required_attribute_source", "status": "grounded",
        }],
        "dimension_evidence": {
        "identity": {"decision": "match", "web_identity_receipt": {
            "decision": "match", "observed_name": "Ryft",
            "observed_domain": "ryftpay.com",
            "observed_linkedin_slug": "ryftpay" if complete_identity else "",
            "evidence_source": "company_web_reverification",
        }},
        "stage": {"decision": "match", "web_evidence": {
            "url": INVESTOR, "quote": FUNDING,
        }},
        "industry": {"decision": "match", "web_evidence": {
            "url": HOME, "quote": HOME_QUOTE,
        }},
        "required_attribute": {"decision": "match", "web_evidence": {
            "url": HOME, "quote": HOME_QUOTE,
        }},
    }})
    signals = [{
        "after_decay": 54.0, "matched_icp_signal": 0,
        "evidence_urls": [INVESTOR],
        "judge_verdict": {
            "decision": "verified", "client_ready": True,
            "authoritative_date": "2026-09-18",
            "authoritative_date_basis": "source_publication_date",
            "verification_trace": {
                "verified_source_context": [{"url": INVESTOR, "text": FUNDING}],
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported", "same_entity_check": "pass",
                    "supporting_quotes": [FUNDING],
                    "evidence_urls_used": [INVESTOR],
                }]},
            },
        },
    }]
    return company, icp, fit, signals


def _retained_contexts(*, pages, complete_identity=True,
                       paragraph=PARAGRAPH):
    company, icp, fit, signals = _case(
        paragraph=paragraph, complete_identity=complete_identity,
    )
    retained = {}
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, fit,
        {investigator.PRIVATE_FETCHED_PAGES_KEY: pages},
        company=company,
    )
    contexts = lead_scorer._matched_company_source_contexts(
        fit, None, None, retained, paragraph=paragraph, company=company,
        verified_signal_source_urls=lead_scorer._verified_signal_context_urls(
            signals
        ),
    )
    return company, icp, fit, signals, retained, contexts


def test_ryft_shaped_fetched_about_facts_reach_real_review_document():
    assert hashlib.sha256(PARAGRAPH.encode()).hexdigest() == (
        "0e655deea50a609dabeb21e29d9a375b4a80073ff00f3dd66ab7fa441651b730"
    )
    # The investor page already occupies verified-signal evidence. The about
    # page was fetched during company fit, even though no final fit dimension
    # cites it. Its separate product and FCA facts must remain available.
    about_text = "About Ryft. " + PRODUCT + " " + REGULATION
    company, icp, fit, signals, retained, contexts = _retained_contexts(
        pages={
            INVESTOR: {"final_url": INVESTOR, "text": FUNDING},
            ABOUT: {"final_url": ABOUT, "text": about_text},
            HOME: {"final_url": HOME, "text": HOME_QUOTE + " " + PRODUCT},
        },
    )
    assert ABOUT in retained
    assert [context["dimension"] for context in contexts] == [
        "industry", "first_party_context",
    ]
    assert contexts[1] == {
        "dimension": "first_party_context", "url": ABOUT,
        "final_url": ABOUT, "text": about_text,
    }
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )
    about_sources = [
        item for item in document["admitted_evidence"]
        if item["evidence_kind"] == "verified_company_source_context"
        and item["source_url"] == ABOUT
    ]
    assert len(about_sources) == 1
    assert PRODUCT in about_sources[0]["admitted_text"][0]
    assert REGULATION in about_sources[0]["admitted_text"][0]
    assert sum(len(text.encode()) for source in document["admitted_evidence"]
               for text in source.get("admitted_text", [])) <= (
                   intent_details._MAX_SOURCE_CONTEXT_BYTES + 4_000
               )


def test_fetched_about_page_survives_intermediate_fit_until_final_match():
    company, _icp, final_fit, signals = _case()
    intermediate = company_fit_unavailable(
        "stage unresolved", details=final_fit.details,
    )
    retained = {}
    about = "About Ryft. " + PRODUCT + " " + REGULATION
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, intermediate,
        {investigator.PRIVATE_FETCHED_PAGES_KEY: {
            ABOUT: {"final_url": ABOUT, "text": about},
        }}, company=company,
    )
    assert ABOUT in retained
    lead_scorer._retain_matched_investigator_source_contexts(
        retained, final_fit,
        {investigator.PRIVATE_FETCHED_PAGES_KEY: {
            INVESTOR: {"final_url": INVESTOR, "text": FUNDING},
        }}, company=company,
    )
    contexts = lead_scorer._matched_company_source_contexts(
        final_fit, None, None, retained, paragraph=PARAGRAPH,
        company=company,
        verified_signal_source_urls=lead_scorer._verified_signal_context_urls(
            signals
        ),
    )
    assert [context["url"] for context in contexts] == [ABOUT]


def test_multiple_relevant_first_party_pages_keep_one_bounded_auxiliary():
    other = "https://ryftpay.com/capabilities/split-payments"
    company, icp, fit, signals, retained, contexts = _retained_contexts(
        pages={
            ABOUT: {"final_url": ABOUT, "text": (
                "About Ryft. " + REGULATION + " " + PRODUCT
            )},
            other: {"final_url": other, "text": (
                "Ryft split payments help marketplaces and platforms "
                "monetise transactions through one integration."
            )},
        },
    )
    assert len(retained) == 1
    assert len(contexts) == 1
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    )
    assert len([
        item for item in document["admitted_evidence"]
        if item["evidence_kind"] == "verified_company_source_context"
    ]) == 1


def test_signal_source_is_deduplicated_only_when_its_full_body_is_budget_safe():
    _company, _icp, _fit, signals = _case()
    assert lead_scorer._verified_signal_context_urls(signals) == (INVESTOR,)
    signals[0]["judge_verdict"]["verification_trace"][
        "verified_source_context"
    ][0]["text"] = "x" * (
        intent_details._MAX_SOURCE_CONTEXT_BYTES
        - intent_details._COMPANY_SOURCE_CONTEXT_RESERVATION_BYTES + 1
    )
    assert lead_scorer._verified_signal_context_urls(signals) == ()


@pytest.mark.parametrize("url,final_url,text,identity_complete", [
    ("https://other.example/about", "https://other.example/about",
     "Ryft is FCA regulated and automates split payments.", True),
    (ABOUT, "https://other.example/about",
     "Ryft is FCA regulated and automates split payments.", True),
    (ABOUT, ABOUT, "Acme is FCA regulated and automates split payments.", True),
    (ABOUT, ABOUT, "Ryft careers and team information.", True),
    (ABOUT, ABOUT, "Ryft is FCA regulated and automates split payments.", False),
], ids=["off_domain", "redirect_off_domain", "wrong_company",
        "unrelated", "incomplete_identity"])
def test_unbound_or_unrelated_fetched_pages_do_not_enter_paragraph(
    url, final_url, text, identity_complete,
):
    _company, _icp, _fit, _signals, retained, contexts = _retained_contexts(
        pages={url: {"final_url": final_url, "text": text}},
        complete_identity=identity_complete,
    )
    assert url not in retained
    assert contexts is None


def test_paragraph_window_is_continuous_bounded_and_uses_clause_not_fit_quote():
    body = (
        "Ryft home. " + HOME_QUOTE + "\n" + "background " * 1_000
        + PRODUCT + " " + REGULATION + "\n" + "footer " * 1_000
    )
    company, icp, fit, signals = _case()
    context = [{"dimension": "industry", "url": HOME, "text": body}]
    document = intent_details.review_evidence(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=context,
    )
    selected = next(
        item["admitted_text"][0] for item in document["admitted_evidence"]
        if item["evidence_kind"] == "verified_company_source_context"
    )
    assert PRODUCT in selected and REGULATION in selected
    assert selected in body
    assert len(selected.encode()) <= intent_details._MAX_SOURCE_CONTEXT_BYTES
    assert any(
        HOME_QUOTE in item.get("admitted_text", [])
        for item in document["admitted_evidence"]
        if item["evidence_kind"] == "verified_company_fact"
    )
    assert len(json.dumps(document, ensure_ascii=False)) <= (
        intent_details._MAX_REVIEW_DOCUMENT_CHARACTERS
    )


@pytest.mark.parametrize("mutation", [
    "off_domain_final", "wrong_company", "missing_identity", "oversized",
])
def test_review_document_rechecks_auxiliary_source_binding(mutation):
    company, icp, fit, signals = _case(
        complete_identity=mutation != "missing_identity",
    )
    context = {
        "dimension": "first_party_context", "url": ABOUT,
        "final_url": (
            "https://other.example/about-us"
            if mutation == "off_domain_final" else ABOUT
        ),
        "text": (
            "Acme is an FCA-regulated payment platform for marketplaces."
            if mutation == "wrong_company" else
            "Ryft is an FCA-regulated payment platform for marketplaces."
        ),
    }
    if mutation == "oversized":
        context["text"] += "x" * investigator.MAX_PAGE_CHARACTERS
    with pytest.raises(ValueError, match="invalid company source context"):
        intent_details.review_evidence(
            company, icp, signals, fit.receipt("company_fit"),
            company_source_contexts=[context],
        )


@pytest.mark.parametrize("fabricated", [False, True])
def test_existing_factual_gate_still_rejects_unquoted_revenue(monkeypatch, fabricated):
    paragraph = PARAGRAPH + (
        " Ryft doubled revenue last month." if fabricated else ""
    )
    about = "About Ryft. " + PRODUCT + " " + REGULATION
    company, icp, fit, signals, _retained, contexts = _retained_contexts(
        pages={ABOUT: {"final_url": ABOUT, "text": about}},
        paragraph=paragraph,
    )
    assert contexts is not None

    async def judge(prompt, **_kwargs):
        document = json.loads(prompt)
        sources = {
            source["source_index"]: source["admitted_text"][0]
            for source in document["admitted_evidence"]
            if source.get("admitted_text")
        }
        funding_index = next(i for i, text in sources.items() if FUNDING in text)
        about_index = next(i for i, text in sources.items() if PRODUCT in text)
        units = []
        for unit in document["intent_details_units"]:
            unsupported = fabricated and "doubled revenue" in unit["text"]
            conditional = unit["text"].startswith(
                "This expansion may create"
            )
            evidence = (
                [{"source_index": funding_index, "quote": FUNDING}]
                if "Series B" in unit["text"] else
                [{"source_index": about_index, "quote": PRODUCT},
                 {"source_index": about_index, "quote": REGULATION}]
                if "FCA-regulated" in unit["text"] else
                [{"source_index": about_index, "quote": PRODUCT}]
            )
            units.append({
                "unit_id": unit["unit_id"],
                "contains_factual_claim": not conditional,
                "status": "UNPROVEN" if unsupported else "VERIFIED",
                "evidence": [] if unsupported or conditional else evidence,
            })
        return json.dumps({
            "unit_grounding": units,
            "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
            **{check: (not fabricated if check == "facts_supported" else True)
               for check in intent_details._CHECKS},
        })

    from qualification.scoring import verification_helpers
    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(
        company, icp, signals, fit.receipt("company_fit"),
        company_source_contexts=contexts,
    ))
    assert receipt["decision"] == ("mismatch" if fabricated else "match")
    assert receipt["checks"]["facts_supported"] is (not fabricated)
    if fabricated:
        assert receipt["failed_factual_units"] == [{
            "unit_id": len(intent_details._statement_units(paragraph)) - 1,
            "status": "UNPROVEN", "source_indexes": [],
        }]
