"""Saved-input commerce-event and paragraph-fact boundaries from Sep. 24."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

from qualification.scoring import intent_details, verification_helpers
from qualification.scoring import intent_verification_three_stage as stage3


REBEL_TARGET = (
    "Launched a new commerce capability, storefront format, or retail channel "
    "expansion in the last 12 months, with proof in a press release, product "
    "page, or company announcement."
)
REBEL_CLAIM = (
    "REBEL entered CPG by adding bulk-sized snacks and pantry products to its "
    "marketplace; the new section launched in mid-May 2026."
)
REBEL_EVENT_EXCERPT = 'added bulk-sized snacks and pantry products to its lineup in mid-May'
REBEL_OUTCOME_EXCERPT = 'snacks are increasing units sold and driving more repeat visits'
REBEL_PARAGRAPH = 'On June 12, 2026, Modern Retail reported that REBEL had entered the CPG category, adding bulk snacks and pantry products to its online marketplace in mid-May. The completed launch created a new shoppable marketplace section and broadened REBEL’s assortment beyond its earlier baby and home categories. This expansion directly increases the catalog and checkout activity flowing through REBEL’s digital storefront, making it a strong match for commerce-platform and marketplace workflows.'



def _row(*, company: str, target: str, claim: str, evidence_type: str, url: str) -> dict:
    return {
        "id": f"{company.lower()}-signal",
        "company": company,
        "website": f"https://{company.lower()}.example",
        "company_linkedin": "",
        "contact_linkedin": "",
        "claim": claim,
        "signal_date": "2026-06-12",
        "signal_type": "intent",
        "claimed_source_urls": [url],
        "_target_signal_text": target,
        "_evidence_type": evidence_type,
        "_integrity_policy": True,
        "_buyer_max_age_days": 365,
    }


def _stage_prompts(row: dict, source: str) -> tuple[str, str]:
    return (
        stage3._build_verification_prompt(row),
        stage3._build_final_judge_prompt(
            row,
            {"results": [{"url": row["claimed_source_urls"][0], "text": source}], "statuses": []},
        ),
    )


def _one_line(value: str) -> str:
    return " ".join(value.split())


def test_saved_rebel_projection_keeps_event_and_source_qualifiers() -> None:
    assert hashlib.sha256(REBEL_EVENT_EXCERPT.encode()).hexdigest() == (
        "c7c4f9564171cb5ccc77b087c177e0b50e549f3d997fe3ea016d385b3b6851b9"
    )
    assert hashlib.sha256(REBEL_PARAGRAPH.encode()).hexdigest() == (
        "e3f6ee6918b3a8461179da88caecad6211c00dfd72707522e5b65a827dcbb378"
    )
    row = _row(
        company="REBEL",
        target=REBEL_TARGET,
        claim=REBEL_CLAIM,
        evidence_type="PRODUCT_LAUNCH",
        url="https://www.modernretail.co/operations/open-box-marketplace-rebel-is-now-selling-better-for-you-snacks/",
    )
    first, final = _stage_prompts(row, REBEL_EVENT_EXCERPT)

    for prompt in (first, final):
        flat = _one_line(prompt)
        assert REBEL_TARGET in flat
        assert REBEL_CLAIM in flat
        assert "an independent article is not itself one of those sources" in flat
        assert "adding merchandise to an existing catalog or marketplace does not by itself" in flat
        assert "do not substitute a separate bundle" in flat
        assert "Existing technology used to sell new goods is not a new shopping capability" in flat
    assert REBEL_EVENT_EXCERPT in final
    assert "modernretail.co" in final


def test_true_storefront_and_completed_acquisition_remain_in_actual_stage_prompts() -> None:
    controls = [
        (
            _row(
                company="Rundoo",
                target="Launched a new commerce capability or storefront in the last 12 months.",
                claim="Rundoo launched e-commerce in its customer app and a new web store.",
                evidence_type="PRODUCT_LAUNCH",
                url="https://www.rundoo.com/blog/e-commerce-is-live",
            ),
            "Rundoo E-Commerce is live on both the Customer app and a brand new web store.",
        ),
        (
            _row(
                company="CoreStack",
                target="Completed an acquisition in the last 12 months.",
                claim="CoreStack completed its acquisition of BetterCloud on March 31, 2026.",
                evidence_type="ACQUISITION",
                url="https://www.corestack.io/news/corestack-acquires-bettercloud",
            ),
            "CoreStack announced its acquisition of BetterCloud. BetterCloud is now part of CoreStack, and the companies are realizing post-acquisition synergies.",
        ),
    ]

    for row, source in controls:
        first, final = _stage_prompts(row, source)
        for prompt in (first, final):
            assert row["_target_signal_text"] in prompt
            assert row["claim"] in prompt
            assert "semantically" in prompt
        assert source in final
    assert "brand new web store" in controls[0][1]
    assert "post-acquisition synergies" in controls[1][1]


def _paragraph_inputs(paragraph: str, sources: str | tuple[str, ...], quote: str):
    url = "https://www.modernretail.co/operations/open-box-marketplace-rebel-is-now-selling-better-for-you-snacks/"
    company = SimpleNamespace(
        company_name="REBEL",
        company_website="https://fromrebel.com",
        intent_details=paragraph,
        intent_signals=[SimpleNamespace(matched_icp_signal=0, description=REBEL_CLAIM, url=url)],
    )
    icp = SimpleNamespace(
        prompt="Find online commerce companies with a recent new shopping capability, store format, or major retail channel.",
        product_service="Commerce software",
        intent_signals=[REBEL_TARGET],
    )
    source_texts = (sources,) if isinstance(sources, str) else sources
    results = [{
        "after_decay": 51,
        "matched_icp_signal": 0,
        "evidence_urls": [url],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": "2026-05-15",
            "authoritative_date_basis": "event",
            "verification_trace": {
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported",
                    "same_entity_check": "pass",
                    "supporting_quotes": [quote],
                    "evidence_urls_used": [url],
                }]},
                "verified_source_context": [
                    {"url": url, "text": source} for source in source_texts
                ],
            },
        },
    }]
    fit = {"gate": "company_fit", "decision": "match", "dimension_evidence": {}}
    return company, icp, results, fit


def _binding(document: dict, quote: str) -> dict:
    for index, values in intent_details._bound_evidence_sources(document).items():
        if any(quote in value for value in values):
            return {"source_index": index, "quote": quote}
    raise AssertionError(f"quote is not admitted: {quote!r}")


def _run_review(monkeypatch, inputs, builder):
    async def judge(prompt, **kwargs):
        assert kwargs["max_tokens"] == 800
        assert "An unconditional increase, effect, or\nperformance outcome is a factual assertion" in kwargs["system_prompt"]
        document = json.loads(prompt)
        return json.dumps(builder(document), separators=(",", ":"))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    return asyncio.run(intent_details.review_intent_details(*inputs))


def test_rebel_unconditional_checkout_outcome_stays_factual_and_event_mismatch(monkeypatch) -> None:
    quote = "added bulk-sized snacks and pantry products to its lineup in mid-May"
    inputs = _paragraph_inputs(REBEL_PARAGRAPH, (REBEL_EVENT_EXCERPT, REBEL_OUTCOME_EXCERPT), quote)

    def response(document):
        units = document["intent_details_units"]
        assert len(units) == 3
        grounding = [
            {"unit_id": 0, "contains_factual_claim": True, "status": "VERIFIED", "evidence": [_binding(document, quote)]},
            {"unit_id": 1, "contains_factual_claim": True, "status": "VERIFIED", "evidence": [_binding(document, quote)]},
            {"unit_id": 2, "contains_factual_claim": True, "status": "VERIFIED", "evidence": [_binding(document, REBEL_OUTCOME_EXCERPT)]},
        ]
        return {
            "unit_grounding": grounding,
            "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
            "facts_supported": True,
            "verified_signals_covered": True,
            "relevance_grounded": True,
            "connects_icp": False,
            "natural_paragraph": True,
        }

    receipt = _run_review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is True
    assert receipt["checks"]["connects_icp"] is False


def test_conditional_implication_and_supported_catalog_expansion_remain_valid(monkeypatch) -> None:
    paragraph = (
        "REBEL added bulk-sized snacks and pantry products to its marketplace in mid-May. "
        "The additions broadened its catalog. This could increase checkout activity."
    )
    quote = "added bulk-sized snacks and pantry products to its lineup in mid-May"
    inputs = list(_paragraph_inputs(paragraph, (REBEL_EVENT_EXCERPT, REBEL_OUTCOME_EXCERPT), quote))
    inputs[1].intent_signals = [
        "Launched a new product category or expanded its assortment in the last year."
    ]

    def response(document):
        assert len(document["intent_details_units"]) == 3
        grounding = [
            {"unit_id": 0, "contains_factual_claim": True, "status": "VERIFIED", "evidence": [_binding(document, quote)]},
            {"unit_id": 1, "contains_factual_claim": True, "status": "VERIFIED", "evidence": [_binding(document, quote)]},
            {"unit_id": 2, "contains_factual_claim": False, "status": "VERIFIED", "evidence": []},
        ]
        return {
            "unit_grounding": grounding,
            "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
            "facts_supported": True,
            "verified_signals_covered": True,
            "relevance_grounded": True,
            "connects_icp": True,
            "natural_paragraph": True,
        }

    receipt = _run_review(monkeypatch, tuple(inputs), response)
    assert receipt["decision"] == "match"
    assert all(receipt["checks"].values())
