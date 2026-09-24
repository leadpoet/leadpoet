"""Bounded unit grounding for the existing Intent Details review call."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import intent_details, verification_helpers
from qualification.scoring.competition import (
    scorer_breakdown_has_retryable_infrastructure_failure,
)


def _inputs(
    paragraph: str,
    source_text: str,
    *,
    supporting_quote: str,
    authoritative_date: str = "2026-03-01",
):
    url = "https://example.test/news"
    company = SimpleNamespace(
        company_name="Example",
        company_website="https://example.test/",
        intent_details=paragraph,
        intent_signals=[SimpleNamespace(
            matched_icp_signal=0,
            description="Submitted claim text is not evidence.",
            url=url,
        )],
    )
    icp = SimpleNamespace(
        prompt="Find companies with a recent supported activity.",
        product_service="Business software",
        intent_signals=["Recent supported activity"],
    )
    results = [{
        "after_decay": 51,
        "matched_icp_signal": 0,
        "evidence_urls": [url],
        "judge_verdict": {
            "decision": "verified",
            "client_ready": True,
            "authoritative_date": authoritative_date,
            "authoritative_date_basis": "event",
            "verification_trace": {
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported",
                    "same_entity_check": "pass",
                    "supporting_quotes": [supporting_quote],
                    "evidence_urls_used": [url],
                }]},
                "verified_source_context": [{
                    "url": url,
                    "text": source_text,
                    "source_publication_date": "2026-03-02",
                }],
            },
        },
    }]
    fit = {
        "gate": "company_fit",
        "decision": "match",
        "dimension_evidence": {
            "industry": {
                "decision": "match",
                "web_evidence": {
                    "url": "https://example.test/",
                    "quote": "Example provides business software.",
                },
            },
        },
    }
    return company, icp, results, fit


def _binding(document, text: str) -> dict:
    for source_index, values in intent_details._bound_evidence_sources(
        document
    ).items():
        if any(text in value for value in values):
            return {"source_index": source_index, "quote": text}
    raise AssertionError(f"No trusted evidence contains {text!r}")


def _response(document, unit_grounding, *, facts_supported, clause=""):
    return {
        **{name: True for name in intent_details._CHECKS},
        "facts_supported": facts_supported,
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        "unit_grounding": unit_grounding,
        "unsupported_factual_clause": clause if not facts_supported else "",
        "unsupported_factual_reason": (
            "The admitted evidence does not establish this factual clause."
            if not facts_supported else ""
        ),
    }


def _verified_unit(document, unit_id: int, quote: str) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": True,
        "status": "VERIFIED",
        "evidence": [_binding(document, quote)],
    }


def _unproven_unit(unit_id: int) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": True,
        "status": "UNPROVEN",
        "evidence": [],
    }


def _nonfactual_unit(unit_id: int) -> dict:
    return {
        "unit_id": unit_id,
        "contains_factual_claim": False,
        "status": "VERIFIED",
        "evidence": [],
    }


def _review(monkeypatch, inputs, response_builder):
    async def judge(prompt, **kwargs):
        assert kwargs["max_tokens"] == 800
        document = json.loads(prompt)
        return json.dumps(response_builder(document), separators=(",", ":"))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    return asyncio.run(intent_details.review_intent_details(*inputs))


def test_typesafe_api_claim_is_an_unproven_complete_unit(monkeypatch):
    paragraph = (
        "Dealroom's September 15, 2026 coverage reports TypeSafe's seed "
        "financing and says Jev is in early access for select developers. "
        "Its API reference documents Jev-powered evaluation of application "
        "state into structured answers."
    )
    supported = (
        "TypeSafe raised seed financing and Jev is in early access for select "
        "developers."
    )
    inputs = _inputs(paragraph, supported, supporting_quote=supported)

    def response(document):
        assert len(document["intent_details_units"]) == 2
        assert "API reference" not in " ".join(
            value
            for values in intent_details._bound_evidence_sources(document).values()
            for value in values
        )
        return _response(
            document,
            [
                _verified_unit(document, 0, supported[:100]),
                _unproven_unit(1),
            ],
            facts_supported=False,
            clause=document["intent_details_units"][1]["text"],
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_common_wealth_daily_growth_cannot_be_omitted(monkeypatch):
    paragraph = (
        "Common Wealth closed a C$12M Series A in March 2026 after bridge "
        "financing supported product innovation and customer growth. In a 2026 "
        "update, the company reports adding about three employers every business "
        "day, with 80% offering retirement benefits for the first time. Flow "
        "Capital reports that Common Wealth was selected to administer Canada's "
        "$30M Personal Support Worker Retirement Savings Innovation Program."
    )
    funding = "Common Wealth closed a C$12M Series A in March 2026."
    first_plans = "Over 80% of employers offer a retirement plan for the first time."
    program = (
        "Common Wealth was selected to administer Canada's $30M Personal "
        "Support Worker Retirement Savings Innovation Program."
    )
    source = f"{funding} {first_plans} {program}"
    inputs = _inputs(paragraph, source, supporting_quote=funding)

    def response(document):
        assert len(document["intent_details_units"]) == 3
        admitted = " ".join(
            value
            for values in intent_details._bound_evidence_sources(document).values()
            for value in values
        )
        assert "three employers every business day" not in admitted
        return _response(
            document,
            [
                _verified_unit(document, 0, funding),
                _unproven_unit(1),
                _verified_unit(document, 2, program[:100]),
            ],
            facts_supported=False,
            clause="adding about three employers every business day",
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_supported_paraphrase_uses_exact_source_span_without_literal_match(
    monkeypatch,
):
    paragraph = "Example secured growth financing to expand its operations."
    quote = "Example raised a $25 million Series A round."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        assert "secured growth financing" not in quote
        return _response(
            document,
            [_verified_unit(document, 0, quote)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_compound_unit_can_bind_two_existing_sources(monkeypatch):
    paragraph = "Example raised a Series A and launched its analytics product."
    financing = "Example raised a Series A."
    launch = "Example launched its analytics product."
    inputs = _inputs(
        paragraph,
        f"{financing} {launch}",
        supporting_quote=financing,
    )

    def response(document):
        signal_source = _binding(document, financing)
        context_source = document["verified_signals"][0]["source_context"][0][
            "source_index"
        ]
        unit = _verified_unit(document, 0, financing)
        unit["evidence"] = [
            signal_source,
            {"source_index": context_source, "quote": launch},
        ]
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_conditional_implication_can_be_a_nonfactual_unit(monkeypatch):
    paragraph = (
        "Example raised a $25 million Series A. This may support future product "
        "delivery."
    )
    quote = "Example raised a $25 million Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        conditional = _nonfactual_unit(1)
        conditional["evidence"] = [_binding(document, quote)]
        return _response(
            document,
            [
                _verified_unit(document, 0, quote),
                conditional,
            ],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"
    assert receipt["checks"]["relevance_grounded"] is True


def test_unproven_compound_unit_can_bind_its_supported_part(monkeypatch):
    paragraph = (
        "Example raised a Series A and added three customers every business day."
    )
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        unit = _unproven_unit(0)
        unit["evidence"] = [_binding(document, quote)]
        return _response(
            document,
            [unit],
            facts_supported=False,
            clause="added three customers every business day",
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "mismatch"
    assert receipt["checks"]["facts_supported"] is False


def test_all_nonfactual_units_conflict_with_complete_signal_coverage(monkeypatch):
    paragraph = "This may support future delivery. This could create demand."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        units = [
            _nonfactual_unit(unit["unit_id"])
            for unit in document["intent_details_units"]
        ]
        return _response(document, units, facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


@pytest.mark.parametrize("mutation", ["omitted", "duplicate"])
def test_incomplete_or_duplicate_unit_ids_remain_retryable(monkeypatch, mutation):
    paragraph = "Example raised a Series A. Example launched a product."
    quote = "Example raised a Series A and launched a product."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        units = [
            _verified_unit(document, unit["unit_id"], quote)
            for unit in document["intent_details_units"]
        ]
        if mutation == "omitted":
            units.pop()
        else:
            units[-1]["unit_id"] = units[0]["unit_id"]
        return _response(document, units, facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"
    assert scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [receipt],
    })


@pytest.mark.parametrize(
    "invalid", ["paragraph_self_quote", "source_index", "source_url"]
)
def test_unbound_or_invalid_source_evidence_is_retryable(monkeypatch, invalid):
    paragraph = "Example claims an undocumented API capability."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        unit = _verified_unit(document, 0, quote)
        if invalid == "paragraph_self_quote":
            unit["evidence"][0]["quote"] = paragraph
        elif invalid == "source_index":
            unit["evidence"][0]["source_index"] = 999
        else:
            unit["evidence"][0]["quote"] = "https://example.test/news"
        return _response(document, [unit], facts_supported=True)

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_authoritative_date_scalar_is_an_allowed_bound_source(monkeypatch):
    paragraph = "Example completed the launch on March 1, 2026."
    quote = "Example completed the launch."
    inputs = _inputs(
        paragraph,
        quote,
        supporting_quote=quote,
        authoritative_date="2026-03-01",
    )

    def response(document):
        return _response(
            document,
            [_verified_unit(document, 0, "2026-03-01")],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_verified_company_fact_quote_is_an_allowed_bound_source(monkeypatch):
    paragraph = "Example provides business software."
    signal_quote = "Example raised a Series A."
    inputs = _inputs(paragraph, signal_quote, supporting_quote=signal_quote)
    company_quote = "Example provides business software."

    def response(document):
        return _response(
            document,
            [_verified_unit(document, 0, company_quote)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "match"


def test_flat_boolean_cannot_override_an_unproven_unit(monkeypatch):
    paragraph = "Example claims an undocumented API capability."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        return _response(
            document,
            [_unproven_unit(0)],
            facts_supported=True,
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_false_diagnostic_must_belong_to_nonverified_factual_unit(monkeypatch):
    paragraph = "Example raised a Series A. Example launched an API."
    quote = "Example raised a Series A."
    inputs = _inputs(paragraph, quote, supporting_quote=quote)

    def response(document):
        return _response(
            document,
            [
                _verified_unit(document, 0, quote),
                _unproven_unit(1),
            ],
            facts_supported=False,
            clause=document["intent_details_units"][0]["text"],
        )

    receipt = _review(monkeypatch, inputs, response)
    assert receipt["decision"] == "unavailable"
    assert receipt["failure_reason_code"] == "malformed_response"


def test_six_unit_two_binding_response_fits_existing_output_bound():
    natural_quote_one = (
        "Common Wealth added employers through its retirement platform every "
        "business day in Canada."
    )[:100]
    natural_quote_two = (
        "TypeSafe evaluates application state and returns structured results "
        "through its documented API."
    )[:100]
    response = {
        **{name: True for name in intent_details._CHECKS},
        "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        "unit_grounding": [
            {
                "unit_id": unit_id,
                "contains_factual_claim": True,
                "status": "VERIFIED",
                "evidence": [
                    {"source_index": 0, "quote": natural_quote_one},
                    {"source_index": 1, "quote": natural_quote_two},
                ],
            }
            for unit_id in range(intent_details._MAX_STATEMENT_UNITS)
        ],
        "unsupported_factual_clause": "",
        "unsupported_factual_reason": "",
    }
    compact = json.dumps(response, separators=(",", ":"), ensure_ascii=False)

    # This is a realistic upper-shape check, not a provider-tokenizer proof.
    assert len(compact.encode("utf-8")) < 2_400
