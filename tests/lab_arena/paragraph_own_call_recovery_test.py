"""Raw paragraph packets are not shared before native evidence validation."""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from lab_arena import broker
from qualification.scoring import intent_details as review, verification_helpers
from tests.lab_arena.test_verifier_provider_cache import (
    JudgmentLedger, _MODEL, _REQUEST, _answer, _broker, _call, _context,
)


NAMES = (
    "arena_intent_details_review",
    "arena_intent_details_review_v3",
    "arena_intent_details_citation_repair",
    "arena_intent_details_semantic_repair",
)

# The November 17 Hilb failure, retained without credentials or unrelated page
# text. The repair omitted the intervening Share text and stitched two spans.
ARTICLE = (
    "Hilb Group has completed the acquisition of a South Carolina‑based property "
    "and casualty and employee benefits agency with full‑service risk solution "
    "capabilities. The deal, which marks the US brokerage's 200th transaction "
    "since its launch in 2009, became effective Feb. 1."
)
HEADER = "Deal marks the company's 200th acquisition Feb 24, 2026"
CLAIM = (
    "Hilb Group completed an acquisition of a South Carolina-based property and "
    "casualty and employee benefits agency on February 1, 2026, announced "
    "February 24, 2026."
)


def retained_document():
    return {
        "company": {"name": "Hilb", "website": "https://www.hilbgroup.com/"},
        "icp": {}, "verified_company_evidence": [],
        "intent_details_units": [
            {"unit_id": 0, "text": CLAIM},
            {"unit_id": 1, "text": "This acquisition may help Hilb Group expand "
             "its insurance brokerage and risk advisory offering."},
        ],
        "verified_signals": [{"matched_icp_signal": 0}],
        "admitted_evidence": [
            {"source_index": 0, "evidence_kind": "verified_source_context",
             "admitted_text": [HEADER + "\nShare\n" + ARTICLE]},
            {"source_index": 1, "evidence_kind": "verified_signal_observation",
             "admitted_text": [ARTICLE],
             "observed_dates": [{"basis": "event_date", "date": "2026-02-01"}]},
        ],
    }


def retained_initial():
    return {
        **{name: True for name in review._CHECKS}, "facts_supported": False,
        "signal_coverage": [{"covered": True, "matched_icp_signal": 0}],
        "unit_grounding": [
            {"unit_id": 0, "contains_factual_claim": True, "status": "UNPROVEN",
             "evidence": [{"source_index": 0, "quote": ARTICLE},
                          {"source_index": 1, "quote": "2026-02-01"}]},
            {"unit_id": 1, "contains_factual_claim": False,
             "status": "VERIFIED", "evidence": []},
        ],
    }


def retained_bad_repair():
    return {"repairs": [{"unit_id": 0, "status": "VERIFIED", "evidence": [
        {"source_index": 0, "quote": HEADER + " " + ARTICLE},
        {"source_index": 1, "quote": "2026-02-01"},
    ]}]}


def response(value):
    result = _answer()
    result["choices"][0]["message"]["content"] = json.dumps(value)
    return result


def native_repair(initial, repair, document):
    with pytest.raises(review._BoundedReviewRepairNeeded) as error:
        review._validate_review_response(json.dumps(initial), document)
    needed = error.value
    merged = review._merge_semantic_repairs(
        json.dumps(repair), needed.held_response, set(needed.issues),
        needed.semantic_unit_ids,
    )
    return review._validate_review_response(
        json.dumps(merged), document, initial_review=False,
    )


def test_retained_failure_is_nonexact_quote_not_a_terminal_initial_negative():
    document = retained_document()
    with pytest.raises(review._BoundedReviewRepairNeeded) as error:
        native_repair(retained_initial(), retained_bad_repair(), document)
    assert error.value.issues == {0: ("nonexact_quote",)}
    # A continuous source quote passes without relaxing any native check.
    repaired = retained_bad_repair()
    repaired["repairs"][0]["evidence"][0]["quote"] = HEADER + " Share " + ARTICLE
    assert native_repair(retained_initial(), repaired, document)["facts_supported"]


def test_full_review_recovers_within_same_call_budget_and_preserves_paid_replay(monkeypatch):
    initial = response(retained_initial())
    invalid = response(retained_bad_repair())
    # The actual next attempt returned a complete, valid negative instead of
    # another invalid positive. It must pass native validation, not be invented.
    valid = response({"repairs": [{"unit_id": 0, "status": "UNPROVEN", "evidence": []}]})
    store = JudgmentLedger()
    instance, _, transport = _broker(store, [
        (200, initial), (200, invalid), (200, initial), (200, valid),
    ])
    context = _context("paragraph-recovery")
    calls = []

    async def request(prompt, **kwargs):
        parameters = {
            "model": _MODEL, "max_tokens": kwargs["max_tokens"],
            "messages": [{"role": "system", "content": kwargs["system_prompt"]},
                         {"role": "user", "content": prompt}],
            "response_format": kwargs["response_format"],
        }
        result = _call(instance, context, parameters, action_sequence=len(calls) + 1)
        assert result.status == 200
        calls.append((parameters, result))
        return json.loads(result.body)["choices"][0]["message"]["content"]

    monkeypatch.setattr(review, "review_evidence", lambda *a, **kw: retained_document())
    monkeypatch.setattr(verification_helpers, "openrouter_chat", request)
    first = asyncio.run(review.review_intent_details(None, None, [], {}))
    second = asyncio.run(review.review_intent_details(None, None, [], {}))
    assert first["decision"] == "unavailable"
    assert first["failure_reason_code"] == "malformed_response"
    assert second["decision"] == "mismatch"
    assert second["failed_factual_units"] == [
        {"unit_id": 0, "status": "UNPROVEN", "source_indexes": []},
    ]
    assert len(calls) == len(transport.sent) == 4
    original = copy.deepcopy(store.calls[calls[1][1].call["call_identity"]])
    own = _call(instance, context, calls[1][0], action_sequence=2)
    assert own.call["idempotent"] is True and own.body == calls[1][1].body
    assert len(transport.sent) == 4
    assert store.calls[own.call["call_identity"]] == original
    assert sum(item["actual"] for item in store.calls.values()) == 84


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_native_paragraph_protocols_require_own_call(name, api):
    schema = {"name": name, "strict": True, "schema": {"type": "object"}}
    parameters = (
        {"response_format": {"type": "json_schema", "json_schema": schema}}
        if api == "chat" else
        {"text": {"format": {"type": "json_schema", **schema}}}
    )
    assert broker._judgment_requires_own_call(parameters) is True


@pytest.mark.parametrize("parameters", [
    _REQUEST,
    {"response_format": {"type": "json_schema", "json_schema": {"name": "verification"}}},
    {"text": {"format": {"type": "json_schema", "name": "unrelated_verdict"}}},
    {"response_format": {"type": "json_object"}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "arena_intent_details_review_other"}}},
    {"response_format": {"type": "json_schema", "json_schema": {}}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": []}}},
    {"text": {"format": {"type": "text", "name": NAMES[0]}}},
    {"response_format": None}, {"text": None},
])
def test_other_protocols_keep_their_existing_reuse(parameters):
    assert broker._judgment_requires_own_call(parameters) is False


def test_current_native_protocols_stay_registered_for_own_call_replay():
    for response_format in (
        review._RESPONSE_FORMAT, review._CITATION_REPAIR_RESPONSE_FORMAT,
        review._SEMANTIC_REPAIR_RESPONSE_FORMAT,
    ):
        assert broker._judgment_requires_own_call({"response_format": response_format})
