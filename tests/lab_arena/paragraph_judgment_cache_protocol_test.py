"""Cache completeness does not turn contradictory paragraph verdicts into authority."""
from __future__ import annotations

import copy
import json

import pytest

from lab_arena import broker
from qualification.scoring import intent_details as review


NEW_NAME = "arena_intent_details_review_v3"
OLD_NAME = "arena_intent_details_review"


def request(operation="openrouter.chat", *, name=NEW_NAME):
    format_doc = copy.deepcopy(review._RESPONSE_FORMAT["json_schema"])
    format_doc["name"] = name
    if operation == "openrouter.chat":
        return {"model": "openai/gpt-4o-mini", "max_tokens": 800,
                "messages": [{"role": "user", "content": "Review the supplied paragraph."}],
                "response_format": {"type": "json_schema", "json_schema": format_doc}}
    return {"model": "openai/gpt-4o-mini", "max_output_tokens": 800,
            "input": "Review the supplied paragraph.",
            "text": {"format": {"type": "json_schema", **format_doc}}}


def verdict(*, aggregate=False, status="VERIFIED"):
    return {"unit_grounding": [
        {"unit_id": 0, "contains_factual_claim": True, "status": status,
         "evidence": [{"source_index": 0, "quote": "Example launched a product."}]},
        {"unit_id": 1, "contains_factual_claim": False, "status": "VERIFIED", "evidence": []},
    ], "signal_coverage": [{"matched_icp_signal": 0, "covered": True}],
        **{name: True for name in review._CHECKS}, "facts_supported": aggregate}


def answer(value, operation="openrouter.chat", *, cost="0.000120"):
    if operation == "openrouter.chat":
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(value)}}], "usage": {"cost": cost}}
    return {"object": "response", "status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": json.dumps(value)}]}],
        "usage": {"cost": cost}}


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("name,aggregate,status,expected", [
    (NEW_NAME, False, "VERIFIED", False),
    (NEW_NAME, True, "VERIFIED", True),
    (NEW_NAME, False, "UNPROVEN", True),
    (NEW_NAME, False, "CONTRADICTED", True),
    # Native validation may safely normalize a positive aggregate downward.
    (NEW_NAME, True, "UNPROVEN", True),
    (OLD_NAME, False, "VERIFIED", True),
    ("unrelated_verdict", False, "VERIFIED", True),
])
def test_only_new_paragraph_protocol_excludes_retryable_conflict(operation, name, aggregate, status, expected):
    value = verdict(aggregate=aggregate, status=status)
    assert broker._complete_judgment_response(operation, request(operation, name=name),
        status=200, body=json.dumps(answer(value, operation)).encode(), call_succeeded=True) is expected


def test_native_parser_still_rejects_aggregate_conflict_without_changing_verdict():
    document = {"intent_details_units": [
        {"unit_id": 0, "text": "Example launched a product."},
        {"unit_id": 1, "text": "This may offer an opportunity."}],
        "verified_signals": [{"matched_icp_signal": 0}],
        "admitted_evidence": [{"source_index": 0, "evidence_kind": "verified_source_context",
                               "admitted_text": ["Example launched a product."]}]}
    with pytest.raises(ValueError, match="factual aggregate conflicts"):
        review._validate_review_response(json.dumps(verdict()), document, initial_review=False)


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_negative_aggregate_with_no_factual_units_is_also_retryable(operation):
    value = verdict()
    value["unit_grounding"] = [value["unit_grounding"][1]]
    document = {"intent_details_units": [{"unit_id": 1, "text": "This may offer an opportunity."}],
                "verified_signals": [{"matched_icp_signal": 0}], "admitted_evidence": []}
    with pytest.raises(ValueError, match="covered verified signals require a factual"):
        review._validate_review_response(json.dumps(value), document)
    assert not broker._complete_judgment_response(operation, request(operation), status=200,
        body=json.dumps(answer(value, operation)).encode(), call_succeeded=True)


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("business_match", [True, False])
def test_grounded_fact_with_pure_conditional_and_independent_business_check_stays_cacheable(operation, business_match):
    value = verdict(aggregate=True)
    value["connects_icp"] = business_match
    document = {"intent_details_units": [
        {"unit_id": 0, "text": "Example launched a product."},
        {"unit_id": 1, "text": "This may offer an opportunity."}],
        "verified_signals": [{"matched_icp_signal": 0}],
        "admitted_evidence": [{"source_index": 0, "evidence_kind": "verified_source_context",
                               "admitted_text": ["Example launched a product."]}]}
    checks = review._validate_review_response(json.dumps(value), document)
    assert checks["facts_supported"] is True and checks["connects_icp"] is business_match
    assert broker._complete_judgment_response(operation, request(operation), status=200,
        body=json.dumps(answer(value, operation)).encode(), call_succeeded=True)


def test_current_paragraph_schema_uses_new_cache_namespace_only():
    assert review._RESPONSE_FORMAT["json_schema"]["name"] == NEW_NAME


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_missing_or_wrong_unit_fields_still_fail_schema_completeness(operation):
    value = verdict()
    for changed in [{**value, "unit_grounding": []},
                    {**value, "facts_supported": "false"}]:
        assert not broker._complete_judgment_response(operation, request(operation), status=200,
            body=json.dumps(answer(changed, operation)).encode(), call_succeeded=True)
