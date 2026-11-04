"""Native paragraph requests retain mixed facts and uncertain outcomes separately.

Model replies are controlled fixtures. These tests prove request/guard behavior,
not that a live model follows the rubric.
"""
import asyncio
from copy import deepcopy
import json

import pytest

from qualification.scoring import intent_details as review, verification_helpers


FACT = 'Acme raised capital on September 1, 2026.'
PRODUCT = 'Acme provides workflow software.'


def document(text):
    return {'intent_details_units': [{'unit_id': 0, 'text': FACT},
                                     {'unit_id': 1, 'text': text}],
            'verified_signals': [{'matched_icp_signal': 0}],
            'admitted_evidence': [
                {'source_index': 0, 'evidence_kind': 'verified_source_context', 'admitted_text': [FACT]},
                {'source_index': 1, 'evidence_kind': 'company_fit', 'admitted_text': [PRODUCT]}]}


def answer(*, factual, supported, business=True):
    return {**{key: True for key in review._CHECKS},
            'facts_supported': supported,
            'connects_icp': business, 'relevance_grounded': business,
            'signal_coverage': [{'matched_icp_signal': 0, 'covered': True}],
            'unit_grounding': [
                {'unit_id': 0, 'contains_factual_claim': True, 'status': 'VERIFIED',
                 'evidence': [{'source_index': 0, 'quote': FACT}]},
                {'unit_id': 1, 'contains_factual_claim': factual,
                 'status': 'VERIFIED' if supported or not factual else 'UNPROVEN',
                 'evidence': [{'source_index': 1, 'quote': PRODUCT}] if factual and supported else []}]}


def invoke(monkeypatch, doc, replies):
    requests = []
    monkeypatch.setattr(review, 'review_evidence', lambda *a, **k: deepcopy(doc))

    async def respond(prompt, **kwargs):
        requests.append({'prompt': prompt, **kwargs})
        return json.dumps(replies[len(requests) - 1])

    monkeypatch.setattr(verification_helpers, 'openrouter_chat', respond)
    result = asyncio.run(review.review_intent_details(None, None, [], {}))
    return result, requests


@pytest.mark.parametrize('text,factual,supported,business,decision', [
    ('This funding may create future growth opportunities.', False, True, True, 'match'),
    ('This funding may create future growth opportunities.', False, True, False, 'mismatch'),
    ('This funding may help Acme expand its workflow software.', True, True, True, 'match'),
    ('This funding could expand its undocumented analytics API.', True, False, False, 'mismatch'),
    ('This funding will double revenue.', True, False, False, 'mismatch'),
])
def test_healthy_conditional_and_mixed_fact_controls(monkeypatch, text, factual, supported, business, decision):
    result, calls = invoke(monkeypatch, document(text), [answer(
        factual=factual, supported=supported, business=business)])
    assert len(calls) == 1
    assert result['decision'] == decision
    assert result['checks']['facts_supported'] is supported
    assert result['checks']['connects_icp'] is business
    system = ' '.join(calls[0]['system_prompt'].split())
    assert 'check every factual clause' in system
    assert 'conditional commercial implication with no new asserted fact' in system
    assert 'An unconditional increase, effect, or performance outcome is a factual assertion' in system
    assert 'Require natural prose, not headings, bullet lists, field labels or internal scoring commentary' in system



@pytest.mark.parametrize('correction_consistent', [False, True])
def test_existing_complete_correction_preserves_guard_and_call_bound(monkeypatch, correction_consistent):
    doc = document('This funding could expand its undocumented analytics API.')
    inconsistent = answer(factual=False, supported=False, business=False)
    corrected = answer(factual=True, supported=False, business=False) if correction_consistent else inconsistent
    result, calls = invoke(monkeypatch, doc, [inconsistent, corrected])
    assert len(calls) == 2
    assert calls[0]['prompt'] == calls[1]['prompt'] == review._review_document_prompt(doc)
    assert all(c['max_tokens'] == 800 and c['model'] == review.REVIEW_MODEL
               and c['max_retries'] == 0 and c['response_format'] == review._RESPONSE_FORMAT for c in calls)
    correction = ' '.join(calls[1]['system_prompt'].split())
    assert "every unit's factual classification, status and evidence" in correction
    assert 'No particular decision is requested' in correction
    if correction_consistent:
        assert result['decision'] == 'mismatch'
        assert result['checks']['facts_supported'] is False
    else:
        assert result['decision'] == 'unavailable'
        assert result['failure_reason_code'] == 'malformed_response'
        assert 'checks' not in result


def test_legacy_contradiction_is_not_promoted_by_the_parser():
    doc = document('This may create an opportunity.')
    invalid = json.dumps(answer(factual=False, supported=False, business=False))
    with pytest.raises(review._BoundedReviewRepairNeeded) as raised:
        review._validate_review_response(invalid, doc)
    assert raised.value.complete_response_correction is True
    with pytest.raises(ValueError, match='factual aggregate conflicts'):
        review._validate_review_response(invalid, doc, initial_review=False)
