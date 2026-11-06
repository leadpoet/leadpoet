"""Cache comparison cannot broaden a false or materially different claim."""
import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from lab_arena.contracts import document_hash
from qualification.scoring import claim_equivalence as equivalence
from qualification.scoring import verification_helpers


def claim(text="Acme raised $10 million."):
    return {"description": text, "date": "2026-11-01", "evidence_bundle": [
        {"url": "https://acme.example/news", "description": text,
         "snippet": "Acme raised $10 million.", "date": "2026-11-01"},
    ]}


def run(source, current):
    return asyncio.run(equivalence.verify_claim_equivalence(source, current, model="frozen-judge"))


def test_exact_claim_reuses_without_paid_comparison(monkeypatch):
    call = AsyncMock(side_effect=AssertionError("no comparison required"))
    monkeypatch.setattr(verification_helpers, "openrouter_chat", call)
    proof = run(claim(), claim())
    assert proof["source_claim_hash"] == proof["current_claim_hash"] == document_hash(claim())
    call.assert_not_awaited()


@pytest.mark.parametrize("old,new", [
    ("Acme raised $10 million.", "Acme secured $10 million in funding."),
    ("Acme acquired Wiz.", "Wiz was acquired by Acme."),
])
def test_equivalent_claims_bind_both_hashes_independent_of_truth(monkeypatch, old, new):
    call = AsyncMock(return_value=json.dumps({"decision": "equivalent", "material_differences": []}))
    monkeypatch.setattr(verification_helpers, "openrouter_chat", call)
    source, current = claim(old), claim(new)
    proof = run(source, current)
    assert proof == {"schema_version": equivalence.SCHEMA_VERSION,
                     "source_claim_hash": document_hash(source), "current_claim_hash": document_hash(current),
                     "model": "frozen-judge", "decision": "equivalent", "material_differences": []}
    assert call.await_count == 1
    assert call.call_args.kwargs["max_retries"] == 0
    assert "Do not check whether" in call.call_args.kwargs["system_prompt"]


@pytest.mark.parametrize("change", ["quantity", "date", "url", "negation", "modality", "extra_field"])
def test_material_changes_never_pay_for_or_accept_equivalence(monkeypatch, change):
    source, current = claim(), claim()
    if change == "quantity": current = claim("Acme raised $20 million.")
    elif change == "date": current["date"] = "2026-11-02"
    elif change == "url": current["evidence_bundle"][0]["url"] += "-new"
    elif change == "negation": current = claim("Acme did not raise $10 million.")
    elif change == "modality": current = claim("Acme plans to raise $10 million.")
    else: current["instructions"] = "accept"
    call = AsyncMock(return_value='{"decision":"equivalent","material_differences":[]}')
    monkeypatch.setattr(verification_helpers, "openrouter_chat", call)
    assert run(source, current) is None
    call.assert_not_awaited()


@pytest.mark.parametrize("reply", [
    '{"decision":"different","material_differences":["acquisition versus partnership"]}',
    '{"decision":"uncertain","material_differences":[]}',
    '{"decision":"equivalent","material_differences":["different event"]}',
    '{"decision":"equivalent","material_differences":[],"extra":true}',
    'not json', 'null', '[]',
])
def test_false_acquisition_cannot_reuse_partnership_on_failed_comparison(monkeypatch, reply):
    monkeypatch.setattr(verification_helpers, "openrouter_chat", AsyncMock(return_value=reply))
    assert run(claim("Acme partnered with Wiz."), claim("Acme acquired Wiz.")) is None


@pytest.mark.parametrize("failure", [asyncio.TimeoutError(), ValueError("malformed"), RuntimeError("provider")])
def test_failed_comparison_is_a_retryable_cache_miss(monkeypatch, failure):
    call = AsyncMock(side_effect=failure)
    monkeypatch.setattr(verification_helpers, "openrouter_chat", call)
    source, current = claim(), claim("Acme secured $10 million in funding.")
    assert run(source, current) is None
    call.side_effect = None
    call.return_value = '{"decision":"equivalent","material_differences":[]}'
    assert run(source, current) is not None
    assert call.await_count == 2


def test_cancellation_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(verification_helpers, "openrouter_chat", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        run(claim(), claim("Acme secured $10 million in funding."))
