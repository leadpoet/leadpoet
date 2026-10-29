"""One source judge evaluates its assigned criterion, not the whole ICP."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.prompts import _common
from tests.test_arena_score_integrity import _company_model, _icp_model


SAVED = json.loads((Path(__file__).parent / "fixtures/epam_bonus_independent_intent_scope.json").read_text())
SIGNAL = SAVED["signal"]
URL = SIGNAL["claimed_source_urls"][0]
SCOPE = (
    "Other independently listed primary or bonus intent requirements are checked "
    "separately; their absence must not change this signal's status or overall verdict."
)


def _signal_document(prompt):
    return json.JSONDecoder().raw_decode(prompt.split("Intent signal to verify:\n", 1)[1].lstrip())[0]


@pytest.mark.parametrize("target,body,status", [
    (SIGNAL["target_icp_signal"], SAVED["source_quote"], "supported"),
    # A true bonus is not evidence for an assigned partnership target.
    ("Announced a strategic partnership or alliance in the last 12 months.",
     SAVED["source_quote"], "contradicted"),
    # A conjunction inside the assigned target remains mandatory.
    ("Launched a new service AND announced a strategic partnership in the last 12 months.",
     SAVED["source_quote"], "contradicted"),
    # Broader buyer wording still controls the matched event's requested state.
    ("Launched a new service currently available to customers.",
     "EPAM announced a new service on October 5, 2026. It will be available next year.",
     "contradicted"),
])
def test_native_scorer_preserves_assigned_target_and_source_scope(monkeypatch, target, body, status):
    calls = []

    async def fetch(urls, *args, **kwargs):
        assert urls == [URL]
        return {"results": [{"url": URL, "text": body,
                              "source_publication_date": "2026-10-05"}], "statuses": []}

    async def judge(_client, _model, prompt):
        calls.append(prompt)
        return {"answer": {
            "overall_verdict": "qualified" if status == "supported" else "disqualified",
            "overall_confidence": "high",
            "signal_evaluations": [{
                "signal_id": "signal-1", "signal_status": status,
                "confidence": "high", "same_entity_check": "pass",
                "verification_mode": "source_grounded", "evidence_urls_used": [URL],
                "source_urls_supplied": [URL], "claim_matches_miner_date": "consistent",
                "supporting_quotes": [body] if status == "supported" else [],
                "contradicting_quotes": [] if status == "supported" else [body],
                "risk_notes": ["source_event_date:2026-10-05"],
            }],
        }, "usage": {}, "model": "offline-native-contract"}

    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    signal = _company_model([{
        "source": "news", "url": URL, "description": SIGNAL["miner_claim"],
        "date": "2026-10-05", "matched_icp_signal": 1, "snippet": "",
    }]).intent_signals[0]
    icp = _icp_model().model_copy(update={
        "prompt": SIGNAL["buyer_original_request"],
        "intent_signals": ["Announced a strategic partnership or alliance.", target],
        "intent_signal_evidence_types": ["PARTNERSHIP", "PRODUCT_LAUNCH"],
    })
    with use_evaluation_date("2026-10-24"):
        score = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, icp, None, "EPAM", "https://www.epam.com",
            integrity_policy=True, stage1_soft_reject=True,
            llm_only_intent_gate=True, evidence_signals=[signal],
        ))
    assert len(calls) == 1
    prompt = calls[0]
    doc = _signal_document(prompt)
    assert doc["buyer_original_request"] == SIGNAL["buyer_original_request"]
    assert doc["target_icp_signal"] == target
    assert doc["criterion_evidence"][0]["url"] == URL
    assert body in prompt
    assert SCOPE in " ".join(prompt.split())
    assert "including an explicit conjunction or dependency" in prompt
    assert "future availability statement for that" in prompt
    assert "exact supplied source URL(s)" in prompt
    assert (score[0] > 0) == (status == "supported")
    if status == "supported":
        assert score[-1] == 1


@pytest.mark.parametrize("evidence_type", [None, "SOCIAL_POSTING", "TECHSTACK", "PODCAST_APPEARANCE"])
@pytest.mark.parametrize("combined,reference", [(False, False), (True, False), (True, True)])
def test_scope_survives_all_dispatchers_and_same_event_reference(evidence_type, combined, reference):
    row = {
        "id": "signal-1", "company": "EPAM", "website": "https://www.epam.com",
        "claim": SIGNAL["miner_claim"], "claimed_source_urls": [URL],
        "_target_signal_text": SIGNAL["target_icp_signal"],
        "_buyer_request_context": SIGNAL["buyer_original_request"],
        "_evidence_type": evidence_type,
    }
    if combined:
        row["_evidence_bundle"] = SIGNAL["criterion_evidence"]
    if reference:
        row["_same_event_resolution"] = True
    with use_evaluation_date("2026-10-24"):
        first = verifier._build_verification_prompt(row)
        final = verifier._build_final_judge_prompt(row, {"results": [{"url": URL, "text": "x" * 60_000}]})
    assert SCOPE in " ".join(first.split())
    assert SCOPE in " ".join(final.split())
    assert _signal_document(first) == _signal_document(final)
    assert len(final) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    if combined:
        assert "ONE target_icp_signal" in final


@pytest.mark.parametrize("evidence_type,expected_sha256", [
    # SHA256 of the reviewed pre-fix no-context dispatcher output.
    (None, 'c35d71621025ae42d5ff8f790de0baf898e09908a00355324c951e173315568c'),
    ('SOCIAL_POSTING', 'c35d71621025ae42d5ff8f790de0baf898e09908a00355324c951e173315568c'),
    ('TECHSTACK', '29f49678f90f15655313af6b71c2b7c04f90646b28b511d828f0375d81f1d99f'),
    ('PODCAST_APPEARANCE', '77c1e3655c29c26c7418fc5da8017e4cb82f048335efb44c3b4feaa6d6d86108'),
])
def test_no_context_legacy_prompt_bytes(evidence_type, expected_sha256):
    row = {"id": "signal-1", "company": "EPAM", "website": "https://www.epam.com",
           "claim": SIGNAL["miner_claim"], "claimed_source_urls": [URL],
           "_target_signal_text": SIGNAL["target_icp_signal"], "_evidence_type": evidence_type}
    with use_evaluation_date("2026-10-24"):
        prompt = verifier._build_final_judge_prompt(row, {"results": [{"url": URL, "text": SAVED["source_quote"]}]})
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected_sha256
    assert "BUYER ORIGINAL REQUEST" not in prompt
