"""Keep the buyer's requested event state through Arena intent verification."""

import asyncio
import json

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.prompts._common import FINAL_JUDGE_PROMPT_MAX_CHARS
from tests.test_arena_score_integrity import _company_model, _icp_model


URL = "https://www.cyberhaven.com/press-releases/cyberhaven-introduces-flow-ai-native-data-security"
TARGET = (
    "Launched a new product or major platform capability in the last 12 months, "
    "per a press release, product page, or changelog."
)
SHIPPED_REQUEST = (
    "Find security vendors that SHIPPED a major detection or identity capability "
    "in the last 12 months. They must also fit our separate company profile."
)
INTRODUCTION = (
    "On July 28, 2026, Cyberhaven introduced Flow, an AI-native data security "
    "platform. Flow will be available in the coming quarter. Customers may "
    "request early access now."
)


def _model_verdict(status: str, body: str) -> dict:
    quote = (
        "Flow will be available in the coming quarter."
        if status == "contradicted"
        else body.split("\n")[0]
    )
    return {"answer": {
        "overall_verdict": "qualified" if status == "supported" else "disqualified",
        "overall_confidence": "high",
        "signal_evaluations": [{
            "signal_id": "signal-1",
            "signal_status": status,
            "verification_mode": "source_grounded",
            "source_urls_supplied": [URL],
            "evidence_urls_used": [URL],
            "source_accessibility": "accessible",
            "same_entity_check": "pass",
            "supporting_quotes": [quote] if status == "supported" else [],
            "contradicting_quotes": [quote] if status == "contradicted" else [],
            "confidence": "high",
            "claim_matches_miner_date": "match",
            "risk_notes": ["source_event_date:2026-07-28"],
        }],
    }, "model": "mock-final-judge", "usage": {}}


@pytest.mark.parametrize("body,buyer_request,target,status", [
    (INTRODUCTION, SHIPPED_REQUEST, TARGET, "contradicted"),
    (
        "On July 28, 2026, Cyberhaven released Flow's data detection capability "
        "to all customers; it is available now.\nA separate identity module "
        "is planned for next year.",
        SHIPPED_REQUEST, TARGET,
        "supported",
    ),
    (
        INTRODUCTION,
        "Find vendors that ANNOUNCED a major data security capability.",
        "Announced a new major capability in the last 12 months, per a press release.",
        "supported",
    ),
    (
        "On July 28, 2026, Cyberhaven made Flow's detection capability available "
        "to all existing customers, who can enable and use it today in public "
        "preview.\nBroader general availability is planned for next quarter.",
        "Find vendors with a new detection capability currently available to existing customers.",
        TARGET,
        "supported",
    ),
    (
        INTRODUCTION + "\nCyberhaven made its older Data Lineage product "
        "generally available in 2025.",
        SHIPPED_REQUEST, TARGET,
        "contradicted",
    ),
])
def test_arena_scorer_passes_request_and_final_judge_controls_event_state(
    monkeypatch, body, buyer_request, target, status,
) -> None:
    prompts = []

    async def fetch(urls, *args, **kwargs):
        assert urls == [URL]
        return {"results": [{
            "url": URL,
            "title": "Cyberhaven Introduces Flow",
            "text": body,
            "source_publication_date": "2026-07-28",
        }], "statuses": []}

    async def judge(_client, _model, prompt):
        prompts.append(prompt)
        signal = json.loads(prompt.split("Intent signal to verify:\n", 1)[1].split(
            "\n\nThree-part verification", 1,
        )[0])
        assert signal["buyer_original_request"] == buyer_request
        assert signal["target_icp_signal"] == target
        assert body in prompt
        assert "BUYER ORIGINAL REQUEST — MATCHED EVENT ONLY" in prompt
        assert "future availability statement for that" in prompt
        assert len(prompt) <= FINAL_JUDGE_PROMPT_MAX_CHARS
        return _model_verdict(status, body)

    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    signal = _company_model([{
        "source": "news", "description": "Cyberhaven introduced Flow on July 28, 2026",
        "url": URL, "date": "2026-07-28", "snippet": "Cyberhaven introduces Flow",
        "matched_icp_signal": 0,
    }]).intent_signals[0]
    icp = _icp_model().model_copy(update={
        "prompt": buyer_request,
        "intent_signals": [target],
        "intent_signal_evidence_types": ["PRODUCT_LAUNCH"],
        "intent_max_age_days": 365,
    })
    with use_evaluation_date("2026-10-01"):
        result = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, icp, None, "Cyberhaven", "https://www.cyberhaven.com",
            stage1_soft_reject=True, llm_only_intent_gate=True,
            integrity_policy=True,
        ))
    assert len(prompts) == 1
    assert (result[0] > 0) == (status == "supported")


def test_missing_buyer_context_preserves_legacy_prompt_and_bounded_transport():
    row = {
        "id": "signal-1", "company": "Cyberhaven",
        "website": "https://www.cyberhaven.com", "company_linkedin": "",
        "claim": "Cyberhaven introduced Flow", "signal_type": "intent",
        "claimed_source_urls": [URL], "_target_signal_text": TARGET,
    }
    from qualification.scoring.prompts._common import build_final_judge_prompt

    legacy = build_final_judge_prompt(row, {"results": [{"url": URL, "text": INTRODUCTION}]})
    assert "buyer_original_request" not in legacy
    assert "BUYER ORIGINAL REQUEST" not in legacy
    row["_buyer_request_context"] = SHIPPED_REQUEST
    current = build_final_judge_prompt(row, {"results": [{"url": URL, "text": "x" * 60_000}]})
    assert len(current) <= FINAL_JUDGE_PROMPT_MAX_CHARS
    assert SHIPPED_REQUEST in current


def test_oversized_buyer_context_fails_closed_before_judge():
    result = asyncio.run(verifier.verify_three_stage(
        object(), company_name="Cyberhaven", company_linkedin="",
        company_website="https://www.cyberhaven.com", source_url=URL,
        miner_claim="Cyberhaven introduced Flow", target_signal_text=TARGET,
        buyer_request_context="x" * 12_001,
    ))
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "candidate_prompt_input_unsafe"
