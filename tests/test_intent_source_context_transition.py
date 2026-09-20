"""Verified source context stays bound through scoring and paragraph review."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import ICPPrompt, IntentSignal
from qualification.scoring import intent_details
from qualification.scoring import intent_verification_three_stage as three_stage
from qualification.scoring.lead_scorer import _score_single_intent_signal


SOURCE_URL = "https://acme.com/news/leadership"
SOURCE_TEXT = (
    "September 14, 2026. Acme appointed a new chief executive effective today. "
    "The executive leads technology and operations."
)
SUPPORTING_QUOTE = "Acme appointed a new chief executive."


def _provider_answer(status: str) -> dict:
    return {
        "answer": {
            "signal_evaluations": [{
                "signal_status": status,
                "confidence": "high",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "evidence_urls_used": [SOURCE_URL],
                "claim_matches_miner_date": "consistent",
                "supporting_quotes": [SUPPORTING_QUOTE],
                "unsupported_parts": [],
                "risk_notes": ["source_publication_date:2026-09-14"],
            }],
            "overall_verdict": (
                "qualified" if status == "supported" else "unverified"
            ),
            "overall_confidence": "high",
            "summary": "Sanitized transition fixture.",
            "missing_or_risks": [],
        },
        "model": "test-model",
        "usage": {},
    }


def _signal() -> IntentSignal:
    return IntentSignal(
        source="news",
        description=(
            "Acme appointed a new chief executive effective September 14, 2026."
        ),
        url=SOURCE_URL,
        date="2026-09-14",
        snippet=SUPPORTING_QUOTE,
        matched_icp_signal=0,
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="source-context-transition",
        prompt="Find software companies with recent leadership changes",
        industry="Software",
        sub_industry="SaaS",
        employee_count="51-200",
        company_stage="",
        geography="United States",
        country="United States",
        product_service="Reporting software",
        intent_signals=["Announced a leadership change in the past year"],
        intent_signal_evidence_types=["LEADERSHIP_CHANGE"],
    )


@pytest.mark.parametrize(
    ("fetched_url", "context_expected"),
    [
        (SOURCE_URL, True),
        (SOURCE_URL + "/", True),
        ("https://acme.com/news/different-event", False),
        (SOURCE_URL + "?article=other", False),
    ],
)
def test_verified_context_stays_bound_from_score_receipt_to_review(
    monkeypatch,
    fetched_url,
    context_expected,
):
    """Canonical equivalents survive; a different resource never crosses gates."""

    signal = _signal()
    provider = AsyncMock(side_effect=[
        _provider_answer("unable_to_verify"),
        _provider_answer("supported"),
    ])
    fetch = AsyncMock(return_value={
        "results": [{
            "url": fetched_url,
            "text": SOURCE_TEXT,
            "source_publication_date": "2026-09-14",
        }],
        "statuses": [],
    })
    monkeypatch.setattr(three_stage, "_call_openrouter", provider)
    monkeypatch.setattr(three_stage, "_fetch_sd_then_exa", fetch)
    monkeypatch.setenv("LEADPOET_COMPETITION_EVALUATION_DATE", "2026-09-20")

    verdicts: list[dict] = []
    score, confidence, date_status, _, matched_index = asyncio.run(
        _score_single_intent_signal(
            signal,
            _icp(),
            None,
            "Acme",
            "https://acme.com",
            company_linkedin="https://www.linkedin.com/company/acme",
            stage1_soft_reject=True,
            llm_only_intent_gate=True,
            integrity_policy=True,
            verdict_out=verdicts,
            evidence_signals=[signal],
        )
    )

    assert fetch.await_count == 1
    verdict = verdicts[-1]
    if not context_expected:
        assert provider.await_count == 1
        assert score == 0.0
        assert confidence == 0
        assert date_status == "verified"
        assert matched_index == -1
        assert verdict["decision"] == "rejected_verifier_error"
        assert verdict["rejection_reason"] == "evidence_fetch_failed"
        assert verdict["client_ready"] is False
        assert "verified_source_context" not in verdict["verification_trace"]
        return

    assert provider.await_count == 2
    assert score == 54.0
    assert confidence == 90
    assert date_status == "in_window"
    assert matched_index == 0
    assert verdict["decision"] == "verified"
    assert verdict["authoritative_date"] == "2026-09-14"
    assert verdict["authoritative_date_basis"] == "publication_date"

    signal_result = {
        "raw": score,
        "after_decay": score,
        "matched_icp_signal": matched_index,
        "evidence_urls": [SOURCE_URL],
        "judge_verdict": verdict,
    }
    company = SimpleNamespace(
        company_name="Acme",
        company_website="https://acme.com",
        intent_signals=[signal],
        intent_details=(
            "Acme appointed a new chief executive on September 14, 2026, "
            "who leads technology and operations, creating a timely reason "
            "to assess its reporting software needs."
        ),
    )
    company_fit = {
        "gate": "company_fit",
        "decision": "match",
        "dimension_evidence": {
            "industry": {
                "decision": "match",
                "web_evidence": {
                    "url": "https://acme.com",
                    "quote": "Acme provides reporting software.",
                },
            },
        },
    }
    review = intent_details.review_evidence(
        company,
        _icp(),
        [signal_result],
        company_fit,
    )
    receipt_context = verdict["verification_trace"].get(
        "verified_source_context", []
    )
    review_context = review["verified_signals"][0].get("source_context", [])

    expected = [{
        "url": SOURCE_URL,
        "text": SOURCE_TEXT,
        "source_publication_date": "2026-09-14",
    }]
    assert receipt_context == expected
    assert review_context == expected
