"""Bounded review of grounded HIRING semantic contradictions."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from tests.test_intent_hiring_functional_role_prompt import (
    TARGET,
    UNIBUDDY_NATIVE_DESCRIPTION,
    UNIBUDDY_TEXT,
    UNIBUDDY_URL,
)


EXPOSURE_QUOTE = (
    "We’re looking for an exceptional mid-level software engineer with a "
    "passion for building great products to join our team. You will gain "
    "exposure to the full stack of the Unibuddy platform across web, native, "
    "and backend to deliver engaging solutions to our users and customers."
)
FOCUS_QUOTE = (
    "This role will be focussed on our Chat, Assistant, and Intelligence products."
)
DUTY_QUOTE = (
    "Demonstrate true ownership of a product - owning its ongoing upkeep, "
    "maintenance, performance, and bug resolution."
)


def _answer(
    *,
    status: str = "contradicted",
    confidence: str = "high",
    overall: str = "disqualified",
    overall_confidence: str | None = None,
    same_entity: str = "pass",
    supporting_quotes: list[str] | None = None,
    contradicting_quotes: list[str] | None = None,
    unsupported_parts: list[str] | None = None,
    url: str = UNIBUDDY_URL,
    claim: str = UNIBUDDY_NATIVE_DESCRIPTION,
) -> dict:
    return {
        "overall_verdict": overall,
        "overall_confidence": overall_confidence or confidence,
        "summary": "Bounded test verdict.",
        "missing_or_risks": [],
        "signal_evaluations": [{
            "signal_id": "signal-1",
            "claim": claim,
            "signal_status": status,
            "verification_mode": "source_grounded",
            "same_entity_check": same_entity,
            "confidence": confidence,
            "evidence_urls_used": [url],
            "supporting_quotes": (
                [EXPOSURE_QUOTE, FOCUS_QUOTE]
                if supporting_quotes is None else supporting_quotes
            ),
            "contradicting_quotes": contradicting_quotes or [],
            "unsupported_parts": (
                [
                    "The posting does not establish direct responsibility for "
                    "building, operating, owning, or maintaining platform components."
                ]
                if unsupported_parts is None else unsupported_parts
            ),
            "claim_matches_miner_date": "consistent",
            "source_accessibility": "Exact supplied source extraction is available.",
            "risk_notes": ["source_publication_date:2026-04-21"],
        }],
    }


def _envelope(answer: dict) -> dict:
    return {"answer": answer, "model": "test-stage3", "usage": {}}


def _eligibility_row() -> dict:
    return {
        "claimed_source_urls": [UNIBUDDY_URL],
        "_integrity_policy": True,
        "_evidence_type": "HIRING",
    }


def test_retained_high_false_negative_shape_is_structurally_eligible() -> None:
    answer = _answer()
    item = answer["signal_evaluations"][0]

    assert verifier._grounded_hiring_contradiction_needs_clarification(
        answer,
        item,
        UNIBUDDY_TEXT,
        row=_eligibility_row(),
        company_quality=True,
        fetched_urls={verifier._normalize_url(UNIBUDDY_URL)},
    )


@pytest.mark.parametrize(
    "control",
    [
        "counterquote",
        "wrong_entity",
        "ungrounded_quote",
        "other_event",
        "not_company_quality",
        "unfetched_url",
        "malformed",
    ],
)
def test_hiring_contradiction_review_bypasses_unsafe_shapes(control: str) -> None:
    answer = _answer()
    item = answer["signal_evaluations"][0]
    row = _eligibility_row()
    company_quality = True
    fetched_urls = {verifier._normalize_url(UNIBUDDY_URL)}
    if control == "counterquote":
        item["contradicting_quotes"] = [FOCUS_QUOTE]
    elif control == "wrong_entity":
        item.update(signal_status="wrong_entity", same_entity_check="fail")
    elif control == "ungrounded_quote":
        item["supporting_quotes"] = ["Text absent from the fetched source."]
    elif control == "other_event":
        row["_same_event_resolution"] = True
    elif control == "not_company_quality":
        company_quality = False
    elif control == "unfetched_url":
        fetched_urls = set()
    elif control == "malformed":
        item.pop("unsupported_parts")

    assert not verifier._grounded_hiring_contradiction_needs_clarification(
        answer,
        item,
        UNIBUDDY_TEXT,
        row=row,
        company_quality=company_quality,
        fetched_urls=fetched_urls,
    )


async def _verify(
    monkeypatch,
    *answers: dict,
    source_text: str = UNIBUDDY_TEXT,
    url: str = UNIBUDDY_URL,
    claim: str = UNIBUDDY_NATIVE_DESCRIPTION,
):
    judge = AsyncMock(side_effect=[_envelope(answer) for answer in answers])
    monkeypatch.setattr(verifier, "_call_openrouter", judge)
    monkeypatch.setattr(
        verifier,
        "_fetch_sd_then_exa",
        AsyncMock(return_value={
            "results": [{
                "url": url,
                "title": "Software Engineer II - Chat Systems",
                "text": source_text,
                "source_publication_date": "2026-04-21",
                "meta": {},
            }],
            "statuses": [{"url": url, "source": "scrapingdog", "stage": "ok"}],
        }),
    )
    result = await verifier.verify_three_stage(
        object(),
        company_name="Unibuddy",
        company_linkedin="",
        company_website="https://unibuddy.com/",
        source_url=url,
        miner_claim=claim,
        target_signal_text=TARGET,
        miner_signal_date="2026-04-21",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        integrity_policy=True,
        company_quality=True,
        verified_company_identity={
            "decision": "match",
            "evidence_source": "company_web_reverification",
            "observed_name": "unibuddy",
            "observed_domain": "unibuddy.com",
            "observed_linkedin_slug": "unibuddy",
        },
        buyer_max_age_days=365,
    )
    return result, judge


@pytest.mark.asyncio
async def test_retained_high_false_negative_gets_one_neutral_review(
    monkeypatch,
) -> None:
    supported = _answer(
        status="supported",
        overall="qualified",
        supporting_quotes=[FOCUS_QUOTE, DUTY_QUOTE],
        unsupported_parts=[],
    )
    result, judge = await _verify(monkeypatch, _answer(), supported)

    assert judge.await_count == 2
    clarification_prompt = judge.await_args_list[1].args[2]
    assert judge.await_args_list[1].kwargs["max_attempts"] == 1
    assert "ONE BOUNDED HIRING SEMANTIC CLARIFICATION" in clarification_prompt
    assert "untrusted model conclusion" in clarification_prompt
    assert "non-exhaustive locators" in clarification_prompt
    assert DUTY_QUOTE in clarification_prompt
    assert TARGET in clarification_prompt
    assert result["decision"] == "approve"
    assert result["evidence_clarification"] == {
        "attempted": True,
        "resolved": True,
        "provider_error": False,
    }


@pytest.mark.asyncio
async def test_true_hiring_negative_remains_rejected_after_review(monkeypatch) -> None:
    url = "https://careers.acme.example/jobs/business-analyst"
    claim = "Acme is hiring a Business Analyst who will gain platform exposure."
    no_duty_quote = (
        "This role has no responsibility to build, operate, own, maintain, or "
        "improve the platform or its components."
    )
    source_text = (
        "About the role. Acme is hiring a Business Analyst who will gain platform "
        "exposure. Responsibilities include weekly reports and meetings. "
        + no_duty_quote
    )
    negative = _answer(
        url=url,
        claim=claim,
        supporting_quotes=[
            "Acme is hiring a Business Analyst who will gain platform exposure."
        ],
    )
    result, judge = await _verify(
        monkeypatch, negative, deepcopy(negative),
        source_text=source_text, url=url, claim=claim,
    )

    assert judge.await_count == 2
    assert result["decision"] == "reject"
    assert result["client_ready"] is False
    assert result["stage3"]["status"] == "contradicted"


@pytest.mark.asyncio
async def test_hiring_semantic_review_never_opens_a_third_call(monkeypatch) -> None:
    unresolved = _answer(
        status="supported",
        confidence="medium",
        overall="qualified",
        overall_confidence="medium",
        unsupported_parts=[],
    )
    result, judge = await _verify(monkeypatch, _answer(), unresolved)

    assert judge.await_count == 2
    assert result["decision"] == "review"
    assert result["client_ready"] is False
    assert result["rejection_reason"] == "stage3_review"
