"""Regressions for the final intent judge's item/overall consistency."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from lab_arena import scoring as arena_scoring
from qualification.scoring import intent_verification_three_stage as three_stage
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match,
    evaluate_company_identity,
)
from qualification.scoring.competition import CompetitionCompanyScorer
from qualification.scoring.intent_verification_three_stage import _decision


def _verdict(
    *,
    overall: str,
    status: str = "supported",
    date_match: str = "no_date_in_content",
    extra_evaluations: list[dict] | None = None,
) -> dict:
    evaluations = [{
        "signal_id": "signal-1",
        "signal_status": status,
        "confidence": "high",
        "same_entity_check": "pass",
        "claim_matches_miner_date": date_match,
        "supporting_quotes": ["Acme launched its new retail channel."],
        "contradicting_quotes": [],
        "unsupported_parts": [],
    }]
    evaluations.extend(extra_evaluations or [])
    return {
        "overall_verdict": overall,
        "overall_confidence": "high",
        "signal_evaluations": evaluations,
    }


def test_company_quality_does_not_approve_just_ice_tea_inconsistent_receipt():
    verdict = _verdict(
        overall="needs_review",
        date_match="contradicted",
    )
    verdict["signal_evaluations"][0].update({
        "claim": (
            "Just Ice Tea doubled its store presence through 2025 retail "
            "rollouts into Target, CVS Pharmacy, Wegmans, and Harris Teeter."
        ),
        "supporting_quotes": [
            "In 2025, the brand doubled its store presence with rollouts "
            "into Target, CVS Pharmacy, Wegmans and Harris Teeter."
        ],
        "risk_notes": [
            "The source establishes the event occurred in 2025 but does not "
            "give a month or day; recency within the 12-month target window "
            "is unresolved."
        ],
    })

    assert _decision(verdict, company_quality=True) == "review"


def test_company_quality_keeps_qualified_no_date_acceptance():
    assert _decision(
        _verdict(overall="qualified", date_match="no_date_in_content"),
        company_quality=True,
    ) == "approve"


def test_company_quality_keeps_qualified_requested_alternative_acceptance():
    optional_unproven = {
        "signal_id": "signal-2",
        "signal_status": "unable_to_verify",
        "confidence": "low",
        "same_entity_check": "unclear",
        "claim_matches_miner_date": "no_date_in_content",
        "supporting_quotes": [],
        "contradicting_quotes": [],
        "unsupported_parts": ["The optional second alternative is unproven."],
    }

    assert _decision(
        _verdict(
            overall="qualified",
            extra_evaluations=[optional_unproven],
        ),
        company_quality=True,
    ) == "approve"


def test_company_quality_does_not_override_overall_disqualification():
    assert _decision(
        _verdict(overall="disqualified"), company_quality=True
    ) == "review"


def test_legacy_item_level_decision_is_unchanged():
    assert _decision(_verdict(overall="needs_review")) == "approve"


def test_stage3_semantic_review_is_terminal_zero_in_score_work_item():
    calls = 0
    semantic_zero = {
        "final_score": 0.0,
        "failure_reason": (
            "Primary intent evidence unverified: verifier did not confirm "
            "the submitted claim"
        ),
        "intent_signals_detail": [{
            "raw": 0.0,
            "after_decay": 0.0,
            "matched_icp_signal": 0,
            "judge_verdict": {
                "decision": "rejected_three_stage",
                "pipeline_decision": "review",
                "rejection_reason": "stage3_review",
                "client_ready": False,
                "verification_trace": {
                    "intent_verdict": _verdict(overall="needs_review"),
                },
            },
        }],
        "verifier_gate_receipts": [{
            "gate": "company_fit",
            "decision": "match",
        }],
    }

    def scorer(_companies, _icp, _is_reference_model):
        nonlocal calls
        calls += 1
        return [semantic_zero]

    result = arena_scoring.score_work_item(
        {"scored_run_id": "just-ice-tea-semantic-review"},
        icp={
            "max_companies": 1,
            "employee_count": ["51-200"],
            "intent_signals": ["Retail channel expansion in the last year"],
        },
        companies=[{
            "company_name": "Just Ice Tea",
            "company_website": "https://justicetea.com",
            "employee_count": "51-200",
        }],
        scorer=scorer,
        max_retries=3,
    )

    assert calls == 1
    assert result == [semantic_zero]
    assert all(
        receipt.get("failure_class") != "company_verification_exhausted"
        for receipt in result[0]["verifier_gate_receipts"]
    )


def test_just_ice_tea_review_flows_through_real_verifier_and_scorers_once(
    monkeypatch,
):
    source_url = (
        "https://www.refreshmentmag.com/news/just-ice-tea-raises-9m-"
        "series-b-to-support-retail-expansion-and-new-flavours"
    )
    source_quote = (
        "In 2025, the brand doubled its store presence with rollouts into "
        "Target, CVS Pharmacy, Wegmans and Harris Teeter."
    )
    source_text = (
        "Just Ice Tea raises $9m Series B to support retail expansion and "
        "new flavours. " + source_quote + " Rafaela Sousa 19 February 2026."
    )
    provider_answer = {
        "answer": {
            "overall_verdict": "needs_review",
            "overall_confidence": "high",
            "summary": (
                "The rollout is supported, but its 12-month recency cannot "
                "be confirmed."
            ),
            "missing_or_risks": [
                "The source dates the rollouts only to 2025."
            ],
            "signal_evaluations": [{
                "signal_id": "signal-1",
                "claim": (
                    "Just Ice Tea doubled its store presence through 2025 "
                    "retail rollouts into Target, CVS Pharmacy, Wegmans, and "
                    "Harris Teeter."
                ),
                "verification_mode": "source_grounded",
                "signal_status": "supported",
                "source_urls_supplied": [source_url],
                "evidence_urls_used": [source_url],
                "source_accessibility": "Exact supplied source extraction.",
                "same_entity_check": "pass",
                "entity_match_reason": "The source identifies Just Ice Tea.",
                "supporting_quotes": [source_quote],
                "contradicting_quotes": [],
                "unsupported_parts": [],
                "source_quality": "Named industry publication.",
                "risk_notes": [
                    "The event occurred in 2025 but has no month or day."
                ],
                "confidence": "high",
                "claim_matches_miner_date": "contradicted",
                "author_type": "n/a",
                "author_employer_matches_lead": "n/a",
                "author_role_matches_spec": "n/a",
                "author_satisfies_role_spec": "n/a",
            }],
        },
        "model": "openai/gpt-6-luna",
        "usage": {},
    }
    provider = AsyncMock(return_value=provider_answer)
    fetch = AsyncMock(return_value={
        "results": [{
            "url": source_url,
            "title": "Just Ice Tea raises $9m Series B",
            "text": source_text,
            "source_publication_date": "",
        }],
        "statuses": [],
    })
    identity = evaluate_company_identity(
        submitted_name="Just Ice Tea",
        submitted_website="https://justicetea.com/",
        submitted_linkedin="https://linkedin.com/company/just-ice-tea",
        observed_name="Just Ice Tea",
        observed_website="https://justicetea.com/",
        observed_linkedin="https://linkedin.com/company/just-ice-tea",
        evidence_source="company_web_reverification",
        company_quality=True,
    )
    fit = company_fit_match(
        "All requested fit dimensions are independently verified.",
        details={
            "dimension_evidence": {
                "identity": {
                    "decision": "match",
                    "web_identity_receipt": identity,
                }
            }
        },
    )

    monkeypatch.setattr(three_stage, "_call_openrouter", provider)
    monkeypatch.setattr(three_stage, "_fetch_sd_then_exa", fetch)
    monkeypatch.setattr(lead_scorer, "_verify_company_fit", AsyncMock(return_value=fit))
    monkeypatch.setenv("LEADPOET_COMPETITION_EVALUATION_DATE", "2026-09-24")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    company = {
        "company_name": "Just Ice Tea",
        "company_website": "https://justicetea.com/",
        "company_linkedin": "https://linkedin.com/company/just-ice-tea",
        "industry": "Food and Beverage",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "country": "United States",
        "state": "Maryland",
        "fit_summary": "Just Ice Tea is a beverage brand.",
        "fit_evidence_urls": ["https://justicetea.com/"],
        "intent_signals": [{
            "description": (
                "Just Ice Tea doubled its store presence through 2025 retail "
                "rollouts into Target, CVS Pharmacy, Wegmans, and Harris Teeter."
            ),
            "url": source_url,
            "date": "2026-02-19",
            "why_now": "The submitted source claims a recent retail rollout.",
            "snippet": source_quote,
            "matched_icp_signal": 0,
        }],
    }
    icp = {
        "icp_id": "icp_20260923_003",
        "prompt": "Find online commerce businesses with recent expansion.",
        "industry": "Food and Beverage",
        "sub_industry": "Beverages",
        "employee_count": ["51-200"],
        "company_stage": "Series B",
        "geography": "United States",
        "country": "United States",
        "product_service": "An online commerce or retail brand.",
        "intent_signals": [
            "Launched a new commerce capability, storefront format, or retail "
            "channel expansion in the last 12 months."
        ],
        "intent_signal_evidence_types": ["PRODUCT_LAUNCH"],
        "intent_signal_max_age_days": [365],
        "intent_max_age_days": 365,
        "max_companies": 1,
    }
    adapter = CompetitionCompanyScorer(company_quality=True)
    calls = 0

    async def scorer(companies, frozen_icp, is_reference_model):
        nonlocal calls
        calls += 1
        return await adapter.score_with_breakdowns(
            companies, frozen_icp, is_reference_model
        )

    scorer.integrity_policy = True
    result = arena_scoring.score_work_item(
        {"scored_run_id": "just-ice-tea-real-stage3-review"},
        icp=icp,
        companies=[company],
        scorer=scorer,
        max_retries=3,
    )

    assert calls == provider.await_count == fetch.await_count == 1
    assert result[0]["final_score"] == 0.0
    assert result[0]["failure_reason"] == (
        "Primary intent evidence unverified: verifier did not confirm the "
        "submitted claim"
    )
    detail = result[0]["intent_signals_detail"][0]
    assert detail["judge_verdict"]["decision"] == "rejected_three_stage"
    assert detail["judge_verdict"]["pipeline_decision"] == "review"
    assert detail["judge_verdict"]["rejection_reason"] == "stage3_review"
    assert detail["judge_verdict"]["verification_trace"][
        "intent_verdict"
    ]["overall_verdict"] == "needs_review"
    assert all(
        receipt.get("failure_class") != "company_verification_exhausted"
        for receipt in result[0]["verifier_gate_receipts"]
    )
