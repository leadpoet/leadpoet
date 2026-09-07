"""Focused regressions for Arena intent and fit failure classification."""

from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match,
    company_fit_mismatch,
    company_fit_unavailable,
)
from qualification.scoring.competition import (
    count_penalizable_false_positives,
    scorer_breakdown_has_retryable_infrastructure_failure,
)


def _detail(
    *,
    decision: str = "rejected_three_stage",
    rejection_reason: str = "stage3_unable_to_verify",
    status: str | None = "unable_to_verify",
    same_entity: str | None = "unclear",
    matched: int = 0,
) -> dict:
    evaluation = {
        "signal_status": status,
        "same_entity_check": same_entity,
        "supporting_quotes": ["The source text was retained in the receipt."],
    }
    return {
        "raw": 0.0,
        "after_decay": 0.0,
        "matched_icp_signal": matched,
        "judge_verdict": {
            "decision": decision,
            "rejection_reason": rejection_reason,
            "pipeline_decision": "reject",
            "verification_trace": {
                "intent_verdict": {"signal_evaluations": [evaluation]},
            },
        },
    }


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            [_detail(status="wrong_entity", same_entity="unclear")],
            "Primary intent evidence unverified: verifier could not confirm "
            "the source-company identity",
        ),
        (
            [_detail(status="wrong_entity", same_entity="pass")],
            "Primary intent evidence unverified: verifier could not confirm "
            "the source-company identity",
        ),
        (
            [_detail(status="wrong_entity", same_entity="fail")],
            "Primary intent evidence mismatch: source is about a different "
            "company",
        ),
        (
            [_detail(status="contradicted", same_entity="pass")],
            "Primary intent evidence mismatch: source contradicts the claim",
        ),
        (
            [
                _detail(
                    decision="rejected_freshness",
                    rejection_reason="signal_out_of_window",
                    status="contradicted",
                    same_entity="pass",
                )
            ],
            "Primary intent evidence unverified: evidence is outside the "
            "allowed freshness window",
        ),
        (
            [
                _detail(
                    decision="rejected_pregate",
                    rejection_reason="duplicate_evidence_domain",
                    status=None,
                    same_entity=None,
                )
            ],
            "Primary intent evidence unverified: duplicate evidence domain",
        ),
        (
            [_detail()],
            "Primary intent evidence unverified: verifier did not confirm the "
            "submitted claim",
        ),
        (
            [
                _detail(),
                _detail(
                    status="wrong_entity", same_entity="fail", matched=1
                ),
            ],
            "Primary intent evidence unverified: verifier did not confirm the "
            "submitted claim",
        ),
        (
            [_detail(status="wrong_entity", same_entity="fail", matched=1)],
            "Primary intent evidence unverified: no submitted evidence targeted "
            "the primary intent",
        ),
    ],
)
def test_all_zero_intent_summary_uses_grounded_verdict(details, expected):
    assert lead_scorer._competition_intent_failure_reason(details) == expected
    assert "fabricat" not in expected.casefold()


def _company() -> CompanyOutput:
    return CompanyOutput.model_validate(
        {
            "company_name": "Strand Therapeutics",
            "company_website": "https://strandtx.com",
            "company_linkedin": "https://linkedin.com/company/strandtx",
            "industry": "Biotechnology",
            "employee_count": "51-200",
            "company_stage": "Series B",
            "country": "United States",
            "description": "Programmable mRNA therapeutics.",
            "intent_signals": [
                {
                    "source": "news",
                    "description": "Closed a $153M Series B.",
                    "url": "https://news.example.com/strand-series-b",
                    "date": "2025-10-22",
                    "snippet": "Strand Therapeutics closed a Series B.",
                    "matched_icp_signal": 0,
                }
            ],
        }
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="funding",
        prompt="Biotechnology companies with recent funding",
        industry="Biotechnology",
        sub_industry="Therapeutics",
        employee_count="51-200",
        company_stage="Series B",
        geography="United States",
        country="United States",
        product_service="Drug development",
        intent_signals=["Announced a Series B or later funding round"],
    )


def test_arena_scorer_does_not_turn_ambiguous_identity_into_fabrication(
    monkeypatch,
):
    details = [_detail(status="wrong_entity", same_entity="unclear")]

    async def fit(*_args, **_kwargs):
        return company_fit_match("fit verified")

    async def score(*_args, **_kwargs):
        return 0.0, 0.0, 0.0, 0, True, details

    async def no_repair(*_args, **_kwargs):
        return None

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent_signal", score
    )
    monkeypatch.setattr(
        lead_scorer, "_attempt_competition_evidence_repair", no_repair
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            _company(), _icp(), 0.0, 0.0, set()
        )
    )

    assert result.failure_reason == (
        "Primary intent evidence unverified: verifier could not confirm the "
        "source-company identity"
    )
    assert result.intent_signals_detail == details


@pytest.mark.parametrize(
    "detail",
    [
        _detail(),
        _detail(status="wrong_entity", same_entity="unclear"),
        _detail(status="wrong_entity", same_entity="fail"),
        _detail(status="contradicted", same_entity="pass"),
        _detail(
            decision="rejected_freshness",
            rejection_reason="signal_out_of_window",
            status="verified",
            same_entity="pass",
        ),
        _detail(
            decision="rejected_pregate",
            rejection_reason="duplicate_evidence_domain",
            status=None,
            same_entity=None,
        ),
    ],
)
def test_failed_primary_keeps_one_existing_penalty_and_no_gate_penalty(detail):
    breakdown = {
        "final_score": 0.0,
        "failure_reason": lead_scorer._competition_intent_failure_reason(
            [detail]
        ),
        "intent_signals_detail": [detail],
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    ) == (0, 1)


def test_fit_unavailable_remains_retryable_but_is_not_a_mismatch_penalty():
    missing = {
        "final_score": 0.0,
        "failure_reason": "Company fit unavailable: identity not proven",
        "verifier_gate_receipts": [
            company_fit_unavailable("identity not proven").receipt("company_fit")
        ],
    }
    outage = {
        "final_score": 0.0,
        "failure_reason": "Company fit unavailable: provider HTTP 503",
        "verifier_gate_receipts": [
            company_fit_unavailable("provider HTTP 503").receipt("company_fit")
        ],
    }
    mismatch = {
        "final_score": 0.0,
        "failure_reason": "Company fit failed: submitted identity conflicts",
        "verifier_gate_receipts": [
            company_fit_mismatch("identity conflict").receipt("company_fit")
        ],
    }

    assert scorer_breakdown_has_retryable_infrastructure_failure(missing)
    assert scorer_breakdown_has_retryable_infrastructure_failure(outage)
    assert count_penalizable_false_positives(
        [missing], icp_has_intent_signals=True
    ) == (0, 0)
    assert count_penalizable_false_positives(
        [mismatch], icp_has_intent_signals=True
    ) == (1, 0)


def test_intent_provider_outage_is_retryable_and_never_penalized():
    detail = _detail()
    detail["judge_verdict"] = {
        "decision": "rejected_verifier_error",
        "pipeline_decision": "unavailable",
        "error_class": "ProviderTimeout",
    }
    breakdown = {
        "final_score": 0.0,
        "failure_reason": "Intent verification unavailable: verifier provider error",
        "intent_signals_detail": [detail],
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)
    assert count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    ) == (0, 0)
