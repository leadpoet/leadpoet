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
    competition_score_from_breakdowns,
    count_penalizable_false_positives,
    has_verified_primary_intent,
    intent_unavailability_requires_retry,
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


def _verified_detail(*, matched: int = 0, after_decay: float = 60.0) -> dict:
    return {
        "raw": after_decay,
        "after_decay": after_decay,
        "matched_icp_signal": matched,
        "judge_verdict": {
            "decision": "verified",
            "pipeline_decision": "accept",
            "client_ready": True,
            "verification_trace": {
                "intent_verdict": {
                    "signal_evaluations": [
                        {
                            "signal_status": "supported",
                            "same_entity_check": "pass",
                        }
                    ]
                }
            },
        },
    }


def _unavailable_detail(*, matched: int = 0) -> dict:
    detail = _detail(matched=matched)
    detail["judge_verdict"] = {
        "decision": "rejected_verifier_error",
        "pipeline_decision": "unavailable",
        "error_class": "ProviderTimeout",
        "verification_trace": {
            "intent_verdict": {
                "signal_evaluations": [
                    {
                        "signal_status": "unable_to_verify",
                        "same_entity_check": "unclear",
                    }
                ]
            }
        },
    }
    return detail


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
            "Primary intent evidence mismatch: source does not establish the "
            "required intent",
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


def test_category_mismatch_keeps_1password_verdict_and_penalty_arithmetic():
    detail = _detail(
        status="contradicted",
        same_entity="pass",
        rejection_reason="stage3_intent_category_mismatch",
    )
    evaluation = detail["judge_verdict"]["verification_trace"][
        "intent_verdict"
    ]["signal_evaluations"][0]
    evaluation.update(
        {
            "claim_status": "supported",
            "required_intent": "formal regulatory or compliance approval",
            "observed_event": "AWS Security Competency distinction",
        }
    )
    failure_reason = lead_scorer._competition_intent_failure_reason([detail])
    rejected = {
        "final_score": 0.0,
        "failure_reason": failure_reason,
        "intent_signals_detail": [detail],
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert failure_reason == (
        "Primary intent evidence mismatch: source does not establish the "
        "required intent"
    )
    assert evaluation["claim_status"] == "supported"
    assert count_penalizable_false_positives(
        [rejected], icp_has_intent_signals=True
    ) == (0, 1)
    assert competition_score_from_breakdowns(
        {"max_companies": 5, "intent_signal": "required"},
        [{"final_score": 54.0}, rejected],
    ) == {
        "per_icp_score": 8.8,
        "fp_gate_count": 0,
        "fp_unverified_primary_count": 1,
        "company_goal": 5,
        "company_scores": [54.0, 0.0],
    }


def test_alex_bank_fit_summary_uses_final_dimension_decisions(monkeypatch):
    company = CompanyOutput.model_validate(
        {
            "company_name": "Alex Bank",
            "company_website": "https://alex.bank",
            "company_linkedin": "https://linkedin.com/company/withalex",
            "industry": "Financial Services",
            "employee_count": "51-200",
            "company_stage": "Series C",
            "country": "Australia",
            "intent_signals": [
                {
                    "source": "news",
                    "description": "Announced a qualifying partnership.",
                    "url": "https://news.example.com/alex-bank-partnership",
                    "date": "2026-08-01",
                    "snippet": "Alex Bank announced a partnership.",
                    "matched_icp_signal": 0,
                }
            ],
        }
    )
    icp = ICPPrompt(
        icp_id="australian-digital-lending",
        prompt="Australian digital lenders",
        industry="Financial Services",
        sub_industry="Digital lending",
        employee_count="51-200",
        company_stage="Series C",
        geography="Australia",
        country="Australia",
        product_service="Digital lending",
    )

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable("homepage identity binding unavailable")

    async def web(*_args, **_kwargs):
        return company_fit_mismatch(
            "Public web pages identify Alex Bank as an Australian digital "
            "bank with Brisbane headquarters, 51-200 employees, Series C",
            details={
                "identity_decision": "mismatch",
                "identity_receipt": {
                    "decision": "mismatch",
                    "submitted_name": "Alex Bank",
                    "submitted_domain": "alex.bank",
                    "submitted_linkedin_slug": "withalex",
                    "observed_name": "Alex Bank",
                    "observed_domain": "alex.bank",
                    "observed_linkedin_slug": "alexbankaus",
                    "evidence_source": "company_web_reverification",
                },
                "dimension_decisions": {
                    "employee_size": "unavailable",
                    "industry": "match",
                    "geography": "match",
                    "stage": "match",
                },
                "required_attribute_decision": "unavailable",
                "dimension_evidence": {
                    dimension: {
                        "url": f"https://alex.bank/evidence/{dimension}",
                        "quote": f"Verified {dimension}",
                    }
                    for dimension in ("industry", "geography", "stage")
                },
                "provider_observations": {
                    "observed_employee_count": None,
                    "observed_linkedin_slug": "alexbankaus",
                },
            },
        )

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)

    fit = asyncio.run(
        lead_scorer._verify_company_fit(
            company,
            icp,
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    breakdown = {
        "final_score": 0.0,
        "failure_reason": lead_scorer._company_fit_failure_reason(
            "Company fit", fit
        ),
        "verifier_gate_receipts": [fit.receipt("company_fit")],
    }

    assert fit.decision == "mismatch"
    assert fit.reason == (
        "company fit mismatch: identity; unproven dimensions: employee_size"
    )
    assert "51-200 employees" not in fit.reason
    assert "required_attribute" not in fit.reason
    assert fit.details["company_fit_dimensions"] == {
        "identity": "mismatch",
        "employee_size": "unavailable",
        "industry": "match",
        "geography": "match",
        "stage": "match",
    }
    assert fit.details["dimension_evidence"]["employee_size"][
        "web_evidence"
    ] == {}
    assert competition_score_from_breakdowns(
        {"max_companies": 5, "intent_signal": "required"},
        [{"final_score": 54.0}, breakdown],
    ) == {
        "per_icp_score": 8.8,
        "fp_gate_count": 1,
        "fp_unverified_primary_count": 0,
        "company_goal": 5,
        "company_scores": [54.0, 0.0],
    }


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


def test_mixed_verified_primary_and_unavailable_signal_keeps_score(monkeypatch):
    details = [_verified_detail(), _unavailable_detail(matched=1)]

    async def fit(*_args, **_kwargs):
        return company_fit_match("fit verified")

    async def score(*_args, **_kwargs):
        return 60.0, 60.0, 1.0, 90, False, details

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent_signal", score
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            _company(), _icp(), 0.0, 0.0, set()
        )
    )
    breakdown = result.model_dump(mode="json")

    assert result.final_score == result.intent_signal_final == 60.0
    assert result.failure_reason is None
    assert result.intent_signals_detail == details
    assert not intent_unavailability_requires_retry(details)
    assert not scorer_breakdown_has_retryable_infrastructure_failure(breakdown)
    assert count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    ) == (0, 0)


def test_scorer_keeps_retryable_zero_without_verified_primary(monkeypatch):
    details = [_unavailable_detail(), _verified_detail(matched=1)]

    async def fit(*_args, **_kwargs):
        return company_fit_match("fit verified")

    async def score(*_args, **_kwargs):
        return 25.0, 25.0, 1.0, 90, False, details

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(
        lead_scorer, "score_company_competition_intent_signal", score
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            _company(), _icp(), 0.0, 0.0, set()
        )
    )
    breakdown = result.model_dump(mode="json")

    assert result.final_score == 0.0
    assert result.failure_reason == (
        "Intent verification unavailable: verifier provider error"
    )
    assert result.intent_signals_detail == details
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


@pytest.mark.parametrize(
    "details",
    [
        [_unavailable_detail(), _verified_detail(matched=1)],
        [_unavailable_detail(), _unavailable_detail(matched=1)],
    ],
)
def test_unavailable_intent_without_verified_primary_remains_retryable(details):
    assert not has_verified_primary_intent(details)
    assert intent_unavailability_requires_retry(details)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("after_decay", float("nan")),
        ("after_decay", float("inf")),
        ("after_decay", "60"),
        ("after_decay", True),
        ("matched_icp_signal", "0"),
    ],
)
def test_malformed_positive_primary_cannot_suppress_retry(field, value):
    primary = _verified_detail()
    primary[field] = value
    details = [primary, _unavailable_detail(matched=1)]

    assert not has_verified_primary_intent(details)
    assert intent_unavailability_requires_retry(details)


def test_contradicted_positive_primary_cannot_suppress_retry():
    primary = _verified_detail()
    primary["judge_verdict"]["verification_trace"]["intent_verdict"][
        "signal_evaluations"
    ][0]["signal_status"] = "contradicted"
    details = [primary, _unavailable_detail(matched=1)]

    assert not has_verified_primary_intent(details)
    assert intent_unavailability_requires_retry(details)


def test_fit_unavailable_still_retries_with_positive_primary():
    details = [_verified_detail(), _unavailable_detail(matched=1)]
    breakdown = {
        "final_score": 60.0,
        "failure_reason": None,
        "intent_signals_detail": details,
        "verifier_gate_receipts": [
            company_fit_unavailable("provider HTTP 503").receipt("company_fit")
        ],
    }

    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_historical_unavailable_zero_stays_retryable_with_positive_detail():
    breakdown = {
        "final_score": 0.0,
        "failure_reason": "Intent verification unavailable: verifier provider error",
        "intent_signals_detail": [_verified_detail(), _unavailable_detail(matched=1)],
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_all_verified_details_keep_existing_nonretryable_result():
    details = [_verified_detail(), _verified_detail(matched=1, after_decay=25.0)]
    breakdown = {
        "final_score": 85.0,
        "failure_reason": None,
        "intent_signals_detail": details,
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert has_verified_primary_intent(details)
    assert not intent_unavailability_requires_retry(details)
    assert not scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_legacy_positive_primary_retains_prior_penalty_accounting():
    breakdown = {
        "final_score": 45.0,
        "failure_reason": None,
        "intent_signals_detail": [
            {
                "raw": 45.0,
                "after_decay": 45.0,
                "matched_icp_signal": "0",
            }
        ],
        "verifier_gate_receipts": [company_fit_match().receipt("company_fit")],
    }

    assert count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    ) == (0, 0)
