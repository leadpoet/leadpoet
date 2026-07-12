from __future__ import annotations

import pytest

from gateway.research_lab import attested_scoring
from research_lab.eval.evaluator import QualificationStyleCompanyScorer


@pytest.mark.asyncio
async def test_non_authoritative_utility_scorer_keeps_existing_local_behavior(monkeypatch):
    scorer = QualificationStyleCompanyScorer()
    expected = [{"final_score": 3.0}]

    async def local(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", local)
    assert await scorer.score_with_breakdowns([], {}, False) is expected


@pytest.mark.asyncio
async def test_research_lab_scorer_uses_v2_as_sole_provider_and_scorer(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=102,
        attested_purpose="research_lab.rebenchmark.v1",
        attested_provider_profile="benchmark_scorer",
    )
    captured = {}

    async def forbidden_host_scoring(*_args, **_kwargs):
        raise AssertionError("authoritative scoring must not call the host scorer")

    async def execute(**kwargs):
        captured.update(kwargs)
        kwargs["attestation_out"].update(
            {"receipt": {"receipt_hash": "sha256:" + "a" * 64}}
        )
        return [{"final_score": 23.0}]

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", forbidden_host_scoring)
    monkeypatch.setattr(
        attested_scoring,
        "execute_required_qualification_company_scores",
        execute,
    )

    result = await scorer.score_with_breakdowns(
        [{"company_name": "Example"}],
        {"industry": "Software"},
        False,
    )

    assert result == [{"final_score": 23.0}]
    assert captured["epoch_id"] == 102
    assert captured["purpose"] == "research_lab.rebenchmark.v1"
    assert captured["provider_credential_profile"] == "benchmark_scorer"
    assert scorer.attested_receipts()[0]["receipt_hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_v2_scoring_failure_propagates_without_host_fallback(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=103,
        attested_purpose="research_lab.candidate_score.v1",
    )

    async def forbidden_host_scoring(*_args, **_kwargs):
        raise AssertionError("host fallback was attempted")

    async def failed(**_kwargs):
        raise attested_scoring.AttestedScoringError("V2 unavailable")

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", forbidden_host_scoring)
    monkeypatch.setattr(
        attested_scoring,
        "execute_required_qualification_company_scores",
        failed,
    )
    with pytest.raises(attested_scoring.AttestedScoringError, match="V2 unavailable"):
        await scorer.score_with_breakdowns([], {}, False)
