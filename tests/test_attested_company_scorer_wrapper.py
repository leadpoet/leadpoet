from __future__ import annotations

import httpx
import pytest

from gateway.research_lab import attested_scoring
from research_lab.eval.evaluator import QualificationStyleCompanyScorer
from research_lab.eval.http_tape import validate_http_tape


async def _authoritative_breakdowns_with_one_provider_call():
    async def live_handler(_request):
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(live_handler)) as client:
        assert (await client.get("https://api.exa.ai/search?q=company")).status_code == 200
    return [{"final_score": 19.5, "failure_reason": None}]


@pytest.mark.asyncio
async def test_shadow_wrapper_returns_authoritative_bytes_and_sends_transient_tape(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=99,
        attested_purpose="research_lab.candidate_score.v1",
    )
    monkeypatch.setattr(
        scorer,
        "_score_with_breakdowns_impl",
        lambda *args, **kwargs: _authoritative_breakdowns_with_one_provider_call(),
    )
    monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", "shadow")
    captured = {}

    async def compare(**kwargs):
        captured.update(kwargs)
        return {"status": "matched"}

    monkeypatch.setattr(attested_scoring, "compare_qualification_company_scores", compare)
    companies = [{"company_name": "Example"}]
    icp = {"industry": "Software"}

    result = await scorer.score_with_breakdowns(companies, icp, False)

    assert result == [{"final_score": 19.5, "failure_reason": None}]
    assert captured["epoch_id"] == 99
    assert captured["purpose"] == "research_lab.candidate_score.v1"
    assert captured["companies"] == companies
    assert captured["icp"] == icp
    assert captured["expected_breakdowns"] == result
    tape = validate_http_tape(captured["provider_tape"])
    assert tape["entry_count"] == 1


@pytest.mark.asyncio
async def test_shadow_attestation_failure_cannot_change_authoritative_score(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=100,
        attested_purpose="research_lab.rebenchmark.v1",
    )
    expected = [{"final_score": 7.0}]

    async def authoritative(*_args, **_kwargs):
        return expected

    async def failed_compare(**_kwargs):
        raise RuntimeError("shadow unavailable")

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", authoritative)
    monkeypatch.setattr(
        attested_scoring,
        "compare_qualification_company_scores",
        failed_compare,
    )
    monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", "shadow")

    assert await scorer.score_with_breakdowns([], {}, True) is expected


@pytest.mark.asyncio
async def test_required_mode_is_release_gated_before_host_provider_call(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=101,
        attested_purpose="research_lab.candidate_score.v1",
    )

    async def forbidden_host_scoring(*_args, **_kwargs):
        raise AssertionError("required mode must not call host scoring")

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", forbidden_host_scoring)
    monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", "required")
    monkeypatch.delenv("RESEARCH_LAB_ATTESTED_SCORING_LIVE_PROVIDER_ENABLED", raising=False)

    with pytest.raises(
        attested_scoring.AttestedScoringError,
        match="not release-enabled",
    ):
        await scorer.score_with_breakdowns([], {}, False)


@pytest.mark.asyncio
async def test_required_mode_uses_enclave_as_sole_provider_and_scorer(monkeypatch):
    scorer = QualificationStyleCompanyScorer(
        attested_epoch_id=102,
        attested_purpose="research_lab.rebenchmark.v1",
    )

    async def forbidden_host_scoring(*_args, **_kwargs):
        raise AssertionError("required mode must not call host scoring")

    captured = {}

    async def execute(**kwargs):
        captured.update(kwargs)
        return {
            "status": "succeeded",
            "receipt": {
                "evidence_roots": {
                    "provider_http_tape": "sha256:" + "a" * 64,
                }
            },
            "result": {
                "breakdowns": [{"final_score": 23.0}],
                "scores": [23.0],
            },
        }

    monkeypatch.setattr(scorer, "_score_with_breakdowns_impl", forbidden_host_scoring)
    monkeypatch.setattr(attested_scoring, "execute_attested_scoring_operation", execute)
    monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_MODE", "required")
    monkeypatch.setenv("RESEARCH_LAB_ATTESTED_SCORING_LIVE_PROVIDER_ENABLED", "true")

    result = await scorer.score_with_breakdowns(
        [{"company_name": "Example"}],
        {"industry": "Software"},
        False,
    )

    assert result == [{"final_score": 23.0}]
    assert captured["payload"]["provider_execution_mode"] == "live_enclave"
    assert "provider_tape" not in captured["payload"]
