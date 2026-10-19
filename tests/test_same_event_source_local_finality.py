"""Existing bounded company finality handles mixed source-local recovery."""
from __future__ import annotations

import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import IntentSignal
from lab_arena import scoring
from qualification.scoring import competition, intent_verification_three_stage as verifier
from qualification.scoring import lead_scorer
from tests.lab_arena.test_lab_arena_scoring import _ICPS, breakdown, scored_company
from tests.test_intent_source_context_transition import _icp
from tests.test_intent_unavailable_same_event_source import (
    ALTERNATIVE, CLAIM, ORIGINAL, fetched_source, judgment, missing_source, run_verifier,
)


def confirmed_alternative_absence():
    return {"results": [], "statuses": [{
        "url": ALTERNATIVE, "source": "none", "sd_stage": "genuine_404",
        "sd_error": "http_404", "exa_stage": "exa_target_not_found",
        "exa_error": "target_not_found", "exa_target_absence": {
            "id_matches_requested_url": True, "status": "error",
            "error_tag": "CRAWL_NOT_FOUND", "error_http_status": 404,
            "confirmed_attempts": 2,
        },
    }]}


def unavailable_alternative(stage="exa_no_results", *, sd_stage="all_tiers_exhausted:http_422"):
    alternative = missing_source()
    alternative["statuses"][0].update(url=ALTERNATIVE, sd_stage=sd_stage, exa_stage=stage)
    return alternative


def _propagate_to_breakdown(monkeypatch, result):
    """Use the actual score-single-signal mapping, then existing company retry."""
    monkeypatch.setattr(verifier, "verify_three_stage", AsyncMock(return_value=result))
    monkeypatch.setenv("LEADPOET_COMPETITION_EVALUATION_DATE", "2026-10-19")
    signal = IntentSignal(source="social_media", description=CLAIM, url=ORIGINAL,
                          date="2026-05-20", snippet=CLAIM, matched_icp_signal=0)
    icp = _icp().model_copy(update={
        "intent_signals": ["Publicly named as a partner at a technology event"],
        "intent_signal_evidence_types": ["PARTNERSHIP"],
    })
    verdicts = []
    asyncio.run(lead_scorer._score_single_intent_signal(
        signal, icp, None, "PixVerse", "https://pixverse.ai/",
        stage1_soft_reject=True, llm_only_intent_gate=True,
        integrity_policy=True, verdict_out=verdicts,
    ))
    assert len(verdicts) == 1
    row = breakdown(0.0, "Intent verification unavailable: evidence_fetch_failed"
                    if result["decision"] == "unavailable" else "Required intent not proven")
    row.update(company_index=0, company_qualified=False, duplicate_company=False,
               company_identity_key="domain:scored0.example.com",
               company_identity_alias_keys=["domain:scored0.example.com"])
    row["intent_signals_detail"] = [{
        "matched_icp_signal": 0, "after_decay": 0.0, "judge_verdict": verdicts[0],
    }]
    return row


def _score_to_finality(row):
    calls = []

    def scorer(*_args):
        calls.append(1)
        return [copy.deepcopy(row)]

    scorer.integrity_policy = True
    result = scoring.score_work_item(
        {"scored_run_id": "same-event-finality-control"},
        icp=_ICPS[0], companies=[scored_company(0)], scorer=scorer,
    )
    return result[0], len(calls)


@pytest.mark.parametrize("case", ["no_locator", "empty_alternative", "confirmed_absent_alternative"])
def test_healthy_same_event_exhaustion_reaches_existing_terminal_company_zero(monkeypatch, case):
    kwargs = {}
    if case == "no_locator":
        kwargs["search"] = {"status": "unproven", "reason": "no_same_event_locator"}
    else:
        kwargs["alternative"] = (confirmed_alternative_absence() if case == "confirmed_absent_alternative"
                                 else unavailable_alternative())
    result, *_ = asyncio.run(run_verifier(monkeypatch, **kwargs))
    assert result["decision"] == "unavailable"
    assert result["rejection_reason"] == "evidence_fetch_failed"
    assert not verifier._confirmed_source_absence(missing_source(), ORIGINAL)
    row = _propagate_to_breakdown(monkeypatch, result)
    assert row["intent_signals_detail"][0]["judge_verdict"]["decision"] == "rejected_verifier_error"
    assert competition.scorer_breakdown_has_company_local_verification_failure(row)
    terminal, calls = _score_to_finality(row)
    assert calls == scoring.MAX_JUDGE_RETRIES == 3
    assert terminal["final_score"] == 0.0
    assert competition.scorer_breakdown_is_terminal_company_verification_failure(terminal)
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(terminal, integrity_policy=True)


@pytest.mark.parametrize("stage,error", [
    ("exa_http_error", "HTTP 401"),
    ("exa_transient_exhausted", "HTTP 429"),
    ("exa_transient_exhausted", "ReadTimeout"),
    ("exa_failed", "provider transport failure"),
])
def test_actual_alternative_provider_errors_never_become_terminal_zero(monkeypatch, stage, error):
    # A genuine local SD anti-bot result cannot hide Exa authentication,
    # rate-limit, timeout or other provider failure on that same source.
    alternative = unavailable_alternative(stage, sd_stage="all_tiers_exhausted:anti_bot_marker")
    alternative["statuses"][0]["exa_error"] = error
    result, *_ = asyncio.run(run_verifier(monkeypatch, alternative=alternative))
    row = _propagate_to_breakdown(monkeypatch, result)
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)
    with pytest.raises(scoring.ScoringError, match="judge_exhausted attempts=3"):
        _score_to_finality(row)


@pytest.mark.parametrize("mutation", [
    {"confirmed_attempts": 1}, {"confirmed_attempts": True},
    {"id_matches_requested_url": False}, {"error_http_status": 401},
    {"error_tag": "CRAWL_UNKNOWN_ERROR"}, {"extra": "untrusted"},
])
def test_absence_requires_the_complete_exact_two_attempt_receipt(monkeypatch, mutation):
    alternative = confirmed_alternative_absence()
    alternative["statuses"][0]["exa_target_absence"].update(mutation)
    result, *_ = asyncio.run(run_verifier(monkeypatch, alternative=alternative))
    row = _propagate_to_breakdown(monkeypatch, result)
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    with pytest.raises(scoring.ScoringError):
        _score_to_finality(row)


def test_healthy_fetched_alternative_semantic_negative_is_terminal_first_attempt(monkeypatch):
    result, *_ = asyncio.run(run_verifier(
        monkeypatch, alternative=fetched_source(),
        response=judgment(status="unable_to_verify", notes=[]),
    ))
    assert result["decision"] in {"reject", "review"}
    row = _propagate_to_breakdown(monkeypatch, result)
    assert row["intent_signals_detail"][0]["judge_verdict"]["decision"] == "rejected_three_stage"
    terminal, calls = _score_to_finality(row)
    assert calls == 1 and terminal["final_score"] == 0.0
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(terminal, integrity_policy=True)


def test_confirmed_absence_cannot_hide_an_additional_provider_failure(monkeypatch):
    alternative = confirmed_alternative_absence()
    result, *_ = asyncio.run(run_verifier(monkeypatch, alternative=alternative))
    result["scrape"]["statuses"].append({
        "url": "https://third-source.example", "source": "none",
        "sd_stage": "all_tiers_exhausted:anti_bot_marker", "exa_stage": "exa_http_error",
    })
    row = _propagate_to_breakdown(monkeypatch, result)
    assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
    with pytest.raises(scoring.ScoringError):
        _score_to_finality(row)
