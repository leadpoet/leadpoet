"""Amounts and counts are not HTTP absence; all source checks still run."""
import asyncio
from unittest.mock import AsyncMock
import pytest
from qualification.scoring import lead_scorer, intent_verification_three_stage as verifier
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_arena_score_integrity import _company_model, _icp_model

@pytest.mark.parametrize("description,accepted", [
    ("Acme raised $404 million in a Series C financing round.", True),
    ("Acme raised $404M in a Series C financing round.", True),
    ("Acme announced 404 new engineering positions.", True),
    ("Acme funding announcement returned HTTP 404.", False),
    ("Acme funding announcement returned error 404.", False),
    ("Acme funding announcement returned 404 Not Found.", False),
    ("Acme funding announcement: page not found.", False),
])
def test_source_status_requires_error_context(monkeypatch, description, accepted):
    signal = _company_model([{"source":"news", "description":description,
        "url":"https://news.example/acme-funding", "date":"2026-10-01",
        "snippet":description, "matched_icp_signal":0}]).intent_signals[0]
    icp = _icp_model().model_copy(update={"intent_signals":["Raised funding in the last 12 months"]})
    # Hold the source-backed semantic verdict constant to isolate the guard.
    review = AsyncMock(return_value={"decision":"approve", "client_ready":True,
        "stage1":{"status":"skipped_source_grounded"},
        "stage3":{"status":"supported", "confidence":"high", "claim_matches_miner_date":"consistent"},
        "verdict":{"signal_evaluations":[{"signal_status":"supported", "same_entity_check":"pass",
            "verification_mode":"source_grounded", "risk_notes":["source_event_date:2026-10-01"]}]},
        "source_publication_dates":["2026-10-01"], "scrape":{"result_count":1}})
    monkeypatch.setattr(verifier, "verify_three_stage", review)
    verdicts=[]
    with use_evaluation_date("2026-10-29"):
        score=asyncio.run(lead_scorer._score_single_intent_signal(signal,icp,None,"Acme",
            "https://acme.com",trust_signal_date=True,stage1_soft_reject=True,
            llm_only_intent_gate=True,integrity_policy=True,verdict_out=verdicts))
    assert review.await_count == 1
    assert (score[0]>0) is accepted
    assert verdicts[-1]["decision"] == ("verified" if accepted else "rejected_pregate_after_fetch")
