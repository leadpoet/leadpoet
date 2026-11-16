"""Missing event dates must not become unsupported content in integrity rounds."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier


CASES = json.loads((Path(__file__).parent / "fixtures/intent_whatnot_missing_date.json").read_text())
URL = "https://blog.teamwhatnot.com/unitedstates/shopify-integration"


def _inputs():
    case = copy.deepcopy(CASES[1])
    row = {"id": "signal-1", "claimed_source_urls": [URL], "_integrity_policy": True}
    contents = {"results": [{"url": URL, "text": case["source_text"]}]}
    return row, case["verdict"], contents


def test_captured_whatnot_date_only_partial_obeys_existing_integrity_policy():
    row, verdict, contents = _inputs()
    before = copy.deepcopy(verdict)
    result = verifier._apply_guardrails(row, verdict, contents=contents)
    item = result["signal_evaluations"][0]
    assert verifier._decision(result, company_quality=True) == "approve"
    assert item["claim_matches_miner_date"] == "no_date_in_content"
    assert not any(note.startswith("source_event_date:") for note in item["risk_notes"])
    assert item["supporting_quotes"] == before["signal_evaluations"][0]["supporting_quotes"]
    assert item["unsupported_parts"] == []


@pytest.mark.parametrize("defect", [
    "policy_off", "wrong_entity", "low_confidence", "overall_unqualified",
    "contradiction", "other_missing_fact", "date_and_quantity", "source_changed",
    "unfetched_source", "wrong_url", "date_contradicted", "other_risk",
    "missing_quote", "unavailable", "duplicate_evaluation",
    "global_missing_fact", "cross_page_join", "old_source_date",
])
def test_date_policy_never_repairs_substantive_or_unbound_failures(defect):
    row, verdict, contents = _inputs()
    item = verdict["signal_evaluations"][0]
    if defect == "policy_off": row["_integrity_policy"] = False
    elif defect == "wrong_entity": item["same_entity_check"] = "fail"
    elif defect == "low_confidence": item["confidence"] = "medium"
    elif defect == "overall_unqualified": verdict["overall_verdict"] = "uncertain"
    elif defect == "contradiction": item["contradicting_quotes"] = ["This integration is planned, not launched."]
    elif defect == "other_missing_fact": item["unsupported_parts"] = ["merchant integration"]
    elif defect == "date_and_quantity": item["unsupported_parts"] = ["event_date", "quantity"]
    elif defect == "source_changed": contents["results"][0]["text"] = "No integration was launched."
    elif defect == "unfetched_source": contents["results"] = []
    elif defect == "wrong_url": item["evidence_urls_used"] = ["https://unrelated.example/integration"]
    elif defect == "date_contradicted": item["claim_matches_miner_date"] = "contradicted"
    elif defect == "other_risk": item["risk_notes"].append("integration_not_available")
    elif defect == "missing_quote": item["supporting_quotes"] = []
    elif defect == "unavailable": item["signal_status"] = "unable_to_verify"
    elif defect == "duplicate_evaluation": verdict["signal_evaluations"].append(copy.deepcopy(item))
    elif defect == "global_missing_fact": verdict["missing_or_risks"] = ["Integration not available yet"]
    elif defect == "cross_page_join":
        item["supporting_quotes"] = ["Whatnot now integrates directly with Shopify"]
        contents["results"] = [
            {"url": URL, "text": "Whatnot now integrates"},
            {"url": URL, "text": "directly with Shopify"},
        ]
    elif defect == "old_source_date": item["risk_notes"].append("source_event_date:2024-01-01")
    result = verifier._apply_guardrails(row, verdict, contents=contents)
    assert verifier._decision(result, company_quality=True) != "approve"


@pytest.mark.asyncio
async def test_full_source_judge_reuses_content_without_a_paid_date_retry(monkeypatch):
    row, verdict, contents = _inputs()
    call = AsyncMock(return_value={"answer": verdict, "model": "test", "usage": {}})
    monkeypatch.setattr(verifier, "_call_openrouter", call)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value=contents))
    result = await verifier.verify_three_stage(
        object(), company_name="Whatnot", company_linkedin="",
        company_website="https://whatnot.com", source_url=URL,
        miner_claim=verdict["signal_evaluations"][0]["claim"],
        target_signal_text="Launched a new product or major platform capability in the last 12 months",
        evidence_type="PRODUCT_LAUNCH", stage1_soft_reject=True,
        integrity_policy=True, buyer_max_age_days=365,
    )
    assert result["decision"] == "approve"
    assert result["client_ready"] is True
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_captured_date_only_failure_reaches_normal_scorer_with_uncertain_date(monkeypatch):
    from qualification.scoring import lead_scorer
    from qualification.scoring.evaluation_clock import use_evaluation_date
    from tests.test_arena_score_integrity import _company_model, _icp_model

    _, verdict, contents = _inputs()
    call = AsyncMock(return_value={"answer": verdict, "model": "test", "usage": {}})
    monkeypatch.setattr(verifier, "_call_openrouter", call)
    monkeypatch.setattr(verifier, "_fetch_sd_then_exa", AsyncMock(return_value=contents))
    signal = _company_model([{
        "source": "news", "description": verdict["signal_evaluations"][0]["claim"],
        "url": URL, "date": "2026-04-29", "snippet": "", "matched_icp_signal": 0,
    }]).intent_signals[0]
    icp = _icp_model().model_copy(update={
        "intent_signals": ["Launched a new product or major platform capability in the last 12 months"],
        "intent_signal_evidence_types": ["PRODUCT_LAUNCH"],
    })
    receipts = []
    with use_evaluation_date("2026-11-15"):
        result = await lead_scorer._score_single_intent_signal(
            signal, icp, None, "Whatnot", "https://whatnot.com",
            trust_signal_date=True, stage1_soft_reject=True,
            llm_only_intent_gate=True, integrity_policy=True, verdict_out=receipts,
        )
    assert result[0] == 54.0
    assert result[2] == "uncertain"
    assert result[3] is None
    assert call.await_count == 1
    assert receipts[-1]["decision"] == "verified"
    assert receipts[-1].get("authoritative_date") is None
