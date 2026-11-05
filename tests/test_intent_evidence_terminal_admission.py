"""V2 full-source admission is strict and leaves v1/native rows unchanged."""
from copy import deepcopy

import pytest

from qualification.scoring.intent_evidence_reuse import reusable_evidence_terminal_judgment
from qualification.scoring.signal_judgment_cache import reusable_terminal_judgment
from tests.test_retained_source_native_retry import _fixture


def _inputs(positive=False):
    fixture = _fixture()
    url = fixture["source_url"]
    row = {"raw": 54.0 if positive else 0.0, "after_decay": 54.0 if positive else 0.0,
        "decay": 1.0 if positive else 0.0, "confidence": 90 if positive else 0,
        "date_status": "in_window" if positive else "verified",
        "date_verdict": "in_window" if positive else "not_evaluated",
        "claim_support_verdict": "supported" if positive else "contradicted",
        "matched_icp_signal": 0, "evidence_type": "PARTNERSHIP", "evidence_urls": [url],
        "judge_verdict": {"decision": "verified" if positive else "rejected_three_stage",
            "pipeline_decision": "approve" if positive else "reject", "client_ready": positive,
            "claim_support_verdict": "supported" if positive else "contradicted",
            "verification_trace": {"evidence_url": url, "evidence_urls": [url],
                "evidence_source": "news", "extracted_signal_date": fixture["miner_date"],
                "verified_source_context": [{"url": url, "text": "A compact source window omitted the quote."}],
                "final_disposition": "approve" if positive else "reject",
                "intent_verdict": deepcopy(fixture["native_retry_answer"] if positive else fixture["wrong_acquisition_answer"])}}}
    evidence = {"schema_version": "leadpoet.intent_evidence_input.v1",
        "row": {"claimed_source_urls": [url], "signal_date": fixture["miner_date"]},
        "contents": {"results": [{"url": url, "text": fixture["source_text"]}]}}
    expected = {"expected_urls": [url], "expected_index": 0, "expected_evidence_type": "PARTNERSHIP"}
    return row, evidence, expected


@pytest.mark.parametrize("positive", [False, True])
def test_full_fresh_snapshot_admits_all_grounded_quotes_without_rewriting_native_row(positive):
    row, evidence, expected = _inputs(positive)
    original, original_evidence = deepcopy(row), deepcopy(evidence)
    assert reusable_terminal_judgment(row, **expected) is False
    assert reusable_evidence_terminal_judgment(row, evidence_input=evidence, **expected) is True
    assert row == original and evidence == original_evidence
    assert reusable_terminal_judgment(row, **expected) is False


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("invalid", ["partial", "unavailable", "error", "incomplete", "missing_source", "wrong_source", "ungrounded_quote", "date", "recovery"])
def test_v2_never_admits_partial_unavailable_incomplete_or_unbound_evidence(positive, invalid):
    row, evidence, expected = _inputs(positive)
    trace = row["judge_verdict"]["verification_trace"]
    evaluation = trace["intent_verdict"]["signal_evaluations"][0]
    if invalid == "partial":
        evaluation["signal_status"] = "partially_supported"
    elif invalid == "unavailable":
        row["judge_verdict"]["pipeline_decision"] = "unavailable"
    elif invalid == "error":
        row["judge_verdict"]["error_class"] = "provider_error"
    elif invalid == "incomplete":
        evidence["contents"]["results"] = []
    elif invalid == "missing_source":
        expected["expected_urls"].append("https://other.example/missing")
        evidence["row"]["claimed_source_urls"] = list(expected["expected_urls"])
    elif invalid == "wrong_source":
        evidence["contents"]["results"][0]["url"] = "https://other.example/source"
    elif invalid == "ungrounded_quote":
        evaluation["supporting_quotes"].append("This second quote was never in the actual source.")
    elif invalid == "date":
        evidence["row"]["signal_date"] = "2026-10-23"
    else:
        trace["source_resolution"] = {"status": "verified"}
    assert reusable_evidence_terminal_judgment(row, evidence_input=evidence, **expected) is False
