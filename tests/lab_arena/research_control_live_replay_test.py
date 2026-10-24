"""Replay saved real provider replies; this test makes no network calls.

An expected invalid reply remains invalid. Passing this test must not be reported
as a perfect reviewer success rate or as proof of production submission accuracy.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import contracts, research_control, source_bundle

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = json.loads((ROOT / "docs/evidence/research-control/development-check-20261024.json").read_text())


@pytest.mark.parametrize("case", EVIDENCE["fixture_results"], ids=lambda row: row["case"])
def test_saved_provider_reply_remains_bound_and_its_failures_stay_failures(case, tmp_path):
    source = ROOT / "tests/fixtures/research_control" / case["case"]
    archive = tmp_path / "source.tar.gz"
    source_bundle.write_source_archive(source, archive)
    prepared = research_control.prepare_request(archive.read_bytes(), max_output_tokens=4096)
    parameters = copy.deepcopy(prepared.parameters)
    parameters["provider"].update(EVIDENCE["provider_request_override"])
    bound = len((contracts.canonical_json(parameters) + "\n").encode()) + 16
    prepared = replace(prepared, parameters=parameters,
                       request_sha256=contracts.document_hash(parameters),
                       input_tokens_upper_bound=bound,
                       requires_context_integrity=bound + parameters["max_tokens"] > prepared.context_window_tokens)
    report, raw = case["report"], case["raw_provider_response"]
    assert EVIDENCE["policy_hash"] == prepared.policy_hash == report["policy_hash"]
    assert report["request_sha256"] == prepared.request_sha256
    assert report["source_archive_sha256"] == prepared.source_archive_sha256
    assert report["source_manifest_sha256"] == prepared.source_manifest_sha256
    assert hashlib.sha256(raw.encode()).hexdigest() == report["response_sha256"]
    if report["valid"]:
        result = research_control.parse_response(json.loads(raw), prepared)
        assert result.classification == case["expected_conditional_controller"]
        assert result.to_document() == report["assessment"]
    else:
        with pytest.raises(research_control.ResearchControlError) as error:
            research_control.parse_response(json.loads(raw), prepared)
        assert type(error.value).__name__ + ":" + error.value.reason == report["failure"]


def test_full_champion_reply_does_not_supply_a_completed_assessment():
    record = EVIDENCE["public_champion"]
    # Its exact original remains private. The public projection excludes opaque
    # signatures and generated reasoning; it cannot replay the original parser.
    response = record["provider_response_projection"]
    assert contracts.document_hash(response) == record["projection_sha256"]
    assert record["valid"] is False
    assert "assessment" not in record
    assert response["finish_reason"] == "length"
    assert response["has_openrouter_metadata"] is False
