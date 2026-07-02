"""Regression tests for score-bundle → Langfuse score export (plan §28.3).

Uses the canonical valid-bundle fixture so a schema drift in the score
bundle shows up here, and pins the exact score names the dashboards key on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import research_lab.observability.score_export as se

FIXTURE = Path(__file__).resolve().parents[1] / "schemas" / "fixtures" / "research_evaluation_score_bundle.valid.json"

EXPECTED_AGGREGATE_SCORES = {
    "base_score",
    "candidate_score",
    "mean_delta",
    "delta_lcb",
    "sd_delta",
    "se_delta",
    "icp_count",
    "successful_icp_count",
    "hard_failure_count",
    "total_cost_usd",
}
EXPECTED_REWARD_SCORES = {
    "eligible_for_probation",
    "eligible_for_crown",
    "eligible_for_improvement_grant",
}


class FakeClient:
    def __init__(self) -> None:
        self.scores: list[dict] = []

    def score(self, **kwargs) -> None:
        self.scores.append(kwargs)


@pytest.fixture()
def bundle() -> dict:
    return json.loads(FIXTURE.read_text())


def test_payload_carries_all_aggregate_and_reward_scores(bundle: dict) -> None:
    payload = se.score_payload_from_bundle(bundle)
    for name in EXPECTED_AGGREGATE_SCORES:
        assert payload[name] == bundle["aggregates"][name], name
    for name in EXPECTED_REWARD_SCORES:
        assert payload[name] == bundle["reward_path"][name], name


def test_payload_health_extras_none_without_health_docs(bundle: dict) -> None:
    payload = se.score_payload_from_bundle(bundle)
    assert payload["baseline_health_gate_passed"] is None
    assert payload["scoring_health_status"] is None


def test_payload_health_extras_present_when_docs_exist(bundle: dict) -> None:
    enriched = {
        **bundle,
        "baseline_health": {"gate_passed": False},
        "scoring_health": {"health_status": "degraded", "sourced_zero_no_error_count": 3},
    }
    payload = se.score_payload_from_bundle(enriched)
    assert payload["baseline_health_gate_passed"] is False
    assert payload["scoring_health_status"] == "degraded"
    assert payload["sourced_zero_no_error_count"] == 3


def test_export_skips_none_values_and_sends_the_rest(monkeypatch, bundle: dict) -> None:
    fake = FakeClient()
    monkeypatch.setattr(se, "get_langfuse_client", lambda: fake)
    se.export_score_bundle_scores("trace-1", bundle)
    sent = {item["name"]: item["value"] for item in fake.scores}
    assert set(sent) == EXPECTED_AGGREGATE_SCORES | EXPECTED_REWARD_SCORES
    assert sent["eligible_for_probation"] is True
    assert sent["eligible_for_crown"] is False
    assert sent["mean_delta"] == bundle["aggregates"]["mean_delta"]


def test_export_metadata_carries_join_keys_and_is_safe(monkeypatch, bundle: dict) -> None:
    fake = FakeClient()
    monkeypatch.setattr(se, "get_langfuse_client", lambda: fake)
    se.export_score_bundle_scores("trace-1", bundle)
    metadata = fake.scores[0]["metadata"]
    assert metadata["score_bundle_hash"] == bundle["score_bundle_hash"]
    assert metadata["run_id"] == bundle["run_id"]
    assert metadata["execution_trace_ref"] == bundle["execution_trace_ref"]
    # The raw miner hotkey must never ride score metadata.
    assert bundle["miner_hotkey"] not in str(metadata)


def test_export_without_trace_id_is_noop(monkeypatch, bundle: dict) -> None:
    fake = FakeClient()
    monkeypatch.setattr(se, "get_langfuse_client", lambda: fake)
    se.export_score_bundle_scores("", bundle)
    assert fake.scores == []


def test_export_client_failure_contained(monkeypatch, bundle: dict) -> None:
    class Exploding:
        def score(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(se, "get_langfuse_client", lambda: Exploding())
    se.export_score_bundle_scores("trace-1", bundle)  # must not raise
