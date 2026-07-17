from __future__ import annotations

import hashlib
import json

import pytest

from gateway.tee.bootstrap_acceptance_corpus_v2 import (
    AcceptanceCorpusBootstrapV2Error,
    _timestamp,
    build_fixture_index,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class _Reader:
    def __init__(self, *, secret_provider=False):
        self.secret_provider = secret_provider

    def rows(self, table, **_kwargs):
        if table == "research_evaluation_score_bundles":
            return [
                {
                    "score_bundle_id": "score:%03d" % index,
                    "created_at": "2026-06-%02dT00:00:00Z" % (1 + index % 30),
                    "score_bundle_hash": _hash("score:%d" % index),
                    "anchored_hash": _hash("anchor:%d" % index),
                    "score_bundle_doc": {"aggregates": {"mean_delta": 2.0}},
                }
                for index in range(100)
            ]
        if table == "research_lab_private_model_benchmark_bundles":
            return [
                {
                    "benchmark_bundle_id": "benchmark:%02d" % index,
                    "benchmark_date": "2026-06-%02d" % (index + 1),
                    "created_at": "2026-06-%02dT01:00:00Z" % (index + 1),
                    "benchmark_bundle_hash": _hash("benchmark:%d" % index),
                    "anchored_hash": _hash("benchmark-anchor:%d" % index),
                    "aggregate_score": 50.0,
                    "benchmark_quality": "complete",
                    "evaluation_epoch": index,
                }
                for index in range(14)
            ]
        if table == "research_lab_auto_research_loop_events":
            return [{
                "event_id": "event:1", "run_id": "run:1",
                "created_at": "2026-06-15T00:00:00Z",
                "anchored_hash": _hash("loop"),
                "candidate_artifact_hash": _hash("artifact"),
                "candidate_patch_hash": _hash("patch"),
                "event_type": "loop_completed", "loop_status": "completed",
            }]
        if table == "research_lab_provider_usage_ledger":
            return [{
                "usage_row_id": "usage:1",
                "recorded_at": "2026-06-16T00:00:00Z",
                "request_fingerprint": _hash("request"),
                "provider_id": "sk-or-secret" if self.secret_provider else "provider:1",
                "endpoint_class": "search", "status": "succeeded",
                "est_cost_microusd": 5,
            }]
        if table == "research_lab_emission_allocation_snapshots":
            return [{
                "allocation_id": "allocation:1",
                "created_at": "2026-06-17T00:00:00Z",
                "allocation_hash": _hash("allocation"),
                "input_hash": _hash("allocation-input"),
                "epoch": 1, "netuid": 71, "snapshot_status": "active",
                "champion_alpha_percent": 1, "reimbursement_alpha_percent": 1,
                "source_add_alpha_percent": 1, "unallocated_alpha_percent": 0,
            }]
        if table == "research_lab_signed_audit_bundles":
            return [
                {
                    "audit_bundle_id": "audit:%d" % index,
                    "created_at": "2026-06-%02dT02:00:00Z" % (1 + index % 30),
                    "epoch": 23000 + index,
                    "audit_bundle_hash": _hash("audit:%d" % index),
                    "anchored_hash": _hash("audit-anchor:%d" % index),
                    "schema_version": "v1",
                }
                for index in range(50)
            ]
        raise AssertionError(table)


class _AllHoldoutReader(_Reader):
    def rows(self, table, **kwargs):
        rows = super().rows(table, **kwargs)
        if table != "research_evaluation_score_bundles":
            return rows
        for index, row in enumerate(rows):
            if index == 42:
                gate = {
                    "decision": "private_holdout_approved",
                    "private_holdout_evaluated": True,
                    "baseline_aggregate_score": 40.0,
                    "candidate_total_score": 42.0,
                    "candidate_delta_vs_daily_baseline": 2.0,
                }
            else:
                gate = {
                    "decision": "rejected_before_private_holdout",
                    "private_holdout_evaluated": False,
                }
            row["score_bundle_doc"]["private_holdout_gate"] = gate
        return rows


def test_bootstrap_uses_real_rows_and_replays_every_promotion_branch(tmp_path):
    fixtures = build_fixture_index(_Reader(), corpus_root=tmp_path / "corpus")
    counts = {}
    statuses = set()
    for fixture in fixtures:
        counts[fixture["kind"]] = counts.get(fixture["kind"], 0) + 1
        if fixture["kind"] == "promotion_branch":
            statuses.add(fixture["metadata"]["status"])
            artifact = json.loads(
                (tmp_path / "corpus" / fixture["artifact_path"]).read_text()
            )
            assert artifact["expected_decision"]["status"] == fixture["metadata"]["status"]

    assert counts == {
        "score_bundle": 100,
        "daily_benchmark": 14,
        "promotion_branch": 6,
        "autoresearch_run": 1,
        "provider_tape": 1,
        "reward_allocation": 1,
        "weight_epoch": 50,
    }
    assert statuses == {
        "disabled",
        "rejected_legacy_patch_candidate",
        "rejected_basis_unavailable",
        "rejected_below_threshold",
        "stale_parent_needs_rescore",
        "promotion_passed",
    }


def test_bootstrap_rejects_secret_markers_before_signing(tmp_path):
    with pytest.raises(
        AcceptanceCorpusBootstrapV2Error,
        match="forbidden secret marker",
    ):
        build_fixture_index(
            _Reader(secret_provider=True),
            corpus_root=tmp_path / "corpus",
        )


def test_timestamp_accepts_postgrest_five_digit_fraction_on_python_39():
    assert (
        _timestamp("2026-07-14T06:27:29.75749+00:00")
        == "2026-07-14T06:27:29.757490Z"
    )


def test_bootstrap_selects_authoritative_pass_when_every_row_has_holdout_gate(
    tmp_path,
):
    fixtures = build_fixture_index(
        _AllHoldoutReader(),
        corpus_root=tmp_path / "corpus",
    )
    passed = next(
        fixture
        for fixture in fixtures
        if fixture["kind"] == "promotion_branch"
        and fixture["metadata"]["status"] == "promotion_passed"
    )
    artifact = json.loads(
        (tmp_path / "corpus" / passed["artifact_path"]).read_text()
    )
    assert artifact["source_score_bundle_hash"] == _hash("score:42")
    assert artifact["expected_decision"]["status"] == "promotion_passed"
