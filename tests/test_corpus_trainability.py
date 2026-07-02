"""Tests for P15/P10/P16 trainability invariants + eligibility pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from research_lab.trajectory_corpus import (
    CorpusSplitPolicyRecord,
    TrajectoryCorpusSourceRecord,
    build_trajectory_corpus_manifest,
)
from gateway.research_lab import corpus_export as ce


def _record(trajectory_id, split, **extra):
    base = dict(
        source_id=f"trajectory_source:{trajectory_id}",
        trajectory_id=trajectory_id,
        trajectory_hash="sha256:" + "0" * 64,
        trajectory_schema_valid=True,
        event_count=3,
        execution_trace_refs=[f"execution_trace:{trajectory_id}"],
        evidence_bundle_refs=[],
        results_ledger_refs=[f"results_ledger:{trajectory_id}"],
        receipt_refs=[],
        cost_ledger_refs=[f"cost_ledger:{trajectory_id}"],
        release_policy_ref="release_policy:x",
        trajectory_rights_ref="trajectory_rights:x",
        distillation_rights_ref="distillation_rights:x",
        pii_review_ref="pii_review:x",
        legal_gate_ref="legal_gate:x",
        island="lead_generation",
        split=split,
        data_state="production_measured",
        measured_data=True,
    )
    base.update(extra)
    return TrajectoryCorpusSourceRecord.from_mapping(base)


def _policy():
    return CorpusSplitPolicyRecord(
        split_policy_id="split_policy:v1",
        train_percent=80,
        validation_percent=10,
        holdout_percent=10,
        deterministic_seed_ref="sha256:" + "a" * 64,
        group_by_fields=("trajectory_id", "brief_id", "customer_ref", "split_cluster_key"),
    )


# ---------------------------------------------------------------------------
# P15 near-duplicate leak guard
# ---------------------------------------------------------------------------


def test_same_cluster_key_across_splits_fails_guard():
    from research_lab.trajectory_corpus import _split_policy_passes

    # A paraphrased ICP: same brief cluster key, different trajectory ids,
    # one in train and one in holdout.
    records = [
        _record("traj-1", "train", split_cluster_key="sha256:brief-cluster-A"),
        _record("traj-2", "holdout", split_cluster_key="sha256:brief-cluster-A"),
    ]
    assert _split_policy_passes(_policy(), records) is False


def test_distinct_cluster_keys_pass_guard():
    from research_lab.trajectory_corpus import _split_policy_passes

    records = [
        _record("traj-1", "train", split_cluster_key="sha256:brief-cluster-A"),
        _record("traj-2", "holdout", split_cluster_key="sha256:brief-cluster-B"),
    ]
    assert _split_policy_passes(_policy(), records) is True


def test_same_brief_id_across_splits_fails_guard():
    from research_lab.trajectory_corpus import _split_policy_passes

    records = [
        _record("traj-1", "train", brief_id="brief-1"),
        _record("traj-2", "validation", brief_id="brief-1"),
    ]
    assert _split_policy_passes(_policy(), records) is False


def test_manifest_leakage_guard_reflects_cluster_collision():
    train = _record("traj-1", "train", split_cluster_key="k")
    holdout = _record("traj-2", "holdout", split_cluster_key="k")
    manifest = build_trajectory_corpus_manifest(
        corpus_id="corpus:1",
        source_records=[train, holdout],
        split_policy=_policy(),
    )
    assert manifest.eval_leakage_guard_passed is False


# ---------------------------------------------------------------------------
# P15 token budget
# ---------------------------------------------------------------------------


def test_over_token_budget_flag_roundtrips():
    record = _record("traj-1", "train", token_count=2_000_000, over_token_budget=True)
    assert record.over_token_budget is True
    assert record.to_dict()["token_count"] == 2_000_000
    restored = TrajectoryCorpusSourceRecord.from_mapping(record.to_dict())
    assert restored.over_token_budget is True


# ---------------------------------------------------------------------------
# P10 readiness gate binding
# ---------------------------------------------------------------------------


def test_bind_readiness_gates_all_pass_enables_training():
    record = _record("traj-1", "train")
    bound = ce.bind_readiness_gates(
        record,
        protected_scan={"ref": "scan:protected:1", "passed": True, "hits": []},
        pii_scan={"ref": "scan:pii:1", "passed": True},
        rights={"ref": "rights:1", "passed": True, "distillation_allowed": True},
        legal={"ref": "legal:1", "passed": True},
    )
    assert bound.eligible_for_training is True
    assert bound.eligible_for_distillation is True
    assert bound.rights_verified is True
    assert bound.pii_review_passed is True


def test_bind_readiness_gates_missing_ref_blocks_training():
    record = _record("traj-1", "train")
    bound = ce.bind_readiness_gates(
        record,
        protected_scan={"ref": "scan:protected:1", "passed": True, "hits": []},
        pii_scan={"passed": True},  # no ref → not a real scan
        rights={"ref": "rights:1", "passed": True, "distillation_allowed": True},
        legal={"ref": "legal:1", "passed": True},
    )
    assert bound.pii_review_passed is False
    assert bound.eligible_for_training is False


def test_bind_readiness_gates_protected_hits_block_training():
    record = _record("traj-1", "train")
    bound = ce.bind_readiness_gates(
        record,
        protected_scan={"ref": "scan:protected:1", "passed": True, "hits": ["raw_prompt"]},
        pii_scan={"ref": "scan:pii:1", "passed": True},
        rights={"ref": "rights:1", "passed": True, "distillation_allowed": True},
        legal={"ref": "legal:1", "passed": True},
    )
    assert bound.contains_raw_evidence_snapshot is True
    assert bound.eligible_for_training is False


def test_bind_readiness_gates_over_budget_blocks_training():
    record = _record("traj-1", "train", token_count=9_000_000, over_token_budget=True)
    bound = ce.bind_readiness_gates(
        record,
        protected_scan={"ref": "scan:protected:1", "passed": True, "hits": []},
        pii_scan={"ref": "scan:pii:1", "passed": True},
        rights={"ref": "rights:1", "passed": True, "distillation_allowed": True},
        legal={"ref": "legal:1", "passed": True},
    )
    assert bound.eligible_for_training is False


# ---------------------------------------------------------------------------
# P15 axis-B → axis-A rewrite seed
# ---------------------------------------------------------------------------


def _axis_b_trace():
    return {
        "run_id": "trace-1",
        "trace_doc": {"trajectory_axis": "axis_b"},
        "calls": [
            {
                "call_kind": "incontainer_trace",
                "stage": "incontainer_model_runtime",
                "s3_ref": "s3://bucket/incontainer/icp.json",
                "sha256": "sha256:ii",
            }
        ],
    }


def test_rewrite_disabled_by_default_produces_no_seed():
    result = ce.rewrite_axis_b_to_axis_a_seed(_axis_b_trace())
    assert result["sft_seed_from_rewritten_axis_a"] is False
    assert result["seed"] is None
    assert result["reason"] == "rewrite_disabled"


def test_rewrite_enabled_grounds_axis_b_into_seed():
    result = ce.rewrite_axis_b_to_axis_a_seed(_axis_b_trace(), enabled=True)
    assert result["sft_seed_from_rewritten_axis_a"] is True
    seed = result["seed"]
    assert seed["source_axis"] == "axis_b"
    assert seed["target_axis"] == "axis_a"
    assert seed["grounded_call_count"] == 1
    assert seed["grounded_call_refs"][0]["s3_ref"].startswith("s3://")
    assert seed["seed_hash"].startswith("sha256:")


def test_rewrite_refuses_axis_a_source():
    trace = _axis_b_trace()
    trace["trace_doc"]["trajectory_axis"] = "axis_a"
    result = ce.rewrite_axis_b_to_axis_a_seed(trace, enabled=True)
    assert result["sft_seed_from_rewritten_axis_a"] is False
    assert result["reason"] == "not_axis_b"


def test_rewrite_refuses_when_no_groundable_calls():
    trace = _axis_b_trace()
    trace["calls"] = []
    result = ce.rewrite_axis_b_to_axis_a_seed(trace, enabled=True)
    assert result["sft_seed_from_rewritten_axis_a"] is False
    assert result["reason"] == "no_groundable_calls"


# ---------------------------------------------------------------------------
# P16 manifest from real projected rows
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, tables):
        self.tables = tables

    async def select_many(self, table, *, columns="*", filters=(), order_by=(), limit=100):
        return [dict(row) for row in self.tables.get(table, [])][:limit]


async def test_build_manifest_from_projected_rows_no_fixtures():
    envelopes = [
        {
            "trajectory_id": "traj-1",
            "run_id": "run-1",
            "island": "lead_generation",
            "brief_id": "brief-1",
            "brief_sanitized_ref": "sha256:brief-cluster-A",
            "corpus_source_record": _record("traj-1", "train").to_dict(),
            "created_at": "2026-07-01T00:00:00Z",
        },
        {
            # Older envelope without an embedded source record → reconstructed.
            "trajectory_id": "traj-2",
            "run_id": "run-2",
            "island": "lead_generation",
            "brief_id": "brief-2",
            "brief_sanitized_ref": "sha256:brief-cluster-B",
            "trajectory_hash": "sha256:" + "1" * 64,
            "created_at": "2026-07-02T00:00:00Z",
        },
    ]
    store = FakeStore(
        {
            ce.RESEARCH_TRAJECTORIES_TABLE: envelopes,
            ce.EXECUTION_TRACES_TABLE: [{"run_id": "trace-uuid-2"}],
        }
    )
    manifest = await ce.build_manifest_from_projected_rows(
        store=store, corpus_id="corpus:real", split_policy=_policy()
    )
    assert manifest.trajectory_count == 2
    assert manifest.uses_local_fixtures is False
    ids = {record.trajectory_id for record in manifest.source_records}
    assert ids == {"traj-1", "traj-2"}
    reconstructed = next(r for r in manifest.source_records if r.trajectory_id == "traj-2")
    assert reconstructed.split_cluster_key == "sha256:brief-cluster-B"
    assert reconstructed.execution_trace_refs == ("execution_trace:trace-uuid-2",)
