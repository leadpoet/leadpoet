"""Read-only and activation-gate tests for historical dev calibration."""

from __future__ import annotations

import asyncio

from gateway.research_lab import dev_eval_runner, store
from scripts import research_lab_historical_dev_calibration as calibration


def test_latest_calibration_rows_are_unique_and_finite():
    rows = [
        {
            "candidate_id": "candidate-a",
            "realized_mean_delta": 0.3,
            "created_at": "2026-07-02T00:00:00Z",
        },
        {
            "candidate_id": "candidate-a",
            "realized_mean_delta": 0.1,
            "created_at": "2026-07-01T00:00:00Z",
        },
        {
            "candidate_id": "candidate-b",
            "realized_mean_delta": float("nan"),
            "created_at": "2026-07-03T00:00:00Z",
        },
    ]
    selected = calibration._latest_unique_calibrations(rows, limit=10)
    assert selected == [rows[0]]


def test_input_loader_uses_only_paginated_selects(monkeypatch):
    calls = []

    async def select_all(table, **kwargs):
        calls.append((table, kwargs))
        if table == "research_lab_score_calibration":
            return [
                {
                    "candidate_id": "candidate-a",
                    "realized_mean_delta": 0.2,
                    "created_at": "2026-07-02T00:00:00Z",
                }
            ]
        return [
            {
                "candidate_id": "candidate-a",
                "candidate_kind": "image_build",
                "candidate_model_manifest_doc": {},
            }
        ]

    monkeypatch.setattr(store, "select_all", select_all)
    selected, candidates = asyncio.run(calibration._load_inputs(10))
    assert [row["candidate_id"] for row in selected] == ["candidate-a"]
    assert set(candidates) == {"candidate-a"}
    assert [table for table, _kwargs in calls] == [
        "research_lab_score_calibration",
        "research_lab_candidate_artifacts",
    ]
    assert all(call[1]["allow_partial"] is True for call in calls)


def test_report_gate_uses_one_snapshot_and_positive_realized_lift(monkeypatch):
    rows = [
        {
            "candidate_id": f"candidate-{index}",
            "run_id": f"run-{index}",
            "node_id": f"node-{index}",
            "lane": "provider",
            "score_bundle_id": f"bundle-{index}",
            "realized_mean_delta": float(index),
        }
        for index in range(30)
    ]
    candidates = {row["candidate_id"]: {} for row in rows}

    async def load_inputs(_limit):
        return rows, candidates

    async def evaluate_one(*, evaluator, calibration, candidate_row):
        del evaluator, candidate_row
        index = int(str(calibration["candidate_id"]).rsplit("-", 1)[1])
        return {
            "candidate_id": calibration["candidate_id"],
            "eligible": True,
            "replayed_dev_score": float(index),
            "realized_mean_delta": float(index),
            "snapshot_manifest_hash": "sha256:" + "a" * 64,
            "dev_set_hash": "sha256:" + "b" * 64,
        }

    monkeypatch.setattr(calibration, "_load_inputs", load_inputs)
    monkeypatch.setattr(calibration, "_evaluate_one", evaluate_one)
    monkeypatch.setattr(
        dev_eval_runner,
        "snapshot_readiness",
        lambda _uri: {"ready": True, "reason": "ready", "dev_set_size": 8},
    )
    monkeypatch.setattr(dev_eval_runner, "DockerReplayDevEvaluator", lambda **_kwargs: object())

    report = asyncio.run(
        calibration.build_report(
            snapshot_uri="/immutable/snapshot",
            limit=30,
            concurrency=4,
        )
    )
    assert report["read_only"] is True
    assert report["provider_network_access"] == "disabled_by_docker_network_none"
    assert report["eligible_pair_count"] == 30
    assert report["spearman_rho"] == 1.0
    assert report["top_quartile_realized_lift"] > 0
    assert report["activation_gate"]["passed"] is True
