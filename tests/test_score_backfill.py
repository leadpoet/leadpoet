"""Tests for the Phase-5 score backfill (calibration + score-aware memory).

Covers:
- calibration-row projection from a scored candidate + bundle (predicted vs
  dev vs realized field mapping),
- record_score_backfill flag gating and (candidate, bundle) idempotency,
- read-time lesson hydration: realized deltas merge onto retrieved lessons by
  node_id when the flag is on, degrade to score-blind when off or when the
  calibration table is unavailable,
- score-aware attempt memory: realized bundle deltas + predicted_delta attach
  to recent_attempts entries; post-score statuses are not clobbered.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from gateway.research_lab import lesson_store, score_backfill
from gateway.research_lab.score_backfill import (
    SCORE_BACKFILL_ENABLED_ENV,
    SCORE_BUNDLE_CURRENT,
    SCORE_CALIBRATION_TABLE,
    build_calibration_row,
    fetch_score_enrichments_by_node,
    record_score_backfill,
)
from gateway.research_lab.worker import _candidate_attempt_memory

PARENT_HASH = "sha256:" + "p" * 64


class _FakeStore:
    def __init__(self, rows_by_table: Mapping[str, list[dict[str, Any]]] | None = None):
        self.rows_by_table = {
            name: list(rows) for name, rows in (rows_by_table or {}).items()
        }
        self.inserted: list[tuple[str, dict[str, Any]]] = []
        self.select_errors: dict[str, Exception] = {}

    async def select_many(self, table, *, columns="*", filters, order_by=(), limit=100):
        if table in self.select_errors:
            raise self.select_errors[table]
        rows = [dict(row) for row in self.rows_by_table.get(table, [])]
        for spec in filters:
            field, value = spec[0], spec[-1]
            if len(spec) == 3 and spec[1] == "in":
                allowed = {str(item) for item in value}
                rows = [row for row in rows if str(row.get(field)) in allowed]
            else:
                rows = [row for row in rows if str(row.get(field)) == str(value)]
        for field, desc in reversed(list(order_by or ())):
            rows.sort(key=lambda row: str(row.get(field) or ""), reverse=bool(desc))
        return rows[:limit]

    async def insert_row(self, table, row):
        self.inserted.append((table, dict(row)))
        stored = dict(row)
        stored.setdefault("created_at", f"2026-07-06T00:00:0{len(self.inserted)}Z")
        self.rows_by_table.setdefault(table, []).append(stored)
        return stored


def _scored_candidate(candidate_id: str = "cand-1", node_id: str = "node-1") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "run_id": "run-1",
        "island": "generalist",
        "current_score_bundle_id": "bundle-1",
        "current_candidate_status": "scored",
        "hypothesis_doc": {"predicted_delta": 1.0, "plan_path_id": "path-a"},
        "candidate_build_doc": {
            "loop_node_id": node_id,
            "loop_dev_score": 61.25,
            "loop_dev_score_version": "research-lab-dev-eval-mechanical-v1",
        },
        "candidate_patch_manifest": {
            "patch_doc": {
                "code_edit": {
                    "lane": "provider",
                    "plan_path_id": "path-a",
                    "target_files": ["sourcing_model/pipeline.py"],
                    "unified_diff_hash": "sha256:" + "d" * 64,
                }
            }
        },
    }


def _score_bundle() -> dict[str, Any]:
    return {
        "score_bundle_id": "bundle-1",
        "island": "generalist",
        "aggregates": {"mean_delta": -0.02, "delta_lcb": -0.11},
    }


def _daily_baseline_bundle(
    *,
    score_bundle_id: str = "bundle-daily",
    daily_delta: float = -1.5,
) -> dict[str, Any]:
    return {
        "score_bundle_id": score_bundle_id,
        "aggregates": {"mean_delta": 40.0, "delta_lcb": 35.0},
        "private_holdout_gate": {
            "decision": "private_holdout_approved",
            "private_holdout_evaluated": True,
            "baseline_aggregate_score": 20.0,
            "candidate_total_score": 20.0 + daily_delta,
            "candidate_delta_vs_daily_baseline": daily_delta,
        },
    }


def test_build_calibration_row_projects_all_three_score_sources():
    row = build_calibration_row(
        candidate=_scored_candidate(),
        score_bundle=_score_bundle(),
        score_bundle_id="bundle-1",
        outcome="promotion_rejected",
        created_by="scorer-1",
    )
    assert row["candidate_id"] == "cand-1"
    assert row["node_id"] == "node-1"
    assert row["lane"] == "provider"
    assert row["plan_path_id"] == "path-a"
    assert row["predicted_delta"] == 1.0
    assert row["dev_score"] == 61.25
    assert row["dev_score_version"] == "research-lab-dev-eval-mechanical-v1"
    assert row["realized_mean_delta"] == -0.02
    assert row["realized_delta_lcb"] == -0.11
    assert row["outcome"] == "promotion_rejected"
    assert row["score_bundle_id"] == "bundle-1"


def test_build_calibration_row_uses_daily_benchmark_not_candidate_vs_zero():
    row = build_calibration_row(
        candidate=_scored_candidate(),
        score_bundle=_daily_baseline_bundle(),
        score_bundle_id="bundle-daily",
    )
    assert row["realized_mean_delta"] == -1.5
    assert row["realized_delta_lcb"] is None


def test_calibration_prefers_live_patch_shape_and_rejects_non_finite_scores():
    candidate = {
        **_scored_candidate(),
        "patch_doc": {"lane": "direct-lane", "plan_path_id": "direct-path"},
        "hypothesis_doc": {"predicted_delta": float("inf")},
        "candidate_build_doc": {
            "loop_node_id": "node-direct",
            "loop_dev_score": float("nan"),
        },
    }
    bundle = {"aggregates": {"mean_delta": "nan", "delta_lcb": "inf"}}
    row = build_calibration_row(
        candidate=candidate,
        score_bundle=bundle,
        score_bundle_id="bundle-direct",
    )
    assert row["lane"] == "direct-lane"
    assert row["plan_path_id"] == "direct-path"
    assert row["predicted_delta"] is None
    assert row["dev_score"] is None
    assert row["realized_mean_delta"] is None
    assert row["realized_delta_lcb"] is None


def test_record_score_backfill_is_flag_gated(monkeypatch):
    monkeypatch.delenv(SCORE_BACKFILL_ENABLED_ENV, raising=False)
    store = _FakeStore()
    result = asyncio.run(
        record_score_backfill(
            candidate=_scored_candidate(),
            score_bundle_row={"score_bundle_id": "bundle-1"},
            score_bundle=_score_bundle(),
            store=store,
        )
    )
    assert result["status"] == "disabled"
    assert not store.inserted


def test_record_score_backfill_persists_once(monkeypatch):
    monkeypatch.setenv(SCORE_BACKFILL_ENABLED_ENV, "true")
    store = _FakeStore()
    first = asyncio.run(
        record_score_backfill(
            candidate=_scored_candidate(),
            score_bundle_row={"score_bundle_id": "bundle-1"},
            score_bundle=_score_bundle(),
            promotion_result={"status": "promotion_approved"},
            store=store,
        )
    )
    second = asyncio.run(
        record_score_backfill(
            candidate=_scored_candidate(),
            score_bundle_row={"score_bundle_id": "bundle-1"},
            score_bundle=_score_bundle(),
            promotion_result={"status": "promotion_approved"},
            store=store,
        )
    )
    assert first["status"] == "recorded"
    assert first["outcome"] == "promotion_approved"
    assert second["status"] == "already_recorded"
    assert len(store.inserted) == 1
    table, row = store.inserted[0]
    assert table == SCORE_CALIBRATION_TABLE
    assert row["outcome"] == "promotion_approved"


def test_record_score_backfill_treats_uniqueness_race_as_success(monkeypatch):
    monkeypatch.setenv(SCORE_BACKFILL_ENABLED_ENV, "true")

    class _RaceStore(_FakeStore):
        async def insert_row(self, table, row):
            await asyncio.sleep(0)
            if self.inserted:
                raise RuntimeError("23505 duplicate key violates unique constraint")
            self.inserted.append((table, dict(row)))
            return dict(row)

    store = _RaceStore()

    async def _run():
        return await asyncio.gather(
            *(
                record_score_backfill(
                    candidate=_scored_candidate(),
                    score_bundle_row={"score_bundle_id": "bundle-1"},
                    score_bundle=_score_bundle(),
                    store=store,
                )
                for _index in range(2)
            )
        )

    results = asyncio.run(_run())
    assert {row["status"] for row in results} == {"recorded", "already_recorded"}
    assert len(store.inserted) == 1


def test_fetch_score_enrichments_returns_newest_row_per_node(monkeypatch):
    store = _FakeStore(
        {
            SCORE_CALIBRATION_TABLE: [
                {
                    "node_id": "node-1",
                    "score_bundle_id": "bundle-old",
                    "realized_mean_delta": -0.5,
                    "realized_delta_lcb": -0.9,
                    "predicted_delta": 1.0,
                    "dev_score": 40.0,
                    "outcome": "promotion_rejected",
                    "created_at": "2026-07-05T00:00:00Z",
                },
                {
                    "node_id": "node-1",
                    "score_bundle_id": "bundle-daily",
                    "realized_mean_delta": 0.25,
                    "realized_delta_lcb": 0.05,
                    "predicted_delta": 1.0,
                    "dev_score": 66.0,
                    "outcome": "promotion_approved",
                    "created_at": "2026-07-06T00:00:00Z",
                },
            ],
            SCORE_BUNDLE_CURRENT: [
                {
                    "score_bundle_id": "bundle-daily",
                    "score_bundle_doc": _daily_baseline_bundle(),
                }
            ],
        }
    )
    enrichments = asyncio.run(
        fetch_score_enrichments_by_node(["node-1", "node-missing"], store=store)
    )
    assert set(enrichments) == {"node-1"}
    assert enrichments["node-1"]["realized_delta"] == -1.5
    assert enrichments["node-1"]["realized_delta_lcb"] is None
    assert enrichments["node-1"]["scored_outcome"] == "promotion_approved"


def test_fetch_score_enrichments_bulk_loads_nodes_and_bundles():
    class _BulkStore:
        def __init__(self):
            self.calls = []

        async def select_all(self, table, **kwargs):
            self.calls.append((table, kwargs))
            if table == SCORE_CALIBRATION_TABLE:
                return [
                    {
                        "node_id": f"node-{index}",
                        "score_bundle_id": f"bundle-{index}",
                        "realized_mean_delta": index / 100,
                        "realized_delta_lcb": 0.0,
                        "created_at": "2026-07-06T00:00:00Z",
                    }
                    for index in range(40)
                ]
            return [
                {
                    "score_bundle_id": f"bundle-{index}",
                    "score_bundle_doc": {
                        "aggregates": {
                            "mean_delta": index / 100,
                            "delta_lcb": 0.0,
                        }
                    },
                }
                for index in range(40)
            ]

    store = _BulkStore()
    node_ids = [f"node-{index}" for index in range(40)]
    enrichments = asyncio.run(
        fetch_score_enrichments_by_node(node_ids, store=store, limit=50)
    )
    assert len(enrichments) == 40
    assert len(store.calls) == 2
    table, kwargs = store.calls[0]
    assert table == SCORE_CALIBRATION_TABLE
    assert kwargs["filters"] == [("node_id", "in", node_ids)]
    bundle_table, bundle_kwargs = store.calls[1]
    assert bundle_table == SCORE_BUNDLE_CURRENT
    assert bundle_kwargs["filters"] == [
        ("score_bundle_id", "in", [f"bundle-{index}" for index in range(40)])
    ]


def test_fetch_score_enrichments_stays_score_blind_when_bundle_is_missing():
    store = _FakeStore(
        {
            SCORE_CALIBRATION_TABLE: [
                {
                    "node_id": "node-1",
                    "score_bundle_id": "bundle-missing",
                    "realized_mean_delta": 99.0,
                    "realized_delta_lcb": 98.0,
                    "predicted_delta": 1.0,
                    "dev_score": 66.0,
                    "outcome": "promotion_approved",
                    "created_at": "2026-07-06T00:00:00Z",
                }
            ]
        }
    )
    enrichment = asyncio.run(
        fetch_score_enrichments_by_node(["node-1"], store=store)
    )["node-1"]
    assert enrichment["realized_delta"] is None
    assert enrichment["realized_delta_lcb"] is None
    assert enrichment["predicted_delta"] == 1.0


def _reflection_event_row(*, node_id: str, created_at: str) -> dict[str, Any]:
    return {
        "event_id": f"event-{node_id}",
        "run_id": "run-1",
        "node_id": node_id,
        "event_type": lesson_store.REFLECTION_EVENT_TYPE,
        "created_at": created_at,
        "event_doc": {
            "lane": "provider",
            "outcome": "candidate_built",
            "target_files": ["sourcing_model/pipeline.py"],
            "reflection": {
                "lesson_id": f"lesson:{node_id}",
                "node_id": node_id,
                "component": "sourcing_model/pipeline.py",
                "champion_base": PARENT_HASH,
                "eval_version": "v1",
                "worked": "patch built",
                "failed": "",
                "why": "narrow edit",
                "next_question": "does it score",
            },
        },
    }


def _lesson_fixture_store() -> _FakeStore:
    return _FakeStore(
        {
            lesson_store.LOOP_EVENTS_TABLE: [
                _reflection_event_row(node_id="node-1", created_at="2026-07-05T00:00:00Z"),
                _reflection_event_row(node_id="node-2", created_at="2026-07-04T00:00:00Z"),
            ],
            SCORE_CALIBRATION_TABLE: [
                {
                    "node_id": "node-1",
                    "score_bundle_id": "bundle-1",
                    "realized_mean_delta": -0.02,
                    "realized_delta_lcb": -0.11,
                    "predicted_delta": 1.0,
                    "dev_score": 61.25,
                    "outcome": "promotion_rejected",
                    "created_at": "2026-07-06T00:00:00Z",
                },
            ],
            SCORE_BUNDLE_CURRENT: [
                {
                    "score_bundle_id": "bundle-1",
                    "score_bundle_doc": _score_bundle(),
                }
            ],
        }
    )


def test_lessons_hydrate_realized_deltas_when_backfill_enabled(monkeypatch):
    monkeypatch.setenv(SCORE_BACKFILL_ENABLED_ENV, "true")
    lessons = asyncio.run(
        lesson_store.fetch_recent_lessons(
            active_parent_hash=PARENT_HASH, store=_lesson_fixture_store()
        )
    )
    by_node = {lesson["node_id"]: lesson for lesson in lessons}
    assert by_node["node-1"]["realized_delta"] == -0.02
    assert by_node["node-1"]["scored_outcome"] == "promotion_rejected"
    assert by_node["node-1"]["predicted_delta"] == 1.0
    assert "realized_delta" not in by_node["node-2"]


def test_lessons_stay_score_blind_when_backfill_disabled(monkeypatch):
    monkeypatch.delenv(SCORE_BACKFILL_ENABLED_ENV, raising=False)
    lessons = asyncio.run(
        lesson_store.fetch_recent_lessons(
            active_parent_hash=PARENT_HASH, store=_lesson_fixture_store()
        )
    )
    assert lessons
    assert all("realized_delta" not in lesson for lesson in lessons)


def test_lessons_survive_missing_calibration_table(monkeypatch):
    monkeypatch.setenv(SCORE_BACKFILL_ENABLED_ENV, "true")
    store = _lesson_fixture_store()
    store.select_errors[SCORE_CALIBRATION_TABLE] = RuntimeError("relation does not exist")
    lessons = asyncio.run(
        lesson_store.fetch_recent_lessons(active_parent_hash=PARENT_HASH, store=store)
    )
    assert lessons
    assert all("realized_delta" not in lesson for lesson in lessons)


def test_attempt_memory_attaches_realized_and_predicted_deltas():
    attempt = _candidate_attempt_memory(
        _scored_candidate(),
        bundle_deltas={"bundle-1": {"score_delta": -0.02, "score_delta_lcb": -0.11}},
    )
    assert attempt["status"] == "scored"
    assert attempt["score_delta"] == -0.02
    assert attempt["score_delta_lcb"] == -0.11
    assert attempt["predicted_delta"] == 1.0


def test_attempt_memory_keeps_post_score_status():
    row = {**_scored_candidate(), "current_candidate_status": "rejected"}
    attempt = _candidate_attempt_memory(
        row, bundle_deltas={"bundle-1": {"score_delta": -0.5, "score_delta_lcb": -0.9}}
    )
    assert attempt["status"] == "rejected"
    assert attempt["score_delta"] == -0.5


def test_attempt_memory_unscored_rows_stay_unchanged():
    row = {**_scored_candidate(), "current_score_bundle_id": ""}
    attempt = _candidate_attempt_memory(row, bundle_deltas={})
    assert "score_delta" not in attempt
    assert "predicted_delta" not in attempt
