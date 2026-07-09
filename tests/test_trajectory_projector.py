"""Tests for the section 9.1 trajectory-capture projector.

Synthetic live-event fixtures model the real emitter shapes in
gateway/research_lab/worker.py + code_loop_engine.py (create_auto_research_loop_event
rows), scoring_worker.py ("scored" candidate evaluation events with the
private_holdout_gate doc), and promotion.py (promotion / version events).
Projected records are checked with the REAL validators:
schemas/research_trajectory.schema.json via research_lab.schema_validation and
research_lab.trajectory_corpus.validate_trajectory_corpus_source_record.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Mapping, Sequence

import pytest

import gateway.research_lab.trajectory_projector as trajectory_projector_mod
from gateway.research_lab.trajectory_projector import (
    CANONICAL_EVENT_TYPES,
    PROJECTOR_ENABLED_ENV,
    RESULTS_LEDGER_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_EVENTS_TABLE,
    build_trajectory_projection,
    find_protected_material,
    project_completed_runs,
    project_run,
    sanitize_capture_payload,
    schema_event_view,
    trajectory_id_for_run,
    verify_anchored_hash,
)
from research_lab.canonical import sha256_json
from research_lab.schema_validation import validate_schema_record
from research_lab.trajectory_corpus import validate_trajectory_corpus_source_record


RUN_ID = "11111111-1111-4111-8111-111111111111"
TICKET_ID = "22222222-2222-4222-8222-222222222222"
BRIEF_ID = "33333333-3333-4333-8333-333333333333"
RECEIPT_ID = "44444444-4444-4444-8444-444444444444"
CANDIDATE_1 = "55555555-5555-4555-8555-555555555551"
CANDIDATE_3 = "55555555-5555-4555-8555-555555555553"
VERSION_ID = "66666666-6666-4666-8666-666666666666"
SCORE_BUNDLE_ID = "score_bundle:abc123"


def test_projector_and_trace_join_keys_default_on(monkeypatch):
    monkeypatch.delenv(PROJECTOR_ENABLED_ENV, raising=False)
    monkeypatch.delenv("RESEARCH_LAB_EXECUTION_TRACE_TRAJECTORY_ID_ENABLED", raising=False)

    assert trajectory_projector_mod.projector_enabled() is True
    assert trajectory_projector_mod._execution_trace_trajectory_id_enabled() is True


# ---------------------------------------------------------------------------
# Fake store
# ---------------------------------------------------------------------------


class FakeStore:
    """In-memory stand-in for gateway.research_lab.store's query helpers."""

    def __init__(self, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {
            name: [dict(row) for row in rows] for name, rows in tables.items()
        }
        self.inserted: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def _matches(row: Mapping[str, Any], filters: Any) -> bool:
        for spec in filters:
            if len(spec) == 2:
                field, value = spec
                if str(row.get(field)) != str(value):
                    return False
            else:
                field, op, value = spec
                if op == "eq":
                    if str(row.get(field)) != str(value):
                        return False
                elif op == "in":
                    if str(row.get(field)) not in {str(item) for item in value}:
                        return False
                else:
                    raise ValueError(f"unsupported fake filter op {op}")
        return True

    def _select(self, table: str, filters: Any, order_by: Any = ()) -> list[dict[str, Any]]:
        rows = [
            dict(row)
            for row in self.tables.get(table, [])
            if self._matches(row, filters)
        ]
        for field, desc in reversed(list(order_by or ())):
            rows.sort(key=lambda row: (row.get(field) is None, row.get(field)), reverse=bool(desc))
        return rows

    async def select_one(self, table: str, *, columns: str = "*", filters: Any):
        rows = self._select(table, filters)
        return rows[0] if rows else None

    async def select_many(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Any,
        order_by: Any = (),
        limit: int = 100,
    ):
        return self._select(table, filters, order_by)[:limit]

    async def select_all(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Any,
        order_by: Any = (),
        batch_size: int = 1000,
        max_rows: int = 10000,
    ):
        return self._select(table, filters, order_by)[:max_rows]

    async def insert_row(self, table: str, row: dict[str, Any]):
        stored = self.tables.setdefault(table, [])
        if table == TRAJECTORIES_TABLE:
            if any(r.get("trajectory_id") == row.get("trajectory_id") for r in stored):
                raise RuntimeError("duplicate key value violates unique constraint")
        if table == TRAJECTORY_EVENTS_TABLE:
            if any(
                r.get("trajectory_id") == row.get("trajectory_id")
                and r.get("seq") == row.get("seq")
                for r in stored
            ):
                raise RuntimeError("duplicate key value violates unique constraint")
        stored.append(dict(row))
        self.inserted.setdefault(table, []).append(dict(row))
        return dict(row)

    def write_count(self) -> int:
        return sum(len(rows) for rows in self.inserted.values())


# ---------------------------------------------------------------------------
# Synthetic live-event fixtures (modelled on the real emit sites)
# ---------------------------------------------------------------------------


def _cost_ledger(total_usd: float, stage: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "stage": stage,
        "total_usd": round(total_usd, 6),
        "actual_openrouter_cost_usd": round(total_usd, 6),
        "actual_openrouter_cost_microusd": int(total_usd * 1_000_000),
        "estimated_cost_usd": round(total_usd, 6),
        "openrouter_call_count": 1,
        "official_scoring": False,
    }


def _provider_usage(prompt: int, completion: int) -> list[dict[str, Any]]:
    return [
        {
            "provider": "openrouter",
            "key_source": "miner_key_ref",
            "model": "anthropic/claude-sonnet-4",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": 0.01,
        }
    ]


def _loop_event(
    seq: int,
    event_type: str,
    *,
    ts: str,
    node_id: str | None = None,
    loop_status: str = "running",
    total_usd: float = 0.0,
    event_doc: dict[str, Any] | None = None,
    provider_usage: list[dict[str, Any]] | None = None,
    candidate_artifact_hash: str | None = None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Shape of a research_lab_auto_research_loop_events row (store.py:658)."""
    return {
        "event_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "run_id": RUN_ID,
        "ticket_id": TICKET_ID,
        "receipt_id": RECEIPT_ID,
        "seq": seq,
        "event_type": event_type,
        "loop_status": loop_status,
        "node_id": node_id,
        "worker_ref": "hosted-worker-1",
        "elapsed_seconds": elapsed_seconds,
        "candidate_artifact_hash": candidate_artifact_hash,
        "candidate_patch_hash": None,
        "provider_usage": provider_usage or [],
        "cost_ledger": _cost_ledger(total_usd, event_type),
        "event_doc": event_doc or {},
        "anchored_hash": "sha256:fixture",
        "created_at": ts,
    }


def _drafted_doc(iteration: int, lane: str, plan_path_id: str) -> dict[str, Any]:
    """event_doc shape from code_loop_engine.py:842 (code_edit_drafted)."""
    return {
        "iteration": iteration,
        "lane": lane,
        "plan_path_id": plan_path_id,
        "loop_direction_plan_hash": "sha256:planhash",
        "target_files": ["qualify/sources.py"],
        "unified_diff_hash": sha256_json({"unified_diff": f"diff-{iteration}"}),
        "hypothesis": {
            "failure_mode": "sparse evidence on niche ICPs",
            "mechanism": "add a second retrieval pass over cached snapshots",
            "expected_improvement": "higher evidence coverage",
            "risk": "slower loop",
            "predicted_delta": 0.5,
        },
    }


def happy_path_tables() -> dict[str, list[dict[str, Any]]]:
    day = "2026-06-20T10:{m:02d}:00Z"
    artifact_1 = "sha256:candidate-artifact-1"
    artifact_3 = "sha256:candidate-artifact-3"
    loop_events = [
        _loop_event(
            0,
            "loop_started",
            ts=day.format(m=0),
            event_doc={
                "run_id": RUN_ID,
                "candidate_kind": "image_build",
                "requested_loop_count": 3,
                "settings": {"max_iterations": 3, "max_candidates": 3},
                "budget_context": {"compute_budget_usd": 5.0},
                "resumed_from_checkpoint": False,
                "source_mode": "image_extract",
                "source_tree_hash": "sha256:tree",
                "parent_image_digest_hash": "sha256:parent-image",
            },
        ),
        _loop_event(
            1,
            "loop_direction_planned",
            ts=day.format(m=1),
            total_usd=0.01,
            event_doc={
                "focus_signature_hash": "sha256:focus",
                "loop_direction_plan": {"plan_hash": "sha256:planhash", "paths": []},
                "prior_attempt_count": 2,
                "source_tree_hash": "sha256:tree",
            },
        ),
        _loop_event(
            2,
            "code_edit_drafted",
            ts=day.format(m=2),
            node_id="node-1",
            total_usd=0.05,
            provider_usage=_provider_usage(1200, 400),
            event_doc=_drafted_doc(1, "evidence_quality", "path-a"),
        ),
        _loop_event(
            3,
            "candidate_build_passed",
            ts=day.format(m=3),
            node_id="node-1",
            total_usd=0.06,
            candidate_artifact_hash=artifact_1,
            elapsed_seconds=180.0,
            event_doc={
                "iteration": 1,
                "candidate_kind": "image_build",
                "candidate_model_manifest_hash": "sha256:manifest-1",
                "candidate_source_diff_hash": "sha256:source-diff-1",
                "build_doc_hash": "sha256:build-1",
            },
        ),
        _loop_event(
            4,
            "code_edit_drafted",
            ts=day.format(m=4),
            node_id="node-2",
            total_usd=0.09,
            provider_usage=_provider_usage(1100, 380),
            event_doc=_drafted_doc(2, "coverage", "path-b"),
        ),
        _loop_event(
            5,
            "candidate_build_failed",
            ts=day.format(m=5),
            node_id="node-2",
            total_usd=0.10,
            event_doc={
                "iteration": 2,
                "target_files": ["qualify/sources.py"],
                "source_diff_hash": "sha256:source-diff-2",
                "error": "docker build failed: missing import",
                "error_hash": "sha256:err",
            },
        ),
        _loop_event(
            6,
            "code_edit_drafted",
            ts=day.format(m=6),
            node_id="node-3",
            total_usd=0.13,
            provider_usage=_provider_usage(1050, 300),
            event_doc=_drafted_doc(3, "latency", "path-c"),
        ),
        _loop_event(
            7,
            "candidate_build_passed",
            ts=day.format(m=7),
            node_id="node-3",
            total_usd=0.14,
            candidate_artifact_hash=artifact_3,
            elapsed_seconds=140.0,
            event_doc={
                "iteration": 3,
                "candidate_kind": "image_build",
                "candidate_model_manifest_hash": "sha256:manifest-3",
                "candidate_source_diff_hash": "sha256:source-diff-3",
            },
        ),
        _loop_event(
            8,
            "candidate_selected",
            ts=day.format(m=8),
            node_id="node-1",
            total_usd=0.14,
            candidate_artifact_hash=artifact_1,
            event_doc={
                "candidate_index": 0,
                "iteration": 1,
                "candidate_kind": "image_build",
                "candidate_source_diff_hash": "sha256:source-diff-1",
                "redacted_summary": "second retrieval pass over cached snapshots",
            },
        ),
        _loop_event(
            9,
            "loop_completed",
            ts=day.format(m=9),
            loop_status="completed",
            total_usd=0.15,
            event_doc={
                "candidate_kind": "image_build",
                "iterations_completed": 3,
                "selected_candidate_count": 1,
                "stop_reason": "max_iterations",
            },
        ),
    ]
    candidate_rows = [
        {
            "candidate_id": CANDIDATE_1,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "island": "lead_generation",
            "parent_artifact_hash": "sha256:parent-image",
            "candidate_artifact_hash": artifact_1,
            "candidate_source_diff_hash": "sha256:source-diff-1",
        },
        {
            "candidate_id": CANDIDATE_3,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "island": "lead_generation",
            "parent_artifact_hash": "sha256:parent-image",
            "candidate_artifact_hash": artifact_3,
            "candidate_source_diff_hash": "sha256:source-diff-3",
        },
    ]
    evaluation_events = [
        {
            "event_id": str(uuid.uuid4()),
            "candidate_id": CANDIDATE_1,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "seq": 3,
            "event_type": "scored",
            "candidate_status": "scored",
            "score_bundle_id": SCORE_BUNDLE_ID,
            "event_doc": {
                "score_bundle_hash": "sha256:bundle",
                "rolling_window_hash": "sha256:window",
                "private_holdout_gate": {
                    "gate_type": "public_score_before_private_holdout",
                    "decision": "accepted",
                    "baseline_aggregate_score": 71.2,
                    "candidate_total_score": 74.4,
                    "candidate_delta_vs_daily_baseline": 3.2,
                    "public_icp_count": 5,
                },
            },
            "created_at": "2026-06-20T11:00:00Z",
        }
    ]
    promotion_events = [
        {
            "promotion_event_id": str(uuid.uuid4()),
            "candidate_id": CANDIDATE_1,
            "source_score_bundle_id": SCORE_BUNDLE_ID,
            "event_type": "promotion_passed",
            "promotion_status": "passed",
            "rolling_window_hash": "sha256:window",
            "improvement_points": 3.2,
            "threshold_points": 1.0,
            "event_doc": {"auto_commit_enabled": True},
            "created_at": "2026-06-20T11:05:00Z",
        },
        {
            "promotion_event_id": str(uuid.uuid4()),
            "candidate_id": CANDIDATE_1,
            "source_score_bundle_id": SCORE_BUNDLE_ID,
            "private_model_version_id": VERSION_ID,
            "event_type": "active_version_created",
            "promotion_status": "merged",
            "improvement_points": 3.2,
            "threshold_points": 1.0,
            "event_doc": {
                "new_model_artifact_hash": artifact_1,
                "candidate_kind": "image_build",
            },
            "created_at": "2026-06-20T11:06:00Z",
        },
    ]
    version_events = [
        {
            "event_id": str(uuid.uuid4()),
            "private_model_version_id": VERSION_ID,
            "event_type": "superseded",
            "version_status": "superseded",
            "reason": "superseded_by_research_lab_image_build_promotion",
            "event_doc": {"source_candidate_id": CANDIDATE_1},
            "created_at": "2026-06-22T09:00:00Z",
        }
    ]
    score_bundles = [
        {
            "score_bundle_id": SCORE_BUNDLE_ID,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "evaluation_epoch": 412,
            "candidate_artifact_hash": artifact_1,
            "score_bundle_hash": "sha256:bundle",
        }
    ]
    return {
        "research_loop_run_queue_current": [
            {
                "run_id": RUN_ID,
                "ticket_id": TICKET_ID,
                "current_queue_status": "completed",
                "current_status_at": "2026-06-20T10:09:00Z",
            }
        ],
        "research_loop_ticket_current": [
            {
                "ticket_id": TICKET_ID,
                "miner_hotkey": "5FfixtureHotkey",
                "island": "lead_generation",
                "brief_id": BRIEF_ID,
                "brief_sanitized_ref": "sha256:sanitized-brief",
                "requested_loop_count": 3,
            }
        ],
        "research_lab_auto_research_loop_events": loop_events,
        "research_lab_candidate_artifacts": candidate_rows,
        "research_lab_candidate_evaluation_events": evaluation_events,
        "research_lab_candidate_promotion_events": promotion_events,
        "research_lab_private_model_version_events": version_events,
        "research_evaluation_score_bundles": score_bundles,
    }


@pytest.fixture
def tables() -> dict[str, list[dict[str, Any]]]:
    return happy_path_tables()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(PROJECTOR_ENABLED_ENV, "true")
    return monkeypatch


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.setenv(PROJECTOR_ENABLED_ENV, "false")
    return monkeypatch


# ---------------------------------------------------------------------------
# Happy path: full mapping
# ---------------------------------------------------------------------------


async def test_happy_path_projects_schema_valid_rows(tables, enabled):
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors
    assert result.trajectory_id == trajectory_id_for_run(RUN_ID)

    envelopes = store.inserted[TRAJECTORIES_TABLE]
    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope["island"] == "lead_generation"
    assert envelope["funder_hotkey"] == "5FfixtureHotkey"
    assert envelope["brief_sanitized_ref"] == "sha256:sanitized-brief"
    assert envelope["champion_base"] == "sha256:parent-image"
    assert "events" not in envelope  # envelope has no events array (scripts/27)
    assert envelope["final"]["settlement"]["receipt_ref"] == f"receipt:{RECEIPT_ID}"
    assert envelope["final"]["settlement"]["crown"]["area"] == "lead_generation"
    assert envelope["final"]["settlement"]["crown"]["started_epoch"] == 412

    event_rows = store.inserted[TRAJECTORY_EVENTS_TABLE]
    types = [row["event_type"] for row in event_rows]
    assert types == [
        "LOOP_FUNDED",
        "NODE_DRAFTED",
        "NODE_EVALUATED",
        "NODE_DRAFTED",
        "NODE_EVALUATED",
        "NODE_DRAFTED",
        "NODE_EVALUATED",
        "PLATEAU_STOP",
        "L2_PROMOTED",
        "CROWNED",
        "FINALIZED",
    ]
    assert [row["seq"] for row in event_rows] == list(range(len(event_rows)))

    # The reassembled trajectory doc passes the real JSON schema.
    doc = {
        key: envelope[key]
        for key in (
            "trajectory_id",
            "schema_version",
            "brief_id",
            "island",
            "funder_hotkey",
            "brief_sanitized_ref",
            "novelty_gate",
            "engine_version",
            "champion_base",
            "created_at",
            "final",
        )
    }
    doc["events"] = [schema_event_view(row["event"]) for row in event_rows]
    assert validate_schema_record("research_trajectory.schema.json", doc) == []

    # Score mapping: candidate_delta_vs_daily_baseline is the score.
    evaluated = [row["event"] for row in event_rows if row["event_type"] == "NODE_EVALUATED"]
    by_node = {event["node_id"]: event for event in evaluated}
    assert by_node["node-1"]["status"] == "scored"
    assert by_node["node-1"]["metrics"]["proxy_score"] == pytest.approx(3.2)
    assert by_node["node-2"]["status"] == "crash"
    assert by_node["node-3"]["status"] == "timeout"  # built, never scored

    promoted = next(row["event"] for row in event_rows if row["event_type"] == "L2_PROMOTED")
    assert promoted["node_id"] == "node-1"
    assert promoted["delta"] == pytest.approx(3.2)
    assert promoted["champion_same_day_l2"] == pytest.approx(71.2)
    crowned = next(row["event"] for row in event_rows if row["event_type"] == "CROWNED")
    assert crowned["area"] == "lead_generation"
    assert crowned["epoch"] == 412

    plateau = next(row["event"] for row in event_rows if row["event_type"] == "PLATEAU_STOP")
    assert plateau["reason"] == "plateau"
    assert plateau["best_node_id"] == "node-1"

    # Ledger: one row per drafted node with keep/crash/timeout outcomes.
    ledger = store.inserted[RESULTS_LEDGER_TABLE]
    statuses = {row["node_id"]: row["status"] for row in ledger}
    assert statuses == {"node-1": "keep", "node-2": "crash", "node-3": "timeout"}
    keep_row = next(row for row in ledger if row["node_id"] == "node-1")
    assert keep_row["delta_vs_parent"] == pytest.approx(3.2)
    assert keep_row["targeted_metric"] == "candidate_delta_vs_daily_baseline"
    assert keep_row["commit"] == "sha256:source-diff-1"
    assert "evidence_quality" in keep_row["description"]
    for row in ledger:
        schema_row = {k: v for k, v in row.items() if k != "source_event_seq"}
        assert validate_schema_record("results_ledger_row.schema.json", schema_row) == []

    # Costs are distributed across canonical events and stay non-negative.
    total_cost = sum(row["cost_usd"] for row in event_rows)
    assert total_cost == pytest.approx(0.15, abs=1e-6)
    assert all(row["cost_usd"] >= 0 for row in event_rows)


async def test_corpus_source_record_passes_real_validator(tables):
    store = FakeStore(tables)
    inputs = await _load_inputs(store)
    projection = build_trajectory_projection(**inputs)
    assert projection.errors == []
    record = projection.corpus_source_record
    assert validate_trajectory_corpus_source_record(record) == []
    assert record.data_state == "production_measured"
    assert record.measured_data is True
    # Capture != training eligibility: readiness stays BLOCKED by design.
    assert record.eligible_for_training is False
    assert record.rights_verified is False
    assert record.trajectory_hash.startswith("sha256:")
    assert all(ref.startswith("results_ledger:") for ref in record.results_ledger_refs)
    assert record.receipt_refs == (f"receipt:{RECEIPT_ID}",)
    assert record.cost_ledger_refs == (f"cost_ledger:{RUN_ID}",)


# ---------------------------------------------------------------------------
# Extras are nested, never standalone rows
# ---------------------------------------------------------------------------


async def test_extra_live_events_are_nested_not_standalone(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    event_rows = store.inserted[TRAJECTORY_EVENTS_TABLE]

    # Every stored row uses one of the 12 CHECK-enforced canonical types and
    # satisfies the event JSONB consistency CHECKs.
    for row in event_rows:
        assert row["event_type"] in CANONICAL_EVENT_TYPES
        event = row["event"]
        assert event["type"] == row["event_type"]
        assert event["seq"] == row["seq"]
        assert event["cost_usd"] == row["cost_usd"]
        assert event["anchored_hash"] == row["anchored_hash"]

    # loop_direction_planned lives inside NODE_DRAFTED.event.planning_context.
    first_draft = next(r["event"] for r in event_rows if r["event_type"] == "NODE_DRAFTED")
    plan = first_draft["planning_context"]["loop_direction_plan"]
    assert plan["live_event_type"] == "loop_direction_planned"
    assert plan["loop_direction_plan"]["plan_hash"] == "sha256:planhash"

    # Build failures live inside PLATEAU_STOP.event.rejected_paths.
    plateau = next(r["event"] for r in event_rows if r["event_type"] == "PLATEAU_STOP")
    assert any(item.get("node_id") == "node-2" for item in plateau["rejected_paths"])

    # The schema view strips the nested extension keys.
    view = schema_event_view(first_draft)
    assert "planning_context" not in view
    assert "draft_context" not in view
    assert view["node_id"] == first_draft["node_id"]


# ---------------------------------------------------------------------------
# Anchored-hash round trip (hosted_loop._append_event pattern)
# ---------------------------------------------------------------------------


async def test_anchored_hash_round_trips(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    for row in store.inserted[TRAJECTORY_EVENTS_TABLE]:
        event = dict(row["event"])
        assert verify_anchored_hash(event)
        stored_hash = event.pop("anchored_hash")
        assert sha256_json(event) == stored_hash
        assert stored_hash == row["anchored_hash"]


def test_hash_pattern_matches_hosted_loop_append_event():
    """Events hashed by hosted_loop's own _append_event verify with ours."""
    from research_lab.hosted_loop import _append_event as hosted_append_event

    events: list[dict[str, Any]] = []
    hosted_append_event(
        events,
        "2026-06-20T10:00:00Z",
        "LOOP_FUNDED",
        0.25,
        {"loop_n": 3, "balance_before": 5.0},
    )
    hosted_append_event(
        events,
        "2026-06-20T10:00:00Z",
        "PLATEAU_STOP",
        0.0,
        {"reason": "plateau", "best_node_id": "node-1"},
    )
    for event in events:
        assert verify_anchored_hash(event)


# ---------------------------------------------------------------------------
# Idempotency / dry-run / flag gating
# ---------------------------------------------------------------------------


async def test_rerun_is_idempotent(tables, enabled):
    store = FakeStore(tables)
    first = await project_run(RUN_ID, store=store, dry_run=False)
    assert first.status == "projected"
    writes_after_first = store.write_count()
    second = await project_run(RUN_ID, store=store, dry_run=False)
    assert second.status == "skipped_existing"
    assert store.write_count() == writes_after_first


async def test_dry_run_writes_nothing(tables, enabled):
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=True)
    assert result.status == "dry_run"
    assert result.event_count > 0
    assert result.ledger_row_count == 3
    assert store.write_count() == 0


async def test_disabled_flag_blocks_writes(tables, disabled):
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "skipped_disabled"
    assert store.write_count() == 0
    # Dry-run stays available while the projector is dormant.
    dry = await project_run(RUN_ID, store=store, dry_run=True)
    assert dry.status == "dry_run"
    assert store.write_count() == 0


# ---------------------------------------------------------------------------
# Protected-material sanitization
# ---------------------------------------------------------------------------


async def test_poisoned_event_is_pointerized_and_still_validates(tables, enabled):
    poisoned = copy.deepcopy(tables)
    drafted = poisoned["research_lab_auto_research_loop_events"][2]
    assert drafted["event_type"] == "code_edit_drafted"
    drafted["event_doc"]["llm_response"] = "SECRET raw model output that must not leak"
    drafted["event_doc"]["hypothesis"]["mechanism"] = (
        "reuse the llm response and cached page content for retrieval"
    )
    store = FakeStore(poisoned)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors

    event_rows = store.inserted[TRAJECTORY_EVENTS_TABLE]
    for row in event_rows:
        assert find_protected_material(row["event"]) == set()
    for row in store.inserted[RESULTS_LEDGER_TABLE]:
        assert find_protected_material(row) == set()
    assert find_protected_material(store.inserted[TRAJECTORIES_TABLE][0]) == set()

    draft_event = next(r["event"] for r in event_rows if r["event"].get("node_id") == "node-1")
    source_doc = draft_event["draft_context"]["source_event_doc"]
    assert "llm_response" not in source_doc
    assert source_doc["llm_response_sha256_ref"].startswith("sha256:")
    assert "SECRET" not in str(source_doc)
    assert "[protected-material-redacted]" in draft_event["hypothesis"]["mechanism"]

    # The record still passes the real schema after sanitization.
    envelope = store.inserted[TRAJECTORIES_TABLE][0]
    doc = {k: v for k, v in envelope.items()}
    doc["events"] = [schema_event_view(row["event"]) for row in event_rows]
    assert validate_schema_record("research_trajectory.schema.json", doc) == []


def test_sanitize_capture_payload_unit():
    payload = {
        "prompt": "you are a helpful assistant",
        "page_content": {"html": "<div>x</div>"},
        "note": "compare against the LLM Response later",
        "unified_diff_hash": "sha256:abc",
        "nested": [{"llm_request": "raw"}],
    }
    sanitized = sanitize_capture_payload(payload)
    assert find_protected_material(sanitized) == set()
    assert sanitized["prompt_sha256_ref"].startswith("sha256:")
    assert sanitized["page_content_sha256_ref"].startswith("sha256:")
    assert "[protected-material-redacted]" in sanitized["note"]
    assert sanitized["unified_diff_hash"] == "sha256:abc"
    assert sanitized["nested"][0]["llm_request_sha256_ref"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Batch projection / backfill behaviour
# ---------------------------------------------------------------------------


async def test_batch_skips_projected_and_survives_poison_runs(tables, enabled):
    broken_run = "99999999-9999-4999-8999-999999999999"
    tables = copy.deepcopy(tables)
    tables["research_loop_run_queue_current"].append(
        {
            "run_id": broken_run,
            "ticket_id": "88888888-8888-4888-8888-888888888888",
            "current_queue_status": "failed",
            "current_status_at": "2026-06-21T00:00:00Z",
        }
    )
    # broken run: no ticket, no loop events -> must log-and-skip, not raise.
    store = FakeStore(tables)
    results = await project_completed_runs(batch_size=10, dry_run=False, store=store)
    by_run = {r.run_id: r for r in results}
    assert by_run[RUN_ID].status == "projected"
    assert by_run[broken_run].status == "skipped_incomplete"

    # Second pass: the projected run is skipped before project_run is called,
    # the broken one is retried and skipped again; nothing new is written.
    writes = store.write_count()
    results2 = await project_completed_runs(batch_size=10, dry_run=False, store=store)
    assert store.write_count() == writes
    assert all(r.status != "projected" for r in results2)


async def test_batch_dry_run_writes_nothing(tables, disabled):
    store = FakeStore(tables)
    results = await project_completed_runs(batch_size=10, dry_run=True, store=store)
    assert [r.status for r in results] == ["dry_run"]
    assert store.write_count() == 0


# ---------------------------------------------------------------------------
# Reason mapping and reflection tolerance
# ---------------------------------------------------------------------------


async def test_budget_exhausted_maps_to_balance_exhausted(tables):
    tables = copy.deepcopy(tables)
    terminal = tables["research_lab_auto_research_loop_events"][-1]
    terminal["event_type"] = "loop_failed"
    terminal["loop_status"] = "failed"
    terminal["event_doc"]["stop_reason"] = "compute_budget_exhausted_after_code_edit"
    tables["research_loop_run_queue_current"][0]["current_queue_status"] = "failed"
    store = FakeStore(tables)
    inputs = await _load_inputs(store)
    projection = build_trajectory_projection(**inputs)
    assert projection.errors == []
    plateau = next(
        row["event"] for row in projection.event_rows if row["event_type"] == "PLATEAU_STOP"
    )
    assert plateau["reason"] == "balance_exhausted"


async def test_typed_lane_reflection_maps_to_node_reflected(tables):
    tables = copy.deepcopy(tables)
    events = tables["research_lab_auto_research_loop_events"]
    events.insert(
        3,
        _loop_event(
            99,
            "reflection_recorded",
            ts="2026-06-20T10:02:30Z",
            total_usd=0.055,
            event_doc={
                "iteration": 1,
                "reflection": {
                    "worked": "retrieval pass compiled",
                    "failed": "coverage unchanged",
                    "why": "cache was cold",
                    "next_question": "warm the snapshot cache first?",
                },
            },
        ),
    )
    # keep live seqs strictly ordered for the walk
    for index, row in enumerate(events):
        row["seq"] = index
    store = FakeStore(tables)
    inputs = await _load_inputs(store)
    projection = build_trajectory_projection(**inputs)
    assert projection.errors == []
    reflected = [
        row["event"] for row in projection.event_rows if row["event_type"] == "NODE_REFLECTED"
    ]
    assert len(reflected) == 1
    assert reflected[0]["node_id"] == "node-1"
    assert reflected[0]["lesson"]["why"] == "cache was cold"
    assert reflected[0]["lesson_provenance"]["champion_base"] == "sha256:parent-image"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _load_inputs(store: FakeStore) -> dict[str, Any]:
    from gateway.research_lab.trajectory_projector import load_projection_inputs

    return await load_projection_inputs(RUN_ID, store)


# ---------------------------------------------------------------------------
# P14: negative examples as first-class rows (trajectoryimprovements.md)
# ---------------------------------------------------------------------------


async def test_promotion_rejection_projects_reverted_event(enabled):
    tables = happy_path_tables()
    # Replace the passed promotion with a below-threshold rejection.
    tables["research_lab_candidate_promotion_events"] = [
        {
            "promotion_event_id": str(uuid.uuid4()),
            "candidate_id": CANDIDATE_1,
            "source_score_bundle_id": SCORE_BUNDLE_ID,
            "event_type": "below_threshold",
            "promotion_status": "rejected",
            "rolling_window_hash": "sha256:window",
            "improvement_points": 0.4,
            "threshold_points": 1.0,
            "event_doc": {},
            "created_at": "2026-06-20T11:05:00Z",
        }
    ]
    tables["research_lab_private_model_version_events"] = []
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors

    event_rows = store.inserted[TRAJECTORY_EVENTS_TABLE]
    types = [row["event_type"] for row in event_rows]
    assert "REVERTED" in types
    assert "L2_PROMOTED" not in types
    reverted = next(row["event"] for row in event_rows if row["event_type"] == "REVERTED")
    assert reverted["reason"] == "below_threshold"
    assert reverted["node_id"] == "node-1"
    context = reverted["rejection_context"]
    assert context["rejection_kind"] == "promotion_gate"
    assert context["improvement_points"] == pytest.approx(0.4)
    assert context["threshold_points"] == pytest.approx(1.0)
    # The stripped schema view still validates (context is an extension key).
    assert schema_event_view(reverted)["type"] == "REVERTED"


async def test_scorer_crash_yields_crash_trace_row(enabled):
    tables = happy_path_tables()
    # The candidate's scoring crashed: a failed evaluation event, no bundle.
    tables["research_lab_candidate_evaluation_events"] = [
        {
            "event_id": str(uuid.uuid4()),
            "candidate_id": CANDIDATE_1,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "seq": 3,
            "event_type": "failed",
            "candidate_status": "failed",
            "event_doc": {"error": "scoring worker crashed mid-evaluation"},
            "created_at": "2026-06-20T11:00:00Z",
        }
    ]
    tables["research_evaluation_score_bundles"] = []
    tables["research_lab_candidate_promotion_events"] = []
    tables["research_lab_private_model_version_events"] = []
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors

    from gateway.research_lab.trajectory_projector import (
        EXECUTION_TRACES_TABLE,
        execution_trace_id_for_node,
    )

    trace_rows = {
        row["run_id"]: row for row in store.inserted[EXECUTION_TRACES_TABLE]
    }
    node1 = trace_rows[execution_trace_id_for_node(RUN_ID, "node-1")]
    assert node1["status"] == "crash"
    assert node1["score_bundle_ref"] == "score_bundle:unavailable"
    assert node1["trace_doc"]["scoring_crashed"] is True
    assert node1["trace_doc"]["scoring_failure_event_type"] == "failed"


async def test_score_bundle_projects_first_class_champion_and_baseline_traces(tables, enabled):
    tables = copy.deepcopy(tables)
    gate = tables["research_lab_candidate_evaluation_events"][0]["event_doc"][
        "private_holdout_gate"
    ]
    gate["baseline_benchmark_bundle_id"] = "benchmark-20260709-a0"
    gate["baseline_benchmark_hash"] = sha256_json({"benchmark": "20260709-a0"})

    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors

    from gateway.research_lab.trajectory_projector import (
        EXECUTION_TRACES_TABLE,
        execution_trace_id_for_baseline_arm_side,
        execution_trace_id_for_champion_side,
        execution_trace_id_for_node,
    )

    trace_rows = {
        row["run_id"]: row for row in store.inserted[EXECUTION_TRACES_TABLE]
    }
    candidate_trace_id = execution_trace_id_for_node(RUN_ID, "node-1")
    champion_trace_id = execution_trace_id_for_champion_side(RUN_ID, SCORE_BUNDLE_ID)
    baseline_ref = "benchmark_bundle:benchmark-20260709-a0"
    baseline_trace_id = execution_trace_id_for_baseline_arm_side(
        RUN_ID, baseline_ref, SCORE_BUNDLE_ID
    )

    assert candidate_trace_id in trace_rows
    assert trace_rows[candidate_trace_id]["role"] == "candidate"
    assert trace_rows[champion_trace_id]["role"] == "champion"
    assert trace_rows[champion_trace_id]["score_bundle_ref"] == SCORE_BUNDLE_ID
    assert trace_rows[champion_trace_id]["trace_doc"]["trace_kind"] == (
        "champion_comparison_side"
    )
    assert trace_rows[baseline_trace_id]["role"] == "baseline_arm"
    assert trace_rows[baseline_trace_id]["score_bundle_ref"] == baseline_ref
    assert trace_rows[baseline_trace_id]["trace_doc"]["trace_kind"] == (
        "baseline_arm_comparison_side"
    )
    assert trace_rows[baseline_trace_id]["trace_doc"]["candidate_score_bundle_ref"] == (
        SCORE_BUNDLE_ID
    )
