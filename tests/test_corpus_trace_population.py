"""Tests for execution_traces / evidence_bundles population (item 5.5).

The projector must populate the two empty v5 corpus tables with POINTER rows:

* ``execution_traces``  -- engine raw-LLM-trace pointers (item 5.1, in loop
  events' provider_usage) and in-container sourcing-model trace pointers
  (item 5.3, in score-bundle per-ICP rows), grouped per run + per node;
* ``evidence_bundles``  -- per scored candidate node: score-bundle pointer +
  per-ICP evidence summary (numbers and refs only) + holdout gate refs.

Row shapes are checked against the REAL scripts/27 SQL: the CHECK enums are
re-parsed from the file so a schema drift fails the test, not production.
Linkage is forward-only (events are append-only + hash-anchored), so the
``--traces-backfill`` path must add rows to already-projected runs WITHOUT
touching their events.
"""

from __future__ import annotations

import copy
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from gateway.research_lab.trajectory_projector import (
    EVIDENCE_BUNDLES_TABLE,
    EVIDENCE_RETENTION_CLASSES,
    EVIDENCE_VERIFICATION_STATES,
    EXECUTION_TRACE_ROLES,
    EXECUTION_TRACE_RUNGS,
    EXECUTION_TRACE_STATUSES,
    EXECUTION_TRACES_TABLE,
    PROJECTOR_ENABLED_ENV,
    RESULTS_LEDGER_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_EVENTS_TABLE,
    backfill_corpus_trace_rows,
    backfill_run_corpus_trace_rows,
    build_trajectory_projection,
    evidence_bundle_id_for_node,
    evidence_bundle_id_for_score_bundle,
    execution_trace_id_for_node,
    execution_trace_id_for_run,
    execution_trace_id_for_score_bundle,
    find_protected_material,
    load_projection_inputs,
    project_run,
    schema_event_view,
    trajectory_id_for_run,
    verify_anchored_hash,
)
from research_lab.canonical import sha256_json
from research_lab.trajectory_corpus import validate_trajectory_corpus_source_record


SCRIPTS_27 = Path(__file__).resolve().parents[1] / "scripts" / "27-research-lab-v5-schemas.sql"

RUN_ID = "aaaaaaaa-1111-4111-8111-111111111111"
TICKET_ID = "aaaaaaaa-2222-4222-8222-222222222222"
BRIEF_ID = "aaaaaaaa-3333-4333-8333-333333333333"
RECEIPT_ID = "aaaaaaaa-4444-4444-8444-444444444444"
CANDIDATE_1 = "aaaaaaaa-5555-4555-8555-555555555551"
SCORE_BUNDLE_ID = "score_bundle:" + "ab" * 32

RUN_ID_2 = "bbbbbbbb-1111-4111-8111-111111111111"
TICKET_ID_2 = "bbbbbbbb-2222-4222-8222-222222222222"
CANDIDATE_2 = "bbbbbbbb-5555-4555-8555-555555555551"
SCORE_BUNDLE_ID_2 = "score_bundle:" + "cd" * 32


# ---------------------------------------------------------------------------
# Fake store (same query surface as tests/test_trajectory_projector.py)
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
        pk = {
            TRAJECTORIES_TABLE: "trajectory_id",
            EXECUTION_TRACES_TABLE: "run_id",
            EVIDENCE_BUNDLES_TABLE: "bundle_id",
            RESULTS_LEDGER_TABLE: "ledger_row_id",
        }.get(table)
        if pk and any(r.get(pk) == row.get(pk) for r in stored):
            raise RuntimeError("duplicate key value violates unique constraint")
        if table == TRAJECTORY_EVENTS_TABLE:
            if any(
                r.get("trajectory_id") == row.get("trajectory_id")
                and r.get("seq") == row.get("seq")
                for r in stored
            ):
                raise RuntimeError("duplicate key value violates unique constraint")
        if table == EVIDENCE_BUNDLES_TABLE:
            if any(r.get("bundle_hash") == row.get("bundle_hash") for r in stored):
                raise RuntimeError("duplicate key value violates unique constraint")
        stored.append(dict(row))
        self.inserted.setdefault(table, []).append(dict(row))
        return dict(row)

    def write_count(self, table: str | None = None) -> int:
        if table is not None:
            return len(self.inserted.get(table, []))
        return sum(len(rows) for rows in self.inserted.values())


# ---------------------------------------------------------------------------
# Fixtures: live rows with item-5.1 raw-trace pointers and item-5.3
# in-container pointers, modelled on the real emitter shapes
# (worker.py provider_usage["raw_trace_ref"] / ["retry_attempts"], and
# score_bundle_doc.aggregates.per_icp_results rows).
# ---------------------------------------------------------------------------


def _cost_ledger(total_usd: float, stage: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "running",
        "stage": stage,
        "total_usd": round(total_usd, 6),
        "actual_openrouter_cost_usd": round(total_usd, 6),
    }


def _raw_trace_ref(tag: str) -> dict[str, str]:
    return {
        "s3_ref": f"s3://trace-bucket/research-lab/raw-traces/{tag}.json.enc",
        "sha256": sha256_json({"raw_trace": tag}),
    }


def _provider_usage_with_trace(
    prompt: int, completion: int, tag: str, *, retry_tag: str | None = None
) -> list[dict[str, Any]]:
    usage: dict[str, Any] = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cost_usd": 0.01,
        "raw_trace_ref": _raw_trace_ref(tag),
    }
    if retry_tag:
        usage["retry_attempts"] = [
            {"outcome": "length", "raw_trace_ref": _raw_trace_ref(retry_tag)}
        ]
    return [usage]


def _loop_event(
    run_id: str,
    ticket_id: str,
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
    return {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{seq}")),
        "schema_version": "1.0",
        "run_id": run_id,
        "ticket_id": ticket_id,
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


def _per_icp_row(index: int, *, run_tag: str, incontainer: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {
        "icp_ref": f"icp-{run_tag}-{index}",
        "icp_hash": sha256_json({"icp": f"{run_tag}-{index}"}),
        "status": "completed",
        "hard_failure": False,
        "base_company_scores": [],
        "candidate_company_scores": [71.0, 68.5, 62.0],
        "base_per_icp_score": 0.0,
        "candidate_per_icp_score": 67.2,
        "delta_vs_base": 67.2,
        "failure_reason": "",
    }
    if incontainer:
        row["incontainer_trace_ref"] = (
            f"s3://trace-bucket/research-lab/incontainer/{run_tag}/{index}.json.enc"
        )
        row["incontainer_trace_sha256"] = sha256_json(
            {"incontainer": f"{run_tag}-{index}"}
        )
        row["incontainer_trace_call_count"] = 4 + index
        # P12 passthrough: the per-ICP scorer judgment pointer rides the row.
        row["scorer_trace_ref"] = (
            f"s3://trace-bucket/research-lab/scorer-traces/{run_tag}/{index}.json"
        )
        row["scorer_trace_sha256"] = sha256_json({"scorer": f"{run_tag}-{index}"})
    return row


def _score_bundle_doc(run_id: str, run_tag: str, artifact: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "bundle_type": "research_evaluation_score_bundle",
        "run_id": run_id,
        "icp_set_hash": sha256_json({"icp_set": run_tag}),
        "scoring_version": "scoring:v9",
        "evaluator_version": "evaluator:v9",
        "candidate_artifact_hash": artifact,
        "score_bundle_hash": sha256_json({"bundle": run_tag}),
        "aggregates": {
            "icp_count": 3,
            "successful_icp_count": 3,
            "hard_failure_count": 0,
            "base_score": 0.0,
            "candidate_score": 67.2,
            "mean_delta": 3.2,
            "delta_lcb": 1.1,
            "total_cost_usd": 0.42,
            "per_icp_results": [
                _per_icp_row(1, run_tag=run_tag),
                _per_icp_row(2, run_tag=run_tag),
                _per_icp_row(3, run_tag=run_tag, incontainer=False),
            ],
        },
        "private_holdout_gate": {
            "gate_type": "public_score_before_private_holdout",
            "decision": "accepted",
            "baseline_benchmark_bundle_id": f"benchmark-{run_tag}",
            "baseline_benchmark_hash": sha256_json({"benchmark": run_tag}),
            "baseline_aggregate_score": 71.2,
            "candidate_total_score": 74.4,
            "candidate_delta_vs_daily_baseline": 3.2,
            "public_icp_count": 2,
            "private_holdout_icp_count": 1,
            "private_holdout_evaluated": True,
            "provider_excluded_icp_ids": [f"icp-{run_tag}-9"],
        },
        "provider_excluded_icp_ids": [f"icp-{run_tag}-9"],
    }


def build_run_tables(
    *,
    run_id: str = RUN_ID,
    ticket_id: str = TICKET_ID,
    candidate_id: str = CANDIDATE_1,
    score_bundle_id: str = SCORE_BUNDLE_ID,
    run_tag: str = "run1",
    with_raw_traces: bool = True,
    with_bundle_doc: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """One completed run: planner + 2 nodes (one scored, one build-crashed)."""
    day = "2026-06-28T10:{m:02d}:00Z"
    artifact = f"sha256:candidate-artifact-{run_tag}"
    planner_usage = (
        _provider_usage_with_trace(900, 250, f"{run_tag}/0001-loop_direction_plan")
        if with_raw_traces
        else []
    )
    draft_usage = (
        _provider_usage_with_trace(
            1200,
            400,
            f"{run_tag}/0003-code_edit_draft",
            retry_tag=f"{run_tag}/0002-code_edit_draft",
        )
        if with_raw_traces
        else []
    )
    draft2_usage = (
        _provider_usage_with_trace(1100, 380, f"{run_tag}/0004-code_edit_draft")
        if with_raw_traces
        else []
    )
    loop_events = [
        _loop_event(
            run_id,
            ticket_id,
            0,
            "loop_started",
            ts=day.format(m=0),
            event_doc={
                "run_id": run_id,
                "candidate_kind": "image_build",
                "requested_loop_count": 2,
                "settings": {"max_iterations": 2},
                "budget_context": {"compute_budget_usd": 5.0},
                "source_tree_hash": "sha256:tree",
                "parent_image_digest_hash": "sha256:parent-image",
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            1,
            "loop_direction_planned",
            ts=day.format(m=1),
            total_usd=0.01,
            provider_usage=planner_usage,
            event_doc={"loop_direction_plan": {"plan_hash": "sha256:planhash"}},
        ),
        _loop_event(
            run_id,
            ticket_id,
            2,
            "code_edit_drafted",
            ts=day.format(m=2),
            node_id="node-1",
            total_usd=0.05,
            provider_usage=draft_usage,
            event_doc={
                "iteration": 1,
                "lane": "evidence_quality",
                "plan_path_id": "path-a",
                "target_files": ["qualify/sources.py"],
                "unified_diff_hash": sha256_json({"diff": f"{run_tag}-1"}),
                "hypothesis": {
                    "failure_mode": "sparse evidence",
                    "mechanism": "second retrieval pass",
                    "predicted_delta": 0.5,
                },
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            3,
            "candidate_build_passed",
            ts=day.format(m=3),
            node_id="node-1",
            total_usd=0.06,
            candidate_artifact_hash=artifact,
            elapsed_seconds=120.0,
            event_doc={
                "iteration": 1,
                "candidate_kind": "image_build",
                "candidate_source_diff_hash": f"sha256:source-diff-{run_tag}",
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            4,
            "reflection_recorded",
            ts=day.format(m=4),
            node_id="node-1",
            total_usd=0.06,
            event_doc={
                "iteration": 1,
                "reflection": {
                    "worked": "retrieval pass compiled",
                    "failed": "coverage unchanged",
                    "why": "cache was cold",
                    "next_question": "warm the cache first?",
                },
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            5,
            "code_edit_drafted",
            ts=day.format(m=5),
            node_id="node-2",
            total_usd=0.09,
            provider_usage=draft2_usage,
            event_doc={
                "iteration": 2,
                "lane": "coverage",
                "plan_path_id": "path-b",
                "unified_diff_hash": sha256_json({"diff": f"{run_tag}-2"}),
                "hypothesis": {
                    "failure_mode": "low coverage",
                    "mechanism": "broaden query",
                    "predicted_delta": 0.2,
                },
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            6,
            "candidate_build_failed",
            ts=day.format(m=6),
            node_id="node-2",
            total_usd=0.10,
            event_doc={
                "iteration": 2,
                "error": "docker build failed: missing import",
                "error_hash": "sha256:err",
                # P4: build-failure diagnostic artifact pointer (present only
                # alongside raw traces so no-pointer historical fixtures stay
                # genuinely pointer-free).
                **(
                    {
                        "diagnostic_artifact_uri": (
                            f"s3://artifact-bucket/research-lab/candidates/{run_id}/"
                            f"diagnostics/002-node-2-candidate_build_failed.json"
                        ),
                        "diagnostic_artifact_hash": sha256_json(
                            {"diagnostic": f"{run_tag}-2"}
                        ),
                        "target_files": ["qualify/sources.py"],
                    }
                    if with_raw_traces
                    else {}
                ),
            },
        ),
        _loop_event(
            run_id,
            ticket_id,
            7,
            "candidate_selected",
            ts=day.format(m=7),
            node_id="node-1",
            total_usd=0.10,
            candidate_artifact_hash=artifact,
            event_doc={"iteration": 1},
        ),
        _loop_event(
            run_id,
            ticket_id,
            8,
            "loop_completed",
            ts=day.format(m=8),
            loop_status="completed",
            total_usd=0.11,
            event_doc={
                "iterations_completed": 2,
                "selected_candidate_count": 1,
                "stop_reason": "max_iterations",
            },
        ),
    ]
    bundle_row: dict[str, Any] = {
        "score_bundle_id": score_bundle_id,
        "run_id": run_id,
        "ticket_id": ticket_id,
        "evaluation_epoch": 412,
        "candidate_artifact_hash": artifact,
        "icp_set_hash": sha256_json({"icp_set": run_tag}),
        "scoring_version": "scoring:v9",
        "evaluator_version": "evaluator:v9",
        "score_bundle_hash": sha256_json({"bundle": run_tag}),
        "created_at": "2026-06-28T11:00:00Z",
    }
    if with_bundle_doc:
        bundle_row["score_bundle_doc"] = _score_bundle_doc(run_id, run_tag, artifact)
    return {
        "research_loop_run_queue_current": [
            {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "current_queue_status": "completed",
                "current_status_at": day.format(m=8),
            }
        ],
        "research_loop_ticket_current": [
            {
                "ticket_id": ticket_id,
                "miner_hotkey": "5FfixtureHotkey",
                "island": "lead_generation",
                "brief_id": BRIEF_ID,
                "brief_sanitized_ref": "sha256:sanitized-brief",
                "requested_loop_count": 2,
            }
        ],
        "research_lab_auto_research_loop_events": loop_events,
        "research_lab_candidate_artifacts": [
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "ticket_id": ticket_id,
                "island": "lead_generation",
                "parent_artifact_hash": "sha256:parent-image",
                "candidate_artifact_hash": artifact,
                "candidate_source_diff_hash": f"sha256:source-diff-{run_tag}",
            }
        ],
        "research_lab_candidate_evaluation_events": [
            {
                "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:scored")),
                "candidate_id": candidate_id,
                "run_id": run_id,
                "ticket_id": ticket_id,
                "seq": 3,
                "event_type": "scored",
                "candidate_status": "scored",
                "score_bundle_id": score_bundle_id,
                "event_doc": {
                    "score_bundle_hash": sha256_json({"bundle": run_tag}),
                    "private_holdout_gate": {
                        "gate_type": "public_score_before_private_holdout",
                        "decision": "accepted",
                        "baseline_aggregate_score": 71.2,
                        "candidate_total_score": 74.4,
                        "candidate_delta_vs_daily_baseline": 3.2,
                        "public_icp_count": 2,
                        "private_holdout_icp_count": 1,
                    },
                },
                "created_at": "2026-06-28T11:00:00Z",
            }
        ],
        "research_lab_candidate_promotion_events": [],
        "research_lab_private_model_version_events": [],
        "research_evaluation_score_bundles": [bundle_row],
    }


def merge_tables(*table_sets: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for tables in table_sets:
        for name, rows in tables.items():
            merged.setdefault(name, []).extend(copy.deepcopy(rows))
    return merged


@pytest.fixture
def tables() -> dict[str, list[dict[str, Any]]]:
    return build_run_tables()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(PROJECTOR_ENABLED_ENV, "true")
    return monkeypatch


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.delenv(PROJECTOR_ENABLED_ENV, raising=False)
    return monkeypatch


NODE1_TRACE_ID = execution_trace_id_for_node(RUN_ID, "node-1")
ENGINE_TRACE_ID = execution_trace_id_for_run(RUN_ID)
EVIDENCE_ID = evidence_bundle_id_for_node(RUN_ID, "node-1", SCORE_BUNDLE_ID)
ORPHAN_SCORE_BUNDLE_ID = "score_bundle:" + "ef" * 32
ORPHAN_TRACE_ID = execution_trace_id_for_score_bundle(RUN_ID, ORPHAN_SCORE_BUNDLE_ID)
ORPHAN_EVIDENCE_ID = evidence_bundle_id_for_score_bundle(
    RUN_ID, ORPHAN_SCORE_BUNDLE_ID
)


# ---------------------------------------------------------------------------
# scripts/27 CHECK-shape conformance (enums parsed from the real SQL)
# ---------------------------------------------------------------------------


def _sql_table_block(sql: str, table: str) -> str:
    start = sql.index(f"CREATE TABLE IF NOT EXISTS public.{table}")
    end = sql.index("CREATE TABLE", start + 1)
    return sql[start:end]

def _sql_enum(block: str, column: str) -> set[str]:
    match = re.search(
        rf"{column}\s+TEXT\s+NOT NULL CHECK \(\s*{column} IN \(([^)]*)\)",
        block,
        re.DOTALL,
    )
    assert match, f"could not parse CHECK enum for {column}"
    return {item.strip().strip("'") for item in match.group(1).split(",") if item.strip()}


def test_module_enums_match_scripts_27_sql():
    sql = SCRIPTS_27.read_text(encoding="utf-8")
    traces = _sql_table_block(sql, "execution_traces")
    assert _sql_enum(traces, "role") == set(EXECUTION_TRACE_ROLES)
    assert _sql_enum(traces, "rung") == set(EXECUTION_TRACE_RUNGS)
    assert _sql_enum(traces, "status") == set(EXECUTION_TRACE_STATUSES)
    evidence = _sql_table_block(sql, "evidence_bundles")
    assert _sql_enum(evidence, "retention_class") == set(EVIDENCE_RETENTION_CLASSES)
    assert _sql_enum(evidence, "verification_state") == set(EVIDENCE_VERIFICATION_STATES)
    # snapshots must be a non-empty array per the CHECK
    assert "jsonb_array_length(snapshots) > 0" in evidence


def _assert_trace_row_matches_sql(row: Mapping[str, Any]) -> None:
    sql = SCRIPTS_27.read_text(encoding="utf-8")
    block = _sql_table_block(sql, "execution_traces")
    assert str(uuid.UUID(str(row["run_id"])))  # UUID PK
    assert row["schema_version"] == "1.0"
    assert row["role"] in _sql_enum(block, "role")
    assert row["rung"] in _sql_enum(block, "rung")
    assert row["status"] in _sql_enum(block, "status")
    for not_null in ("artifact_hash", "icp_set_hash", "outputs_ref", "score_bundle_ref"):
        assert str(row[not_null])
    assert isinstance(row["calls"], list)
    assert isinstance(row["evidence_bundles"], list)
    assert isinstance(row["judge_verdicts"], list)
    assert isinstance(row["eval_version"], dict)
    assert isinstance(row["cost_ledger"], dict)
    assert row["trace_doc"] is None or isinstance(row["trace_doc"], dict)
    assert row["lane_id"] is None or str(uuid.UUID(str(row["lane_id"])))


def _assert_evidence_row_matches_sql(row: Mapping[str, Any]) -> None:
    sql = SCRIPTS_27.read_text(encoding="utf-8")
    block = _sql_table_block(sql, "evidence_bundles")
    assert str(uuid.UUID(str(row["bundle_id"])))
    assert row["schema_version"] == "1.0"
    assert row["run_id"] is None or str(uuid.UUID(str(row["run_id"])))
    assert str(row["artifact_hash"])
    assert row["retention_class"] in _sql_enum(block, "retention_class")
    assert row["verification_state"] in _sql_enum(block, "verification_state")
    assert str(row["bundle_hash"]).startswith("sha256:")
    assert isinstance(row["snapshots"], list) and len(row["snapshots"]) > 0
    assert row["bundle_doc"] is None or isinstance(row["bundle_doc"], dict)


# ---------------------------------------------------------------------------
# Happy path: rows written, pointer aggregation, linkage
# ---------------------------------------------------------------------------


async def test_projection_writes_trace_and_evidence_rows(tables, enabled):
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors
    # engine row + node-1 row + node-2 row (node-2 has a draft raw trace)
    assert result.execution_trace_count == 3
    assert result.evidence_bundle_count == 1

    trace_rows = store.inserted[EXECUTION_TRACES_TABLE]
    for row in trace_rows:
        _assert_trace_row_matches_sql(row)
    evidence_rows = store.inserted[EVIDENCE_BUNDLES_TABLE]
    for row in evidence_rows:
        _assert_evidence_row_matches_sql(row)

    by_id = {row["run_id"]: row for row in trace_rows}
    assert set(by_id) == {
        ENGINE_TRACE_ID,
        NODE1_TRACE_ID,
        execution_trace_id_for_node(RUN_ID, "node-2"),
    }
    assert evidence_rows[0]["bundle_id"] == EVIDENCE_ID
    assert evidence_rows[0]["run_id"] == RUN_ID


async def test_engine_and_node_pointer_aggregation(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    by_id = {row["run_id"]: row for row in store.inserted[EXECUTION_TRACES_TABLE]}

    # Engine row: only the non-node-attributed planner trace.
    engine = by_id[ENGINE_TRACE_ID]
    assert engine["status"] == "completed"
    assert [call["stage"] for call in engine["calls"]] == ["loop_direction_planned"]
    # P11 (amended): the planner is a fixed-order engine stage — code chooses
    # the next call, so the derived emitter is "code" (axis-B), not the old
    # hardcoded "model".
    assert engine["calls"][0]["call_emitter"] == "code"
    assert engine["calls"][0]["purpose"] == "plan_next_iteration"
    assert engine["calls"][0]["s3_ref"].startswith("s3://trace-bucket/")
    assert engine["calls"][0]["sha256"].startswith("sha256:")
    assert engine["calls"][0]["node_id"] is None
    assert engine["outputs_ref"] == f"trajectory:{trajectory_id_for_run(RUN_ID)}"
    assert engine["evidence_bundles"] == []
    assert engine["trace_doc"]["trace_kind"] == "engine_loop"
    assert engine["trace_doc"]["lab_run_ref"] == f"research_loop_run:{RUN_ID}"

    # Node-1 row: draft trace + its failed-length retry attempt, grouped by
    # stage/iteration/attempt, plus the in-container per-ICP pointers.
    node1 = by_id[NODE1_TRACE_ID]
    assert node1["status"] == "completed"  # scored
    engine_calls = [c for c in node1["calls"] if c["call_kind"] == "engine_raw_trace"]
    assert len(engine_calls) == 2  # response + retry attempt pointer
    assert {c["stage"] for c in engine_calls} == {"code_edit_drafted"}
    assert {c["iteration"] for c in engine_calls} == {1}
    assert sorted(c["attempt"] for c in engine_calls) == [1, 2]
    incontainer = [c for c in node1["calls"] if c["call_kind"] == "incontainer_trace"]
    assert len(incontainer) == 2  # third per-ICP row has no incontainer ref
    assert all(c["call_emitter"] == "code" for c in incontainer)
    assert all(c["s3_ref"].startswith("s3://trace-bucket/research-lab/incontainer/") for c in incontainer)
    assert all(c["sha256"].startswith("sha256:") for c in incontainer)
    assert sorted(c["call_count"] for c in incontainer) == [5, 6]
    assert node1["score_bundle_ref"] == SCORE_BUNDLE_ID
    assert node1["outputs_ref"] == SCORE_BUNDLE_ID
    assert node1["evidence_bundles"] == [f"evidence_bundle:{EVIDENCE_ID}"]
    assert node1["icp_set_hash"] == sha256_json({"icp_set": "run1"})
    assert node1["eval_version"]["scoring_version"] == "scoring:v9"
    # P2: per-ICP scorer judgment pointers project as first-class verdicts.
    verdicts = node1["judge_verdicts"]
    assert len(verdicts) == 2  # third per-ICP row has no scorer trace ref
    assert all(v["verdict_kind"] == "scorer_judgment_trace" for v in verdicts)
    assert all(
        v["s3_ref"].startswith("s3://trace-bucket/research-lab/scorer-traces/")
        for v in verdicts
    )
    assert all(v["teacher_model_flag"] is True for v in verdicts)
    assert all(v["storage_state"] == "raw_trace_ref" for v in verdicts)
    assert node1["trace_doc"]["reflection_count"] == 1
    assert node1["trace_doc"]["judge_verdict_count"] == 2

    # Node-2 (build crashed, never scored): draft trace + P4 diagnostic
    # artifact pointer, crash status.
    node2 = by_id[execution_trace_id_for_node(RUN_ID, "node-2")]
    assert node2["status"] == "crash"
    assert node2["score_bundle_ref"] == "score_bundle:unavailable"
    assert [c["call_kind"] for c in node2["calls"]] == [
        "engine_raw_trace",
        "build_diagnostic_artifact",
    ]
    diagnostic = node2["calls"][1]
    assert diagnostic["artifact_kind"] == "code_build_failure"
    assert diagnostic["s3_ref"].startswith("s3://artifact-bucket/")
    assert diagnostic["sha256"].startswith("sha256:")
    assert diagnostic["iteration"] == 2
    assert diagnostic["target_files"] == ["qualify/sources.py"]
    assert node2["judge_verdicts"] == []  # never scored


async def test_evidence_bundle_snapshots_and_gate_refs(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    row = store.inserted[EVIDENCE_BUNDLES_TABLE][0]
    assert row["retention_class"] == "live_verification"
    assert row["verification_state"] == "active"

    snapshots = row["snapshots"]
    assert len(snapshots) == 3  # one per per-ICP row
    per_icp = {snap["icp_ref"]: snap for snap in snapshots}
    assert set(per_icp) == {"icp-run1-1", "icp-run1-2", "icp-run1-3"}
    first = per_icp["icp-run1-1"]
    assert first["snapshot_kind"] == "per_icp_score_evidence"
    assert first["candidate_company_score_count"] == 3
    assert first["candidate_per_icp_score"] == pytest.approx(67.2)
    assert first["incontainer_trace_ref"].startswith("s3://")
    assert first["incontainer_trace_sha256"].startswith("sha256:")
    assert first["incontainer_trace_call_count"] == 5
    # numbers and refs only: no free-text failure_reason, no company payloads
    assert "failure_reason" not in first
    assert "candidate_company_scores" not in first
    assert "incontainer_trace_ref" not in per_icp["icp-run1-3"]

    doc = row["bundle_doc"]
    assert doc["score_bundle_ref"] == SCORE_BUNDLE_ID
    assert doc["score_bundle_hash"] == sha256_json({"bundle": "run1"})
    assert doc["execution_trace_ref"] == f"execution_trace:{NODE1_TRACE_ID}"
    assert doc["holdout_gate"]["decision"] == "accepted"
    assert doc["holdout_gate"]["baseline_benchmark_bundle_ref"] == "benchmark_bundle:benchmark-run1"
    assert doc["holdout_gate"]["public_icp_count"] == 2
    assert doc["holdout_gate"]["provider_excluded_icp_ids"] == ["icp-run1-9"]
    assert doc["aggregates"]["icp_count"] == 3
    assert doc["reflections"][0]["iteration"] == 1
    assert doc["incontainer_trace_count"] == 2


async def test_orphan_score_bundle_gets_fallback_trace_and_evidence_rows(
    tables, enabled
):
    tables = copy.deepcopy(tables)
    orphan_artifact = sha256_json({"candidate_artifact": "orphan-score-bundle"})
    tables["research_evaluation_score_bundles"].append(
        {
            "score_bundle_id": ORPHAN_SCORE_BUNDLE_ID,
            "run_id": RUN_ID,
            "ticket_id": TICKET_ID,
            "evaluation_epoch": 413,
            "bundle_status": "scored",
            "candidate_artifact_hash": orphan_artifact,
            "icp_set_hash": sha256_json({"icp_set": "orphan"}),
            "scoring_version": "scoring:v9",
            "evaluator_version": "evaluator:v9",
            "score_bundle_hash": sha256_json({"bundle": "orphan"}),
            "score_bundle_doc": _score_bundle_doc(RUN_ID, "orphan", orphan_artifact),
            "created_at": "2026-06-28T11:05:00Z",
        }
    )
    store = FakeStore(tables)

    result = await project_run(RUN_ID, store=store, dry_run=False)

    assert result.status == "projected", result.errors
    assert result.execution_trace_count == 4
    assert result.evidence_bundle_count == 2
    traces = {row["run_id"]: row for row in store.inserted[EXECUTION_TRACES_TABLE]}
    fallback_trace = traces[ORPHAN_TRACE_ID]
    _assert_trace_row_matches_sql(fallback_trace)
    assert fallback_trace["trace_doc"]["trace_kind"] == "score_bundle_only"
    assert fallback_trace["trace_doc"]["fallback_reason"] == (
        "score_bundle_not_linked_to_node"
    )
    assert fallback_trace["score_bundle_ref"] == ORPHAN_SCORE_BUNDLE_ID
    assert fallback_trace["outputs_ref"] == ORPHAN_SCORE_BUNDLE_ID
    assert fallback_trace["evidence_bundles"] == [
        f"evidence_bundle:{ORPHAN_EVIDENCE_ID}"
    ]
    assert {call["call_kind"] for call in fallback_trace["calls"]} == {
        "incontainer_trace"
    }

    evidence = {
        row["bundle_id"]: row for row in store.inserted[EVIDENCE_BUNDLES_TABLE]
    }[ORPHAN_EVIDENCE_ID]
    _assert_evidence_row_matches_sql(evidence)
    assert evidence["artifact_hash"] == orphan_artifact
    assert len(evidence["snapshots"]) == 3
    assert evidence["bundle_doc"]["source"] == "score_bundle_fallback"
    assert evidence["bundle_doc"]["score_bundle_ref"] == ORPHAN_SCORE_BUNDLE_ID
    assert evidence["bundle_doc"]["execution_trace_ref"] == (
        f"execution_trace:{ORPHAN_TRACE_ID}"
    )
    assert find_protected_material(fallback_trace) == set()
    assert find_protected_material(evidence) == set()


async def test_forward_only_linkage_embedded_in_new_events(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    event_rows = store.inserted[TRAJECTORY_EVENTS_TABLE]
    evaluated = {
        row["event"]["node_id"]: row["event"]
        for row in event_rows
        if row["event_type"] == "NODE_EVALUATED"
    }
    assert evaluated["node-1"]["execution_trace_ref"] == f"execution_trace:{NODE1_TRACE_ID}"
    assert (
        evaluated["node-1"]["evaluation_context"]["evidence_bundle_ref"]
        == f"evidence_bundle:{EVIDENCE_ID}"
    )
    # node-2 has a draft raw trace -> trace ref but no evidence (never scored)
    assert evaluated["node-2"]["execution_trace_ref"] == (
        f"execution_trace:{execution_trace_id_for_node(RUN_ID, 'node-2')}"
    )
    assert evaluated["node-2"]["evaluation_context"]["evidence_bundle_ref"] is None
    # anchored hashes still verify with the embedded refs
    for row in event_rows:
        assert verify_anchored_hash(row["event"])
    # the ref survives the schema view ($defs/event declares execution_trace_ref)
    view = schema_event_view(evaluated["node-1"])
    assert view["execution_trace_ref"] == f"execution_trace:{NODE1_TRACE_ID}"
    assert "evaluation_context" not in view


async def test_corpus_source_record_carries_prefixed_refs(tables):
    store = FakeStore(tables)
    inputs = await load_projection_inputs(RUN_ID, store)
    projection = build_trajectory_projection(**inputs)
    assert projection.errors == []
    record = projection.corpus_source_record
    assert validate_trajectory_corpus_source_record(record) == []
    assert len(record.execution_trace_refs) == 3
    assert all(ref.startswith("execution_trace:") for ref in record.execution_trace_refs)
    assert record.evidence_bundle_refs == (f"evidence_bundle:{EVIDENCE_ID}",)


# ---------------------------------------------------------------------------
# Idempotency: deterministic ids, re-runs write nothing new
# ---------------------------------------------------------------------------


async def test_reprojection_and_backfill_write_nothing_new(tables, enabled):
    store = FakeStore(tables)
    first = await project_run(RUN_ID, store=store, dry_run=False)
    assert first.status == "projected"
    writes = store.write_count()

    second = await project_run(RUN_ID, store=store, dry_run=False)
    assert second.status == "skipped_existing"
    assert store.write_count() == writes

    third = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=False)
    assert third.status == "skipped_traces_existing"
    assert store.write_count() == writes

    batch = await backfill_corpus_trace_rows(batch_size=10, dry_run=False, store=store)
    assert [r.status for r in batch] == ["skipped_traces_existing"]
    assert store.write_count() == writes


# ---------------------------------------------------------------------------
# --traces-backfill: already-projected (historical) runs gain rows, events
# are never retrofitted
# ---------------------------------------------------------------------------


async def test_traces_backfill_adds_rows_without_touching_events(tables, enabled):
    store = FakeStore(tables)
    await project_run(RUN_ID, store=store, dry_run=False)
    # Simulate a run projected BEFORE item 5.5: pointer rows absent.
    store.tables[EXECUTION_TRACES_TABLE] = []
    store.tables[EVIDENCE_BUNDLES_TABLE] = []
    event_writes = store.write_count(TRAJECTORY_EVENTS_TABLE)
    envelope_writes = store.write_count(TRAJECTORIES_TABLE)

    result = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=False)
    assert result.status == "traces_backfilled"
    assert result.execution_trace_count == 3
    assert result.evidence_bundle_count == 1
    # deterministic ids: restored rows carry the same ids
    restored = {row["run_id"] for row in store.tables[EXECUTION_TRACES_TABLE]}
    assert NODE1_TRACE_ID in restored and ENGINE_TRACE_ID in restored
    assert store.tables[EVIDENCE_BUNDLES_TABLE][0]["bundle_id"] == EVIDENCE_ID
    # forward-only: append-only tables untouched
    assert store.write_count(TRAJECTORY_EVENTS_TABLE) == event_writes
    assert store.write_count(TRAJECTORIES_TABLE) == envelope_writes


async def test_traces_backfill_dry_run_writes_nothing(tables, disabled):
    store = FakeStore(tables)
    async def _project():
        return await project_run(RUN_ID, store=store, dry_run=False)
    # project with the flag on, then drop pointer rows
    import os
    os.environ[PROJECTOR_ENABLED_ENV] = "true"
    try:
        await _project()
    finally:
        os.environ.pop(PROJECTOR_ENABLED_ENV, None)
    store.tables[EXECUTION_TRACES_TABLE] = []
    store.tables[EVIDENCE_BUNDLES_TABLE] = []
    writes = store.write_count()

    result = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=True)
    assert result.status == "traces_dry_run"
    assert result.execution_trace_count == 3
    assert result.evidence_bundle_count == 1
    assert store.write_count() == writes

    batch = await backfill_corpus_trace_rows(batch_size=10, dry_run=True, store=store)
    assert [r.status for r in batch] == ["traces_dry_run"]
    assert store.write_count() == writes


async def test_traces_backfill_skips_unprojected_and_disabled(tables, disabled):
    store = FakeStore(tables)
    # not projected yet -> --backfill owns it
    result = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=True)
    assert result.status == "skipped_unprojected"
    # disabled flag blocks writes (dry-run above stays available)
    blocked = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=False)
    assert blocked.status == "skipped_disabled"
    assert await backfill_corpus_trace_rows(dry_run=False, store=store) == []
    assert store.write_count() == 0


async def test_traces_backfill_batch_cap(tables, enabled):
    run2 = build_run_tables(
        run_id=RUN_ID_2,
        ticket_id=TICKET_ID_2,
        candidate_id=CANDIDATE_2,
        score_bundle_id=SCORE_BUNDLE_ID_2,
        run_tag="run2",
    )
    store = FakeStore(merge_tables(tables, run2))
    assert (await project_run(RUN_ID, store=store, dry_run=False)).status == "projected"
    assert (await project_run(RUN_ID_2, store=store, dry_run=False)).status == "projected"
    store.tables[EXECUTION_TRACES_TABLE] = []
    store.tables[EVIDENCE_BUNDLES_TABLE] = []

    first_pass = await backfill_corpus_trace_rows(batch_size=1, dry_run=False, store=store)
    assert [r.status for r in first_pass].count("traces_backfilled") == 1
    second_pass = await backfill_corpus_trace_rows(batch_size=1, dry_run=False, store=store)
    assert [r.status for r in second_pass].count("traces_backfilled") == 1
    third_pass = await backfill_corpus_trace_rows(batch_size=1, dry_run=False, store=store)
    assert all(r.status == "skipped_traces_existing" for r in third_pass)
    trace_ids = {row["run_id"] for row in store.tables[EXECUTION_TRACES_TABLE]}
    assert execution_trace_id_for_node(RUN_ID, "node-1") in trace_ids
    assert execution_trace_id_for_node(RUN_ID_2, "node-1") in trace_ids


# ---------------------------------------------------------------------------
# Absence tolerance (historical runs: no raw traces, thin/absent bundle docs)
# ---------------------------------------------------------------------------


async def test_historical_run_without_pointers_still_gets_valid_evidence(enabled):
    tables = build_run_tables(with_raw_traces=False, with_bundle_doc=False)
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors
    # No raw traces anywhere: no engine row, no node-2 row; the scored node
    # still gets a pointer row (score bundle IS a pointer).
    trace_rows = store.inserted[EXECUTION_TRACES_TABLE]
    assert [row["run_id"] for row in trace_rows] == [NODE1_TRACE_ID]
    node1 = trace_rows[0]
    _assert_trace_row_matches_sql(node1)
    assert node1["calls"] == []
    assert node1["score_bundle_ref"] == SCORE_BUNDLE_ID
    # Evidence row falls back to a single non-empty summary snapshot.
    evidence = store.inserted[EVIDENCE_BUNDLES_TABLE][0]
    _assert_evidence_row_matches_sql(evidence)
    assert len(evidence["snapshots"]) == 1
    summary = evidence["snapshots"][0]
    assert summary["snapshot_kind"] == "score_bundle_summary"
    assert summary["score_bundle_ref"] == SCORE_BUNDLE_ID
    assert summary["per_icp_rows_available"] is False
    assert summary["icp_count"] == 3  # public 2 + private 1 from the gate
    # node-2 gets no trace ref on its NODE_EVALUATED (no row written)
    evaluated = {
        row["event"]["node_id"]: row["event"]
        for row in store.inserted[TRAJECTORY_EVENTS_TABLE]
        if row["event_type"] == "NODE_EVALUATED"
    }
    assert evaluated["node-2"]["execution_trace_ref"] is None


async def test_run_with_zero_sources_is_skipped_by_backfill(enabled):
    tables = build_run_tables(with_raw_traces=False, with_bundle_doc=False)
    # Remove scoring entirely: no bundles, no scored events.
    tables["research_evaluation_score_bundles"] = []
    tables["research_lab_candidate_evaluation_events"] = []
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors
    assert result.execution_trace_count == 0
    assert result.evidence_bundle_count == 0
    assert store.write_count(EXECUTION_TRACES_TABLE) == 0
    assert store.write_count(EVIDENCE_BUNDLES_TABLE) == 0
    backfill = await backfill_run_corpus_trace_rows(RUN_ID, store=store, dry_run=False)
    assert backfill.status == "skipped_no_trace_sources"


# ---------------------------------------------------------------------------
# Scanner safety: pointers only, never protected content
# ---------------------------------------------------------------------------


async def test_poisoned_inputs_never_reach_written_rows(tables, enabled):
    poisoned = copy.deepcopy(tables)
    bundle = poisoned["research_evaluation_score_bundles"][0]
    per_icp = bundle["score_bundle_doc"]["aggregates"]["per_icp_results"]
    per_icp[0]["llm_response"] = "SECRET raw model output"
    per_icp[0]["icp_ref"] = "icp with the llm response inside"
    per_icp[0]["failure_reason"] = "provider returned the full prompt verbatim"
    bundle["score_bundle_doc"]["private_holdout_gate"]["decision"] = (
        "accepted despite page content mismatch"
    )
    drafted = poisoned["research_lab_auto_research_loop_events"][2]
    drafted["event_doc"]["prompt"] = "you are a helpful assistant"

    store = FakeStore(poisoned)
    result = await project_run(RUN_ID, store=store, dry_run=False)
    assert result.status == "projected", result.errors
    for table in (EXECUTION_TRACES_TABLE, EVIDENCE_BUNDLES_TABLE):
        for row in store.inserted[table]:
            assert find_protected_material(row) == set(), table
    flat = str(store.inserted[EVIDENCE_BUNDLES_TABLE])
    assert "SECRET" not in flat
    assert "full prompt" not in flat.lower()


# ---------------------------------------------------------------------------
# Flag gating and dry-run on the projection path
# ---------------------------------------------------------------------------


async def test_project_run_dry_run_reports_trace_counts_without_writes(tables, disabled):
    store = FakeStore(tables)
    result = await project_run(RUN_ID, store=store, dry_run=True)
    assert result.status == "dry_run"
    assert result.execution_trace_count == 3
    assert result.evidence_bundle_count == 1
    assert store.write_count() == 0


def test_deterministic_ids_are_stable_uuids():
    assert execution_trace_id_for_run(RUN_ID) == ENGINE_TRACE_ID
    assert execution_trace_id_for_node(RUN_ID, "node-1") == NODE1_TRACE_ID
    assert evidence_bundle_id_for_node(RUN_ID, "node-1", SCORE_BUNDLE_ID) == EVIDENCE_ID
    assert (
        execution_trace_id_for_score_bundle(RUN_ID, ORPHAN_SCORE_BUNDLE_ID)
        == ORPHAN_TRACE_ID
    )
    assert (
        evidence_bundle_id_for_score_bundle(RUN_ID, ORPHAN_SCORE_BUNDLE_ID)
        == ORPHAN_EVIDENCE_ID
    )
    for value in (
        ENGINE_TRACE_ID,
        NODE1_TRACE_ID,
        EVIDENCE_ID,
        ORPHAN_TRACE_ID,
        ORPHAN_EVIDENCE_ID,
    ):
        assert str(uuid.UUID(value)) == value
    # distinct kinds/scopes never collide
    assert len({
        ENGINE_TRACE_ID,
        NODE1_TRACE_ID,
        execution_trace_id_for_node(RUN_ID, "node-2"),
        execution_trace_id_for_node(RUN_ID_2, "node-1"),
        EVIDENCE_ID,
        ORPHAN_TRACE_ID,
        ORPHAN_EVIDENCE_ID,
        trajectory_id_for_run(RUN_ID),
    }) == 8


async def test_run_summary_contract_marks_summary_presence(enabled, monkeypatch):
    """P14 / v5 §8.3 run-summary contract: presence is recorded on the engine
    row; with enforcement on, a summary-less completed terminal is a crash."""
    store = FakeStore(build_run_tables())
    await project_run(RUN_ID, store=store, dry_run=False)
    engine = {
        row["run_id"]: row for row in store.inserted[EXECUTION_TRACES_TABLE]
    }[ENGINE_TRACE_ID]
    # Fixture terminal event predates the contract → absent, not enforced.
    assert engine["trace_doc"]["run_summary_present"] is False
    assert engine["status"] == "completed"

    monkeypatch.setenv("RESEARCH_LAB_RUN_SUMMARY_CONTRACT_ENFORCED", "true")
    store2 = FakeStore(build_run_tables())
    await project_run(RUN_ID, store=store2, dry_run=False)
    engine2 = {
        row["run_id"]: row for row in store2.inserted[EXECUTION_TRACES_TABLE]
    }[ENGINE_TRACE_ID]
    assert engine2["status"] == "crash"

    # A terminal event carrying the run_summary block stays completed.
    monkeypatch.setenv("RESEARCH_LAB_RUN_SUMMARY_CONTRACT_ENFORCED", "true")
    tables = build_run_tables()
    terminal = tables["research_lab_auto_research_loop_events"][-1]
    assert terminal["event_type"] == "loop_completed"
    terminal["event_doc"]["run_summary"] = {
        "schema_version": "1.0",
        "status": "completed",
        "stop_reason": "max_iterations",
        "iterations_completed": 2,
        "selected_candidate_count": 1,
        "wall_clock_seconds": 480.0,
        "cost_ledger": terminal["cost_ledger"],
        "openrouter_call_count": 3,
    }
    store3 = FakeStore(tables)
    await project_run(RUN_ID, store=store3, dry_run=False)
    engine3 = {
        row["run_id"]: row for row in store3.inserted[EXECUTION_TRACES_TABLE]
    }[ENGINE_TRACE_ID]
    assert engine3["status"] == "completed"
    assert engine3["trace_doc"]["run_summary_present"] is True
