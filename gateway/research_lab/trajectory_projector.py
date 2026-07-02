"""Research Lab v5 trajectory-capture projector (fableanalysis.md section 9.1).

An async/batch projector -- NOT a hot-path dual-write -- that reads the live
Research Lab event stream (``research_lab_auto_research_loop_events`` plus the
candidate / score / promotion event tables for a completed run) and projects it
into the v5 corpus tables created by ``scripts/27-research-lab-v5-schemas.sql``:

* ``research_trajectories``        -- one envelope row per run (no events array)
* ``research_trajectory_events``   -- append-only canonical event log
* ``research_lab_results_ledger``  -- one row per (run x drafted node)

Design facts this module encodes (verified against the code/schemas):

* ``research_trajectory_events`` CHECK-enforces the 12 canonical uppercase
  event types AND ``event->>'type' = event_type`` -- extra live events
  (``loop_direction_planned``, ``plan_alignment_judged``,
  ``source_inspection_*``, repair events) can never be standalone rows.  They
  are nested inside the nearest canonical event's ``event`` JSONB
  (``NODE_DRAFTED.event.planning_context``, ``PLATEAU_STOP.event.rejected_paths``
  and friends).
* ``schemas/research_trajectory.schema.json`` validates the *reassembled*
  trajectory document (envelope fields + ``events`` array + ``final``) and its
  ``$defs/event`` uses ``additionalProperties: false`` -- so schema validation
  runs against a *schema view* of each event with the nested extension keys
  stripped.  The stored/hashed event keeps the extensions.
* Anchored hashing replicates ``research_lab/hosted_loop.py::_append_event``
  exactly: the event dict is built as ``{seq, ts, type, cost_usd, **payload}``
  and ``anchored_hash = sha256_json(event-without-hash)`` is then embedded.
  ``build_hosted_trajectory`` itself is fixture-shaped and deliberately unused.
* The corpus protected-material scanner fails any payload containing keys or
  string markers like ``prompt`` / ``llm_response`` / ``page_content`` -- so
  every free-form payload is sanitized to pointers (``sha256:`` refs) before
  it enters a projected record.
* Capture != training eligibility: records are written with
  ``data_state="production_measured"`` and all rights/PII/legal booleans
  false, so corpus readiness stays BLOCKED by design.

Idempotency: trajectory ids are deterministic (uuid5 of the run id), and a run
whose envelope row already exists is skipped, so the projector is safely
re-runnable and can backfill every historical run from its existing events.

Flags / entry points:

* ``RESEARCH_LAB_TRAJECTORY_PROJECTOR_ENABLED`` (default false) gates all
  writes; dry-run projection is always allowed (read-only).
* ``project_run(run_id, ...)`` and ``project_completed_runs(...)`` for a
  worker periodic pass or cron (wiring is a follow-up, not done here).
* CLI: ``python3 -m gateway.research_lab.trajectory_projector --backfill``
  (dry-run by default; pass ``--no-dry-run`` to write).

Follow-ups owned elsewhere: applying scripts/27, the worker-pass hook,
code-edit-lane ``reflection_recorded`` emission, raw prompt/response capture
(``execution_traces`` refs stay empty until that lands), and folding the store
facade below into ``gateway/research_lab/store.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_lab.canonical import coerce_iso_z, sha256_json, utc_now_iso
from research_lab.schema_validation import validate_schema_record
from research_lab.trajectory_corpus import (
    PROTECTED_CORPUS_KEYS,
    PROTECTED_CORPUS_MARKERS,
    TRAJECTORY_CORPUS_CONTRACT_VERSION,
    TrajectoryCorpusSourceRecord,
    validate_trajectory_corpus_source_record,
)

logger = logging.getLogger(__name__)

PROJECTOR_ENABLED_ENV = "RESEARCH_LAB_TRAJECTORY_PROJECTOR_ENABLED"

TRAJECTORY_SCHEMA_NAME = "research_trajectory.schema.json"
RESULTS_LEDGER_SCHEMA_NAME = "results_ledger_row.schema.json"

TRAJECTORIES_TABLE = "research_trajectories"
TRAJECTORY_EVENTS_TABLE = "research_trajectory_events"
RESULTS_LEDGER_TABLE = "research_lab_results_ledger"

LOOP_EVENTS_TABLE = "research_lab_auto_research_loop_events"
RUN_QUEUE_TABLE = "research_loop_run_queue_current"
TICKET_TABLE = "research_loop_ticket_current"
CANDIDATE_ARTIFACTS_TABLE = "research_lab_candidate_artifacts"
EVALUATION_EVENTS_TABLE = "research_lab_candidate_evaluation_events"
PROMOTION_EVENTS_TABLE = "research_lab_candidate_promotion_events"
VERSION_EVENTS_TABLE = "research_lab_private_model_version_events"
SCORE_BUNDLES_TABLE = "research_evaluation_score_bundles"

# The 12 canonical v5 event types CHECK-enforced by scripts/27.
CANONICAL_EVENT_TYPES: tuple[str, ...] = (
    "PROBE",
    "LOOP_FUNDED",
    "NODE_DRAFTED",
    "NODE_EVALUATED",
    "NODE_REFLECTED",
    "PLATEAU_STOP",
    "L2_PROMOTED",
    "LANE_ENTERED",
    "PROBATION_SET_SCORED",
    "CROWNED",
    "REVERTED",
    "FINALIZED",
)

TARGETED_METRIC = "candidate_delta_vs_daily_baseline"
CUSTOMER_REF_STAND_IN = "customer_ref:none"  # miner-funded briefs have no customer
ENGINE_VERSION = "gateway_code_edit_image_build:v1"

_TERMINAL_LOOP_EVENT_TYPES = frozenset({"loop_completed", "loop_failed"})
_QUEUE_TERMINAL_STATUSES = ("completed", "failed")

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / TRAJECTORY_SCHEMA_NAME
)
_schema_event_fields_cache: frozenset[str] | None = None


def _schema_event_fields() -> frozenset[str]:
    """Property names allowed on a schema-view event ($defs/event is closed)."""
    global _schema_event_fields_cache
    if _schema_event_fields_cache is None:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        _schema_event_fields_cache = frozenset(
            schema["$defs"]["event"]["properties"].keys()
        )
    return _schema_event_fields_cache


def projector_enabled() -> bool:
    return os.getenv(PROJECTOR_ENABLED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------------
# Store facade
# ---------------------------------------------------------------------------


class GatewayProjectorStore:
    """Thin async facade over gateway.research_lab.store's public helpers.

    Exists so tests (and future stores) can inject fakes; the projector only
    uses these four query shapes.  Follow-up: fold into store.py itself.
    """

    async def select_one(self, table: str, *, columns: str = "*", filters: Any):
        from gateway.research_lab import store

        return await store.select_one(table, columns=columns, filters=filters)

    async def select_many(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Any,
        order_by: Any = (),
        limit: int = 100,
    ):
        from gateway.research_lab import store

        return await store.select_many(
            table, columns=columns, filters=filters, order_by=order_by, limit=limit
        )

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
        from gateway.research_lab import store

        return await store.select_all(
            table,
            columns=columns,
            filters=filters,
            order_by=order_by,
            batch_size=batch_size,
            max_rows=max_rows,
        )

    async def insert_row(self, table: str, row: dict[str, Any]):
        from gateway.research_lab import store

        return await store.insert_row(table, row)


def _deterministic_uuid(*parts: Any) -> str:
    from gateway.research_lab.store import deterministic_uuid

    return deterministic_uuid(*parts)


def trajectory_id_for_run(run_id: str) -> str:
    return _deterministic_uuid("research_trajectory", str(run_id))


# ---------------------------------------------------------------------------
# Protected-material sanitization (mirrors trajectory_corpus scanner rules)
# ---------------------------------------------------------------------------

_MARKER_PATTERNS = tuple(
    re.compile(re.escape(marker), re.IGNORECASE) for marker in PROTECTED_CORPUS_MARKERS
)
_REDACTED = "[protected-material-redacted]"


def _sanitize_text(value: str) -> str:
    out = value
    for pattern in _MARKER_PATTERNS:
        out = pattern.sub(_REDACTED, out)
    return out


def sanitize_capture_payload(value: Any) -> Any:
    """Replace protected keys with sha256 pointer refs and redact markers.

    The corpus scanner fails any payload containing keys like ``prompt`` /
    ``llm_response`` / ``page_content`` (unless ``_ref``-suffixed) or string
    markers like ``"llm response"``.  Content behind a protected key is
    replaced by a ``sha256:`` pointer under ``<key>_sha256_ref`` -- pointers,
    never content.
    """
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in PROTECTED_CORPUS_KEYS and not key_text.endswith(
                ("_ref", "_refs")
            ):
                try:
                    pointer = sha256_json(item)
                except Exception:
                    pointer = sha256_json(str(item))
                sanitized[f"{key}_sha256_ref"] = pointer
                continue
            sanitized[str(key)] = sanitize_capture_payload(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_capture_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def find_protected_material(value: Any, path: str = "") -> set[str]:
    """Local mirror of trajectory_corpus._find_protected_corpus_material."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_CORPUS_KEYS and not key_text.endswith(
                ("_ref", "_refs")
            ):
                found.add(key_path)
            found.update(find_protected_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(find_protected_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_CORPUS_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


# ---------------------------------------------------------------------------
# Canonical event construction (hosted_loop._append_event hashing pattern)
# ---------------------------------------------------------------------------


def _append_canonical_event(
    events: list[dict[str, Any]],
    *,
    ts: str,
    event_type: str,
    cost_usd: float,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a canonical event using hosted_loop.py's exact _append_event
    anchored-hash pattern: hash is computed over the event-without-hash (with
    ``seq``/``ts``/``type``/``cost_usd`` first) and then embedded."""
    seq = len(events)
    event = {
        "seq": seq,
        "ts": ts,
        "type": event_type,
        "cost_usd": round(float(cost_usd), 6),
        **dict(payload),
    }
    event["anchored_hash"] = sha256_json(event)
    events.append(event)
    return event


def verify_anchored_hash(event: Mapping[str, Any]) -> bool:
    body = {key: value for key, value in event.items() if key != "anchored_hash"}
    return sha256_json(body) == event.get("anchored_hash")


def schema_event_view(event: Mapping[str, Any]) -> dict[str, Any]:
    """Strip nested extension keys so the event passes the closed $defs/event."""
    fields = _schema_event_fields()
    return {key: value for key, value in event.items() if key in fields}


# ---------------------------------------------------------------------------
# Small coercion helpers
# ---------------------------------------------------------------------------


def _ts(value: Any, fallback: str | None = None) -> str:
    try:
        return coerce_iso_z(value)
    except Exception:
        return fallback or utc_now_iso()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_uuid(value: Any) -> str | None:
    import uuid

    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _doc(row: Mapping[str, Any], key: str = "event_doc") -> dict[str, Any]:
    value = row.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _ledger_total_usd(row: Mapping[str, Any]) -> float:
    ledger = row.get("cost_ledger")
    if not isinstance(ledger, Mapping):
        return 0.0
    for key in ("total_usd", "actual_openrouter_cost_usd"):
        if ledger.get(key) is not None:
            return _f(ledger.get(key))
    return 0.0


def _first_provider_tokens(row: Mapping[str, Any]) -> tuple[int, int, str]:
    usage = row.get("provider_usage")
    if isinstance(usage, Sequence):
        for item in usage:
            if isinstance(item, Mapping):
                return (
                    max(0, _i(item.get("prompt_tokens"))),
                    max(0, _i(item.get("completion_tokens"))),
                    str(item.get("model") or "") or "unknown",
                )
    return 0, 0, "unknown"


def _budget_balance_usd(doc: Mapping[str, Any], payment: Mapping[str, Any] | None) -> float:
    budget = doc.get("budget_context")
    if isinstance(budget, Mapping):
        for key in (
            "balance_usd",
            "compute_budget_usd",
            "budget_usd",
            "total_budget_usd",
            "remaining_budget_usd",
            "loop_budget_usd",
        ):
            if budget.get(key) is not None:
                return max(0.0, _f(budget.get(key)))
    if payment:
        for key in ("amount_usd", "verified_amount_usd", "payment_amount_usd"):
            if payment.get(key) is not None:
                return max(0.0, _f(payment.get(key)))
    return 0.0


def _plateau_reason(stop_reason: str, queue_status: str) -> str:
    text = (stop_reason or "").lower()
    if "budget" in text or "exhaust" in text or "balance" in text:
        return "balance_exhausted"
    if "funder" in text:
        return "funder_stop"
    if queue_status == "failed" and not text:
        return "plateau"
    return "plateau"


def _node_metrics(
    *,
    proxy_score: float,
    latency_s: float,
    schema_validity: bool,
) -> dict[str, Any]:
    """Minimal node_metrics passing schemas/research_trajectory.schema.json.

    Only fields computable from live events today are populated; the rest are
    zero-valued schema slots (section 9.1 item 4 -- richer metric emission is
    a follow-up owned elsewhere).
    """
    return {
        "proxy_score": round(float(proxy_score), 6),
        "evidence_defect_rate_by_category": {},
        "coverage": 0.0,
        "cost_per_icp": 0.0,
        "latency_s": max(0.0, round(float(latency_s), 3)),
        "schema_validity": bool(schema_validity),
        "complexity": {
            "diff_loc": 0,
            "prompt_pack_tokens": 0,
            "component_count": 0,
        },
    }


# ---------------------------------------------------------------------------
# Projection dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryProjection:
    run_id: str
    trajectory_id: str
    envelope_row: dict[str, Any]
    event_rows: list[dict[str, Any]]
    ledger_rows: list[dict[str, Any]]
    trajectory_doc: dict[str, Any]
    corpus_source_record: TrajectoryCorpusSourceRecord
    errors: list[str] = field(default_factory=list)


@dataclass
class ProjectionResult:
    run_id: str
    status: str  # projected | dry_run | skipped_existing | skipped_disabled |
    #             skipped_incomplete | failed
    trajectory_id: str | None = None
    event_count: int = 0
    ledger_row_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "trajectory_id": self.trajectory_id,
            "event_count": self.event_count,
            "ledger_row_count": self.ledger_row_count,
            "errors": list(self.errors),
        }


@dataclass
class _NodeState:
    node_id: str
    drafted_row: Mapping[str, Any]
    lane: str
    plan_path_id: str | None
    unified_diff_hash: str
    build_passed_row: Mapping[str, Any] | None = None
    build_failed_row: Mapping[str, Any] | None = None
    candidate_artifact_hash: str | None = None
    candidate_source_diff_hash: str | None = None
    candidate_id: str | None = None
    evaluated: bool = False
    gate_doc: Mapping[str, Any] | None = None
    score_bundle_id: str | None = None
    promotion_passed: bool = False
    version_created: bool = False
    selected: bool = False
    cost_usd: float = 0.0
    draft_seq: int | None = None
    evaluated_seq: int | None = None


# ---------------------------------------------------------------------------
# Core builder (pure: rows in, projection out)
# ---------------------------------------------------------------------------


def build_trajectory_projection(
    *,
    run_id: str,
    queue_row: Mapping[str, Any] | None,
    ticket: Mapping[str, Any] | None,
    loop_events: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]] = (),
    evaluation_events: Sequence[Mapping[str, Any]] = (),
    promotion_events: Sequence[Mapping[str, Any]] = (),
    version_events: Sequence[Mapping[str, Any]] = (),
    score_bundles: Sequence[Mapping[str, Any]] = (),
    payment: Mapping[str, Any] | None = None,
) -> TrajectoryProjection:
    run_id = str(run_id)
    ticket = ticket or {}
    queue_row = queue_row or {}
    ordered = sorted(loop_events, key=lambda row: _i(row.get("seq")))
    if not ordered:
        raise ValueError(f"run {run_id} has no live loop events to project")

    tid = trajectory_id_for_run(run_id)
    island = str(ticket.get("island") or "unknown_island")
    funder_hotkey = str(ticket.get("miner_hotkey") or "") or None
    brief_sanitized_ref = str(ticket.get("brief_sanitized_ref") or "sha256:unknown-brief")
    brief_id = _coerce_uuid(ticket.get("brief_id")) or _deterministic_uuid(
        "research_brief", brief_sanitized_ref
    )

    first_doc = _doc(ordered[0])
    champion_base = (
        str(first_doc.get("parent_image_digest_hash") or "")
        or next(
            (
                str(row.get("parent_artifact_hash"))
                for row in candidate_rows
                if row.get("parent_artifact_hash")
            ),
            "",
        )
        or "unknown_champion_base"
    )
    created_at = _ts(ordered[0].get("created_at"))

    # -- index post-loop data ------------------------------------------------
    candidates_by_hash: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        artifact_hash = str(row.get("candidate_artifact_hash") or "")
        if artifact_hash:
            candidates_by_hash[artifact_hash] = row

    scored_by_candidate: dict[str, Mapping[str, Any]] = {}
    for row in evaluation_events:
        if str(row.get("event_type")) == "scored":
            scored_by_candidate[str(row.get("candidate_id"))] = row

    bundles_by_id: dict[str, Mapping[str, Any]] = {
        str(row.get("score_bundle_id")): row for row in score_bundles
    }
    default_epoch = next(
        (
            _i(row.get("evaluation_epoch"))
            for row in score_bundles
            if row.get("evaluation_epoch") is not None
        ),
        0,
    )

    # -- walk the live loop event stream --------------------------------------
    canonical: list[dict[str, Any]] = []
    nodes: dict[str, _NodeState] = {}
    node_order: list[str] = []
    pending_planning: dict[str, Any] | None = None
    pending_inspections: list[dict[str, Any]] = []
    pending_alignment: dict[str, Any] | None = None
    rejected_paths: list[dict[str, Any]] = []
    selected_node_ids: list[str] = []
    receipt_id = next(
        (str(row.get("receipt_id")) for row in ordered if row.get("receipt_id")), None
    )
    terminal_row: Mapping[str, Any] | None = None
    last_anchor_total = 0.0
    last_ts = created_at
    iterations_completed = 0
    requested_loop_count = _i(ticket.get("requested_loop_count"), 1)

    def _emit(
        row: Mapping[str, Any],
        event_type: str,
        payload: Mapping[str, Any],
        *,
        zero_cost: bool = False,
    ) -> dict[str, Any]:
        nonlocal last_anchor_total, last_ts
        ts = _ts(row.get("created_at"), last_ts)
        last_ts = ts
        total = _ledger_total_usd(row)
        cost = 0.0
        if not zero_cost:
            cost = max(0.0, round(total - last_anchor_total, 6))
        last_anchor_total = max(last_anchor_total, total)
        return _append_canonical_event(
            canonical, ts=ts, event_type=event_type, cost_usd=cost, payload=payload
        )

    def _live_ref(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "live_event_id": str(row.get("event_id") or ""),
            "live_seq": _i(row.get("seq")),
            "live_event_type": str(row.get("event_type") or ""),
        }

    for row in ordered:
        event_type = str(row.get("event_type") or "")
        doc = _doc(row)
        node_id = str(row.get("node_id") or "") or None
        iterations_completed = max(iterations_completed, _i(doc.get("iteration")))

        if event_type in ("loop_started", "loop_resumed"):
            run_context = sanitize_capture_payload(
                {
                    **_live_ref(row),
                    "run_id": run_id,
                    "ticket_id": str(row.get("ticket_id") or ""),
                    "customer_ref": CUSTOMER_REF_STAND_IN,
                    "candidate_kind": doc.get("candidate_kind"),
                    "resumed_from_checkpoint": bool(doc.get("resumed_from_checkpoint")),
                    "settings": doc.get("settings"),
                    "budget_context": doc.get("budget_context"),
                    "source_tree_hash": doc.get("source_tree_hash"),
                    "parent_image_digest_hash": doc.get("parent_image_digest_hash"),
                    "payment_ref": (
                        f"payment:{payment.get('payment_id')}" if payment else None
                    ),
                }
            )
            _emit(
                row,
                "LOOP_FUNDED",
                {
                    "loop_n": max(0, _i(doc.get("requested_loop_count"), requested_loop_count)),
                    "balance_before": _budget_balance_usd(doc, payment),
                    "run_context": run_context,
                },
            )
        elif event_type == "loop_direction_planned":
            pending_planning = sanitize_capture_payload(
                {**_live_ref(row), **doc}
            )
        elif event_type.startswith("source_inspection"):
            pending_inspections.append(
                sanitize_capture_payload({**_live_ref(row), **doc})
            )
        elif event_type == "plan_alignment_judged":
            pending_alignment = sanitize_capture_payload({**_live_ref(row), **doc})
        elif event_type == "code_edit_drafted":
            hypothesis_doc = doc.get("hypothesis")
            hypothesis_doc = dict(hypothesis_doc) if isinstance(hypothesis_doc, Mapping) else {}
            lane = str(doc.get("lane") or "") or "code_edit"
            unified_diff_hash = str(doc.get("unified_diff_hash") or "") or sha256_json(
                {"unified_diff_missing": node_id or _i(row.get("seq"))}
            )
            state = _NodeState(
                node_id=node_id or f"node-{len(node_order)}",
                drafted_row=row,
                lane=lane,
                plan_path_id=(
                    str(doc.get("plan_path_id")) if doc.get("plan_path_id") else None
                ),
                unified_diff_hash=unified_diff_hash,
            )
            nodes[state.node_id] = state
            node_order.append(state.node_id)
            tokens_in, tokens_out, model_used = _first_provider_tokens(row)
            planning_context = sanitize_capture_payload(
                {
                    "loop_direction_plan": pending_planning,
                    "source_inspections": list(pending_inspections),
                    "plan_alignment": pending_alignment,
                }
            )
            pending_inspections = []
            pending_alignment = None
            draft_context = sanitize_capture_payload(
                {
                    **_live_ref(row),
                    "iteration": _i(doc.get("iteration")),
                    "lane": lane,
                    "plan_path_id": state.plan_path_id,
                    "target_files": doc.get("target_files"),
                    "unified_diff_hash": unified_diff_hash,
                    "hypothesis_extra": {
                        key: value
                        for key, value in hypothesis_doc.items()
                        if key not in ("failure_mode", "mechanism", "predicted_delta")
                    },
                    "source_event_doc": doc,
                }
            )
            event = _emit(
                row,
                "NODE_DRAFTED",
                {
                    "node_id": state.node_id,
                    "parent_id": None,
                    "operator": "draft",
                    "component": (_sanitize_text(lane)[:128] or "code_edit"),
                    "patch_type": "CODE_EDIT",
                    "hypothesis": {
                        "failure_mode": _sanitize_text(
                            str(hypothesis_doc.get("failure_mode") or "unspecified")
                        )
                        or "unspecified",
                        "mechanism": _sanitize_text(
                            str(hypothesis_doc.get("mechanism") or "unspecified")
                        )
                        or "unspecified",
                        "predicted_delta": _f(hypothesis_doc.get("predicted_delta")),
                        "falsifier": f"{TARGETED_METRIC} <= 0 on the daily benchmark",
                    },
                    "patch_ref": unified_diff_hash,
                    "model_used": model_used[:128],
                    "tokens": {"in": tokens_in, "out": tokens_out},
                    "planning_context": planning_context,
                    "draft_context": draft_context,
                },
            )
            state.draft_seq = event["seq"]
            state.cost_usd += event["cost_usd"]
        elif event_type == "candidate_build_passed" and node_id and node_id in nodes:
            state = nodes[node_id]
            state.build_passed_row = row
            state.candidate_artifact_hash = (
                str(row.get("candidate_artifact_hash") or "") or None
            )
            state.candidate_source_diff_hash = (
                str(doc.get("candidate_source_diff_hash") or "") or None
            )
            candidate_row = candidates_by_hash.get(state.candidate_artifact_hash or "")
            if candidate_row:
                state.candidate_id = str(candidate_row.get("candidate_id"))
            _emit_node_evaluated(
                _emit,
                state,
                row,
                scored_by_candidate,
                bundles_by_id,
            )
        elif node_id and node_id in nodes and _is_build_failure(event_type):
            state = nodes[node_id]
            state.build_failed_row = row
            rejected_paths.append(
                sanitize_capture_payload(
                    {
                        **_live_ref(row),
                        "node_id": node_id,
                        "stage": event_type,
                        "error": str(doc.get("error") or "")[:500],
                        "error_hash": doc.get("error_hash"),
                    }
                )
            )
            _emit_node_evaluated(
                _emit,
                state,
                row,
                scored_by_candidate,
                bundles_by_id,
            )
        elif event_type in (
            "code_edit_validation_failed",
            "code_edit_alignment_rejected",
            "no_viable_patch",
            "code_edit_repair_requested",
            "code_edit_repair_drafted",
            "code_edit_repair_failed",
            "candidate_patch_apply_failed",
        ):
            rejected_paths.append(
                sanitize_capture_payload(
                    {**_live_ref(row), "node_id": node_id, **doc}
                )
            )
        elif event_type == "candidate_selected":
            if node_id:
                selected_node_ids.append(node_id)
                if node_id in nodes:
                    nodes[node_id].selected = True
        elif event_type == "reflection_recorded":
            reflected_node = node_id or (node_order[-1] if node_order else None)
            if reflected_node:
                reflection = doc.get("reflection")
                reflection = dict(reflection) if isinstance(reflection, Mapping) else {}
                lane = nodes[reflected_node].lane if reflected_node in nodes else "auto_research"
                _emit(
                    row,
                    "NODE_REFLECTED",
                    {
                        "node_id": reflected_node,
                        "lesson": {
                            "worked": _sanitize_text(str(reflection.get("worked") or "")),
                            "failed": _sanitize_text(str(reflection.get("failed") or "")),
                            "why": _sanitize_text(str(reflection.get("why") or "")),
                            "next_question": _sanitize_text(
                                str(reflection.get("next_question") or "")
                            ),
                        },
                        "lesson_embedding_ref": None,
                        "lesson_provenance": {
                            "champion_base": champion_base,
                            "component": _sanitize_text(lane)[:128] or "auto_research",
                            "eval_version": "unversioned",
                        },
                        "reflection_context": sanitize_capture_payload(
                            {**_live_ref(row), "iteration": _i(doc.get("iteration"))}
                        ),
                    },
                )
        elif event_type in _TERMINAL_LOOP_EVENT_TYPES:
            terminal_row = row
        # checkpoint_saved / heartbeats / build_started etc. are intentionally
        # not canonical; their information survives via the nested contexts.

    # -- PLATEAU_STOP ----------------------------------------------------------
    queue_status = str(queue_row.get("current_queue_status") or "")
    terminal_doc = _doc(terminal_row) if terminal_row else {}
    stop_reason = str(terminal_doc.get("stop_reason") or queue_status or "unknown")
    best_node_id = (
        selected_node_ids[0]
        if selected_node_ids
        else next(
            (state.node_id for state in nodes.values() if state.gate_doc),
            "none",
        )
    )
    stop_anchor_row = terminal_row or ordered[-1]
    _emit(
        stop_anchor_row,
        "PLATEAU_STOP",
        {
            "reason": _plateau_reason(stop_reason, queue_status),
            "best_node_id": best_node_id,
            "rejected_paths": rejected_paths,
            "stop_context": sanitize_capture_payload(
                {
                    **_live_ref(stop_anchor_row),
                    "stop_reason": _sanitize_text(stop_reason),
                    "queue_status": queue_status,
                    "iterations_completed": max(
                        iterations_completed, _i(terminal_doc.get("iterations_completed"))
                    ),
                    "selected_candidate_count": _i(
                        terminal_doc.get("selected_candidate_count")
                    ),
                }
            ),
        },
    )

    # -- post-loop promotion lifecycle ----------------------------------------
    nodes_by_candidate = {
        state.candidate_id: state for state in nodes.values() if state.candidate_id
    }
    crown_doc: dict[str, Any] | None = None
    promo_ordered = sorted(
        promotion_events, key=lambda row: str(row.get("created_at") or "")
    )
    version_ids: list[str] = []
    for row in promo_ordered:
        promo_type = str(row.get("event_type") or "")
        candidate_id = str(row.get("candidate_id") or "")
        state = nodes_by_candidate.get(candidate_id)
        promo_node_id = state.node_id if state else (candidate_id or "unknown_node")
        gate = state.gate_doc if state and state.gate_doc else {}
        promo_doc = _doc(row)
        if promo_type == "promotion_passed":
            if state:
                state.promotion_passed = True
            _append_canonical_event(
                canonical,
                ts=_ts(row.get("created_at"), last_ts),
                event_type="L2_PROMOTED",
                cost_usd=0.0,
                payload={
                    "node_id": promo_node_id,
                    "l2_score": _f(gate.get("candidate_total_score")),
                    "champion_same_day_l2": _f(gate.get("baseline_aggregate_score")),
                    "delta": _f(gate.get(TARGETED_METRIC)),
                    "entry_bar_passed": True,
                    "promotion_context": sanitize_capture_payload(
                        {
                            "candidate_id": candidate_id,
                            "rolling_window_hash": row.get("rolling_window_hash"),
                            "improvement_points": _f(row.get("improvement_points")),
                            "threshold_points": _f(row.get("threshold_points")),
                        }
                    ),
                },
            )
        elif promo_type == "active_version_created":
            if state:
                state.version_created = True
            epoch = default_epoch
            bundle = bundles_by_id.get(str(row.get("source_score_bundle_id") or ""))
            if bundle and bundle.get("evaluation_epoch") is not None:
                epoch = _i(bundle.get("evaluation_epoch"))
            crown_doc = {
                "area": island,
                "started_epoch": max(0, epoch),
                "grant_share": 0.0,
                "term_k": 1,
            }
            version_id = str(row.get("private_model_version_id") or "")
            if version_id:
                version_ids.append(version_id)
            _append_canonical_event(
                canonical,
                ts=_ts(row.get("created_at"), last_ts),
                event_type="CROWNED",
                cost_usd=0.0,
                payload={
                    "area": island,
                    "epoch": max(0, epoch),
                    "promotion_context": sanitize_capture_payload(
                        {
                            "candidate_id": candidate_id,
                            "private_model_version_ref": (
                                f"private_model_version:{version_id}" if version_id else None
                            ),
                            "new_model_artifact_hash": promo_doc.get(
                                "new_model_artifact_hash"
                            ),
                        }
                    ),
                },
            )

    for row in sorted(version_events, key=lambda row: str(row.get("created_at") or "")):
        if str(row.get("event_type") or "") != "superseded":
            continue
        version_id = str(row.get("private_model_version_id") or "")
        if version_ids and version_id not in version_ids:
            continue
        _append_canonical_event(
            canonical,
            ts=_ts(row.get("created_at"), last_ts),
            event_type="FINALIZED",
            cost_usd=0.0,
            payload={
                "area": island,
                "epoch": max(0, default_epoch),
                "finalization_context": sanitize_capture_payload(
                    {
                        "private_model_version_ref": (
                            f"private_model_version:{version_id}" if version_id else None
                        ),
                        "reason": _sanitize_text(str(row.get("reason") or "")),
                    }
                ),
            },
        )

    # -- results ledger rows ---------------------------------------------------
    ledger_created_at = _ts(stop_anchor_row.get("created_at"), last_ts)
    ledger_rows: list[dict[str, Any]] = []
    for ordered_node_id in node_order:
        state = nodes[ordered_node_id]
        status = _ledger_status(state)
        delta = (
            _f(state.gate_doc.get(TARGETED_METRIC)) if state.gate_doc else None
        )
        delta_text = f"{delta:.6f}" if delta is not None else "n/a"
        description = _sanitize_text(
            f"CODE_EDIT on {state.lane} targeted {TARGETED_METRIC}; "
            f"decision={status}; delta={delta_text}."
        )[:2048]
        ledger_rows.append(
            {
                "ledger_row_id": _deterministic_uuid(
                    "research_lab_results_ledger", run_id, state.node_id
                ),
                "schema_version": "1.0",
                "trajectory_id": tid,
                "node_id": state.node_id,
                "commit": state.candidate_source_diff_hash
                or state.unified_diff_hash,
                "island": island,
                "brief_id": brief_id,
                "targeted_metric": TARGETED_METRIC,
                "delta_vs_parent": delta,
                "cost_usd": round(max(0.0, state.cost_usd), 6),
                "status": status,
                "description": description,
                "source_event_seq": (
                    state.evaluated_seq
                    if state.evaluated_seq is not None
                    else state.draft_seq
                ),
                "created_at": ledger_created_at,
            }
        )

    # -- envelope / trajectory doc / final -------------------------------------
    final_doc = {
        "settlement": {
            "loops_consumed": max(0, iterations_completed),
            "probation_charged": False,
            "balance_returned": 0.0,
            "crown": crown_doc,
            "grant_state": "none",
            "receipt_ref": f"receipt:{receipt_id}" if receipt_id else None,
        }
    }
    novelty_gate = {
        "result": "pass",
        "similarity": 0.0,
        "nearest_prior_receipt": None,
    }
    trajectory_doc = {
        "trajectory_id": tid,
        "schema_version": "1.0",
        "brief_id": brief_id,
        "island": island,
        "funder_hotkey": funder_hotkey,
        "brief_sanitized_ref": brief_sanitized_ref,
        "novelty_gate": novelty_gate,
        "engine_version": ENGINE_VERSION,
        "champion_base": champion_base,
        "created_at": created_at,
        "events": [schema_event_view(event) for event in canonical],
        "final": final_doc,
    }
    envelope_row = {
        "trajectory_id": tid,
        "schema_version": "1.0",
        "brief_id": brief_id,
        "island": island,
        "funder_hotkey": funder_hotkey,
        "brief_sanitized_ref": brief_sanitized_ref,
        "novelty_gate": novelty_gate,
        "engine_version": ENGINE_VERSION,
        "champion_base": champion_base,
        "final": final_doc,
        "created_at": created_at,
    }
    event_rows = [
        {
            "trajectory_id": tid,
            "seq": event["seq"],
            "ts": event["ts"],
            "event_type": event["type"],
            "cost_usd": event["cost_usd"],
            "anchored_hash": event["anchored_hash"],
            "event": event,
        }
        for event in canonical
    ]

    corpus_source_record = TrajectoryCorpusSourceRecord(
        source_id=f"trajectory_source:{tid}",
        trajectory_id=tid,
        trajectory_hash=sha256_json(trajectory_doc),
        trajectory_schema_valid=True,
        event_count=len(canonical),
        execution_trace_refs=(),  # raw-trace capture is a follow-up (section 9.1 item 5)
        evidence_bundle_refs=(),
        results_ledger_refs=tuple(
            f"results_ledger:{row['ledger_row_id']}" for row in ledger_rows
        ),
        receipt_refs=(f"receipt:{receipt_id}",) if receipt_id else (),
        cost_ledger_refs=(f"cost_ledger:{run_id}",),
        release_policy_ref="release_policy:unassigned",
        trajectory_rights_ref="trajectory_rights:unassigned",
        distillation_rights_ref="distillation_rights:unassigned",
        pii_review_ref="pii_review:unassigned",
        legal_gate_ref="legal_gate:unassigned",
        island=island,
        split="train",
        data_state="production_measured",
        measured_data=True,
        rights_verified=False,
        distillation_rights_verified=False,
        pii_review_passed=False,
        legal_gate_passed=False,
        protected_data_scanned=True,
        eligible_for_training=False,  # readiness stays BLOCKED by design
        eligible_for_distillation=False,
    )

    projection = TrajectoryProjection(
        run_id=run_id,
        trajectory_id=tid,
        envelope_row=envelope_row,
        event_rows=event_rows,
        ledger_rows=ledger_rows,
        trajectory_doc=trajectory_doc,
        corpus_source_record=corpus_source_record,
    )
    projection.errors = validate_projection(projection)
    return projection


def _is_build_failure(event_type: str) -> bool:
    if event_type == "candidate_build_failed":
        return True
    return event_type.endswith("_failed") and (
        "build" in event_type or "private_test" in event_type or "image" in event_type
    )


def _emit_node_evaluated(
    emit: Any,
    state: _NodeState,
    row: Mapping[str, Any],
    scored_by_candidate: Mapping[str, Mapping[str, Any]],
    bundles_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    if state.evaluated:
        return
    state.evaluated = True
    doc = _doc(row)
    scored_row = (
        scored_by_candidate.get(state.candidate_id) if state.candidate_id else None
    )
    gate: Mapping[str, Any] = {}
    score_bundle_id: str | None = None
    if scored_row:
        scored_doc = _doc(scored_row)
        raw_gate = scored_doc.get("private_holdout_gate")
        gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else {}
        score_bundle_id = (
            str(scored_row.get("score_bundle_id"))
            if scored_row.get("score_bundle_id")
            else None
        )
        state.gate_doc = gate
        state.score_bundle_id = score_bundle_id
    if state.build_failed_row is not None and state.build_passed_row is None:
        status = "crash"
    elif gate:
        status = "scored"
    else:
        status = "timeout"  # built but never scored
    proxy_score = _f(gate.get("candidate_delta_vs_daily_baseline")) if gate else 0.0
    evaluation_context = sanitize_capture_payload(
        {
            "candidate_ref": (
                f"candidate:{state.candidate_id}" if state.candidate_id else None
            ),
            "candidate_artifact_hash": state.candidate_artifact_hash,
            "candidate_source_diff_hash": state.candidate_source_diff_hash,
            "score_bundle_ref": score_bundle_id,
            "gate": gate or None,
            "build_event_doc": doc,
            "build_outcome": (
                "failed" if state.build_failed_row is not None else "passed"
            ),
        }
    )
    event = emit(
        row,
        "NODE_EVALUATED",
        {
            "node_id": state.node_id,
            "status": status,
            "rung": "L0",
            "metrics": _node_metrics(
                proxy_score=proxy_score,
                latency_s=_f(row.get("elapsed_seconds")),
                schema_validity=state.build_failed_row is None,
            ),
            "paired_lcb_vs_parent": None,
            "fixtures": [],
            "cache_hits": {"snapshot": 0, "verdict": 0},
            "execution_trace_ref": None,  # raw-trace capture is a follow-up
            "evaluation_context": evaluation_context,
        },
    )
    state.evaluated_seq = event["seq"]
    state.cost_usd += event["cost_usd"]


def _ledger_status(state: _NodeState) -> str:
    """keep/discard/crash/timeout per node, mirroring hosted_loop._ledger_status."""
    if state.build_failed_row is not None and state.build_passed_row is None:
        return "crash"
    if state.gate_doc:
        if state.promotion_passed or state.version_created:
            return "keep"
        return "discard"
    if state.build_passed_row is not None:
        return "timeout"  # built, never scored
    return "discard"  # drafted, never built (validation/alignment/apply failure)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_projection(projection: TrajectoryProjection) -> list[str]:
    """Validate every record before anything is written."""
    errors: list[str] = []
    errors.extend(
        f"trajectory_doc: {error}"
        for error in validate_schema_record(
            TRAJECTORY_SCHEMA_NAME, projection.trajectory_doc
        )
    )
    for row in projection.ledger_rows:
        schema_row = {
            key: value for key, value in row.items() if key != "source_event_seq"
        }
        errors.extend(
            f"results_ledger[{row['node_id']}]: {error}"
            for error in validate_schema_record(RESULTS_LEDGER_SCHEMA_NAME, schema_row)
        )
    errors.extend(
        f"corpus_source_record: {error}"
        for error in validate_trajectory_corpus_source_record(
            projection.corpus_source_record
        )
    )
    for row in projection.event_rows:
        event = row["event"]
        if event.get("type") not in CANONICAL_EVENT_TYPES:
            errors.append(f"event seq={row['seq']}: non-canonical type {event.get('type')!r}")
        if event.get("type") != row["event_type"]:
            errors.append(f"event seq={row['seq']}: event.type != event_type column")
        if not verify_anchored_hash(event):
            errors.append(f"event seq={row['seq']}: anchored_hash mismatch")
        protected = find_protected_material(event)
        if protected:
            errors.append(
                f"event seq={row['seq']}: protected material at {sorted(protected)[:3]}"
            )
    protected = find_protected_material(projection.envelope_row)
    if protected:
        errors.append(f"envelope: protected material at {sorted(protected)[:3]}")
    for row in projection.ledger_rows:
        protected = find_protected_material(row)
        if protected:
            errors.append(
                f"ledger[{row['node_id']}]: protected material at {sorted(protected)[:3]}"
            )
    return errors


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def load_projection_inputs(
    run_id: str, store: Any
) -> dict[str, Any]:
    run_id = str(run_id)
    queue_row = await store.select_one(
        RUN_QUEUE_TABLE, filters=(("run_id", run_id),)
    )
    ticket = None
    if queue_row and queue_row.get("ticket_id"):
        ticket = await store.select_one(
            TICKET_TABLE, filters=(("ticket_id", str(queue_row["ticket_id"])),)
        )
    loop_events = await store.select_all(
        LOOP_EVENTS_TABLE,
        filters=(("run_id", run_id),),
        order_by=(("seq", False),),
        max_rows=5000,
    )
    candidate_rows = await store.select_many(
        CANDIDATE_ARTIFACTS_TABLE, filters=(("run_id", run_id),), limit=200
    )
    evaluation_events = await store.select_many(
        EVALUATION_EVENTS_TABLE,
        filters=(("run_id", run_id),),
        order_by=(("seq", False),),
        limit=500,
    )
    candidate_ids = [
        str(row.get("candidate_id"))
        for row in candidate_rows
        if row.get("candidate_id")
    ]
    promotion_events: list[dict[str, Any]] = []
    version_events: list[dict[str, Any]] = []
    if candidate_ids:
        promotion_events = await store.select_many(
            PROMOTION_EVENTS_TABLE,
            filters=(("candidate_id", "in", candidate_ids),),
            limit=500,
        )
        version_ids = [
            str(row.get("private_model_version_id"))
            for row in promotion_events
            if row.get("private_model_version_id")
        ]
        if version_ids:
            version_events = await store.select_many(
                VERSION_EVENTS_TABLE,
                filters=(("private_model_version_id", "in", version_ids),),
                limit=200,
            )
    score_bundles = await store.select_many(
        SCORE_BUNDLES_TABLE, filters=(("run_id", run_id),), limit=100
    )
    return {
        "run_id": run_id,
        "queue_row": queue_row,
        "ticket": ticket,
        "loop_events": loop_events,
        "candidate_rows": candidate_rows,
        "evaluation_events": evaluation_events,
        "promotion_events": promotion_events,
        "version_events": version_events,
        "score_bundles": score_bundles,
    }


async def project_run(
    run_id: str,
    *,
    store: Any | None = None,
    dry_run: bool = True,
) -> ProjectionResult:
    """Project one completed run into the v5 corpus tables.

    Best-effort: never raises; every failure is logged and reported in the
    returned ProjectionResult.
    """
    run_id = str(run_id)
    store = store or GatewayProjectorStore()
    tid = trajectory_id_for_run(run_id)
    try:
        if not dry_run and not projector_enabled():
            logger.warning(
                "research_lab_trajectory_projector_disabled run_id=%s set %s=1 to enable writes",
                run_id,
                PROJECTOR_ENABLED_ENV,
            )
            return ProjectionResult(run_id=run_id, status="skipped_disabled", trajectory_id=tid)
        existing = await store.select_one(
            TRAJECTORIES_TABLE, filters=(("trajectory_id", tid),)
        )
        if existing:
            return ProjectionResult(
                run_id=run_id, status="skipped_existing", trajectory_id=tid
            )
        inputs = await load_projection_inputs(run_id, store)
        if not inputs["loop_events"]:
            return ProjectionResult(
                run_id=run_id,
                status="skipped_incomplete",
                trajectory_id=tid,
                errors=["no live loop events for run"],
            )
        projection = build_trajectory_projection(**inputs)
        if projection.errors:
            logger.warning(
                "research_lab_trajectory_projection_invalid run_id=%s errors=%s",
                run_id,
                "; ".join(projection.errors[:5]),
            )
            return ProjectionResult(
                run_id=run_id,
                status="failed",
                trajectory_id=tid,
                errors=projection.errors,
            )
        if dry_run:
            return ProjectionResult(
                run_id=run_id,
                status="dry_run",
                trajectory_id=tid,
                event_count=len(projection.event_rows),
                ledger_row_count=len(projection.ledger_rows),
            )
        # Envelope first (events FK-reference it), then append-only events in
        # seq order, then ledger rows.
        await store.insert_row(TRAJECTORIES_TABLE, projection.envelope_row)
        for row in projection.event_rows:
            await store.insert_row(TRAJECTORY_EVENTS_TABLE, row)
        for row in projection.ledger_rows:
            await store.insert_row(RESULTS_LEDGER_TABLE, row)
        logger.info(
            "research_lab_trajectory_projected run_id=%s trajectory_id=%s events=%s ledger_rows=%s",
            run_id,
            tid,
            len(projection.event_rows),
            len(projection.ledger_rows),
        )
        return ProjectionResult(
            run_id=run_id,
            status="projected",
            trajectory_id=tid,
            event_count=len(projection.event_rows),
            ledger_row_count=len(projection.ledger_rows),
        )
    except Exception as exc:  # never raise out of a projection
        logger.warning(
            "research_lab_trajectory_projection_failed run_id=%s error=%s",
            run_id,
            str(exc)[:500],
        )
        return ProjectionResult(
            run_id=run_id, status="failed", trajectory_id=tid, errors=[str(exc)[:500]]
        )


async def project_completed_runs(
    *,
    batch_size: int = 25,
    dry_run: bool = True,
    store: Any | None = None,
    max_candidates: int = 5000,
) -> list[ProjectionResult]:
    """Project up to ``batch_size`` not-yet-projected completed/failed runs.

    Suitable for a worker periodic pass or cron (wiring is a follow-up).
    Best-effort: a projection failure logs and skips that run; the batch never
    raises.
    """
    store = store or GatewayProjectorStore()
    results: list[ProjectionResult] = []
    try:
        queue_rows = await store.select_all(
            RUN_QUEUE_TABLE,
            columns="run_id,ticket_id,current_queue_status,current_status_at",
            filters=(("current_queue_status", "in", list(_QUEUE_TERMINAL_STATUSES)),),
            order_by=(("current_status_at", False),),
            max_rows=max_candidates,
        )
    except Exception as exc:
        logger.warning(
            "research_lab_trajectory_projector_discovery_failed error=%s",
            str(exc)[:500],
        )
        return results
    projected = 0
    for row in queue_rows:
        if projected >= max(1, int(batch_size)):
            break
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        try:
            existing = await store.select_one(
                TRAJECTORIES_TABLE,
                filters=(("trajectory_id", trajectory_id_for_run(run_id)),),
            )
        except Exception as exc:
            logger.warning(
                "research_lab_trajectory_projector_existence_check_failed run_id=%s error=%s",
                run_id,
                str(exc)[:200],
            )
            continue
        if existing:
            continue
        result = await project_run(run_id, store=store, dry_run=dry_run)
        results.append(result)
        if result.status in ("projected", "dry_run"):
            projected += 1
    return results


# ---------------------------------------------------------------------------
# CLI: python3 -m gateway.research_lab.trajectory_projector --backfill --dry-run
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gateway.research_lab.trajectory_projector",
        description=(
            "Project completed Research Lab runs into the v5 trajectory corpus "
            "tables (scripts/27). Dry-run by default; writes additionally "
            f"require {PROJECTOR_ENABLED_ENV}=1."
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="project every not-yet-projected completed/failed run",
    )
    parser.add_argument("--run-id", default=None, help="project a single run id")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="max runs to project per pass (backfill loops until drained)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate and report without writing (default: true)",
    )
    return parser


async def _cli_main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.backfill and not args.run_id:
        _build_arg_parser().print_help()
        return 2
    store = GatewayProjectorStore()
    results: list[ProjectionResult] = []
    if args.run_id:
        results.append(await project_run(args.run_id, store=store, dry_run=args.dry_run))
    if args.backfill:
        while True:
            batch = await project_completed_runs(
                batch_size=args.batch_size, dry_run=args.dry_run, store=store
            )
            results.extend(batch)
            attempted = [r for r in batch if r.status in ("projected", "dry_run")]
            if not attempted:
                break
            if args.dry_run:
                # A dry run writes nothing, so the discovery loop would find the
                # same runs forever; one pass reports everything projectable.
                break
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "projector_enabled": projector_enabled(),
                "corpus_contract_version": TRAJECTORY_CORPUS_CONTRACT_VERSION,
                "summary": summary,
                "results": [result.to_dict() for result in results],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(r.status != "failed" for r in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
