"""Research Lab v5 trajectory-capture projector (fableanalysis.md section 9.1).

An async/batch projector -- NOT a hot-path dual-write -- that reads the live
Research Lab event stream (``research_lab_auto_research_loop_events`` plus the
candidate / score / promotion event tables for a completed run) and projects it
into the v5 corpus tables created by ``scripts/27-research-lab-v5-schemas.sql``:

* ``research_trajectories``        -- one envelope row per run (no events array)
* ``research_trajectory_events``   -- append-only canonical event log
* ``research_lab_results_ledger``  -- one row per (run x drafted node)
* ``execution_traces``             -- pointer/metadata aggregation rows: one
  "engine" row per run (non-node-attributed raw LLM trace pointers and
  metadata-only reasoning captures) plus one row per node with node-scoped
  pointers (draft-stage raw traces, in-container sourcing-model trace refs),
  plus score-bundle fallback rows for scored bundles that are not linked to a
  node -- fablefollowup.md item 5.5
* ``evidence_bundles``             -- one row per scored candidate node (or
  score-bundle fallback) pointing at its score bundle + per-ICP evidence
  summary (numbers/refs only)

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
``execution_traces`` / ``evidence_bundles`` row ids are equally deterministic
(uuid5 over run/node + kind), and every row is existence-checked before insert,
so trace/evidence population is upsert-or-skip idempotent too.

Trajectory linkage is FORWARD-ONLY by design: ``research_trajectory_events``
is append-only by grant (UPDATE is revoked from service_role) and by trigger,
and ``anchored_hash`` covers the whole event body -- retrofitting refs into
already-projected events is impossible without breaking both.  New projections
embed ``execution_trace_ref`` (a real ``$defs/event`` field) on NODE_EVALUATED
and ``evidence_bundle_ref`` inside the nested ``evaluation_context``; runs
projected before this feature get their ``execution_traces`` /
``evidence_bundles`` rows via ``--traces-backfill`` WITHOUT event retrofit
(the deterministic row ids make the linkage recomputable from the run id).

Flags / entry points:

* ``RESEARCH_LAB_TRAJECTORY_PROJECTOR_ENABLED`` (default false) gates all
  writes; dry-run projection is always allowed (read-only).
* ``project_run(run_id, ...)`` and ``project_completed_runs(...)`` for a
  worker periodic pass or cron (wiring is a follow-up, not done here).
* CLI: ``python3 -m gateway.research_lab.trajectory_projector --backfill``
  (dry-run by default; pass ``--no-dry-run`` to write).
* CLI: ``python3 -m gateway.research_lab.trajectory_projector
  --traces-backfill`` adds missing ``execution_traces`` / ``evidence_bundles``
  rows to ALREADY-projected runs (dry-run by default).  Historical runs
  tolerate absent pointers everywhere: raw traces did not exist before item
  5.1 landed, so old runs mostly gain evidence-bundle rows only.

Follow-ups owned elsewhere: applying scripts/27, scorer-judge trace capture
(item 5.4 -- ``judge_verdicts`` stays ``[]`` until it lands), in-container
per-ICP trace refs reaching score-bundle docs (item 5.3 reader below is
tolerant of their absence), and folding the store facade below into
``gateway/research_lab/store.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_lab.axis_provenance import axis_rollup, provenance_for_stage
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


def _build_corpus_source_record(data: Mapping[str, Any]) -> TrajectoryCorpusSourceRecord:
    """Create a corpus source record across staged runtime schema versions."""
    if hasattr(TrajectoryCorpusSourceRecord, "from_mapping"):
        return TrajectoryCorpusSourceRecord.from_mapping(data)
    allowed = {item.name for item in fields(TrajectoryCorpusSourceRecord)}
    return TrajectoryCorpusSourceRecord(
        **{key: value for key, value in dict(data).items() if key in allowed}
    )


RESULTS_LEDGER_SCHEMA_NAME = "results_ledger_row.schema.json"

TRAJECTORIES_TABLE = "research_trajectories"
TRAJECTORY_EVENTS_TABLE = "research_trajectory_events"
RESULTS_LEDGER_TABLE = "research_lab_results_ledger"
EXECUTION_TRACES_TABLE = "execution_traces"
EVIDENCE_BUNDLES_TABLE = "evidence_bundles"

# scripts/27 CHECK-enforced enums (mirrored verbatim; tests re-parse the SQL).
EXECUTION_TRACE_ROLES: tuple[str, ...] = (
    "champion",
    "candidate",
    "shadow",
    "baseline_arm",
    "reference",
)
EXECUTION_TRACE_RUNGS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "anchor")
EXECUTION_TRACE_STATUSES: tuple[str, ...] = ("completed", "crash", "timeout")
EVIDENCE_RETENTION_CLASSES: tuple[str, ...] = (
    "live_verification",
    "regression_anchor",
    "general_snapshot",
)
EVIDENCE_VERIFICATION_STATES: tuple[str, ...] = (
    "active",
    "content_deleted",
    "hash_attested",
)

# Caps keep pointer rows bounded on pathological runs (batch-capped writes).
_EXECUTION_TRACE_CALL_CAP = 500
_EVIDENCE_SNAPSHOT_CAP = 64

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

    async def update_row(self, table: str, values: dict[str, Any], *, filters: Any):
        from gateway.research_lab import store

        return await store.update_row(table, values, filters=filters)


def _deterministic_uuid(*parts: Any) -> str:
    from gateway.research_lab.store import deterministic_uuid

    return deterministic_uuid(*parts)


def trajectory_id_for_run(run_id: str) -> str:
    return _deterministic_uuid("research_trajectory", str(run_id))


def execution_trace_id_for_run(run_id: str) -> str:
    """Deterministic ``execution_traces.run_id`` for the run's engine row."""
    return _deterministic_uuid("execution_trace", str(run_id), "engine")


def execution_trace_id_for_node(run_id: str, node_id: str) -> str:
    """Deterministic ``execution_traces.run_id`` for a node-scoped row."""
    return _deterministic_uuid("execution_trace", str(run_id), "node", str(node_id))


def execution_trace_id_for_score_bundle(run_id: str, score_bundle_id: str) -> str:
    """Deterministic trace row for a score bundle with no node linkage."""
    return _deterministic_uuid(
        "execution_trace", str(run_id), "score_bundle", str(score_bundle_id)
    )


def evidence_bundle_id_for_node(run_id: str, node_id: str, score_bundle_id: str) -> str:
    """Deterministic ``evidence_bundles.bundle_id`` for a scored node."""
    return _deterministic_uuid(
        "evidence_bundle", str(run_id), str(node_id), str(score_bundle_id)
    )


def evidence_bundle_id_for_score_bundle(run_id: str, score_bundle_id: str) -> str:
    """Deterministic evidence row for a score bundle with no node linkage."""
    return _deterministic_uuid(
        "evidence_bundle", str(run_id), "score_bundle", str(score_bundle_id)
    )


def _score_bundle_ref(score_bundle_id: str) -> str:
    """Live bundle ids are already ``score_bundle:``-prefixed; normalize."""
    text = str(score_bundle_id)
    return text if text.startswith("score_bundle:") else f"score_bundle:{text}"


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
# Corpus pointer collectors (fablefollowup.md item 5.5 -- pointers only)
# ---------------------------------------------------------------------------


def _iter_raw_trace_pointers(value: Any, model: Any = None):
    """Yield every ``raw_trace_ref``-shaped pointer in a live event row.

    Item 5.1 pointers ride ``provider_usage[*].raw_trace_ref`` and
    ``provider_usage[*].retry_attempts[*].raw_trace_ref`` today; the walk is
    deliberately recursive+tolerant so event-doc-nested pointers (or future
    shapes) are found too.  Yields ``(pointer_mapping, nearest_model)``.
    """
    if isinstance(value, Mapping):
        current_model = value.get("model") or model
        pointer = value.get("raw_trace_ref")
        if isinstance(pointer, Mapping):
            yield pointer, current_model
        for key, item in value.items():
            if str(key) == "raw_trace_ref":
                continue
            yield from _iter_raw_trace_pointers(item, current_model)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_raw_trace_pointers(item, model)


def _looks_like_provider_usage_item(value: Mapping[str, Any]) -> bool:
    return (
        isinstance(value.get("raw_trace_ref"), Mapping)
        or isinstance(value.get("reasoning_logs"), Mapping)
        or isinstance(value.get("reasoning_capture"), Mapping)
        or str(value.get("provider") or "").lower() == "openrouter"
        or bool(value.get("response_id"))
    )


def _iter_provider_usage_items(value: Any, model: Any = None):
    """Yield provider-usage-shaped mappings, including nested retry attempts."""
    if isinstance(value, Mapping):
        current_model = value.get("model") or model
        if _looks_like_provider_usage_item(value):
            yield value, current_model
        for key, item in value.items():
            if str(key) in {"raw_trace_ref", "reasoning_logs", "reasoning_capture"}:
                continue
            yield from _iter_provider_usage_items(item, current_model)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_provider_usage_items(item, model)


def _reasoning_hashes_from_logs(logs: Mapping[str, Any]) -> list[str]:
    hashes: list[str] = []
    raw_hashes = logs.get("reasoning_hashes")
    if isinstance(raw_hashes, Sequence) and not isinstance(raw_hashes, (str, bytes, bytearray)):
        hashes.extend(str(item) for item in raw_hashes if str(item))
    for key in ("reasoning_hash", "reasoning_details_hash"):
        if logs.get(key):
            hashes.append(str(logs[key]))
    choices = logs.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            for key in ("reasoning_hash", "reasoning_details_hash"):
                if choice.get(key):
                    hashes.append(str(choice[key]))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in hashes:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item[:256])
    return deduped[:32]


def _reasoning_fields_from_logs(logs: Mapping[str, Any]) -> list[str]:
    fields: set[str] = set()
    raw_fields = logs.get("fields_present")
    if isinstance(raw_fields, Sequence) and not isinstance(raw_fields, (str, bytes, bytearray)):
        fields.update(str(item)[:80] for item in raw_fields if str(item))
    for key in ("reasoning", "reasoning_details"):
        if logs.get(key) is not None:
            fields.add(key)
    choices = logs.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            choice_fields = choice.get("fields_present")
            if isinstance(choice_fields, Sequence) and not isinstance(choice_fields, (str, bytes, bytearray)):
                fields.update(str(item)[:80] for item in choice_fields if str(item))
            for key in ("reasoning", "reasoning_details"):
                if choice.get(key) is not None:
                    fields.add(key)
    return sorted(fields)


def _provider_reasoning_token_count(item: Mapping[str, Any], capture: Mapping[str, Any]) -> int | None:
    for holder in (capture, item, item.get("generation_stats")):
        if not isinstance(holder, Mapping):
            continue
        for key in ("reasoning_token_count", "native_tokens_reasoning", "reasoning_tokens", "tokens_reasoning"):
            if holder.get(key) is not None:
                return max(0, _i(holder.get(key)))
        details = holder.get("completion_tokens_details")
        if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
            return max(0, _i(details.get("reasoning_tokens")))
    return None


def _reasoning_capture_doc(item: Mapping[str, Any], *, raw_trace_ref_present: bool) -> dict[str, Any]:
    logs = item.get("reasoning_logs") if isinstance(item.get("reasoning_logs"), Mapping) else {}
    capture = item.get("reasoning_capture") if isinstance(item.get("reasoning_capture"), Mapping) else {}
    fields_present = list(capture.get("fields_present") or []) if capture else []
    if not fields_present and isinstance(logs, Mapping):
        fields_present = _reasoning_fields_from_logs(logs)
    hashes = list(capture.get("reasoning_hashes") or []) if capture else []
    if not hashes and isinstance(logs, Mapping):
        hashes = _reasoning_hashes_from_logs(logs)
    returned = bool(capture.get("returned")) if capture else bool(logs)
    requested = bool(capture.get("requested")) if capture else (returned or bool(item.get("reasoning_request_dropped")))
    token_count = _provider_reasoning_token_count(item, capture)
    storage_state = (
        "raw_trace_ref"
        if raw_trace_ref_present
        else ("metadata_only" if returned else ("requested_but_absent" if requested else "not_requested"))
    )
    storage_policy = str(
        capture.get("storage_policy")
        or (logs.get("storage_policy") if isinstance(logs, Mapping) else "")
        or ("raw_trace_s3_ref" if raw_trace_ref_present else "no_reasoning_returned")
    )[:160]
    doc: dict[str, Any] = {
        "requested": requested,
        "returned": returned,
        "fields_present": [str(field)[:80] for field in fields_present][:16],
        "reasoning_hashes": [str(item_hash)[:256] for item_hash in hashes][:32],
        "reasoning_token_count": token_count,
        "storage_state": storage_state,
        "storage_policy": storage_policy,
        "raw_trace_ref_present": raw_trace_ref_present,
    }
    if item.get("reasoning_request_dropped") or capture.get("request_dropped"):
        doc["request_dropped"] = True
    drop_hash = item.get("reasoning_effort_drop_error_hash") or capture.get("drop_error_hash")
    if drop_hash:
        doc["drop_error_hash"] = str(drop_hash)[:256]
    effort = item.get("requested_reasoning_effort") or capture.get("requested_reasoning_effort")
    if effort:
        doc["requested_reasoning_effort"] = str(effort)[:80]
    return sanitize_capture_payload(doc)


def _collect_engine_trace_calls(
    loop_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Engine-side raw LLM trace pointers grouped by stage/iteration/attempt.

    One ``calls`` entry per distinct ``{s3_ref, sha256}`` pointer or
    metadata-only reasoning capture, tagged with the emitting live event's
    stage (event_type), iteration, node and seq so the run's execution can be
    replayed pointer-by-pointer.  Pointers/hashes/metadata only -- never raw
    request/response/reasoning content.
    """
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(loop_events, key=lambda item: _i(item.get("seq"))):
        doc = _doc(row)
        stage = str(row.get("event_type") or "unknown_stage")
        node_id = str(row.get("node_id") or "") or None
        attempt = 0
        for item, model in _iter_provider_usage_items(
            [row.get("provider_usage"), doc]
        ):
            pointer = item.get("raw_trace_ref") if isinstance(item.get("raw_trace_ref"), Mapping) else {}
            s3_ref = str(pointer.get("s3_ref") or "") if isinstance(pointer, Mapping) else ""
            sha256 = str(pointer.get("sha256") or "") if isinstance(pointer, Mapping) else ""
            reasoning_logs = item.get("reasoning_logs") if isinstance(item.get("reasoning_logs"), Mapping) else {}
            if not s3_ref and not sha256 and not reasoning_logs:
                continue
            call_stage = str(item.get("call_stage") or stage or "unknown_stage")
            call_iteration = _i(item.get("loop_iteration"), _i(doc.get("iteration")))
            key = (
                "raw" if (s3_ref or sha256) else "metadata",
                s3_ref,
                sha256,
                str(row.get("event_id") or ""),
                str(item.get("response_id") or ""),
                call_stage,
            )
            if key in seen:
                continue
            seen.add(key)
            attempt += 1
            raw_trace_ref_present = bool(s3_ref or sha256)
            # P11: emitter/teacher/purpose are derived per call from the
            # auditable stage mapping — fixed engine stages are code-emitted
            # (axis-B); only source-inspection rounds are model-emitted.
            provenance = provenance_for_stage(call_stage)
            calls.append(
                sanitize_capture_payload(
                    {
                        "call_emitter": provenance["call_emitter"],
                        "purpose": provenance["purpose"] or call_stage[:128],
                        "component": provenance["component"],
                        "call_kind": "engine_raw_trace" if raw_trace_ref_present else "engine_reasoning_metadata",
                        "stage": call_stage[:128],
                        "iteration": call_iteration,
                        "attempt": attempt,
                        "node_id": node_id,
                        "live_event_id": str(row.get("event_id") or ""),
                        "live_seq": _i(row.get("seq")),
                        "model": (str(model or "") or "unknown")[:128],
                        "provider": str(item.get("provider") or "openrouter")[:80],
                        "response_id": str(item.get("response_id") or "")[:160],
                        "s3_ref": s3_ref,
                        "sha256": sha256,
                        "storage_state": "raw_trace_ref" if raw_trace_ref_present else "metadata_only",
                        "reasoning_capture": _reasoning_capture_doc(
                            item,
                            raw_trace_ref_present=raw_trace_ref_present,
                        ),
                        "teacher_model_flag": provenance["teacher_model_flag"],
                    }
                )
            )
    return calls


def _collect_incontainer_calls(
    bundle_row: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Item 5.3 in-container sourcing-model trace pointers from a score bundle.

    Per-ICP rows carry ``incontainer_trace_ref`` (S3 uri string) +
    ``incontainer_trace_sha256`` + ``incontainer_trace_call_count``.  That
    capture may still be landing, so the reader is a tolerant recursive scan
    over the whole bundle row/doc: absent refs simply yield zero entries.
    """
    if not isinstance(bundle_row, Mapping):
        return []
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _walk(value: Any, icp_ref: str | None) -> None:
        if isinstance(value, Mapping):
            row_icp = (
                str(value.get("icp_ref") or value.get("icp_hash") or "") or icp_ref
            )
            ref = value.get("incontainer_trace_ref")
            s3_ref = ""
            sha256 = str(value.get("incontainer_trace_sha256") or "")
            if isinstance(ref, str) and ref.strip():
                s3_ref = ref.strip()
            elif isinstance(ref, Mapping):  # tolerate {s3_ref, sha256} shape
                s3_ref = str(ref.get("s3_ref") or "")
                sha256 = sha256 or str(ref.get("sha256") or "")
            if s3_ref or sha256:
                key = (s3_ref, sha256)
                if key not in seen:
                    seen.add(key)
                    # P11 (amended): the in-container champion pipeline is
                    # axis-B by construction (v5 §8.3) — derived from the
                    # auditable mapping, not a per-stream constant.
                    incontainer_provenance = provenance_for_stage(
                        "incontainer_model_runtime"
                    )
                    entry = {
                        "call_emitter": incontainer_provenance["call_emitter"],
                        "purpose": incontainer_provenance["purpose"],
                        "component": incontainer_provenance["component"],
                        "stage": "incontainer_model_runtime",
                        "call_kind": "incontainer_trace",
                        "icp_ref": (row_icp or "unknown_icp")[:256],
                        "s3_ref": s3_ref,
                        "sha256": sha256,
                        "call_count": max(
                            0, _i(value.get("incontainer_trace_call_count"))
                        ),
                        "teacher_model_flag": incontainer_provenance[
                            "teacher_model_flag"
                        ],
                    }
                    # P13: truncated captures are flagged and filterable.
                    if value.get("incontainer_trace_truncated_count"):
                        entry["truncated"] = True
                        entry["truncated_call_count"] = max(
                            0, _i(value.get("incontainer_trace_truncated_count"))
                        )
                    # P5: dropped captures stay visible (and countable) in the
                    # corpus instead of masquerading as populated rows.
                    if value.get("incontainer_trace_dropped"):
                        entry["dropped"] = True
                        entry["dropped_call_count"] = max(
                            0, _i(value.get("incontainer_trace_dropped_call_count"))
                        )
                    calls.append(sanitize_capture_payload(entry))
            for item in value.values():
                _walk(item, row_icp)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item, icp_ref)

    _walk(bundle_row, None)
    return calls


def _collect_judge_verdicts(
    bundle_row: Mapping[str, Any] | None,
    *,
    candidate_id: str | None = None,
    score_bundle_id: str | None = None,
    scorer_trace_refs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """P2/P12: scorer judgment trace pointers as first-class judge verdicts.

    Sources, deduped by ``(s3_ref, sha256)``:

    * per-ICP bundle rows carrying ``scorer_trace_ref`` (the P12 passthrough);
    * the scored event doc's ``scorer_trace_refs`` map (``{icp_ref:
      {s3_ref, sha256}}``) — the only source for pre-P12 historical rows.

    Pointer/metadata only; the structured verdict content lives in the S3
    scorer trace doc.
    """
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(icp_ref: str, s3_ref: str, sha256: str, source: str) -> None:
        if not s3_ref and not sha256:
            return
        key = (s3_ref, sha256)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            sanitize_capture_payload(
                {
                    "verdict_kind": "scorer_judgment_trace",
                    "icp_ref": (icp_ref or "unknown_icp")[:256],
                    "s3_ref": s3_ref,
                    "sha256": sha256,
                    "candidate_ref": (
                        f"candidate:{candidate_id}" if candidate_id else None
                    ),
                    "score_bundle_ref": (
                        _score_bundle_ref(score_bundle_id) if score_bundle_id else None
                    ),
                    "is_reference_model": False,
                    "stage": "scorer_judgment",
                    "source": source,
                    "storage_state": "raw_trace_ref" if s3_ref else "metadata_only",
                    "teacher_model_flag": provenance_for_stage("scorer_judgment")[
                        "teacher_model_flag"
                    ],
                }
            )
        )

    for row in _bundle_per_icp_rows(bundle_row):
        _add(
            str(row.get("icp_ref") or row.get("icp_hash") or ""),
            str(row.get("scorer_trace_ref") or ""),
            str(row.get("scorer_trace_sha256") or ""),
            "per_icp_row",
        )
    for icp_ref, pointer in (scorer_trace_refs or {}).items():
        if isinstance(pointer, Mapping):
            _add(
                str(icp_ref),
                str(pointer.get("s3_ref") or ""),
                str(pointer.get("sha256") or ""),
                "scored_event",
            )
    return entries


def _collect_build_diagnostic_calls(
    loop_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """P4: build-failure diagnostic artifact pointers as corpus calls.

    Recursively scans event docs for ``diagnostic_artifact_uri`` /
    ``diagnostic_artifact_hash`` (bad diff, stderr/stdout, exit code live in
    the S3 artifact — never copied here). Dedupes by ``(s3_ref, sha256)``.
    """
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(loop_events, key=lambda item: _i(item.get("seq"))):
        doc = _doc(row)
        node_id = str(row.get("node_id") or "") or None

        def _walk(value: Any, iteration: int) -> None:
            if isinstance(value, Mapping):
                row_iteration = _i(value.get("iteration"), iteration)
                s3_ref = str(value.get("diagnostic_artifact_uri") or "")
                sha256 = str(value.get("diagnostic_artifact_hash") or "")
                if s3_ref or sha256:
                    key = (s3_ref, sha256)
                    if key not in seen:
                        seen.add(key)
                        target_files = value.get("target_files") or doc.get("target_files") or []
                        if not isinstance(target_files, Sequence) or isinstance(
                            target_files, (str, bytes)
                        ):
                            target_files = []
                        calls.append(
                            sanitize_capture_payload(
                                {
                                    "call_emitter": "code",
                                    "call_kind": "build_diagnostic_artifact",
                                    "artifact_kind": "code_build_failure",
                                    "s3_ref": s3_ref,
                                    "sha256": sha256,
                                    "node_id": node_id,
                                    "iteration": row_iteration,
                                    "target_files": [
                                        str(path)[:200] for path in list(target_files)[:16]
                                    ],
                                    "live_event_id": str(row.get("event_id") or ""),
                                    "live_seq": _i(row.get("seq")),
                                    "storage_state": (
                                        "raw_trace_ref" if s3_ref else "metadata_only"
                                    ),
                                    "diagnostic_write_failed": bool(
                                        value.get("diagnostic_artifact_error")
                                    ),
                                    "teacher_model_flag": False,
                                }
                            )
                        )
                for item in value.values():
                    _walk(item, row_iteration)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _walk(item, iteration)

        _walk(doc, _i(doc.get("iteration")))
    return calls


def _cap_pointer_entries(
    entries: Sequence[Mapping[str, Any]], cap: int, kind: str
) -> list[dict[str, Any]]:
    items = [dict(entry) for entry in entries]
    if len(items) <= cap:
        return items
    omitted = len(items) - cap
    return [*items[:cap], {"call_kind": kind, "omitted_count": omitted}]


def _bundle_doc(bundle_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bundle_row, Mapping):
        return {}
    doc = bundle_row.get("score_bundle_doc")
    return dict(doc) if isinstance(doc, Mapping) else {}


def _bundle_per_icp_rows(bundle_row: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    doc = _bundle_doc(bundle_row)
    for holder in (doc.get("aggregates"), doc):
        if isinstance(holder, Mapping):
            rows = holder.get("per_icp_results")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                found = [row for row in rows if isinstance(row, Mapping)]
                if found:
                    return found
    return []


def _score_count(row: Mapping[str, Any], key: str) -> int:
    scores = row.get(key)
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        return len(scores)
    return 0


def _per_icp_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    """One evidence snapshot per scored ICP: numbers and refs only.

    ``failure_reason`` free text is deliberately dropped (it can embed
    provider error strings); status/exclusion booleans carry the signal.
    """
    snapshot: dict[str, Any] = {
        "snapshot_kind": "per_icp_score_evidence",
        "icp_ref": _sanitize_text(str(row.get("icp_ref") or ""))[:256],
        "icp_hash": str(row.get("icp_hash") or "")[:256],
        "status": _sanitize_text(str(row.get("status") or "unknown"))[:64],
        "hard_failure": bool(row.get("hard_failure")),
        "provider_excluded": bool(row.get("provider_excluded")),
        "candidate_per_icp_score": _f(row.get("candidate_per_icp_score")),
        "base_per_icp_score": _f(row.get("base_per_icp_score")),
        "delta_vs_base": _f(row.get("delta_vs_base")),
        "candidate_company_score_count": _score_count(row, "candidate_company_scores"),
        "base_company_score_count": _score_count(row, "base_company_scores"),
    }
    # P14: a zero-company ICP is a first-class negative example — record the
    # empty-output judgment context instead of leaving the row ambiguous.
    if row.get("sourced_zero_no_error"):
        snapshot["sourced_zero_no_error"] = True
        snapshot["judgment_context"] = "candidate_sourced_zero_companies"
    # P12 dense reward: per-claim L0 verdicts (check ids/statuses only) and
    # the per-ICP scorer judgment pointer become corpus-reachable evidence.
    scorer_ref = row.get("scorer_trace_ref")
    if isinstance(scorer_ref, str) and scorer_ref.strip():
        snapshot["scorer_trace_ref"] = scorer_ref.strip()
        snapshot["scorer_trace_sha256"] = str(row.get("scorer_trace_sha256") or "")
    l0_findings = row.get("l0_findings")
    if isinstance(l0_findings, Sequence) and not isinstance(l0_findings, (str, bytes)):
        bounded = [dict(item) for item in l0_findings if isinstance(item, Mapping)][:64]
        if bounded:
            snapshot["l0_findings"] = bounded
    incontainer_ref = row.get("incontainer_trace_ref")
    if isinstance(incontainer_ref, str) and incontainer_ref.strip():
        snapshot["incontainer_trace_ref"] = incontainer_ref.strip()
        snapshot["incontainer_trace_sha256"] = str(
            row.get("incontainer_trace_sha256") or ""
        )
        snapshot["incontainer_trace_call_count"] = max(
            0, _i(row.get("incontainer_trace_call_count"))
        )
    elif row.get("incontainer_trace_dropped"):
        # P5: dropped in-container captures stay visible in evidence snapshots.
        snapshot["incontainer_trace_dropped"] = True
        snapshot["incontainer_trace_dropped_call_count"] = max(
            0, _i(row.get("incontainer_trace_dropped_call_count"))
        )
        snapshot["incontainer_trace_sha256"] = str(
            row.get("incontainer_trace_sha256") or ""
        )
    return snapshot


def _holdout_gate_summary(gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Numbers-and-refs projection of the private-holdout gate doc."""
    if not isinstance(gate, Mapping) or not gate:
        return {}
    summary: dict[str, Any] = {
        "gate_type": _sanitize_text(str(gate.get("gate_type") or ""))[:64],
        "decision": _sanitize_text(str(gate.get("decision") or ""))[:64],
        "public_icp_count": _i(gate.get("public_icp_count")),
        "private_holdout_icp_count": _i(gate.get("private_holdout_icp_count")),
        "private_holdout_evaluated": bool(gate.get("private_holdout_evaluated")),
        "baseline_aggregate_score": _f(gate.get("baseline_aggregate_score")),
        "candidate_total_score": _f(gate.get("candidate_total_score")),
        "candidate_delta_vs_daily_baseline": _f(gate.get(TARGETED_METRIC)),
        "provider_excluded_icp_ids": [
            _sanitize_text(str(item))[:256]
            for item in (
                gate.get("provider_excluded_icp_ids")
                if isinstance(gate.get("provider_excluded_icp_ids"), Sequence)
                and not isinstance(gate.get("provider_excluded_icp_ids"), (str, bytes))
                else []
            )
            if str(item)
        ][:64],
    }
    baseline_bundle_id = str(gate.get("baseline_benchmark_bundle_id") or "")
    if baseline_bundle_id:
        summary["baseline_benchmark_bundle_ref"] = (
            f"benchmark_bundle:{baseline_bundle_id}"[:256]
        )
    baseline_hash = str(gate.get("baseline_benchmark_hash") or "")
    if baseline_hash:
        summary["baseline_benchmark_hash"] = baseline_hash[:256]
    return summary


def _aggregates_summary(bundle_row: Mapping[str, Any] | None) -> dict[str, Any]:
    doc = _bundle_doc(bundle_row)
    aggregates = doc.get("aggregates")
    if not isinstance(aggregates, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in ("icp_count", "successful_icp_count", "hard_failure_count"):
        if aggregates.get(key) is not None:
            summary[key] = _i(aggregates.get(key))
    for key in (
        "base_score",
        "candidate_score",
        "mean_delta",
        "delta_lcb",
        "total_cost_usd",
    ):
        if aggregates.get(key) is not None:
            summary[key] = _f(aggregates.get(key))
    return summary


def _eval_version_doc(bundle_row: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"engine_version": ENGINE_VERSION}
    if isinstance(bundle_row, Mapping):
        doc = _bundle_doc(bundle_row)
        for key in ("scoring_version", "evaluator_version"):
            value = bundle_row.get(key) or doc.get(key)
            if value:
                out[key] = str(value)[:128]
    return out


def _score_bundle_artifact_hash(
    bundle_row: Mapping[str, Any] | None, score_bundle_id: str
) -> str:
    """Best available artifact hash for a score-bundle-only trace row."""
    doc = _bundle_doc(bundle_row)
    row = bundle_row if isinstance(bundle_row, Mapping) else {}
    for key in (
        "candidate_artifact_hash",
        "private_model_manifest_hash",
        "candidate_patch_hash",
        "score_bundle_hash",
    ):
        value = row.get(key) or doc.get(key)
        if value:
            return str(value)
    return sha256_json({"score_bundle_artifact_unavailable": str(score_bundle_id)})


def _score_bundle_execution_status(bundle_row: Mapping[str, Any] | None) -> str:
    """Map score-bundle lifecycle states onto execution_traces statuses."""
    if not isinstance(bundle_row, Mapping):
        return "timeout"
    status = str(
        bundle_row.get("bundle_status")
        or bundle_row.get("current_event_status")
        or bundle_row.get("current_event_type")
        or ""
    ).lower()
    if status in {"failed", "rejected", "tombstoned"}:
        return "crash"
    if status in {"scored", "verified"}:
        return "completed"
    doc = _bundle_doc(bundle_row)
    if bundle_row.get("score_bundle_hash") or doc.get("score_bundle_hash"):
        return "completed"
    return "timeout"


def _score_bundle_total_cost_usd(bundle_row: Mapping[str, Any] | None) -> float:
    summary = _aggregates_summary(bundle_row)
    if summary.get("total_cost_usd") is not None:
        return _f(summary.get("total_cost_usd"))
    doc = _bundle_doc(bundle_row)
    for holder in (doc.get("aggregates"), doc, bundle_row):
        if isinstance(holder, Mapping) and holder.get("total_cost_usd") is not None:
            return _f(holder.get("total_cost_usd"))
    return 0.0


def _trace_role(bundle_row: Mapping[str, Any] | None, default: str = "candidate") -> str:
    """P15: derive the execution-trace role from the real arm.

    Bundles (or their docs) may stamp ``evaluation_role`` as
    champion/candidate/shadow/baseline_arm/reference; a stored-daily-baseline
    reference evaluation projects as ``baseline_arm``. Unknown/absent → the
    caller's default — but the value is data-driven, never a per-stream
    constant, so future champion/shadow projection streams label correctly.
    """
    doc = _bundle_doc(bundle_row)
    for holder in (bundle_row or {}, doc, doc.get("aggregates") or {}):
        if not isinstance(holder, Mapping):
            continue
        role = str(holder.get("evaluation_role") or "").strip()
        if role in EXECUTION_TRACE_ROLES:
            return role
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
    if str(aggregates.get("reference_evaluation_mode") or "") == "stored_daily_baseline_reference":
        return "baseline_arm"
    return default


def _node_execution_status(state: "_NodeState") -> str:
    """Map node lifecycle onto execution_traces' completed/crash/timeout CHECK."""
    if state.build_failed_row is not None and state.build_passed_row is None:
        return "crash"
    if state.scoring_crashed and not state.score_bundle_id:
        # P14: a crashed scoring is a crash row, not an invisible node.
        return "crash"
    if state.gate_doc:
        return "completed"
    return "timeout"  # built-but-never-scored, or drafted-but-never-built


def _execution_trace_trajectory_id_enabled() -> bool:
    """P18: include the explicit ``trajectory_id`` join key on execution trace
    rows. Requires the scripts/64 column — leave off until applied (an unknown
    column fails the insert)."""
    return os.getenv(
        "RESEARCH_LAB_EXECUTION_TRACE_TRAJECTORY_ID_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _run_summary_contract_enforced() -> bool:
    """P14 / v5 §8.3: when on, a terminal loop event WITHOUT the run-summary
    block classifies the run as crash — absence of the summary IS the crash
    detector. Default off until the fleet has emitted summaries long enough
    that historical rows are the exception."""
    return os.getenv(
        "RESEARCH_LAB_RUN_SUMMARY_CONTRACT_ENFORCED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _terminal_run_summary(terminal_row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if terminal_row is None:
        return None
    summary = _doc(terminal_row).get("run_summary")
    return summary if isinstance(summary, Mapping) and summary else None


def _engine_execution_status(
    queue_status: str, terminal_row: Mapping[str, Any] | None
) -> str:
    terminal_type = str(terminal_row.get("event_type") or "") if terminal_row else ""
    if queue_status == "completed" or terminal_type == "loop_completed":
        if (
            _run_summary_contract_enforced()
            and terminal_row is not None
            and _terminal_run_summary(terminal_row) is None
        ):
            return "crash"
        return "completed"
    if queue_status == "failed" or terminal_type == "loop_failed":
        return "crash"
    return "timeout"


def _build_corpus_trace_rows(
    *,
    run_id: str,
    trajectory_id: str,
    champion_base: str,
    nodes: Mapping[str, "_NodeState"],
    node_order: Sequence[str],
    engine_calls: Sequence[Mapping[str, Any]],
    build_diagnostic_calls: Sequence[Mapping[str, Any]] = (),
    bundles_by_id: Mapping[str, Mapping[str, Any]],
    reflections_by_node: Mapping[str, Sequence[Mapping[str, Any]]],
    queue_status: str,
    terminal_row: Mapping[str, Any] | None,
    total_cost_usd: float,
    fallback_ts: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build execution_traces + evidence_bundles rows (pointers only).

    Row policy (absence-tolerant by construction):

    * engine row  -- written iff the run has non-node-attributed raw-trace
      pointers (historical runs have none and get no engine row);
    * node row    -- written iff the node was scored (score bundle pointer
      exists) OR node-attributed trace pointers exist;
    * evidence row -- written iff the node was scored; ``snapshots`` falls
      back to a single score-bundle summary entry when the bundle doc carries
      no per-ICP rows (scripts/27 CHECKs snapshots non-empty);
    * score-bundle fallback row -- written iff a score bundle exists for the
      run but no scored node claimed it.  This covers late/reused/manual
      scoring paths whose rich bundle data would otherwise be present in the
      source table but absent from the v5 corpus pointer tables.
    """
    execution_trace_rows: list[dict[str, Any]] = []
    evidence_bundle_rows: list[dict[str, Any]] = []
    run_ref = f"research_loop_run:{run_id}"
    trajectory_ref = f"trajectory:{trajectory_id}"
    cost_ledger_ref = f"cost_ledger:{run_id}"
    default_icp_set_hash = next(
        (
            str(row.get("icp_set_hash"))
            for row in bundles_by_id.values()
            if isinstance(row, Mapping) and row.get("icp_set_hash")
        ),
        "",
    ) or sha256_json({"icp_set_hash_unavailable_for_run": run_id})
    first_bundle_id = next(iter(bundles_by_id), "")
    used_score_bundle_ids: set[str] = set()

    for node_id in node_order:
        state = nodes[node_id]
        bundle_row = bundles_by_id.get(state.score_bundle_id or "")
        node_calls = [
            dict(call) for call in engine_calls if call.get("node_id") == node_id
        ]
        node_diagnostic_calls = [
            dict(call)
            for call in build_diagnostic_calls
            if call.get("node_id") == node_id
        ]
        incontainer_calls = _collect_incontainer_calls(bundle_row)
        judge_verdicts = _collect_judge_verdicts(
            bundle_row,
            candidate_id=state.candidate_id,
            score_bundle_id=state.score_bundle_id,
            scorer_trace_refs=state.scorer_trace_refs,
        )
        if (
            not state.score_bundle_id
            and not node_calls
            and not incontainer_calls
            and not node_diagnostic_calls
            and not state.scoring_crashed
        ):
            continue  # nothing to point at for this node
        trace_id = execution_trace_id_for_node(run_id, node_id)
        anchor_row = state.build_passed_row or state.build_failed_row or state.drafted_row
        created_at = _ts(anchor_row.get("created_at"), fallback_ts)
        node_icp_set_hash = (
            str(bundle_row.get("icp_set_hash"))
            if isinstance(bundle_row, Mapping) and bundle_row.get("icp_set_hash")
            else default_icp_set_hash
        )
        evidence_refs: list[str] = []
        score_bundle_ref = "score_bundle:unavailable"
        outputs_ref = (
            f"candidate:{state.candidate_id}"
            if state.candidate_id
            else "outputs:unavailable"
        )

        if state.score_bundle_id:
            used_score_bundle_ids.add(state.score_bundle_id)
            score_bundle_ref = _score_bundle_ref(state.score_bundle_id)
            outputs_ref = score_bundle_ref
            evidence_id = evidence_bundle_id_for_node(
                run_id, node_id, state.score_bundle_id
            )
            evidence_refs = [f"evidence_bundle:{evidence_id}"]
            per_icp_rows = _bundle_per_icp_rows(bundle_row)
            bundle_doc = _bundle_doc(bundle_row)
            score_bundle_hash = str(
                (bundle_row or {}).get("score_bundle_hash")
                or bundle_doc.get("score_bundle_hash")
                or ""
            )
            # The bundle doc's gate is canonical (the scored event's copy is a
            # thinner derivative); fall back to the event gate for old bundles.
            raw_bundle_gate = bundle_doc.get("private_holdout_gate")
            gate = (
                raw_bundle_gate
                if isinstance(raw_bundle_gate, Mapping) and raw_bundle_gate
                else state.gate_doc
            )
            if per_icp_rows:
                snapshots = _cap_pointer_entries(
                    [_per_icp_snapshot(row) for row in per_icp_rows],
                    _EVIDENCE_SNAPSHOT_CAP,
                    "per_icp_truncation_marker",
                )
            else:
                gate_summary = _holdout_gate_summary(gate)
                snapshots = [
                    {
                        "snapshot_kind": "score_bundle_summary",
                        "score_bundle_ref": score_bundle_ref,
                        "score_bundle_hash": score_bundle_hash,
                        "icp_count": _i(
                            _aggregates_summary(bundle_row).get("icp_count")
                        )
                        or (
                            gate_summary.get("public_icp_count", 0)
                            + gate_summary.get("private_holdout_icp_count", 0)
                        ),
                        "per_icp_rows_available": False,
                    }
                ]
            evidence_doc = sanitize_capture_payload(
                {
                    "evidence_kind": "candidate_score_evidence",
                    "node_id": node_id,
                    "candidate_ref": (
                        f"candidate:{state.candidate_id}" if state.candidate_id else None
                    ),
                    "lab_run_ref": run_ref,
                    "trajectory_ref": trajectory_ref,
                    "execution_trace_ref": f"execution_trace:{trace_id}",
                    "score_bundle_ref": score_bundle_ref,
                    "score_bundle_hash": score_bundle_hash,
                    "evaluation_epoch": (
                        _i(bundle_row.get("evaluation_epoch"))
                        if isinstance(bundle_row, Mapping)
                        and bundle_row.get("evaluation_epoch") is not None
                        else None
                    ),
                    "icp_set_hash": node_icp_set_hash,
                    "holdout_gate": _holdout_gate_summary(gate) or None,
                    "aggregates": _aggregates_summary(bundle_row) or None,
                    "reflections": [
                        dict(item) for item in reflections_by_node.get(node_id, ())
                    ],
                    "per_icp_snapshot_count": len(snapshots),
                    "incontainer_trace_count": len(incontainer_calls),
                    "incontainer_call_count_total": sum(
                        max(0, _i(call.get("call_count"))) for call in incontainer_calls
                    ),
                }
            )
            snapshots = sanitize_capture_payload(snapshots)
            evidence_created_at = _ts(
                (bundle_row or {}).get("created_at"), created_at
            )
            evidence_bundle_rows.append(
                {
                    "bundle_id": evidence_id,
                    "schema_version": "1.0",
                    "run_id": _coerce_uuid(run_id),
                    "artifact_hash": state.candidate_artifact_hash
                    or state.unified_diff_hash,
                    "retention_class": "live_verification",
                    "verification_state": "active",
                    "bundle_hash": sha256_json(
                        {
                            "evidence_bundle_id": evidence_id,
                            "snapshots": snapshots,
                            "bundle_doc": evidence_doc,
                        }
                    ),
                    "merkle_anchor_ref": None,
                    "deletion_request_ref": None,
                    "snapshots": snapshots,
                    "bundle_doc": evidence_doc,
                    "created_at": evidence_created_at,
                }
            )

        trace_doc = sanitize_capture_payload(
            {
                "trace_kind": "candidate_node",
                "node_id": node_id,
                "lane": _sanitize_text(state.lane)[:128],
                "plan_path_id": state.plan_path_id,
                "lab_run_ref": run_ref,
                "trajectory_ref": trajectory_ref,
                "candidate_ref": (
                    f"candidate:{state.candidate_id}" if state.candidate_id else None
                ),
                "engine_call_count": len(node_calls),
                "incontainer_trace_count": len(incontainer_calls),
                "build_diagnostic_count": len(node_diagnostic_calls),
                "judge_verdict_count": len(judge_verdicts),
                "reflection_count": len(reflections_by_node.get(node_id, ())),
                **(
                    {
                        "scoring_crashed": True,
                        "scoring_failure_event_type": state.scoring_failure_event_type,
                    }
                    if state.scoring_crashed and not state.score_bundle_id
                    else {}
                ),
                # P11: v5 §8.3 trace-level axis rollup — conjunction over the
                # control-flow-driving calls (axis_a only if all model-emitted).
                "trajectory_axis": axis_rollup([*node_calls, *incontainer_calls]),
            }
        )
        execution_trace_rows.append(
            {
                "run_id": trace_id,
                "schema_version": "1.0",
                "artifact_hash": state.candidate_artifact_hash
                or state.unified_diff_hash,
                "role": _trace_role(bundle_row),
                "rung": "L0",
                "status": _node_execution_status(state),
                "lane_id": None,  # live lanes are labels, not UUIDs (see trace_doc)
                "icp_set_hash": node_icp_set_hash,
                "eval_version": _eval_version_doc(bundle_row),
                "calls": _cap_pointer_entries(
                    [*node_calls, *node_diagnostic_calls, *incontainer_calls],
                    _EXECUTION_TRACE_CALL_CAP,
                    "call_truncation_marker",
                ),
                "evidence_bundles": evidence_refs,
                "judge_verdicts": _cap_pointer_entries(
                    judge_verdicts,
                    _EXECUTION_TRACE_CALL_CAP,
                    "judge_verdict_truncation_marker",
                ),
                "outputs_ref": outputs_ref,
                "score_bundle_ref": score_bundle_ref,
                "cost_ledger": {
                    "cost_ledger_ref": cost_ledger_ref,
                    "node_cost_usd": round(max(0.0, state.cost_usd), 6),
                },
                "attestation_ref": None,
                "trace_doc": trace_doc,
                "created_at": created_at,
            }
        )

    for score_bundle_id, bundle_row in bundles_by_id.items():
        if not score_bundle_id or score_bundle_id == "None":
            continue
        if score_bundle_id in used_score_bundle_ids:
            continue

        score_bundle_ref = _score_bundle_ref(score_bundle_id)
        trace_id = execution_trace_id_for_score_bundle(run_id, score_bundle_id)
        evidence_id = evidence_bundle_id_for_score_bundle(run_id, score_bundle_id)
        evidence_ref = f"evidence_bundle:{evidence_id}"
        bundle_doc = _bundle_doc(bundle_row)
        score_bundle_hash = str(
            bundle_row.get("score_bundle_hash")
            or bundle_doc.get("score_bundle_hash")
            or ""
        )
        artifact_hash = _score_bundle_artifact_hash(bundle_row, score_bundle_id)
        node_icp_set_hash = (
            str(bundle_row.get("icp_set_hash") or bundle_doc.get("icp_set_hash") or "")
            or default_icp_set_hash
        )
        created_at = _ts(bundle_row.get("created_at"), fallback_ts)
        incontainer_calls = _collect_incontainer_calls(bundle_row)
        judge_verdicts = _collect_judge_verdicts(
            bundle_row, score_bundle_id=score_bundle_id
        )
        per_icp_rows = _bundle_per_icp_rows(bundle_row)
        raw_bundle_gate = bundle_doc.get("private_holdout_gate")
        gate = raw_bundle_gate if isinstance(raw_bundle_gate, Mapping) else {}
        gate_summary = _holdout_gate_summary(gate)
        aggregates_summary = _aggregates_summary(bundle_row)
        if per_icp_rows:
            snapshots = _cap_pointer_entries(
                [_per_icp_snapshot(row) for row in per_icp_rows],
                _EVIDENCE_SNAPSHOT_CAP,
                "per_icp_truncation_marker",
            )
        else:
            snapshots = [
                {
                    "snapshot_kind": "score_bundle_summary",
                    "score_bundle_ref": score_bundle_ref,
                    "score_bundle_hash": score_bundle_hash,
                    "icp_count": _i(aggregates_summary.get("icp_count"))
                    or (
                        gate_summary.get("public_icp_count", 0)
                        + gate_summary.get("private_holdout_icp_count", 0)
                    ),
                    "per_icp_rows_available": False,
                }
            ]
        evidence_doc = sanitize_capture_payload(
            {
                "evidence_kind": "score_bundle_evidence",
                "source": "score_bundle_fallback",
                "fallback_reason": "score_bundle_not_linked_to_node",
                "lab_run_ref": run_ref,
                "trajectory_ref": trajectory_ref,
                "execution_trace_ref": f"execution_trace:{trace_id}",
                "score_bundle_ref": score_bundle_ref,
                "score_bundle_hash": score_bundle_hash,
                "score_bundle_status": bundle_row.get("bundle_status")
                or bundle_row.get("current_event_status")
                or bundle_row.get("current_event_type"),
                "candidate_artifact_hash": artifact_hash,
                "evaluation_epoch": (
                    _i(bundle_row.get("evaluation_epoch"))
                    if bundle_row.get("evaluation_epoch") is not None
                    else None
                ),
                "icp_set_hash": node_icp_set_hash,
                "holdout_gate": gate_summary or None,
                "aggregates": aggregates_summary or None,
                "per_icp_snapshot_count": len(snapshots),
                "incontainer_trace_count": len(incontainer_calls),
                "incontainer_call_count_total": sum(
                    max(0, _i(call.get("call_count"))) for call in incontainer_calls
                ),
            }
        )
        snapshots = sanitize_capture_payload(snapshots)
        evidence_bundle_rows.append(
            {
                "bundle_id": evidence_id,
                "schema_version": "1.0",
                "run_id": _coerce_uuid(run_id),
                "artifact_hash": artifact_hash,
                "retention_class": "live_verification",
                "verification_state": "active",
                "bundle_hash": sha256_json(
                    {
                        "evidence_bundle_id": evidence_id,
                        "snapshots": snapshots,
                        "bundle_doc": evidence_doc,
                    }
                ),
                "merkle_anchor_ref": None,
                "deletion_request_ref": None,
                "snapshots": snapshots,
                "bundle_doc": evidence_doc,
                "created_at": created_at,
            }
        )
        trace_doc = sanitize_capture_payload(
            {
                "trace_kind": "score_bundle_only",
                "source": "score_bundle_fallback",
                "fallback_reason": "score_bundle_not_linked_to_node",
                "lab_run_ref": run_ref,
                "trajectory_ref": trajectory_ref,
                "score_bundle_ref": score_bundle_ref,
                "score_bundle_hash": score_bundle_hash,
                "candidate_artifact_hash": artifact_hash,
                "incontainer_trace_count": len(incontainer_calls),
                "per_icp_snapshot_count": len(snapshots),
                "trajectory_axis": axis_rollup(incontainer_calls),
            }
        )
        execution_trace_rows.append(
            {
                "run_id": trace_id,
                "schema_version": "1.0",
                "artifact_hash": artifact_hash,
                "role": _trace_role(bundle_row),
                "rung": "L0",
                "status": _score_bundle_execution_status(bundle_row),
                "lane_id": None,
                "icp_set_hash": node_icp_set_hash,
                "eval_version": _eval_version_doc(bundle_row),
                "calls": _cap_pointer_entries(
                    incontainer_calls,
                    _EXECUTION_TRACE_CALL_CAP,
                    "call_truncation_marker",
                ),
                "evidence_bundles": [evidence_ref],
                "judge_verdicts": _cap_pointer_entries(
                    judge_verdicts,
                    _EXECUTION_TRACE_CALL_CAP,
                    "judge_verdict_truncation_marker",
                ),
                "outputs_ref": score_bundle_ref,
                "score_bundle_ref": score_bundle_ref,
                "cost_ledger": {
                    "cost_ledger_ref": cost_ledger_ref,
                    "score_bundle_cost_usd": round(
                        max(0.0, _score_bundle_total_cost_usd(bundle_row)), 6
                    ),
                },
                "attestation_ref": None,
                "trace_doc": trace_doc,
                "created_at": created_at,
            }
        )

    run_scoped_calls = [
        dict(call) for call in engine_calls if not call.get("node_id")
    ]
    run_scoped_calls.extend(
        dict(call) for call in build_diagnostic_calls if not call.get("node_id")
    )
    if run_scoped_calls:
        execution_trace_rows.insert(
            0,
            {
                "run_id": execution_trace_id_for_run(run_id),
                "schema_version": "1.0",
                "artifact_hash": champion_base,
                # The engine loop generates candidates; role stays data-driven
                # for bundle-backed rows above.
                "role": "candidate",
                "rung": "L0",
                "status": _engine_execution_status(queue_status, terminal_row),
                "lane_id": None,
                "icp_set_hash": default_icp_set_hash,
                "eval_version": _eval_version_doc(bundles_by_id.get(first_bundle_id)),
                "calls": _cap_pointer_entries(
                    run_scoped_calls,
                    _EXECUTION_TRACE_CALL_CAP,
                    "call_truncation_marker",
                ),
                # Node-scoped evidence refs live on the node rows; the engine
                # row's evidence is discoverable via evidence_bundles.run_id.
                "evidence_bundles": [],
                "judge_verdicts": [],  # scorer-judge traces are item 5.4
                "outputs_ref": trajectory_ref,
                "score_bundle_ref": (
                    _score_bundle_ref(first_bundle_id)
                    if first_bundle_id
                    else "score_bundle:unavailable"
                ),
                "cost_ledger": {
                    "cost_ledger_ref": cost_ledger_ref,
                    "total_usd": round(max(0.0, total_cost_usd), 6),
                },
                "attestation_ref": None,
                "trace_doc": sanitize_capture_payload(
                    {
                        "trace_kind": "engine_loop",
                        "lab_run_ref": run_ref,
                        "trajectory_ref": trajectory_ref,
                        "engine_call_count": len(run_scoped_calls),
                        "node_ids": [str(node_id) for node_id in node_order][:64],
                        "trajectory_axis": axis_rollup(run_scoped_calls),
                        # P14 run-summary contract: countable without forensics.
                        "run_summary_present": _terminal_run_summary(terminal_row)
                        is not None,
                    }
                ),
                "created_at": fallback_ts,
            },
        )
    # P18: explicit trajectory join key (scripts/64 column, env-gated so
    # pre-migration fleets keep inserting cleanly).
    if _execution_trace_trajectory_id_enabled():
        trajectory_uuid = _coerce_uuid(trajectory_id) or str(trajectory_id)
        for row in execution_trace_rows:
            row["trajectory_id"] = trajectory_uuid
    return execution_trace_rows, evidence_bundle_rows


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
    execution_trace_rows: list[dict[str, Any]] = field(default_factory=list)
    evidence_bundle_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ProjectionResult:
    run_id: str
    status: str  # projected | dry_run | skipped_existing | skipped_disabled |
    #             skipped_incomplete | failed | traces_backfilled |
    #             traces_dry_run | skipped_traces_existing |
    #             skipped_unprojected | skipped_no_trace_sources
    trajectory_id: str | None = None
    event_count: int = 0
    ledger_row_count: int = 0
    execution_trace_count: int = 0
    evidence_bundle_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "trajectory_id": self.trajectory_id,
            "event_count": self.event_count,
            "ledger_row_count": self.ledger_row_count,
            "execution_trace_count": self.execution_trace_count,
            "evidence_bundle_count": self.evidence_bundle_count,
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
    # P2: {icp_ref: {s3_ref, sha256}} scorer judgment pointers from the
    # scored event doc — projected into execution_traces.judge_verdicts.
    scorer_trace_refs: Mapping[str, Mapping[str, Any]] | None = None
    # P14: candidate scoring crashed/was rejected with no bundle — the node
    # still gets an execution_traces row with status="crash".
    scoring_crashed: bool = False
    scoring_failure_event_type: str = ""


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
    failed_eval_by_candidate: dict[str, Mapping[str, Any]] = {}
    for row in evaluation_events:
        event_type = str(row.get("event_type"))
        if event_type == "scored":
            scored_by_candidate[str(row.get("candidate_id"))] = row
        elif event_type in {"failed", "rejected"}:
            # P14: crashed/rejected scorings must still yield corpus rows.
            failed_eval_by_candidate.setdefault(str(row.get("candidate_id")), row)

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

    # Engine raw-trace pointers (item 5.1) are collected up front so the
    # NODE_EVALUATED emitter can embed each node's deterministic
    # execution_trace ref only when a row will actually be written.
    engine_calls = _collect_engine_trace_calls(ordered)
    # P4: build-failure diagnostic pointers become first-class corpus calls.
    build_diagnostic_calls = _collect_build_diagnostic_calls(ordered)
    node_ids_with_engine_calls = frozenset(
        str(call.get("node_id")) for call in engine_calls if call.get("node_id")
    )
    reflections_by_node: dict[str, list[dict[str, Any]]] = {}

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
                run_id=run_id,
                node_ids_with_engine_calls=node_ids_with_engine_calls,
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
                run_id=run_id,
                node_ids_with_engine_calls=node_ids_with_engine_calls,
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
                reflections_by_node.setdefault(reflected_node, []).append(
                    {
                        "live_event_id": str(row.get("event_id") or ""),
                        "live_seq": _i(row.get("seq")),
                        "iteration": _i(doc.get("iteration")),
                    }
                )
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
                            {
                                **_live_ref(row),
                                "iteration": _i(doc.get("iteration")),
                                # P14: mechanical template reflections are NOT
                                # model chain-of-thought — CoT curation must
                                # exclude anything not "model" here.
                                "reflection_source": _sanitize_text(
                                    str(doc.get("reflection_source") or "mechanical")
                                )[:40],
                            }
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
        elif promo_type in {
            "below_threshold",
            "promotion_failed",
            "public_holdout_rejected",
        }:
            # P14: promotion-gate rejections become countable REVERTED rows —
            # the corpus was survivorship-biased toward crowned winners.
            rejection_epoch = default_epoch
            rejection_bundle = bundles_by_id.get(
                str(row.get("source_score_bundle_id") or "")
            )
            if rejection_bundle and rejection_bundle.get("evaluation_epoch") is not None:
                rejection_epoch = _i(rejection_bundle.get("evaluation_epoch"))
            _append_canonical_event(
                canonical,
                ts=_ts(row.get("created_at"), last_ts),
                event_type="REVERTED",
                cost_usd=0.0,
                payload={
                    "node_id": promo_node_id,
                    "area": island,
                    "epoch": max(0, rejection_epoch),
                    "reason": promo_type,
                    "delta": _f(gate.get(TARGETED_METRIC)),
                    "revert_evidence_ref": (
                        f"promotion_event:{row.get('promotion_event_id')}"
                        if row.get("promotion_event_id")
                        else None
                    ),
                    "rejection_context": sanitize_capture_payload(
                        {
                            "rejection_kind": "promotion_gate",
                            "candidate_id": candidate_id,
                            "promotion_event_type": promo_type,
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

    # -- execution_traces / evidence_bundles pointer rows (item 5.5) -----------
    # P14: mark nodes whose scoring crashed/was rejected without a bundle so
    # they still yield crash-status execution rows.
    for state in nodes.values():
        if not state.candidate_id or state.score_bundle_id:
            continue
        failed_row = failed_eval_by_candidate.get(state.candidate_id)
        if failed_row is not None:
            state.scoring_crashed = True
            state.scoring_failure_event_type = str(failed_row.get("event_type") or "")

    execution_trace_rows, evidence_bundle_rows = _build_corpus_trace_rows(
        run_id=run_id,
        trajectory_id=tid,
        champion_base=champion_base,
        nodes=nodes,
        node_order=node_order,
        engine_calls=engine_calls,
        build_diagnostic_calls=build_diagnostic_calls,
        bundles_by_id=bundles_by_id,
        reflections_by_node=reflections_by_node,
        queue_status=queue_status,
        terminal_row=terminal_row,
        total_cost_usd=last_anchor_total,
        fallback_ts=ledger_created_at,
    )

    # P15 token-budget invariant: count tokens against the declared trainer
    # max (flag, never drop). P10: the protected scan is a computed result
    # over the actual projected payloads, not a caller-supplied boolean.
    total_tokens = 0
    for row in ordered:
        for item, _model in _iter_provider_usage_items([row.get("provider_usage"), _doc(row)]):
            total_tokens += max(0, _i(item.get("total_tokens")))
    trainer_max_tokens = _i(os.getenv("RESEARCH_LAB_TRAINER_MAX_TOKENS", "1000000"))
    protected_hits = find_protected_material(
        {
            "trajectory_doc": trajectory_doc,
            "execution_trace_rows": execution_trace_rows,
            "evidence_bundle_rows": evidence_bundle_rows,
        }
    )

    corpus_source_record = _build_corpus_source_record(
        {
            "source_id": f"trajectory_source:{tid}",
            "trajectory_id": tid,
            "trajectory_hash": sha256_json(trajectory_doc),
            "trajectory_schema_valid": True,
            "event_count": len(canonical),
            "execution_trace_refs": tuple(
                f"execution_trace:{row['run_id']}" for row in execution_trace_rows
            ),
            "evidence_bundle_refs": tuple(
                f"evidence_bundle:{row['bundle_id']}" for row in evidence_bundle_rows
            ),
            "results_ledger_refs": tuple(
                f"results_ledger:{row['ledger_row_id']}" for row in ledger_rows
            ),
            "receipt_refs": (f"receipt:{receipt_id}",) if receipt_id else (),
            "cost_ledger_refs": (f"cost_ledger:{run_id}",),
            "release_policy_ref": "release_policy:unassigned",
            "trajectory_rights_ref": "trajectory_rights:unassigned",
            "distillation_rights_ref": "distillation_rights:unassigned",
            "pii_review_ref": "pii_review:unassigned",
            "legal_gate_ref": "legal_gate:unassigned",
            "island": island,
            # P15 leak-guard keys: paraphrased ICPs from the same brief share the
            # cluster key (sanitized-brief signature) and can never straddle splits.
            "brief_id": brief_id,
            "customer_ref": CUSTOMER_REF_STAND_IN,
            "split_cluster_key": brief_sanitized_ref or brief_id,
            "token_count": total_tokens,
            "over_token_budget": bool(trainer_max_tokens and total_tokens > trainer_max_tokens),
            "split": "train",
            "data_state": "production_measured",
            "measured_data": True,
            "rights_verified": False,
            "distillation_rights_verified": False,
            "pii_review_passed": False,
            "legal_gate_passed": False,
            # P10: computed from the scan that actually ran, not self-declared.
            "protected_data_scanned": True,
            "contains_raw_evidence_snapshot": bool(protected_hits),
            "eligible_for_training": False,  # readiness stays BLOCKED by design
            "eligible_for_distillation": False,
        }
    )

    projection = TrajectoryProjection(
        run_id=run_id,
        trajectory_id=tid,
        envelope_row=envelope_row,
        event_rows=event_rows,
        ledger_rows=ledger_rows,
        trajectory_doc=trajectory_doc,
        corpus_source_record=corpus_source_record,
        execution_trace_rows=execution_trace_rows,
        evidence_bundle_rows=evidence_bundle_rows,
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
    *,
    run_id: str = "",
    node_ids_with_engine_calls: frozenset[str] = frozenset(),
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
        # P2: keep the scored event's scorer-trace pointer map so the node's
        # execution trace can project judge_verdicts.
        raw_scorer_refs = scored_doc.get("scorer_trace_refs")
        if isinstance(raw_scorer_refs, Mapping):
            state.scorer_trace_refs = {
                str(icp_ref): dict(pointer)
                for icp_ref, pointer in raw_scorer_refs.items()
                if isinstance(pointer, Mapping)
            }
        # Bundle→trace linkage check (warn-only): a bundle stamped with an
        # execution_trace:<uuid> ref must point at THIS node's deterministic
        # row id, else the forward join is silently broken. Legacy
        # gateway_qualification_worker:* refs predate the linkage and are
        # expected on historical bundles — no warning for those.
        if score_bundle_id and run_id:
            bundle_row = bundles_by_id.get(score_bundle_id)
            bundle_doc = (
                bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
            )
            embedded_ref = (
                str(bundle_doc.get("execution_trace_ref") or "")
                if isinstance(bundle_doc, Mapping)
                else ""
            )
            if embedded_ref.startswith("execution_trace:"):
                expected_ref = (
                    f"execution_trace:{execution_trace_id_for_node(run_id, state.node_id)}"
                )
                if embedded_ref != expected_ref:
                    logger.warning(
                        "research_lab_trajectory_bundle_trace_ref_mismatch run_id=%s node_id=%s "
                        "bundle_ref=%s expected=%s (bundle→trace join broken for this bundle)",
                        run_id,
                        state.node_id,
                        embedded_ref,
                        expected_ref,
                    )
    if state.build_failed_row is not None and state.build_passed_row is None:
        status = "crash"
    elif gate:
        status = "scored"
    else:
        status = "timeout"  # built but never scored
    proxy_score = _f(gate.get("candidate_delta_vs_daily_baseline")) if gate else 0.0
    # Forward-only linkage (item 5.5): the deterministic execution-trace /
    # evidence-bundle ids are embeddable BEFORE the rows are built.  The
    # predicate mirrors _build_corpus_trace_rows' row policy so an embedded
    # ref always resolves to a written row.
    execution_trace_ref: str | None = None
    if run_id and (
        state.score_bundle_id or state.node_id in node_ids_with_engine_calls
    ):
        execution_trace_ref = (
            f"execution_trace:{execution_trace_id_for_node(run_id, state.node_id)}"
        )
    evidence_bundle_ref: str | None = None
    if run_id and state.score_bundle_id:
        evidence_bundle_ref = "evidence_bundle:" + evidence_bundle_id_for_node(
            run_id, state.node_id, state.score_bundle_id
        )
    evaluation_context = sanitize_capture_payload(
        {
            "candidate_ref": (
                f"candidate:{state.candidate_id}" if state.candidate_id else None
            ),
            "candidate_artifact_hash": state.candidate_artifact_hash,
            "candidate_source_diff_hash": state.candidate_source_diff_hash,
            "score_bundle_ref": score_bundle_id,
            "evidence_bundle_ref": evidence_bundle_ref,
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
            "execution_trace_ref": execution_trace_ref,
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
    errors.extend(_validate_execution_trace_rows(projection.execution_trace_rows))
    errors.extend(_validate_evidence_bundle_rows(projection.evidence_bundle_rows))
    errors.extend(_validate_trace_linkage(projection))
    return errors


def _validate_execution_trace_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Enforce scripts/27 execution_traces CHECK shapes before any write."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        label = f"execution_trace[{row.get('run_id')}]"
        row_id = str(row.get("run_id") or "")
        if _coerce_uuid(row_id) is None:
            errors.append(f"{label}: run_id must be a UUID")
        if row_id in seen_ids:
            errors.append(f"{label}: duplicate deterministic row id")
        seen_ids.add(row_id)
        if row.get("schema_version") != "1.0":
            errors.append(f"{label}: schema_version must be '1.0'")
        if row.get("role") not in EXECUTION_TRACE_ROLES:
            errors.append(f"{label}: role {row.get('role')!r} violates CHECK")
        if row.get("rung") not in EXECUTION_TRACE_RUNGS:
            errors.append(f"{label}: rung {row.get('rung')!r} violates CHECK")
        if row.get("status") not in EXECUTION_TRACE_STATUSES:
            errors.append(f"{label}: status {row.get('status')!r} violates CHECK")
        for key in ("artifact_hash", "icp_set_hash", "outputs_ref", "score_bundle_ref"):
            if not str(row.get(key) or ""):
                errors.append(f"{label}: {key} is NOT NULL in scripts/27")
        for key in ("calls", "evidence_bundles", "judge_verdicts"):
            if not isinstance(row.get(key), list):
                errors.append(f"{label}: {key} must be a JSON array")
        for key in ("eval_version", "cost_ledger"):
            if not isinstance(row.get(key), Mapping):
                errors.append(f"{label}: {key} must be a JSON object")
        trace_doc = row.get("trace_doc")
        if trace_doc is not None and not isinstance(trace_doc, Mapping):
            errors.append(f"{label}: trace_doc must be NULL or a JSON object")
        protected = find_protected_material(dict(row))
        if protected:
            errors.append(f"{label}: protected material at {sorted(protected)[:3]}")
    return errors


def _validate_evidence_bundle_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Enforce scripts/27 evidence_bundles CHECK shapes before any write."""
    errors: list[str] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        label = f"evidence_bundle[{row.get('bundle_id')}]"
        bundle_id = str(row.get("bundle_id") or "")
        if _coerce_uuid(bundle_id) is None:
            errors.append(f"{label}: bundle_id must be a UUID")
        if bundle_id in seen_ids:
            errors.append(f"{label}: duplicate deterministic bundle id")
        seen_ids.add(bundle_id)
        if row.get("schema_version") != "1.0":
            errors.append(f"{label}: schema_version must be '1.0'")
        if row.get("run_id") is not None and _coerce_uuid(row.get("run_id")) is None:
            errors.append(f"{label}: run_id must be NULL or a UUID")
        if not str(row.get("artifact_hash") or ""):
            errors.append(f"{label}: artifact_hash is NOT NULL in scripts/27")
        if row.get("retention_class") not in EVIDENCE_RETENTION_CLASSES:
            errors.append(
                f"{label}: retention_class {row.get('retention_class')!r} violates CHECK"
            )
        if row.get("verification_state") not in EVIDENCE_VERIFICATION_STATES:
            errors.append(
                f"{label}: verification_state {row.get('verification_state')!r} violates CHECK"
            )
        bundle_hash = str(row.get("bundle_hash") or "")
        if not bundle_hash.startswith("sha256:"):
            errors.append(f"{label}: bundle_hash must be sha256:-prefixed")
        if bundle_hash in seen_hashes:
            errors.append(f"{label}: bundle_hash violates UNIQUE")
        seen_hashes.add(bundle_hash)
        snapshots = row.get("snapshots")
        if not isinstance(snapshots, list) or not snapshots:
            errors.append(f"{label}: snapshots must be a non-empty JSON array")
        bundle_doc = row.get("bundle_doc")
        if bundle_doc is not None and not isinstance(bundle_doc, Mapping):
            errors.append(f"{label}: bundle_doc must be NULL or a JSON object")
        protected = find_protected_material(dict(row))
        if protected:
            errors.append(f"{label}: protected material at {sorted(protected)[:3]}")
    return errors


def _validate_trace_linkage(projection: TrajectoryProjection) -> list[str]:
    """Every embedded forward-only ref must resolve to a row in this projection."""
    errors: list[str] = []
    trace_refs = {
        f"execution_trace:{row['run_id']}" for row in projection.execution_trace_rows
    }
    evidence_refs = {
        f"evidence_bundle:{row['bundle_id']}" for row in projection.evidence_bundle_rows
    }
    for row in projection.event_rows:
        event = row["event"]
        if event.get("type") != "NODE_EVALUATED":
            continue
        embedded_trace = event.get("execution_trace_ref")
        if embedded_trace and embedded_trace not in trace_refs:
            errors.append(
                f"event seq={row['seq']}: execution_trace_ref {embedded_trace!r} "
                "has no matching execution_traces row"
            )
        context = event.get("evaluation_context")
        embedded_evidence = (
            context.get("evidence_bundle_ref") if isinstance(context, Mapping) else None
        )
        if embedded_evidence and embedded_evidence not in evidence_refs:
            errors.append(
                f"event seq={row['seq']}: evidence_bundle_ref {embedded_evidence!r} "
                "has no matching evidence_bundles row"
            )
    return errors


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def _insert_missing(
    store: Any, table: str, pk_field: str, row: Mapping[str, Any]
) -> bool:
    """Existence-checked insert: deterministic ids make this upsert-or-skip.

    Returns True when the row was written, False when it already existed
    (including the insert race where another writer lands the same PK first).
    """
    existing = await store.select_one(table, filters=((pk_field, row[pk_field]),))
    if existing:
        return False
    try:
        await store.insert_row(table, dict(row))
    except Exception as exc:
        message = str(exc).lower()
        if "duplicate" in message or "unique" in message or "conflict" in message:
            return False
        raise
    return True


def _trace_call_key(call: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    s3_ref = str(call.get("s3_ref") or "")
    sha256 = str(call.get("sha256") or "")
    if s3_ref or sha256:
        return ("raw", s3_ref, sha256, "", "")
    return (
        "metadata",
        str(call.get("live_event_id") or ""),
        str(call.get("response_id") or ""),
        str(call.get("stage") or ""),
        str(call.get("attempt") or ""),
    )


def _merge_trace_calls(
    existing_calls: Any,
    proposed_calls: Any,
) -> tuple[bool, list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = [
        dict(item) for item in (existing_calls if isinstance(existing_calls, list) else [])
        if isinstance(item, Mapping)
    ]
    seen = {_trace_call_key(item) for item in merged}
    changed = False
    for item in proposed_calls if isinstance(proposed_calls, list) else []:
        if not isinstance(item, Mapping):
            continue
        key = _trace_call_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
        changed = True
    if len(merged) > _EXECUTION_TRACE_CALL_CAP:
        merged = _cap_pointer_entries(
            merged,
            _EXECUTION_TRACE_CALL_CAP,
            "call_truncation_marker",
        )
        changed = True
    return changed, merged


async def _upsert_execution_trace_row(store: Any, row: Mapping[str, Any]) -> str:
    """Insert a trace row, or merge newly discovered calls into an existing row."""
    existing = await store.select_one(
        EXECUTION_TRACES_TABLE, filters=(("run_id", row["run_id"]),)
    )
    if not existing:
        if await _insert_missing(store, EXECUTION_TRACES_TABLE, "run_id", row):
            return "inserted"
        existing = await store.select_one(
            EXECUTION_TRACES_TABLE, filters=(("run_id", row["run_id"]),)
        )
    if not existing:
        return "unchanged"
    changed, calls = _merge_trace_calls(existing.get("calls"), row.get("calls"))
    if not changed:
        return "unchanged"
    await store.update_row(
        EXECUTION_TRACES_TABLE,
        {"calls": calls},
        filters=(("run_id", row["run_id"]),),
    )
    return "updated"


async def _execution_trace_row_needs_write(store: Any, row: Mapping[str, Any]) -> bool:
    existing = await store.select_one(
        EXECUTION_TRACES_TABLE, filters=(("run_id", row["run_id"]),)
    )
    if not existing:
        return True
    changed, _calls = _merge_trace_calls(existing.get("calls"), row.get("calls"))
    return changed


async def _write_missing_corpus_trace_rows(
    store: Any, projection: TrajectoryProjection
) -> tuple[int, int]:
    """Write/merge execution_traces and missing evidence_bundles; return counts."""
    trace_changed = 0
    for row in projection.execution_trace_rows:
        trace_status = await _upsert_execution_trace_row(store, row)
        if trace_status in {"inserted", "updated"}:
            trace_changed += 1
    evidence_written = 0
    for row in projection.evidence_bundle_rows:
        if await _insert_missing(store, EVIDENCE_BUNDLES_TABLE, "bundle_id", row):
            evidence_written += 1
    return trace_changed, evidence_written


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
                execution_trace_count=len(projection.execution_trace_rows),
                evidence_bundle_count=len(projection.evidence_bundle_rows),
            )
        # Envelope first (events FK-reference it), then append-only events in
        # seq order, then ledger rows, then the pointer rows.  A crash between
        # envelope and pointer writes is repaired by --traces-backfill.
        await store.insert_row(TRAJECTORIES_TABLE, projection.envelope_row)
        for row in projection.event_rows:
            await store.insert_row(TRAJECTORY_EVENTS_TABLE, row)
        for row in projection.ledger_rows:
            await store.insert_row(RESULTS_LEDGER_TABLE, row)
        trace_written, evidence_written = await _write_missing_corpus_trace_rows(
            store, projection
        )
        logger.info(
            "research_lab_trajectory_projected run_id=%s trajectory_id=%s events=%s "
            "ledger_rows=%s execution_traces=%s evidence_bundles=%s",
            run_id,
            tid,
            len(projection.event_rows),
            len(projection.ledger_rows),
            trace_written,
            evidence_written,
        )
        return ProjectionResult(
            run_id=run_id,
            status="projected",
            trajectory_id=tid,
            event_count=len(projection.event_rows),
            ledger_row_count=len(projection.ledger_rows),
            execution_trace_count=trace_written,
            evidence_bundle_count=evidence_written,
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


async def backfill_run_corpus_trace_rows(
    run_id: str,
    *,
    store: Any | None = None,
    dry_run: bool = True,
) -> ProjectionResult:
    """Add missing execution_traces / evidence_bundles rows for ONE
    already-projected run (fablefollowup.md item 5.5 re-projection command).

    Forward-only: NEVER touches the run's envelope/events/ledger rows -- the
    event log is append-only + hash-anchored, so historical runs get pointer
    rows without event retrofit.  Absence-tolerant: a historical run with no
    raw traces and no score bundles reports ``skipped_no_trace_sources``.
    Best-effort: never raises.
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
            return ProjectionResult(
                run_id=run_id, status="skipped_disabled", trajectory_id=tid
            )
        envelope = await store.select_one(
            TRAJECTORIES_TABLE, filters=(("trajectory_id", tid),)
        )
        if not envelope:
            # Not projected yet: --backfill owns it (and writes traces inline).
            return ProjectionResult(
                run_id=run_id, status="skipped_unprojected", trajectory_id=tid
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
            return ProjectionResult(
                run_id=run_id,
                status="failed",
                trajectory_id=tid,
                errors=projection.errors,
            )
        if not projection.execution_trace_rows and not projection.evidence_bundle_rows:
            return ProjectionResult(
                run_id=run_id, status="skipped_no_trace_sources", trajectory_id=tid
            )
        missing_traces = [
            row
            for row in projection.execution_trace_rows
            if await _execution_trace_row_needs_write(store, row)
        ]
        missing_evidence = [
            row
            for row in projection.evidence_bundle_rows
            if not await store.select_one(
                EVIDENCE_BUNDLES_TABLE, filters=(("bundle_id", row["bundle_id"]),)
            )
        ]
        if not missing_traces and not missing_evidence:
            return ProjectionResult(
                run_id=run_id, status="skipped_traces_existing", trajectory_id=tid
            )
        if dry_run:
            return ProjectionResult(
                run_id=run_id,
                status="traces_dry_run",
                trajectory_id=tid,
                execution_trace_count=len(missing_traces),
                evidence_bundle_count=len(missing_evidence),
            )
        trace_written = 0
        for row in missing_traces:
            trace_status = await _upsert_execution_trace_row(store, row)
            if trace_status in {"inserted", "updated"}:
                trace_written += 1
        evidence_written = 0
        for row in missing_evidence:
            if await _insert_missing(store, EVIDENCE_BUNDLES_TABLE, "bundle_id", row):
                evidence_written += 1
        logger.info(
            "research_lab_corpus_traces_backfilled run_id=%s trajectory_id=%s "
            "execution_traces=%s evidence_bundles=%s",
            run_id,
            tid,
            trace_written,
            evidence_written,
        )
        return ProjectionResult(
            run_id=run_id,
            status="traces_backfilled",
            trajectory_id=tid,
            execution_trace_count=trace_written,
            evidence_bundle_count=evidence_written,
        )
    except Exception as exc:  # never raise out of a backfill pass
        logger.warning(
            "research_lab_corpus_traces_backfill_failed run_id=%s error=%s",
            run_id,
            str(exc)[:500],
        )
        return ProjectionResult(
            run_id=run_id, status="failed", trajectory_id=tid, errors=[str(exc)[:500]]
        )


async def backfill_corpus_trace_rows(
    *,
    batch_size: int = 25,
    dry_run: bool = True,
    store: Any | None = None,
    max_candidates: int = 5000,
    max_attempts: int | None = None,
) -> list[ProjectionResult]:
    """Scan already-projected terminal runs and add missing pointer rows.

    Batch-capped: stops after ``batch_size`` runs produced (or, in dry-run,
    would produce) writes. ``max_attempts`` separately caps row inspections so
    persistent per-run failures cannot flood every worker maintenance pass.
    """
    store = store or GatewayProjectorStore()
    results: list[ProjectionResult] = []
    if not dry_run and not projector_enabled():
        logger.warning(
            "research_lab_trajectory_projector_disabled set %s=1 to enable writes",
            PROJECTOR_ENABLED_ENV,
        )
        return results
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
            "research_lab_corpus_traces_backfill_discovery_failed error=%s",
            str(exc)[:500],
        )
        return results
    processed = 0
    attempted = 0
    attempt_limit = max(1, int(max_attempts if max_attempts is not None else max_candidates))
    for row in queue_rows:
        if processed >= max(1, int(batch_size)):
            break
        if attempted >= attempt_limit:
            break
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        attempted += 1
        result = await backfill_run_corpus_trace_rows(
            run_id, store=store, dry_run=dry_run
        )
        results.append(result)
        if result.status in ("traces_backfilled", "traces_dry_run"):
            processed += 1
    return results


async def summarize_reasoning_capture_coverage(
    *,
    store: Any | None = None,
    max_events: int = 5000,
    max_traces: int = 5000,
) -> dict[str, Any]:
    """Private coverage report for OpenRouter reasoning/raw-trace capture."""
    store = store or GatewayProjectorStore()
    loop_events = await store.select_all(
        LOOP_EVENTS_TABLE,
        columns="run_id,event_id,seq,event_type,node_id,provider_usage,event_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_events)),
    )
    execution_traces = await store.select_all(
        EXECUTION_TRACES_TABLE,
        columns="run_id,calls,trace_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_traces)),
    )

    projected_raw_refs: set[tuple[str, str]] = set()
    projected_reasoning_calls = 0
    projected_metadata_only_calls = 0
    for trace in execution_traces:
        calls = trace.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            s3_ref = str(call.get("s3_ref") or "")
            sha256 = str(call.get("sha256") or "")
            if s3_ref or sha256:
                projected_raw_refs.add((s3_ref, sha256))
            reasoning_capture = call.get("reasoning_capture")
            if isinstance(reasoning_capture, Mapping):
                projected_reasoning_calls += 1
                if reasoning_capture.get("storage_state") == "metadata_only":
                    projected_metadata_only_calls += 1

    openrouter_calls = 0
    reasoning_requested = 0
    reasoning_returned = 0
    raw_trace_stored = 0
    raw_trace_projected = 0
    missing_projected_raw_trace_refs = 0
    requested_without_raw_trace_ref = 0
    reasoning_request_dropped = 0
    metadata_only_reasoning = 0
    models: dict[str, dict[str, int]] = {}
    for row in loop_events:
        doc = _doc(row)
        for item, model in _iter_provider_usage_items([row.get("provider_usage"), doc]):
            provider = str(item.get("provider") or "openrouter").lower()
            if provider != "openrouter" and not item.get("reasoning_logs") and not item.get("raw_trace_ref"):
                continue
            openrouter_calls += 1
            model_key = (str(model or item.get("model") or "unknown") or "unknown")[:128]
            model_stats = models.setdefault(
                model_key,
                {
                    "openrouter_calls": 0,
                    "reasoning_requested": 0,
                    "reasoning_returned": 0,
                    "raw_trace_stored": 0,
                    "raw_trace_projected": 0,
                },
            )
            model_stats["openrouter_calls"] += 1
            capture = item.get("reasoning_capture") if isinstance(item.get("reasoning_capture"), Mapping) else {}
            logs = item.get("reasoning_logs") if isinstance(item.get("reasoning_logs"), Mapping) else {}
            requested = bool(capture.get("requested")) or bool(logs) or bool(item.get("reasoning_request_dropped"))
            returned = bool(capture.get("returned")) or bool(logs)
            pointer = item.get("raw_trace_ref") if isinstance(item.get("raw_trace_ref"), Mapping) else {}
            s3_ref = str(pointer.get("s3_ref") or "") if isinstance(pointer, Mapping) else ""
            sha256 = str(pointer.get("sha256") or "") if isinstance(pointer, Mapping) else ""
            has_raw = bool(s3_ref or sha256)
            if requested:
                reasoning_requested += 1
                model_stats["reasoning_requested"] += 1
            if returned:
                reasoning_returned += 1
                model_stats["reasoning_returned"] += 1
            if item.get("reasoning_request_dropped") or capture.get("request_dropped"):
                reasoning_request_dropped += 1
            if has_raw:
                raw_trace_stored += 1
                model_stats["raw_trace_stored"] += 1
                if (s3_ref, sha256) in projected_raw_refs:
                    raw_trace_projected += 1
                    model_stats["raw_trace_projected"] += 1
                else:
                    missing_projected_raw_trace_refs += 1
            elif requested:
                requested_without_raw_trace_ref += 1
                if returned:
                    metadata_only_reasoning += 1

    return {
        "schema_version": "1.0",
        "source": "research_lab_openrouter_reasoning_capture",
        "loop_events_scanned": len(loop_events),
        "execution_traces_scanned": len(execution_traces),
        "openrouter_calls": openrouter_calls,
        "reasoning_requested": reasoning_requested,
        "reasoning_returned": reasoning_returned,
        "raw_trace_stored": raw_trace_stored,
        "raw_trace_projected": raw_trace_projected,
        "missing_projected_raw_trace_refs": missing_projected_raw_trace_refs,
        "requested_without_raw_trace_ref": requested_without_raw_trace_ref,
        "metadata_only_reasoning": metadata_only_reasoning,
        "reasoning_request_dropped": reasoning_request_dropped,
        "projected_reasoning_calls": projected_reasoning_calls,
        "projected_metadata_only_calls": projected_metadata_only_calls,
        "models": models,
    }


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
    parser.add_argument(
        "--traces-backfill",
        action="store_true",
        help=(
            "add missing execution_traces/evidence_bundles pointer rows to "
            "already-projected runs (append-only events are never touched); "
            "combine with --run-id to target one run"
        ),
    )
    parser.add_argument(
        "--reasoning-coverage",
        action="store_true",
        help="print private OpenRouter reasoning/raw-trace capture coverage without writing",
    )
    parser.add_argument(
        "--capture-coverage",
        action="store_true",
        help=(
            "print the P7 all-channel capture coverage report (engine, scorer, "
            "in-container, diagnostics, judge verdicts, crash rows)"
        ),
    )
    parser.add_argument(
        "--reconcile-pointers",
        action="store_true",
        help="P6: HEAD every recent S3 trace pointer and report verified/missing/hash_mismatch",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="with --reconcile-pointers: fetch objects and verify sha256 (slow)",
    )
    parser.add_argument(
        "--import-shadow-windows",
        action="store_true",
        help=(
            "P18: import shadow-monitor window reports as DB rows keyed to the "
            "promotion they adjudicate (requires scripts/64)"
        ),
    )
    parser.add_argument("--run-id", default=None, help="project a single run id")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="max runs to project per pass (backfill loops until drained)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5000,
        help="max live loop events to scan for --reasoning-coverage",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=5000,
        help="max execution_traces rows to scan for --reasoning-coverage",
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
    if (
        not args.backfill
        and not args.run_id
        and not args.traces_backfill
        and not args.reasoning_coverage
        and not args.capture_coverage
        and not args.reconcile_pointers
        and not args.import_shadow_windows
    ):
        _build_arg_parser().print_help()
        return 2
    store = GatewayProjectorStore()
    if args.capture_coverage or args.reconcile_pointers or args.import_shadow_windows:
        from gateway.research_lab.trace_reconciler import (
            import_shadow_windows,
            reconcile_trace_pointers,
            summarize_capture_coverage,
        )

        report: dict[str, Any] = {"dry_run": True, "projector_enabled": projector_enabled()}
        if args.capture_coverage:
            report["capture_coverage"] = await summarize_capture_coverage(
                store=store, max_rows=args.max_events
            )
        if args.reconcile_pointers:
            report["pointer_reconciliation"] = await reconcile_trace_pointers(
                store=store, verify_hash=args.verify_hash, max_rows=args.max_events
            )
        if args.import_shadow_windows:
            report["dry_run"] = False
            report["shadow_window_import"] = await import_shadow_windows(store=store)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.reasoning_coverage:
        coverage = await summarize_reasoning_capture_coverage(
            store=store,
            max_events=args.max_events,
            max_traces=args.max_traces,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "projector_enabled": projector_enabled(),
                    "corpus_contract_version": TRAJECTORY_CORPUS_CONTRACT_VERSION,
                    "coverage": coverage,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    results: list[ProjectionResult] = []
    if args.run_id and not args.traces_backfill:
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
    if args.traces_backfill:
        if args.run_id:
            results.append(
                await backfill_run_corpus_trace_rows(
                    args.run_id, store=store, dry_run=args.dry_run
                )
            )
        else:
            while True:
                batch = await backfill_corpus_trace_rows(
                    batch_size=args.batch_size, dry_run=args.dry_run, store=store
                )
                results.extend(batch)
                attempted = [
                    r
                    for r in batch
                    if r.status in ("traces_backfilled", "traces_dry_run")
                ]
                if not attempted:
                    break
                if args.dry_run:
                    # Dry-run writes nothing; one pass reports every run that
                    # would gain pointer rows.
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
