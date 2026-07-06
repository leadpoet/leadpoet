"""Post-score backfill: write realized results back into loop memory (Phase 5).

Today realized scores land in ``research_evaluation_score_bundles`` and the
results ledger but inform nothing retroactively: lessons are written at
reflection time (pre-score), attempt memory stores only "tried", and each
hypothesis's ``predicted_delta`` is never reconciled against reality.

This module closes that loop with one append-only calibration row per scored
candidate (``research_lab_score_calibration``, scripts/71):

- **prediction calibration**: ``(candidate_id, lane, plan_path_id,
  predicted_delta, dev_score, realized_delta, outcome)`` — deliberately the
  future training table for the deferred learned quality predictor (revisit
  at ~1k rows; the predictor itself stays unbuilt).
- **score-aware lessons**: ``lesson_store`` hydrates these rows onto retrieved
  reflection lessons by ``node_id`` at read time (trajectory events are
  immutable by design — we never mutate a recorded reflection).
- **score-aware attempt memory**: the hosted worker joins realized deltas onto
  ``active_parent_outcome_memory.recent_attempts`` (that join reads score
  bundles directly; this table is not on that path).

Flag-gated OFF by default (``RESEARCH_LAB_SCORE_BACKFILL_ENABLED``, §8.3 one
flag at a time). Strictly best-effort and additive: any failure logs and the
scored candidate proceeds untouched; with the flag off, prompts degrade
gracefully to score-blind memory.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

SCORE_BACKFILL_ENABLED_ENV = "RESEARCH_LAB_SCORE_BACKFILL_ENABLED"
SCORE_CALIBRATION_TABLE = "research_lab_score_calibration"


def score_backfill_enabled() -> bool:
    """§8.3 staged enable: score backfill is OFF by default."""
    return os.environ.get(SCORE_BACKFILL_ENABLED_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class GatewayCalibrationStore:
    """Thin async facade over gateway.research_lab.store helpers."""

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

    async def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        from gateway.research_lab import store

        return await store.insert_row(table, row)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN guard


def _candidate_code_edit_doc(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = candidate.get("candidate_patch_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    patch_doc = manifest.get("patch_doc") if isinstance(manifest.get("patch_doc"), Mapping) else {}
    code_edit = patch_doc.get("code_edit") if isinstance(patch_doc.get("code_edit"), Mapping) else {}
    return code_edit


def build_calibration_row(
    *,
    candidate: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    score_bundle_id: str,
    outcome: str = "",
    created_by: str = "",
) -> dict[str, Any]:
    """Project one scored candidate into a calibration row (pure/deterministic)."""
    code_edit = _candidate_code_edit_doc(candidate)
    hypothesis = candidate.get("hypothesis_doc")
    hypothesis = hypothesis if isinstance(hypothesis, Mapping) else {}
    build_doc = candidate.get("candidate_build_doc")
    build_doc = build_doc if isinstance(build_doc, Mapping) else {}
    aggregates = score_bundle.get("aggregates")
    aggregates = aggregates if isinstance(aggregates, Mapping) else {}
    return {
        "schema_version": "1.0",
        "candidate_id": str(candidate.get("candidate_id") or "")[:120],
        "run_id": str(candidate.get("run_id") or "")[:120],
        "node_id": str(build_doc.get("loop_node_id") or "")[:128],
        "island": str(candidate.get("island") or score_bundle.get("island") or "")[:128],
        "lane": str(code_edit.get("lane") or "")[:80],
        "plan_path_id": str(code_edit.get("plan_path_id") or hypothesis.get("plan_path_id") or "")[:120],
        "predicted_delta": _float_or_none(hypothesis.get("predicted_delta")),
        "dev_score": _float_or_none(build_doc.get("loop_dev_score")),
        "dev_score_version": str(build_doc.get("loop_dev_score_version") or "")[:120],
        "realized_mean_delta": _float_or_none(aggregates.get("mean_delta")),
        "realized_delta_lcb": _float_or_none(aggregates.get("delta_lcb")),
        "outcome": str(outcome or "")[:80],
        "score_bundle_id": str(score_bundle_id or "")[:120],
        "created_by": str(created_by or "")[:120],
    }


async def record_score_backfill(
    *,
    candidate: Mapping[str, Any],
    score_bundle_row: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
    promotion_result: Mapping[str, Any] | None = None,
    store: Any | None = None,
    created_by: str = "",
) -> dict[str, Any]:
    """Persist the calibration row for one scored candidate (idempotent).

    ``outcome`` is the promotion controller's status when available (the
    closest thing to the ledger's keep/discard at score time). Re-running for
    the same (candidate, bundle) is a no-op.
    """
    if not score_backfill_enabled():
        return {"status": "disabled"}
    store = store or GatewayCalibrationStore()
    score_bundle_id = str(
        (score_bundle_row or {}).get("score_bundle_id")
        or score_bundle.get("score_bundle_id")
        or ""
    )
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id or not score_bundle_id:
        return {"status": "skipped", "reason": "missing_candidate_or_bundle_id"}
    existing = await store.select_many(
        SCORE_CALIBRATION_TABLE,
        columns="calibration_id",
        filters=[("candidate_id", candidate_id), ("score_bundle_id", score_bundle_id)],
        order_by=(),
        limit=1,
    )
    if existing:
        return {"status": "already_recorded", "candidate_id": candidate_id}
    outcome = str((promotion_result or {}).get("status") or "")
    row = build_calibration_row(
        candidate=candidate,
        score_bundle=score_bundle,
        score_bundle_id=score_bundle_id,
        outcome=outcome,
        created_by=created_by,
    )
    await store.insert_row(SCORE_CALIBRATION_TABLE, row)
    return {
        "status": "recorded",
        "candidate_id": candidate_id,
        "score_bundle_id": score_bundle_id,
        "node_id": row["node_id"],
        "outcome": outcome,
    }


async def fetch_score_enrichments_by_node(
    node_ids: Sequence[str],
    *,
    store: Any | None = None,
    limit: int = 50,
) -> dict[str, dict[str, Any]]:
    """Newest calibration row per node_id, for read-time lesson hydration.

    Best-effort by contract: callers wrap failures (e.g. scripts/71 not yet
    applied) and fall back to score-blind lessons.
    """
    wanted = [str(node_id) for node_id in node_ids if str(node_id or "").strip()]
    if not wanted:
        return {}
    store = store or GatewayCalibrationStore()
    enrichments: dict[str, dict[str, Any]] = {}
    for node_id in dict.fromkeys(wanted):  # preserve order, dedupe
        rows = await store.select_many(
            SCORE_CALIBRATION_TABLE,
            columns=(
                "node_id,candidate_id,lane,plan_path_id,predicted_delta,dev_score,"
                "realized_mean_delta,realized_delta_lcb,outcome,created_at"
            ),
            filters=[("node_id", node_id)],
            order_by=[("created_at", True)],
            limit=1,
        )
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            enrichments[node_id] = {
                "realized_delta": _float_or_none(row.get("realized_mean_delta")),
                "realized_delta_lcb": _float_or_none(row.get("realized_delta_lcb")),
                "predicted_delta": _float_or_none(row.get("predicted_delta")),
                "dev_score": _float_or_none(row.get("dev_score")),
                "scored_outcome": str(row.get("outcome") or "")[:80],
            }
        if len(enrichments) >= max(1, int(limit)):
            break
    return enrichments
