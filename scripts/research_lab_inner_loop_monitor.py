#!/usr/bin/env python3
"""Read-only production monitor for automatic Research Lab inner-loop rollout."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _freshness(rows: Sequence[Mapping[str, Any]], now: datetime) -> dict[str, Any]:
    stamps = [_timestamp(row.get("created_at")) for row in rows]
    stamps = [stamp for stamp in stamps if stamp is not None]
    newest = max(stamps) if stamps else None
    return {
        "latest_created_at": newest.isoformat() if newest else None,
        "age_seconds": round((now - newest).total_seconds(), 3) if newest else None,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


async def _safe_select_all(
    store: Any,
    table: str,
    *,
    columns: str,
    query: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Keep one unavailable telemetry source from hiding all other evidence."""
    try:
        rows = await store.select_all(table, columns=columns, **dict(query))
    except Exception as exc:  # noqa: BLE001 - monitor reports read failures
        return [], f"{type(exc).__name__}:{str(exc)[:180]}"
    return [dict(row) for row in rows if isinstance(row, Mapping)], ""


async def _collect(days: int) -> dict[str, Any]:
    from gateway.research_lab import store
    from gateway.research_lab.dev_eval_runner import snapshot_readiness
    from gateway.research_lab.inner_loop_activation import (
        INNER_LOOP_EVENT_TABLE,
        spearman_correlation,
        top_quartile_lift,
    )
    from gateway.research_lab.score_backfill import SCORE_CALIBRATION_TABLE

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=max(1, int(days)))).isoformat()
    common = {
        "filters": (("created_at", "gte", cutoff),),
        "order_by": (("created_at", True),),
        "batch_size": 500,
        "max_rows": 50000,
        "allow_partial": True,
    }
    source_results = await asyncio.gather(
        _safe_select_all(
            store,
            SCORE_CALIBRATION_TABLE,
            columns=(
                "candidate_id,run_id,node_id,lane,plan_path_id,predicted_delta,"
                "dev_score,realized_mean_delta,realized_delta_lcb,outcome,created_at"
            ),
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_results_ledger",
            columns="ledger_row_id,status,delta_vs_parent,cost_usd,created_at",
            query=common,
        ),
        _safe_select_all(
            store,
            INNER_LOOP_EVENT_TABLE,
            columns="seq,event_type,phase,run_id,evidence_doc,created_at",
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_allocator_selection_records",
            columns="selection_record_id,created_at",
            query=common,
        ),
    )
    source_names = (
        "calibration",
        "results_ledger",
        "activation_events",
        "allocator_records",
    )
    source_errors = {
        name: error
        for name, (_rows, error) in zip(source_names, source_results)
        if error
    }
    calibration_rows, ledger_rows, activation_rows, allocator_rows = (
        rows for rows, _error in source_results
    )

    observations = [
        row for row in activation_rows if row.get("event_type") == "run_observed"
    ]
    transitions = [
        row for row in activation_rows if row.get("event_type") == "phase_transition"
    ]
    latest_transition = max(transitions, key=lambda row: int(row.get("seq") or 0), default={})
    latest_observation = max(observations, key=lambda row: int(row.get("seq") or 0), default={})
    latest_doc = (
        dict(latest_observation.get("evidence_doc") or {})
        if isinstance(latest_observation.get("evidence_doc"), Mapping)
        else {}
    )

    evidence_docs = [
        dict(row.get("evidence_doc") or {})
        for row in observations
        if isinstance(row.get("evidence_doc"), Mapping)
    ]
    fallback_reasons = Counter(
        str(doc.get("fallback_reason") or "none") for doc in evidence_docs
    )
    runtimes = [
        value
        for value in (_finite(doc.get("runtime_seconds")) for doc in evidence_docs)
        if value is not None
    ]
    candidate_count = sum(int(doc.get("candidate_count") or 0) for doc in evidence_docs)
    evaluated_count = sum(
        int(doc.get("evaluated_candidate_count") or 0) for doc in evidence_docs
    )
    eligible_count = sum(
        int(doc.get("eligible_candidate_count") or 0) for doc in evidence_docs
    )

    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"scored": 0, "dev_scored": 0}
    )
    pairs: list[tuple[float, float]] = []
    for row in calibration_rows:
        lane = str(row.get("lane") or "unknown")
        coverage[lane]["scored"] += 1
        dev_score = _finite(row.get("dev_score"))
        realized = _finite(row.get("realized_mean_delta"))
        if dev_score is not None:
            coverage[lane]["dev_scored"] += 1
        if dev_score is not None and realized is not None:
            pairs.append((dev_score, realized))

    status_counts = Counter(str(row.get("status") or "unknown") for row in ledger_rows)
    total_cost = sum(
        max(0.0, value)
        for value in (_finite(row.get("cost_usd")) for row in ledger_rows)
        if value is not None
    )
    kept_delta = sum(
        value
        for row in ledger_rows
        for value in [_finite(row.get("delta_vs_parent"))]
        if str(row.get("status") or "") == "keep" and value is not None
    )

    snapshot_uri = str(os.getenv("RESEARCH_LAB_DEV_SNAPSHOT_URI") or "").strip()
    snapshot = (
        await asyncio.to_thread(snapshot_readiness, snapshot_uri)
        if snapshot_uri
        else {"ready": False, "reason": "snapshot_uri_unset"}
    )
    alerts: list[str] = []
    alerts.extend(f"monitor_data_source_error:{name}" for name in source_errors)
    if not snapshot.get("ready"):
        alerts.append("snapshot_not_ready_or_tampered")
    if (_finite(snapshot.get("snapshot_age_seconds")) or 0.0) > 14 * 86400:
        alerts.append("snapshot_older_than_14_days")
    alert_fields = {
        "unclassified_error_count": "unclassified_evaluator_errors",
        "silent_miss_count": "silent_evaluation_misses",
        "paid_finalist_invariant_violations": "paid_finalist_invariant_violation",
        "protected_workflow_invariant_violations": "protected_workflow_invariant_violation",
    }
    for field, label in alert_fields.items():
        if sum(int(doc.get(field) or 0) for doc in evidence_docs) > 0:
            alerts.append(label)
    if any(bool(doc.get("candidate_width_mismatch")) for doc in evidence_docs):
        alerts.append("candidate_width_mismatch")

    return {
        "generated_at": now.isoformat(),
        "window_days": max(1, int(days)),
        "data_source_errors": source_errors,
        "effective_phase": str(
            latest_transition.get("phase")
            or latest_observation.get("phase")
            or "observe"
        ),
        "run_counts_by_phase": dict(
            sorted(Counter(str(row.get("phase") or "unknown") for row in observations).items())
        ),
        "candidate_execution": {
            "built": candidate_count,
            "evaluated": evaluated_count,
            "eligible": eligible_count,
            "eligibility_rate": (
                round(eligible_count / candidate_count, 6) if candidate_count else None
            ),
            "snapshot_misses": sum(
                int(doc.get("snapshot_miss_count") or 0) for doc in evidence_docs
            ),
            "true_misses": sum(
                int(doc.get("true_miss_count") or 0) for doc in evidence_docs
            ),
            "evaluation_failures": sum(
                int(doc.get("evaluation_failure_count") or 0) for doc in evidence_docs
            ),
            "unclassified_errors": sum(
                int(doc.get("unclassified_error_count") or 0) for doc in evidence_docs
            ),
            "zero_outputs": sum(
                int(doc.get("zero_output_count") or 0) for doc in evidence_docs
            ),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
        },
        "latest_selection": {
            "run_id": latest_observation.get("run_id"),
            "shadow_winner": latest_doc.get("hypothetical_winner_node_id"),
            "actual_paid_candidate": latest_doc.get("actual_paid_candidate_node_id"),
            "ranking_applied": bool(latest_doc.get("ranking_applied")),
        },
        "runtime_seconds": {
            "sample_count": len(runtimes),
            "mean": round(sum(runtimes) / len(runtimes), 3) if runtimes else None,
            "p95": _percentile(runtimes, 0.95),
        },
        "snapshot": snapshot,
        "calibration": {
            "row_count": len(calibration_rows),
            "pair_count": len(pairs),
            "spearman_rho": spearman_correlation(pairs),
            "top_quartile_realized_lift": top_quartile_lift(pairs),
            "coverage_by_lane": {
                lane: {
                    **counts,
                    "coverage": round(counts["dev_scored"] / counts["scored"], 6)
                    if counts["scored"]
                    else None,
                }
                for lane, counts in sorted(coverage.items())
            },
            "freshness": _freshness(calibration_rows, now),
        },
        "allocator_freshness": _freshness(allocator_rows, now),
        "ledger_yield": {
            "status_counts": dict(sorted(status_counts.items())),
            "kept_delta_sum": round(kept_delta, 6),
            "total_cost_usd": round(total_cost, 2),
            "kept_delta_per_usd": round(kept_delta / total_cost, 6)
            if total_cost
            else None,
        },
        "alerts": sorted(set(alerts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(_collect(args.days))
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 1 if summary["alerts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
