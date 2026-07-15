#!/usr/bin/env python3
"""Read-only production monitor for V2 Git-tree autoresearch."""

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
    except Exception as exc:  # noqa: BLE001 - read-only monitor reports failures
        return [], f"{type(exc).__name__}:{str(exc)[:180]}"
    return [dict(row) for row in rows if isinstance(row, Mapping)], ""


def _tree_policy(tree: Mapping[str, Any]) -> Mapping[str, Any]:
    tree_doc = tree.get("tree_doc")
    if not isinstance(tree_doc, Mapping):
        return {}
    policy = tree_doc.get("policy")
    return dict(policy) if isinstance(policy, Mapping) else {}


def _tree_runtime_seconds(
    tree: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> float | None:
    started = _timestamp(tree.get("created_at"))
    ended = max(
        (
            stamp
            for stamp in (_timestamp(row.get("created_at")) for row in events)
            if stamp is not None
        ),
        default=None,
    )
    if started is None or ended is None:
        return None
    return max(0.0, (ended - started).total_seconds())


async def _collect(days: int) -> dict[str, Any]:
    from gateway.research_lab import store
    from gateway.research_lab.calibration_metrics import (
        spearman_correlation,
        top_quartile_lift,
    )
    from gateway.research_lab.dev_eval_runner import snapshot_readiness
    from gateway.research_lab.git_tree_models import (
        GitTreeContractError,
        TreePolicy,
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
            "research_lab_autoresearch_tree_current",
            columns=(
                "tree_id,run_id,tree_doc,created_at,current_event_seq,"
                "current_event_type,current_event_doc,current_event_at,"
                "current_round_index,current_frontier_hash"
            ),
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_autoresearch_tree_node_current",
            columns=(
                "tree_id,node_id,parent_node_id,root_branch_id,depth,child_ordinal,"
                "created_at,current_event_type,current_event_doc,current_event_at"
            ),
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_autoresearch_tree_events",
            columns="tree_id,seq,event_type,node_id,event_doc,created_at",
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_autoresearch_operation_current",
            columns=(
                "logical_operation_id,tree_id,node_id,operation_kind,"
                "operation_status,settled_cost_microusd,provider_call_count,"
                "settlement_doc,created_at"
            ),
            query=common,
        ),
        _safe_select_all(
            store,
            "research_lab_autoresearch_tree_handoffs",
            columns="tree_id,run_id,candidate_id,node_id,handoff_hash,created_at",
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
        "trees",
        "tree_nodes",
        "tree_events",
        "tree_operations",
        "tree_handoffs",
        "allocator_records",
    )
    source_errors = {
        name: error
        for name, (_rows, error) in zip(source_names, source_results)
        if error
    }
    (
        calibration_rows,
        ledger_rows,
        tree_rows,
        node_rows,
        event_rows,
        operation_rows,
        handoff_rows,
        allocator_rows,
    ) = (rows for rows, _error in source_results)

    events_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nodes_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    handoffs_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        events_by_tree[str(row.get("tree_id") or "")].append(row)
    for row in node_rows:
        nodes_by_tree[str(row.get("tree_id") or "")].append(row)
    for row in handoff_rows:
        handoffs_by_tree[str(row.get("tree_id") or "")].append(row)

    node_statuses: Counter[str] = Counter()
    evaluated_count = 0
    eligible_count = 0
    evaluation_failures = 0
    true_misses = 0
    zero_outputs = 0
    for row in node_rows:
        doc = row.get("current_event_doc")
        doc = dict(doc) if isinstance(doc, Mapping) else {}
        status = str(doc.get("status") or row.get("current_event_type") or "unknown")
        node_statuses[status] += 1
        evaluation = doc.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        evaluated_count += 1
        eligible_count += int(bool(evaluation.get("eligible")))
        evaluation_failures += int(evaluation.get("failure_count") or 0)
        true_misses += int(evaluation.get("true_miss_count") or 0)
        zero_outputs += int(evaluation.get("zero_output_count") or 0)

    operation_statuses = Counter(
        str(row.get("operation_status") or "unknown") for row in operation_rows
    )
    operation_kinds = Counter(
        str(row.get("operation_kind") or "unknown") for row in operation_rows
    )
    settled_cost_microusd = sum(
        max(0, int(row.get("settled_cost_microusd") or 0))
        for row in operation_rows
    )
    provider_call_count = sum(
        max(0, int(row.get("provider_call_count") or 0))
        for row in operation_rows
    )

    final_events = [row for row in event_rows if row.get("event_type") == "final_selected"]
    latest_final = max(
        final_events,
        key=lambda row: _timestamp(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        default={},
    )
    latest_selection_doc = latest_final.get("event_doc")
    latest_selection_doc = (
        dict(latest_selection_doc.get("selection") or {})
        if isinstance(latest_selection_doc, Mapping)
        else {}
    )
    latest_tree_id = str(latest_final.get("tree_id") or "")
    latest_handoff = (handoffs_by_tree.get(latest_tree_id) or [{}])[0]

    topology_violations: list[str] = []
    for tree in tree_rows:
        tree_id = str(tree.get("tree_id") or "")
        policy = _tree_policy(tree)
        nodes = nodes_by_tree.get(tree_id, [])
        max_nodes = int(policy.get("max_nodes") or 0)
        max_depth = int(policy.get("max_depth") or 0)
        branch_factor = int(policy.get("branch_factor") or 0)
        root_branches = {
            str(row.get("root_branch_id") or "")
            for row in nodes
            if int(row.get("depth") or 0) == 1
        }
        if max_nodes and len(nodes) > max_nodes:
            topology_violations.append(f"{tree_id}:node_cap")
        if max_depth and any(int(row.get("depth") or 0) > max_depth for row in nodes):
            topology_violations.append(f"{tree_id}:depth_cap")
        if branch_factor and len(root_branches) > branch_factor:
            topology_violations.append(f"{tree_id}:branch_cap")

    runtimes = [
        value
        for value in (
            _tree_runtime_seconds(tree, events_by_tree.get(str(tree.get("tree_id") or ""), []))
            for tree in tree_rows
        )
        if value is not None
    ]

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
    try:
        live_policy = TreePolicy.from_env()
        policy_doc: Mapping[str, Any] = live_policy.to_dict()
        policy_error = ""
    except GitTreeContractError as exc:
        policy_doc = {}
        policy_error = f"{type(exc).__name__}:{str(exc)[:180]}"

    final_count_by_tree = Counter(str(row.get("tree_id") or "") for row in final_events)
    alerts: list[str] = []
    alerts.extend(f"monitor_data_source_error:{name}" for name in source_errors)
    if policy_error:
        alerts.append("tree_policy_invalid")
    if not snapshot.get("ready"):
        alerts.append("snapshot_not_ready_or_tampered")
    if (_finite(snapshot.get("snapshot_age_seconds")) or 0.0) > 14 * 86400:
        alerts.append("snapshot_older_than_14_days")
    if operation_statuses.get("indeterminate", 0):
        alerts.append("indeterminate_tree_operations")
    if any(row.get("current_event_type") == "tree_failed" for row in tree_rows):
        alerts.append("tree_failed")
    if topology_violations:
        alerts.append("tree_topology_invariant_violation")
    if any(count > 1 for count in final_count_by_tree.values()):
        alerts.append("multiple_finalists_for_tree")
    if any(len(rows) > 1 for rows in handoffs_by_tree.values()):
        alerts.append("multiple_paid_handoffs_for_tree")
    if any(tree_id not in handoffs_by_tree for tree_id in final_count_by_tree):
        alerts.append("finalist_missing_official_handoff")

    return {
        "schema_version": "research_lab.git_tree_monitor.v1",
        "generated_at": now.isoformat(),
        "window_days": max(1, int(days)),
        "data_source_errors": source_errors,
        "tree_policy": dict(policy_doc),
        "tree_policy_error": policy_error,
        "trees": {
            "count": len(tree_rows),
            "current_event_counts": dict(
                sorted(
                    Counter(
                        str(row.get("current_event_type") or "unknown")
                        for row in tree_rows
                    ).items()
                )
            ),
            "event_counts": dict(
                sorted(Counter(str(row.get("event_type") or "unknown") for row in event_rows).items())
            ),
            "final_selection_count": len(final_events),
            "official_handoff_count": len(handoff_rows),
            "topology_violations": sorted(topology_violations),
        },
        "nodes": {
            "planned": len(node_rows),
            "evaluated": evaluated_count,
            "eligible": eligible_count,
            "eligibility_rate": (
                round(eligible_count / evaluated_count, 6)
                if evaluated_count
                else None
            ),
            "status_counts": dict(sorted(node_statuses.items())),
            "true_misses": true_misses,
            "evaluation_failures": evaluation_failures,
            "zero_outputs": zero_outputs,
        },
        "operations": {
            "count": len(operation_rows),
            "status_counts": dict(sorted(operation_statuses.items())),
            "kind_counts": dict(sorted(operation_kinds.items())),
            "settled_cost_microusd": settled_cost_microusd,
            "provider_call_count": provider_call_count,
        },
        "latest_selection": {
            "tree_id": latest_tree_id or None,
            "run_id": latest_selection_doc.get("run_id"),
            "selected_node_id": latest_selection_doc.get("selected_node_id"),
            "selection_hash": (
                latest_final.get("event_doc", {}).get("selection_hash")
                if isinstance(latest_final.get("event_doc"), Mapping)
                else None
            ),
            "official_candidate_id": latest_handoff.get("candidate_id"),
            "paid_finalist_count": latest_selection_doc.get("paid_finalist_count"),
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
