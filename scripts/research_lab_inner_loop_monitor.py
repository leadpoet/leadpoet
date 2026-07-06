"""Inner-loop activation monitoring summary (plan "Monitoring" section).

Computes, from the persisted tables:
- **Dev-score coverage**: scored-candidate share by lane
  (research_lab_score_calibration rows with a dev_score, Phase 5 required).
- **Dev<->live rank agreement**: among runs with >=2 scored candidates, how
  often the dev-best candidate is also the live-best (the number that
  justifies — or kills — raising the candidate cap to 3).
- **Calibration drift**: mean |predicted_delta - realized_delta| over time.
- **Yield per run**: drafted/kept/crashed counts + realized delta per $ from
  research_lab_results_ledger.

Snapshot miss-rate by lane is log-based: grep worker logs for
``research_lab_loop_dev_eval_result`` (one structured line per evaluated
candidate).

Usage:
    python scripts/research_lab_inner_loop_monitor.py [--days 14] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from typing import Any


async def _collect(days: int) -> dict[str, Any]:
    from gateway.research_lab import store
    from gateway.research_lab.score_backfill import SCORE_CALIBRATION_TABLE

    calibration_rows = await store.select_many(
        SCORE_CALIBRATION_TABLE,
        columns=(
            "candidate_id,run_id,node_id,lane,plan_path_id,predicted_delta,dev_score,"
            "realized_mean_delta,realized_delta_lcb,outcome,created_at"
        ),
        filters=[],
        order_by=[("created_at", True)],
        limit=2000,
    )
    ledger_rows = await store.select_many(
        "research_lab_results_ledger",
        columns="ledger_row_id,status,delta_vs_parent,cost_usd,created_at",
        filters=[],
        order_by=[("created_at", True)],
        limit=2000,
    )

    # Dev-score coverage by lane.
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"scored": 0, "dev_scored": 0})
    for row in calibration_rows:
        lane = str(row.get("lane") or "unknown")
        coverage[lane]["scored"] += 1
        if row.get("dev_score") is not None:
            coverage[lane]["dev_scored"] += 1

    # Dev<->live rank agreement among multi-finalist runs.
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        if row.get("dev_score") is not None and row.get("realized_mean_delta") is not None:
            by_run[str(row.get("run_id") or "")].append(row)
    agreement_total = 0
    agreement_hits = 0
    for run_id, rows in by_run.items():
        if not run_id or len(rows) < 2:
            continue
        agreement_total += 1
        dev_best = max(rows, key=lambda r: float(r["dev_score"]))
        live_best = max(rows, key=lambda r: float(r["realized_mean_delta"]))
        if dev_best.get("candidate_id") == live_best.get("candidate_id"):
            agreement_hits += 1

    # Calibration drift: predicted vs realized.
    errors = [
        abs(float(row["predicted_delta"]) - float(row["realized_mean_delta"]))
        for row in calibration_rows
        if row.get("predicted_delta") is not None
        and row.get("realized_mean_delta") is not None
    ]

    # Ledger yield.
    status_counts: dict[str, int] = defaultdict(int)
    kept_delta = 0.0
    total_cost = 0.0
    for row in ledger_rows:
        status_counts[str(row.get("status") or "unknown")] += 1
        try:
            total_cost += max(0.0, float(row.get("cost_usd") or 0.0))
        except (TypeError, ValueError):
            pass
        if str(row.get("status")) == "keep" and row.get("delta_vs_parent") is not None:
            try:
                kept_delta += float(row["delta_vs_parent"])
            except (TypeError, ValueError):
                pass

    return {
        "window_days_requested": days,
        "calibration_row_count": len(calibration_rows),
        "dev_score_coverage_by_lane": {
            lane: {
                **counts,
                "coverage": round(counts["dev_scored"] / counts["scored"], 3)
                if counts["scored"]
                else None,
            }
            for lane, counts in sorted(coverage.items())
        },
        "dev_live_rank_agreement": {
            "multi_finalist_runs": agreement_total,
            "dev_best_was_live_best": agreement_hits,
            "agreement_rate": round(agreement_hits / agreement_total, 3)
            if agreement_total
            else None,
        },
        "calibration_drift": {
            "sample_count": len(errors),
            "mean_abs_error": round(sum(errors) / len(errors), 6) if errors else None,
        },
        "ledger_yield": {
            "status_counts": dict(sorted(status_counts.items())),
            "kept_delta_sum": round(kept_delta, 6),
            "total_cost_usd": round(total_cost, 2),
            "kept_delta_per_usd": round(kept_delta / total_cost, 6) if total_cost else None,
        },
        "notes": [
            "Snapshot miss-rate by lane: grep worker logs for research_lab_loop_dev_eval_result.",
            "Plateau-stop firing rate: grep loop events for stop_reason=dev_score_plateau.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14, help="Lookback window label")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON only")
    args = parser.parse_args()
    summary = asyncio.run(_collect(args.days))
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
