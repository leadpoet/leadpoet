"""Shared readiness gate for the daily public-baseline rebenchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import Any, Mapping

from leadpoet_canonical.production_parity_boundary_v2 import (
    configured_rebenchmark_now_v2,
)

from .config import ResearchLabGatewayConfig
from .public_baseline_runner import BASELINE_ID
from .public_baseline_store import load_completed_run


logger = logging.getLogger(__name__)


def _valid_completed_row(row: Mapping[str, Any]) -> bool:
    results = row.get("per_icp_results")
    summary = row.get("score_summary_doc")
    report = row.get("public_report_doc")
    try:
        aggregate = float(row.get("aggregate_score"))
        expected = int(row.get("expected_icp_count") or 0)
        completed = int(row.get("completed_icp_count") or 0)
    except (TypeError, ValueError):
        return False
    if (
        str(row.get("status") or "") != "completed"
        or not math.isfinite(aggregate)
        or not 0.0 <= aggregate <= 100.0
        or expected <= 0
        or completed != expected
        or not isinstance(results, list)
        or len(results) != expected
        or not isinstance(summary, Mapping)
        or not isinstance(report, Mapping)
        or not summary
        or not report
    ):
        return False
    refs: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping):
            return False
        ref = str(result.get("icp_ref") or "")
        if (
            not ref
            or ref in refs
            or str(result.get("status") or "") != "completed"
            or not isinstance(result.get("summary"), Mapping)
        ):
            return False
        refs.add(ref)
    return (
        str(summary.get("baseline", {}).get("id") or "") == BASELINE_ID
        if isinstance(summary.get("baseline"), Mapping)
        else False
    )


async def daily_public_baseline_readiness(
    config: ResearchLabGatewayConfig,
    *,
    now: datetime | None = None,
    include_commitments: bool = False,
) -> dict[str, Any]:
    """Return whether today's public baseline has a complete durable result."""

    if not config.public_baseline_rebenchmark_enabled:
        return {"available": True, "reason": "daily_baseline_disabled"}
    benchmark_date = configured_rebenchmark_now_v2(
        now=now or datetime.now(timezone.utc)
    ).date().isoformat()
    try:
        row = await load_completed_run(
            benchmark_date=benchmark_date,
            baseline_id=BASELINE_ID,
        )
    except Exception as exc:
        logger.warning(
            "research_lab_daily_baseline_gate_unavailable type=%s",
            type(exc).__name__,
        )
        return {
            "available": False,
            "reason": "daily_baseline_gate_unavailable",
            "benchmark_date": benchmark_date,
        }
    if not isinstance(row, Mapping) or not _valid_completed_row(row):
        return {
            "available": False,
            "reason": "daily_baseline_not_published",
            "benchmark_date": benchmark_date,
        }
    result = {
        "available": True,
        "reason": "daily_baseline_published",
        "benchmark_date": benchmark_date,
        "baseline_run_id": str(row.get("run_id") or ""),
        "baseline_id": BASELINE_ID,
        "rolling_window_hash": str(row.get("rolling_window_hash") or ""),
    }
    if include_commitments:
        summaries = row["score_summary_doc"].get("per_icp_summaries") or []
        scores = [float(value.get("score") or 0.0) for value in summaries]
        result["completion_commitments"] = {
            "all_icp_count": len(summaries),
            "minimum_icp_score": round(min(scores), 6) if scores else 0.0,
            "maximum_icp_score": round(max(scores), 6) if scores else 0.0,
        }
    return result


async def autoresearch_daily_baseline_readiness(
    config: ResearchLabGatewayConfig,
    *,
    now: datetime | None = None,
    include_commitments: bool = False,
) -> dict[str, Any]:
    """Compatibility name for callers being replaced by the Arena service."""

    return await daily_public_baseline_readiness(
        config,
        now=now,
        include_commitments=include_commitments,
    )
