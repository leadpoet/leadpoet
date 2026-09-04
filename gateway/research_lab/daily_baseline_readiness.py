"""Shared readiness gate for the daily public-baseline rebenchmark."""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import math
from typing import Any, Mapping

from leadpoet_canonical.production_parity_boundary_v2 import (
    configured_rebenchmark_now_v2,
)

from .config import ResearchLabGatewayConfig
from .daily_icp_set import (
    DAILY_ICP_COUNT,
    DailyIcpSetUnavailable,
    daily_icp_set_from_input_doc,
)
from .public_baseline_runner import (
    BASELINE_ENTRYPOINT,
    BASELINE_ID,
    BASELINE_REPOSITORY,
)
from .public_baseline_store import load_completed_run


logger = logging.getLogger(__name__)


def _same_number(left: Any, right: Any) -> bool:
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(first) and math.isfinite(second) and math.isclose(
        first, second, abs_tol=1e-6
    )


def valid_completed_row(row: Mapping[str, Any]) -> bool:
    results = row.get("per_icp_results")
    summary = row.get("score_summary_doc")
    report = row.get("public_report_doc")
    window = row.get("window_doc")
    input_doc = row.get("benchmark_input_doc")
    try:
        benchmark_date = date.fromisoformat(str(row.get("benchmark_date") or ""))
        benchmark_date_text = benchmark_date.isoformat()
        expected_set_id = int(benchmark_date.strftime("%Y%m%d"))
        aggregate = float(row.get("aggregate_score"))
        expected = int(row.get("expected_icp_count") or 0)
        completed = int(row.get("completed_icp_count") or 0)
    except (TypeError, ValueError):
        return False
    if not isinstance(input_doc, Mapping):
        return False
    try:
        frozen_set = daily_icp_set_from_input_doc(
            input_doc,
            required_set_id=expected_set_id,
        )
    except DailyIcpSetUnavailable:
        return False
    if (
        str(row.get("status") or "") != "completed"
        or str(row.get("baseline_id") or "") != BASELINE_ID
        or str(row.get("baseline_repository") or "") != BASELINE_REPOSITORY
        or str(row.get("baseline_entrypoint") or "") != BASELINE_ENTRYPOINT
        or not math.isfinite(aggregate)
        or not 0.0 <= aggregate <= 100.0
        or expected != DAILY_ICP_COUNT
        or completed != expected
        or not isinstance(results, list)
        or len(results) != expected
        or not isinstance(summary, Mapping)
        or not isinstance(report, Mapping)
        or not isinstance(window, Mapping)
        or not summary
        or not report
        or dict(input_doc) != frozen_set.input_doc
        or dict(window) != frozen_set.public_doc
    ):
        return False
    window_refs = window.get("icp_refs")
    if (
        window.get("icp_count") != expected
        or not isinstance(window_refs, list)
        or len(window_refs) != expected
        or len({str(value) for value in window_refs}) != expected
    ):
        return False
    summaries = summary.get("per_icp_summaries")
    if not isinstance(summaries, list) or len(summaries) != expected:
        return False
    result_summaries: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            return False
        ref = str(result.get("icp_ref") or "")
        if (
            not ref
            or ref in result_summaries
            or str(result.get("status") or "") != "completed"
            or not isinstance(result.get("summary"), Mapping)
        ):
            return False
        result_summaries[ref] = result["summary"]
    summary_by_ref: dict[str, Mapping[str, Any]] = {}
    for item in summaries:
        if not isinstance(item, Mapping):
            return False
        ref = str(item.get("icp_ref") or "")
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            return False
        if (
            not ref
            or ref in summary_by_ref
            or not math.isfinite(score)
            or not 0.0 <= score <= 100.0
            or ref not in result_summaries
            or not _same_number(result_summaries[ref].get("score"), score)
        ):
            return False
        summary_by_ref[ref] = item
    if (
        set(result_summaries) != set(summary_by_ref)
        or set(result_summaries) != {str(value) for value in window_refs}
    ):
        return False
    calculated = sum(float(item["score"]) for item in summaries) / expected
    report_rows = report.get("per_icp")
    if not isinstance(report_rows, list) or len(report_rows) != expected:
        return False
    report_by_ref: dict[str, Mapping[str, Any]] = {}
    for item in report_rows:
        if not isinstance(item, Mapping):
            return False
        ref = str(item.get("icp_ref") or "")
        if (
            not ref
            or ref in report_by_ref
            or ref not in summary_by_ref
            or not _same_number(item.get("score"), summary_by_ref[ref].get("score"))
        ):
            return False
        report_by_ref[ref] = item
    summary_baseline = summary.get("baseline")
    report_baseline = report.get("baseline")
    try:
        report_completed = int(report.get("completed_icp_count") or 0)
    except (TypeError, ValueError):
        return False
    return (
        set(report_by_ref) == set(summary_by_ref)
        and isinstance(summary_baseline, Mapping)
        and isinstance(report_baseline, Mapping)
        and str(summary_baseline.get("id") or "") == BASELINE_ID
        and str(report_baseline.get("id") or "") == BASELINE_ID
        and str(summary.get("benchmark_date") or "") == benchmark_date_text
        and str(report.get("benchmark_date") or "") == benchmark_date_text
        and summary.get("icp_set_id") == expected_set_id
        and report.get("icp_set_id") == expected_set_id
        and report_completed == expected
        and _same_number(summary.get("aggregate_score"), aggregate)
        and _same_number(report.get("aggregate_score"), aggregate)
        and _same_number(calculated, aggregate)
    )


async def daily_public_baseline_readiness(
    config: ResearchLabGatewayConfig,
    *,
    now: datetime | None = None,
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
    if not isinstance(row, Mapping) or not valid_completed_row(row):
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
        "aggregate_score": float(row.get("aggregate_score") or 0.0),
        "completed_icp_count": int(row.get("completed_icp_count") or 0),
    }
    return result
