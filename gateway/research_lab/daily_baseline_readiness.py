"""Shared effective-readiness gate for daily-baseline-dependent work."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import Any, Mapping

from research_lab.canonical import sha256_json

from leadpoet_canonical.production_parity_boundary_v2 import (
    configured_rebenchmark_now_v2,
)

from .config import ResearchLabGatewayConfig
from .promotion import load_active_private_model
from .store import select_many


logger = logging.getLogger(__name__)


def _positive_counts(values: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in values.items()
        if int(value) > 0
    }


def _assignment_completion_commitments(
    score_summary: Mapping[str, Any],
    *,
    rolling_window_hash: str,
    expected_policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    assignment = score_summary.get("category_assignment")
    if not isinstance(assignment, Mapping):
        return None
    expected_counts = {
        "public": int(expected_policy.get("public_total_icps") or 0),
        "private": int(expected_policy.get("private_total_icps") or 0),
        "conditional": int(expected_policy.get("conditional_total_icps") or 0),
    }
    if (
        str(assignment.get("rolling_window_hash") or "") != rolling_window_hash
        or str(assignment.get("policy_hash") or "")
        != str(expected_policy.get("policy_hash") or "")
        or dict(assignment.get("policy") or {}) != dict(expected_policy)
        or dict(assignment.get("category_counts") or {}) != expected_counts
    ):
        return None

    items = assignment.get("items")
    summaries = score_summary.get("per_icp_summaries")
    expected_total = sum(expected_counts.values())
    if (
        not isinstance(items, list)
        or not isinstance(summaries, list)
        or len(items) != expected_total
        or len(summaries) != expected_total
    ):
        return None

    refs: set[str] = set()
    hashes: set[str] = set()
    observed_counts = {category: 0 for category in expected_counts}
    observed_strength_counts = {
        category: {} for category in expected_counts
    }
    summaries_by_ref: dict[str, Mapping[str, Any]] = {}
    summary_hashes: set[str] = set()
    for summary in summaries:
        if not isinstance(summary, Mapping):
            return None
        ref = str(summary.get("icp_ref") or "")
        icp_hash = str(summary.get("icp_hash") or "")
        try:
            score = float(summary.get("score"))
        except (TypeError, ValueError):
            return None
        if (
            not ref
            or not icp_hash
            or ref in summaries_by_ref
            or icp_hash in summary_hashes
            or not math.isfinite(score)
            or not 0.0 <= score <= 100.0
        ):
            return None
        summaries_by_ref[ref] = summary
        summary_hashes.add(icp_hash)

    item_scores: list[float] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        ref = str(item.get("icp_ref") or "")
        icp_hash = str(item.get("icp_hash") or "")
        category = str(item.get("category") or "")
        strength = str(item.get("strength_label") or "")
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            return None
        summary = summaries_by_ref.get(ref)
        try:
            summary_score = float(summary.get("score")) if summary else math.nan
        except (TypeError, ValueError):
            return None
        if (
            not ref
            or not icp_hash
            or ref in refs
            or icp_hash in hashes
            or category not in observed_counts
            or not math.isfinite(score)
            or not 0.0 <= score <= 100.0
            or not isinstance(summary, Mapping)
            or str(summary.get("icp_hash") or "") != icp_hash
            or round(summary_score, 6) != round(score, 6)
        ):
            return None
        refs.add(ref)
        hashes.add(icp_hash)
        observed_counts[category] += 1
        category_strengths = observed_strength_counts[category]
        category_strengths[strength] = int(category_strengths.get(strength, 0)) + 1
        item_scores.append(score)
    if observed_counts != expected_counts:
        return None

    if set(summaries_by_ref) != refs or summary_hashes != hashes:
        return None

    expected_strength_counts = {
        "public": _positive_counts(
            {
                "weak": int(expected_policy.get("public_weak_total") or 0),
                "strong": int(expected_policy.get("public_strong_total") or 0),
            }
        ),
        "private": _positive_counts(
            {
                "weak": int(expected_policy.get("private_weak_total") or 0),
                "strong": int(expected_policy.get("private_strong_total") or 0),
            }
        ),
        "conditional": {"center": expected_counts["conditional"]},
    }
    if observed_strength_counts != expected_strength_counts:
        return None

    category_scores = assignment.get("category_scores")
    if not isinstance(category_scores, Mapping):
        return None
    for category in expected_counts:
        expected_score = round(
            sum(
                float(item["score"])
                for item in items
                if item.get("category") == category
            )
            / expected_counts[category],
            6,
        )
        try:
            observed_score = float(category_scores.get(category))
        except (TypeError, ValueError):
            return None
        if observed_score != expected_score:
            return None
    expected_aggregate = round(sum(item_scores) / expected_total, 6)
    try:
        observed_aggregate = float(assignment.get("aggregate_score"))
    except (TypeError, ValueError):
        return None
    if observed_aggregate != expected_aggregate:
        return None

    assignment_doc = {
        key: value
        for key, value in assignment.items()
        if key not in {"assignment_hash", "policy"}
    }
    assignment_hash = str(assignment.get("assignment_hash") or "")
    if assignment_hash != sha256_json(assignment_doc):
        return None
    return {
        "all_icp_count": expected_total,
        "per_icp_summaries_hash": sha256_json(
            {"per_icp_summaries": summaries}
        ),
        "category_assignment_hash": assignment_hash,
        "conditional_policy_hash": str(assignment.get("policy_hash") or ""),
        "category_counts": observed_counts,
        "category_strength_counts": observed_strength_counts,
        "minimum_icp_score": round(min(item_scores), 6),
        "maximum_icp_score": round(max(item_scores), 6),
    }


def _assignment_is_complete(
    score_summary: Mapping[str, Any],
    *,
    rolling_window_hash: str,
    expected_policy: Mapping[str, Any],
) -> bool:
    assignment = score_summary.get("category_assignment")
    if not isinstance(assignment, Mapping):
        return False
    expected_counts = {
        "public": int(expected_policy.get("public_total_icps") or 0),
        "private": int(expected_policy.get("private_total_icps") or 0),
        "conditional": int(expected_policy.get("conditional_total_icps") or 0),
    }
    if (
        str(assignment.get("rolling_window_hash") or "") != rolling_window_hash
        or str(assignment.get("policy_hash") or "")
        != str(expected_policy.get("policy_hash") or "")
        or dict(assignment.get("policy") or {}) != dict(expected_policy)
        or dict(assignment.get("category_counts") or {}) != expected_counts
    ):
        return False

    items = assignment.get("items")
    summaries = score_summary.get("per_icp_summaries")
    expected_total = sum(expected_counts.values())
    if (
        not isinstance(items, list)
        or not isinstance(summaries, list)
        or len(items) != expected_total
        or len(summaries) != expected_total
    ):
        return False

    refs: set[str] = set()
    hashes: set[str] = set()
    observed_counts = {category: 0 for category in expected_counts}
    for item in items:
        if not isinstance(item, Mapping):
            return False
        ref = str(item.get("icp_ref") or "")
        icp_hash = str(item.get("icp_hash") or "")
        category = str(item.get("category") or "")
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            return False
        if (
            not ref
            or not icp_hash
            or ref in refs
            or icp_hash in hashes
            or category not in observed_counts
            or not math.isfinite(score)
            or not 0.0 <= score <= 100.0
        ):
            return False
        refs.add(ref)
        hashes.add(icp_hash)
        observed_counts[category] += 1
    if observed_counts != expected_counts:
        return False

    summary_refs = {
        str(item.get("icp_ref") or "")
        for item in summaries
        if isinstance(item, Mapping) and str(item.get("icp_ref") or "")
    }
    if summary_refs != refs:
        return False

    assignment_doc = {
        key: value
        for key, value in assignment.items()
        if key not in {"assignment_hash", "policy"}
    }
    return str(assignment.get("assignment_hash") or "") == sha256_json(
        assignment_doc
    )


async def autoresearch_daily_baseline_readiness(
    config: ResearchLabGatewayConfig,
    *,
    now: datetime | None = None,
    include_commitments: bool = False,
) -> dict[str, Any]:
    """Resolve effective autoresearch readiness from durable daily authority.

    The operator maintenance event remains desired state. This gate prevents
    new autoresearch or candidate work until the exact active model has a
    complete same-day baseline and its linked public report is published.
    """

    if not config.private_baseline_rebenchmark_enabled:
        return {"available": True, "reason": "daily_baseline_disabled"}

    benchmark_date = (now or configured_rebenchmark_now_v2()).astimezone(
        timezone.utc
    ).date().isoformat()
    try:
        active = await load_active_private_model(config, register_bootstrap=False)
        artifact = active.artifact
        policy = config.conditional_validation_policy()
        policy_doc = policy.to_dict() if policy.enabled else {}
        benchmark_rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,benchmark_date,private_model_artifact_hash,"
                "private_model_manifest_hash,rolling_window_hash,aggregate_score,"
                "benchmark_quality,current_benchmark_status,score_summary_doc,"
                "benchmark_attempt,created_at"
            ),
            filters=(
                ("benchmark_date", benchmark_date),
                ("private_model_artifact_hash", artifact.model_artifact_hash),
                ("private_model_manifest_hash", artifact.manifest_hash),
                ("current_benchmark_status", "completed"),
            ),
            order_by=(("benchmark_attempt", True), ("created_at", True)),
            limit=10,
        )
        report_rows = await select_many(
            "research_lab_public_benchmark_report_current",
            columns=(
                "report_id,benchmark_date,benchmark_bundle_id,"
                "private_model_artifact_hash,private_model_manifest_hash,"
                "rolling_window_hash,benchmark_quality,current_report_status,"
                "benchmark_attempt,created_at"
            ),
            filters=(
                ("benchmark_date", benchmark_date),
                ("current_report_status", "published"),
                ("private_model_artifact_hash", artifact.model_artifact_hash),
                ("private_model_manifest_hash", artifact.manifest_hash),
            ),
            order_by=(("benchmark_attempt", True), ("created_at", True)),
            limit=10,
        )
    except Exception as exc:
        logger.warning(
            "research_lab_autoresearch_daily_baseline_gate_unavailable type=%s",
            type(exc).__name__,
        )
        return {
            "available": False,
            "reason": "daily_baseline_gate_unavailable",
            "benchmark_date": benchmark_date,
        }

    reports_by_bundle: dict[str, Mapping[str, Any]] = {}
    for report_row in report_rows:
        report_bundle_id = str(report_row.get("benchmark_bundle_id") or "")
        if report_bundle_id:
            reports_by_bundle.setdefault(report_bundle_id, report_row)
    for row in benchmark_rows:
        bundle_id = str(row.get("benchmark_bundle_id") or "")
        window_hash = str(row.get("rolling_window_hash") or "")
        score_summary = (
            row.get("score_summary_doc")
            if isinstance(row.get("score_summary_doc"), Mapping)
            else {}
        )
        summaries = score_summary.get("per_icp_summaries")
        report = reports_by_bundle.get(bundle_id)
        try:
            aggregate_score = float(row.get("aggregate_score"))
        except (TypeError, ValueError):
            continue
        if (
            not bundle_id
            or not window_hash
            or not isinstance(summaries, list)
            or not summaries
            or not math.isfinite(aggregate_score)
            or not 0.0 <= aggregate_score <= 100.0
            or str(row.get("benchmark_quality") or "") != "passed"
            or str(row.get("current_benchmark_status") or "") != "completed"
            or not isinstance(report, Mapping)
            or str(report.get("current_report_status") or "") != "published"
            or str(report.get("benchmark_quality") or "") != "passed"
            or str(report.get("benchmark_date") or "") != benchmark_date
            or str(report.get("rolling_window_hash") or "") != window_hash
        ):
            continue
        if policy.enabled and not _assignment_is_complete(
            score_summary,
            rolling_window_hash=window_hash,
            expected_policy=policy_doc,
        ):
            continue
        completion_commitments = None
        if policy.enabled and include_commitments:
            completion_commitments = _assignment_completion_commitments(
                score_summary,
                rolling_window_hash=window_hash,
                expected_policy=policy_doc,
            )
            if completion_commitments is None:
                continue
        result = {
            "available": True,
            "reason": "daily_baseline_published",
            "benchmark_date": benchmark_date,
            "report_id": str(report.get("report_id") or ""),
            "benchmark_bundle_id": bundle_id,
            "rolling_window_hash": window_hash,
        }
        if completion_commitments is not None:
            result["completion_commitments"] = completion_commitments
        return result
    return {
        "available": False,
        "reason": "daily_baseline_not_published",
        "benchmark_date": benchmark_date,
    }
