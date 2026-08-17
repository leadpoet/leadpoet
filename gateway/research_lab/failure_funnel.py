"""Private Research Lab failure-funnel report loader.

This module stays independent of the large API/auth import graph so the
read-only reporting contract can be tested without initializing the gateway.
"""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import Any, Mapping

from .store import call_rpc


logger = logging.getLogger(__name__)

_SAFE_CODE = re.compile(r"^[a-z0-9_:-]{1,120}$")
_MODEL_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")
_TELEMETRY_COUNT_FIELDS = (
    "bundle_count",
    "coverage_mismatch_count",
    "invalid_scoring_health_count",
    "degraded_scoring_health_count",
    "scoring_health_icp_count",
    "icp_row_count",
    "funnel_row_count",
    "company_label_count",
    "company_failure_count",
    "company_positive_count",
    "detailed_reason_gap_count",
    "detailed_reason_excess_count",
    "detailed_pass_gap_count",
    "detailed_pass_excess_count",
    "company_label_gap_count",
    "company_label_excess_count",
    "unclassified_failure_count",
    "infrastructure_failure_count",
    "unclassified_icp_failure_count",
    "scoring_run_count",
    "expected_execution_count",
    "execution_count",
    "terminal_execution_count",
    "failed_execution_count",
    "invalid_funnel_row_count",
    "nonterminal_execution_count",
    "degraded_execution_count",
)
_MINER_REASON_CODES = {
    "employee_count_mismatch",
    "employee_count_missing",
    "company_stage_mismatch",
    "company_stage_missing",
    "company_unverifiable",
    "intent_fabricated",
    "failed_prechecks",
    "duplicate_company",
    "required_attribute_not_proven",
    "company_fit_not_proven",
    "other",
    "no_companies_qualified",
    "no_scoreable_companies",
    "model_invalid_output",
    "model_runtime_skipped",
}
_MODEL_QUALITY_RUNTIME_REASONS = {
    "candidate_model_runtime_invalid_json",
    "candidate_model_runtime_adapter_failed",
    "candidate_model_runtime_invalid_output",
}
_MODEL_RUNTIME_SKIPPED_REASONS = {
    "candidate_model_runtime_skipped_after_timeout",
    "candidate_model_runtime_skipped_after_invalid_json",
    "candidate_model_runtime_skipped_after_adapter_failed",
    "candidate_model_runtime_skipped_after_invalid_output",
}
_MODEL_ZERO_REASON_TRANSLATIONS = {
    "candidate_model_zero_companies": ("sourcing", "no_companies_qualified"),
    "candidate_model_zero_scoreable_companies": (
        "scoring",
        "no_scoreable_companies",
    ),
}
_MINER_STAGE_CODES = {
    "sourcing",
    "firmographic",
    "verifier",
    "company_fit",
    "attribute",
    "intent",
    "scoring",
    "identity",
    "uniqueness",
    "pre_checks",
    "unclassified",
    "infrastructure",
}
_MINER_UNIT_CODES = {"companies", "icp_attempts"}
_INFRASTRUCTURE_MARKERS = (
    "infrastructure",
    "provider",
    "transport",
    "timeout",
    "network",
    "supabase",
    "persist",
)


def missing_failure_funnel(
    ticket_id: str,
    candidate_id: str | None,
    *,
    report_available: bool,
) -> dict[str, Any]:
    """Return an explicit no-data state; never turn missing telemetry into zero."""
    return {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": ticket_id,
        "candidate_id": candidate_id,
        "stages": [],
        "rejections": [],
        "model_revisions": [],
        "telemetry": {
            "status": "missing",
            "report_available": report_available,
        },
    }


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("failure-funnel count must be a non-negative integer")
    return value


def _safe_code(value: Any) -> str:
    code = str(value or "").strip().lower()
    if not _SAFE_CODE.fullmatch(code):
        raise ValueError("failure-funnel code is invalid")
    return code


def _decode_failure_funnel_report(
    report: Any,
    *,
    ticket_id: str,
    candidate_id: str | None,
) -> dict[str, Any] | None:
    if isinstance(report, list) and len(report) == 1 and isinstance(report[0], Mapping):
        report = report[0]
    if not isinstance(report, Mapping):
        return None
    report = dict(report)
    if report.get("schema_version") != "research_lab_failure_funnel.v1":
        return None
    if str(report.get("ticket_id") or "") != ticket_id:
        return None
    reported_candidate_id = report.get("candidate_id")
    normalized_candidate_id = (
        None if reported_candidate_id is None else str(reported_candidate_id)
    )
    if normalized_candidate_id != candidate_id:
        return None
    telemetry_raw = report.get("telemetry")
    if not isinstance(telemetry_raw, Mapping):
        return None
    telemetry_raw = dict(telemetry_raw)
    status = str(telemetry_raw.get("status") or "")
    if status not in {"complete", "partial", "missing"}:
        return None

    stages_raw = report.get("stages")
    rejections_raw = report.get("rejections")
    revisions_raw = report.get("model_revisions")
    if not isinstance(stages_raw, list) or not isinstance(rejections_raw, list):
        return None
    if not isinstance(revisions_raw, list):
        return None

    stages: list[dict[str, Any]] = []
    for raw in stages_raw:
        if not isinstance(raw, Mapping):
            return None
        stage = _safe_code(raw.get("stage"))
        unit = _safe_code(raw.get("unit"))
        reviewed = _count(raw.get("reviewed"))
        passed = _count(raw.get("passed"))
        rejected = _count(raw.get("rejected"))
        if passed + rejected != reviewed:
            return None
        stages.append(
            {
                "stage": stage,
                "unit": unit,
                "reviewed": reviewed,
                "passed": passed,
                "rejected": rejected,
            }
        )

    rejections: list[dict[str, Any]] = []
    for raw in rejections_raw:
        if not isinstance(raw, Mapping):
            return None
        rejections.append(
            {
                "stage": _safe_code(raw.get("stage")),
                "reason_code": _safe_code(raw.get("reason_code")),
                "unit": _safe_code(raw.get("unit")),
                "count": _count(raw.get("count")),
            }
        )

    model_revisions: list[str] = []
    for raw in revisions_raw:
        revision = str(raw or "").strip().lower()
        if not _MODEL_REVISION.fullmatch(revision):
            return None
        model_revisions.append(revision)

    if any(field not in telemetry_raw for field in _TELEMETRY_COUNT_FIELDS):
        return None
    telemetry: dict[str, Any] = {"status": status}
    for field in _TELEMETRY_COUNT_FIELDS:
        if field in telemetry_raw:
            telemetry[field] = _count(telemetry_raw[field])
    if isinstance(telemetry_raw.get("report_available"), bool):
        telemetry["report_available"] = telemetry_raw["report_available"]
    return {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": ticket_id,
        "candidate_id": candidate_id,
        "stages": stages,
        "rejections": rejections,
        "model_revisions": sorted(set(model_revisions)),
        "telemetry": telemetry,
    }


def miner_failure_funnel_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the whole-pool, provider-neutral projection safe for the owner."""
    combined: Counter[tuple[str, str, str]] = Counter()
    for raw in report.get("rejections") or ():
        if not isinstance(raw, Mapping):
            continue
        stage = str(raw.get("stage") or "unclassified")
        reason = str(raw.get("reason_code") or "other")
        unit = str(raw.get("unit") or "companies")
        blob = f"{stage} {reason}".lower()
        if reason in _MODEL_QUALITY_RUNTIME_REASONS:
            stage = "scoring"
            reason = "model_invalid_output"
        elif reason in _MODEL_RUNTIME_SKIPPED_REASONS:
            stage = "sourcing"
            reason = "model_runtime_skipped"
        elif reason in _MODEL_ZERO_REASON_TRANSLATIONS:
            stage, reason = _MODEL_ZERO_REASON_TRANSLATIONS[reason]
        elif any(marker in blob for marker in _INFRASTRUCTURE_MARKERS):
            stage = "infrastructure"
            reason = "external_service_failure"
        else:
            if stage not in _MINER_STAGE_CODES:
                stage = "unclassified"
            if reason not in _MINER_REASON_CODES:
                reason = "unclassified_failure"
        if unit not in _MINER_UNIT_CODES:
            unit = "units"
        combined[(stage, reason, unit)] += int(raw.get("count") or 0)

    telemetry_raw = report.get("telemetry")
    telemetry: dict[str, Any] = {"status": "missing"}
    if isinstance(telemetry_raw, Mapping):
        raw_status = str(telemetry_raw.get("status") or "missing")
        telemetry["status"] = (
            raw_status if raw_status in {"complete", "partial", "missing"} else "missing"
        )
        for field in _TELEMETRY_COUNT_FIELDS:
            value = telemetry_raw.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                telemetry[field] = value
        if isinstance(telemetry_raw.get("report_available"), bool):
            telemetry["report_available"] = telemetry_raw["report_available"]
    stages = []
    for raw in report.get("stages") or ():
        if not isinstance(raw, Mapping):
            continue
        stage = str(raw.get("stage") or "unclassified")
        unit = str(raw.get("unit") or "companies")
        stages.append(
            {
                "stage": stage if stage in _MINER_STAGE_CODES else "unclassified",
                "unit": unit if unit in _MINER_UNIT_CODES else "units",
                "reviewed": int(raw.get("reviewed") or 0),
                "passed": int(raw.get("passed") or 0),
                "rejected": int(raw.get("rejected") or 0),
            }
        )
    return {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": str(report.get("ticket_id") or ""),
        "candidate_id": report.get("candidate_id"),
        "stages": stages,
        "rejections": [
            {
                "stage": stage,
                "reason_code": reason,
                "unit": unit,
                "count": count,
            }
            for (stage, reason, unit), count in sorted(combined.items())
        ],
        "telemetry": telemetry,
    }


async def build_ticket_failure_funnel(
    ticket_id: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Load the service-only aggregate without making diagnostics brittle.

    The report migration can be rolled out before or after the gateway. A
    missing function or temporary reporting read failure therefore degrades
    only this optional panel and never hides the existing loop diagnostics.
    """
    try:
        report = await call_rpc(
            "get_research_lab_failure_funnel",
            {
                "p_ticket_id": ticket_id,
                "p_candidate_id": candidate_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 - optional reporting projection
        logger.warning(
            "research_lab_failure_funnel_unavailable ticket_id=%s error_type=%s",
            ticket_id,
            type(exc).__name__,
        )
        return missing_failure_funnel(ticket_id, candidate_id, report_available=False)
    try:
        decoded = _decode_failure_funnel_report(
            report,
            ticket_id=ticket_id,
            candidate_id=candidate_id,
        )
    except Exception as exc:  # noqa: BLE001 - malformed optional projection
        logger.warning(
            "research_lab_failure_funnel_invalid ticket_id=%s error_type=%s",
            ticket_id,
            type(exc).__name__,
        )
        decoded = None
    if decoded is None:
        return missing_failure_funnel(ticket_id, candidate_id, report_available=True)
    return decoded
