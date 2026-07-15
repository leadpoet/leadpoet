"""Aggregate Research Lab no-buildable-candidate diagnostics."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from .candidate_diagnostics import (
    NO_BUILDABLE_CANDIDATE_EVENT_TYPE,
    build_candidate_generation_failure_summary,
    canonical_loop_event_order,
    public_candidate_generation_failure_summary,
    sanitize_diagnostic_text,
)
from .store import select_all

_PUBLIC_CARD_COLUMNS = ",".join(
    (
        "current_run_id",
        "ticket_id",
        "current_outcome_label",
        "current_event_type",
        "current_candidate_count",
        "current_event_doc",
        "current_status_at",
    )
)
_LOOP_EVENT_COLUMNS = "run_id,seq,event_type,event_doc,created_at"
_RUN_ID_BATCH_SIZE = 150
_EVENT_FETCH_TIMEOUT_SECONDS = 10.0


def _event_doc(row: Mapping[str, Any]) -> Mapping[str, Any]:
    doc = row.get("event_doc") or row.get("current_event_doc")
    return doc if isinstance(doc, Mapping) else {}


def _run_id(row: Mapping[str, Any]) -> str:
    return str(row.get("run_id") or row.get("current_run_id") or "").strip()


def _ticket_id(row: Mapping[str, Any]) -> str:
    return str(row.get("ticket_id") or "").strip()


def _event_type(row: Mapping[str, Any]) -> str:
    return sanitize_diagnostic_text(
        row.get("event_type") or row.get("current_event_type"),
        max_length=120,
    )


def _primary_reason_from_public_row(row: Mapping[str, Any]) -> str:
    doc = _event_doc(row)
    failure = doc.get("candidate_generation_failure")
    if isinstance(failure, Mapping):
        reason = str(failure.get("primary_reason") or "").strip()
        if reason:
            return sanitize_diagnostic_text(reason, max_length=120)
    return ""


def _latest_stage_from_public_row(row: Mapping[str, Any]) -> str:
    doc = _event_doc(row)
    failure = doc.get("candidate_generation_failure")
    if isinstance(failure, Mapping):
        stage = str(failure.get("latest_stage") or failure.get("latest_event_type") or "").strip()
        if stage:
            return sanitize_diagnostic_text(stage, max_length=120)
    return sanitize_diagnostic_text(row.get("current_event_type") or row.get("event_type"), max_length=120)


def _category_for_reason(reason: str, latest_stage: str) -> str:
    text = f"{reason} {latest_stage}".lower()
    if any(marker in text for marker in ("binding_plan", "no_viable_patch", "source_missing", "source path")):
        return "source_path_failure"
    if "loop_direction" in text or "planner" in text:
        return "planner_failure"
    if any(marker in text for marker in ("patch", "build", "candidate_image", "artifact")):
        return "patch_or_build_failure"
    return "other_no_buildable"


def build_candidate_generation_failure_report(
    *,
    loop_event_rows: Sequence[Mapping[str, Any]],
    public_card_rows: Sequence[Mapping[str, Any]],
    days: int,
    generated_at: datetime | None = None,
    partial: bool = False,
    partial_reason: str = "",
) -> dict[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    events_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in loop_event_rows:
        run_id = _run_id(row)
        if run_id:
            events_by_run[run_id].append(row)

    summaries: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for row in public_card_rows:
        if str(row.get("current_outcome_label") or row.get("outcome_label") or "") != NO_BUILDABLE_CANDIDATE_EVENT_TYPE:
            continue
        run_id = _run_id(row)
        ticket_id = _ticket_id(row)
        loop_rows = events_by_run.get(run_id, [])
        failure_summary = build_candidate_generation_failure_summary(
            loop_rows,
            queue_reason=str(_event_doc(row).get("queue_reason") or ""),
            terminal_error=str(_event_doc(row).get("queue_reason") or ""),
            candidate_count=int(row.get("current_candidate_count") or row.get("candidate_count") or 0),
        )
        raw_terminal_available = any(
            _event_type(item) in {"loop_failed", NO_BUILDABLE_CANDIDATE_EVENT_TYPE}
            for item in loop_rows
        )
        public_reason = _primary_reason_from_public_row(row)
        primary_reason = (
            str(failure_summary.get("primary_reason") or "no_valid_image_build_finalists")
            if raw_terminal_available
            else public_reason or "no_valid_image_build_finalists"
        )
        latest_stage = (
            str(failure_summary.get("latest_stage") or "")
            if raw_terminal_available
            else _latest_stage_from_public_row(row)
        )
        summaries.append(
            {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "primary_reason": sanitize_diagnostic_text(primary_reason, max_length=120),
                "latest_stage": sanitize_diagnostic_text(latest_stage, max_length=120),
                "failure_category": _category_for_reason(primary_reason, latest_stage),
                "public_summary": public_candidate_generation_failure_summary(failure_summary),
            }
        )
        if run_id:
            seen_runs.add(run_id)

    # Catch current/raw loop failures that have not projected to a public card yet.
    for run_id, rows in events_by_run.items():
        if run_id in seen_runs:
            continue
        ordered_rows = canonical_loop_event_order(rows)
        terminal_rows = [
            row
            for row in ordered_rows
            if _event_type(row) in {"loop_failed", NO_BUILDABLE_CANDIDATE_EVENT_TYPE}
        ]
        if not terminal_rows:
            continue
        terminal_doc = _event_doc(terminal_rows[-1])
        run_summary = terminal_doc.get("run_summary") if isinstance(terminal_doc.get("run_summary"), Mapping) else {}
        try:
            selected_count = int(run_summary.get("selected_candidate_count") or 0)
        except (TypeError, ValueError):
            selected_count = 0
        if selected_count > 0:
            continue
        summary = build_candidate_generation_failure_summary(ordered_rows, candidate_count=0)
        reason = str(summary.get("primary_reason") or "no_valid_image_build_finalists")
        latest_stage = str(summary.get("latest_stage") or "")
        summaries.append(
            {
                "run_id": run_id,
                "ticket_id": "",
                "primary_reason": sanitize_diagnostic_text(reason, max_length=120),
                "latest_stage": sanitize_diagnostic_text(latest_stage, max_length=120),
                "failure_category": _category_for_reason(reason, latest_stage),
                "public_summary": public_candidate_generation_failure_summary(summary),
            }
        )

    by_reason = Counter(item["primary_reason"] for item in summaries)
    by_stage = Counter(item["latest_stage"] or "unknown" for item in summaries)
    by_category = Counter(item["failure_category"] for item in summaries)
    report = {
        "schema_version": "1.0",
        "report_type": "research_lab_candidate_generation_failures",
        "generated_at": generated.isoformat(),
        "days": max(0, int(days)),
        "partial": bool(partial),
        "total_no_buildable_candidate": len(summaries),
        "counts": {
            "by_primary_reason": dict(sorted(by_reason.items())),
            "by_latest_stage": dict(sorted(by_stage.items())),
            "by_failure_category": dict(sorted(by_category.items())),
        },
        "sample_runs": summaries[:25],
    }
    if partial:
        report["partial_reason"] = sanitize_diagnostic_text(partial_reason, max_length=160)
    return report


def _chunks(values: Sequence[str], size: int = _RUN_ID_BATCH_SIZE) -> Sequence[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), max(1, int(size)))]


def _unique_run_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    run_ids: list[str] = []
    for row in rows:
        run_id = _run_id(row)
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        run_ids.append(run_id)
    return run_ids


async def _fetch_loop_events_for_runs(
    run_ids: Sequence[str],
    *,
    event_filters: tuple[tuple[Any, ...], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in _chunks(list(run_ids)):
        if not batch:
            continue
        rows.extend(
            await select_all(
                "research_lab_auto_research_loop_events",
                columns=_LOOP_EVENT_COLUMNS,
                filters=(*event_filters, ("run_id", "in", list(batch))),
                order_by=(("seq", False), ("created_at", False)),
                max_rows=10000,
                allow_partial=True,
            )
        )
    return rows


async def _fetch_candidate_generation_event_rows(
    public_cards: Sequence[Mapping[str, Any]],
    *,
    event_filters: tuple[tuple[Any, ...], ...],
) -> list[dict[str, Any]]:
    terminal_events = await select_all(
        "research_lab_auto_research_loop_events",
        columns=_LOOP_EVENT_COLUMNS,
        filters=(*event_filters, ("event_type", "in", ["loop_failed", NO_BUILDABLE_CANDIDATE_EVENT_TYPE])),
        order_by=(("created_at", True),),
        max_rows=50000,
        allow_partial=True,
    )
    run_ids = _unique_run_ids([*public_cards, *terminal_events])
    return await _fetch_loop_events_for_runs(run_ids, event_filters=event_filters)


async def fetch_candidate_generation_failure_report(days: int) -> dict[str, Any]:
    safe_days = max(0, min(int(days), 90))
    cutoff = datetime.now(timezone.utc)
    if safe_days:
        cutoff = cutoff.replace(microsecond=0)
        since_dt = (cutoff - timedelta(days=safe_days)).isoformat()
        event_filters: tuple[tuple[Any, ...], ...] = (("created_at", "gte", since_dt),)
        card_filters: tuple[tuple[Any, ...], ...] = (("current_status_at", "gte", since_dt),)
    else:
        event_filters = ()
        card_filters = ()
    public_cards = await select_all(
        "research_lab_public_loop_card_current",
        columns=_PUBLIC_CARD_COLUMNS,
        filters=(*card_filters, ("current_outcome_label", NO_BUILDABLE_CANDIDATE_EVENT_TYPE)),
        order_by=(("current_status_at", True),),
        max_rows=50000,
        allow_partial=True,
    )
    partial = False
    partial_reason = ""
    try:
        loop_events = await asyncio.wait_for(
            _fetch_candidate_generation_event_rows(public_cards, event_filters=event_filters),
            timeout=_EVENT_FETCH_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        # Keep the internal report available from public projected cards even
        # when the optimized event lookup is temporarily slow/unavailable.
        loop_events = []
        partial = True
        partial_reason = f"optimized_event_fetch_failed:{type(exc).__name__}"
    return build_candidate_generation_failure_report(
        loop_event_rows=loop_events,
        public_card_rows=public_cards,
        days=safe_days,
        partial=partial,
        partial_reason=partial_reason,
    )
