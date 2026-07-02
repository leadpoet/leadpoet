"""Passive scanner over canonical Research Lab stores."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from research_lab.observability.redaction import redact_for_langfuse

from .clusterer import cluster_events
from .config import ImprovementEngineConfig
from .diagnoser import draft_root_cause
from .dataset_generator import draft_dataset_spec
from .evaluator_generator import draft_evaluator_spec
from .fix_generator import draft_fix_spec
from .fingerprints import normalize_reason
from .models import ENGINE_VERSION, EngineIssue, EngineTraceEvent
from .prioritizer import priority_for_score, severity_score
from .store import fetch_recent_rows, persist_issue


def _since_iso(lookback_hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat().replace("+00:00", "Z")


def _event_time(row: Mapping[str, Any]) -> str:
    return str(row.get("created_at") or row.get("current_status_at") or "")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_bundle_events(row: Mapping[str, Any]) -> list[EngineTraceEvent]:
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
    scoring_health = doc.get("scoring_health") if isinstance(doc.get("scoring_health"), Mapping) else {}
    baseline_health = doc.get("baseline_health") if isinstance(doc.get("baseline_health"), Mapping) else {}
    events: list[EngineTraceEvent] = []
    run_id = str(row.get("run_id") or doc.get("run_id") or "")
    ticket_id = str(row.get("ticket_id") or doc.get("ticket_id") or "")
    score_hash = str(row.get("score_bundle_hash") or doc.get("score_bundle_hash") or "")
    created_at = _event_time(row)
    delta = _as_float(aggregates.get("mean_delta"))
    if int(scoring_health.get("sourced_zero_no_error_count") or 0) > 0:
        events.append(
            EngineTraceEvent(
                failure_category="candidate_model_zero_companies",
                runtime_stage="private_eval_pair",
                normalized_failure_reason="sourced_zero_no_error",
                component="sourcing_model",
                run_id=run_id,
                ticket_id=ticket_id,
                score_bundle_hash=score_hash,
                event_at=created_at,
                score_delta=delta,
                severity_hint=2.0,
                metadata={"scoring_health": _compact_health(scoring_health)},
            )
        )
    if str(row.get("bundle_status") or "") in {"failed", "rejected"}:
        events.append(
            EngineTraceEvent(
                failure_category="score_bundle_gateway_rejected",
                runtime_stage="score_bundle_post",
                normalized_failure_reason=str(row.get("bundle_status") or "score_bundle_not_scored"),
                component="score_bundle",
                run_id=run_id,
                ticket_id=ticket_id,
                score_bundle_hash=score_hash,
                event_at=created_at,
                score_delta=delta,
                severity_hint=2.5,
            )
        )
    if baseline_health and baseline_health.get("gate_passed") is False:
        events.append(
            EngineTraceEvent(
                failure_category="provider_error",
                runtime_stage="baseline_health",
                normalized_failure_reason="baseline_health_gate_failed",
                component="scoring_worker",
                run_id=run_id,
                ticket_id=ticket_id,
                score_bundle_hash=score_hash,
                event_at=created_at,
                severity_hint=2.0,
                metadata={"baseline_health": _compact_health(baseline_health)},
            )
        )
    return events


def _loop_event_to_engine(row: Mapping[str, Any]) -> EngineTraceEvent | None:
    event_type = str(row.get("event_type") or "")
    doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    reason = str(doc.get("failure_reason_code") or doc.get("failure_class") or doc.get("reason") or event_type)
    category = ""
    component = "auto_research_loop"
    if event_type in {"candidate_patch_apply_failed", "candidate_test_failed", "candidate_image_build_failed", "candidate_build_failed", "candidate_build_infra_failed"}:
        if event_type == "candidate_build_infra_failed":
            return None
        category = "candidate_build_failed"
        component = "code_build"
    elif event_type == "patch_validation_failed":
        category = "patch_scope_violation"
        component = "code_edit"
    elif event_type == "loop_failed":
        category = str(doc.get("failure_category") or "candidate_build_failed")
    elif event_type in {"loop_direction_plan_failed", "plan_alignment_rejected"}:
        category = "patch_scope_violation"
        component = "loop_planner"
    if not category:
        return None
    return EngineTraceEvent(
        failure_category=category,
        runtime_stage=event_type,
        normalized_failure_reason=normalize_reason(reason),
        component=component,
        run_id=str(row.get("run_id") or ""),
        ticket_id=str(row.get("ticket_id") or ""),
        candidate_id=str(row.get("candidate_id") or ""),
        event_type=event_type,
        event_at=_event_time(row),
        severity_hint=2.0,
        metadata={"event_doc_hash": _safe_hash_doc(doc)},
    )


def _candidate_event_to_engine(row: Mapping[str, Any]) -> EngineTraceEvent | None:
    status = str(row.get("candidate_status") or row.get("current_candidate_status") or "")
    reason = str(row.get("reason") or "")
    if status not in {"failed", "rejected"} and "cost" not in reason.lower():
        return None
    category = "cost_budget_exceeded" if "cost" in reason.lower() or "budget" in reason.lower() else "candidate_build_failed"
    return EngineTraceEvent(
        failure_category=category,
        runtime_stage="candidate_evaluation",
        normalized_failure_reason=normalize_reason(reason or status),
        component="scoring_worker",
        run_id=str(row.get("run_id") or ""),
        ticket_id=str(row.get("ticket_id") or ""),
        candidate_id=str(row.get("candidate_id") or ""),
        event_at=_event_time(row),
        severity_hint=2.0,
    )


def _compact_health(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in sorted(value)
        if key.endswith("_count") or key.endswith("_rate") or key in {"gate_passed", "health_status", "public_holdout_decision"}
    }


def _safe_hash_doc(value: Any) -> str:
    import hashlib
    import json

    safe = redact_for_langfuse(value, mode="prod")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def issue_from_events(
    fingerprint: str,
    events: Sequence[EngineTraceEvent],
    config: ImprovementEngineConfig | None = None,
) -> EngineIssue:
    ordered = sorted(events, key=lambda item: item.event_at or "")
    category = ordered[0].failure_category
    score = severity_score(ordered)
    linked_trace_ids = tuple(sorted({event.trace_id for event in ordered if event.trace_id})[:25])
    linked_hashes = tuple(sorted({event.score_bundle_hash for event in ordered if event.score_bundle_hash})[:25])
    linked_run_ids = tuple(sorted({event.run_id for event in ordered if event.run_id})[:50])
    linked_ticket_ids = tuple(sorted({event.ticket_id for event in ordered if event.ticket_id})[:50])
    root = draft_root_cause(ordered)
    temp_issue = EngineIssue(
        issue_key=f"engine_issue:{fingerprint.split(':', 1)[1][:24]}",
        title=_issue_title(category, ordered),
        status="open",
        priority=priority_for_score(score, category=category),
        category=category,
        fingerprint=fingerprint,
        first_seen_at=ordered[0].event_at,
        last_seen_at=ordered[-1].event_at,
        occurrence_count=len(ordered),
        severity_score=score,
        confidence=min(0.95, 0.50 + (len(ordered) * 0.05)),
        root_cause_doc=root,
        suggested_fix_doc={},
        evaluator_spec_doc={},
        dataset_spec_doc={},
        linked_trace_ids=linked_trace_ids,
        linked_score_bundle_hashes=linked_hashes,
        linked_run_ids=linked_run_ids,
        linked_ticket_ids=linked_ticket_ids,
        created_by_engine_version=ENGINE_VERSION,
    )
    return EngineIssue(
        **{
            **temp_issue.__dict__,
            "suggested_fix_doc": draft_fix_spec(temp_issue, config),
            "evaluator_spec_doc": draft_evaluator_spec(temp_issue),
            "dataset_spec_doc": draft_dataset_spec(temp_issue),
        }
    )


def _issue_title(category: str, events: Sequence[EngineTraceEvent]) -> str:
    components = Counter(event.component for event in events if event.component)
    component = components.most_common(1)[0][0] if components else "Research Lab"
    return f"{category.replace('_', ' ').title()} recurring in {component}"


async def collect_events(config: ImprovementEngineConfig) -> list[EngineTraceEvent]:
    since = _since_iso(config.lookback_hours)
    rows_limit = min(config.max_traces_per_scan, 5000)
    events: list[EngineTraceEvent] = []
    loop_rows = await fetch_recent_rows(
        "research_lab_auto_research_loop_events",
        filters=(("created_at", "gte", since),),
        limit=rows_limit,
    )
    for row in loop_rows:
        event = _loop_event_to_engine(row)
        if event is not None:
            events.append(event)
    candidate_rows = await fetch_recent_rows(
        "research_lab_candidate_evaluation_current",
        filters=(),
        limit=rows_limit,
    )
    for row in candidate_rows:
        event = _candidate_event_to_engine(row)
        if event is not None:
            events.append(event)
    bundle_rows = await fetch_recent_rows(
        "research_evaluation_score_bundle_current",
        filters=(("created_at", "gte", since),),
        limit=rows_limit,
    )
    for row in bundle_rows:
        events.extend(_score_bundle_events(row))
    return events[: config.max_traces_per_scan]


async def scan_for_issues(
    *,
    config: ImprovementEngineConfig | None = None,
    dry_run: bool = True,
    persist: bool = False,
) -> dict[str, Any]:
    cfg = config or ImprovementEngineConfig.from_env()
    events = await collect_events(cfg)
    clusters = cluster_events(events)
    issues = [
        issue_from_events(fingerprint, cluster, cfg)
        for fingerprint, cluster in clusters.items()
        if len(cluster) >= cfg.min_cluster_size
    ]
    persisted = []
    if persist:
        for issue in issues:
            persisted.append(await persist_issue(issue, dry_run=dry_run))
    return {
        "engine_version": ENGINE_VERSION,
        "enabled": cfg.enabled,
        "mode": cfg.mode,
        "dry_run": dry_run,
        "event_count": len(events),
        "cluster_count": len(clusters),
        "issue_count": len(issues),
        "issues": [issue.to_row() for issue in issues],
        "persisted": persisted,
    }
