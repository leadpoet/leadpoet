#!/usr/bin/env python3
"""Read-only Research Lab operator health snapshot.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment.
Prints only aggregate counts and compact refs; no secrets or raw private payloads.

Every probe is an isolated check: a single failing check (network error, schema
drift, unexpected payload) is reported as a ``CHECK-FAILED`` line and the
remaining checks still run. The script exits nonzero when any check failed to
execute; ``alert_*`` lines are findings and do not change the exit code.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import re
import sys
from typing import Any, Callable
from urllib import parse, request


DEFAULT_BASELINE_JUMP_THRESHOLD = 3.0
DEFAULT_ACTIVE_STALE_SECONDS = 10 * 60
ACTIVE_CANDIDATE_STATUSES = {"queued", "assigned", "evaluating"}
PROVIDER_FAILURE_CATEGORIES = {
    "runtime_provider_error",
    "provider_http_4xx",
    "provider_http_5xx",
    "reference_model_runtime_provider_error",
}


def _supabase_config() -> tuple[str, dict[str, str]]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("missing required env vars: SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    return f"{url}/rest/v1/", headers


def _get(table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    base_url, headers = _supabase_config()
    url = base_url + table + "?" + parse.urlencode(params, doseq=True, safe="*,.:->")
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _compact(value: object, limit: int = 12) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field) or "") for row in rows))


def _parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    text = re.sub(r"([+-]\d{2})$", r"\1:00", text)
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    match = re.match(r"^(.*[ T]\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:\d{2})$", text)
    if match:
        fraction = (match.group(2) + "000000")[:6]
        text = f"{match.group(1)}.{fraction}{match.group(3)}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _oldest_age_hours(rows: list[dict[str, Any]], field: str = "current_status_at") -> float | None:
    values = [_parse_ts(row.get(field)) for row in rows]
    parsed = [value for value in values if value is not None]
    if not parsed:
        return None
    return round((datetime.now(timezone.utc) - min(parsed)).total_seconds() / 3600.0, 2)


def _age_seconds(row: dict[str, Any], field: str = "current_status_at") -> float | None:
    parsed = _parse_ts(row.get(field))
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()


def _age_minutes(row: dict[str, Any], field: str = "current_status_at") -> float | None:
    age = _age_seconds(row, field)
    return None if age is None else round(age / 60.0, 1)


def _stale_active_rows(
    rows: list[dict[str, Any]],
    *,
    status_field: str,
    active_statuses: set[str],
    stale_after_seconds: int = DEFAULT_ACTIVE_STALE_SECONDS,
) -> list[dict[str, Any]]:
    stale = []
    for row in rows:
        if str(row.get(status_field) or "") not in active_statuses:
            continue
        age = _age_seconds(row)
        if age is None or age >= stale_after_seconds:
            stale.append(row)
    return stale


def _failure_class(row: dict[str, Any]) -> str:
    doc = row.get("event_doc")
    if isinstance(doc, dict) and doc.get("failure_class"):
        return str(doc["failure_class"])
    return str(row.get("reason") or "unknown")


def _candidate_lane_and_files(value: Any) -> tuple[str, tuple[str, ...]]:
    manifest = value if isinstance(value, dict) else {}
    patch_doc = manifest.get("patch_doc") if isinstance(manifest.get("patch_doc"), dict) else {}
    code_edit = patch_doc.get("code_edit") if isinstance(patch_doc.get("code_edit"), dict) else {}
    lane = str(code_edit.get("lane") or patch_doc.get("lane") or "unknown").strip() or "unknown"
    raw_files = code_edit.get("target_files") or patch_doc.get("target_files") or ()
    files: list[str] = []
    if isinstance(raw_files, list):
        for item in raw_files:
            path = str(item or "").strip()
            if path and not _looks_sensitive(path):
                files.append(path[:160])
    return lane[:80], tuple(files[:12])


def _score_bundle_health_status(row: dict[str, Any]) -> str:
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), dict) else {}
    health = doc.get("scoring_health") if isinstance(doc.get("scoring_health"), dict) else {}
    return str(health.get("health_status") or "missing")


def _score_bundle_public_holdout_decision(row: dict[str, Any]) -> str:
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), dict) else {}
    health = doc.get("scoring_health") if isinstance(doc.get("scoring_health"), dict) else {}
    return str(health.get("public_holdout_decision") or "missing")


def _score_bundle_reference_mode(row: dict[str, Any]) -> str:
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), dict) else {}
    gate = doc.get("private_holdout_gate") if isinstance(doc.get("private_holdout_gate"), dict) else {}
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), dict) else {}
    return str(
        gate.get("reference_evaluation_mode")
        or aggregates.get("reference_evaluation_mode")
        or "unknown"
    )


def _event_cost_usd(row: dict[str, Any]) -> float:
    ledger = row.get("cost_ledger") if isinstance(row.get("cost_ledger"), dict) else {}
    for key in ("total_usd", "actual_openrouter_cost_usd", "estimated_cost_usd"):
        try:
            return float(ledger.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _safe_div(numerator: float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "sk-or-",
            "service_role",
            "api_key",
            "secret",
            "proxy",
            "://",
        )
    )


# ---------------------------------------------------------------------------
# Pure helpers for the newer checks (unit-testable without network access).
# ---------------------------------------------------------------------------


def _stale_scoring_cards(
    cards: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Public loop cards stuck at ``scoring`` with no actively scoring candidate."""
    active_runs = {
        str(row.get("run_id") or "")
        for row in candidates
        if str(row.get("current_candidate_status") or "") in ACTIVE_CANDIDATE_STATUSES
        and str(row.get("run_id") or "")
    }
    stale = []
    for card in cards:
        if str(card.get("current_event_type") or "") != "scoring":
            continue
        run_id = str(card.get("current_run_id") or "")
        if run_id not in active_runs:
            stale.append(card)
    return stale


def _latest_completed_score_by_day(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Map benchmark_date -> aggregate_score of the newest completed bundle that day."""
    latest: dict[str, tuple[datetime, float]] = {}
    for row in rows:
        status = str(row.get("current_benchmark_status") or "")
        if status and status != "completed":
            continue
        day = str(row.get("benchmark_date") or "")
        if not day:
            continue
        created = _parse_ts(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)
        try:
            score = float(row.get("aggregate_score") or 0.0)
        except (TypeError, ValueError):
            continue
        known = latest.get(day)
        if known is None or created > known[0]:
            latest[day] = (created, score)
    return {day: score for day, (_, score) in latest.items()}


def _baseline_day_jumps(
    score_by_day: dict[str, float],
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Consecutive-day |score delta| findings above ``threshold`` (§0-N1 signal)."""
    days = sorted(score_by_day)
    jumps = []
    for previous, current in zip(days, days[1:]):
        delta = round(score_by_day[current] - score_by_day[previous], 6)
        if abs(delta) > threshold:
            jumps.append((previous, current, delta))
    return jumps


def _duplicate_day_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """benchmark_date -> bundle count for days with more than one bundle."""
    counts = Counter(
        str(row.get("benchmark_date") or "")
        for row in rows
        if str(row.get("benchmark_date") or "")
    )
    return {day: count for day, count in sorted(counts.items()) if count > 1}


def _summary_failure_categories(summary: dict[str, Any]) -> list[str]:
    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}
    value = diagnostics.get("failure_categories")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _unresolved_provider_error_icps(summaries: list[dict[str, Any]]) -> list[str]:
    """ICPs booked as zero into the day's reference with provider failure categories."""
    flagged = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        try:
            score = float(summary.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0.0:
            continue
        categories = set(_summary_failure_categories(summary))
        if categories & PROVIDER_FAILURE_CATEGORIES or any("provider" in item for item in categories):
            flagged.append(str(summary.get("icp_ref") or ""))
    return flagged


def _zero_company_no_error_icps(summaries: list[dict[str, Any]]) -> list[str]:
    """Zero-company ICPs with no recorded failure category (bug #35 signal)."""
    flagged = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        try:
            company_count = int(summary.get("company_count") or 0)
        except (TypeError, ValueError):
            company_count = 0
        if company_count <= 0 and not _summary_failure_categories(summary):
            flagged.append(str(summary.get("icp_ref") or ""))
    return flagged


# ---------------------------------------------------------------------------
# Checks. Each returns printable lines and may raise; the harness isolates it.
# ---------------------------------------------------------------------------


def check_queue_and_loops(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    queue = fetch(
        "research_loop_run_queue_current",
        {
            "select": "run_id,ticket_id,current_queue_status,current_reason,worker_ref,current_status_at",
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    loops = fetch(
        "research_lab_auto_research_loop_current",
        {
            "select": "run_id,current_loop_status,current_event_type,current_worker_ref,current_status_at",
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    queue_by_run = {str(row.get("run_id")): row for row in queue}
    mismatches = []
    for loop in loops:
        run_id = str(loop.get("run_id") or "")
        queue_row = queue_by_run.get(run_id)
        if not queue_row:
            continue
        queue_status = str(queue_row.get("current_queue_status") or "")
        loop_status = str(loop.get("current_loop_status") or "")
        if queue_status in {"completed", "failed"} and loop_status not in {"completed", "failed"}:
            mismatches.append((run_id, queue_status, loop_status, loop.get("current_event_type")))
    credit_blocked = [
        row
        for row in queue
        if str(row.get("current_queue_status") or "") == "paused"
        and str(row.get("current_reason") or "") == "blocked_for_credit"
    ]
    stale_started_queue = _stale_active_rows(
        queue,
        status_field="current_queue_status",
        active_statuses={"started"},
    )
    stale_running_loops = _stale_active_rows(
        loops,
        status_field="current_loop_status",
        active_statuses={"running", "paused"},
    )
    lines = [
        f"queue_status_counts={_counts(queue, 'current_queue_status')}",
        f"loop_status_counts={_counts(loops, 'current_loop_status')}",
        f"terminal_queue_nonterminal_loop_mismatches={len(mismatches)}",
        f"alert_stale_started_queue_count={len(stale_started_queue)}",
        f"alert_stale_running_loop_count={len(stale_running_loops)}",
    ]
    for run_id, queue_status, loop_status, event_type in mismatches[:10]:
        lines.append(f"  mismatch run={_compact(run_id)} queue={queue_status} loop={loop_status} loop_event={event_type}")
    for row in stale_started_queue[:10]:
        lines.append(
            "  stale_started_queue"
            f" run={_compact(row.get('run_id'))}"
            f" ticket={_compact(row.get('ticket_id'))}"
            f" worker={_compact(row.get('worker_ref'))}"
            f" age_minutes={_age_minutes(row)}"
            f" at={row.get('current_status_at')}"
        )
    for row in stale_running_loops[:10]:
        lines.append(
            "  stale_running_loop"
            f" run={_compact(row.get('run_id'))}"
            f" worker={_compact(row.get('current_worker_ref'))}"
            f" event={row.get('current_event_type')}"
            f" age_minutes={_age_minutes(row)}"
            f" at={row.get('current_status_at')}"
        )
    lines.append(f"credit_blocked_paused_count={len(credit_blocked)}")
    for row in credit_blocked[:10]:
        lines.append(
            "  credit_blocked"
            f" run={_compact(row.get('run_id'))}"
            f" ticket={_compact(row.get('ticket_id'))}"
            f" at={row.get('current_status_at')}"
        )
    return lines


def check_candidates(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    candidates = fetch(
        "research_lab_candidate_evaluation_current",
        {
            "select": (
                "candidate_id,run_id,candidate_artifact_hash,parent_artifact_hash,current_candidate_status,"
                "current_reason,current_score_bundle_id,candidate_patch_manifest,current_status_at"
            ),
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    candidate_events = fetch(
        "research_lab_candidate_evaluation_events",
        {"select": "candidate_id,reason,event_doc,created_at", "event_type": "eq.failed", "order": "created_at.desc", "limit": "100"},
    )
    stale_parent_backlog = [
        row
        for row in candidates
        if str(row.get("current_candidate_status") or "") == "rejected"
        and str(row.get("current_reason") or "") == "stale_parent_needs_rescore"
    ]
    baseline_not_ready = [
        row
        for row in candidates
        if str(row.get("current_candidate_status") or "") == "queued"
        and str(row.get("current_reason") or "") == "baseline_not_ready"
    ]
    active_scoring = [
        row
        for row in candidates
        if str(row.get("current_candidate_status") or "") in {"assigned", "evaluating"}
    ]
    stale_active_scoring = _stale_active_rows(
        active_scoring,
        status_field="current_candidate_status",
        active_statuses={"assigned", "evaluating"},
    )
    lane_counts: Counter[str] = Counter()
    target_file_counts: Counter[str] = Counter()
    for row in candidates:
        lane, files = _candidate_lane_and_files(row.get("candidate_patch_manifest"))
        lane_counts[lane] += 1
        for path in files:
            target_file_counts[path] += 1
    lines = [
        f"candidate_status_counts={_counts(candidates, 'current_candidate_status')}",
        f"candidate_failure_class_counts={dict(Counter(_failure_class(row) for row in candidate_events))}",
        f"candidate_lane_counts={dict(lane_counts.most_common(10))}",
        f"candidate_target_file_counts={dict(target_file_counts.most_common(10))}",
        f"stale_parent_backlog_count={len(stale_parent_backlog)}",
    ]
    for row in stale_parent_backlog[:10]:
        lines.append(
            "  stale_parent"
            f" candidate={_compact(row.get('candidate_id'))}"
            f" run={_compact(row.get('run_id'))}"
            f" at={row.get('current_status_at')}"
        )
    lines.append(
        "baseline_not_ready_queued_count="
        f"{len(baseline_not_ready)} oldest_age_hours={_oldest_age_hours(baseline_not_ready)}"
    )
    lines.append(f"active_scoring_count={len(active_scoring)}")
    lines.append(f"alert_stale_active_scoring_count={len(stale_active_scoring)}")
    for row in stale_active_scoring[:10]:
        lines.append(
            "  stale_active_scoring"
            f" candidate={_compact(row.get('candidate_id'), 24)}"
            f" run={_compact(row.get('run_id'))}"
            f" status={row.get('current_candidate_status')}"
            f" age_minutes={_age_minutes(row)}"
            f" at={row.get('current_status_at')}"
        )
    return lines


def check_promotions(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    candidates = fetch(
        "research_lab_candidate_evaluation_current",
        {
            "select": "candidate_id,run_id,current_candidate_status,current_score_bundle_id,current_status_at",
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    promotion_events = fetch(
        "research_lab_candidate_promotion_events",
        {
            "select": "candidate_id,event_type,promotion_status,source_score_bundle_id,created_at,event_doc",
            "order": "created_at.desc",
            "limit": "1000",
        },
    )
    terminal_promotion_events = {
        "below_threshold",
        "stale_parent_detected",
        "rebase_queued",
        "public_holdout_rejected",
        "promotion_disabled",
        "scoring_health_quarantined",
        "promotion_failed",
        "promotion_passed",
        "active_version_created",
        "unsupported_candidate_kind",
    }
    promotion_events_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in promotion_events:
        promotion_events_by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(row)
    scored_without_terminal_decision = [
        row
        for row in candidates
        if str(row.get("current_candidate_status") or "") == "scored"
        and not any(
            str(event.get("event_type") or "") in terminal_promotion_events
            for event in promotion_events_by_candidate.get(str(row.get("candidate_id") or ""), [])
        )
    ]
    reward_pending = [
        row
        for row in promotion_events
        if str(row.get("promotion_status") or "") == "reward_pending_uid"
        or str(row.get("event_type") or "") == "champion_reward_pending_uid"
    ]
    passed_candidates = {
        str(row.get("candidate_id") or "")
        for row in promotion_events
        if str(row.get("event_type") or "") == "promotion_passed"
    }
    activated_candidates = {
        str(row.get("candidate_id") or "")
        for row in promotion_events
        if str(row.get("event_type") or "") == "active_version_created"
    }
    passed_without_activation = sorted(passed_candidates - activated_candidates)
    lines = [
        f"promotion_decision_counts={_counts(promotion_events, 'event_type')}",
        f"promotion_status_counts={_counts(promotion_events, 'promotion_status')}",
        f"scored_candidates_without_terminal_promotion_decision={len(scored_without_terminal_decision)}",
    ]
    for row in scored_without_terminal_decision[:10]:
        lines.append(
            "  missing_promotion_decision"
            f" candidate={_compact(row.get('candidate_id'))}"
            f" run={_compact(row.get('run_id'))}"
            f" score_bundle={_compact(row.get('current_score_bundle_id'))}"
            f" at={row.get('current_status_at')}"
        )
    lines.append(f"alert_reward_pending_uid_events={len(reward_pending)}")
    for row in reward_pending[:10]:
        lines.append(
            "  reward_pending_uid"
            f" candidate={_compact(row.get('candidate_id'))}"
            f" at={row.get('created_at')}"
        )
    lines.append(
        f"alert_promotion_passed_without_active_version_created={len(passed_without_activation)}"
    )
    for candidate_id in passed_without_activation[:10]:
        lines.append(f"  promotion_passed_no_activation candidate={_compact(candidate_id)}")
    return lines


def check_score_bundles(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    score_bundles = fetch(
        "research_evaluation_score_bundle_current",
        {
            "select": (
                "score_bundle_id,run_id,parent_artifact_hash,candidate_artifact_hash,"
                "evaluation_epoch,created_at,score_bundle_doc"
            ),
            "order": "created_at.desc",
            "limit": "100",
        },
    )
    health_counts = Counter(_score_bundle_health_status(row) for row in score_bundles)
    public_holdout_decision_counts = Counter(_score_bundle_public_holdout_decision(row) for row in score_bundles)
    mode_counts = Counter(_score_bundle_reference_mode(row) for row in score_bundles)
    lines = [
        f"score_bundle_count_recent={len(score_bundles)}",
        f"score_bundle_health_counts={dict(health_counts)}",
        f"score_bundle_public_holdout_decision_counts={dict(public_holdout_decision_counts)}",
        f"score_bundle_reference_evaluation_mode_counts={dict(mode_counts)}",
    ]
    for row in score_bundles[:5]:
        doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), dict) else {}
        aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), dict) else {}
        lines.append(
            "  score_bundle"
            f" id={_compact(row.get('score_bundle_id'))}"
            f" run={_compact(row.get('run_id'))}"
            f" epoch={row.get('evaluation_epoch')}"
            f" mode={_score_bundle_reference_mode(row)}"
            f" mean_delta={aggregates.get('mean_delta')}"
            f" icp_count={aggregates.get('icp_count')}"
        )
    return lines


def check_costs(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    queue = fetch(
        "research_loop_run_queue_current",
        {"select": "run_id,current_queue_status", "order": "current_status_at.desc", "limit": "1000"},
    )
    candidates = fetch(
        "research_lab_candidate_evaluation_current",
        {"select": "candidate_id,current_candidate_status", "order": "current_status_at.desc", "limit": "1000"},
    )
    promotion_events = fetch(
        "research_lab_candidate_promotion_events",
        {"select": "candidate_id,event_type", "order": "created_at.desc", "limit": "1000"},
    )
    loop_events = fetch(
        "research_lab_auto_research_loop_events",
        {"select": "run_id,event_type,cost_ledger,created_at", "order": "created_at.desc", "limit": "1000"},
    )
    promotion_events_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in promotion_events:
        promotion_events_by_candidate.setdefault(str(row.get("candidate_id") or ""), []).append(row)
    cost_by_run: dict[str, float] = {}
    for row in loop_events:
        run_id = str(row.get("run_id") or "")
        if run_id and run_id not in cost_by_run:
            cost_by_run[run_id] = _event_cost_usd(row)
    total_cost_usd = round(sum(cost_by_run.values()), 6)
    admitted_runs = len({str(row.get("run_id") or "") for row in queue if str(row.get("run_id") or "")})
    completed_runs = len({str(row.get("run_id") or "") for row in queue if str(row.get("current_queue_status") or "") == "completed"})
    candidate_count = len(candidates)
    scored_candidate_count = len([row for row in candidates if str(row.get("current_candidate_status") or "") == "scored"])
    promotion_eligible_count = len(
        [
            row
            for row in candidates
            if str(row.get("current_candidate_status") or "") == "scored"
            and any(
                str(event.get("event_type") or "") in {"promotion_passed", "active_version_created", "below_threshold"}
                for event in promotion_events_by_candidate.get(str(row.get("candidate_id") or ""), [])
            )
        ]
    )
    return [
        "cost_visibility="
        f"total_event_cost_usd={total_cost_usd} "
        f"per_admitted_run={_safe_div(total_cost_usd, admitted_runs)} "
        f"per_completed_run={_safe_div(total_cost_usd, completed_runs)} "
        f"per_candidate={_safe_div(total_cost_usd, candidate_count)} "
        f"per_scored_candidate={_safe_div(total_cost_usd, scored_candidate_count)} "
        f"per_promotion_eligible_candidate={_safe_div(total_cost_usd, promotion_eligible_count)}"
    ]


def check_reimbursements(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    reimbursements = fetch(
        "research_reimbursement_award_current",
        {"select": "run_id,miner_hotkey,current_award_status,run_day,eligible_cost_microusd,target_reimbursement_microusd,current_status_at", "order": "current_status_at.desc", "limit": "50"},
    )
    return [f"reimbursement_status_counts={_counts(reimbursements, 'current_award_status')}"]


def check_arweave_anchors(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    # NOTE: the ``_current`` view exposes the events-side tx id as
    # ``current_arweave_tx_id`` (see scripts/38-research-lab-arweave-audit-anchors.sql);
    # selecting ``arweave_tx_id`` here previously 400'd the whole script.
    anchors = fetch(
        "research_lab_arweave_epoch_audit_anchor_current",
        {"select": "epoch,audit_kind,current_anchor_status,current_arweave_tx_id,current_status_at,payload_hash", "order": "current_status_at.desc", "limit": "10"},
    )
    lines = [f"arweave_anchor_status_counts={_counts(anchors, 'current_anchor_status')}"]
    for row in anchors[:5]:
        lines.append(
            "  arweave_anchor"
            f" epoch={row.get('epoch')}"
            f" kind={row.get('audit_kind')}"
            f" status={row.get('current_anchor_status')}"
            f" tx={_compact(row.get('current_arweave_tx_id'))}"
            f" at={row.get('current_status_at')}"
        )
    return lines


def check_scoring_cards_without_active_candidate(
    fetch: Callable[..., list[dict[str, Any]]] = _get,
) -> list[str]:
    cards = fetch(
        "research_lab_public_loop_card_current",
        {
            "select": "card_id,ticket_id,current_event_type,current_outcome_label,current_run_id,current_status_at",
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    candidates = fetch(
        "research_lab_candidate_evaluation_current",
        {"select": "candidate_id,run_id,current_candidate_status", "order": "current_status_at.desc", "limit": "1000"},
    )
    stale = _stale_scoring_cards(cards, candidates)
    lines = [f"alert_scoring_cards_without_active_candidate={len(stale)}"]
    for card in stale[:10]:
        lines.append(
            "  stale_scoring_card"
            f" card={_compact(card.get('card_id'), 24)}"
            f" ticket={_compact(card.get('ticket_id'))}"
            f" run={_compact(card.get('current_run_id'))}"
            f" at={card.get('current_status_at')}"
        )
    return lines


def check_daily_baseline_history(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    threshold = float(
        os.getenv("RESEARCH_LAB_HEALTH_BASELINE_JUMP_THRESHOLD", str(DEFAULT_BASELINE_JUMP_THRESHOLD))
    )
    rows = fetch(
        "research_lab_private_model_benchmark_current",
        {
            "select": "benchmark_bundle_id,benchmark_date,aggregate_score,created_at,current_benchmark_status",
            "order": "created_at.desc",
            "limit": "60",
        },
    )
    score_by_day = _latest_completed_score_by_day(rows)
    recent_days = sorted(score_by_day)[-7:]
    jumps = _baseline_day_jumps(score_by_day, threshold)
    duplicates = _duplicate_day_counts(rows)
    lines = [
        "baseline_recent_day_scores="
        + str({day: round(score_by_day[day], 4) for day in recent_days}),
        f"alert_baseline_day_jumps_above_{threshold}={len(jumps)}",
    ]
    for previous, current, delta in jumps[:10]:
        lines.append(f"  baseline_jump {previous}->{current} delta={delta}")
    lines.append(f"alert_duplicate_same_day_benchmark_bundles={len(duplicates)}")
    for day, count in list(duplicates.items())[:10]:
        lines.append(f"  duplicate_benchmark_day date={day} bundles={count}")
    return lines


def check_latest_baseline_icp_health(fetch: Callable[..., list[dict[str, Any]]] = _get) -> list[str]:
    rows = fetch(
        "research_lab_private_model_benchmark_current",
        {
            "select": "benchmark_bundle_id,benchmark_date,aggregate_score,created_at,current_benchmark_status,score_summary_doc",
            "order": "created_at.desc",
            "limit": "3",
        },
    )
    latest = None
    for row in rows:
        status = str(row.get("current_benchmark_status") or "")
        if status and status != "completed":
            continue
        if isinstance(row.get("score_summary_doc"), dict):
            latest = row
            break
    if latest is None:
        return ["latest_baseline_doc=missing"]
    doc = latest["score_summary_doc"]
    summaries = doc.get("per_icp_summaries") if isinstance(doc.get("per_icp_summaries"), list) else []
    provider_error_icps = _unresolved_provider_error_icps(summaries)
    zero_no_error_icps = _zero_company_no_error_icps(summaries)
    lines = [
        "latest_baseline"
        f" date={latest.get('benchmark_date')}"
        f" bundle={_compact(latest.get('benchmark_bundle_id'), 24)}"
        f" aggregate_score={latest.get('aggregate_score')}"
        f" icp_count={len(summaries)}",
        f"alert_baseline_zero_score_provider_error_icps={len(provider_error_icps)}",
    ]
    for ref in provider_error_icps[:10]:
        lines.append(f"  baseline_provider_error_icp ref={_compact(ref, 40)}")
    lines.append(f"alert_zero_company_icps_without_recorded_error={len(zero_no_error_icps)}")
    for ref in zero_no_error_icps[:10]:
        lines.append(f"  zero_company_no_error_icp ref={_compact(ref, 40)}")
    return lines


CHECKS: list[tuple[str, Callable[[], list[str]]]] = [
    ("queue_and_loops", check_queue_and_loops),
    ("candidates", check_candidates),
    ("promotions", check_promotions),
    ("score_bundles", check_score_bundles),
    ("costs", check_costs),
    ("reimbursements", check_reimbursements),
    ("arweave_anchors", check_arweave_anchors),
    ("scoring_cards_without_active_candidate", check_scoring_cards_without_active_candidate),
    ("daily_baseline_history", check_daily_baseline_history),
    ("latest_baseline_icp_health", check_latest_baseline_icp_health),
]


def run_health_checks(
    checks: list[tuple[str, Callable[[], list[str]]]] | None = None,
    printer: Callable[[str], None] = print,
) -> int:
    """Run every check, isolating failures. Returns nonzero if any check crashed."""
    selected = CHECKS if checks is None else checks
    failed: list[str] = []
    for name, check in selected:
        try:
            lines = check()
        except Exception as exc:  # noqa: BLE001 - each probe must not sink the others
            failed.append(name)
            printer(f"CHECK-FAILED {name}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        for line in lines:
            printer(line)
    if failed:
        printer(f"health_checks_failed={failed}")
        return 1
    return 0


def main() -> int:
    try:
        _supabase_config()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("Research Lab operator health")
    return run_health_checks()


if __name__ == "__main__":
    sys.exit(main())
