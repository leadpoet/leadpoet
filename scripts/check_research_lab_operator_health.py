#!/usr/bin/env python3
"""Read-only Research Lab operator health snapshot.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment.
Prints only aggregate counts and compact refs; no secrets or raw private payloads.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any
from urllib import parse, request


def _env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise SystemExit(f"missing required env var: {name}")
    return value


SUPABASE_URL = _env("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY")
BASE_URL = f"{SUPABASE_URL}/rest/v1/"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Accept": "application/json",
}


def _get(table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = BASE_URL + table + "?" + parse.urlencode(params, doseq=True, safe="*,.:->")
    req = request.Request(url, headers=HEADERS)
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
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
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


def main() -> int:
    queue = _get(
        "research_loop_run_queue_current",
        {
            "select": "run_id,ticket_id,current_queue_status,current_reason,current_status_at",
            "order": "current_status_at.desc",
            "limit": "1000",
        },
    )
    loops = _get(
        "research_lab_auto_research_loop_current",
        {"select": "run_id,current_loop_status,current_event_type,current_status_at", "order": "current_status_at.desc", "limit": "1000"},
    )
    candidates = _get(
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
    candidate_events = _get(
        "research_lab_candidate_evaluation_events",
        {"select": "candidate_id,reason,event_doc,created_at", "event_type": "eq.failed", "order": "created_at.desc", "limit": "100"},
    )
    score_bundles = _get(
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
    promotion_events = _get(
        "research_lab_candidate_promotion_events",
        {
            "select": "candidate_id,event_type,promotion_status,source_score_bundle_id,created_at,event_doc",
            "order": "created_at.desc",
            "limit": "1000",
        },
    )
    loop_events = _get(
        "research_lab_auto_research_loop_events",
        {"select": "run_id,event_type,cost_ledger,created_at", "order": "created_at.desc", "limit": "1000"},
    )
    reimbursements = _get(
        "research_reimbursement_award_current",
        {"select": "run_id,miner_hotkey,current_award_status,run_day,eligible_cost_microusd,target_reimbursement_microusd,current_status_at", "order": "current_status_at.desc", "limit": "50"},
    )
    anchors = _get(
        "research_lab_arweave_epoch_audit_anchor_current",
        {"select": "epoch,audit_kind,current_anchor_status,arweave_tx_id,current_status_at,payload_hash", "order": "current_status_at.desc", "limit": "10"},
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
    credit_blocked = [
        row
        for row in queue
        if str(row.get("current_queue_status") or "") == "paused"
        and str(row.get("current_reason") or "") == "blocked_for_credit"
    ]
    active_scoring = [
        row
        for row in candidates
        if str(row.get("current_candidate_status") or "") in {"assigned", "evaluating"}
    ]
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
    lane_counts: Counter[str] = Counter()
    target_file_counts: Counter[str] = Counter()
    for row in candidates:
        lane, files = _candidate_lane_and_files(row.get("candidate_patch_manifest"))
        lane_counts[lane] += 1
        for path in files:
            target_file_counts[path] += 1
    health_counts = Counter(_score_bundle_health_status(row) for row in score_bundles)
    public_holdout_decision_counts = Counter(_score_bundle_public_holdout_decision(row) for row in score_bundles)
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

    print("Research Lab operator health")
    print(f"queue_status_counts={_counts(queue, 'current_queue_status')}")
    print(f"loop_status_counts={_counts(loops, 'current_loop_status')}")
    print(f"terminal_queue_nonterminal_loop_mismatches={len(mismatches)}")
    for run_id, queue_status, loop_status, event_type in mismatches[:10]:
        print(f"  mismatch run={_compact(run_id)} queue={queue_status} loop={loop_status} loop_event={event_type}")
    print(f"candidate_status_counts={_counts(candidates, 'current_candidate_status')}")
    print(f"candidate_failure_class_counts={dict(Counter(_failure_class(row) for row in candidate_events))}")
    print(f"promotion_decision_counts={_counts(promotion_events, 'event_type')}")
    print(f"promotion_status_counts={_counts(promotion_events, 'promotion_status')}")
    print(f"scored_candidates_without_terminal_promotion_decision={len(scored_without_terminal_decision)}")
    for row in scored_without_terminal_decision[:10]:
        print(
            "  missing_promotion_decision",
            f"candidate={_compact(row.get('candidate_id'))}",
            f"run={_compact(row.get('run_id'))}",
            f"score_bundle={_compact(row.get('current_score_bundle_id'))}",
            f"at={row.get('current_status_at')}",
        )
    print(f"candidate_lane_counts={dict(lane_counts.most_common(10))}")
    print(f"candidate_target_file_counts={dict(target_file_counts.most_common(10))}")
    print(f"stale_parent_backlog_count={len(stale_parent_backlog)}")
    for row in stale_parent_backlog[:10]:
        print(
            "  stale_parent",
            f"candidate={_compact(row.get('candidate_id'))}",
            f"run={_compact(row.get('run_id'))}",
            f"at={row.get('current_status_at')}",
        )
    print(
        "baseline_not_ready_queued_count="
        f"{len(baseline_not_ready)} oldest_age_hours={_oldest_age_hours(baseline_not_ready)}"
    )
    print(f"credit_blocked_paused_count={len(credit_blocked)}")
    for row in credit_blocked[:10]:
        print(
            "  credit_blocked",
            f"run={_compact(row.get('run_id'))}",
            f"ticket={_compact(row.get('ticket_id'))}",
            f"at={row.get('current_status_at')}",
        )
    print(f"active_scoring_count={len(active_scoring)}")
    print(f"score_bundle_count_recent={len(score_bundles)}")
    print(f"score_bundle_health_counts={dict(health_counts)}")
    print(f"score_bundle_public_holdout_decision_counts={dict(public_holdout_decision_counts)}")
    print(
        "cost_visibility="
        f"total_event_cost_usd={total_cost_usd} "
        f"per_admitted_run={_safe_div(total_cost_usd, admitted_runs)} "
        f"per_completed_run={_safe_div(total_cost_usd, completed_runs)} "
        f"per_candidate={_safe_div(total_cost_usd, candidate_count)} "
        f"per_scored_candidate={_safe_div(total_cost_usd, scored_candidate_count)} "
        f"per_promotion_eligible_candidate={_safe_div(total_cost_usd, promotion_eligible_count)}"
    )
    for row in score_bundles[:5]:
        doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), dict) else {}
        aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), dict) else {}
        print(
            "  score_bundle",
            f"id={_compact(row.get('score_bundle_id'))}",
            f"run={_compact(row.get('run_id'))}",
            f"epoch={row.get('evaluation_epoch')}",
            f"mean_delta={aggregates.get('mean_delta')}",
            f"icp_count={aggregates.get('icp_count')}",
        )
    print(f"reimbursement_status_counts={_counts(reimbursements, 'current_award_status')}")
    print(f"arweave_anchor_status_counts={_counts(anchors, 'current_anchor_status')}")
    for row in anchors[:5]:
        print(
            "  arweave_anchor",
            f"epoch={row.get('epoch')}",
            f"kind={row.get('audit_kind')}",
            f"status={row.get('current_anchor_status')}",
            f"tx={_compact(row.get('arweave_tx_id'))}",
            f"at={row.get('current_status_at')}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
