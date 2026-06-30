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
        {"select": "candidate_id,run_id,current_candidate_status,current_reason,current_status_at", "order": "current_status_at.desc", "limit": "1000"},
    )
    candidate_events = _get(
        "research_lab_candidate_evaluation_events",
        {"select": "candidate_id,reason,event_doc,created_at", "event_type": "eq.failed", "order": "created_at.desc", "limit": "100"},
    )
    score_bundles = _get(
        "research_evaluation_score_bundle_current",
        {"select": "score_bundle_id,run_id,evaluation_epoch,created_at,score_bundle_doc", "order": "created_at.desc", "limit": "20"},
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

    print("Research Lab operator health")
    print(f"queue_status_counts={_counts(queue, 'current_queue_status')}")
    print(f"loop_status_counts={_counts(loops, 'current_loop_status')}")
    print(f"terminal_queue_nonterminal_loop_mismatches={len(mismatches)}")
    for run_id, queue_status, loop_status, event_type in mismatches[:10]:
        print(f"  mismatch run={_compact(run_id)} queue={queue_status} loop={loop_status} loop_event={event_type}")
    print(f"candidate_status_counts={_counts(candidates, 'current_candidate_status')}")
    print(f"candidate_failure_class_counts={dict(Counter(_failure_class(row) for row in candidate_events))}")
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
