#!/usr/bin/env python3
"""Static and fixture checks for Research Lab public loop activity."""

from __future__ import annotations

import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.public_activity import (  # noqa: E402
    contains_secret_material,
    derive_public_loop_outcome,
    public_loop_api_item,
    public_loop_outcome_closes_ticket,
    public_loop_ticket_id_matches_lookup,
    sanitize_public_text,
    topic_group_items,
    topic_signature_hash,
    topic_tags_from_texts,
)


MIGRATION = ROOT / "scripts" / "40-research-lab-public-loop-activity.sql"
STATUS_LABEL_MIGRATION = ROOT / "scripts" / "55-research-lab-public-loop-status-labels.sql"
CANDIDATE_DIAGNOSTICS_MIGRATION = ROOT / "scripts" / "69-research-lab-candidate-generation-diagnostics.sql"
FORBIDDEN_GRANT_RE = re.compile(
    r"\bGRANT\b(?!\s+EXECUTE\b)[^;]*\b(?:anon|authenticated)\b",
    re.IGNORECASE | re.DOTALL,
)


def main() -> int:
    errors = verify_sql(MIGRATION.read_text(encoding="utf-8"))
    errors.extend(verify_status_label_sql(STATUS_LABEL_MIGRATION.read_text(encoding="utf-8")))
    errors.extend(verify_candidate_diagnostics_sql(CANDIDATE_DIAGNOSTICS_MIGRATION.read_text(encoding="utf-8")))
    errors.extend(verify_projection_fixtures())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab public loop activity verified: SQL safety, deterministic tags, outcomes, API shaping.")
    return 0


def verify_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    for marker in (
        "begin;",
        "commit;",
        "create table if not exists public.research_lab_public_loop_cards",
        "create table if not exists public.research_lab_public_loop_card_events",
        "create or replace view public.research_lab_public_loop_card_current",
        "before update or delete on public.research_lab_public_loop_cards",
        "before update or delete on public.research_lab_public_loop_card_events",
        "alter table public.research_lab_public_loop_cards enable row level security",
        "alter table public.research_lab_public_loop_card_events enable row level security",
        "grant select, insert on table public.research_lab_public_loop_cards to service_role",
        "grant select, insert on table public.research_lab_public_loop_card_events to service_role",
        "revoke all on table public.research_lab_public_loop_card_current from anon, authenticated",
        "event_ref",
        "topic_signature_hash",
        "candidate_patch_manifest",
        "private_model_manifest_doc",
        "hidden_icp",
        "judge_prompt",
        "proxy[_-]?url",
        "://[^/]+:[^/@]+@",
    ):
        if marker not in lowered:
            errors.append(f"migration missing marker: {marker}")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("migration must not grant privileges to anon/authenticated")
    if re.search(r"GRANT\s+[^;]*(UPDATE|DELETE)[^;]*TO\s+service_role", sql, re.IGNORECASE):
        errors.append("append-only tables must not grant UPDATE/DELETE to service_role")
    return errors


def verify_status_label_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    for marker in (
        "begin;",
        "commit;",
        "research_lab_public_loop_card_events_event_type_check",
        "research_lab_public_loop_card_events_outcome_label_check",
        "research_lab_public_loop_card_events_outcome_band_check",
        "waiting_for_baseline",
        "needs_rescore",
        "blocked_for_credit",
        "not_started",
        "'blocked'",
    ):
        if marker not in lowered:
            errors.append(f"status label migration missing marker: {marker}")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("status label migration must not grant privileges to anon/authenticated")
    return errors


def verify_candidate_diagnostics_sql(sql: str) -> list[str]:
    errors: list[str] = []
    lowered = sql.lower()
    for marker in (
        "begin;",
        "commit;",
        "no_buildable_candidate",
        "candidate_patch_parse_failed",
        "candidate_patch_empty_or_noop",
        "candidate_patch_test_failed",
        "candidate_artifact_missing",
        "candidate_repair_exhausted",
        "candidate_generation_fallback_requested",
        "candidate_generation_fallback_drafted",
        "candidate_generation_fallback_failed",
        "allocator_decision",
        "awaiting_payment",
    ):
        if marker not in lowered:
            errors.append(f"candidate diagnostics migration missing marker: {marker}")
    if FORBIDDEN_GRANT_RE.search(sql):
        errors.append("candidate diagnostics migration must not grant privileges to anon/authenticated")
    return errors


def verify_projection_fixtures() -> list[str]:
    errors: list[str] = []
    text = "Improve evidence freshness and reduce overbroad matches in role targeting."
    tags_a = topic_tags_from_texts("generalist", text)
    tags_b = topic_tags_from_texts("generalist", text)
    if tags_a != tags_b:
        errors.append("topic tags must be deterministic")
    for expected in ("evidence_freshness", "overbroad_matches", "role_targeting"):
        if expected not in tags_a:
            errors.append(f"missing expected topic tag: {expected}")
    if topic_signature_hash("generalist", tags_a) != topic_signature_hash("generalist", tags_b):
        errors.append("topic signature must be deterministic")
    if sanitize_public_text("sk-or-v1-test") != "[redacted]":
        errors.append("secret-like text must be redacted")
    if not contains_secret_material("candidate_patch_manifest"):
        errors.append("candidate patch marker must be detected")

    scored = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[{"run_id": "r", "current_queue_status": "completed", "current_status_at": "2026-01-01T00:01:00+00:00"}],
        receipt_rows=[{"receipt_id": "x", "run_id": "r", "current_receipt_status": "completed", "current_status_at": "2026-01-01T00:02:00+00:00"}],
        candidate_rows=[
            {
                "candidate_id": "candidate:1",
                "run_id": "r",
                "receipt_id": "x",
                "current_candidate_status": "scored",
                "candidate_artifact_hash": "sha256:a",
                "redacted_public_summary": "Better scoring calibration",
                "current_status_at": "2026-01-01T00:03:00+00:00",
            }
        ],
        score_bundle_rows=[
            {
                "candidate_artifact_hash": "sha256:a",
                "score_bundle_doc": {"aggregates": {"mean_delta": 2.5, "delta_lcb": 0.5}},
                "current_status_at": "2026-01-01T00:04:00+00:00",
            }
        ],
        promotion_event_rows=[],
        improvement_threshold_points=1.0,
    )
    if scored.outcome_label != "scored_promising" or scored.outcome_band != "passed_threshold":
        errors.append("positive scored fixture should be promising/passed_threshold")
    if not public_loop_outcome_closes_ticket(
        {"current_outcome_label": scored.outcome_label, "current_outcome_band": scored.outcome_band}
    ):
        errors.append("scored public loop should close open-ticket cap")
    failed = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[],
        receipt_rows=[],
        candidate_rows=[{"current_candidate_status": "failed", "current_status_at": "2026-01-01T00:01:00+00:00"}],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if failed.outcome_label != "failed" or failed.outcome_band != "failed":
        errors.append("failed fixture should be failed/failed")
    if not public_loop_outcome_closes_ticket(
        {"current_outcome_label": failed.outcome_label, "current_outcome_band": failed.outcome_band}
    ):
        errors.append("failed public loop should close open-ticket cap")
    blocked = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[
            {
                "run_id": "r",
                "current_queue_status": "paused",
                "current_reason": "blocked_for_credit",
                "current_status_at": "2026-01-01T00:01:00+00:00",
            }
        ],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if blocked.outcome_label != "blocked_for_credit" or blocked.outcome_band != "blocked":
        errors.append("credit-blocked fixture should be blocked_for_credit/blocked")
    waiting = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[],
        receipt_rows=[],
        candidate_rows=[
            {
                "current_candidate_status": "queued",
                "current_reason": "baseline_not_ready",
                "current_status_at": "2026-01-01T00:01:00+00:00",
            }
        ],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if waiting.outcome_label != "waiting_for_baseline" or waiting.outcome_band != "pending":
        errors.append("baseline wait fixture should be waiting_for_baseline/pending")
    needs_rescore = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[],
        receipt_rows=[],
        candidate_rows=[
            {
                "current_candidate_status": "rejected",
                "current_reason": "stale_parent_needs_rescore",
                "current_status_at": "2026-01-01T00:01:00+00:00",
            }
        ],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if needs_rescore.outcome_label != "needs_rescore" or needs_rescore.outcome_band != "blocked":
        errors.append("stale-parent fixture should be needs_rescore/blocked")
    recovered_stale_parent = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-07-05T16:13:28+00:00"},
        queue_rows=[
            {
                "run_id": "new-run",
                "current_queue_status": "completed",
                "current_status_at": "2026-07-06T05:35:34+00:00",
            }
        ],
        receipt_rows=[],
        candidate_rows=[
            {
                "candidate_id": "candidate:old",
                "run_id": "old-run",
                "current_candidate_status": "rejected",
                "current_reason": "stale_parent_needs_rescore",
                "current_status_at": "2026-07-06T20:34:54+00:00",
            },
            {
                "candidate_id": "candidate:new",
                "run_id": "new-run",
                "current_candidate_status": "scored",
                "candidate_artifact_hash": "sha256:new",
                "redacted_public_summary": "Recovered run scored cleanly",
                "current_status_at": "2026-07-06T05:53:55+00:00",
            },
        ],
        score_bundle_rows=[
            {
                "candidate_artifact_hash": "sha256:new",
                "score_bundle_doc": {"aggregates": {"mean_delta": 2.4, "delta_lcb": 1.1}},
                "current_status_at": "2026-07-06T05:53:56+00:00",
            }
        ],
        promotion_event_rows=[
            {
                "event_type": "stale_parent_detected",
                "promotion_status": "rebase_required",
                "created_at": "2026-07-06T21:34:54+00:00",
            }
        ],
    )
    if recovered_stale_parent.outcome_label != "scored_promising":
        errors.append("recovered stale-parent fixture should remain scored_promising")
    if recovered_stale_parent.last_activity_at == "2026-07-06T20:34:54+00:00":
        errors.append("recovered stale-parent tombstone must not drive public last_activity_at")
    if recovered_stale_parent.last_activity_at != "2026-07-06T05:53:56+00:00":
        errors.append(
            "recovered stale-parent fixture should use scored replacement activity, "
            f"got {recovered_stale_parent.last_activity_at}"
        )
    not_started = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "opened", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if not_started.outcome_label != "awaiting_payment" or not_started.outcome_band != "pending":
        errors.append("ticket-without-run fixture should be awaiting_payment/pending")
    no_candidate = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[
            {
                "run_id": "r",
                "current_queue_status": "completed",
                "current_status_at": "2026-01-01T00:01:00+00:00",
            }
        ],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
    )
    if no_candidate.outcome_label != "completed_no_candidate":
        errors.append("completed run without candidate should be completed_no_candidate")
    if not public_loop_outcome_closes_ticket(
        {"current_outcome_label": no_candidate.outcome_label, "current_outcome_band": no_candidate.outcome_band}
    ):
        errors.append("completed_no_candidate public loop should close open-ticket cap")
    no_buildable = derive_public_loop_outcome(
        ticket={"ticket_id": "t", "current_ticket_status": "running", "created_at": "2026-01-01T00:00:00+00:00"},
        queue_rows=[
            {
                "run_id": "r",
                "current_queue_status": "failed",
                "current_reason": "no_valid_image_build_finalists",
                "current_status_at": "2026-01-01T00:01:00+00:00",
            }
        ],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
        auto_loop_event_rows=[
            {
                "run_id": "r",
                "event_type": "source_inspection_failed",
                "seq": 1,
                "event_doc": {
                    "stage": "source_inspection_call_failed",
                    "error": "OpenRouter candidate generation failed: HTTP 404: No endpoints found for requested parameters",
                },
            }
        ],
    )
    if no_buildable.outcome_label != "no_buildable_candidate" or no_buildable.outcome_band != "failed":
        errors.append("failed zero-candidate fixture should be no_buildable_candidate/failed")
    failure_doc = no_buildable.event_doc.get("candidate_generation_failure", {})
    if failure_doc.get("primary_reason") != "provider_route_unavailable":
        errors.append("no_buildable_candidate fixture should classify provider_route_unavailable")
    if no_buildable.event_doc.get("public_status_label") != "No buildable candidate":
        errors.append("no_buildable_candidate fixture should carry display label")
    if not public_loop_outcome_closes_ticket(
        {"current_outcome_label": no_buildable.outcome_label, "current_outcome_band": no_buildable.outcome_band}
    ):
        errors.append("no_buildable_candidate public loop should close open-ticket cap")
    if public_loop_outcome_closes_ticket({"current_outcome_label": "running", "current_outcome_band": "pending"}):
        errors.append("running public loop must remain open for ticket cap")
    if not public_loop_ticket_id_matches_lookup(
        "49a0d110-1234-4567-89ab-123456789abc",
        "49a0d110",
    ):
        errors.append("short public loop ticket prefix must match full ticket id")
    if public_loop_ticket_id_matches_lookup(
        "49a0d110-1234-4567-89ab-123456789abc",
        "not-a-uuid-prefix",
    ):
        errors.append("invalid public loop ticket lookup must not match")

    row = {
        "card_id": "public_loop_card:00000000-0000-0000-0000-000000000000",
        "ticket_id": "00000000-0000-0000-0000-000000000000",
        "miner_hotkey": "5Hotkey",
        "research_area": "generalist",
        "research_focus_summary": "Improve source routing",
        "current_topic_tags": ["source_routing"],
        "current_topic_signature_hash": "sha256:" + "0" * 64,
        "current_outcome_label": "running",
        "current_outcome_band": "pending",
        "current_candidate_count": 0,
        "current_scored_candidate_count": 0,
    }
    item = public_loop_api_item(row, similar_recent_loop_count=3)
    if item["miner_hotkey"] != "5Hotkey" or item["similar_recent_loop_count"] != 3:
        errors.append("API item must expose full miner hotkey and similar count")
    groups = topic_group_items([row])
    if len(groups) != 1 or groups[0]["running"] != 1:
        errors.append("topic group fixture count mismatch")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
