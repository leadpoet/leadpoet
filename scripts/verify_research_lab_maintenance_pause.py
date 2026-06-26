#!/usr/bin/env python3
"""Verify Research Lab maintenance pause/resume implementation markers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = {
        "sql": (
            ROOT / "scripts" / "44-research-lab-maintenance-pause.sql",
            {
                "research_lab_gateway_control_events",
                "research_lab_gateway_control_current",
                "'paused'",
                "'checkpoint_saved'",
                "'loop_paused'",
                "'loop_resumed'",
                "current_queue_status IN ('queued', 'started', 'paused')",
                "is_existing_run_requeue BOOLEAN",
                "q.run_id <> NEW.run_id",
            },
        ),
        "api": (
            ROOT / "gateway" / "research_lab" / "api.py",
            {
                "_require_autoresearch_not_paused",
                "research_lab_maintenance_paused",
                "ACTIVE_AUTORESEARCH_QUEUE_STATUSES = {\"queued\", \"started\", \"paused\"}",
            },
        ),
        "worker": (
            ROOT / "gateway" / "research_lab" / "worker.py",
            {
                "is_autoresearch_maintenance_paused",
                "autoresearch_queue_capacity_doc",
                "_mark_paused",
                "latest_auto_research_checkpoint",
                "find_queued_receipt_for_run",
                "event_type=\"paused\"",
                "resume_state=resume_state",
            },
        ),
        "maintenance": (
            ROOT / "gateway" / "research_lab" / "maintenance.py",
            {
                "autoresearch_queue_capacity_doc",
                "research_lab_maintenance_resume_capacity_limited",
                "requeue_paused_autoresearch_runs",
            },
        ),
        "engine": (
            ROOT / "gateway" / "research_lab" / "loop_engine.py",
            {
                "checkpoint_saved",
                "loop_paused",
                "loop_resumed",
                "resume_state",
                "should_pause",
                "pause_after_minimum_runtime",
                "_selected_candidates_from_checkpoint",
            },
        ),
        "admin": (
            ROOT / "gateway" / "research_lab" / "admin.py",
            {
                "pause-autoresearch",
                "resume-autoresearch",
                "wait-drained",
                "requeue_paused_autoresearch_runs",
            },
        ),
    }
    missing: list[str] = []
    for label, (path, markers) in checks.items():
        text = path.read_text(encoding="utf-8")
        missing.extend(f"{label}:{marker}" for marker in sorted(markers) if marker not in text)
    if missing:
        for marker in missing:
            print(f"missing maintenance pause marker: {marker}")
        return 1
    print("Research Lab maintenance pause/resume verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
