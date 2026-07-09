from __future__ import annotations

from datetime import datetime, timezone

from gateway.research_lab import candidate_generation_report as report_mod
from gateway.research_lab.candidate_generation_report import (
    build_candidate_generation_failure_report,
)


def _event(run_id: str, event_type: str, event_doc: dict | None = None) -> dict:
    return {
        "run_id": run_id,
        "ticket_id": f"ticket-{run_id}",
        "event_type": event_type,
        "event_doc": event_doc or {},
    }


def test_candidate_generation_failure_report_aggregates_public_and_fallback_rows():
    loop_events = [
        _event("run-a", "loop_direction_planned"),
        _event(
            "run-a",
            "candidate_generation_fallback_requested",
            {
                "ranked_path_fallback_attempted": True,
                "previous_path_id": "path-a",
                "next_path_id": "path-b",
                "fallback_index": 1,
            },
        ),
        _event("run-a", "candidate_selected"),
        _event(
            "run-b",
            "no_viable_patch",
            {
                "reason": "required file is not present in editable_files",
                "terminal_after_ranked_paths_exhausted": True,
            },
        ),
        _event("run-b", "loop_failed", {"run_summary": {"selected_candidate_count": 0}}),
    ]
    public_cards = [
        {
            "run_id": "run-a",
            "ticket_id": "ticket-run-a",
            "current_outcome_label": "no_buildable_candidate",
            "current_event_type": "candidate_selected",
            "current_candidate_count": 0,
            "current_event_doc": {
                "candidate_generation_failure": {
                    "primary_reason": "binding_plan_source_missing",
                    "latest_stage": "no_viable_patch",
                }
            },
        }
    ]

    report = build_candidate_generation_failure_report(
        loop_event_rows=loop_events,
        public_card_rows=public_cards,
        days=7,
        generated_at=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )

    assert report["total_no_buildable_candidate"] == 2
    assert report["counts"]["by_primary_reason"]["binding_plan_source_missing"] == 2
    assert report["counts"]["ranked_path_fallback"] == {
        "attempted": 1,
        "succeeded": 1,
        "exhausted": 1,
    }
    assert {row["failure_category"] for row in report["sample_runs"]} >= {"source_path_failure"}


def test_candidate_generation_failure_report_ignores_terminal_runs_with_selected_candidates():
    report = build_candidate_generation_failure_report(
        loop_event_rows=[
            _event(
                "run-selected",
                "loop_failed",
                {
                    "run_summary": {
                        "selected_candidate_count": 1,
                        "stop_reason": "max_iterations",
                    }
                },
            )
        ],
        public_card_rows=[],
        days=7,
        generated_at=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )

    assert report["total_no_buildable_candidate"] == 0
    assert report["sample_runs"] == []


async def test_fetch_candidate_generation_failure_report_uses_paginated_select(monkeypatch):
    calls = []

    async def fake_select_all(table, **kwargs):
        calls.append((table, kwargs))
        return []

    monkeypatch.setattr(report_mod, "select_all", fake_select_all)

    report = await report_mod.fetch_candidate_generation_failure_report(7)

    assert report["schema_version"] == "1.0"
    assert [table for table, _kwargs in calls] == [
        "research_lab_auto_research_loop_events",
        "research_lab_public_loop_card_current",
    ]
    assert all(kwargs["max_rows"] == 50000 for _table, kwargs in calls)
    assert all(kwargs["allow_partial"] is True for _table, kwargs in calls)
