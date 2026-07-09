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
        if table == "research_lab_public_loop_card_current":
            return [
                {
                    "current_run_id": "run-a",
                    "ticket_id": "ticket-run-a",
                    "current_outcome_label": "no_buildable_candidate",
                    "current_event_type": "loop_failed",
                    "current_candidate_count": 0,
                    "current_event_doc": {
                        "candidate_generation_failure": {
                            "primary_reason": "binding_plan_source_missing",
                            "latest_stage": "loop_failed",
                        }
                    },
                }
            ]
        if table == "research_lab_auto_research_loop_events" and any(
            flt == ("event_type", "in", ["loop_failed", "no_buildable_candidate"])
            for flt in kwargs.get("filters", ())
        ):
            return [{"run_id": "run-b", "event_type": "loop_failed", "event_doc": {"run_summary": {}}}]
        if table == "research_lab_auto_research_loop_events" and any(
            len(flt) == 3 and flt[0] == "run_id" and flt[1] == "in"
            for flt in kwargs.get("filters", ())
        ):
            return [
                {"run_id": "run-a", "event_type": "loop_failed", "event_doc": {"run_summary": {}}},
                {"run_id": "run-b", "event_type": "loop_failed", "event_doc": {"run_summary": {}}},
            ]
        return []

    monkeypatch.setattr(report_mod, "select_all", fake_select_all)

    report = await report_mod.fetch_candidate_generation_failure_report(7)

    assert report["schema_version"] == "1.0"
    assert report["partial"] is False
    assert [table for table, _kwargs in calls] == [
        "research_lab_public_loop_card_current",
        "research_lab_auto_research_loop_events",
        "research_lab_auto_research_loop_events",
    ]
    assert calls[0][1]["columns"] == report_mod._PUBLIC_CARD_COLUMNS
    assert ("current_outcome_label", "no_buildable_candidate") in calls[0][1]["filters"]
    assert calls[1][1]["columns"] == report_mod._LOOP_EVENT_COLUMNS
    assert ("event_type", "in", ["loop_failed", "no_buildable_candidate"]) in calls[1][1]["filters"]
    assert calls[2][1]["columns"] == report_mod._LOOP_EVENT_COLUMNS
    assert any(
        len(flt) == 3 and flt[0] == "run_id" and flt[1] == "in" and set(flt[2]) == {"run-a", "run-b"}
        for flt in calls[2][1]["filters"]
    )
    assert all(kwargs["allow_partial"] is True for _table, kwargs in calls)


async def test_fetch_candidate_generation_failure_report_returns_partial_on_event_fetch_failure(monkeypatch):
    calls = []

    async def fake_select_all(table, **kwargs):
        calls.append((table, kwargs))
        if table == "research_lab_public_loop_card_current":
            return [
                {
                    "current_run_id": "run-a",
                    "ticket_id": "ticket-run-a",
                    "current_outcome_label": "no_buildable_candidate",
                    "current_event_type": "loop_failed",
                    "current_candidate_count": 0,
                    "current_event_doc": {
                        "candidate_generation_failure": {
                            "primary_reason": "no_viable_patch",
                            "latest_stage": "loop_failed",
                        }
                    },
                }
            ]
        raise TimeoutError("synthetic event fetch timeout")

    monkeypatch.setattr(report_mod, "select_all", fake_select_all)

    report = await report_mod.fetch_candidate_generation_failure_report(1)

    assert report["partial"] is True
    assert report["partial_reason"] == "optimized_event_fetch_failed:TimeoutError"
    assert report["total_no_buildable_candidate"] == 1
    assert report["counts"]["by_primary_reason"] == {"no_viable_patch": 1}
    assert [table for table, _kwargs in calls] == [
        "research_lab_public_loop_card_current",
        "research_lab_auto_research_loop_events",
    ]
