from __future__ import annotations

from datetime import datetime, timezone

from gateway.research_lab import candidate_generation_report as report_mod
from gateway.research_lab.candidate_diagnostics import (
    build_candidate_generation_failure_summary,
    canonical_loop_event_order,
)
from gateway.research_lab.candidate_generation_report import (
    build_candidate_generation_failure_report,
)


def _event(
    run_id: str,
    event_type: str,
    event_doc: dict | None = None,
    *,
    seq: int | None = None,
    created_at: str | None = None,
) -> dict:
    row = {
        "run_id": run_id,
        "ticket_id": f"ticket-{run_id}",
        "event_type": event_type,
        "event_doc": event_doc or {},
    }
    if seq is not None:
        row["seq"] = seq
    if created_at is not None:
        row["created_at"] = created_at
    return row


def test_diagnostics_ignore_earlier_unread_iteration_after_later_source_read():
    rows = [
        _event("run-ordered", "code_edit_validation_failed", {
            "iteration": 1,
            "error": "code_edit_no_source_files_read",
        }, seq=1),
        _event("run-ordered", "source_inspection_resolved", {
            "iteration": 2,
            "read_file_count": 3,
            "read_files": ["sourcing_model/discovery.py"],
        }, seq=2),
        _event("run-ordered", "no_viable_patch", {
            "iteration": 2,
            "reason": "No existing test file appears in runtime_source_context.editable_files.",
        }, seq=3),
        _event("run-ordered", "loop_failed", {"stop_reason": "binding_plan_unimplementable"}, seq=4),
    ]
    summary = build_candidate_generation_failure_summary(list(reversed(rows)))
    assert summary["primary_reason"] == "binding_plan_source_missing"
    assert summary["latest_event_type"] == "loop_failed"


def test_diagnostics_preserve_genuine_terminal_unread_source():
    summary = build_candidate_generation_failure_summary(
        [
            _event("run-unread", "code_edit_validation_failed", {
                "iteration": 1,
                "error": "code_edit_no_source_files_read",
                "stage": "source_inspection_exhausted_without_read",
            }, seq=2),
            _event("run-unread", "loop_failed", {}, seq=3),
        ]
    )
    assert summary["primary_reason"] == "source_context_empty_or_unread"


def test_diagnostics_source_failure_is_not_terminal_after_later_read():
    summary = build_candidate_generation_failure_summary(
        [
            _event("run-recovered", "source_inspection_failed", {"iteration": 1}, seq=1),
            _event("run-recovered", "source_inspection_resolved", {
                "iteration": 2,
                "read_file_count": 1,
                "read_files": ["sourcing_model/discovery.py"],
            }, seq=2),
            _event("run-recovered", "no_viable_patch", {
                "iteration": 2,
                "failure_class": "no_safe_patch",
                "reason": "no safe patch after inspection",
            }, seq=3),
            _event("run-recovered", "loop_failed", {}, seq=4),
        ]
    )
    assert summary["primary_reason"] == "no_viable_patch"


def test_diagnostics_ignore_unread_from_nonterminal_iteration_without_later_read():
    summary = build_candidate_generation_failure_summary(
        [
            _event("run-later-failure", "code_edit_validation_failed", {
                "iteration": 1,
                "error": "code_edit_no_source_files_read",
            }, seq=1),
            _event("run-later-failure", "candidate_patch_parse_failed", {
                "iteration": 2,
                "error": "invalid draft JSON",
            }, seq=2),
            _event("run-later-failure", "loop_failed", {}, seq=3),
        ]
    )
    assert summary["primary_reason"] == "candidate_patch_parse_failed"


def test_diagnostics_earlier_structured_refusal_does_not_override_later_build_failure():
    summary = build_candidate_generation_failure_summary(
        [
            _event("run-later-build", "no_viable_patch", {
                "iteration": 1,
                "failure_class": "binding_plan_unimplementable",
                "reason": "required symbol is absent",
            }, seq=1),
            _event("run-later-build", "candidate_image_build_failed", {
                "iteration": 2,
                "error": "build failed",
            }, seq=2),
            _event("run-later-build", "loop_failed", {}, seq=3),
        ]
    )
    assert summary["primary_reason"] == "candidate_image_build_failed"


def test_canonical_event_order_handles_mixed_and_missing_sequence_values():
    rows = [
        _event("run-mixed", "missing-late", seq=None, created_at="2026-07-10T00:00:03Z"),
        _event("run-mixed", "second", seq=2, created_at="2026-07-10T00:00:02Z"),
        _event("run-mixed", "missing-early", seq=None, created_at="2026-07-10T00:00:01Z"),
        _event("run-mixed", "first", seq=1, created_at="2026-07-10T00:00:04Z"),
    ]
    assert [row["event_type"] for row in canonical_loop_event_order(rows)] == [
        "first",
        "second",
        "missing-early",
        "missing-late",
    ]


def test_structured_binding_failure_precedes_stale_unread_warning():
    summary = build_candidate_generation_failure_summary(
        [
            _event("run-structured", "code_edit_validation_failed", {
                "iteration": 1,
                "error": "code_edit_no_source_files_read",
            }, seq=1),
            _event("run-structured", "source_inspection_resolved", {
                "iteration": 2,
                "read_file_count": 1,
            }, seq=2),
            _event("run-structured", "no_viable_patch", {
                "iteration": 2,
                "failure_class": "binding_plan_unimplementable",
                "reason": "required validation path is unavailable",
            }, seq=3),
        ]
    )
    assert summary["primary_reason"] == "binding_plan_source_missing"


def test_candidate_generation_failure_report_aggregates_public_and_terminal_rows():
    loop_events = [
        _event("run-a", "loop_direction_planned"),
        _event("run-a", "candidate_selected"),
        _event(
            "run-b",
            "no_viable_patch",
            {
                "reason": "required file is not present in editable_files",
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
    assert "ranked_path_fallback" not in report["counts"]
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


def test_report_recomputes_stale_public_reason_when_terminal_events_are_available():
    report = build_candidate_generation_failure_report(
        loop_event_rows=[
            _event("run-stale-card", "code_edit_validation_failed", {
                "iteration": 1,
                "error": "code_edit_no_source_files_read",
            }, seq=1),
            _event("run-stale-card", "source_inspection_resolved", {
                "iteration": 2,
                "read_file_count": 1,
            }, seq=2),
            _event("run-stale-card", "no_viable_patch", {
                "iteration": 2,
                "failure_class": "binding_plan_unimplementable",
                "reason": "validation file unavailable",
            }, seq=3),
            _event("run-stale-card", "loop_failed", {
                "run_summary": {"selected_candidate_count": 0},
            }, seq=4),
        ],
        public_card_rows=[
            {
                "current_run_id": "run-stale-card",
                "ticket_id": "ticket-run-stale-card",
                "current_outcome_label": "no_buildable_candidate",
                "current_candidate_count": 0,
                "current_event_doc": {
                    "candidate_generation_failure": {
                        "primary_reason": "source_context_empty_or_unread",
                    }
                },
            }
        ],
        days=7,
        generated_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert report["total_no_buildable_candidate"] == 1
    assert report["sample_runs"][0]["primary_reason"] == "binding_plan_source_missing"


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
    assert calls[2][1]["order_by"] == (("seq", False), ("created_at", False))
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
