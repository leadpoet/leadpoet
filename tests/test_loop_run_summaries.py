"""Owner-readable run-level summaries for zero-candidate loop runs.

A paid loop that ends no_buildable_candidate produces no candidate
diagnostics, leaving the miner blind to why the run stopped. The run-level
summary exposes structured fields only (stop_reason, last stage, call count,
the miner's own spend) and must never carry free-text reason/error strings,
which quote the model's analysis of the private source tree.
"""

from __future__ import annotations

import json

from gateway.research_lab.candidate_diagnostics import _sanitized_run_summary_from_terminal_event


TERMINAL_DOC = {
    "run_summary": {
        "status": "failed",
        "stop_reason": "loop_direction_no_new_safe_path",
        "wall_clock_seconds": 92.209,
        "iterations_completed": 1,
        "openrouter_call_count": 6,
        "selected_candidate_count": 0,
        "cost_ledger": {
            "actual_openrouter_cost_usd": 1.140948,
            "estimated_cost_usd": 3.0,
        },
    },
    "stop_reason": "loop_direction_no_new_safe_path",
    "failure_reason": "no_viable_patch",
    "candidate_generation_failure": {
        "latest_stage": "loop_failed",
        "public_label": "No buildable candidate",
        "stage_counts": {"loop_failed": 1, "no_viable_patch": 2},
    },
    # Private material that must NOT appear in the sanitized projection.
    "reason": "The binding plan requires inspection of the downstream gate paths",
    "error": "ValueError: code-edit response requires a non-empty candidates array",
}


def _summary():
    return _sanitized_run_summary_from_terminal_event(
        "611e99d7-1f29-4ad5-8d1a-f91739017ce2",
        "failed",
        TERMINAL_DOC,
        ["insufficient_source_context"],
    )


def test_exposes_the_fields_miners_need():
    doc = _summary()
    assert doc["stop_reason"] == "loop_direction_no_new_safe_path"
    assert doc["failure_reason"] == "no_viable_patch"
    assert doc["last_completed_stage"] == "loop_failed"
    assert doc["openrouter_call_count"] == 6
    assert doc["iterations_completed"] == 1
    assert doc["selected_candidate_count"] == 0
    assert doc["actual_compute_used_usd"] == 1.140948
    assert doc["estimated_cost_usd"] == 3.0
    assert doc["failure_classes"] == ["insufficient_source_context"]
    assert doc["stage_counts"] == {"loop_failed": 1, "no_viable_patch": 2}
    assert doc["public_label"] == "No buildable candidate"


def test_never_leaks_free_text_reason_or_error():
    blob = json.dumps(_summary())
    assert "binding plan" not in blob
    assert "downstream gate" not in blob
    assert "ValueError" not in blob
    assert "candidates array" not in blob


def test_tolerates_missing_and_malformed_sections():
    doc = _sanitized_run_summary_from_terminal_event("run-x", "failed", {}, [])
    assert doc["run_id"] == "run-x"
    assert doc["stop_reason"] == ""
    assert doc["stage_counts"] == {}
    assert "actual_compute_used_usd" not in doc
    doc = _sanitized_run_summary_from_terminal_event(
        "run-y", "completed", {"run_summary": "corrupt", "candidate_generation_failure": 3}, []
    )
    assert doc["loop_status"] == "completed"


def test_booleans_never_masquerade_as_counts():
    doc = _sanitized_run_summary_from_terminal_event(
        "run-z",
        "failed",
        {"run_summary": {"openrouter_call_count": True, "cost_ledger": {"actual_openrouter_cost_usd": True}}},
        [],
    )
    assert "openrouter_call_count" not in doc
    assert "actual_compute_used_usd" not in doc


def test_public_projection_never_carries_the_miners_spend():
    from gateway.research_lab.candidate_diagnostics import (
        public_run_summary_from_terminal_event,
    )

    doc = public_run_summary_from_terminal_event(
        "611e99d7-1f29-4ad5-8d1a-f91739017ce2",
        "failed",
        TERMINAL_DOC,
        ["insufficient_source_context"],
    )
    serialized = json.dumps(doc)
    assert "usd" not in serialized.lower()
    assert "1.140948" not in serialized
    assert doc["stop_reason"] == "loop_direction_no_new_safe_path"
    assert doc["failure_reason"] == "no_viable_patch"
    assert doc["last_completed_stage"] == "loop_failed"
    assert doc["stage_counts"] == {"loop_failed": 1, "no_viable_patch": 2}
    assert doc["failure_classes"] == ["insufficient_source_context"]
    assert doc["openrouter_call_count"] == 6
    assert "requires inspection" not in serialized
    assert "ValueError" not in serialized


def test_public_outcome_event_doc_carries_the_run_summary():
    from gateway.research_lab.public_activity import derive_public_loop_outcome

    run_id = "611e99d7-1f29-4ad5-8d1a-f91739017ce2"
    outcome = derive_public_loop_outcome(
        ticket={"ticket_id": "t-1", "current_ticket_status": "failed"},
        queue_rows=[
            {
                "run_id": run_id,
                "current_queue_status": "failed",
                "current_reason": "no_buildable_candidate",
                "current_status_at": "2026-07-12T11:16:30Z",
            }
        ],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
        auto_loop_event_rows=[
            {
                "run_id": run_id,
                "event_type": "patch_drafted",
                "loop_status": "running",
                "seq": 5,
                "event_doc": {"failure_class": "insufficient_source_context"},
            },
            {
                "run_id": run_id,
                "event_type": "loop_failed",
                "loop_status": "failed",
                "seq": 9,
                "event_doc": TERMINAL_DOC,
            },
            {
                # A different run's terminal event must not win the projection.
                "run_id": "other-run",
                "event_type": "loop_completed",
                "loop_status": "completed",
                "seq": 99,
                "event_doc": {"stop_reason": "loop_completed"},
            },
        ],
    )
    summary = outcome.event_doc.get("run_summary")
    assert summary is not None
    assert summary["run_id"] == run_id
    assert summary["stop_reason"] == "loop_direction_no_new_safe_path"
    assert summary["failure_classes"] == ["insufficient_source_context"]
    assert "usd" not in json.dumps(summary).lower()


def test_public_outcome_omits_run_summary_without_a_terminal_event():
    from gateway.research_lab.public_activity import derive_public_loop_outcome

    outcome = derive_public_loop_outcome(
        ticket={"ticket_id": "t-2", "current_ticket_status": "running"},
        queue_rows=[
            {
                "run_id": "run-live",
                "current_queue_status": "running",
                "current_reason": "",
                "current_status_at": "2026-07-13T07:00:00Z",
            }
        ],
        receipt_rows=[],
        candidate_rows=[],
        score_bundle_rows=[],
        promotion_event_rows=[],
        auto_loop_event_rows=[
            {
                "run_id": "run-live",
                "event_type": "loop_started",
                "loop_status": "running",
                "seq": 1,
                "event_doc": {},
            }
        ],
    )
    assert "run_summary" not in outcome.event_doc
