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
