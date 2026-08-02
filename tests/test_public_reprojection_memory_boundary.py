from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import pytest

from gateway.research_lab import public_activity
from gateway.research_lab import worker


def _compact_event_row(
    *,
    event_type: str,
    seq: int,
    failure_class: str,
    terminal: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "run-1",
        "event_type": event_type,
        "loop_status": "failed",
        "seq": seq,
        "created_at": f"2026-08-01T00:00:0{seq}+00:00",
        "event_failure_class": failure_class,
        "event_stage": "candidate_generation",
        "event_iteration": 2,
        "cost_stop_reason": "candidate_patch_parse_failed",
        "provider_0_call_stage": "draft",
        "provider_0_error_class": "upstream_timeout",
        "provider_0_http_status": "504",
    }
    if terminal:
        row.update(
            {
                "event_stop_reason": "no_valid_image_build_finalists",
                "run_stop_reason": "no_valid_image_build_finalists",
                "run_wall_clock_seconds": 90,
                "run_openrouter_call_count": 3,
                "run_iterations_completed": 2,
                "run_selected_candidate_count": 0,
                "run_actual_openrouter_cost_usd": 0.25,
                "run_estimated_cost_usd": 0.3,
                "generation_public_label": "No buildable candidate",
                "generation_latest_stage": "candidate_generation",
                "generation_stage_counts": {
                    "no_viable_patch": 1,
                    "loop_failed": 1,
                },
            }
        )
    return row


def _full_event_row(compact: dict[str, Any]) -> dict[str, Any]:
    event_doc: dict[str, Any] = {
        "failure_class": compact["event_failure_class"],
        "stage": compact["event_stage"],
        "iteration": compact["event_iteration"],
    }
    if compact.get("event_stop_reason"):
        event_doc["stop_reason"] = compact["event_stop_reason"]
        event_doc["run_summary"] = {
            "stop_reason": compact["run_stop_reason"],
            "wall_clock_seconds": compact["run_wall_clock_seconds"],
            "openrouter_call_count": compact["run_openrouter_call_count"],
            "iterations_completed": compact["run_iterations_completed"],
            "selected_candidate_count": compact[
                "run_selected_candidate_count"
            ],
            "cost_ledger": {
                "actual_openrouter_cost_usd": compact[
                    "run_actual_openrouter_cost_usd"
                ],
                "estimated_cost_usd": compact["run_estimated_cost_usd"],
            },
        }
        event_doc["candidate_generation_failure"] = {
            "public_label": compact["generation_public_label"],
            "latest_stage": compact["generation_latest_stage"],
            "stage_counts": compact["generation_stage_counts"],
        }
    return {
        "run_id": compact["run_id"],
        "event_type": compact["event_type"],
        "loop_status": compact["loop_status"],
        "seq": compact["seq"],
        "created_at": compact["created_at"],
        "provider_usage": [
            {
                "call_stage": compact["provider_0_call_stage"],
                "failed_request": {
                    "error_class": compact["provider_0_error_class"],
                    "http_status": compact["provider_0_http_status"],
                    "unused_private_body": "x" * 100_000,
                },
                "unused_private_request": "x" * 100_000,
            }
        ],
        "cost_ledger": {"stop_reason": compact["cost_stop_reason"]},
        "event_doc": {
            **event_doc,
            "unused_private_source_and_model_evidence": "x" * 500_000,
        },
    }


def test_compact_event_projection_is_behaviorally_equivalent() -> None:
    compact_rows = [
        _compact_event_row(
            event_type="no_viable_patch",
            seq=1,
            failure_class="binding_plan_unimplementable",
        ),
        _compact_event_row(
            event_type="loop_failed",
            seq=2,
            failure_class="candidate_generation_failed",
            terminal=True,
        ),
    ]
    full_rows = [_full_event_row(row) for row in compact_rows]
    reconstructed = [
        public_activity._reconstruct_projection_auto_loop_event(row)
        for row in compact_rows
    ]

    common = {
        "ticket": {
            "current_ticket_status": "failed",
            "created_at": "2026-08-01T00:00:00+00:00",
        },
        "queue_rows": [
            {
                "run_id": "run-1",
                "current_queue_status": "failed",
                "current_reason": "candidate_generation_failed",
                "current_status_at": "2026-08-01T00:00:03+00:00",
            }
        ],
        "receipt_rows": [],
        "candidate_rows": [],
        "score_bundle_rows": [],
        "promotion_event_rows": [],
    }
    full_outcome = public_activity.derive_public_loop_outcome(
        **common, auto_loop_event_rows=full_rows
    )
    compact_outcome = public_activity.derive_public_loop_outcome(
        **common, auto_loop_event_rows=reconstructed
    )
    assert asdict(compact_outcome) == asdict(full_outcome)
    assert "unused_private_source_and_model_evidence" not in str(reconstructed)


def test_compact_score_projection_preserves_promotion_metric() -> None:
    score_doc = {
        "aggregates": {"mean_delta": 4.5, "delta_lcb": 3.5},
        "unread_private_icp_results": ["x" * 100_000],
    }
    reconstructed = public_activity._reconstruct_projection_score_bundle(
        {
            "candidate_artifact_hash": "sha256:" + "1" * 64,
            "current_status_at": "2026-08-01T00:00:00+00:00",
            "created_at": "2026-08-01T00:00:00+00:00",
            "score_aggregates": score_doc["aggregates"],
            "score_private_holdout_gate": None,
            "score_improvement_gate": None,
        }
    )
    assert public_activity._score_bundle_delta(reconstructed) == (
        public_activity._score_bundle_delta({"score_bundle_doc": score_doc})
    )
    assert "unread_private_icp_results" not in reconstructed["score_bundle_doc"]


@pytest.mark.asyncio
async def test_fetch_projection_inputs_never_selects_full_private_documents(
    monkeypatch,
) -> None:
    observed: dict[str, str] = {}

    async def select_one(table: str, *, columns: str, **_kwargs: Any):
        observed[table] = columns
        return {
            "ticket_id": "ticket-1",
            "current_ticket_status": "opened",
            "created_at": "2026-08-01T00:00:00+00:00",
        }

    async def select_many(table: str, *, columns: str, **_kwargs: Any):
        observed[table] = columns
        if table == "research_loop_run_queue_current":
            return [{"run_id": "run-1", "current_status_at": "2026-08-01T00:00:00+00:00"}]
        if table == "research_evaluation_score_bundle_current":
            return []
        return []

    async def select_all(table: str, *, columns: str, **_kwargs: Any):
        observed[table] = columns
        return []

    async def promotions(_rows):
        return []

    monkeypatch.setattr(public_activity, "select_one", select_one)
    monkeypatch.setattr(public_activity, "select_many", select_many)
    monkeypatch.setattr(public_activity, "select_all", select_all)
    monkeypatch.setattr(public_activity, "_promotion_events_for_candidates", promotions)

    assert await public_activity._fetch_projection_inputs("ticket-1") is not None
    assert all(columns != "*" for columns in observed.values())
    auto_columns = observed["research_lab_auto_research_loop_events"]
    assert ",event_doc," not in f",{auto_columns},"
    assert ",provider_usage," not in f",{auto_columns},"
    assert "event_doc->" in auto_columns
    assert "provider_usage->0->" in auto_columns
    score_columns = observed["research_evaluation_score_bundle_current"]
    assert "score_bundle_doc->aggregates" in score_columns
    assert ",score_bundle_doc," not in f",{score_columns},"


class _FakeProcess:
    def __init__(self, return_code: int) -> None:
        self.returncode: int | None = None
        self.pid = 12345
        self._return_code = return_code

    async def wait(self) -> int:
        self.returncode = self._return_code
        return self._return_code


@pytest.mark.asyncio
async def test_worker_runs_reprojection_in_exact_runtime_subprocess(
    monkeypatch,
) -> None:
    observed: dict[str, Any] = {}

    async def create(*argv: str, **kwargs: Any):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return _FakeProcess(0)

    monkeypatch.setattr(worker.asyncio, "create_subprocess_exec", create)
    await worker._run_public_reprojection_subprocess()

    assert observed["argv"][0] == worker.sys.executable
    assert observed["argv"][1:5] == (
        "-m",
        "gateway.research_lab.maintenance_process",
        "--task",
        "public-reprojection",
    )
    assert observed["kwargs"]["cwd"] == str(
        worker.Path(worker.__file__).resolve().parents[2]
    )
    assert observed["kwargs"]["start_new_session"] is (worker.os.name == "posix")


@pytest.mark.asyncio
async def test_worker_fails_visible_on_reprojection_subprocess_error(
    monkeypatch,
) -> None:
    async def create(*_argv: str, **_kwargs: Any):
        return _FakeProcess(17)

    monkeypatch.setattr(worker.asyncio, "create_subprocess_exec", create)
    with pytest.raises(RuntimeError, match="status 17"):
        await worker._run_public_reprojection_subprocess()


@pytest.mark.asyncio
async def test_worker_kills_timed_out_reprojection_process_group(
    monkeypatch,
) -> None:
    process = _FakeProcess(0)
    kill_calls: list[tuple[int, int]] = []
    wait_calls = 0
    real_wait_for = asyncio.wait_for

    async def create(*_argv: str, **_kwargs: Any):
        return process

    async def wait_for(awaitable, *, timeout: float):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(worker.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(worker.asyncio, "wait_for", wait_for)
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        await worker._run_public_reprojection_subprocess()
    if worker.os.name == "posix":
        assert kill_calls == [(process.pid, worker.signal.SIGTERM)]
    assert process.returncode == 0
