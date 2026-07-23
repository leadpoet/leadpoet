from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab import global_icp_queue


GENERATION_ID = "11111111-1111-4111-8111-111111111111"
CANDIDATE_ID = "candidate:" + "a" * 64


@pytest.mark.asyncio
async def test_phase_completion_requires_every_declared_job_to_be_done(
    monkeypatch,
) -> None:
    candidate = {
        "queue_generation_id": GENERATION_ID,
        "public_total": 2,
        "private_total": 2,
        "conditional_total": 0,
    }
    jobs = {
        "public": [
            {"status": "done"},
            {"status": "failed"},
        ],
        "private": [
            {"status": "done"},
        ],
    }

    async def select_candidate(*_args: Any, **_kwargs: Any):
        return dict(candidate)

    async def select_jobs(_table: str, *, filters, **_kwargs: Any):
        phase = next(value for field, value in filters if field == "phase")
        return [dict(row) for row in jobs.get(phase, [])]

    monkeypatch.setattr(global_icp_queue, "select_one", select_candidate)
    monkeypatch.setattr(global_icp_queue, "select_many", select_jobs)

    assert not await global_icp_queue.public_set_complete(GENERATION_ID)
    assert not await global_icp_queue.phase_set_complete(
        GENERATION_ID,
        "private",
    )

    jobs["public"] = [{"status": "done"}, {"status": "done"}]
    jobs["private"] = [{"status": "done"}, {"status": "done"}]
    assert await global_icp_queue.public_set_complete(GENERATION_ID)
    assert await global_icp_queue.phase_set_complete(
        GENERATION_ID,
        "private",
    )


@pytest.mark.asyncio
async def test_assembly_rejects_execution_failures_but_allows_gate_skips(
    monkeypatch,
) -> None:
    candidate = {
        "queue_generation_id": GENERATION_ID,
        "candidate_id": CANDIDATE_ID,
        "public_total": 2,
        "private_total": 2,
        "conditional_total": 0,
        "gate_status": "passed",
        "preliminary_gate_status": "not_required",
        "assembly_status": "pending",
    }
    jobs = [
        {"phase": "public", "status": "done"},
        {"phase": "public", "status": "done"},
        {"phase": "private", "status": "done"},
        {"phase": "private", "status": "failed"},
    ]

    async def select_candidate(*_args: Any, **_kwargs: Any):
        return dict(candidate)

    async def select_jobs(*_args: Any, **_kwargs: Any):
        return [dict(row) for row in jobs]

    monkeypatch.setattr(global_icp_queue, "select_one", select_candidate)
    monkeypatch.setattr(global_icp_queue, "select_many", select_jobs)

    assert (
        await global_icp_queue.candidate_ready_to_assemble(GENERATION_ID)
        is None
    )

    candidate["gate_status"] = "rejected"
    jobs[-2]["status"] = "failed"
    assert await global_icp_queue.candidate_ready_to_assemble(
        GENERATION_ID
    ) == candidate


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_phase", ["public", "private"])
async def test_exhausted_job_fails_generation_without_gate_or_assembly(
    monkeypatch,
    failed_phase: str,
) -> None:
    job = {
        "job_id": "22222222-2222-4222-8222-222222222222",
        "candidate_id": CANDIDATE_ID,
        "queue_generation_id": GENERATION_ID,
        "phase": failed_phase,
        "claimed_by": "worker:test",
        "attempt_count": 3,
    }
    claims = [job, None]
    failed_generations: list[dict[str, Any]] = []
    completion_callbacks: list[tuple[bool, bool]] = []
    projection_pending: dict[str, Any] | None = None

    async def no_recovery(**_kwargs: Any) -> int:
        return 0

    async def no_pending(*_args: Any, **_kwargs: Any):
        return []

    async def claim(**_kwargs: Any):
        return claims.pop(0)

    async def score(_job):
        raise TimeoutError("evaluation timed out")

    async def fail_generation(**kwargs: Any):
        nonlocal projection_pending
        assert kwargs["job"] == job
        assert kwargs["failure_class"] == "adapter_timeout"
        projection_pending = {
            "queue_generation_id": GENERATION_ID,
            "candidate_id": CANDIDATE_ID,
            "failure_projection_attempt_count": 1,
            "failure_doc": {
                "failure_class": "adapter_timeout",
                "failed_phase": failed_phase,
            },
        }
        return dict(projection_pending)

    async def claim_projection(**_kwargs: Any):
        nonlocal projection_pending
        generation = projection_pending
        projection_pending = None
        return dict(generation) if generation is not None else None

    async def complete_projection(**kwargs: Any):
        assert kwargs["generation"]["queue_generation_id"] == GENERATION_ID
        return True

    async def completed(
        _job,
        _result,
        failed,
        committed,
        _error,
    ):
        completion_callbacks.append((failed, committed))

    async def candidate_failed(generation):
        failed_generations.append(dict(generation))

    async def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("an exhausted generation cannot reach a gate")

    monkeypatch.setattr(global_icp_queue, "recover_stale_leases", no_recovery)
    monkeypatch.setattr(global_icp_queue, "select_many", no_pending)
    monkeypatch.setattr(global_icp_queue, "claim_next_job", claim)
    monkeypatch.setattr(
        global_icp_queue,
        "fail_generation_for_job",
        fail_generation,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "claim_failed_generation_projection",
        claim_projection,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "complete_failed_generation_projection",
        complete_projection,
    )
    monkeypatch.setattr(global_icp_queue, "public_set_complete", forbidden)
    monkeypatch.setattr(
        global_icp_queue,
        "candidate_ready_to_assemble",
        forbidden,
    )

    counters = await global_icp_queue.run_queue_scoring_pass(
        worker_ref="worker:test",
        lease_seconds=60,
        score_icp=score,
        compute_public_score=lambda _rows: 0.0,
        assemble_candidate=forbidden,
        retryable_job_error=lambda _job, _error: False,
        job_failure_class=lambda _job, _error: "adapter_timeout",
        job_completed=completed,
        candidate_failed=candidate_failed,
    )

    assert counters["failed"] == 1
    assert counters["gates_decided"] == 0
    assert counters["assembled"] == 0
    assert completion_callbacks == [(True, True)]
    assert len(failed_generations) == 1


@pytest.mark.asyncio
async def test_concurrent_exhausted_jobs_project_one_terminal_generation(
    monkeypatch,
) -> None:
    jobs = [
        {
            "job_id": f"22222222-2222-4222-8222-{index:012d}",
            "candidate_id": CANDIDATE_ID,
            "queue_generation_id": GENERATION_ID,
            "phase": "private",
            "claimed_by": f"worker:{index}",
            "attempt_count": 3,
        }
        for index in range(2)
    ]
    claim_lock = asyncio.Lock()
    fail_lock = asyncio.Lock()
    generation_closed = False
    projection_pending: dict[str, Any] | None = None
    projected: list[str] = []

    async def no_recovery(**_kwargs: Any) -> int:
        return 0

    async def no_pending(*_args: Any, **_kwargs: Any):
        return []

    async def claim(**_kwargs: Any):
        async with claim_lock:
            return jobs.pop(0) if jobs else None

    async def score(_job):
        await asyncio.sleep(0)
        raise ValueError("invalid candidate output")

    async def fail_generation(**_kwargs: Any):
        nonlocal generation_closed, projection_pending
        async with fail_lock:
            if generation_closed:
                return None
            generation_closed = True
            projection_pending = {
                "queue_generation_id": GENERATION_ID,
                "candidate_id": CANDIDATE_ID,
                "failure_projection_attempt_count": 1,
                "failure_doc": {
                    "failure_class": "candidate_scoring_error",
                },
            }
            return dict(projection_pending)

    async def claim_projection(**_kwargs: Any):
        nonlocal projection_pending
        async with fail_lock:
            generation = projection_pending
            projection_pending = None
            return dict(generation) if generation is not None else None

    async def complete_projection(**_kwargs: Any):
        return True

    async def candidate_failed(generation):
        projected.append(str(generation["queue_generation_id"]))

    monkeypatch.setattr(global_icp_queue, "recover_stale_leases", no_recovery)
    monkeypatch.setattr(global_icp_queue, "select_many", no_pending)
    monkeypatch.setattr(global_icp_queue, "claim_next_job", claim)
    monkeypatch.setattr(
        global_icp_queue,
        "fail_generation_for_job",
        fail_generation,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "claim_failed_generation_projection",
        claim_projection,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "complete_failed_generation_projection",
        complete_projection,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "candidate_ready_to_assemble",
        lambda *_args, **_kwargs: None,
    )

    async def run(worker_ref: str):
        return await global_icp_queue.run_queue_scoring_pass(
            worker_ref=worker_ref,
            lease_seconds=60,
            score_icp=score,
            compute_public_score=lambda _rows: 0.0,
            assemble_candidate=lambda *_args: None,
            retryable_job_error=lambda _job, _error: False,
            job_failure_class=lambda _job, _error: "candidate_scoring_error",
            candidate_failed=candidate_failed,
            max_jobs=1,
        )

    await asyncio.gather(run("worker:0"), run("worker:1"))
    assert projected == [GENERATION_ID]


@pytest.mark.asyncio
async def test_failed_generation_projection_retries_after_worker_restart(
    monkeypatch,
) -> None:
    projection_claims = [
        {
            "queue_generation_id": GENERATION_ID,
            "candidate_id": CANDIDATE_ID,
            "failure_projection_attempt_count": 1,
            "failure_doc": {"failure_class": "adapter_timeout"},
        },
        {
            "queue_generation_id": GENERATION_ID,
            "candidate_id": CANDIDATE_ID,
            "failure_projection_attempt_count": 2,
            "failure_doc": {"failure_class": "adapter_timeout"},
        },
    ]
    callback_attempts = 0
    completed_attempts: list[int] = []

    async def no_recovery(**_kwargs: Any) -> int:
        return 0

    async def no_pending(*_args: Any, **_kwargs: Any):
        return []

    async def no_job(**_kwargs: Any):
        return None

    async def claim_projection(**_kwargs: Any):
        return projection_claims.pop(0)

    async def complete_projection(**kwargs: Any):
        completed_attempts.append(
            int(kwargs["generation"]["failure_projection_attempt_count"])
        )
        return True

    async def candidate_failed(_generation):
        nonlocal callback_attempts
        callback_attempts += 1
        if callback_attempts == 1:
            raise RuntimeError("projection transport unavailable")

    monkeypatch.setattr(global_icp_queue, "recover_stale_leases", no_recovery)
    monkeypatch.setattr(global_icp_queue, "select_many", no_pending)
    monkeypatch.setattr(global_icp_queue, "claim_next_job", no_job)
    monkeypatch.setattr(
        global_icp_queue,
        "claim_failed_generation_projection",
        claim_projection,
    )
    monkeypatch.setattr(
        global_icp_queue,
        "complete_failed_generation_projection",
        complete_projection,
    )

    async def run(worker_ref: str):
        return await global_icp_queue.run_queue_scoring_pass(
            worker_ref=worker_ref,
            lease_seconds=60,
            score_icp=lambda _job: None,  # type: ignore[arg-type,return-value]
            compute_public_score=lambda _rows: 0.0,
            assemble_candidate=lambda *_args: None,  # type: ignore[arg-type,return-value]
            candidate_failed=candidate_failed,
        )

    await run("worker:before-restart")
    assert completed_attempts == []
    await run("worker:after-restart")
    assert callback_attempts == 2
    assert completed_attempts == [2]


def test_terminal_failure_migration_serializes_and_closes_generation() -> None:
    sql = (
        Path(__file__).parents[1]
        / "scripts"
        / "116-research-lab-global-queue-terminal-failure.sql"
    ).read_text()

    assert "FOR UPDATE" in sql
    assert "assembly_status = 'failed'" in sql
    assert "status IN ('held', 'queued', 'claimed')" in sql
    assert "expected_attempt_count" in sql
    assert "research_lab_fail_scoring_queue_generation" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "failure_projection_status = 'projected'" in sql
