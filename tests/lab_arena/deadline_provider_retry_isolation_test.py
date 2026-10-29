"""Scoring policy for a dispatched provider retry closed by the stage deadline."""

from __future__ import annotations

import pytest

from lab_arena import scoring


def _run(
    attempt: int,
    cause: str,
    *,
    status: str = "failed",
    terminal_doc: object = None,
    output_ref: str | None = None,
) -> dict:
    return {
        "run_id": "run-%d-%s" % (attempt, status),
        "submission_id": "submission-1",
        "stage": 1,
        "icp_position": 0,
        "attempt": attempt,
        "status": status,
        "terminal_cause": cause,
        "terminal_doc": terminal_doc,
        "output_ref": output_ref,
    }


def _deadline_retry(*, marker: object = True, previous_status: str = "leased") -> dict:
    return _run(
        2,
        "stage_closed",
        terminal_doc={
            "previous_status": previous_status,
            "deadline_provider_retry_exhausted": marker,
        },
    )


def _plan(runs: list[dict]) -> dict:
    return scoring.build_scoring_plan(
        round_id="arena-2026-09-22-test",
        stage=1,
        runs=runs,
    )


def test_deadline_closed_dispatched_retry_preserves_incomplete_work():
    plan = _plan([_run(1, "provider_error"), _deadline_retry()])

    assert plan["work_items"] == []
    assert plan["zero_rows"] == []
    assert plan["incomplete_rows"] == [
        {
            "submission_id": "submission-1",
            "icp_position": 0,
            "cause": "stage_closed",
        }
    ]


@pytest.mark.parametrize(
    "runs",
    [
        [_run(1, "provider_error"), _run(2, "stage_closed")],
        [_run(1, "provider_error"), _deadline_retry(marker=False)],
        [_run(1, "provider_error"), _deadline_retry(previous_status="pending")],
        [_run(1, "worker_lost"), _deadline_retry()],
        [_run(1, "result_rejected"), _deadline_retry()],
        [_deadline_retry()],
        [
            _run(
                1,
                "stage_closed",
                terminal_doc={"deadline_provider_retry_exhausted": True},
            )
        ],
    ],
)
def test_deadline_retry_marker_cannot_invent_a_score(runs):
    plan = _plan(runs)
    assert plan["work_items"] == []
    assert plan["zero_rows"] == []
    assert plan["incomplete_rows"] == [{"submission_id": "submission-1", "icp_position": 0,
                                        "cause": "stage_closed"}]


def test_confirmed_model_failure_keeps_its_zero_after_deadline_retry():
    plan = _plan([_run(1, "invalid_output"), _deadline_retry()])

    assert plan["work_items"] == []
    assert plan["zero_rows"] == [{"submission_id": "submission-1", "icp_position": 0,
                                 "cause": "invalid_output"}]
    assert "incomplete_rows" not in plan


def test_accepted_attempt_keeps_authority_over_deadline_marker():
    accepted = _run(
        1,
        "accepted",
        status="accepted",
        output_ref="arena/test/output.json",
    )

    plan = _plan([accepted, _deadline_retry()])

    assert plan["zero_rows"] == []
    assert not plan.get("incomplete_rows")
    assert plan["work_items"] == [
        {
            "scored_run_id": accepted["run_id"],
            "submission_id": "submission-1",
            "icp_position": 0,
            "output_ref": "arena/test/output.json",
        }
    ]
