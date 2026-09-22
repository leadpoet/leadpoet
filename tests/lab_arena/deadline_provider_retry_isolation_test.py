"""Scoring policy for a dispatched provider retry closed by the stage deadline."""

from __future__ import annotations

import pytest

from lab_arena import contracts, scoring


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


def test_deadline_closed_dispatched_retry_uses_prior_provider_error_zero():
    plan = _plan([_run(1, "provider_error"), _deadline_retry()])

    assert plan["work_items"] == []
    assert plan["zero_rows"] == [
        {
            "submission_id": "submission-1",
            "icp_position": 0,
            "cause": "provider_error",
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
def test_unproved_or_unsupported_deadline_retry_stays_fail_closed(runs):
    with pytest.raises(contracts.ArenaContractError, match="stage must cancel"):
        _plan(runs)


def test_accepted_attempt_keeps_authority_over_deadline_marker():
    accepted = _run(
        1,
        "accepted",
        status="accepted",
        output_ref="arena/test/output.json",
    )

    plan = _plan([accepted, _deadline_retry()])

    assert plan["zero_rows"] == []
    assert plan["work_items"] == [
        {
            "scored_run_id": accepted["run_id"],
            "submission_id": "submission-1",
            "icp_position": 0,
            "output_ref": "arena/test/output.json",
        }
    ]
