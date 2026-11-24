"""The next funded reference must not start on missing, unsettled, or retryable work."""

import pytest


def job(slot, attempt=1, status="accepted", amount=3, unresolved=0):
    return dict(
        job_id=f"run-{slot}-{attempt}",
        slot=slot,
        attempt=attempt,
        status=status,
        artifact_sha256=(
            "sha256:" + "1" * 64 if status in ("accepted", "rejected") else None
        ),
        miner_cost_microusd=amount,
        budget_microusd=30,
        unresolved=unresolved,
        generator_calls=1 if amount else 0,
        model="model",
        scenario_id="P7-s01",
        batch_id="canary",
        harness_commit="a" * 40,
        miner_hotkey="fake",
    )


def test_frontier_requires_exact_terminal_range_including_safe_replacements():
    from scripts.collection_reference_followup import finished_slots

    completed = [job(1), job(2, 1, "failed"), job(2, 2, "failed"), job(3)]
    assert finished_slots(completed, 1, 4)
    for bad in (
        completed[:-1],
        completed[:2] + completed[3:],
        completed + [job(0)],
        completed + [job(1)],
        completed + [job(2, 3)],
        [dict(completed[0], unresolved=1), *completed[1:]],
        [dict(completed[0], status="running"), *completed[1:]],
    ):
        assert not finished_slots(bad, 1, 4)
    # There is no retryable liability if a failed attempt spent all its original budget.
    assert finished_slots([job(1), job(2, 1, "failed", 30), job(3)], 1, 4)
