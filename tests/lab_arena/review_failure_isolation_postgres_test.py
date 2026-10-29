"""Incomplete judge work preserves healthy work and blocks final authority."""

from dataclasses import replace
import time

import pytest

from lab_arena.service import ServiceError
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_service_round import (
    _start_round, _run_stage_one_to_scoring, assert_canary_absent,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge


@pytest.fixture(scope="module")
def database():
    yield from current_service_database()


def test_baseline_and_miner_review_failures_preserve_work_and_hold_publication(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path,
        challengers=["IsolatedFailure", "UnaffectedMiner"],
        runners=["alpha", "beta"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults, per_icp_cost_policy=True,
    )
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    participants = _start_round(harness, day=28, epoch=30528)
    harness.sandbox.judge_failures.update({
        ("PublicBaseline", 0), ("IsolatedFailure", 0),
    })
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.run_stage_with_runners(2)
    assert harness.service.advance_round(harness.round_id)["status"] == "waiting"
    # Preserve the current retry admission boundary and the queued row timestamps.
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT max(extract(epoch FROM created_at+interval '30 seconds'-clock_timestamp())) "
            "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='pending'",
            (harness.round_id,),
        )
        delay = float(cursor.fetchone()[0])
    assert 0 < delay <= 30
    time.sleep(delay + 0.01)
    harness.run_stage_with_runners(2)
    harness.clock.advance_to(harness.schedule()["stage_1_scoring_close"])
    assert harness.service.advance_round(harness.round_id)["status"] == "closed"
    result = harness.service.advance_round(harness.round_id)
    assert result["status"] == "retry", result
    assert result["reason"] == "judge_incomplete", result
    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "stage1_judged"
    assert row["publication_doc"] is None
    assert row["reward_basis_doc"] is None
    assert row["reward_activated_at"] is None
    assert row["cancel_reason"] is None
    for participant in row["participants"]:
        submission_id = participant["submission_id"]
        flavor = harness.flavors[submission_id]
        executions = harness.service.store.list_runs(
            harness.round_id, submission_id=submission_id, kind="execute",
        )
        assert len(executions) == 10
        assert all(run["status"] == "accepted" for run in executions)
        judges = harness.service.store.list_runs(
            harness.round_id, submission_id=submission_id, kind="score",
        )
        healthy = [run for run in judges if run["icp_position"] != 0]
        assert len(healthy) == 9
        assert all(run["status"] == "accepted" for run in healthy)
        first = [run for run in judges if run["icp_position"] == 0]
        if flavor == "UnaffectedMiner":
            assert len(first) == 1 and first[0]["status"] == "accepted"
        else:
            # Current host recovery can append retries after the ordinary pair.
            assert len(first) >= 2
            assert {run["attempt"] for run in first} == set(range(1, len(first) + 1))
            assert all(run["status"] == "failed" and run["terminal_cause"] == "judge_error"
                       for run in first[:2])
            assert all(run["status"] in ("failed", "pending") for run in first)
            assert all(run["output_ref"] is None for run in first)
            assert all(run["terminal_cause"] is None for run in first if run["status"] == "pending")
        with pytest.raises(ServiceError, match="results_not_public"):
            harness.service.public_results(harness.round_id, submission_id)
    assert_canary_absent(harness, connect)
