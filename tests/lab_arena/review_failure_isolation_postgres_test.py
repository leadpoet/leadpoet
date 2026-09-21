"""A failed review zeros one ICP while the real round still publishes."""

from dataclasses import replace

import pytest

from lab_arena import contracts, verify
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    _start_round, _run_stage_one_to_scoring, assert_canary_absent,
)

from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge


@pytest.fixture(scope="module")
def database():
    migration = "349-lab-arena-scoring-effective-outcome.sql"
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "289-lab-arena-per-icp-cost-policy.sql",
        "292-lab-arena-null-final-score-publication.sql",
    )
    if migration not in migrations:
        migrations += (migration,)
    yield from database_with_lab_arena_migration(migrations)


def test_baseline_and_miner_review_failures_publish_only_affected_icps_as_zero(
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
        ("PublicBaseline", 0),
        ("IsolatedFailure", 0),
        ("IsolatedFailure", 10),
    })
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("published", runners=2)

    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "published" and row["cancel_reason"] is None
    for participant in row["participants"]:
        submission_id = participant["submission_id"]
        flavor = harness.flavors[submission_id]
        public = harness.service.public_results(harness.round_id, submission_id)
        scores = public["scores"]["stage_1"] + public["scores"]["stage_2"]
        assert len(scores) == contracts.BENCHMARK_ICP_COUNT
        failed_positions = (
            {0} if participant["is_king"] else
            {0, 10} if flavor == "IsolatedFailure" else set()
        )
        assert {
            item["icp_position"] for item in scores
            if item["per_icp_score"] == 0
        } == failed_positions
        assert public["submission_scores"]["final"] > 0
        assert public["submission_scores"]["final"] == verify.stage_score(
            [item["per_icp_score"] for item in scores],
            contracts.BENCHMARK_ICP_COUNT,
        )
        runs = harness.service.store.list_runs(
            harness.round_id, submission_id=submission_id, kind="score"
        )
        executions = harness.service.store.list_runs(
            harness.round_id, submission_id=submission_id, kind="execute"
        )
        published = next(
            entry for entry in row["publication_doc"]["final_ranking"]
            if entry["submission_id"] == submission_id
        )
        costs = {
            entry["icp_position"]: entry
            for entry in published["cost_summary"]["per_icp"]
        }
        for position in failed_positions:
            failures = [run for run in runs if run["icp_position"] == position]
            assert {run["attempt"] for run in failures} == {1, 2}
            assert all(
                run["status"] == "failed" and run["terminal_cause"] == "judge_error"
                for run in failures
            )
            execution = next(
                run for run in executions
                if run["icp_position"] == position and run["status"] == "accepted"
            )
            assert execution["per_icp_score"] == 0
            assert execution["qualification_doc"] == {"companies": []}
            assert costs[position]["qualified_company_count"] == 0
            assert costs[position]["eligible"] is False
    assert_canary_absent(harness, connect)
