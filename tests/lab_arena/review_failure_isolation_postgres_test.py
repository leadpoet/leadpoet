"""Failed reviews keep unknown scores until real bound retries recover."""

from dataclasses import replace
import json

import pytest

from lab_arena import contracts, verify
from tests.lab_arena.lab_arena_pg_harness import (
    current_service_database,
)
from tests.lab_arena.test_lab_arena_service_round import (
    _start_round, _run_stage_one_to_scoring, assert_canary_absent,
)

from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge


@pytest.fixture(scope="module")
def database():
    yield from current_service_database()


def test_baseline_and_miner_review_failures_recover_without_invented_zero_scores(
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
    failed_history = []
    failed_billing = {}
    for stage, affected in ((1, {('PublicBaseline', 0), ('IsolatedFailure', 0)}),
                            (2, {('IsolatedFailure', 10)})):
        harness.run_stage_with_runners(2)
        runs = harness.service.store.list_runs(harness.round_id, stage=stage,
                                              kind='score')
        failed = [run for run in runs if run['status'] == 'failed']
        assert len(failed) == 2 * len(affected)
        assert {(harness.flavors[run['submission_id']], run['icp_position'])
                for run in failed} == affected
        assert {run['attempt'] for run in failed} == {1, 2}
        assert all(run['terminal_cause'] == 'judge_error' for run in failed)
        failed_history.extend(failed)
        failed_billing.update({run['run_id']:
            harness.service.store.list_ledger(run_id=run['run_id']) for run in failed})
        pending = [run for run in runs if run['status'] == 'pending']
        assert len(pending) == len(affected)
        assert {run['attempt'] for run in pending} == {3}
        assert {run['scored_run_id'] for run in pending} == {
            run['scored_run_id'] for run in failed}
        for retry in pending:
            source = harness.service.store.get_run(retry['scored_run_id'])
            assert source['per_icp_score'] is None
            assert source['qualification_doc'] is None
        held = harness.service.store.get_round(harness.round_id)
        assert held['publication_doc'] is None and held['reward_basis_doc'] is None
        assert harness.status() == 'stage%d_scoring' % stage
        # SQL 472 keeps unknown review outcomes pending. Remove only this
        # stage's fault, then obey the actual retry backoff and accepted API.
        harness.sandbox.judge_failures.difference_update(affected)
        harness.wait_for_score_retry_backoff(max_attempt=3)
        harness.run_stage_with_runners(2)
        for before in failed_history:
            assert harness.service.store.get_run(before['run_id']) == before
            assert harness.service.store.list_ledger(run_id=before['run_id']) == failed_billing[before['run_id']]
        if stage == 1:
            harness.advance_until('stage2_scoring', runners=2)
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
        assert all(item['per_icp_score'] > 0 for item in scores)
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
            attempts = [run for run in runs if run["icp_position"] == position]
            failures = [run for run in attempts if run['status'] == 'failed']
            assert {run["attempt"] for run in failures} == {1, 2}
            assert all(
                run["status"] == "failed" and run["terminal_cause"] == "judge_error"
                for run in failures
            )
            execution = next(
                run for run in executions
                if run["icp_position"] == position and run["status"] == "accepted"
            )
            accepted = [run for run in attempts if run['status'] == 'accepted']
            assert len(accepted) == 1 and accepted[0]['attempt'] == 3
            assert accepted[0]['scored_run_id'] == execution['run_id']
            assert accepted[0]['terminal_cause'] == 'accepted'
            companies = json.loads(harness.objects.get(execution['output_ref']))['companies']
            measured = harness.service._verified_breakdowns(
                accepted[0], icp=harness.service.benchmark_icps(harness.round_id)[position],
                companies=companies, policy=row['configuration_doc']['scorer_policy'])
            assert measured and all(item['final_score'] > 0 for item in measured)
            assert execution['per_icp_score'] > 0
            assert execution['qualification_doc']['companies']
            assert costs[position]["qualified_company_count"] == 5
            # This fixture has free execution. Only the real accepted retry
            # establishes positive qualification; the failed reviews did not.
            assert costs[position]["competition_sourcing_microusd"] == 0
            assert costs[position]["eligible"] is True
    assert_canary_absent(harness, connect)
