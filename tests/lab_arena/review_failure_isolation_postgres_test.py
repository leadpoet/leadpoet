"""Failed judgments retain evidence and cannot become qualification zeroes."""
from copy import deepcopy
from dataclasses import replace
import time
from lab_arena.service import ServiceError
import json
from pathlib import Path

import pytest

from lab_arena import contracts, verify
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, current_service_database, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    _start_round, _run_stage_one_to_scoring, assert_canary_absent,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge


@pytest.fixture
def historical_database():
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "289-lab-arena-per-icp-cost-policy.sql", "292-lab-arena-null-final-score-publication.sql",
    )
    migration = "349-lab-arena-scoring-effective-outcome.sql"
    yield from database_with_lab_arena_migration(migrations + (() if migration in migrations else (migration,)))


@pytest.fixture
def database():
    yield from current_service_database()


def test_exact_349_close_prefers_accepted_recovery_and_keeps_other_failures(historical_database):
    pg, dsn = historical_database
    rid, hotkey = "arena-2026-09-28-review", "5" + "A" * 47
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc) VALUES(%s,'stage1_scoring',%s::jsonb)",
            (rid, json.dumps({"mode": "live", "rewards_enabled": False})))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) VALUES('review-submission',%s,%s,'frozen')", (rid, hotkey))
        for position in range(3):
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,terminal_cause,output_ref) VALUES(%s,%s,%s,'review-submission',%s,1,%s,1,'accepted','accepted','saved-output')",
                (f"execution-{position}", f"assignment-{position}", rid, hotkey, position))
        for run_id, assignment, source, position, status in (
            ("failed-old", "old-namespace", "execution-0", 0, "failed"),
            ("accepted-recovery", "new-namespace", "execution-0", 0, "accepted"),
            ("failed-only", "failure-namespace", "execution-1", 1, "failed"),
            ("healthy", "healthy-namespace", "execution-2", 2, "accepted"),
        ):
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,scored_run_id,status,terminal_cause,output_ref) VALUES(%s,%s,%s,'review-submission',%s,1,%s,1,'score',%s,%s,%s,%s)",
                (run_id, assignment, rid, hotkey, position, source, status, "accepted" if status == "accepted" else "judge_error", "saved-score" if status == "accepted" else None))
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s", (rid,))
        evidence = cur.fetchone()[0]
        cur.execute("SELECT public.lab_arena_close_scoring(%s,1::smallint)", (rid,))
        closed = cur.fetchone()[0]
        assert closed["status"] == "closed" and closed["round_status"] == "stage1_judged"
        assert closed["incomplete_assignments"] == 1
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s", (rid,))
        assert cur.fetchone()[0] == evidence
        migration = Path(__file__).parents[2] / "scripts/349-lab-arena-scoring-effective-outcome.sql"
        cur.execute(migration.read_text())
        cur.execute(migration.read_text())
        cur.execute("SELECT public.lab_arena_close_scoring(%s,1::smallint)", (rid,))
        assert cur.fetchone()[0]["status"] == "existing"
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s", (rid,))
        assert cur.fetchone()[0] == evidence


def test_current_judge_failures_hold_null_scores_preserve_peers_and_recover(database, tmp_path, monkeypatch):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness = IntegrityHarness(connect, tmp_path,
        challengers=["IsolatedFailure", "UnaffectedMiner"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(harness.service.config.defaults, per_icp_cost_policy=True)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    participants = _start_round(harness, day=28, epoch=30528)
    harness.sandbox.judge_failures.update({("PublicBaseline", 0), ("IsolatedFailure", 0), ("IsolatedFailure", 10)})
    _run_stage_one_to_scoring(harness, participants, runners=2)
    retained_failures = {}
    retained_accepted = {}
    retained_bills = {}

    for stage in (1, 2):
        if stage == 2:
            harness.clock.advance_to(harness.schedule()["stage_2_start"])
            harness.advance_until("stage2_scoring", runners=2)
        harness.run_stage_with_runners(2)
        all_runs = harness.service.store.list_runs(harness.round_id)
        failures = [run for run in all_runs if run["kind"] == "score" and run["stage"] == stage and run["status"] == "failed"]
        assert len(failures) == (2 if stage == 1 else 1)
        assert all(run["attempt"] == 1 and run["terminal_cause"] == "judge_error" for run in failures)
        retained_failures.update({run["run_id"]: deepcopy(run) for run in failures})
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT entry_id,to_jsonb(l) FROM public.lab_arena_ledger l WHERE round_id=%s", (harness.round_id,))
            retained_bills.update(dict(cur.fetchall()))
        accepted = {run["run_id"]: deepcopy(run) for run in all_runs if run["kind"] == "score" and run["status"] == "accepted"}
        retained_accepted.update(accepted)
        failed_sources = {run["scored_run_id"] for run in failures}
        assert all(run["per_icp_score"] is None and run["qualification_doc"] is None for run in all_runs
            if run["run_id"] in failed_sources)
        assert any(run["submission_id"] == next(sid for sid, flavor in harness.flavors.items() if flavor == "UnaffectedMiner") for run in accepted.values())
        held = harness.service.advance_round(harness.round_id)
        assert held["status"] == "waiting" and held["round_status"] == f"stage{stage}_scoring"
        row = harness.service.store.get_round(harness.round_id)
        assert harness.service._unfinished_judgments(row)
        assert row["publication_doc"] is None and row["cancel_reason"] is None
        assert row["reward_activated_at"] is None and row["baseline_promoted_at"] is None
        for source in failed_sources:
            assert harness.service.store.get_run(source)["per_icp_score"] is None
        harness.run_stage_with_runners(2)
        assert all(harness.service.store.get_run(rid) == value for rid, value in retained_accepted.items())
        assert all(harness.service.store.get_run(rid) == value for rid, value in retained_failures.items())
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT run_id,clock_timestamp()<created_at+interval '30 seconds',public.lab_arena__judge_calls_clear_v1(assignment_id) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND stage=%s AND attempt=2 AND status='pending'", (harness.round_id, stage))
            pending = cur.fetchall()
            assert len(pending) == len(failures) and all(backoff and clear for _, backoff, clear in pending)
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=ANY(%s)", ([rid for rid, _, _ in pending],))
        harness.sandbox.judge_failures.difference_update({("PublicBaseline", 0), ("IsolatedFailure", 0)} if stage == 1 else {("IsolatedFailure", 10)})
        harness.run_stage_with_runners(2)
        assert harness.service.close_scoring(harness.round_id, stage)["status"] == "closed"
        assert harness.service.score_stage(harness.round_id, stage)["status"] == "ok"

    assert harness.service.publish(harness.round_id)["status"] == "ok"
    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "published" and row["cancel_reason"] is None
    for participant in row["participants"]:
        sid = participant["submission_id"]
        public = harness.service.public_results(harness.round_id, sid)
        scores = public["scores"]["stage_1"] + public["scores"]["stage_2"]
        assert len(scores) == contracts.BENCHMARK_ICP_COUNT
        assert all(item["per_icp_score"] > 0 for item in scores)
        assert public["submission_scores"]["final"] == verify.stage_score([item["per_icp_score"] for item in scores], contracts.BENCHMARK_ICP_COUNT)
        published = next(entry for entry in row["publication_doc"]["final_ranking"] if entry["submission_id"] == sid)
        costs = published["cost_summary"]["per_icp"]
        assert len(costs) == contracts.BENCHMARK_ICP_COUNT
        assert {entry["icp_position"] for entry in costs} == set(range(contracts.BENCHMARK_ICP_COUNT))
        assert all(entry["competition_sourcing_microusd"] == 0 and entry["eligible"] is True for entry in costs)
        assert all(entry["qualified_company_count"] == 5 for entry in costs)
    assert all(harness.service.store.get_run(rid) == value for rid, value in retained_failures.items())
    assert all(harness.service.store.get_run(rid) == value for rid, value in retained_accepted.items())
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT entry_id,to_jsonb(l) FROM public.lab_arena_ledger l WHERE round_id=%s", (harness.round_id,))
        bills = dict(cur.fetchall())
    assert retained_bills and all(bills[key] == value for key, value in retained_bills.items())
    assert_canary_absent(harness, connect)


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
