"""Run a partial Arena day through signed HTTP, broker, PostgreSQL, and publication."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab_arena import broker as br, contracts, scoring, submission_runtime
from lab_arena.api import create_app
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.company_only_capacity_e2e_test import _install_company_only_sandbox
from tests.lab_arena.successful_call_cost_round_test import ChargedTransport
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.parallel_twenty_runtime_e2e_test import (
    _ExecutionEvidence, _http_runner, _postgres_lease_clock,
    bounded_socket_shutdown_poll,
)
from tests.lab_arena.partial_execution_close422_postgres_test import (
    base_database, current_database, database, migrated,
)
from tests.lab_arena.test_lab_arena_service_round import _start_round, keypair

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def integrated(migrated):
    psycopg, dsn = migrated
    migrations = list((ROOT / "scripts").glob("423-*.sql"))
    assert len(migrations) == 1
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(migrations[0].read_text())
        cur.execute(migrations[0].read_text())
        transition = ROOT / "scripts/424-lab-arena-partial-baseline-stage-transition.sql"
        cur.execute(transition.read_text())
        cur.execute(transition.read_text())
    return migrated


def _execute_exact(runner, harness, count):
    with _postgres_lease_clock(harness):
        for _ in range(count):
            lease = runner.claim_one()
            assert lease["status"] == "leased", lease
            assert runner._slots.acquire(blocking=False)
            runner._run_lease(lease)
            assert runner.completed[-1]["result"]["status"] == "accepted"


def _advance_with_http_scores(harness, runner, target):
    for _ in range(30):
        status = harness.status()
        if status == target:
            return
        if status in ("stage1_scoring", "stage2_scoring"):
            with _postgres_lease_clock(harness):
                while runner.run_once(max_claims=100):
                    pass
        if status == "stage1_scored":
            harness.clock.advance_to(harness.schedule()["stage_2_start"])
        result = harness.service.advance_round(harness.round_id)
        assert result.get("status") not in ("cancelled", "terminal", "retry", "stale"), (status, result, harness.service.store.get_round(harness.round_id).get("cancel_reason"))
    raise AssertionError(f"round did not reach {target}: {harness.status()}")


def _project_closed_stage_to_real_clock(connect, round_id, field):
    """Align a test-only accelerated schedule with PostgreSQL's real clock."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("""
            UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(
              configuration_doc,ARRAY['schedule',%s],to_jsonb(%s::text))
            WHERE round_id=%s
        """, (field, cutoff, round_id))
        cur.execute("SET LOCAL session_replication_role=origin")
    return cutoff


def test_partial_baseline_and_challenger_publish_without_unjust_promotion(integrated, tmp_path):
    psycopg, dsn = integrated
    connect = lambda: psycopg.connect(**dsn)
    harness = QualityHarness(
        connect, tmp_path, challengers=["FullCandidate", "PartialCandidate"],
        runners=["alpha"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        execution_sequence_from="2026-01-01T00:00:00Z",
        parallel_twenty_icp_execution=False,
        runner_slot_ceiling=10,
        per_icp_cost_policy=True,
        integrity_from="2026-01-01T00:00:00Z",
        cost_per_company_microusd=contracts.PER_ICP_QUALIFIED_PAIR_CAP_MICROUSD,
        execution_icp_cap_microusd=contracts.PER_ICP_EXECUTION_CAP_MICROUSD,
        company_quality_from=None,
        contacts_from=None,
        intent_details_from="2026-01-01T00:00:00Z",
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": daily_icps()[:10],
    }
    payer = submission_runtime.SubmissionProviderKeys(
        store=harness.service.store,
        credentials=fixtures.FakeCredentialManager(),
        organizer_keys=fixtures.CANARY_KEYS,
    )
    transport = ChargedTransport()
    harness.service.config.broker_factory = lambda _service, _round: br.Broker(
        store=harness.service.store,
        key_for=lambda provider: fixtures.CANARY_KEYS[provider],
        credential_for=payer.credential_for,
        funding_source_for=payer.funding_source_for,
        price_table=fixtures.price_table(),
        judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
        transport=transport,
        clock=harness.clock,
    )
    for flavor in harness.challengers:
        miner = keypair("svc-miner-" + flavor).ss58_address
        harness.chain.runners.append(miner)
        harness.chain.permits[miner] = False
        harness.chain.active[miner] = False
        harness.chain.stakes[miner] = 0.0
    assert _start_round(harness, day=28, epoch=62028) == 3
    round_id = harness.round_id
    frozen = harness.service.store.get_round(round_id)["configuration_doc"]
    assert frozen["execution_sequence_policy"] == contracts.BASELINE_SCORED_FIRST_POLICY
    assert frozen["integrity_policy"] == "arena_integrity_v1"
    assert frozen["sourcing_cost_eligibility_policy"] == "successful_calls_per_icp_v1"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(round_id)["assignments"] == 10

    observed = _install_company_only_sandbox(harness)
    evidence = _ExecutionEvidence(harness, expected_high_water=1)
    harness.sandbox.run_icp = evidence.run_icp
    runner_label = "svc-runner-alpha"
    with TestClient(create_app(harness.service)) as http:
        runner = _http_runner(
            harness, http, tmp_path / "http-runner",
            key_label=runner_label, local_capacity=10,
        )
        try:
            _execute_exact(runner, harness, 9)
            before = harness.service.store.list_runs(round_id, stage=1, kind="execute")
            baseline_id = next(p["submission_id"] for p in harness.service.store.get_round(round_id)["participants"] if p["is_king"])
            assert len(before) == 10 and sum(r["status"] == "accepted" for r in before) == 9
            pending_position = next(r["icp_position"] for r in before if r["status"] == "pending")
            accepted_before = {r["run_id"]: (r["output_ref"], r["result_doc"])
                               for r in before if r["status"] == "accepted"}
            scheduled_stage1_close = harness.schedule()["stage_1_close"]
            _project_closed_stage_to_real_clock(connect, round_id, "stage_1_close")
            harness.clock.advance_to(scheduled_stage1_close)
            close1 = harness.service.advance_round(round_id)
            assert close1["status"] == "ok", close1
            assert harness.status() == "stage1_closed"
            stage1 = harness.service.store.get_round(round_id)["stage1_scoring_plan_doc"]
            assert stage1["incomplete_rows"] == [{
                "submission_id": baseline_id, "icp_position": pending_position,
                "cause": "stage_closed",
            }]
            assert len(stage1["work_items"]) == 9 and stage1["zero_rows"] == []
            _advance_with_http_scores(harness, runner, "stage2")
            assert len(harness.service.store.list_runs(round_id, stage=2, kind="execute")) == 20
            _execute_exact(runner, harness, 19)
            stage2_before = harness.service.store.list_runs(round_id, stage=2, kind="execute")
            accepted_before.update({
                r["run_id"]: (r["output_ref"], r["result_doc"])
                for r in stage2_before if r["status"] == "accepted"
            })
            incomplete_submission = next(r["submission_id"] for r in stage2_before if r["status"] == "pending")
            scheduled_stage2_close = harness.schedule()["stage_2_close"]
            _project_closed_stage_to_real_clock(connect, round_id, "stage_2_close")
            harness.clock.advance_to(scheduled_stage2_close)
            close2 = harness.service.advance_round(round_id)
            assert close2["status"] == "ok", close2
            assert harness.status() == "stage2_closed"
            stage2 = harness.service.store.get_round(round_id)["stage2_scoring_plan_doc"]
            assert len(stage2["incomplete_rows"]) == 1
            assert stage2["incomplete_rows"][0]["submission_id"] == incomplete_submission
            assert len(stage2["work_items"]) == 19 and stage2["zero_rows"] == []
            _advance_with_http_scores(harness, runner, "published")
        finally:
            runner.close()

    published = harness.service.store.get_round(round_id)
    assert published["status"] == "published"
    assert {key: value for key, value in published["configuration_doc"].items() if key != "schedule"} == {
        key: value for key, value in frozen.items() if key != "schedule"}
    assert published["publication_doc"]["king_decision"]["outcome"] == "no_king"
    ranking = {r["submission_id"]: r for r in published["publication_doc"]["final_ranking"]}
    assert ranking[baseline_id]["final_score"] is None
    assert ranking[incomplete_submission]["final_score"] is None
    for submission_id in (baseline_id, incomplete_submission):
        assert ranking[submission_id]["eligible"] is False
        assert ranking[submission_id]["eligibility_reason"] == "execution_incomplete"
        assert ranking[submission_id]["cost_summary"] is None
    full_submission = next(s for s in ranking if s not in (baseline_id, incomplete_submission))
    assert ranking[full_submission]["final_score"] is not None
    assert ranking[full_submission]["eligible"] is True
    assert evidence.successful_charged_calls == 28
    assert len(harness.service.store.list_runs(round_id, kind="score")) == 28
    assert all(r["status"] == "accepted" for r in harness.service.store.list_runs(round_id, kind="score"))
    after = harness.service.store.list_runs(round_id, kind="execute")
    for run in after:
        if run["run_id"] in accepted_before:
            assert (run["output_ref"], run["result_doc"]) == accepted_before[run["run_id"]]
    assert sum(bool(r["terminal_doc"] and r["terminal_doc"].get("infrastructure_incomplete") is True) for r in after) == 2
    assert sum(r["per_icp_score"] is not None for r in after) == 28
    assert harness.service.store.list_ledger(submission_id=full_submission)
