"""Future baseline-first rounds through the real service and PostgreSQL."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import capacity, contracts, public_dashboard, verify
from leadpoet_canonical import arena_weights
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_cost_zero_round_test import _accepted_state
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
    )


def _baseline_first_judge(companies, icp, reference):
    assert reference is False
    indexes, _ = verify.bucket_skip(icp, companies)
    rows = []
    for index in indexes:
        baseline = companies[index]["company_name"].startswith("PublicBaseline")
        rows.append(
            {
                "final_score": 80.0 if baseline else 10.0,
                "company_index": index,
                "company_identity_key": canonical_company_identity(
                    companies[index]
                ).key,
                "company_qualified": True,
                "duplicate_company": False,
                "verifier_gate_receipts": [
                    {"gate": "company_fit", "decision": "match"}
                ],
                "intent_signals_detail": [],
                "failure_reason": "",
            }
        )
    return rows


def test_baseline_is_fully_scored_before_five_miners_execute_and_publish(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    flavors = ["Miner%d" % index for index in range(5)]
    harness = IntegrityHarness(
        connect, tmp_path, challengers=flavors, runners=["alpha", "beta"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
        rewards_enabled=True,
    )

    # The runtime migration is safe to replay after it has already patched
    # the installed integrity, scoring, and transition functions.
    migration = Path("scripts/347-lab-arena-baseline-scored-first.sql").read_text()
    with connect() as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)

    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    participants = fixtures._start_round(harness, day=31, epoch=62_031)
    assert participants == 6
    committed = harness.service.store.get_round(harness.round_id)
    configuration = committed["configuration_doc"]
    assert configuration["execution_sequence_policy"] == (
        contracts.BASELINE_SCORED_FIRST_POLICY
    )
    assert "parallel_twenty_icp_execution" not in configuration
    assert capacity.daily_challenger_capacity(configuration) >= len(flavors)
    bank_ref = committed["benchmark_ref"]
    bank_bytes = harness.objects.get(bank_ref)
    baseline_id = next(
        item["submission_id"] for item in committed["participants"]
        if item["is_king"]
    )
    miner_ids = {
        item["submission_id"] for item in committed["participants"]
        if not item["is_king"]
    }

    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(harness.round_id)
    assert opened["assignments"] == contracts.BENCHMARK_ICP_COUNT
    baseline_runs = harness.service.store.list_runs(
        harness.round_id, kind="execute"
    )
    assert {run["submission_id"] for run in baseline_runs} == {baseline_id}
    assert {run["icp_position"] for run in baseline_runs} == set(range(20))
    assert harness.service.store.list_runs(
        harness.round_id, stage=2, kind="execute"
    ) == []

    harness.advance_until("stage1_scored", runners=2)
    scored_baseline = harness.service.store.list_runs(
        harness.round_id, stage=1, kind="execute"
    )
    assert len(scored_baseline) == 20
    assert all(run["per_icp_score"] is not None for run in scored_baseline)
    assert harness.service._baseline_fully_succeeded(
        harness.service.store.get_round(harness.round_id)
    )
    assert harness.service.store.list_runs(
        harness.round_id, stage=2, kind="execute"
    ) == []

    opened = harness.service.advance_round(harness.round_id)
    assert opened["assignments"] == 5 * contracts.BENCHMARK_ICP_COUNT
    miner_runs = harness.service.store.list_runs(
        harness.round_id, stage=2, kind="execute"
    )
    assert {run["submission_id"] for run in miner_runs} == miner_ids
    assert all(
        sum(run["submission_id"] == submission_id for run in miner_runs) == 20
        for submission_id in miner_ids
    )
    assert harness.service.store.get_round(harness.round_id)["status"] == "stage2"

    harness.advance_until("published", runners=2, max_steps=120)
    published = harness.service.store.get_round(harness.round_id)
    assert published["benchmark_ref"] == bank_ref
    assert harness.objects.get(bank_ref) == bank_bytes
    publication = published["publication_doc"]
    assert len(publication["stage1_ranking"]) == 5
    assert {row["submission_id"] for row in publication["stage1_ranking"]} == miner_ids
    assert publication["finalists"] == sorted(miner_ids)
    assert len(publication["final_ranking"]) == 6
    assert publication["king_decision"]["outcome"] == "no_king"
    dashboard = public_dashboard.submissions_snapshot(
        harness.service, harness.round_id
    )
    dashboard_scores = {
        item["submission_id"]: item["stage1_score"]
        for item in dashboard["submissions"]
    }
    assert set(dashboard_scores) == miner_ids | {baseline_id}
    assert all(score is not None for score in dashboard_scores.values())
    for result in publication["final_ranking"]:
        summary = result["cost_summary"]
        assert len(summary["per_icp"]) == 20
        assert summary["competition_sourcing_microusd"] == summary["execution"][
            "successful_microusd"
        ]
        assert summary["judge"]["call_count"] > 0
        public_result = harness.service.public_results(
            harness.round_id, result["submission_id"]
        )
        assert public_result["submission_scores"]["final"] == result["final_score"]
        expected_stage = "stage_1" if result["submission_id"] == baseline_id else "stage_2"
        assert len(public_result["scores"][expected_stage]) == 20

    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    activated = harness.service.store.get_round(harness.round_id)
    assert activated["reward_basis_doc"]["king_outcome"] == "no_king"
    burn = fixtures.keypair("baseline-first-burn").ss58_address
    state = _accepted_state(
        harness, int(activated["effective_reward_epoch"]), burn
    )
    weights = arena_weights.derive_arena_weights(state, [burn])
    assert weights["champion_share_ppb"] == 0
    assert weights["burned_residual_ppb"] == 1_000_000_000


def test_baseline_only_future_round_scores_and_publishes_without_stage_two_runs(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=[], runners=["alpha", "beta"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    assert fixtures._start_round(harness, day=30, epoch=62_030) == 1

    harness.advance_until("published", runners=2, max_steps=120)
    row = harness.service.store.get_round(harness.round_id)
    baseline_id = row["participants"][0]["submission_id"]
    assert len(
        harness.service.store.list_runs(
            harness.round_id, stage=1, kind="execute"
        )
    ) == contracts.BENCHMARK_ICP_COUNT
    assert harness.service.store.list_runs(
        harness.round_id, stage=2, kind="execute"
    ) == []
    assert harness.service.store.list_runs(
        harness.round_id, stage=2, kind="score"
    ) == []
    assert row["publication_doc"]["stage1_ranking"] == []
    assert row["publication_doc"]["finalists"] == []
    assert [
        item["submission_id"]
        for item in row["publication_doc"]["final_ranking"]
    ] == [baseline_id]
    assert row["publication_doc"]["king_decision"]["outcome"] == "no_king"


def test_sep22_open_round_opt_in_changes_only_the_execution_policy(
    database, tmp_path
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=[], runners=["alpha", "beta"]
    )
    harness.round_id = "arena-2026-09-22"
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from=None,
        parallel_twenty_icp_execution=True,
    )
    harness.service.create_round(
        datetime(2026, 9, 22, tzinfo=timezone.utc),
        round_id=harness.round_id,
    )
    before = harness.service.store.get_round(harness.round_id)
    assert before["configuration_doc"]["parallel_twenty_icp_execution"] is True

    migration = Path(
        "scripts/348-arena-2026-09-22-baseline-scored-first.sql"
    ).read_text()
    with connect() as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
    after = harness.service.store.get_round(harness.round_id)
    expected_configuration = dict(before["configuration_doc"])
    expected_configuration.pop("parallel_twenty_icp_execution")
    expected_configuration["execution_sequence_policy"] = (
        contracts.BASELINE_SCORED_FIRST_POLICY
    )
    assert after["configuration_doc"] == expected_configuration
    assert {
        key: value for key, value in after.items()
        if key not in {"configuration_doc", "updated_at"}
    } == {
        key: value for key, value in before.items()
        if key not in {"configuration_doc", "updated_at"}
    }
