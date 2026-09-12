"""PostgreSQL journey for integrity scoring with delayed public details."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contracts, icp_disclosure, integrity, service as svc, verify
from lab_arena.promotion import GitPromoter
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_integrity_round import IntegrityHarness, MIGRATIONS


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(
        MIGRATIONS
        + (
            "223-lab-arena-cancelled-call-late-settlement.sql",
            "225-lab-arena-openrouter-delayed-cost-reconciliation.sql",
            "227-lab-arena-champion-funding.sql",
        )
    )


def test_integrity_round_finishes_on_day_one_and_reveals_details_on_day_two(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["DelayedWinner"], runners=["alpha"]
    )
    harness.chain.epoch = 27_000
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
        rewards_enabled=True,
    )

    def judge(companies, icp, reference):
        assert reference is False
        indexes, _ = verify.bucket_skip(icp, companies)
        rows = []
        for index in indexes:
            baseline = companies[index]["company_name"].startswith("PublicBaseline")
            score = (
                40.0
                if baseline
                else 60.0
                if str(icp["icp_id"]).startswith("confirmation_")
                else 80.0
            )
            rows.append(
                {
                    "final_score": score,
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

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    cutoff = harness.clock.now + timedelta(hours=12)
    round_id = "arena-2026-11-02-delayint"
    configuration = harness.service.create_round(cutoff, round_id=round_id)
    harness.round_id = round_id
    assert configuration["integrity_policy"] == integrity.POLICY
    assert (
        configuration["benchmark_disclosure_policy"]
        == icp_disclosure.DELAYED_DISCLOSURE_POLICY
    )
    winner = harness.submit("DelayedWinner", round_id)

    with pytest.raises(svc.ServiceError, match="benchmark_not_public") as hidden:
        harness.service.public_benchmark(round_id)
    assert hidden.value.status == 403
    with pytest.raises(svc.ServiceError, match="results_not_public"):
        harness.service.public_results(round_id, winner)
    with pytest.raises(svc.ServiceError, match="source_not_public"):
        harness.service.public_submission_code(winner)
    assert "icps" not in harness.service.public_round(round_id)

    scheduled_cutoff = datetime.fromisoformat(
        configuration["schedule"]["submission_cutoff"].replace("Z", "+00:00")
    )
    harness.clock.advance_to(configuration["schedule"]["submission_cutoff"])
    published = harness.advance_until("published")
    assert published["king_outcome"] == "crowned"
    assert len(harness.service.store.list_runs(round_id, stage=3, kind="execute")) == 10
    publication = published["publication_doc"]
    ranking = next(
        row for row in publication["final_ranking"]
        if row["submission_id"] == winner
    )
    assert ranking["main_score"] == 80.0
    assert ranking["final_score"] == 60.0
    assert harness.service.public_submission_code(winner)["files"]

    day_one_results = harness.service.public_results(round_id, winner)
    aggregate_scores = day_one_results["submission_scores"]
    assert aggregate_scores == {"stage_1": 80.0, "final": 60.0}
    assert day_one_results["public_icp_status"] == "pending"
    assert day_one_results["public_icp_count"] == 0
    assert day_one_results["outputs"] == {}
    assert day_one_results["run_results"] == []
    assert day_one_results["scores"] == {
        "stage_1": [],
        "stage_2": [],
        "confirmation": [],
    }
    with pytest.raises(svc.ServiceError, match="benchmark_not_public") as hidden:
        harness.service.public_benchmark(round_id)
    assert hidden.value.status == 403

    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok",
        "promoted": 1,
    }
    reward = harness.service.activate_reward(round_id)
    assert reward["status"] == "activated"
    day_one_stored = harness.service.store.get_round(round_id)
    preserved = {
        "promotion_doc": day_one_stored["promotion_doc"],
        "baseline_promoted_at": day_one_stored["baseline_promoted_at"],
        "reward_basis_doc": day_one_stored["reward_basis_doc"],
        "reward_activated_at": day_one_stored["reward_activated_at"],
        "effective_reward_epoch": day_one_stored["effective_reward_epoch"],
    }
    assert harness.service.public_reward_basis(
        int(day_one_stored["effective_reward_epoch"])
    ) == day_one_stored["reward_basis_doc"]

    # Existing rounds use their frozen policy after a service restart, even
    # when the new service would no longer opt future rounds into it.
    harness.service = harness.build_service()
    restarted_results = harness.service.public_results(round_id, winner)
    assert restarted_results == day_one_results
    assert harness.service.public_submission_code(winner)["files"]
    with pytest.raises(svc.ServiceError, match="benchmark_not_public"):
        harness.service.public_benchmark(round_id)

    reveal_at = scheduled_cutoff + timedelta(hours=24)
    harness.clock.now = reveal_at - timedelta(microseconds=1)
    assert harness.service.public_results(round_id, winner) == day_one_results
    with pytest.raises(svc.ServiceError, match="benchmark_not_public"):
        harness.service.public_benchmark(round_id)

    harness.clock.now = reveal_at
    benchmark = harness.service.public_benchmark(round_id)
    assert benchmark["disclosure_policy"] == (
        icp_disclosure.DELAYED_DISCLOSURE_POLICY
    )
    assert len(benchmark["icps"]) == contracts.BENCHMARK_ICP_COUNT
    assert len(benchmark["confirmation_bank"]["icps"]) == (
        contracts.CONFIRMATION_ICP_COUNT
    )
    assert contracts.document_hash(benchmark["confirmation_bank"]) == (
        published["confirmation_bank_hash"]
    )
    assert benchmark["private_icp_count"] == 0

    day_two_results = harness.service.public_results(round_id, winner)
    assert day_two_results["public_icp_status"] == "ready"
    assert day_two_results["public_icp_count"] == (
        contracts.MAX_EVALUATION_ICP_COUNT
    )
    assert day_two_results["submission_scores"] == aggregate_scores
    assert len(day_two_results["outputs"]) == contracts.MAX_EVALUATION_ICP_COUNT
    assert len(day_two_results["run_results"]) == (
        contracts.MAX_EVALUATION_ICP_COUNT
    )
    assert len(day_two_results["scores"]["stage_1"]) == contracts.STAGE_1_ICP_COUNT
    assert len(day_two_results["scores"]["stage_2"]) == contracts.STAGE_2_ICP_COUNT
    assert len(day_two_results["scores"]["confirmation"]) == (
        contracts.CONFIRMATION_ICP_COUNT
    )

    day_two_stored = harness.service.store.get_round(round_id)
    assert {
        key: day_two_stored[key] for key in preserved
    } == preserved
    assert harness.service.public_reward_basis(
        int(day_two_stored["effective_reward_epoch"])
    ) == preserved["reward_basis_doc"]
    fixtures.assert_canary_absent(harness, connect)
