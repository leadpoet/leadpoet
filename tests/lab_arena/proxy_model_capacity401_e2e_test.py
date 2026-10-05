"""Modern company-only round completes under the model execution claim gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from lab_arena import contracts
from lab_arena.api import create_app
from lab_arena.promotion import GitPromoter
from tests.lab_arena import proxy_model_capacity401_postgres_test as capacity401
from tests.lab_arena.company_only_capacity_e2e_test import (
    _drain_both,
    _http_runner,
    _install_company_only_sandbox,
    _postgres_lease_clock,
    _submit_without_review,
)
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_service_round import promotion_repository


database = capacity401.database
migrated399 = capacity401.migrated399
migrated = capacity401.migrated


def test_company_only_baseline_miners_costs_promotion_and_publication(
    database, migrated, tmp_path
):
    psycopg, dsn = database
    harness = QualityHarness(
        lambda: psycopg.connect(**dsn), tmp_path,
        challengers=["CapacityMinerA", "CapacityMinerB"],
        runners=["alpha", "beta"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        runner_slot_ceiling=20,
        promotion_margin=0.5,
        rewards_enabled=True,
        per_icp_cost_policy=True,
        contacts_from=None,
        company_quality_from=None,
        intent_details_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
    )
    bank = deepcopy(daily_icps()[:10])
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": deepcopy(bank),
    }
    observed = _install_company_only_sandbox(harness)
    harness.chain.epoch = 65_000
    harness.clock.now = datetime.now(timezone.utc) + timedelta(seconds=1)
    harness.round_id = "arena-2098-10-07-capacity"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30), round_id=harness.round_id
    )
    assert contracts.benchmark_icp_count(configuration) == 10
    assert configuration["execution_sequence_policy"] == (
        contracts.BASELINE_SCORED_FIRST_POLICY
    )
    submissions = [
        _submit_without_review(harness, flavor)
        for flavor in harness.challengers
    ]
    assert harness.service.review_pending_submissions()["reviewed"] == 2
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    committed = harness.service.store.get_round(harness.round_id)
    assert len(committed["participants"]) == 3
    baseline_id = next(
        row["submission_id"] for row in committed["participants"]
        if row["is_king"]
    )
    harness.flavors[baseline_id] = "PublicBaseline"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 10

    with TestClient(create_app(harness.service)) as http:
        small = _http_runner(
            harness, http, tmp_path / "validator-small",
            key_label="svc-runner-alpha", local_capacity=10,
        )
        large = _http_runner(
            harness, http, tmp_path / "validator-large",
            key_label="svc-runner-beta", local_capacity=20,
        )
        try:
            for _ in range(30):
                status = harness.status()
                if status == "published":
                    break
                if status in (
                    "stage1", "stage1_scoring", "stage2", "stage2_scoring"
                ):
                    _drain_both(harness, (small, large))
                if status == "stage1_scored":
                    assert harness.service._baseline_fully_succeeded(
                        harness.service.store.get_round(harness.round_id)
                    )
                    harness.clock.advance_to(harness.schedule()["stage_2_start"])
                transition = harness.service.advance_round(harness.round_id)
                assert transition.get("status") not in (
                    "cancelled", "terminal", "retry", "stale"
                ), (status, transition)
            else:
                raise AssertionError("round did not publish")
        finally:
            small.close()
            large.close()

    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert observed == {"execute": 30, "score": 30}
    assert len(published["publication_doc"]["final_ranking"]) == 3
    for entry in published["publication_doc"]["final_ranking"]:
        assert entry["eligible"]
        cost = entry["cost_summary"]
        assert len(cost["per_icp"]) == 10
        assert cost["competition_sourcing_microusd"] == cost["execution"][
            "successful_microusd"
        ]
    assert published["king_outcome"] == "crowned"
    winner = published["publication_doc"]["king_decision"]["winner_submission_id"]
    assert winner in submissions
    remote_root = tmp_path / "promotion"
    remote_root.mkdir()
    remote = promotion_repository(remote_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok", "promoted": 1,
    }
