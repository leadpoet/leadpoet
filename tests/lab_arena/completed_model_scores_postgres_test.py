"""Completed models become public while other models still await judging."""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from lab_arena.service import ServiceError
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
    _verified_test_pool,
)
from tests.lab_arena.completed_model_scores_test import _wait_for_scores
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_integrity_round import IntegrityHarness


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_completed_scores_and_diagnostics_precede_round_publication(database, tmp_path, monkeypatch):
    psycopg2, dsn = database
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["Best", "Weak"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults, execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    _enable_verified_proxy_runtime(harness)
    def fractional_judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        for item in rows:
            # More precision than the persisted NUMERIC(12, 6) score, with a
            # miner ahead of the baseline before the round can pick a winner.
            baseline_company = companies[item["company_index"]]["company_name"].startswith("PublicBaseline")
            item["final_score"] = 40.123456789 if baseline_company else 80.234567891
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", fractional_judge)
    fixtures._start_round(harness, day=31, epoch=62_031)
    service = harness.service
    rid = harness.round_id
    baseline = next(p["submission_id"] for p in service.store.get_round(rid)["participants"] if p["is_king"])
    harness.advance_until("stage1_scoring", runners=2)
    assert service.completed_submission_scores(service._round(rid)) == {}
    with pytest.raises(ServiceError, match="results_not_public"):
        service.public_results(rid, baseline)

    harness.run_stage_with_runners(2)
    service._completed_scores_cache.clear()
    before = service.store.list_runs(rid)
    completed = _wait_for_scores(service, service._round(rid), lambda scores: baseline in scores, attempts=6000)
    assert set(completed) == {baseline}
    assert completed[baseline]["final_score"] > 0
    # Public reads perform no scoring writes, transition, promotion or reward.
    assert service.store.list_runs(rid) == before
    assert all(r["per_icp_score"] is None for r in before)
    summary = service.public_competition()["latest_round"]
    assert summary["baseline"]["final_score"] == completed[baseline]["final_score"]
    assert summary["champion"] is None
    cards = service.public_submissions(rid)["submissions"]
    assert next(c for c in cards if c["is_baseline"])["status"] == "scored"
    assert all(c["final_score"] is None for c in cards if not c["is_baseline"])
    result = service.public_results(rid, baseline)
    assert result["score_status"] == "complete"
    assert result["submission"]["is_baseline"] is True
    assert result["submission_scores"]["final"] == completed[baseline]["final_score"]
    assert len(result["scores"]["stage_1"]) == 20
    assert result["company_diagnostics"]

    harness.advance_until("stage2_scoring", runners=2)
    # The first completed miner is visible without closing all miner judging.
    scheduled_now = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    runner = fixtures.Harness.runner(harness, 0, parallel=1)
    runner._config.proxy_worker_pool = _verified_test_pool(2)
    early = {}
    plan = service._round(rid)["stage2_scoring_plan_doc"]
    miner_items = {}
    for item in plan["work_items"]:
        miner_items.setdefault(item["submission_id"], set()).add(item["scored_run_id"])
    try:
        for _ in range(100):
            assert runner.run_once(max_claims=1)
            chosen = service._scoring_outputs(rid, 2)
            if any(all(chosen.get(run_id, {}).get("status") == "accepted" for run_id in run_ids)
                   for run_ids in miner_items.values()):
                break
    finally:
        runner.close()
        harness.clock.now = scheduled_now
    service._completed_scores_cache.clear()
    early = _wait_for_scores(service, service._round(rid), lambda scores: len(scores) > 1, attempts=6000)
    assert len(early) == 2
    pending = [r for r in service.store.list_runs(rid, kind="score", stage=2) if r["status"] != "accepted"]
    assert pending
    miner = next(s for s in early if s != baseline)
    assert service.public_results(rid, miner)["submission_scores"]["final"] == early[miner]["final_score"]
    for card in service.public_submissions(rid)["submissions"]:
        assert card["is_champion"] is False
        assert (card["final_score"] is not None) == (card["submission_id"] in early)
    row = service._round(rid)
    assert row["status"] == "stage2_scoring"
    assert row["publication_doc"] is None
    assert row["reward_activated_at"] is None

    # Frozen final publication must agree with the early completed score.
    harness.advance_until("published", runners=2, max_steps=120)
    row = service._round(rid)
    final = {r["submission_id"]: r["final_score"] for r in row["publication_doc"]["final_ranking"]}
    for sid, entry in early.items():
        assert entry["final_score"] == final[sid]
    assert row["publication_doc"]["king_decision"]["outcome"] == "crowned"
    assert service.completed_submission_scores(row) == {}
    assert service.public_results(rid, miner)["score_status"] == "published"
