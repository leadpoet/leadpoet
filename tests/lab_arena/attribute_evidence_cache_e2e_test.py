"""Old accepted baseline and new hint-bound cached judgments publish normally."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from lab_arena import contracts
from lab_arena.api import create_app
from lab_arena.promotion import GitPromoter
from tests.lab_arena.company_only_capacity_e2e_test import (
    _drain_both,
    _http_runner,
    _install_company_only_sandbox,
    _submit_without_review,
)
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_service_round import promotion_repository


import json
import os
from lab_arena import broker, runtime, scoring, shim
from leadpoet_canonical import arena_weights
from qualification.scoring import competition
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_cost_zero_round_test import _accepted_state
from tests.lab_arena.idle_worker_claims416_postgres_test import database, migrated


def _install_hints(harness):
    counts = _install_company_only_sandbox(harness)
    original = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        result = original(spec, **kwargs)
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            # A fresh judge has a confirmed local provider charge. Cached
            # assignments do not enter this sandbox or create this ledger row.
            with harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    status, _, _ = shim.dispatch(
                        "openrouter.chat",
                        {
                            "model": "google/gemini-2.5-flash",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "cache-cost-fixture "
                                    + document["scored_run_id"],
                                }
                            ],
                        },
                        10,
                    )
                    assert status == 200
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            return result
        output = json.loads(result.output_bytes)
        is_second = any(
            "CapacityMinerB" in c["company_name"] for c in output["companies"]
        )
        if is_second:
            output = json.loads(
                json.dumps(output)
                .replace("CapacityMinerB", "CapacityMinerA")
                .replace("capacityminerb", "capacityminera")
            )
        position = int(document["icp"]["icp_id"].rsplit("_", 1)[-1]) - 1
        for company in output["companies"]:
            url = company["company_website"].rstrip("/") + "/subscription"
            quote = company["company_name"] + " sells a paid platform subscription."
            if is_second and position % 3 == 1:
                quote += " Expanded plan evidence."
            if is_second and position % 3 == 2:
                url += "/different"
            company["required_attribute"] = {
                "text": "Sells a subscription platform",
                "passed": False,
                "evidence_url": url,
                "evidence_quote": quote,
                "explanation": "Untrusted source hint checked independently",
            }
        return runtime.fake_result(
            exit_code=0, output_bytes=json.dumps(output).encode()
        )

    harness.sandbox.run_icp = run_icp
    return counts


def test_cached_judgments_and_old_baseline_publish_after_projection_upgrade(
    database, migrated, tmp_path, monkeypatch
):
    original_send = fixtures.FakeProviderTransport.send

    def priced_local_response(self, **kwargs):
        request = json.loads(kwargs.get("body") or b"{}")
        messages = request.get("messages") or []
        if messages and str(messages[0].get("content", "")).startswith(
            "cache-cost-fixture "
        ):
            return broker.ProviderResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": "fixture"}}
                        ],
                        "usage": {"cost": "0.000010"},
                    }
                ).encode(),
            )
        return original_send(self, **kwargs)

    monkeypatch.setattr(fixtures.FakeProviderTransport, "send", priced_local_response)
    psycopg, dsn = database
    harness = QualityHarness(
        lambda: psycopg.connect(**dsn),
        tmp_path,
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
        "status": "ready",
        "set_id": int(kwargs["set_id"]),
        "icps": deepcopy(bank),
    }
    current_projection = competition.effective_competition_input

    def legacy_projection(*args, **kwargs):
        effective = current_projection(*args, **kwargs)
        for row in effective["companies"]:
            row.pop("required_attribute", None)
        return effective

    monkeypatch.setattr(competition, "effective_competition_input", legacy_projection)
    observed = _install_hints(harness)
    old_baseline = {}
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
        _submit_without_review(harness, flavor) for flavor in harness.challengers
    ]
    assert harness.service.review_pending_submissions()["reviewed"] == 2
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    committed = harness.service.store.get_round(harness.round_id)
    assert len(committed["participants"]) == 3
    baseline_id = next(
        row["submission_id"] for row in committed["participants"] if row["is_king"]
    )
    harness.flavors[baseline_id] = "PublicBaseline"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 10

    with TestClient(create_app(harness.service)) as http:
        small = _http_runner(
            harness,
            http,
            tmp_path / "validator-small",
            key_label="svc-runner-alpha",
            local_capacity=10,
        )
        large = _http_runner(
            harness,
            http,
            tmp_path / "validator-large",
            key_label="svc-runner-beta",
            local_capacity=20,
        )
        try:
            for _ in range(30):
                status = harness.status()
                if status == "published":
                    break
                if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
                    _drain_both(harness, (small, large))
                if status == "stage1_scored":
                    assert harness.service._baseline_fully_succeeded(
                        harness.service.store.get_round(harness.round_id)
                    )
                    old_baseline = {
                        r["run_id"]: (
                            r["judgment_cache_key"],
                            r["judgment_scope_doc"],
                            r["result_doc"],
                            r["output_ref"],
                        )
                        for r in harness.service.store.list_runs(
                            harness.round_id, stage=1, kind="score"
                        )
                        if r["status"] == "accepted"
                    }
                    assert len(old_baseline) == 10
                    monkeypatch.setattr(
                        competition, "effective_competition_input", current_projection
                    )
                    harness.clock.advance_to(harness.schedule()["stage_2_start"])
                transition = harness.service.advance_round(harness.round_id)
                assert transition.get("status") not in (
                    "cancelled",
                    "terminal",
                    "retry",
                    "stale",
                ), (status, transition)
            else:
                raise AssertionError("round did not publish")
        finally:
            small.close()
            large.close()

    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert observed == {"execute": 30, "score": 26}
    score_runs = harness.service.store.list_runs(harness.round_id, kind="score")
    cached = [r for r in score_runs if r["judgment_cache_source_run_id"] != r["run_id"]]
    assert len(cached) == 4
    for row in cached:
        assert harness.service.store.list_ledger(run_id=row["run_id"]) == []
    assert any(
        harness.service.store.list_ledger(run_id=row["run_id"])
        for row in score_runs
        if row not in cached
    )
    for run_id, saved in old_baseline.items():
        row = harness.service.store.get_run(run_id)
        assert row["status"] == "accepted"
        assert (
            row["judgment_cache_key"],
            row["judgment_scope_doc"],
            row["result_doc"],
            row["output_ref"],
        ) == saved
    assert old_baseline
    assert len(published["publication_doc"]["final_ranking"]) == 3
    for entry in published["publication_doc"]["final_ranking"]:
        assert entry["eligible"]
        cost = entry["cost_summary"]
        assert len(cost["per_icp"]) == 10
        assert (
            cost["competition_sourcing_microusd"]
            == cost["execution"]["successful_microusd"]
        )
        assert cost["judge"]["successful_microusd"] > 0
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
        "status": "ok",
        "promoted": 1,
    }
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    activated = harness.service.store.get_round(harness.round_id)
    winner_hotkey = activated["reward_basis_doc"]["king_hotkey"]
    burn = fixtures.keypair("attribute-evidence-cache-burn").ss58_address
    state = _accepted_state(harness, int(activated["effective_reward_epoch"]), burn)
    weights = arena_weights.derive_arena_weights(state, [burn, winner_hotkey])
    assert weights["champion_share_ppb"] == 300_000_000
    assert weights["burned_residual_ppb"] == 700_000_000
    fixtures.assert_canary_absent(harness, lambda: database[0].connect(**database[1]))
