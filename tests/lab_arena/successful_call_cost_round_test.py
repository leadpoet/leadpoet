"""Production round transitions with a controlled, charged provider failure."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from lab_arena import broker as br, contracts, runtime, scoring, shim, submission_runtime
from lab_arena.promotion import GitPromoter
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration


class ChargedTransport(fixtures.FakeProviderTransport):
    def send(self, **kwargs):
        body = kwargs.get("body")
        request = json.loads(body) if body else {}
        if request.get("model") in fixtures.PRICED_MODELS:
            prompt = request["messages"][0]["content"]
            if prompt == "judge transport failure":
                raise br.ProviderTransportError("controlled lost response")
            failed = prompt == "charged provider failure"
            document = {
                "model": request["model"],
                "usage": {"cost": "0.000020" if failed else "0.000010"},
            }
            if failed:
                document["error"] = {"code": 503, "message": "Unavailable"}
            else:
                document["choices"] = [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]
            return br.ProviderResponse(503 if failed else 200, {"content-type": "application/json"}, json.dumps(document).encode())
        return super().send(**kwargs)


class CostHarness(fixtures.Harness):
    def objects_key(self):
        return "successful-call-cost-" + self.tmp.name

    def build_service(self):
        service = super().build_service()
        service.config.defaults = replace(service.config.defaults, cost_per_company_microusd=25, rewards_enabled=True)
        payer = submission_runtime.SubmissionProviderKeys(
            store=service.store, credentials=fixtures.FakeCredentialManager(), organizer_keys=fixtures.CANARY_KEYS,
        )
        transport = ChargedTransport()
        service.config.broker_factory = lambda _service, _row: br.Broker(
            store=service.store,
            key_for=lambda provider: fixtures.CANARY_KEYS[provider],
            credential_for=payer.credential_for,
            funding_source_for=payer.funding_source_for,
            price_table=fixtures.price_table(),
            judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
            transport=transport,
            clock=self.clock,
        )
        return service


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_successful_call_cost_survives_publication_restart_and_promotion(database, tmp_path):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = CostHarness(connect, tmp_path, challengers=["Bravo"], runners=["alpha"])
    original = harness.sandbox.run_icp
    crashed_once = set()

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        is_score = document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION
        prompts = ["judge transport failure"] if is_score else ["successful empty completion", "charged provider failure"]
        # The real shim/socket, gateway broker and SQL ledger handle each call.
        # Only the external provider boundary is controlled in this test.
        with harness.sandbox.lock:
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for prompt in prompts:
                    status, _, _ = shim.dispatch("openrouter.chat", {
                        "model": next(iter(scoring.DEFAULT_JUDGE_MODELS.values())) if is_score else "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                    }, 5000)
                    assert status == (200 if prompt == "successful empty completion" else 502)
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            if not is_score:
                submission_id = spec.source_dir.parent.name.removeprefix("submission-")
                if harness.flavors.get(submission_id) == "Bravo" and not crashed_once:
                    crashed_once.add(submission_id)
                    return runtime.fake_result(exit_code=1, output_bytes=None, stderr=b"submitted model crashed after completed API call")
        return original(spec, **kwargs)

    harness.sandbox.run_icp = run_icp
    participants = fixtures._start_round(harness, day=20, epoch=24900)
    config = harness.service.store.get_round(harness.round_id)["configuration_doc"]
    assert config["sourcing_cost_eligibility_policy"] == "successful_calls_v1"
    fixtures._run_stage_one_to_scoring(harness, participants, runners=1)
    harness.service = harness.build_service()
    assert harness.service.store.get_round(harness.round_id)["configuration_doc"] == config
    harness.advance_until("published", runners=1)

    row = harness.service.store.get_round(harness.round_id)
    for result in row["publication_doc"]["final_ranking"]:
        assert result["eligible"] is True, json.dumps(result, sort_keys=True)
        costs = result["cost_summary"]
        attempts = 21 if result["submission_id"] in crashed_once else 20
        assert costs["execution"]["settled_microusd"] == attempts * 30
        assert costs["competition_sourcing_microusd"] == attempts * 10
        assert costs["execution"]["successful_microusd"] == attempts * 10
        assert costs["eligibility_cap_microusd"] == 500
        assert costs["judge"]["uncertain_calls"] == 20
        assert costs["execution"]["settled_microusd"] > costs["eligibility_cap_microusd"]
        public = harness.service.public_results(harness.round_id, result["submission_id"])
        assert len(public["scores"]["stage_1"] + public["scores"]["stage_2"]) == 20
    assert len(crashed_once) == 1

    assert row["king_outcome"] == "crowned"
    remote_root = tmp_path / "promotion"
    remote_root.mkdir()
    remote = fixtures.promotion_repository(remote_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-objects")
    harness.clock.now = datetime.fromisoformat(row["published_at"].replace("Z", "+00:00"))
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    assert subprocess.check_output(["git", "--git-dir", str(remote), "show", "lab:flavor.txt"]).decode() == "Bravo"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    activated = harness.service.store.get_round(harness.round_id)
    assert activated["reward_basis_doc"]["king_hotkey"] == row["king_hotkey"]
    assert harness.service.activate_reward(harness.round_id)["status"] == "existing"
    fixtures.assert_canary_absent(harness, connect)


@pytest.mark.parametrize("with_contacts", [False, True])
def test_new_cost_policy_preserves_quality_and_contact_publication(database, tmp_path, with_contacts):
    from tests.lab_arena.company_quality_round_test import test_quality_round_publishes_coverage_winner_after_restart
    test_quality_round_publishes_coverage_winner_after_restart(database, tmp_path, with_contacts)
