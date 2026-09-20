"""Full-round proof for the baseline cost-eligibility consequence.

The provider and chain boundaries are controlled.  The service, runner/shim,
broker, PostgreSQL ledger, scorer, publication, Git promotion, reward signing,
and accepted weight-state paths are the production implementations.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import broker as br, runtime, scoring, shim, submission_runtime, verify
from lab_arena.promotion import GitPromoter
from leadpoet_canonical import arena_weights
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena import test_integrity_round as integrity_fixtures
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


class EligibilityCostTransport(fixtures.FakeProviderTransport):
    """Return explicit successful and failed charges at the provider edge."""

    def send(self, **kwargs):
        body = kwargs.get("body")
        request = json.loads(body) if body else {}
        if request.get("model") in fixtures.PRICED_MODELS:
            prompt = request["messages"][0]["content"]
            if prompt == "cost-test unsuccessful call":
                return br.ProviderResponse(
                    503,
                    {"content-type": "application/json"},
                    json.dumps(
                        {
                            "model": request["model"],
                            "usage": {"cost": "0.000020"},
                            "error": {"code": 503, "message": "controlled failure"},
                        }
                    ).encode(),
                )
            cost = (
                "0.000030"
                if prompt == "cost-test expensive successful empty"
                else "0.000010"
            )
            return br.ProviderResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "model": request["model"],
                        "usage": {"cost": cost},
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": ""},
                            }
                        ],
                    }
                ).encode(),
            )
        return super().send(**kwargs)


class BaselineCostHarness(integrity_fixtures.IntegrityHarness):
    """One isolated Arena whose call costs vary by frozen participant."""

    def __init__(self, *args, **kwargs):
        self.expensive_baseline = False
        self.expensive_flavors: set[str] = set()
        super().__init__(*args, **kwargs)
        self._install_cost_calls()

    def objects_key(self):
        return "baseline-cost-zero-" + self.tmp.name

    def build_service(self):
        service = super().build_service()
        service.config.defaults = replace(
            service.config.defaults,
            cost_per_company_microusd=5,
            rewards_enabled=True,
        )
        payer = submission_runtime.SubmissionProviderKeys(
            store=service.store,
            credentials=fixtures.FakeCredentialManager(),
            organizer_keys=fixtures.CANARY_KEYS,
        )
        transport = EligibilityCostTransport()
        service.config.broker_factory = lambda _service, _row: br.Broker(
            store=service.store,
            key_for=lambda provider: fixtures.CANARY_KEYS[provider],
            credential_for=payer.credential_for,
            funding_source_for=payer.funding_source_for,
            provider_funding_source_for=payer.provider_funding_source_for,
            retry_miner_credential_for=payer.retry_miner_credential_for,
            mark_provider_fallback=payer.mark_provider_fallback,
            provider_restart_required_for=payer.provider_restart_required_for,
            price_table=fixtures.price_table(),
            judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
            transport=transport,
            clock=self.clock,
        )
        return service

    def _install_cost_calls(self):
        original = self.sandbox.run_icp

        def run_icp(spec, **kwargs):
            input_document = json.loads(
                (spec.input_dir / runtime.INPUT_FILE_NAME).read_text()
            )
            if input_document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
                submission_id = spec.source_dir.parent.name.removeprefix("submission-")
                submission = self.service.store.get_submission(submission_id)
                flavor = self.flavors.get(submission_id)
                if flavor is None:
                    flavor = (spec.source_dir / "flavor.txt").read_text(encoding="utf-8")
                expensive = bool(submission.get("is_king")) and self.expensive_baseline
                expensive = expensive or flavor in self.expensive_flavors
                successful_prompt = (
                    "cost-test expensive successful empty"
                    if expensive
                    else "cost-test efficient successful empty"
                )
                with self.sandbox.lock:
                    os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                    try:
                        successful, _, _ = shim.dispatch(
                            "openrouter.chat",
                            {
                                "model": "openai/gpt-4o-mini",
                                "messages": [{"role": "user", "content": successful_prompt}],
                                "max_tokens": 100,
                            },
                            5000,
                        )
                        failed, _, _ = shim.dispatch(
                            "openrouter.chat",
                            {
                                "model": "openai/gpt-4o-mini",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "cost-test unsuccessful call",
                                    }
                                ],
                                "max_tokens": 100,
                            },
                            5000,
                        )
                    finally:
                        os.environ.pop(shim.WORKER_SOCKET_ENV, None)
                assert successful == 200
                assert failed == 502
            return original(spec, **kwargs)

        self.sandbox.run_icp = run_icp


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture
def fixed_scores(monkeypatch):
    scores = {
        "PublicBaseline": 5.0,
        "EfficientWinner": 2.0,
        "PriorWinner": 10.0,
        "LowEligible": 0.5,
        "ExpensiveHigh": 99.0,
    }

    def deterministic_scorer(companies, icp, is_reference_model):
        assert is_reference_model is False
        flavor = str(companies[0]["company_name"]).split(" Company ", 1)[0]
        indexes, _ = verify.bucket_skip(icp, companies)
        return [
            {
                "final_score": scores[flavor],
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
            for index in indexes
        ]

    monkeypatch.setattr(fixtures, "deterministic_scorer", deterministic_scorer)
    return scores


def _ranking_by_id(round_row):
    return {
        row["submission_id"]: row
        for row in round_row["publication_doc"]["final_ranking"]
    }


def _baseline_participant(round_row):
    return next(row for row in round_row["participants"] if row["is_king"])


def _submission_for_flavor(harness, flavor):
    submission_id = next(
        submission_id
        for submission_id, candidate in harness.flavors.items()
        if candidate == flavor
    )
    return harness.service.store.get_submission(submission_id)


def _assert_cost_evidence(row, *, successful_microusd, eligible, reason):
    assert row["eligible"] is eligible
    assert row["eligibility_reason"] == reason
    costs = row["cost_summary"]
    assert costs["qualified_company_count"] == 100
    assert costs["eligibility_cap_microusd"] == 500
    assert costs["competition_sourcing_microusd"] == successful_microusd
    assert costs["execution"]["successful_microusd"] == successful_microusd
    # The controlled failed call was billed and remains in total accounting,
    # but is excluded from successful-call promotion spend.
    assert costs["execution"]["settled_microusd"] == successful_microusd + 400


def _accepted_state(harness, epoch, burn_hotkey):
    harness.chain.epoch = epoch
    harness.chain.accepted_weight_epoch_scope = lambda: {
        "genesis_hash": "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
        "epoch": epoch,
        "valid_from_block": 100,
        "valid_until_block": 459,
    }
    harness.service.config.accepted_burn_hotkey = burn_hotkey
    state = harness.service.public_weight_state(epoch)["state"]
    restarted = harness.build_service()
    restarted.config.accepted_burn_hotkey = burn_hotkey
    assert restarted.public_weight_state(epoch)["state"] == state
    assert arena_weights.verify_accepted_weight_state_signature(
        state,
        public_key_der=harness.signer.public_key_der,
        expected_public_key_hash=harness.signer.public_key_hash,
    ) == state["state_hash"]
    return state


def test_cost_ineligible_baseline_scores_zero_and_eligible_challenger_carries_downstream_authority(
    database, tmp_path, fixed_scores
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = BaselineCostHarness(
        connect, tmp_path, challengers=["EfficientWinner"], runners=["alpha"]
    )
    harness.expensive_baseline = True
    participants = fixtures._start_round(harness, day=3, epoch=41_000)
    fixtures._run_stage_one_to_scoring(harness, participants, runners=1)
    harness.advance_until("published", runners=1)

    published = harness.service.store.get_round(harness.round_id)
    baseline = _baseline_participant(published)
    winner = _submission_for_flavor(harness, "EfficientWinner")
    ranking = _ranking_by_id(published)
    baseline_result = ranking[baseline["submission_id"]]
    winner_result = ranking[winner["submission_id"]]
    _assert_cost_evidence(
        baseline_result,
        successful_microusd=600,
        eligible=False,
        reason="cost_per_company_exceeded",
    )
    _assert_cost_evidence(
        winner_result,
        successful_microusd=200,
        eligible=True,
        reason="eligible",
    )
    assert baseline_result["final_score"] == 0.0
    assert winner_result["final_score"] == fixed_scores["EfficientWinner"]
    assert published["publication_doc"]["king_decision"] == {
        "outcome": "crowned",
        "king_submission_id": winner["submission_id"],
        "king_hotkey": winner["miner_hotkey"],
        "winner_submission_id": winner["submission_id"],
    }

    baseline_runs = [
        row
        for row in harness.service.store.list_runs(
            harness.round_id,
            submission_id=baseline["submission_id"],
            kind="execute",
        )
        if row["status"] == "accepted"
    ]
    assert len(baseline_runs) == 20
    assert {row["per_icp_score"] for row in baseline_runs} == {
        fixed_scores["PublicBaseline"]
    }
    public_baseline = harness.service.public_results(
        harness.round_id, baseline["submission_id"]
    )
    assert public_baseline["submission_scores"]["final"] == 0.0
    assert {
        score["per_icp_score"]
        for score in public_baseline["scores"]["stage_1"]
        + public_baseline["scores"]["stage_2"]
    } == {fixed_scores["PublicBaseline"]}

    # Publication is write-once and survives a fresh service instance.
    restarted = harness.build_service()
    assert restarted.public_round(harness.round_id)["publication"] == published[
        "publication_doc"
    ]

    promotion_root = tmp_path / "winner-promotion"
    promotion_root.mkdir()
    remote = fixtures.promotion_repository(promotion_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "winner-promotion-cache"
    )
    harness.clock.now = datetime.fromisoformat(
        published["published_at"].replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok",
        "promoted": 1,
    }
    assert (
        subprocess.check_output(
            ["git", "--git-dir", str(remote), "show", "lab:flavor.txt"]
        ).decode()
        == "EfficientWinner"
    )
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    activated = harness.service.store.get_round(harness.round_id)
    assert activated["reward_basis_doc"]["king_hotkey"] == winner["miner_hotkey"]

    burn = fixtures.keypair("baseline-cost-zero-burn-one").ss58_address
    reward_epoch = int(activated["effective_reward_epoch"])
    state = _accepted_state(harness, reward_epoch, burn)
    assert state["reward_basis"]["king_hotkey"] == winner["miner_hotkey"]
    derived = arena_weights.derive_arena_weights(
        state, [burn, winner["miner_hotkey"]]
    )
    assert derived["champion_share_ppb"] == 300_000_000
    assert derived["burned_residual_ppb"] == 700_000_000
    assert derived["sparse_uids"] == [0, 1]
    fixtures.assert_canary_absent(harness, connect)


def test_no_qualifying_challenger_keeps_prior_champion_reward_weights_and_next_baseline(
    database, tmp_path, fixed_scores
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = BaselineCostHarness(
        connect, tmp_path, challengers=["PriorWinner"], runners=["alpha"]
    )

    first_count = fixtures._start_round(harness, day=4, epoch=42_000)
    fixtures._run_stage_one_to_scoring(harness, first_count, runners=1)
    harness.advance_until("published", runners=1)
    first = harness.service.store.get_round(harness.round_id)
    first_ranking = _ranking_by_id(first)
    first_baseline = _baseline_participant(first)
    assert first_ranking[first_baseline["submission_id"]]["eligible"] is True
    assert first_ranking[first_baseline["submission_id"]]["final_score"] == fixed_scores[
        "PublicBaseline"
    ]
    prior_winner = _submission_for_flavor(harness, "PriorWinner")
    assert first["king_hotkey"] == prior_winner["miner_hotkey"]

    promotion_root = tmp_path / "incumbent-promotion"
    promotion_root.mkdir()
    remote = fixtures.promotion_repository(promotion_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "incumbent-promotion-cache"
    )

    def promoted_baseline(_url, _limit):
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    harness.service.config.baseline_source_fetcher = promoted_baseline
    harness.clock.now = datetime.fromisoformat(
        first["published_at"].replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok",
        "promoted": 1,
    }
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    first_activated = harness.service.store.get_round(harness.round_id)
    first_basis = first_activated["reward_basis_doc"]

    harness.challengers = ["LowEligible", "ExpensiveHigh"]
    harness.expensive_baseline = True
    harness.expensive_flavors = {"ExpensiveHigh"}
    harness.clock.now = datetime.now(timezone.utc)
    second_count = fixtures._start_round(harness, day=5, epoch=42_002)
    fixtures._run_stage_one_to_scoring(harness, second_count, runners=1)
    harness.advance_until("published", runners=1)
    second = harness.service.store.get_round(harness.round_id)
    second_baseline = _baseline_participant(second)
    low = _submission_for_flavor(harness, "LowEligible")
    expensive = _submission_for_flavor(harness, "ExpensiveHigh")
    ranking = _ranking_by_id(second)
    _assert_cost_evidence(
        ranking[second_baseline["submission_id"]],
        successful_microusd=600,
        eligible=False,
        reason="cost_per_company_exceeded",
    )
    _assert_cost_evidence(
        ranking[low["submission_id"]],
        successful_microusd=200,
        eligible=True,
        reason="eligible",
    )
    _assert_cost_evidence(
        ranking[expensive["submission_id"]],
        successful_microusd=600,
        eligible=False,
        reason="cost_per_company_exceeded",
    )
    assert ranking[second_baseline["submission_id"]]["final_score"] == 0.0
    assert ranking[low["submission_id"]]["final_score"] == fixed_scores["LowEligible"]
    assert ranking[expensive["submission_id"]]["final_score"] == fixed_scores[
        "ExpensiveHigh"
    ]
    assert second["publication_doc"]["king_decision"] == {
        "outcome": "no_king",
        "king_submission_id": None,
        "king_hotkey": "",
        "winner_submission_id": None,
    }
    assert harness.service.promote_pending_baselines() == {
        "status": "ok",
        "promoted": 0,
    }
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    second_activated = harness.service.store.get_round(harness.round_id)
    second_basis = second_activated["reward_basis_doc"]
    assert second_basis["king_hotkey"] == prior_winner["miner_hotkey"]
    assert second_basis["king_outcome"] == "defended"
    assert second_basis["king_start_epoch"] == first_basis["king_start_epoch"]
    assert second_basis["champion_reward_factor_ppm"] == first_basis[
        "champion_reward_factor_ppm"
    ]

    burn = fixtures.keypair("baseline-cost-zero-burn-two").ss58_address
    reward_epoch = int(second_activated["effective_reward_epoch"])
    state = _accepted_state(harness, reward_epoch, burn)
    assert state["reward_basis"] == second_basis
    derived = arena_weights.derive_arena_weights(
        state, [burn, prior_winner["miner_hotkey"]]
    )
    assert derived["champion_share_ppb"] == 300_000_000
    assert derived["burned_residual_ppb"] == 700_000_000

    # The next daily round freezes the still-current promoted source again.
    harness.challengers = []
    harness.expensive_baseline = False
    harness.expensive_flavors = set()
    harness.clock.now = datetime.now(timezone.utc)
    harness.chain.epoch = 42_004
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-06",
    )
    harness.round_id = configuration["round_id"]
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    third = harness.service.store.get_round(harness.round_id)
    next_baseline = _baseline_participant(third)
    next_submission = harness.service.store.get_submission(
        next_baseline["submission_id"]
    )
    with tarfile.open(
        fileobj=io.BytesIO(harness.objects.get(next_submission["source_ref"])),
        mode="r:gz",
    ) as archive:
        assert archive.extractfile("flavor.txt").read().decode() == "PriorWinner"
    fixtures.assert_canary_absent(harness, connect)
