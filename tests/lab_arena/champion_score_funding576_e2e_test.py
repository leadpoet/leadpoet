"""A promoted champion's verified fallback survives judging and reward activation.

Only the provider and sandbox boundaries are controlled. The signed HTTP API,
runner socket, broker, PostgreSQL authority, promotion and weights are real.
"""

from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab_arena import broker as br, contracts, runner, weight_state
from lab_arena.api import create_app
from lab_arena.chain import MetagraphSnapshot
from lab_arena.submission_runtime import SubmissionProviderKeys
from leadpoet_canonical import arena_weights
from leadpoet_canonical.lab_arena_rewards import slot_allocations
from tests.lab_arena.champion_funding_round_test import (
    HOST_KEYS,
    ChampionHarness,
    _activate,
    _baseline_id,
    _promote_first_winner,
    _start_round,
)
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge,
    _enable_verified_proxy_runtime,
)
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.parallel_twenty_runtime_e2e_test import bounded_socket_shutdown_poll
from tests.lab_arena.test_lab_arena_service_round import assert_canary_absent, keypair


ROOT = Path(__file__).parents[2]


@pytest.fixture
def current_database():
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database

    database = full_thinking_database()
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            for migration in (
                "575-lab-arena-score-host-continuation.sql",
                "576-lab-arena-champion-score-funding.sql",
            ):
                cursor.execute((ROOT / "scripts" / migration).read_text())
        yield pg, dsn
    finally:
        database.close()


class CurrentChampionHarness(ChampionHarness):
    """Keep the historical fixture, with the current production score callback."""

    def objects_key(self):
        return "champion-score-funding576"

    def build_service(self):
        service = super().build_service()
        payer = SubmissionProviderKeys(
            store=service.store,
            credentials=self.credential_manager,
            organizer_keys=HOST_KEYS,
        )
        original = service.config.broker_factory

        def broker_factory(owner, round_row):
            broker = original(owner, round_row)
            broker._champion_score_for = payer.champion_score_for
            return broker

        service.config.broker_factory = broker_factory
        service.config.defaults = replace(
            service.config.defaults,
            execution_sequence_from="2000-01-01T00:00:00Z",
            per_icp_cost_policy=True,
            integrity_from="2000-01-01T00:00:00Z",
            normalize_intent_scale=False,
        )

        def metagraph(*, finalized=True):
            assert finalized
            hotkeys = tuple(key.ss58_address for key in fixtures.KEYS.values())
            return MetagraphSnapshot(
                netuid=71, block_number=123, block_hash="0x" + "a" * 64,
                hotkeys=hotkeys, coldkeys=hotkeys,
                validator_permit=tuple(key in self.runner_keys for key in hotkeys),
                stake=tuple(100_000.0 if key in self.runner_keys else 0.0 for key in hotkeys),
                active=tuple(key in self.runner_keys for key in hotkeys),
            )

        self.chain.metagraph = metagraph
        return service


def _bind_http(harness, stack):
    client = stack.enter_context(TestClient(create_app(harness.service)))
    harness.api_factory = lambda: runner.HttpArenaApiClient(
        "http://localhost", client=client
    )


def test_verified_fallback_preserves_execution_and_completes_host_judging_and_rewards(
    current_database, tmp_path, bounded_socket_shutdown_poll, monkeypatch,
):
    pg, dsn = current_database
    connect = lambda: pg.connect(**dsn)
    harness = CurrentChampionHarness(
        connect, tmp_path,
        challengers=["Alpha", "Bravo", "Charlie"],
        runners=["alpha", "beta", "gamma"],
    )
    harness.clock.now = datetime.now(timezone.utc)
    _enable_verified_proxy_runtime(harness)

    def challenger_wins(companies, icp, reference):
        return [
            dict(row, final_score=100.0 - row["final_score"])
            for row in _baseline_first_judge(companies, icp, reference)
        ]

    monkeypatch.setattr(fixtures, "deterministic_scorer", challenger_wins)
    with ExitStack() as stack:
        stack.callback(harness.service.store.close)
        _bind_http(harness, stack)
        first = _start_round(
            harness, day=21, epoch=42576,
            challengers=("Alpha", "Bravo", "Charlie"),
        )
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.BENCHMARK_ICP_COUNT
        harness.advance_until("published", runners=3)
        first = harness.service.store.get_round(harness.round_id)
        assert first["king_outcome"] == "crowned"
        winner = first["publication_doc"]["king_decision"]
        winner_id = winner["winner_submission_id"]
        winner_hotkey = winner["king_hotkey"]
        _promote_first_winner(harness, tmp_path, first)
        _activate(harness, 1_000_000)

        # The promoted package still charges its original owner's account.
        round_row = _start_round(
            harness, day=22, epoch=harness.chain.epoch + 10, pool_percent=25,
        )
        baseline_id = _baseline_id(round_row)
        assert round_row["champion_submission_id"] == winner_id
        assert round_row["champion_hotkey"] == winner_hotkey
        configuration = round_row["configuration_doc"]
        benchmark = harness.objects.get(round_row["benchmark_ref"])
        harness.provider_transport.champion_key = "deepline-miner-" + winner_id
        harness.provider_transport.arm_account_failure(402)
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.BENCHMARK_ICP_COUNT

        worker = harness.runner(0, parallel=2)
        scheduled = harness.clock.now
        harness.clock.now = datetime.now(timezone.utc)
        try:
            for _ in range(9):
                assert worker.run_once(max_claims=1) == 1
        finally:
            worker.close()
            harness.clock.now = scheduled
        executions = harness.service.store.list_runs(
            harness.round_id, submission_id=baseline_id, kind="execute"
        )
        stable = {
            run["run_id"]: harness.objects.get(run["output_ref"])
            for run in executions if run["status"] == "accepted"
        }
        assert len(stable) == 8
        failed = [run for run in executions if run["status"] == "failed"]
        assert len(failed) == 1 and failed[0]["icp_position"] == 8
        assert failed[0]["champion_restart_required"] is True
        assert harness.provider_transport.failure_count == br.CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS
        assert harness.service.store.get_round(harness.round_id)["champion_fallback_providers"] == ["deepline"]
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s "
                "AND entry_kind='settlement' AND funding_source='miner_key' "
                "AND amount_microusd=0 AND terminal_response->>'status'='402'",
                (failed[0]["run_id"],),
            )
            assert cursor.fetchone()[0] == br.CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS

        # A new service/runner must preserve accepted source outputs and the
        # immutable round configuration while it uses host funding for retries.
        harness.service = harness.build_service()
        stack.callback(harness.service.store.close)
        _bind_http(harness, stack)
        harness.run_stage_with_runners(1)
        assert harness.service.store.get_round(harness.round_id)["configuration_doc"] == configuration
        assert harness.objects.get(round_row["benchmark_ref"]) == benchmark
        for run_id, output in stable.items():
            preserved = harness.service.store.get_run(run_id)
            assert preserved["status"] == "accepted"
            assert harness.objects.get(preserved["output_ref"]) == output
        assert harness.service.advance_round(harness.round_id)["status"] == "ok"
        assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.BENCHMARK_ICP_COUNT
        transport_cursor = len(harness.provider_transport.sent)
        harness.run_stage_with_runners(1)
        stage_one_scores = harness.service.store.list_runs(
            harness.round_id, submission_id=baseline_id, stage=1, kind="score"
        )
        expected_sources = {
            "deepline": "host", "openrouter": "miner_key", "scrapingdog": "miner_key",
        }
        assert len(stage_one_scores) == contracts.BENCHMARK_ICP_COUNT
        assert all(run["status"] == "accepted" for run in stage_one_scores)
        assert all(run["champion_funding_sources"] == expected_sources for run in stage_one_scores)
        dispatched = [
            authorization
            for method, url, authorization in harness.provider_transport.sent[transport_cursor:]
            if method == "POST" and url.startswith("https://code.deepline.com/api/v2/integrations/")
        ]
        assert dispatched and set(dispatched) == {"Bearer " + HOST_KEYS["deepline"]}

        harness.advance_until("published", runners=1)
        published = harness.service.store.get_round(harness.round_id)
        assert published["configuration_doc"] == configuration
        accepted_executions = [
            run for run in harness.service.store.list_runs(
                harness.round_id, submission_id=baseline_id, kind="execute"
            ) if run["status"] == "accepted"
        ]
        assert len(accepted_executions) == contracts.BENCHMARK_ICP_COUNT
        assert {run["icp_position"] for run in accepted_executions} == set(
            range(contracts.BENCHMARK_ICP_COUNT)
        )
        for run in accepted_executions:
            if run["run_id"] in stable:
                assert harness.objects.get(run["output_ref"]) == stable[run["run_id"]]
        scores = harness.service.store.list_runs(
            harness.round_id, submission_id=baseline_id, kind="score"
        )
        assert len(scores) == contracts.BENCHMARK_ICP_COUNT
        assert all(run["status"] == "accepted" for run in scores)
        assert all(run["champion_funding_sources"] == expected_sources for run in scores)
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE l.funding_source='host' "
                "AND l.terminal_response->>'status'='200') "
                "FROM public.lab_arena_ledger l JOIN public.lab_arena_runs r USING(run_id) "
                "WHERE r.round_id=%s AND r.kind='score' AND l.provider='deepline' "
                "AND l.entry_kind='settlement'",
                (harness.round_id,),
            )
            settlements, successful_host = cursor.fetchone()
        assert settlements == successful_host == contracts.BENCHMARK_ICP_COUNT
        basis = _activate(harness, 500_000)
        policy = basis["slot_policy"]
        owned = [
            (tier, slot)
            for tier, slot in zip(policy["tiers"], basis["reward_slots"])
            if slot and slot["miner_hotkey"] == winner_hotkey
        ]
        assert owned
        for _tier, slot in owned:
            if "decay" in policy:
                start = slot["start_epoch"] or basis["effective_reward_epoch"]
                assert basis["effective_reward_epoch"] - start < policy["decay"]["epochs_per_halving"]
        expected_share = sum(
            Fraction(policy["pool_percent"], 100)
            * Fraction(tier["allocation_percent"], 100) / 2
            for tier, _slot in owned
        )
        assert slot_allocations(
            basis, basis["effective_reward_epoch"], [winner_hotkey]
        ) == {winner_hotkey: expected_share}
        burn = keypair("champion-score-funding576-burn").ss58_address
        accepted = weight_state.build_accepted_weight_state(
            harness.signer, network="finney", genesis_hash="1" * 64, netuid=71,
            epoch=int(basis["effective_reward_epoch"]),
            valid_from_block=1, valid_until_block=360, reward_basis=basis,
            burn_hotkey=burn, issued_at=basis["published_at"],
        )
        arena_weights.verify_accepted_weight_state_signature(
            accepted, public_key_der=harness.signer.public_key_der,
            expected_public_key_hash=harness.signer.public_key_hash,
        )
        weights = arena_weights.derive_arena_weights(accepted, [winner_hotkey, burn])
        assert weights["champion_share_ppb"] == int(expected_share * 1_000_000_000)
        assert weights["burned_residual_ppb"] == 1_000_000_000 - weights["champion_share_ppb"]
        assert_canary_absent(harness, connect)
