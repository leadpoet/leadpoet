"""Real Arena persistence and normal signers enforce the 140-epoch week."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import re
import pytest
from pathlib import Path
import urllib.request

from bittensor_wallet import Keypair
from fastapi.testclient import TestClient
from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot

from lab_arena import contracts, rewards, submission_similarity, wiring
from lab_arena.api import create_app
from lab_arena.local_weight_signer import load_public_chain_signing_profile
from lab_arena.promotion import GitPromoter
from lab_arena.validator import ArenaWeightOrchestrator, ArenaWeightPaths
from leadpoet_canonical import arena_weights
from tests.lab_arena.test_lab_arena_normal_weight_flow import (
    _ExternalSource, _HostChain, _NormalWeightHarness, _local_signer_client,
    _public_api, _test_client_urlopen, assert_canary_absent, daily_icps,
    frozen_image_judge_contract, integrated_database, local_worker_socket_polling,
    promotion_repository,
)
from tests.lab_arena.test_lab_arena_service_round import _start_round


SOURCE_EPOCH = 35001
WEEK = 140


@pytest.mark.parametrize("total_emissions", [False, True])
def test_weekly_clock_scoring_publication_persistence_and_three_normal_signers(
    integrated_database, tmp_path, monkeypatch, total_emissions,
):
    policy = rewards.reward_slot_policy_document(date(2026, 11, 11 if total_emissions else 1))
    monkeypatch.setattr(rewards, "reward_slot_policy_document", lambda *_, **__: deepcopy(policy))
    psycopg2, dsn = integrated_database
    connect = lambda: psycopg2.connect(**dsn)
    if total_emissions:
        with connect() as database, database.cursor() as cursor:
            scripts = Path("scripts")
            cursor.execute((scripts / "591-lab-arena-legacy-baseline-reward-comparison.sql").read_text())
            prior = (scripts / "592-arena-2026-10-29-authorized-champion.sql").read_text()
            cursor.execute(re.search(r'DO \$oct29_snapshot592\$.*?\$oct29_snapshot592\$;', prior, re.S)[0])
            cursor.execute((scripts / "681-lab-arena-total-emissions-reward-slots.sql").read_text())
    harness = _NormalWeightHarness(
        connect, tmp_path, challengers=["NormalSlotWinner71"], runners=["alpha", "beta"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults, rewards_enabled=True, benchmark_icp_count=2,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]), "icps": daily_icps()[:2],
    }
    participants = _start_round(harness, day=29, epoch=SOURCE_EPOCH - 1)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == participants
    harness.advance_until("published", runners=2)
    repository = tmp_path / "weekly-promotion"
    repository.mkdir()
    remote = promotion_repository(repository)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "weekly-promotion-cache",
    )
    assert harness.service.promote_pending_baselines()["status"] == "ok"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    published = harness.service.store.get_round(harness.round_id)
    basis = published["reward_basis_doc"]
    assert published["baseline_promoted_at"] is not None
    assert basis["effective_reward_epoch"] == SOURCE_EPOCH
    assert basis["reward_slots"][0] is None
    assert (basis["reward_slots"][1] == basis["reward_slots"][2]) if total_emissions else (basis["reward_slots"][1] is None)
    achievement = basis["reward_slots"][2]
    assert achievement["winner_score"] == 83.6
    assert achievement["baseline_score"] == 69.8
    winner = harness.service.store.get_submission(achievement["submission_id"])
    payload = harness.objects.get_bounded(winner["source_ref"], winner["source_size_bytes"])
    identity = submission_similarity.inspect_archive(payload)
    assert identity.archive_sha256 == winner["source_archive_sha256"]
    assert identity.normalized_sha256 == winner["source_normalized_sha256"]
    frozen_basis_bytes = contracts.canonical_json(basis)

    profile = load_public_chain_signing_profile(
        "finney", path=Path("lab_arena/chain_signing_profile_v2.json"),
    )
    burn = Keypair.create_from_uri("//ArenaWeeklyFlowBurn").ss58_address
    keys = [Keypair.create_from_uri("//ArenaWeeklyFlowValidator%d" % index) for index in range(3)]
    for key in keys:
        harness.chain.runners.append(key.ss58_address)
        harness.chain.stakes[key.ss58_address] = 1
    paths = [ArenaWeightPaths(tmp_path / ("weekly-validator-%d" % index)) for index in range(3)]

    def gateway(epoch, *, enabled):
        harness.chain.epoch = epoch
        snapshot = SubnetEpochSnapshot(
            network_genesis_hash="0x" + profile["genesis_hash"], netuid=71, head_kind="finalized",
            block_hash="0x" + "2" * 64, current_block=104, last_epoch_block=100,
            pending_epoch_at=0, subnet_epoch_index=10 + epoch - SOURCE_EPOCH,
            tempo=360, blocks_since_last_step=4, observed_at="2026-11-09T00:00:00Z",
        )
        adapter = wiring.ChainReadsAdapter(harness.chain)
        adapter._cutover = SubnetEpochCutover(
            network_genesis_hash=snapshot.network_genesis_hash, netuid=71,
            cutover_block=100, cutover_block_hash="0x" + "3" * 64,
            first_subnet_epoch_index=10, first_settlement_epoch_id=SOURCE_EPOCH,
            last_legacy_epoch_id=SOURCE_EPOCH - 1,
        )
        monkeypatch.setattr(wiring.chain_module, "finalized_epoch_snapshot", lambda _: snapshot)
        assert adapter.accepted_weight_epoch_scope()["genesis_hash"] == "0x" + profile["genesis_hash"]
        harness.clock.now = datetime.now(timezone.utc)
        service = harness.build_service()
        service.config.chain = adapter
        service.config.accepted_burn_hotkey = burn
        service.config.weekly_slot_decay_enabled = enabled
        return service

    # This durable state predates activation. Enabling the feature must not
    # add a clock or replace any signed field on an already accepted epoch.
    old_gateway = gateway(SOURCE_EPOCH, enabled=False)
    old_state = old_gateway.public_weight_state(SOURCE_EPOCH)["state"]
    old_bytes = contracts.canonical_json(old_state)
    assert "weekly_slot_decay" not in old_state
    enabled_gateway = gateway(SOURCE_EPOCH, enabled=True)
    assert contracts.canonical_json(enabled_gateway.public_weight_state(SOURCE_EPOCH)["state"]) == old_bytes

    states, epoch_vectors = [], []
    controls = (
        ((SOURCE_EPOCH + WEEK - 1, 400_000_000, [65535, 43690]),
         (SOURCE_EPOCH + WEEK, 200_000_000, [65535, 16384]))
        if total_emissions else
        ((SOURCE_EPOCH + WEEK - 1, 150_000_000, [65535, 11565]),
         (SOURCE_EPOCH + WEEK, 75_000_000, [65535, 5314]))
    )
    for epoch, expected_share, expected_weights in controls:
        service = gateway(epoch, enabled=True)
        vectors = []
        with TestClient(create_app(service)) as http, monkeypatch.context() as patcher:
            patcher.setattr(urllib.request, "urlopen", _test_client_urlopen(http))
            if epoch == SOURCE_EPOCH + WEEK:
                # A new gateway must recover the verified signed checkpoint
                # even when the original frozen archive cannot be fetched.
                def unavailable_archive(*_args, **_kwargs):
                    raise AssertionError("signed checkpoint re-read the old source archive")

                patcher.setattr(harness.objects, "get_bounded", unavailable_archive)
            for index, key in enumerate(keys):
                api = _public_api(key, harness.clock)
                state = api.accepted_weight_state(epoch)
                assert state["reward_basis"] == basis
                clock = state["weekly_slot_decay"]
                assert clock == {
                    "epochs_per_halving": WEEK,
                    "slots": [None if slot is None else {
                        "start_epoch": SOURCE_EPOCH,
                        "source_normalized_sha256": identity.normalized_sha256,
                    } for slot in basis["reward_slots"]],
                }
                arena_weights.verify_accepted_weight_state_signature(
                    state, public_key_der=harness.signer.public_key_der,
                    expected_public_key_hash=harness.signer.public_key_hash,
                )
                derived = arena_weights.derive_arena_weights(state, [burn, achievement["miner_hotkey"]])
                assert derived["champion_share_ppb"] == expected_share
                assert derived["burned_residual_ppb"] == 1_000_000_000 - expected_share
                assert derived["sparse_weights_u16"] == expected_weights
                source = _ExternalSource(
                    [burn, achievement["miner_hotkey"]], key.public_key.hex(),
                    profile["genesis_hash"], epoch=epoch,
                )
                host = _HostChain(source, key.ss58_address)

                def orchestrator():
                    return ArenaWeightOrchestrator(
                        api=api, chain=host,
                        signer=_local_signer_client(
                            key=key, source=source, profile=profile,
                            arena_signer=harness.signer, burn=burn,
                        ),
                        validator_hotkey=key.ss58_address,
                        expected_signing_key_hash=harness.signer.public_key_hash,
                        paths=paths[index], extrinsic_period=int(profile["extrinsic_period"]),
                    )

                assert orchestrator().run_once(epoch) == "broadcast"
                signed_bytes = paths[index].signed(epoch).read_bytes()
                signed = json.loads(signed_bytes)
                assert signed["accepted_state"] == state
                vectors.append((signed["sparse_uids"], signed["sparse_weights_u16"]))
                assert vectors[-1] == ([0, 1], expected_weights)
                source.expected_weights = list(zip(*vectors[-1]))
                restarted = orchestrator()
                assert restarted.run_once(epoch) == "rebroadcast"
                assert paths[index].signed(epoch).read_bytes() == signed_bytes
                source.included = True
                assert restarted.run_once(epoch) == "included_pending_reveal"
                assert not paths[index].outcome(epoch).exists()
                source.revealed = True
                assert restarted.run_once(epoch) == "finalized"
                assert paths[index].signed(epoch).read_bytes() == signed_bytes
                outcome = json.loads(paths[index].outcome(epoch).read_bytes())
                assert outcome["outcome"]["revealed_weights"] == [list(item) for item in source.expected_weights]
                assert outcome["report_document"]["state_hash"] == state["state_hash"]
            assert vectors == [vectors[0]] * 3
            states.append(state)
            epoch_vectors.append(vectors[0])
        assert len(service.public_chain_outcomes(epoch)["outcomes"]) == 3
        # A separate gateway object must load exactly the same persisted state.
        assert gateway(epoch, enabled=True).public_weight_state(epoch)["state"] == state

    assert epoch_vectors[0] != epoch_vectors[1]
    with connect() as database, database.cursor() as cursor:
        cursor.execute(
            "SELECT epoch, state_doc FROM public.lab_arena_accepted_weight_states "
            "WHERE network = 'finney' AND netuid = 71 ORDER BY epoch"
        )
        saved = cursor.fetchall()
    assert [row[0] for row in saved] == [SOURCE_EPOCH, SOURCE_EPOCH + WEEK - 1, SOURCE_EPOCH + WEEK]
    assert contracts.canonical_json(saved[0][1]) == old_bytes
    assert [row[1] for row in saved[1:]] == states
    assert contracts.canonical_json(harness.service.store.get_round(harness.round_id)["reward_basis_doc"]) == frozen_basis_bytes
    assert_canary_absent(harness, connect)
