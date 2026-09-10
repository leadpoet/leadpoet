"""One executable proof from Arena scoring through two normal-validator outcomes."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from lab_arena import signing
from lab_arena.store import ArenaStoreError
from lab_arena.promotion import GitPromoter
from lab_arena.validator import ArenaWeightOrchestrator, ArenaWeightPaths
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, _run_stage_one_to_scoring, _start_round, promotion_repository,
    assert_canary_absent,
)
from tests.postgres_migration_harness import SCRIPTS
from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner
from validator_tee.enclave.hotkey_authority_v2 import load_chain_signing_profile


@pytest.fixture(scope="module")
def integrated_database():
    database = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:-1])
    psycopg2, dsn = next(database)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE public.research_lab_emission_allocation_snapshots (epoch bigint, netuid integer, snapshot_status text, allocation_doc jsonb, lab_cap_alpha_percent numeric)")
            cursor.execute("CREATE TABLE public.fulfillment_score_consensus (miner_hotkey text, reward_pct numeric, reward_expires_epoch bigint, is_winner boolean, computed_at timestamptz)")
            cursor.execute("CREATE TABLE public.banned_hotkeys (hotkey text)")
            cursor.execute((SCRIPTS / DEFAULT_MIGRATIONS[-1]).read_text(encoding="utf-8"))
        yield psycopg2, dsn
    finally:
        connection.close()
        database.close()


class _Era:
    def encode(self, value): self.value = value
    def birth(self, current): return current - (current % int(self.value["period"]))


class _ExternalSource:
    def __init__(self, hotkeys, validator_public_hex, genesis):
        self.hotkeys = list(hotkeys); self.validator_public_hex = validator_public_hex
        self.genesis = genesis; self.included = False; self.revealed = False
        self.extrinsic = None; self.expected_weights = []

    def snapshot(self):
        return {
            "header": {"block": 104}, "finalized_block_hash": "0x" + "2" * 64,
            "epoch_authority": {"settlement_epoch_id": 32001, "last_epoch_block": 100,
                "pending_epoch_at": 460, "subnet_epoch_index": 10, "tempo": 360,
                "blocks_since_last_step": 4, "current_block": 104},
            "metagraph": {"hotkeys": self.hotkeys},
        }

    def read_finalized_snapshot(self, **_): return self.snapshot()
    def read_chain_signing_runtime(self, **_):
        return {"runtime_block": 104, "finalized_block": 104,
            "finalized_block_hash": "0x" + "2" * 64, "spec_version": 438,
            "transaction_version": 1, "genesis_hash": self.genesis}
    def read_canonical_block_hash(self, **_): return "1" * 64
    def read_finalized_account_nonce(self, **_): return 0
    def read_finalized_head(self): return {"block": 110, "block_hash": "0x" + "8" * 64}
    def find_finalized_extrinsic_inclusion(self, **_):
        if not self.included:
            from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2Error
            raise ValidatorChainSourceV2Error("authorized extrinsic range is not finalized")
        return {"finalized_block": 110, "finalized_block_hash": "0x" + "8" * 64,
            "state_transition_hash": "sha256:" + "9" * 64}
    def prove_timelocked_reveal_transition(self, **_):
        if not self.revealed:
            return None
        return {
            "reveal_block": 110, "reveal_block_hash": "0x" + "4" * 64,
            "validator_uid": 7, "last_update": 110,
            "weights": list(self.expected_weights),
            "transition_hash": "sha256:" + "6" * 64,
        }


class _Drand:
    def generate_commit(self, **_): return b"c" * 32, 12


class _SignerClient:
    def __init__(self, signer): self.signer = signer
    def prepare_arena_weight_extrinsic_v1(self, request): return self.signer.prepare(request)
    def confirm_arena_weight_extrinsic_v1(self, request): return self.signer.confirm(request)
    def sign_arena_chain_outcome_v1(self, request): return self.signer.sign_chain_outcome(request)
    def recover_arena_weight_extrinsic_v1(self, request): return self.signer.recover(request)


class _HostChain:
    def __init__(self, source, validator_hotkey):
        self.source = source; self.config = SimpleNamespace(netuid=71, network_name="finney")
        self.broadcasts = []; self.validator_hotkey = validator_hotkey
        self.client = SimpleNamespace(
            runtime_config=SimpleNamespace(create_scale_object=lambda _name: _Era()),
            get_account_nonce=lambda _hotkey: 0,
            get_block_hash=lambda block_id: "0x" + (source.genesis if block_id == 0 else "1" * 64),
            rpc_request=self._broadcast,
        )
    def _broadcast(self, method, params):
        assert method == "author_submitExtrinsic"; self.broadcasts.append(params[0]); self.source.extrinsic = params[0]
    def finalized_head(self): return SimpleNamespace(number=104, hash="0x" + "2" * 64)
    def refresh_metagraph(self): return SimpleNamespace(hotkeys=tuple(self.source.hotkeys))


class _Api:
    def __init__(self, service): self.service = service
    def signing_key(self): return self.service.signing_key_document()
    def accepted_weight_state(self, epoch): return self.service.public_weight_state(epoch)["state"]
    def submit_chain_outcome(self, document): return self.service.record_chain_outcome(document)


def test_scoring_reward_two_normal_validators_restart_and_chain_readback(integrated_database, tmp_path, monkeypatch):
    psycopg2, dsn = integrated_database
    connect = lambda: psycopg2.connect(**dsn)
    with connect() as db, db.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_schema_version_v1(), public.lab_arena_weight_state_schema_v1()")
        core_schema, weight_schema = cursor.fetchone()
    assert core_schema == {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197}
    assert weight_schema == {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202}
    harness = Harness(connect, tmp_path, challengers=["NormalWinner"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(harness.service.config.defaults, rewards_enabled=True)
    participants = _start_round(harness, day=30, epoch=32000)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("published", runners=2)
    repository_root = tmp_path / "promotion"; repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-cache")
    assert harness.service.promote_pending_baselines()["status"] == "ok"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"

    burn = Keypair.create_from_uri("//ArenaFlowBurn").ss58_address
    fulfillment = Keypair.create_from_uri("//ArenaFlowFulfillment").ss58_address
    with connect() as db, db.cursor() as cursor:
        cursor.execute("INSERT INTO public.research_lab_emission_allocation_snapshots VALUES (31999,71,'active','{}'::jsonb,30)")
        db.commit()
    with pytest.raises(ArenaStoreError, match="allocation_invalid"):
        harness.service.store.weight_inputs(
            31999, 71, burn, fulfillment_enabled=True, leaderboard_enabled=True,
        )
    invalid_row = {"lab_cap_percent": 30, "unallocated_percent": 0,
        "reimbursement_allocations": [{"miner_hotkey": fulfillment, "paid_alpha_percent": -1}],
        "champion_allocations": [], "queued_champion_allocations": []}
    with connect() as db, db.cursor() as cursor:
        cursor.execute("UPDATE public.research_lab_emission_allocation_snapshots SET allocation_doc = %s::jsonb WHERE epoch = 31999 AND netuid = 71", (json.dumps(invalid_row),))
        db.commit()
    with pytest.raises(ArenaStoreError, match="allocation_invalid"):
        harness.service.store.weight_inputs(
            31999, 71, burn, fulfillment_enabled=True, leaderboard_enabled=True,
        )
    invalid_row["reimbursement_allocations"][0]["paid_alpha_percent"] = 31
    with connect() as db, db.cursor() as cursor:
        cursor.execute("UPDATE public.research_lab_emission_allocation_snapshots SET allocation_doc = %s::jsonb WHERE epoch = 31999 AND netuid = 71", (json.dumps(invalid_row),))
        db.commit()
    with pytest.raises(ArenaStoreError, match="allocation_over_cap"):
        harness.service.store.weight_inputs(
            31999, 71, burn, fulfillment_enabled=True, leaderboard_enabled=True,
        )
    with connect() as db, db.cursor() as cursor:
        cursor.execute("DELETE FROM public.research_lab_emission_allocation_snapshots WHERE epoch = 31999 AND netuid = 71")
        allocation = {"lab_cap_percent": 30, "unallocated_percent": 30,
            "reimbursement_allocations": [], "champion_allocations": [], "queued_champion_allocations": []}
        cursor.execute("INSERT INTO public.research_lab_emission_allocation_snapshots VALUES (%s,71,'active',%s::jsonb,30)", (32001, json.dumps(allocation)))
        cursor.execute("INSERT INTO public.fulfillment_score_consensus VALUES (%s,0.40,32002,TRUE,now())", (fulfillment,))
        db.commit()
    harness.service.config.accepted_burn_hotkey = burn
    harness.service.config.fulfillment_enabled = True
    harness.service.config.leaderboard_emissions_enabled = True
    harness.chain.accepted_weight_epoch_scope = lambda: {"genesis_hash": "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03", "epoch": 32001, "valid_from_block": 100, "valid_until_block": 459}
    harness.chain.epoch = 32001
    state = harness.service.public_weight_state(32001)["state"]
    assert state == harness.service.public_weight_state(32001)["state"]
    with pytest.raises(ArenaStoreError, match="weight_state_conflict"):
        harness.service.store.publish_weight_state(
            "finney", 71, 32001, "sha256:" + "0" * 64,
            {**state, "state_hash": "sha256:" + "0" * 64},
        )
    harness.clock.now = datetime.now(timezone.utc)

    profile = load_chain_signing_profile(Path("validator_tee/enclave/chain_signing_profile_v2.json"))
    monkeypatch.setattr("validator_tee.enclave.hotkey_authority_v2.load_chain_signing_profile", lambda: profile)
    outcomes = []
    for index in range(2):
        key = Keypair.create_from_uri("//ArenaNormalValidator%d" % index)
        hotkeys = [burn, fulfillment]
        if state["reward_basis"]["king_hotkey"]:
            hotkeys.append(state["reward_basis"]["king_hotkey"])
        source = _ExternalSource(hotkeys, key.public_key.hex(), profile["genesis_hash"])
        protected = ArenaWeightSigner(
            validator_hotkey=key.ss58_address, hotkey_public_key_hex=key.public_key.hex(),
            chain_profile=profile, chain_source=source, drand_backend=_Drand(),
            sign_sr25519=key.sign, arena_public_key_der=harness.signer.public_key_der,
            verify_sr25519=lambda signature, message, k=key: k.verify(message, signature),
            arena_public_key_hash=harness.signer.public_key_hash, network="finney", netuid=71,
            burn_hotkey=burn,
        )
        host_chain = _HostChain(source, key.ss58_address)
        paths = ArenaWeightPaths(tmp_path / ("v%d" % index))
        orchestrator = ArenaWeightOrchestrator(
            api=_Api(harness.service), chain=host_chain, signer=_SignerClient(protected),
            validator_hotkey=key.ss58_address, expected_signing_key_hash=harness.signer.public_key_hash,
            paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
        )
        assert orchestrator.run_once(32001) == "broadcast"
        signed_path = paths.signed(32001)
        outcome_path = paths.outcome(32001)
        signed_bytes = signed_path.read_bytes()
        signed = json.loads(signed_bytes)
        source.expected_weights = list(zip(signed["sparse_uids"], signed["sparse_weights_u16"]))
        restarted = ArenaWeightOrchestrator(
            api=_Api(harness.service), chain=host_chain, signer=_SignerClient(protected),
            validator_hotkey=key.ss58_address, expected_signing_key_hash=harness.signer.public_key_hash,
            paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
        )
        assert restarted.run_once(32001) == "rebroadcast"
        assert signed_path.read_bytes() == signed_bytes
        source.included = True
        assert restarted.run_once(32001) == "included_pending_reveal"
        assert signed_path.read_bytes() == signed_bytes
        assert not outcome_path.exists()
        assert len(host_chain.broadcasts) == 2
        source.revealed = True
        assert restarted.run_once(32001) == "finalized"
        outcomes.append(outcome_path.read_bytes())
    assert len(harness.service.public_chain_outcomes(32001)["outcomes"]) == 2
    assert outcomes[0] != outcomes[1]
    first_report = harness.service.public_chain_outcomes(32001)["outcomes"][0]
    with connect() as db, db.cursor() as cursor:
        cursor.execute("DELETE FROM public.lab_arena_chain_outcomes WHERE request_id = %s", (first_report["request_id"],))
        db.commit()
    harness.clock.now += timedelta(minutes=10)
    assert harness.service.record_chain_outcome(first_report) == {"status": "recorded"}
    assert harness.service.record_chain_outcome(first_report) == {"status": "recorded"}
    assert_canary_absent(harness, connect)
    with pytest.raises(Exception, match="not_current"):
        harness.service.public_weight_state(32002)
