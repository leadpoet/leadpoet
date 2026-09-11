"""One executable proof from Arena scoring through two normal-validator outcomes."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from lab_arena import contracts, signing
from lab_arena.api import create_app
from lab_arena.service import ServiceError
from lab_arena.local_weight_signer import LocalArenaWeightSigner
from lab_arena.store import ArenaStoreError
from lab_arena.promotion import GitPromoter
from lab_arena.validator import ArenaWeightOrchestrator, ArenaWeightPaths
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
    LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, _run_stage_one_to_scoring, _start_round, promotion_repository,
    assert_canary_absent,
)
from tests.postgres_migration_harness import SCRIPTS
from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner
from validator_tee.enclave.arena_hotkey import load_chain_signing_profile

RETIRED_INCENTIVE_TABLES = (
    "research_reimbursement_awards",
    "research_reimbursement_schedules",
    "research_reimbursement_award_events",
    "research_weight_input_snapshots",
    "research_lab_champion_reward_obligations",
    "research_lab_champion_reward_events",
    "research_lab_emission_allocation_snapshots",
    "research_lab_arweave_epoch_audit_anchors",
    "research_lab_arweave_epoch_audit_anchor_events",
    "research_lab_signed_audit_bundles",
    "research_lab_signed_audit_bundle_events",
    "research_lab_attested_weight_bundles",
    "research_lab_attested_weight_bundles_v2",
    "research_lab_attested_publication_events_v2",
    "research_lab_attested_weight_finalizations_v2",
    "research_lab_chain_realized_settlement_activation_v1",
    "research_lab_chain_realized_epoch_settlements_v1",
    "research_lab_chain_realized_obligation_credits_v1",
    "research_lab_allocation_settlement_frontiers_v2",
    "research_lab_allocation_settlement_frontier_activation_v2",
    "research_lab_compact_weight_submissions_v2",
    "research_lab_compact_weight_publication_intents_v2",
    "research_lab_compact_weight_authorities_v2",
)
SHARED_EPOCH_TABLES = (
    "research_lab_stateful_subnet_epoch_candidates_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
    "research_lab_stateful_subnet_epoch_cutover_state_v1",
)
RETIRED_EPOCH_WRITE_FUNCTIONS = (
    "research_lab_stateful_subnet_epoch_cutover_preflight_v1",
    "research_lab_stateful_subnet_epoch_legacy_high_water_v1",
    "research_lab_stateful_subnet_epoch_cutover_fence_v1",
    "research_lab_stateful_subnet_epoch_cutover_bind_v1",
    "research_lab_stateful_subnet_epoch_cutover_bind_v2",
    "research_lab_stateful_subnet_epoch_stage_v1",
    "research_lab_stateful_subnet_epoch_stage_v2",
    "research_lab_stateful_subnet_epoch_activate_v1",
    "research_lab_stateful_subnet_epoch_refresh_fence_v1",
    "research_lab_fresh_network_epoch_cutover_public_state_v1",
    "research_lab_champion_lifetime_credit_contract_v1",
)
TEMPORARY_WEIGHT_TRIGGER_FUNCTIONS = (
    "enforce_temporary_testnet401_execution_result_epoch_scope_v1",
    "enforce_temporary_testnet401_weight_submission_epoch_scope_v1",
)


@pytest.fixture(scope="module")
def integrated_database():
    staged_migrations = tuple(
        migration
        for migration in DEFAULT_MIGRATIONS
        if migration
        not in (
            LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION,
            LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
        )
    )
    database = database_with_lab_arena_migration(staged_migrations)
    psycopg2, dsn = next(database)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            # Production has these legacy objects. A fresh Arena database does
            # not, so seed their names to prove the retirement upgrade path.
            for table in RETIRED_INCENTIVE_TABLES:
                cursor.execute(f"CREATE TABLE public.{table} (id BIGINT)")
            for table in SHARED_EPOCH_TABLES:
                cursor.execute(f"CREATE TABLE public.{table} (id BIGINT)")
            # These are the production dependency directions that determine
            # safe drop order. Minimal columns keep the workflow test small.
            dependency_edges = (
                ("research_reimbursement_award_events", "research_reimbursement_awards"),
                ("research_lab_champion_reward_events", "research_lab_champion_reward_obligations"),
                ("research_lab_attested_publication_events_v2", "research_lab_attested_weight_bundles_v2"),
                ("research_lab_attested_weight_finalizations_v2", "research_lab_attested_publication_events_v2"),
                ("research_lab_compact_weight_publication_intents_v2", "research_lab_compact_weight_submissions_v2"),
                ("research_lab_compact_weight_authorities_v2", "research_lab_compact_weight_submissions_v2"),
                ("research_lab_chain_realized_obligation_credits_v1", "research_lab_chain_realized_epoch_settlements_v1"),
                ("research_lab_allocation_settlement_frontier_activation_v2", "research_lab_allocation_settlement_frontiers_v2"),
                ("research_lab_stateful_subnet_epoch_cutovers_v1", "research_lab_attested_weight_bundles_v2"),
                ("research_lab_stateful_subnet_epoch_cutovers_v1", "research_lab_attested_weight_finalizations_v2"),
                ("research_lab_stateful_subnet_epoch_cutovers_v1", "research_lab_stateful_subnet_epoch_candidates_v1"),
                ("research_lab_stateful_subnet_epoch_boundaries_v1", "research_lab_stateful_subnet_epoch_cutovers_v1"),
                ("research_lab_stateful_subnet_epoch_snapshots_v1", "research_lab_stateful_subnet_epoch_cutovers_v1"),
            )
            parents = {parent for _, parent in dependency_edges}
            for parent in parents:
                cursor.execute(f"ALTER TABLE public.{parent} ADD PRIMARY KEY (id)")
            for index, (child, parent) in enumerate(dependency_edges):
                cursor.execute(
                    f"ALTER TABLE public.{child} ADD CONSTRAINT retired_fk_{index} "
                    f"FOREIGN KEY (id) REFERENCES public.{parent}(id)"
                )
            for table in RETIRED_INCENTIVE_TABLES + SHARED_EPOCH_TABLES:
                cursor.execute(f"INSERT INTO public.{table} VALUES (1)")
            cursor.execute(
                "CREATE VIEW public.research_lab_epoch_payouts AS "
                "SELECT id AS epoch FROM public.research_lab_emission_allocation_snapshots"
            )
            cursor.execute(
                "CREATE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1() "
                "RETURNS TABLE(id BIGINT) LANGUAGE sql STABLE AS "
                "'SELECT id FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1'"
            )
            for function in RETIRED_EPOCH_WRITE_FUNCTIONS:
                cursor.execute(
                    f"CREATE FUNCTION public.{function}() RETURNS void "
                    "LANGUAGE sql AS 'SELECT'"
                )
            cursor.execute(
                "CREATE TABLE public.research_lab_attested_execution_results_v2 (id BIGINT)"
            )
            cursor.execute("CREATE TABLE public.transparency_log (id BIGINT)")
            for function in TEMPORARY_WEIGHT_TRIGGER_FUNCTIONS:
                cursor.execute(
                    f"CREATE FUNCTION public.{function}() RETURNS trigger "
                    "LANGUAGE plpgsql AS 'BEGIN RETURN NEW; END'"
                )
            cursor.execute(
                "CREATE TRIGGER enforce_temporary_testnet401_execution_result_epoch_scope_v1 "
                "BEFORE INSERT ON public.research_lab_attested_execution_results_v2 "
                "FOR EACH ROW EXECUTE FUNCTION public.enforce_temporary_testnet401_execution_result_epoch_scope_v1()"
            )
            cursor.execute(
                "CREATE TRIGGER enforce_temporary_testnet401_weight_submission_epoch_scope_v1 "
                "BEFORE INSERT ON public.transparency_log FOR EACH ROW EXECUTE FUNCTION "
                "public.enforce_temporary_testnet401_weight_submission_epoch_scope_v1()"
            )
            cursor.execute("INSERT INTO public.research_lab_attested_execution_results_v2 VALUES (9)")
            cursor.execute("INSERT INTO public.transparency_log VALUES (10)")
            cursor.execute(
                "CREATE VIEW public.research_lab_stateful_subnet_epoch_mapping_v1 "
                "AS SELECT id FROM public.research_lab_stateful_subnet_epoch_cutovers_v1"
            )
            cursor.execute(
                "CREATE TABLE public.research_lab_scoring_runs (id BIGINT)"
            )
            cursor.execute("INSERT INTO public.research_lab_scoring_runs VALUES (7)")
            cursor.execute(
                "CREATE FUNCTION public.persist_research_lab_chain_realized_test() "
                "RETURNS void LANGUAGE sql AS 'SELECT'"
            )
            cursor.execute(
                "CREATE FUNCTION public.research_lab_compact_checkpoint_graph_contract_v1() "
                "RETURNS JSONB LANGUAGE sql STABLE AS 'SELECT ''{}''::jsonb'"
            )
            cursor.execute(
                (
                    SCRIPTS / LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION
                ).read_text(encoding="utf-8")
            )
            cursor.execute(
                (
                    SCRIPTS / LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION
                ).read_text(encoding="utf-8")
            )
            cursor.execute(
                "SELECT to_regclass('public.' || name) FROM unnest(%s::text[]) name",
                (list(RETIRED_INCENTIVE_TABLES),),
            )
            assert all(row[0] is None for row in cursor.fetchall())
            cursor.execute(
                "SELECT to_regclass('public.research_lab_epoch_payouts')"
            )
            assert cursor.fetchone()[0] is None
            cursor.execute(
                "SELECT to_regprocedure('public.' || name || '()') "
                "FROM unnest(%s::text[]) name",
                (list(TEMPORARY_WEIGHT_TRIGGER_FUNCTIONS),),
            )
            assert all(row[0] is None for row in cursor.fetchall())
            cursor.execute(
                "SELECT (SELECT id FROM public.research_lab_attested_execution_results_v2), "
                "(SELECT id FROM public.transparency_log)"
            )
            assert cursor.fetchone() == (9, 10)
            cursor.execute(
                "SELECT to_regclass('public.research_lab_scoring_runs'), "
                "to_regclass('public.lab_arena_accepted_weight_states'), "
                "to_regprocedure('public.persist_research_lab_chain_realized_test()')"
            )
            assert cursor.fetchone() == (
                "research_lab_scoring_runs",
                "lab_arena_accepted_weight_states",
                None,
            )
            cursor.execute("SELECT id FROM public.research_lab_scoring_runs")
            assert cursor.fetchone() == (7,)
            cursor.execute(
                "SELECT to_regprocedure('public.research_lab_compact_checkpoint_graph_contract_v1()')"
            )
            assert cursor.fetchone()[0] is not None
            cursor.execute(
                "SELECT id FROM public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()"
            )
            assert cursor.fetchone() == (1,)
            cursor.execute(
                "SELECT to_regprocedure('public.' || name || '()') "
                "FROM unnest(%s::text[]) name",
                (list(RETIRED_EPOCH_WRITE_FUNCTIONS),),
            )
            assert all(row[0] is None for row in cursor.fetchall())
            cursor.execute(
                "SELECT to_regclass('public.research_lab_stateful_subnet_epoch_mapping_v1'), "
                "to_regprocedure('public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()')"
            )
            mapping_view, public_reader = cursor.fetchone()
            assert mapping_view is None and public_reader is not None
            cursor.execute(
                "SELECT to_regclass('public.' || name) FROM unnest(%s::text[]) name",
                (list(SHARED_EPOCH_TABLES),),
            )
            assert all(row[0] is not None for row in cursor.fetchall())
            cursor.execute(
                "SELECT confrelid::regclass::text FROM pg_constraint "
                "WHERE conrelid = 'public.research_lab_stateful_subnet_epoch_cutovers_v1'::regclass "
                "AND contype = 'f'"
            )
            assert cursor.fetchall() == [
                ("research_lab_stateful_subnet_epoch_candidates_v1",)
            ]
            cursor.execute(
                (
                    SCRIPTS / LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION
                ).read_text(encoding="utf-8")
            )
            cursor.execute("SELECT public.lab_arena_incentive_retirement_schema_v1()")
            assert cursor.fetchone()[0]["version"] == 203
        yield psycopg2, dsn
    finally:
        connection.close()
        database.close()


class _Era:
    def encode(self, value): self.value = value
    def birth(self, current): return current - (current % int(self.value["period"]))


class _ExternalSource:
    def __init__(self, hotkeys, validator_public_hex, genesis, *, epoch=32001):
        self.hotkeys = list(hotkeys); self.validator_public_hex = validator_public_hex
        self.genesis = genesis; self.epoch = epoch
        self.included = False; self.revealed = False
        self.extrinsic = None; self.expected_weights = []

    def snapshot(self):
        return {
            "header": {"block": 104}, "finalized_block_hash": "0x" + "2" * 64,
            "epoch_authority": {"settlement_epoch_id": self.epoch, "last_epoch_block": 100,
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
            "reveal_block": 110, "reveal_block_hash": "4" * 64,
            "validator_uid": 7, "last_update": 110,
            "weights": list(self.expected_weights),
            "transition_hash": "sha256:" + "6" * 64,
        }


class _Drand:
    def generate_commit(self, **_): return b"c" * 32, 12


def _local_signer_client(*, key, source, profile, arena_signer, burn):
    signer = ArenaWeightSigner(
        validator_hotkey=key.ss58_address,
        hotkey_public_key_hex=key.public_key.hex(),
        chain_profile=profile,
        chain_source=source,
        drand_backend=_Drand(),
        sign_sr25519=key.sign,
        arena_public_key_der=arena_signer.public_key_der,
        verify_sr25519=lambda signature, message: key.verify(message, signature),
        arena_public_key_hash=arena_signer.public_key_hash,
        network="finney",
        netuid=71,
        burn_hotkey=burn,
    )
    return LocalArenaWeightSigner(
        signer,
        chain_source=source,
        extrinsic_period=int(profile["extrinsic_period"]),
        validator_hotkey=key.ss58_address,
        network="finney",
        netuid=71,
        chain_profile=profile,
    )


def _weight_request(
    key, *, epoch, network="finney", netuid=71, timestamp=None,
    round_id="weight-state",
):
    return contracts.build_signed_request(
        scope=contracts.SCOPE_WEIGHT_STATE,
        round_id=round_id,
        hotkey=key.ss58_address,
        body={"epoch": int(epoch), "network": network, "netuid": netuid},
        timestamp=int(timestamp or datetime.now(timezone.utc).timestamp()),
        sign_message=lambda message: key.sign(message.encode("utf-8")).hex(),
    )


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
    def finalized_weight_submission_context(self, _hotkey):
        return self.finalized_head(), self.refresh_metagraph(), True


class _Api:
    def __init__(self, service): self.service = service
    def signing_key(self): return self.service.signing_key_document()
    def accepted_weight_state(self, epoch): return self.service.public_weight_state(epoch)["state"]
    def submit_chain_outcome(self, document): return self.service.record_chain_outcome(document)


@pytest.mark.parametrize(
    ("round_day", "round_epoch", "reward_epoch", "uses_prior_basis"),
    ((30, 32000, 32001, False), (31, 32002, 32004, True)),
    ids=("new-scoring-basis", "no-new-miner-or-scoring"),
)
def test_scoring_reward_two_normal_validators_restart_and_chain_readback(
    integrated_database, tmp_path, monkeypatch,
    round_day, round_epoch, reward_epoch, uses_prior_basis,
):
    psycopg2, dsn = integrated_database
    connect = lambda: psycopg2.connect(**dsn)
    with connect() as db, db.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_schema_version_v1(), public.lab_arena_weight_state_schema_v1()")
        core_schema, weight_schema = cursor.fetchone()
    assert core_schema == {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197}
    assert weight_schema == {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202}
    harness = Harness(connect, tmp_path, challengers=["NormalWinner"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(harness.service.config.defaults, rewards_enabled=True)
    # Beta is a valid but unplanned worker: runner configuration cannot gate it.
    harness.service.config.defaults.runner_hotkeys = (harness.runner_keys[0],)
    harness.chain.stakes[harness.runner_keys[1]] = 75_000
    harness.chain.active[harness.runner_keys[1]] = False
    participants = _start_round(harness, day=round_day, epoch=round_epoch)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("published", runners=2)
    repository_root = tmp_path / "promotion"; repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-cache")
    assert harness.service.promote_pending_baselines()["status"] == "ok"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"

    burn = Keypair.create_from_uri("//ArenaFlowBurn").ss58_address
    harness.service.config.accepted_burn_hotkey = burn
    harness.chain.accepted_weight_epoch_scope = lambda: {"genesis_hash": "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03", "epoch": reward_epoch, "valid_from_block": 100, "valid_until_block": 459}
    harness.chain.epoch = reward_epoch
    state = harness.service.public_weight_state(reward_epoch)["state"]
    assert state == harness.service.public_weight_state(reward_epoch)["state"]
    if uses_prior_basis:
        # No new round, miner, or scoring run produced this state. The last
        # activated signed basis remains the governing economic authority.
        assert state["reward_basis"]["effective_reward_epoch"] == round_epoch + 1
        assert harness.service.public_reward_basis(reward_epoch) == state["reward_basis"]
    with pytest.raises(ArenaStoreError, match="weight_state_conflict"):
        harness.service.store.publish_weight_state(
            "finney", 71, reward_epoch, "sha256:" + "0" * 64,
            {**state, "state_hash": "sha256:" + "0" * 64},
        )
    harness.clock.now = datetime.now(timezone.utc)

    profile = load_chain_signing_profile(Path("validator_tee/enclave/chain_signing_profile_v2.json"))
    outcomes, vectors = [], []
    miner = Keypair.create_from_uri("//ArenaWeightMiner")
    harness.chain.runners.append(miner.ss58_address)
    harness.chain.permits[miner.ss58_address] = False
    outsider = Keypair.create_from_uri("//ArenaWeightOutsider")
    permitted = Keypair.create_from_uri("//ArenaWeightPermitted")
    harness.chain.runners.append(permitted.ss58_address)
    harness.chain.stakes[permitted.ss58_address] = 1
    weight_reads = []
    original_public_weight_state = harness.service.public_weight_state
    monkeypatch.setattr(
        harness.service, "public_weight_state",
        lambda epoch: weight_reads.append(epoch) or original_public_weight_state(epoch),
    )
    with TestClient(create_app(harness.service)) as public:
        assert public.post("/arena/v1/weight-state", json={}).status_code == 400
        denied_miner = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(miner, epoch=reward_epoch),
        )
        assert denied_miner.status_code == 403
        assert denied_miner.json()["code"] == "runner_validator_required"
        denied_outsider = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(outsider, epoch=reward_epoch),
        )
        assert denied_outsider.status_code == 403
        assert denied_outsider.json()["code"] == "runner_hotkey_unregistered"
        forged = _weight_request(permitted, epoch=reward_epoch)
        forged["signature"] = "0x" + "00" * 64
        assert public.post(
            "/arena/v1/weight-state", json=forged
        ).status_code == 401
        stale = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(
                miner, epoch=reward_epoch,
                timestamp=int(harness.clock.now.timestamp()) - 301,
            ),
        )
        assert stale.status_code == 400
        wrong_action = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(
                permitted, epoch=reward_epoch, round_id="not-weight-state",
            ),
        )
        assert wrong_action.status_code == 400
        wrong_scope = _weight_request(permitted, epoch=reward_epoch)
        wrong_scope["scope"] = contracts.SCOPE_CLAIM
        wrong_scope["signature"] = "0x" + permitted.sign(
            contracts.signed_request_message(wrong_scope).encode("utf-8")
        ).hex()
        assert public.post(
            "/arena/v1/weight-state", json=wrong_scope
        ).status_code == 400
        wrong_network = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(
                permitted, epoch=reward_epoch, network="test",
            ),
        )
        assert wrong_network.status_code == 400
        wrong_netuid = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(permitted, epoch=reward_epoch, netuid=72),
        )
        assert wrong_netuid.status_code == 400
        non_integer_netuid = public.post(
            "/arena/v1/weight-state",
            json=_weight_request(permitted, epoch=reward_epoch, netuid=71.0),
        )
        assert non_integer_netuid.status_code == 400
        with monkeypatch.context() as patcher:
            patcher.setattr(
                harness.chain, "metagraph",
                lambda finalized=True: (_ for _ in ()).throw(
                    RuntimeError("private chain diagnostic")
                ),
            )
            unavailable = public.post(
                "/arena/v1/weight-state",
                json=_weight_request(miner, epoch=reward_epoch),
            )
        assert unavailable.status_code == 503
        assert unavailable.json()["code"] == "validator_snapshot_unavailable"
        assert "private chain diagnostic" not in unavailable.text
        assert weight_reads == []
        assert public.get(
            "/arena/v1/reward-basis", params={"epoch": reward_epoch}
        ).status_code == 404
        public_round = public.get(
            "/arena/v1/rounds/%s" % harness.round_id
        ).json()
        assert "reward_basis" not in public_round
    for index in range(2):
        key = Keypair.create_from_uri("//ArenaNormalValidator%d" % index)
        harness.chain.runners.append(key.ss58_address)
        harness.chain.stakes[key.ss58_address] = 1 if index == 0 else 75_000
        harness.service._require_validator_authority(key.ss58_address)
        if index == 0:
            with pytest.raises(ServiceError, match="runner_stake_below_minimum"):
                harness.service._benchmark_validator_uid(harness.chain.metagraph(), key.ss58_address)
        with TestClient(create_app(harness.service)) as public:
            assert public.get(
                "/arena/v1/weight-state", params={"epoch": reward_epoch}
            ).status_code == 405
            response = public.post(
                "/arena/v1/weight-state",
                json=_weight_request(key, epoch=reward_epoch),
            )
            assert response.status_code == 200 and response.json()["state"] == state
        hotkeys = [burn]
        if state["reward_basis"]["king_hotkey"]:
            hotkeys.append(state["reward_basis"]["king_hotkey"])
        source = _ExternalSource(
            hotkeys,
            key.public_key.hex(),
            profile["genesis_hash"],
            epoch=reward_epoch,
        )
        signer_client = _local_signer_client(
            key=key,
            source=source,
            profile=profile,
            arena_signer=harness.signer,
            burn=burn,
        )
        host_chain = _HostChain(source, key.ss58_address)
        paths = ArenaWeightPaths(tmp_path / ("v%d" % index))
        orchestrator = ArenaWeightOrchestrator(
            api=_Api(harness.service), chain=host_chain, signer=signer_client,
            validator_hotkey=key.ss58_address, expected_signing_key_hash=harness.signer.public_key_hash,
            paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
        )
        assert orchestrator.run_once(reward_epoch) == "broadcast"
        signed_path = paths.signed(reward_epoch)
        outcome_path = paths.outcome(reward_epoch)
        signed_bytes = signed_path.read_bytes()
        signed = json.loads(signed_bytes)
        vectors.append((signed["sparse_uids"], signed["sparse_weights_u16"]))
        source.expected_weights = list(zip(signed["sparse_uids"], signed["sparse_weights_u16"]))
        # A new in-process signer has no memory of the first attempt.  It must
        # authenticate and restore the exact bytes from the durable journal.
        restarted_signer = _local_signer_client(
            key=key,
            source=source,
            profile=profile,
            arena_signer=harness.signer,
            burn=burn,
        )
        restarted = ArenaWeightOrchestrator(
            api=_Api(harness.service), chain=host_chain, signer=restarted_signer,
            validator_hotkey=key.ss58_address, expected_signing_key_hash=harness.signer.public_key_hash,
            paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
        )
        assert restarted.run_once(reward_epoch) == "rebroadcast"
        assert signed_path.read_bytes() == signed_bytes
        source.included = True
        assert restarted.run_once(reward_epoch) == "included_pending_reveal"
        assert signed_path.read_bytes() == signed_bytes
        assert not outcome_path.exists()
        assert len(host_chain.broadcasts) == 2
        source.revealed = True
        assert restarted.run_once(reward_epoch) == "finalized"
        outcomes.append(outcome_path.read_bytes())
    assert len(harness.service.public_chain_outcomes(reward_epoch)["outcomes"]) == 2
    assert outcomes[0] != outcomes[1]
    assert vectors[0] == vectors[1]
    first_report = harness.service.public_chain_outcomes(reward_epoch)["outcomes"][0]
    assert first_report["finalized_block_hash"] == "4" * 64
    assert first_report["extrinsic_hash"].startswith("0x")
    with connect() as db, db.cursor() as cursor:
        cursor.execute("DELETE FROM public.lab_arena_chain_outcomes WHERE request_id = %s", (first_report["request_id"],))
        db.commit()
    harness.clock.now += timedelta(minutes=10)
    assert harness.service.record_chain_outcome(first_report) == {"status": "recorded"}
    assert harness.service.record_chain_outcome(first_report) == {"status": "recorded"}
    for invalid_hash in ("0x" + "4" * 64, "4" * 63, "g" * 64, "A" * 64):
        with pytest.raises(Exception, match="finalized_block_hash is invalid"):
            harness.service.record_chain_outcome(
                {
                    **first_report,
                    "finalized_block_hash": invalid_hash,
                    "request_id": "sha256:" + "f" * 64,
                }
            )
    assert_canary_absent(harness, connect)
    with pytest.raises(Exception, match="not_current"):
        harness.service.public_weight_state(reward_epoch + 1)
