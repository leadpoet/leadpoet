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
            query=self._query,
        )
    def _query(self, *, module, storage_function, params, block_hash):
        assert module == "SubtensorModule"
        assert block_hash == "0x" + "2" * 64
        if storage_function == "Uids": return SimpleNamespace(value=7)
        if storage_function == "LastUpdate": return SimpleNamespace(value=[0] * 8)
        if storage_function == "WeightsSetRateLimit": return SimpleNamespace(value=100)
        raise AssertionError(storage_function)
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
    harness.service.config.accepted_burn_hotkey = burn
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
    outcomes = []
    for index in range(2):
        key = Keypair.create_from_uri("//ArenaNormalValidator%d" % index)
        hotkeys = [burn]
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
