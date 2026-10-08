"""One executable proof from Arena scoring through normal-validator outcomes."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from lab_arena import broker, contracts, rewards, scoring, signing, submission_runtime
from leadpoet_canonical import arena_weights
from lab_arena.api import create_app
from lab_arena.service import ServiceError
from lab_arena.local_weight_signer import (
    LocalArenaWeightSigner,
    load_public_chain_signing_profile,
)
from lab_arena.store import ArenaStoreError
from lab_arena.promotion import GitPromoter
from lab_arena.validator import (
    ArenaPublicApi,
    ArenaWeightOrchestrator,
    ArenaWeightPaths,
)
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
    LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION,
    CURRENT_REWARD_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, _run_stage_one_to_scoring, _start_round, promotion_repository,
    assert_canary_absent, CANARY_KEYS, FakeProviderTransport, price_table, daily_icps,
)
from tests.postgres_migration_harness import SCRIPTS
from lab_arena.weight_signer import ArenaWeightSigner

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
    staged_migrations += ("216-lab-arena-validator-participation.sql",)
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
            # Finish the current schema after proving the retirement upgrade.
            # New rounds use the same cost/publication policy as production.
            applied_migrations = staged_migrations + (
                LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION,
                LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
            )
            for migration in CURRENT_REWARD_SERVICE_MIGRATIONS:
                if migration not in applied_migrations:
                    cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
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
            from lab_arena.chain_source import ValidatorChainSourceV2Error
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


class _BufferedResponse:
    """The small urllib response surface used by ``ArenaPublicApi``."""

    def __init__(self, body):
        self._body = bytes(body)

    def read(self, limit=-1):
        return self._body if limit is None or limit < 0 else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _test_client_urlopen(client):
    """Route urllib calls into FastAPI without permitting a network fallback."""

    def urlopen(request, *, timeout):
        assert timeout > 0
        parsed = urlsplit(request.full_url)
        assert (parsed.scheme, parsed.netloc) == ("https", "arena.example")
        target = parsed.path + (("?" + parsed.query) if parsed.query else "")
        response = client.request(
            request.get_method(), target, content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url, response.status_code, "test gateway response",
                dict(response.headers), io.BytesIO(response.content),
            )
        return _BufferedResponse(response.content)

    return urlopen


def _public_api(key, clock):
    return ArenaPublicApi(
        "https://arena.example", keypair=key, network="finney", netuid=71,
        now=lambda: clock().timestamp(),
    )


class _NormalWeightHarness(Harness):
    """Use the normal per-provider payer path after the first real crown."""

    def build_service(self):
        service = super().build_service()
        payer = submission_runtime.SubmissionProviderKeys(
            store=service.store, credentials=service.config.credential_manager,
            organizer_keys=CANARY_KEYS,
        )

        def broker_factory(_service, _round_row):
            return broker.Broker(
                store=service.store, key_for=lambda provider: CANARY_KEYS[provider],
                credential_for=payer.credential_for,
                funding_source_for=payer.funding_source_for,
                provider_funding_source_for=payer.provider_funding_source_for,
                retry_miner_credential_for=payer.retry_miner_credential_for,
                mark_provider_fallback=payer.mark_provider_fallback,
                provider_restart_required_for=payer.provider_restart_required_for,
                price_table=price_table(), judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
                transport=FakeProviderTransport(), clock=self.clock,
            )

        service.config.broker_factory = broker_factory
        return service


@pytest.mark.parametrize(
    ("round_day", "round_epoch", "reward_epoch", "uses_prior_basis"),
    ((30, 32000, 32001, False), (31, 32002, 32004, True)),
    ids=("new-scoring-basis", "no-new-miner-or-scoring"),
)
def test_scoring_reward_normal_validators_restart_and_chain_readback(
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
    # These actual judge fixtures score 83.6 against the public baseline's
    # 69.8. No published score is replaced to manufacture an achievement.
    harness = _NormalWeightHarness(connect, tmp_path, challengers=["NormalSlotWinner71"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(harness.service.config.defaults, rewards_enabled=True)
    # Beta is a valid but unplanned worker: runner configuration cannot gate it.
    harness.service.config.defaults.runner_hotkeys = (harness.runner_keys[0],)
    harness.chain.stakes[harness.runner_keys[1]] = 75_000
    harness.chain.active[harness.runner_keys[1]] = False
    working_key = Keypair.create_from_uri("//svc-runner-alpha")
    assert working_key.ss58_address == harness.runner_keys[0]
    harness.clock.now = datetime.now(timezone.utc)
    if not uses_prior_basis:
        with TestClient(create_app(harness.service)) as http, monkeypatch.context() as patcher:
            patcher.setattr(urllib.request, "urlopen", _test_client_urlopen(http))
            patcher.setattr(
                harness.service.store,
                "has_recent_participation",
                lambda *_: (_ for _ in ()).throw(
                    AssertionError("weight access queried participation")
                ),
            )
            assert _public_api(working_key, harness.clock).accepted_weight_state(
                reward_epoch
            ) is None
    participants = _start_round(harness, day=round_day, epoch=round_epoch)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("published", runners=2)
    repository_root = tmp_path / "promotion"; repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-cache")
    assert harness.service.promote_pending_baselines()["status"] == "ok"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"

    activated = harness.service.store.get_round(harness.round_id)
    basis = activated["reward_basis_doc"]
    assert basis["schema_version"] == "leadpoet.lab_arena.reward_basis.v2"
    assert basis["slot_policy"] == rewards.reward_slot_policy_document()
    achievement = basis["reward_slots"][0]
    assert achievement is not None
    assert basis["reward_slots"] == [achievement] * 3
    assert achievement["winner_score"] - achievement["baseline_score"] >= 10
    source_round = harness.service.store.get_round(achievement["round_id"])
    assert source_round["baseline_promoted_at"] is not None
    ranking = source_round["publication_doc"]["final_ranking"]
    for field, submission_field in (
        ("winner_score", "submission_id"),
        ("baseline_score", "baseline_submission_id"),
    ):
        published = next(row for row in ranking if row["submission_id"] == achievement[submission_field])
        assert achievement[field] == published["final_score"]

    burn = Keypair.create_from_uri("//ArenaFlowBurn").ss58_address
    harness.service.config.accepted_burn_hotkey = burn
    harness.chain.accepted_weight_epoch_scope = lambda: {"genesis_hash": "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03", "epoch": reward_epoch, "valid_from_block": 100, "valid_until_block": 459}
    harness.chain.epoch = reward_epoch
    # A fresh gateway process must recover the accepted state from PostgreSQL.
    restarted_gateway = harness.build_service()
    restarted_gateway.config.accepted_burn_hotkey = burn
    with TestClient(create_app(restarted_gateway)) as http, monkeypatch.context() as patcher:
        patcher.setattr(urllib.request, "urlopen", _test_client_urlopen(http))
        working_api = _public_api(working_key, harness.clock)
        state = working_api.accepted_weight_state(reward_epoch)
        assert state is not None
        assert state == working_api.accepted_weight_state(reward_epoch)
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

    expired_key = Keypair.create_from_uri("//svc-runner-beta")
    assert restarted_gateway.store.has_recent_participation(
        "finney", 71, working_key.ss58_address
    )
    assert not restarted_gateway.store.has_recent_participation(
        "finney", 71, expired_key.ss58_address
    )

    # The high-stake validator receives the signed state after its accepted
    # jobs. Weight access is separate from the scoring participation record.
    with TestClient(create_app(restarted_gateway)) as http:
        allowed = http.post(
            "/arena/v1/weight-state",
            json=_weight_request(working_key, epoch=reward_epoch),
        )
        assert allowed.status_code == 200 and allowed.json()["state"] == state
        assert allowed.headers["cache-control"] == "no-store"

    # Build a historical accepted-job record without changing the production
    # clock. The participation trigger is disabled only for this fixture write.
    with connect() as db, db.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
            "lab_arena_runs_participation"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET participation_accepted_at = "
            "clock_timestamp() - interval '25 hours' "
            "WHERE runner_hotkey = %s AND participation_accepted_at IS NOT NULL",
            (working_key.ss58_address,),
        )
        assert cursor.rowcount > 0
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
            "lab_arena_runs_participation"
        )
    restarted_gateway = harness.build_service()
    restarted_gateway.config.accepted_burn_hotkey = burn
    assert not restarted_gateway.store.has_recent_participation(
        "finney", 71, working_key.ss58_address
    )

    profile = load_public_chain_signing_profile(
        "finney",
        path=Path("lab_arena/chain_signing_profile_v2.json"),
    )
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
    no_work_key = Keypair.create_from_uri("//ArenaNormalValidatorNoWork")
    signing_cases = (
        ("expired-accepted-work", working_key, 100_000),
        ("no-accepted-work", no_work_key, 100_000),
        ("exact-threshold-no-work", expired_key, 75_000),
        ("below-threshold", Keypair.create_from_uri("//ArenaNormalValidator0"), 1),
    )
    with TestClient(create_app(restarted_gateway)) as http, monkeypatch.context() as patcher:
        patcher.setattr(urllib.request, "urlopen", _test_client_urlopen(http))
        for label, key, stake in signing_cases:
            if key.ss58_address not in harness.chain.runners:
                harness.chain.runners.append(key.ss58_address)
            harness.chain.stakes[key.ss58_address] = stake
            restarted_gateway._require_validator_authority(key.ss58_address)
            if stake < 75_000:
                with pytest.raises(ServiceError, match="runner_stake_below_minimum"):
                    restarted_gateway._benchmark_validator_uid(
                        harness.chain.metagraph(), key.ss58_address
                    )
            assert not restarted_gateway.store.has_recent_participation(
                "finney", 71, key.ss58_address
            )
            api = _public_api(key, harness.clock)
            assert api.accepted_weight_state(reward_epoch) == state
            hotkeys = [burn]
            if state["reward_basis"]["king_hotkey"]:
                hotkeys.append(state["reward_basis"]["king_hotkey"])
            derived = arena_weights.derive_arena_weights(state, hotkeys)
            assert derived["champion_share_ppb"] == 300_000_000
            assert derived["burned_residual_ppb"] == 700_000_000
            assert derived["sparse_uids"] == [0, 1]
            assert derived["sparse_weights_u16"] == [65535, 28086]
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
            paths = ArenaWeightPaths(tmp_path / label)
            orchestrator = ArenaWeightOrchestrator(
                api=api, chain=host_chain, signer=signer_client,
                validator_hotkey=key.ss58_address, expected_signing_key_hash=harness.signer.public_key_hash,
                paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
            )
            assert orchestrator.run_once(reward_epoch) == "broadcast"
            signed_path = paths.signed(reward_epoch)
            outcome_path = paths.outcome(reward_epoch)
            signed_bytes = signed_path.read_bytes()
            signed = json.loads(signed_bytes)
            assert signed["accepted_state"]["reward_basis"] == basis
            assert signed["sparse_uids"] == derived["sparse_uids"]
            assert signed["sparse_weights_u16"] == derived["sparse_weights_u16"]
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
                api=api, chain=host_chain, signer=restarted_signer,
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
    assert len(harness.service.public_chain_outcomes(reward_epoch)["outcomes"]) == 4
    assert len(set(outcomes)) == 4
    assert all(vector == vectors[0] for vector in vectors)
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


def test_v2_multiple_payees_preserves_pending_v1_signed_recovery(tmp_path):
    """New slot signing must not change an older journal's signed authority."""
    arena_signer = signing.LocalSigner.generate()
    key = Keypair.create_from_uri("//ArenaSlotRecoveryValidator")
    burn = Keypair.create_from_uri("//ArenaSlotRecoveryBurn").ss58_address
    owners = [Keypair.create_from_uri("//ArenaSlotRecoveryOwner%d" % i).ss58_address for i in range(3)]
    profile = load_public_chain_signing_profile(
        "finney", path=Path("lab_arena/chain_signing_profile_v2.json"),
    )
    old_epoch, new_epoch = 33000, 33001
    legacy_slot_policy = rewards.reward_slot_policy_document()
    legacy_slot_policy.pop("decay", None)

    def signed_state(epoch, *, slots=None):
        basis = rewards.reward_basis_document(
            round_id="recovery-%d" % epoch, published_at="2026-10-08T00:00:00Z",
            finalized_epoch=epoch - 1, king_outcome="crowned", king_hotkey=owners[0],
            slot_policy=legacy_slot_policy if slots is not None else None,
            reward_slots=slots,
        )
        basis = signing.sign_document(arena_signer, basis, hash_field="reward_basis_hash")
        body = {
            "schema_version": arena_weights.ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION,
            "network": "finney", "genesis_hash": profile["genesis_hash"], "netuid": 71,
            "epoch": epoch, "valid_from_block": 100, "valid_until_block": 459,
            "reward_basis": basis, "burn_hotkey": burn, "issued_at": "2026-10-08T00:00:00Z",
        }
        body["state_hash"] = contracts.document_hash(body)
        body["signature"] = {
            "algorithm": arena_signer.algorithm,
            "public_key_hash": arena_signer.public_key_hash,
            "signature_b64": base64.b64encode(arena_signer.sign(
                (arena_weights.WEIGHT_STATE_SIGNATURE_PREFIX + body["state_hash"]).encode()
            )).decode(),
        }
        return body

    slots = [
        {
            "round_id": "achievement-%d" % i, "submission_id": "winner-%d" % i,
            "miner_hotkey": owner, "baseline_submission_id": "baseline-%d" % i,
            "baseline_score": 60, "winner_score": score,
        }
        for i, (owner, score) in enumerate(zip(owners, (70, 65, 61)))
    ]
    old_state = signed_state(old_epoch)
    new_state = signed_state(new_epoch, slots=slots)
    source = _ExternalSource([burn] + owners, key.public_key.hex(), profile["genesis_hash"], epoch=old_epoch)
    host = _HostChain(source, key.ss58_address)
    paths = ArenaWeightPaths(tmp_path)
    lookups, reports = [], []
    active_state = [old_state]

    def lookup(epoch):
        lookups.append(epoch)
        assert epoch == active_state[0]["epoch"]
        return active_state[0]

    api = SimpleNamespace(
        signing_key=lambda: signing.signing_key_document(arena_signer.public_key_der),
        accepted_weight_state=lookup,
        submit_chain_outcome=lambda document: reports.append(document) or {"status": "recorded"},
    )

    def orchestrator():
        return ArenaWeightOrchestrator(
            api=api, chain=host,
            signer=_local_signer_client(key=key, source=source, profile=profile, arena_signer=arena_signer, burn=burn),
            validator_hotkey=key.ss58_address, expected_signing_key_hash=arena_signer.public_key_hash,
            paths=paths, extrinsic_period=int(profile["extrinsic_period"]),
        )

    assert orchestrator().run_once(old_epoch) == "broadcast"
    old_bytes = paths.signed(old_epoch).read_bytes()
    old_signed = json.loads(old_bytes)
    assert old_signed["accepted_state"]["reward_basis"]["schema_version"] == "leadpoet.lab_arena.reward_basis.v1"
    assert old_signed["sparse_weights_u16"] == [65535, 28086]
    source.epoch = new_epoch
    active_state[0] = new_state
    restarted = orchestrator()
    assert restarted.run_once(new_epoch) == "broadcast"
    new_bytes = paths.signed(new_epoch).read_bytes()
    new_signed = json.loads(new_bytes)
    # The three registered owners split the 30% pot as 15%, 9%, and 6%
    # of total emissions; the remaining 70% stays at the burn destination.
    assert new_signed["sparse_uids"] == [0, 1, 2, 3]
    assert new_signed["sparse_weights_u16"] == [65535, 14043, 8426, 5617]
    for state, epoch, signed in (
        (old_state, old_epoch, old_signed), (new_state, new_epoch, new_signed),
    ):
        source.expected_weights = list(zip(signed["sparse_uids"], signed["sparse_weights_u16"]))
        source.included = source.revealed = True
        if epoch == old_epoch:
            restarted.poll_prior_outcomes(new_epoch)
            assert paths.outcome(old_epoch).exists()
        else:
            assert restarted.run_once(epoch) == "finalized"
        outcome = json.loads(paths.outcome(epoch).read_bytes())
        assert outcome["outcome"]["revealed_weights"] == [list(item) for item in source.expected_weights]
        assert outcome["report_document"]["state_hash"] == state["state_hash"]
    assert lookups == [old_epoch, new_epoch]
    assert [report["epoch"] for report in reports] == [old_epoch, new_epoch]
    assert paths.signed(old_epoch).read_bytes() == old_bytes
    assert paths.signed(new_epoch).read_bytes() == new_bytes


def test_slot_decay_persists_through_daily_publication_and_normal_validator_recovery(
    integrated_database, tmp_path, monkeypatch,
):
    """Published source epochs age each slot, including after policy upgrades."""
    psycopg2, dsn = integrated_database
    connect = lambda: psycopg2.connect(**dsn)
    harness = _NormalWeightHarness(
        connect, tmp_path, challengers=["NormalSlotWinner71"], runners=["alpha", "beta"],
    )
    # One ICP in each stage exercises the same configured competition path.
    # These fixtures return the same five companies and scores for every ICP.
    harness.service.config.defaults = replace(
        harness.service.config.defaults, rewards_enabled=True, benchmark_icp_count=2,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]), "icps": daily_icps()[:2],
    }
    repository_root = tmp_path / "decay-promotion"
    repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "decay-promotion-cache",
    )

    def promoted_baseline(_url, _limit):
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout

    def publish(day, epoch, challengers, *, baseline_flavor="PublicBaseline", legacy=False):
        harness.challengers = challengers
        participants = _start_round(harness, day=day, epoch=epoch)
        round_row = harness.service.store.get_round(harness.round_id)
        for participant in round_row["participants"]:
            if participant["is_king"]:
                harness.flavors[participant["submission_id"]] = baseline_flavor
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        assert harness.service.advance_round(harness.round_id)["assignments"] == participants
        harness.advance_until("published", runners=2)
        assert harness.service.promote_pending_baselines()["status"] == "ok"
        with monkeypatch.context() as patcher:
            if legacy:
                old_policy = rewards.reward_slot_policy_document()
                old_policy.pop("decay")
                patcher.setattr(rewards, "reward_slot_policy_document", lambda: old_policy)
            assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
        return harness.service.store.get_round(harness.round_id)["reward_basis_doc"]

    # Genuine published fixtures produce 83.6 versus 69.8. This legacy v2
    # activation has no decay fields; its original epoch still starts the clock
    # when a later daily basis first enables decay.
    original = publish(40, 34000, ["NormalSlotWinner71"], legacy=True)
    assert original["effective_reward_epoch"] == 34001
    assert "decay" not in original["slot_policy"]
    assert all("start_epoch" not in slot for slot in original["reward_slots"])
    original_bytes = json.dumps(original, sort_keys=True)
    owner = original["reward_slots"][0]["miner_hotkey"]
    assert original["reward_slots"][0]["winner_score"] == pytest.approx(83.6)
    assert original["reward_slots"][0]["baseline_score"] == pytest.approx(69.8)
    harness.service.config.baseline_source_fetcher = promoted_baseline

    burn = Keypair.create_from_uri("//ArenaDecayFlowBurn").ss58_address
    harness.service.config.accepted_burn_hotkey = burn
    profile = load_public_chain_signing_profile(
        "finney", path=Path("lab_arena/chain_signing_profile_v2.json"),
    )
    keys = [Keypair.create_from_uri("//ArenaDecayFlowValidator%d" % i) for i in range(2)]
    for key in keys:
        harness.chain.runners.append(key.ss58_address)
        harness.chain.stakes[key.ss58_address] = 1
    paths = [ArenaWeightPaths(tmp_path / ("decay-validator-%d" % i)) for i in range(2)]

    def verify_normal_flow(basis, epoch, expected_share_ppb, *, replacement_owner=None):
        harness.chain.epoch = epoch
        harness.chain.accepted_weight_epoch_scope = lambda: {
            "genesis_hash": profile["genesis_hash"], "epoch": epoch,
            "valid_from_block": 100, "valid_until_block": 459,
        }
        harness.clock.now = datetime.now(timezone.utc)
        restarted_gateway = harness.build_service()
        restarted_gateway.config.accepted_burn_hotkey = burn
        metagraph = [burn, owner] + ([replacement_owner] if replacement_owner else [])
        vectors = []
        with TestClient(create_app(restarted_gateway)) as http, monkeypatch.context() as patcher:
            patcher.setattr(urllib.request, "urlopen", _test_client_urlopen(http))
            for index, key in enumerate(keys):
                api = _public_api(key, harness.clock)
                state = api.accepted_weight_state(epoch)
                assert state is not None and state["reward_basis"] == basis
                derived = arena_weights.derive_arena_weights(state, metagraph)
                assert derived["champion_share_ppb"] == expected_share_ppb
                assert derived["burned_residual_ppb"] == 1_000_000_000 - expected_share_ppb
                source = _ExternalSource(metagraph, key.public_key.hex(), profile["genesis_hash"], epoch=epoch)
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
                vectors.append((signed["sparse_uids"], signed["sparse_weights_u16"]))
                assert vectors[-1] == (derived["sparse_uids"], derived["sparse_weights_u16"])
                source.expected_weights = list(zip(*vectors[-1]))
                restarted = orchestrator()
                assert restarted.run_once(epoch) == "rebroadcast"
                source.included = True
                assert restarted.run_once(epoch) == "included_pending_reveal"
                source.revealed = True
                assert restarted.run_once(epoch) == "finalized"
                assert paths[index].signed(epoch).read_bytes() == signed_bytes
                outcome = json.loads(paths[index].outcome(epoch).read_bytes())
                assert outcome["outcome"]["revealed_weights"] == [list(item) for item in source.expected_weights]
                assert outcome["report_document"]["state_hash"] == state["state_hash"]
        assert vectors[0] == vectors[1]
        assert len(harness.service.public_chain_outcomes(epoch)["outcomes"]) == 2
        return vectors[0]

    first_carry = publish(41, 34139, [], baseline_flavor="NormalSlotWinner71")
    assert first_carry["slot_policy"]["decay"] == {"epochs_per_halving": 140, "max_halvings": 4}
    assert [slot["start_epoch"] for slot in first_carry["reward_slots"]] == [34001] * 3
    before = verify_normal_flow(first_carry, 34140, 300_000_000)
    after = verify_normal_flow(first_carry, 34141, 150_000_000)
    assert before != after

    # The promoted archive is now the next round's genuine 83.6 baseline.
    # The 86.2 fixture qualifies only for +1 and cannot reset either older slot.
    replacement = publish(42, 34279, ["SlotDecayReplacement22768"], baseline_flavor="NormalSlotWinner71")
    assert [slot["start_epoch"] for slot in replacement["reward_slots"]] == [34001, 34001, None]
    assert [slot["round_id"] for slot in replacement["reward_slots"]] == [original["round_id"], original["round_id"], replacement["round_id"]]
    replaced = replacement["reward_slots"][2]
    assert replaced["winner_score"] == pytest.approx(86.2)
    assert replaced["baseline_score"] == pytest.approx(83.6)
    replacement_owner = replaced["miner_hotkey"]
    assert replacement_owner != owner
    verify_normal_flow(replacement, 34280, 180_000_000, replacement_owner=replacement_owner)
    verify_normal_flow(replacement, 34281, 120_000_000, replacement_owner=replacement_owner)

    # Daily publication refreshes basis eligibility, while each slot keeps its
    # own source clock. Explicit expected totals also prove residuals burn.
    for day, finalized_epoch, before_share, after_share in (
        (43, 34419, 90_000_000, 60_000_000),
        (44, 34559, 45_000_000, 30_000_000),
        (45, 34699, 22_500_000, 22_500_000),
    ):
        carried = publish(day, finalized_epoch, [], baseline_flavor="SlotDecayReplacement22768")
        assert [slot["start_epoch"] for slot in carried["reward_slots"]] == [34001, 34001, 34280]
        assert [slot["round_id"] for slot in carried["reward_slots"]] == [original["round_id"], original["round_id"], replacement["round_id"]]
        verify_normal_flow(carried, finalized_epoch + 1, before_share, replacement_owner=replacement_owner)
        verify_normal_flow(carried, finalized_epoch + 2, after_share, replacement_owner=replacement_owner)
    assert json.dumps(harness.service.store.get_round(original["round_id"])["reward_basis_doc"], sort_keys=True) == original_bytes
    assert_canary_absent(harness, connect)
