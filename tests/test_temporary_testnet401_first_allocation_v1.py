from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
from gateway.research_lab import (
    champion_settlement_v2 as settlement,
    temporary_testnet401_first_allocation_v1 as origin,
    v2_authority,
)
from gateway.tee import coordinator_allocation_source_v2 as allocation_source
from gateway.tee.coordinator_allocation_source_v2 import (
    CoordinatorAllocationSourceV2,
    CoordinatorAllocationSourceV2Error,
)
from gateway.tee.coordinator_chain_source_v2 import (
    CoordinatorChainSourceV2,
    CoordinatorChainSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    ExecutionJobV2Error,
)
from gateway.tee.provider_broker_v2 import (
    ProviderBrokerV2,
    credential_reference_hash,
    expected_provider_credential_slots,
    measured_retry_policy_hashes,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from leadpoet_canonical.attested_v2 import merkle_root
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    frontier_artifact_hashes_v2,
)
from leadpoet_canonical.chain_source_v2 import last_update_storage_key


def _cutover() -> dict:
    return {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": origin.TESTNET401_GENESIS_HASH,
        "netuid": origin.TESTNET401_NETUID,
        "cutover_block": origin.TESTNET401_CUTOVER_BLOCK,
        "cutover_block_hash": origin.TESTNET401_CUTOVER_BLOCK_HASH,
        "first_subnet_epoch_index": origin.TESTNET401_FIRST_SETTLEMENT_EPOCH,
        "first_settlement_epoch_id": origin.TESTNET401_FIRST_SETTLEMENT_EPOCH,
        "last_legacy_epoch_id": origin.TESTNET401_FIRST_SETTLEMENT_EPOCH - 1,
        "mapping_hash": origin.TESTNET401_CUTOVER_MAPPING_HASH,
    }


def _cutover_graph() -> dict:
    return {
        "root_receipt_hash": origin.TESTNET401_CUTOVER_RECEIPT_HASH,
        "receipts": [
            {
                "receipt_hash": origin.TESTNET401_CUTOVER_RECEIPT_HASH,
                "role": "gateway_coordinator",
                "purpose": "research_lab.subnet_epoch_cutover.v2",
                "status": "succeeded",
                "epoch_id": origin.TESTNET401_FIRST_SETTLEMENT_EPOCH,
                "output_root": origin.TESTNET401_CUTOVER_AUTHORITY_HASH,
                "parent_receipt_hashes": [origin.TESTNET401_SNAPSHOT_RECEIPT_HASH],
            }
        ],
    }


def _profile() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "validator_tee/enclave/chain_signing_profile_test_v2.json"
        ).read_text(encoding="utf-8")
    )


def _scale_u64_vector(values: list[int]) -> str:
    encoded = bytes([len(values) << 2]) + b"".join(
        int(value).to_bytes(8, "little") for value in values
    )
    return "0x" + encoded.hex()


def _scale_weight_vector(values: list[tuple[int, int]]) -> str:
    encoded = bytes([len(values) << 2]) + b"".join(
        int(uid).to_bytes(2, "little") + int(weight).to_bytes(2, "little")
        for uid, weight in values
    )
    return "0x" + encoded.hex()


def _chain_source() -> CoordinatorChainSourceV2:
    return CoordinatorChainSourceV2(
        execute_provider=lambda _request: {},
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": _cutover(),
            "chain_signing_profile": _profile(),
        },
    )


def test_chain_origin_uses_real_test_profile_and_exact_finalized_state(monkeypatch):
    source = _chain_source()
    hotkeys = ["unused-%d" % index for index in range(12)]
    hotkeys[origin.TESTNET401_BURN_UID] = origin.TESTNET401_BURN_HOTKEY
    hotkeys[origin.TESTNET401_VALIDATOR_UID] = origin.TESTNET401_VALIDATOR_HOTKEY
    last_updates = [0] * len(hotkeys)
    last_updates[origin.TESTNET401_VALIDATOR_UID] = (
        origin.TESTNET401_PRE_CUTOVER_LAST_UPDATE
    )
    calls = []

    def chain_call(*, method, params, **_kwargs):
        calls.append((method, params))
        if params[0] == last_update_storage_key(netuid=401):
            return _scale_u64_vector(last_updates)
        return _scale_weight_vector([(origin.TESTNET401_BURN_UID, 65_535)])

    monkeypatch.setattr(source, "_chain_call", chain_call)
    context = SimpleNamespace(job_id="allocation-v2:testnet401")
    result = source.prove_fresh_testnet401_allocation_origin(
        netuid=401,
        snapshot={
            "finalized_block_hash": "b" * 64,
            "header": {"block": 7_961_507},
            "metagraph": {"block": 7_961_507, "hotkeys": hotkeys},
        },
        context=context,
    )

    assert result["last_update_block"] == 7_431_466
    assert result["weights"] == [[0, 65_535]]
    assert len(calls) == 2


def test_chain_origin_rejects_nonempty_weight_state(monkeypatch):
    source = _chain_source()
    hotkeys = ["unused-%d" % index for index in range(12)]
    hotkeys[0] = origin.TESTNET401_BURN_HOTKEY
    hotkeys[9] = origin.TESTNET401_VALIDATOR_HOTKEY
    last_updates = [0] * len(hotkeys)
    last_updates[9] = origin.TESTNET401_PRE_CUTOVER_LAST_UPDATE
    responses = iter(
        (
            _scale_u64_vector(last_updates),
            _scale_weight_vector([(0, 60_000), (11, 20_000)]),
        )
    )
    monkeypatch.setattr(source, "_chain_call", lambda **_kwargs: next(responses))

    with pytest.raises(
        CoordinatorChainSourceV2Error,
        match="finalized origin is not empty",
    ):
        source.prove_fresh_testnet401_allocation_origin(
            netuid=401,
            snapshot={
                "finalized_block_hash": "b" * 64,
                "header": {"block": 7_961_507},
                "metagraph": {"block": 7_961_507, "hotkeys": hotkeys},
            },
            context=SimpleNamespace(job_id="allocation-v2:testnet401"),
        )


class _Reader:
    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return list(self.rows.get(policy_id, ()))


class _MeasuredChain:
    def read_finalized_metagraph(self, *, netuid, context, attempt_number=0):
        assert netuid == 401
        assert context.purpose == "research_lab.allocation.v2"
        assert attempt_number == 0
        hotkeys = ["unused-%d" % index for index in range(12)]
        hotkeys[origin.TESTNET401_BURN_UID] = origin.TESTNET401_BURN_HOTKEY
        hotkeys[origin.TESTNET401_VALIDATOR_UID] = (
            origin.TESTNET401_VALIDATOR_HOTKEY
        )
        return {
            "finalized_block_hash": "b" * 64,
            "header": {"block": 22_058 * 360 + 1},
            "workflow_epoch_id": 22_058,
            "metagraph": {"hotkeys": hotkeys},
        }

    def fresh_testnet401_cutover_scope(self, *, netuid):
        assert netuid == 401
        return _cutover()

    def prove_fresh_testnet401_allocation_origin(self, **_kwargs):
        return {"schema_version": "leadpoet.temporary_testnet401_chain_origin.v1"}


def _resolver(reader):
    return CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=_MeasuredChain(),
        config_supplier=lambda: None,
        network_supplier=lambda: "test",
    )


def test_measured_first_allocation_requires_cutover_parent_and_empty_history(
    monkeypatch,
):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_authority_graphs_from_context",
        lambda _context: [graph],
    )
    monkeypatch.setattr(
        origin,
        "validate_receipt_graph",
        lambda _graph: None,
    )
    reader = _Reader()
    required = set()
    context = ExecutionContextV2(
        job_id="allocation-v2:testnet401",
        purpose="research_lab.allocation.v2",
        epoch_id=22_058,
        parent_receipt_hashes=(origin.TESTNET401_CUTOVER_RECEIPT_HASH,),
    )

    assert (
        _resolver(reader)._finalized_champion_history(
            epoch=22_058,
            netuid=401,
            champion_rows=({"start_epoch": 22_051},),
            context=context,
            required_parents=required,
            chain_state={"finalized_block_hash": "b" * 64},
            fresh_network_origin_out={},
        )
        == []
    )
    assert required == {origin.TESTNET401_CUTOVER_RECEIPT_HASH}
    assert (
        "finalized_allocation_authorities",
        {"netuid": 401, "start_epoch": 22_042, "end_epoch": 22_057},
    ) in reader.calls


def test_measured_full_first_allocation_accepts_empty_legacy_reward_sources(
    monkeypatch,
):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    reader = _Reader()
    config = SimpleNamespace(
        reimbursement_dynamic_alpha_price_enabled=False,
        reimbursement_require_live_alpha_price=False,
        reimbursement_miner_alpha_per_epoch=100.0,
        reimbursement_usd_per_0_1_percent_epoch=0.666667,
        reimbursement_policy_doc=lambda enabled: {
            "policy_id": "policy:testnet401",
            "enabled": bool(enabled),
            "research_lab_emission_percent": 20.0,
            "reward_epochs": 20,
            "reimbursement_epochs": 20,
            "reimbursement_max_cost_multiplier_with_champions": 1.0,
            "champion_placeholder_alpha_percent": 0.0001,
            "champion_queue_trigger_ratio": 0.5,
            "usd_per_0_1_percent_epoch": 0.666667,
        },
    )
    resolver = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=_MeasuredChain(),
        config_supplier=lambda: config,
        network_supplier=lambda: "test",
    )
    context = ExecutionContextV2(
        job_id="allocation-v2:testnet401",
        purpose="research_lab.allocation.v2",
        epoch_id=22_058,
        parent_receipt_hashes=(origin.TESTNET401_CUTOVER_RECEIPT_HASH,),
        external_receipt_graphs=[graph],
    )

    result = resolver.resolve(
        payload={"epoch": 22_058, "netuid": 401},
        context=context,
    )

    assert result["source_state"]["fresh_network_origin"][
        "cutover_receipt_hash"
    ] == origin.TESTNET401_CUTOVER_RECEIPT_HASH
    assert result["source_state"]["champion_obligations"] == []
    assert result["source_state"]["settlement_frontier"]["mode"] == (
        "legacy_full_history_bootstrap"
    )


def test_measured_full_first_allocation_does_not_record_replayed_activation(
    monkeypatch,
):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    credentials = {
        slot: "%s-secret" % slot
        for slot in expected_provider_credential_slots()
    }
    retries = measured_retry_policy_hashes("sha256:" + "9" * 64)
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(value)
            for slot, value in credentials.items()
        },
        retry_policy_hashes=retries,
        transport=lambda **_kwargs: {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": b"[]",
            "tls_peer_chain_hash": "sha256:" + "8" * 64,
            "tls_protocol": "TLSv1.3",
        },
        artifact_sink=lambda body, **_kwargs: {
            "artifact_id": "sha256:" + "7" * 64,
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: "2026-09-08T17:01:48Z",
    )
    broker.provision_credentials(credentials)
    reader = SupabaseSourceReaderV2(
        execute_provider=broker.execute,
        retry_policy_hash=retries["supabase"],
        sleep=lambda _seconds: None,
    )
    config = SimpleNamespace(
        reimbursement_dynamic_alpha_price_enabled=False,
        reimbursement_require_live_alpha_price=False,
        reimbursement_miner_alpha_per_epoch=100.0,
        reimbursement_usd_per_0_1_percent_epoch=0.666667,
        reimbursement_policy_doc=lambda enabled: {
            "policy_id": "policy:testnet401",
            "enabled": bool(enabled),
            "research_lab_emission_percent": 20.0,
            "reward_epochs": 20,
            "reimbursement_epochs": 20,
            "reimbursement_max_cost_multiplier_with_champions": 1.0,
            "champion_placeholder_alpha_percent": 0.0001,
            "champion_queue_trigger_ratio": 0.5,
            "usd_per_0_1_percent_epoch": 0.666667,
        },
    )
    context = ExecutionContextV2(
        job_id="allocation-v2:testnet401:strict-replay",
        purpose="research_lab.allocation.v2",
        epoch_id=22_058,
        parent_receipt_hashes=(origin.TESTNET401_CUTOVER_RECEIPT_HASH,),
        external_receipt_graphs=[graph],
    )

    result = CoordinatorAllocationSourceV2(
        reader=reader,
        chain_source=_MeasuredChain(),
        config_supplier=lambda: config,
        network_supplier=lambda: "test",
    ).resolve(payload={"epoch": 22_058, "netuid": 401}, context=context)

    assert result["source_state"]["fresh_network_origin"]
    activation_attempts = [
        attempt
        for attempt in context.transport_attempts
        if ":allocation_settlement_frontier_activation:" in str(
            attempt["logical_operation_id"]
        )
    ]
    assert len(activation_attempts) == 1
    with pytest.raises(ExecutionJobV2Error, match="transport attempt is duplicated"):
        reader.read(
            policy_id="allocation_settlement_frontier_activation",
            parameters={"netuid": 401},
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )


def test_measured_first_allocation_rejects_existing_history(monkeypatch):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source,
        "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(
        allocation_source,
        "_receipt_authority_graphs_from_context",
        lambda _context: [graph],
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    reader = _Reader({"finalized_allocation_authorities": [{"epoch_id": 22_057}]})

    with pytest.raises(
        CoordinatorAllocationSourceV2Error,
        match="allocation history is not empty",
    ):
        _resolver(reader)._finalized_champion_history(
            epoch=22_058,
            netuid=401,
            champion_rows=({"start_epoch": 22_051},),
            context=ExecutionContextV2(
                job_id="allocation-v2:testnet401",
                purpose="research_lab.allocation.v2",
                epoch_id=22_058,
                parent_receipt_hashes=(origin.TESTNET401_CUTOVER_RECEIPT_HASH,),
            ),
            required_parents=set(),
            chain_state={"finalized_block_hash": "b" * 64},
        )


@pytest.mark.asyncio
async def test_fresh_readiness_requires_empty_testnet401_tables(monkeypatch):
    calls = []

    async def select_all(table, **kwargs):
        calls.append((table, kwargs.get("filters")))
        return []

    monkeypatch.setattr("gateway.research_lab.store.select_all", select_all)
    result = await settlement.champion_v2_cutover_readiness(
        epoch=22_058,
        netuid=401,
        _fresh_testnet401_empty_origin=True,
    )
    assert result["ready"] is True
    assert (
        settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1,
        (("netuid", 401),),
    ) in calls
    assert (
        "research_lab_emission_allocation_snapshots",
        (("netuid", 401), ("epoch", "gte", 22_042)),
    ) in calls
    assert (
        "research_lab_allocation_settlement_frontiers_v2",
        (("netuid", 401), ("allocation_epoch", "gte", 22_042)),
    ) in calls


@pytest.mark.asyncio
async def test_fresh_readiness_preserves_old_nonfinalized_testnet401_snapshots(
    monkeypatch,
):
    old_epochs = [20_613, 20_614, 20_615, 20_616, 20_621]

    async def select_all(table, **kwargs):
        if table != "research_lab_emission_allocation_snapshots":
            return []
        lower_bound = next(
            item[2]
            for item in kwargs["filters"]
            if len(item) == 3 and item[0] == "epoch" and item[1] == "gte"
        )
        return [{"epoch": value} for value in old_epochs if value >= lower_bound]

    monkeypatch.setattr("gateway.research_lab.store.select_all", select_all)
    result = await settlement.champion_v2_cutover_readiness(
        epoch=22_058,
        netuid=401,
        _fresh_testnet401_empty_origin=True,
    )
    assert result["ready"] is True


@pytest.mark.asyncio
async def test_host_loads_only_exact_durable_cutover_without_activation(
    monkeypatch,
):
    graph = _cutover_graph()

    async def select_many(*_args, **_kwargs):
        return []

    async def select_all(*_args, **_kwargs):
        return [
            {
                "schema_version": "leadpoet.subnet_epoch_cutover_authority.v3",
                "previous_epoch_scheme": "fresh_network_v1",
                "cutover_authority_hash": origin.TESTNET401_CUTOVER_AUTHORITY_HASH,
                "cutover_receipt_hash": origin.TESTNET401_CUTOVER_RECEIPT_HASH,
                "manifest_doc": _cutover(),
            }
        ]

    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setattr("gateway.research_lab.store.select_many", select_many)
    monkeypatch.setattr("gateway.research_lab.store.select_all", select_all)
    monkeypatch.setattr(
        "Leadpoet.utils.subnet_epoch.load_subnet_epoch_cutover",
        lambda: SubnetEpochCutover.from_mapping(_cutover()),
    )
    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store.load_receipt_graph_v2",
        lambda _root: None,
    )

    async def load_graph(_root):
        return graph

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store.load_receipt_graph_v2",
        load_graph,
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    loaded = await v2_authority._load_fresh_testnet401_first_allocation_parent_v1(
        netuid=401
    )
    assert loaded == graph


def _allocation_outcome(*, epoch: int, predecessor: dict | None = None) -> dict:
    frontier = build_allocation_settlement_frontier_v2(
        mode="bounded_delta_v1" if predecessor else "legacy_full_history_bootstrap",
        netuid=401,
        allocation_epoch=epoch,
        predecessor_frontier_hash=(
            predecessor["frontier_hash"] if predecessor else None
        ),
        reward_checkpoints=[],
    )
    allocation = {"allocation_hash": "sha256:" + "a" * 64}
    source_state = {
        "epoch": epoch,
        "netuid": 401,
        "policy_id": "policy:test",
        "policy": {},
        "reimbursement_obligation_count": 0,
        "champion_obligation_count": 0,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
        "skipped": {"reimbursements": [], "champions": []},
    }
    result = {
        "allocation": allocation,
        "allocation_inputs": {
            "epoch": epoch,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [],
        },
        "source_state": source_state,
        "source_state_hash": sha256_json(source_state),
    }
    return {
        "result": result,
        "execution_receipt": {"receipt_hash": "sha256:" + "b" * 64},
    }


@pytest.mark.asyncio
async def test_host_build_attaches_fresh_parent_and_skips_missing_activation(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    graph = _cutover_graph()
    captured = {}

    async def fresh_parent(**_kwargs):
        return graph

    async def no_frontier(**_kwargs):
        return None

    async def readiness(**kwargs):
        assert kwargs["_fresh_testnet401_empty_origin"] is True
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def load_parents(**kwargs):
        assert kwargs["finalized_champion_history"] == []
        return []

    async def execute(**kwargs):
        captured["parents"] = kwargs["parent_graphs"]
        return _allocation_outcome(epoch=22_058)

    async def persist_frontier(**kwargs):
        captured["frontier"] = kwargs["frontier"]
        return {}

    async def persist_links(*_args, **_kwargs):
        return {"status": "persisted"}

    async def forbidden(**_kwargs):
        raise AssertionError("chain activation path must not run")

    monkeypatch.setattr(
        v2_authority,
        "_load_fresh_testnet401_first_allocation_parent_v1",
        fresh_parent,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_allocation_settlement_frontier_context_v2",
        no_frontier,
    )
    monkeypatch.setattr(settlement, "champion_v2_cutover_readiness", readiness)
    monkeypatch.setattr(v2_authority, "_load_allocation_parent_graphs_v2", load_parents)
    monkeypatch.setattr(v2_authority, "ensure_chain_realized_settlements_v1", forbidden)
    monkeypatch.setattr(
        attested_v2_store,
        "persist_allocation_settlement_frontier_v2",
        persist_frontier,
    )
    monkeypatch.setattr(v2_authority, "_persist_business_links", persist_links)
    monkeypatch.setattr(v2_authority, "validate_receipt_graphs", lambda _graphs: None)
    monkeypatch.setattr(v2_authority, "execute_coordinator_v2", execute)
    monkeypatch.setitem(v2_authority.build_allocation_v2.__kwdefaults__, "execute", execute)

    result = await v2_authority.build_allocation_v2(
        epoch_id=22_058,
        netuid=401,
        policy={},
    )
    assert result["status"] == "matched"
    assert captured["parents"] == [graph]
    assert captured["frontier"]["allocation_epoch"] == 22_058


@pytest.mark.asyncio
async def test_host_build_attaches_cutover_to_unfinalized_frontier_successor(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    cutover_graph = _cutover_graph()
    frontier_graph = {"root_receipt_hash": "sha256:" + "b" * 64}
    prior = _allocation_outcome(epoch=22_062)["result"]["source_state"][
        "settlement_frontier"
    ]
    context = {
        "frontier": prior,
        "source": {"receipt": {"parent_receipt_hashes": []}},
    }
    captured = {}

    async def load_context(**_kwargs):
        return context

    async def load_parents(**kwargs):
        assert kwargs["settlement_frontier_context"] is context
        return [frontier_graph]

    async def execute(**kwargs):
        captured["parents"] = kwargs["parent_graphs"]
        return _allocation_outcome(epoch=22_063, predecessor=prior)

    async def forbidden(**_kwargs):
        raise AssertionError("settlement activation must not run")

    monkeypatch.setattr(
        attested_v2_store,
        "load_allocation_settlement_frontier_context_v2",
        load_context,
    )
    monkeypatch.setattr(
        v2_authority,
        "_load_fresh_testnet401_first_allocation_parent_v1",
        lambda **_kwargs: None,
    )

    async def fresh(**_kwargs):
        return cutover_graph

    monkeypatch.setattr(
        v2_authority,
        "_load_fresh_testnet401_first_allocation_parent_v1",
        fresh,
    )
    monkeypatch.setattr(v2_authority, "_load_allocation_parent_graphs_v2", load_parents)
    monkeypatch.setattr(v2_authority, "ensure_chain_realized_settlements_v1", forbidden)
    monkeypatch.setattr(v2_authority, "execute_coordinator_v2", execute)
    monkeypatch.setitem(v2_authority.build_allocation_v2.__kwdefaults__, "execute", execute)
    monkeypatch.setattr(v2_authority, "_validate_allocation_parent_graphs", lambda _graphs: [])
    monkeypatch.setattr(attested_v2_store, "persist_allocation_settlement_frontier_v2", lambda **_kwargs: None)

    async def persist_frontier(**_kwargs):
        return {}

    async def persist_links(*_args, **_kwargs):
        return {"status": "persisted"}

    monkeypatch.setattr(attested_v2_store, "persist_allocation_settlement_frontier_v2", persist_frontier)
    monkeypatch.setattr(v2_authority, "_persist_business_links", persist_links)
    result = await v2_authority.build_allocation_v2(
        epoch_id=22_063, netuid=401, policy={},
    )
    assert result["status"] == "matched"
    assert captured["parents"] == [frontier_graph, cutover_graph]


class _RecoveryMeasuredChain(_MeasuredChain):
    def read_finalized_metagraph(self, *, netuid, context, attempt_number=0):
        result = super().read_finalized_metagraph(
            netuid=netuid, context=context, attempt_number=attempt_number,
        )
        result["header"]["block"] = 22_063 * 360 + 1
        result["workflow_epoch_id"] = 22_063
        return result


def _unfinalized_frontier_rows():
    from tests.test_coordinator_allocation_source_v2 import (
        _signed_coordinator_receipt,
    )

    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap", netuid=401,
        allocation_epoch=22_062, predecessor_frontier_hash=None,
        reward_checkpoints=[],
    )
    allocation = {"allocation_hash": "sha256:" + "a" * 64}
    source_state = {
        "settlement_frontier": frontier,
        "fresh_network_origin": {
            "schema_version": "leadpoet.temporary_testnet401_first_allocation_origin.v1",
            "network_genesis_hash": origin.TESTNET401_GENESIS_HASH,
            "netuid": 401,
            "cutover_block": origin.TESTNET401_CUTOVER_BLOCK,
            "cutover_mapping_hash": origin.TESTNET401_CUTOVER_MAPPING_HASH,
            "cutover_authority_hash": origin.TESTNET401_CUTOVER_AUTHORITY_HASH,
            "cutover_receipt_hash": origin.TESTNET401_CUTOVER_RECEIPT_HASH,
            "snapshot_receipt_hash": origin.TESTNET401_SNAPSHOT_RECEIPT_HASH,
        },
    }
    source_state_hash = sha256_json(source_state)
    artifacts = sorted(
        set(frontier_artifact_hashes_v2(frontier)) | {source_state_hash}
    )
    receipt = _signed_coordinator_receipt(
        purpose="research_lab.allocation.v2", job_id="allocation:22062",
        epoch_id=22_062, input_root="sha256:" + "6" * 64,
        output_root=sha256_json({"allocation": allocation}),
        artifact_root=merkle_root(artifacts, domain="leadpoet-artifact-v2"),
        parents=(origin.TESTNET401_CUTOVER_RECEIPT_HASH,),
    )
    execution = {
        "schema_version": "leadpoet.attested_execution_result.v2",
        "receipt_hash": receipt["receipt_hash"],
        "role": "gateway_coordinator", "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2", "job_id": receipt["job_id"],
        "sequence": receipt["sequence"], "epoch_id": 22_062,
        "release_hash": "sha256:" + "7" * 64,
        "result_doc": {
            "allocation": allocation, "source_state": source_state,
            "source_state_hash": source_state_hash,
        },
        "result_hash": sha256_json({
            "allocation": allocation, "source_state": source_state,
            "source_state_hash": source_state_hash,
        }),
        "artifact_hashes": artifacts, "artifact_root": receipt["artifact_root"],
        "input_root": receipt["input_root"], "output_root": receipt["output_root"],
    }
    row = {
        "schema_version": frontier["schema_version"], "netuid": 401,
        "allocation_epoch": 22_062,
        "settled_through_epoch": frontier["settled_through_epoch"],
        "frontier_hash": frontier["frontier_hash"],
        "predecessor_frontier_hash": None,
        "source_receipt_hash": receipt["receipt_hash"],
        "source_state_hash": source_state_hash, "frontier_doc": frontier,
    }
    return frontier, allocation, receipt, execution, row


def _signed_frontier_rows_from_result(*, result, parents, digit):
    from tests.test_coordinator_allocation_source_v2 import (
        _signed_coordinator_receipt,
    )

    frontier = result["source_state"]["settlement_frontier"]
    allocation = result["allocation"]
    source_state_hash = result["source_state_hash"]
    artifacts = sorted(
        set(frontier_artifact_hashes_v2(frontier)) | {source_state_hash}
    )
    receipt = _signed_coordinator_receipt(
        purpose="research_lab.allocation.v2",
        job_id="allocation:%d" % int(frontier["allocation_epoch"]),
        epoch_id=int(frontier["allocation_epoch"]),
        input_root="sha256:" + digit * 64,
        output_root=sha256_json({"allocation": allocation}),
        artifact_root=merkle_root(artifacts, domain="leadpoet-artifact-v2"),
        parents=tuple(parents),
    )
    execution = {
        "schema_version": "leadpoet.attested_execution_result.v2",
        "receipt_hash": receipt["receipt_hash"],
        "role": "gateway_coordinator", "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2", "job_id": receipt["job_id"],
        "sequence": receipt["sequence"],
        "epoch_id": int(frontier["allocation_epoch"]),
        "release_hash": "sha256:" + "7" * 64,
        "result_doc": dict(result), "result_hash": sha256_json(dict(result)),
        "artifact_hashes": artifacts, "artifact_root": receipt["artifact_root"],
        "input_root": receipt["input_root"], "output_root": receipt["output_root"],
    }
    row = {
        "schema_version": frontier["schema_version"], "netuid": 401,
        "allocation_epoch": int(frontier["allocation_epoch"]),
        "settled_through_epoch": int(frontier["settled_through_epoch"]),
        "frontier_hash": frontier["frontier_hash"],
        "predecessor_frontier_hash": frontier["predecessor_frontier_hash"],
        "source_receipt_hash": receipt["receipt_hash"],
        "source_state_hash": source_state_hash, "frontier_doc": frontier,
    }
    return allocation, receipt, execution, row


class _ChainedRecoveryReader(_Reader):
    def __init__(self, *, activation, first_row, latest_row, executions, receipts, allocation):
        super().__init__()
        self.activation = activation
        self.first_row = first_row
        self.latest_row = latest_row
        self.executions = executions
        self.receipts = receipts
        self.allocation = allocation
        self.query_keys = set()
        self.empty_authority_reads = set()

    def read(self, *, policy_id, parameters, **_kwargs):
        query_key = (policy_id, json.dumps(parameters, sort_keys=True))
        if query_key in self.query_keys:
            raise ExecutionJobV2Error("transport attempt is duplicated")
        self.query_keys.add(query_key)
        self.calls.append((policy_id, dict(parameters)))
        if policy_id == "allocation_settlement_frontier_activation":
            return [dict(self.activation)]
        if policy_id == "allocation_settlement_frontiers":
            return [dict(self.latest_row)]
        if policy_id == "allocation_settlement_frontier_by_epoch":
            return [dict(self.first_row)]
        if policy_id == "attested_execution_result_by_receipt":
            return [dict(self.executions[parameters["receipt_hash"]])]
        if policy_id == "attested_receipt_by_hash":
            receipt = self.receipts[parameters["receipt_hash"]]
            return [{"receipt_hash": receipt["receipt_hash"], "receipt_doc": receipt}]
        if policy_id == "allocation_history":
            return [{
                "epoch": int(self.latest_row["allocation_epoch"]),
                "netuid": 401,
                "allocation_hash": self.allocation["allocation_hash"],
                "allocation_doc": dict(self.allocation),
            }]
        if policy_id in {
            "finalized_allocation_authorities",
            "legacy_finalized_allocation_migrations",
            "chain_realized_epoch_settlements",
            "chain_realized_obligation_credits",
            "compact_finalized_authority_cutover",
        }:
            self.empty_authority_reads.add(policy_id)
        return []


def test_full_coordinator_load_chains_e62_e63_e64_without_duplicate_reads(monkeypatch):
    from tests.test_coordinator_allocation_source_v2 import (
        _config as standard_config,
    )

    first_frontier, first_allocation, first_receipt, first_execution, first_row = (
        _unfinalized_frontier_rows()
    )
    activation = {
        "schema_version": "leadpoet.research_lab_allocation_settlement_frontier_activation.v2",
        "netuid": 401, "first_allocation_epoch": 22_062,
        "first_frontier_hash": first_frontier["frontier_hash"],
        "source_receipt_hash": first_receipt["receipt_hash"],
    }
    graph_roots = {
        origin.TESTNET401_CUTOVER_RECEIPT_HASH: _cutover_graph(),
        first_receipt["receipt_hash"]: {
            "root_receipt_hash": first_receipt["receipt_hash"]
        },
    }
    monkeypatch.setattr(
        allocation_source, "_receipt_graphs_by_declared_root",
        lambda _graphs, roots: {root: graph_roots[root] for root in roots},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    monkeypatch.setattr(
        allocation_source, "validate_signed_execution_receipt",
        lambda _receipt: None,
    )

    class ChainedMeasuredChain(_MeasuredChain):
        def __init__(self):
            self.proof_epochs = []

        def read_finalized_metagraph(self, *, netuid, context, attempt_number=0):
            result = super().read_finalized_metagraph(
                netuid=netuid, context=context, attempt_number=attempt_number,
            )
            result["header"]["block"] = int(context.epoch_id) * 360 + 1
            result["workflow_epoch_id"] = int(context.epoch_id)
            return result

        def prove_fresh_testnet401_allocation_origin(self, **kwargs):
            self.proof_epochs.append(int(kwargs["context"].epoch_id))
            return {"schema_version": "leadpoet.temporary_testnet401_chain_origin.v1"}

    chain = ChainedMeasuredChain()

    def resolve_epoch(*, epoch, latest_row, latest_allocation, executions, receipts):
        reader = _ChainedRecoveryReader(
            activation=activation, first_row=first_row, latest_row=latest_row,
            executions=executions, receipts=receipts,
            allocation=latest_allocation,
        )
        parents = (
            origin.TESTNET401_CUTOVER_RECEIPT_HASH,
            first_receipt["receipt_hash"],
            *(
                (latest_row["source_receipt_hash"],)
                if latest_row["source_receipt_hash"] != first_receipt["receipt_hash"]
                else ()
            ),
        )
        context = ExecutionContextV2(
            job_id="allocation-v2:testnet401:%d" % epoch,
            purpose="research_lab.allocation.v2", epoch_id=epoch,
            parent_receipt_hashes=parents,
            external_receipt_graphs=[graph_roots[root] for root in parents],
        )
        result = CoordinatorAllocationSourceV2(
            reader=reader, chain_source=chain,
            config_supplier=standard_config, network_supplier=lambda: "test",
        ).resolve(payload={"epoch": epoch, "netuid": 401}, context=context)
        assert len(reader.calls) == len(reader.query_keys)
        assert reader.empty_authority_reads == {
            "finalized_allocation_authorities",
            "legacy_finalized_allocation_migrations",
            "chain_realized_epoch_settlements",
            "chain_realized_obligation_credits",
            "compact_finalized_authority_cutover",
        }
        return result

    result_63 = resolve_epoch(
        epoch=22_063, latest_row=first_row,
        latest_allocation=first_allocation,
        executions={first_receipt["receipt_hash"]: first_execution},
        receipts={first_receipt["receipt_hash"]: first_receipt},
    )
    allocation_63, receipt_63, execution_63, row_63 = (
        _signed_frontier_rows_from_result(
            result=result_63,
            parents=(
                origin.TESTNET401_CUTOVER_RECEIPT_HASH,
                first_receipt["receipt_hash"],
            ),
            digit="8",
        )
    )
    graph_roots[receipt_63["receipt_hash"]] = {
        "root_receipt_hash": receipt_63["receipt_hash"]
    }
    result_64 = resolve_epoch(
        epoch=22_064, latest_row=row_63, latest_allocation=allocation_63,
        executions={
            first_receipt["receipt_hash"]: first_execution,
            receipt_63["receipt_hash"]: execution_63,
        },
        receipts={
            first_receipt["receipt_hash"]: first_receipt,
            receipt_63["receipt_hash"]: receipt_63,
        },
    )
    assert result_63["source_state"]["settlement_frontier"][
        "predecessor_frontier_hash"
    ] == first_frontier["frontier_hash"]
    assert result_64["source_state"]["settlement_frontier"][
        "predecessor_frontier_hash"
    ] == row_63["frontier_hash"]
    assert chain.proof_epochs == [22_063, 22_064]


def test_full_coordinator_load_recovers_exact_unfinalized_frontier(monkeypatch):
    from tests.test_coordinator_allocation_source_v2 import (
        _config as standard_config,
    )

    frontier, allocation, receipt, execution, row = _unfinalized_frontier_rows()
    activation = {
        "schema_version": "leadpoet.research_lab_allocation_settlement_frontier_activation.v2",
        "netuid": 401, "first_allocation_epoch": 22_062,
        "first_frontier_hash": frontier["frontier_hash"],
        "source_receipt_hash": receipt["receipt_hash"],
    }

    class RecoveryReader(_Reader):
        def read(self, *, policy_id, parameters, **_kwargs):
            self.calls.append((policy_id, dict(parameters)))
            values = {
                "allocation_settlement_frontier_activation": [activation],
                "allocation_settlement_frontiers": [row],
                "allocation_settlement_frontier_by_epoch": [row],
                "attested_execution_result_by_receipt": [execution],
                "attested_receipt_by_hash": [{
                    "receipt_hash": receipt["receipt_hash"],
                    "receipt_doc": receipt,
                }],
                "allocation_history": [{
                    "epoch": 22_062, "netuid": 401,
                    "allocation_hash": allocation["allocation_hash"],
                    "allocation_doc": allocation,
                }],
            }
            return [dict(item) for item in values.get(policy_id, [])]

    graph_roots = {
        origin.TESTNET401_CUTOVER_RECEIPT_HASH: _cutover_graph(),
        receipt["receipt_hash"]: {"root_receipt_hash": receipt["receipt_hash"]},
    }
    monkeypatch.setattr(
        allocation_source, "_receipt_graphs_by_declared_root",
        lambda _graphs, roots: {root: graph_roots[root] for root in roots},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    monkeypatch.setattr(allocation_source, "validate_signed_execution_receipt", lambda _receipt: None)
    context = ExecutionContextV2(
        job_id="allocation-v2:testnet401-recovery",
        purpose="research_lab.allocation.v2", epoch_id=22_063,
        parent_receipt_hashes=(
            origin.TESTNET401_CUTOVER_RECEIPT_HASH, receipt["receipt_hash"],
        ),
        external_receipt_graphs=list(graph_roots.values()),
    )
    reader = RecoveryReader()
    resolver = CoordinatorAllocationSourceV2(
        reader=reader, chain_source=_RecoveryMeasuredChain(),
        config_supplier=standard_config,
        network_supplier=lambda: "test",
    )
    result = resolver.resolve(
        payload={"epoch": 22_063, "netuid": 401}, context=context,
    )
    assert result["source_state"]["settlement_frontier"][
        "predecessor_frontier_hash"
    ] == frontier["frontier_hash"]
    assert "fresh_network_origin" not in result["source_state"]
    assert set(context.parent_receipt_hashes) == {
        origin.TESTNET401_CUTOVER_RECEIPT_HASH, receipt["receipt_hash"],
    }
    assert reader.calls.count((
        "chain_realized_settlement_activation", {"netuid": 401},
    )) == 1
    assert sum(
        policy == "allocation_settlement_frontier_activation"
        for policy, _parameters in reader.calls
    ) == 1


def test_unfinalized_recovery_advances_across_two_missed_epochs(monkeypatch):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source, "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)

    for epoch, source_epoch, digit in ((22_063, 22_062, "a"), (22_064, 22_063, "b")):
        allocation = {"allocation_hash": "sha256:" + digit * 64}
        receipt_hash = "sha256:" + ("c" if digit == "a" else "d") * 64
        reader = _Reader({
            "allocation_history": [{
                "epoch": source_epoch, "netuid": 401,
                "allocation_hash": allocation["allocation_hash"],
                "allocation_doc": allocation,
            }],
        })
        required = {receipt_hash}
        CoordinatorAllocationSourceV2(
            reader=reader, chain_source=_RecoveryMeasuredChain(),
            config_supplier=lambda: None, network_supplier=lambda: "test",
        )._validate_fresh_testnet401_unfinalized_frontier(
            epoch=epoch, netuid=401,
            chain_state={"finalized_block_hash": "b" * 64},
            context=ExecutionContextV2(
                job_id="recovery:%d" % epoch,
                purpose="research_lab.allocation.v2", epoch_id=epoch,
                parent_receipt_hashes=(
                    origin.TESTNET401_CUTOVER_RECEIPT_HASH, receipt_hash,
                ),
            ),
            required_parents=required,
            frontier_source={
                "epoch": source_epoch, "netuid": 401,
                "receipt_hash": receipt_hash, "allocation": allocation,
            },
        )
        assert ("allocation_history", {
            "netuid": 401, "start_epoch": source_epoch,
            "end_epoch": epoch - 1,
        }) in reader.calls
        assert ("finalized_allocation_authorities", {
            "netuid": 401, "start_epoch": 22_042,
            "end_epoch": epoch - 1,
        }) in reader.calls
        assert required == {
            receipt_hash, origin.TESTNET401_CUTOVER_RECEIPT_HASH,
        }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"allocation_history": []}, "allocation history"),
        ({"allocation_history": [
            {"epoch": 22_062, "netuid": 401,
             "allocation_hash": "sha256:" + "f" * 64,
             "allocation_doc": {"allocation_hash": "sha256:" + "f" * 64}},
        ]}, "allocation history"),
        ({"compact_finalized_authority_cutover": [{"epoch_id": 22_062}]},
         "authority is not empty"),
        ({"finalized_allocation_authorities": [{"epoch_id": 22_062}]},
         "authority is not empty"),
    ],
)
def test_unfinalized_recovery_rejects_nonmatching_or_finalized_history(
    monkeypatch, rows, message,
):
    graph = _cutover_graph()
    monkeypatch.setattr(
        allocation_source, "_receipt_graphs_by_declared_root",
        lambda _graphs, _roots: {origin.TESTNET401_CUTOVER_RECEIPT_HASH: graph},
    )
    monkeypatch.setattr(origin, "validate_receipt_graph", lambda _graph: None)
    allocation = {"allocation_hash": "sha256:" + "a" * 64}
    defaults = {
        "allocation_history": [{
            "epoch": 22_062, "netuid": 401,
            "allocation_hash": allocation["allocation_hash"],
            "allocation_doc": allocation,
        }],
    }
    defaults.update(rows)
    receipt_hash = "sha256:" + "c" * 64
    resolver = CoordinatorAllocationSourceV2(
        reader=_Reader(defaults), chain_source=_RecoveryMeasuredChain(),
        config_supplier=lambda: None, network_supplier=lambda: "test",
    )
    with pytest.raises(CoordinatorAllocationSourceV2Error, match=message):
        resolver._validate_fresh_testnet401_unfinalized_frontier(
            epoch=22_063, netuid=401,
            chain_state={"finalized_block_hash": "b" * 64},
            context=ExecutionContextV2(
                job_id="recovery:negative", purpose="research_lab.allocation.v2",
                epoch_id=22_063,
                parent_receipt_hashes=(
                    origin.TESTNET401_CUTOVER_RECEIPT_HASH, receipt_hash,
                ),
            ),
            required_parents={receipt_hash},
            frontier_source={
                "epoch": 22_062, "netuid": 401,
                "receipt_hash": receipt_hash, "allocation": allocation,
            },
        )


@pytest.mark.asyncio
async def test_same_epoch_frontier_replay_never_reenters_fresh_origin(monkeypatch):
    from gateway.research_lab import attested_v2_store

    outcome = _allocation_outcome(epoch=22_058)
    frontier = outcome["result"]["source_state"]["settlement_frontier"]

    async def current_frontier(**_kwargs):
        return {
            "frontier": frontier,
            "source": {"receipt": {"parent_receipt_hashes": []}},
        }

    async def forbidden(**_kwargs):
        raise AssertionError("fresh or execution path must not run")

    async def persist_links(*_args, **_kwargs):
        return {"status": "reused"}

    monkeypatch.setattr(
        attested_v2_store,
        "load_allocation_settlement_frontier_context_v2",
        current_frontier,
    )
    monkeypatch.setattr(
        v2_authority,
        "_load_fresh_testnet401_first_allocation_parent_v1",
        forbidden,
    )
    monkeypatch.setattr(v2_authority, "ensure_chain_realized_settlements_v1", forbidden)
    monkeypatch.setattr(v2_authority, "execute_coordinator_v2", forbidden)
    monkeypatch.setitem(v2_authority.build_allocation_v2.__kwdefaults__, "execute", forbidden)
    monkeypatch.setattr(
        v2_authority,
        "_current_allocation_frontier_outcome_v2",
        lambda *_args, **_kwargs: outcome,
    )
    monkeypatch.setattr(v2_authority, "_persist_business_links", persist_links)

    result = await v2_authority.build_allocation_v2(
        epoch_id=22_058,
        netuid=401,
        policy={},
    )
    assert result["status"] == "matched"
