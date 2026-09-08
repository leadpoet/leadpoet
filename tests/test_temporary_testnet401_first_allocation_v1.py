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
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json
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


def _allocation_outcome(*, epoch: int) -> dict:
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=401,
        allocation_epoch=epoch,
        predecessor_frontier_hash=None,
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
