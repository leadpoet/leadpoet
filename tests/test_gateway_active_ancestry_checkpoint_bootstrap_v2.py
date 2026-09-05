from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from gateway.tee import bootstrap_active_ancestry_checkpoints_v2 as bootstrap
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    RECEIPT_GRAPH_SCHEMA_VERSION,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
LINEAGE = "sha256:" + "d" * 64
COMMIT = "1" * 40
PCR0 = "2" * 96


def test_load_release_manifest_accepts_exact_historical_running_gateway(
    tmp_path,
):
    from tests.test_release_channel_v2 import _historical_gateway_manifest

    manifest = _historical_gateway_manifest(COMMIT)
    path = tmp_path / "running-release.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert bootstrap._load_release_manifest(path) == manifest


def test_load_release_manifest_rejects_incomplete_historical_running_gateway(
    tmp_path,
):
    from tests.test_release_channel_v2 import _historical_gateway_manifest

    manifest = _historical_gateway_manifest(COMMIT)
    manifest["roles"].pop("gateway_autoresearch")
    path = tmp_path / "running-release.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="release manifest roles are incomplete"):
        bootstrap._load_release_manifest(path)


@pytest.mark.asyncio
async def test_whole_bootstrap_accepts_running_historical_gateway(
    monkeypatch,
):
    from tests.test_release_channel_v2 import _historical_gateway_manifest

    manifest = _historical_gateway_manifest(COMMIT)
    coordinator = manifest["roles"]["gateway_coordinator"]

    class HistoricalClient(_Client):
        async def v2_get_boot_identity(self):
            return {
                "boot_identity_hash": HASH_A,
                "commit_sha": coordinator["commit_sha"],
                "pcr0": coordinator["pcr0"],
                "build_manifest_hash": coordinator["execution_manifest_hash"],
                "dependency_lock_hash": coordinator["dependency_lock_hash"],
                "build_identity_hash": coordinator["build_identity_hash"],
                "physical_role": "gateway_coordinator",
            }

    async def awaiting_first_allocation(**_kwargs):
        return {"status": "awaiting_first_allocation"}

    async def resolve_epoch(_value):
        return 24000

    async def load_empty(**_kwargs):
        return []

    verified_releases = []

    def verifier_builder(releases):
        verified_releases.append(deepcopy(releases))
        return lambda identity: dict(identity)

    monkeypatch.setattr(bootstrap, "_lineage_id", lambda: LINEAGE)
    boot_verifier = bootstrap._LazyApprovedReleaseBootVerifier(
        current_release=manifest,
        verifier_builder=verifier_builder,
    )
    result = await bootstrap.bootstrap_active_ancestry_checkpoints_v2(
        netuid=71,
        release_manifest=manifest,
        client=HistoricalClient(),
        boot_verifier=boot_verifier,
        resolve_epoch=resolve_epoch,
        load_allocation_graphs=load_empty,
        load_sourcing_graphs=load_empty,
        load_proofs=lambda *_args, **_kwargs: {},
        load_checkpointed_graphs=lambda *_args, **_kwargs: {},
        persist_checkpoint=lambda *_args, **_kwargs: None,
        ensure_allocation_frontier=awaiting_first_allocation,
    )

    assert result["status"] == "complete"
    assert result["active_root_count"] == 0
    assert verified_releases == [{COMMIT: manifest}]


def _proof(root: str, suffix: str) -> dict:
    return {
        "proof_hash": "sha256:" + suffix * 64,
        "certificate": {
            "claim": {"output_root_receipt_hash": root},
        },
    }


def _full_graph(root: str, receipts: tuple[str, ...] | None = None) -> dict:
    roots = receipts or (root,)
    return {
        "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
        "root_receipt_hash": root,
        "boot_identities": [],
        "receipts": [{"receipt_hash": item} for item in roots],
        "transport_attempts": [],
        "host_operations": [],
    }


def _bounded_graph(root: str, proof: dict) -> dict:
    return {
        "schema_version": CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        "root_receipt_hash": root,
        "boot_identities": [],
        "receipts": [{"receipt_hash": root}],
        "transport_attempts": [],
        "host_operations": [],
        "ancestry_proof": deepcopy(proof),
    }


class _Client:
    def __init__(self, *, supported: bool = True, corrupt_health: bool = False):
        self.supported = supported
        self.corrupt_health = corrupt_health

    async def coordinator_v2_health(self):
        return {
            "authority": "legacy" if self.corrupt_health else "v2_only",
            "role": "gateway_coordinator",
            "physical_role": "gateway_coordinator",
            "worker_count": 1,
            "configured_worker_count": 0,
            "workers_alive": True,
            "ancestry_checkpoints": True,
            "ancestry_lineage_id": LINEAGE,
            "boot_identity_hash": HASH_A,
            "supported_operations": (
                [bootstrap.OP_ANCESTRY_CHECKPOINT_BOOTSTRAP_V2]
                if self.supported
                else []
            ),
        }

    async def v2_get_boot_identity(self):
        return {
            "boot_identity_hash": HASH_A,
            "commit_sha": COMMIT,
            "pcr0": PCR0,
            "build_manifest_hash": HASH_B,
            "dependency_lock_hash": HASH_C,
            "physical_role": "gateway_coordinator",
        }


class _Config:
    @staticmethod
    def reimbursement_policy_doc(*, enabled=True):
        assert enabled is True
        return {"enabled": True}


class _Harness:
    def __init__(self, active_roots: list[str], graphs: dict[str, dict]):
        self.active_roots = list(active_roots)
        self.graphs = {root: deepcopy(graph) for root, graph in graphs.items()}
        self.proofs: dict[str, dict] = {}
        self.bounded: dict[str, dict] = {}
        self.persisted: list[str] = []
        self.execute_calls: list[dict] = []
        self.selector_durable: list[list[str]] = []
        self.proof_queries: list[list[str]] = []
        self.selection_count = 0
        self.selection_roots: list[str] | None = None
        self.tamper_readback = False
        self.compact_persistence = False

    def current_root(self) -> str:
        if self.selection_roots is None:
            return self.active_roots[0]
        index = min(self.selection_count, len(self.selection_roots) - 1)
        return self.selection_roots[index]

    async def load_allocation(self, *, epoch_id, netuid, policy):
        assert epoch_id >= 0
        assert netuid == 71
        assert policy == {"enabled": True}
        if self.selection_roots is None:
            self.selection_count += 1
            return [
                deepcopy(self.bounded.get(root, self.graphs[root]))
                for root in self.active_roots
            ]
        root = self.current_root()
        self.selection_count += 1
        if root in self.bounded:
            return [deepcopy(self.bounded[root])]
        return [deepcopy(self.graphs[root])]

    async def load_sourcing(self, *, current_epoch, window):
        assert current_epoch >= 0
        assert window == 30
        return []

    async def load_proofs(self, roots, **_kwargs):
        self.proof_queries.append(list(roots))
        selected = {
            root: deepcopy(self.proofs[root]) for root in roots if root in self.proofs
        }
        if self.tamper_readback and self.persisted:
            selected.pop(self.persisted[-1], None)
        return selected

    async def load_bounded(self, roots):
        return {
            root: deepcopy(self.bounded[root]) for root in roots if root in self.bounded
        }

    async def persist(self, proof, *, checkpointed_graph, **_kwargs):
        root = proof["certificate"]["claim"]["output_root_receipt_hash"]
        self.persisted.append(root)
        self.proofs[root] = deepcopy(proof)
        self.bounded[root] = deepcopy(checkpointed_graph)
        if self.compact_persistence:
            self.bounded[root]["schema_version"] = (
                COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
            )
        return {
            "root_receipt_hash": root,
            "proof_hash": proof["proof_hash"],
            "root_activated": True,
        }

    async def execute(self, **kwargs):
        self.execute_calls.append(kwargs)
        graph = kwargs["parent_graphs"][0]
        frontier = list(kwargs["parent_ancestry_proofs"])
        frontier_roots = {
            proof["certificate"]["claim"]["output_root_receipt_hash"]
            for proof in frontier
        }
        selected_root = graph["root_receipt_hash"]
        new = [_proof(selected_root, "e")]
        all_roots = sorted(
            frontier_roots
            | {item["certificate"]["claim"]["output_root_receipt_hash"] for item in new}
        )
        return {
            "result": {
                "schema_version": "result",
                "selected_root_receipt_hashes": [graph["root_receipt_hash"]],
                "checkpoint_proofs": new,
                "checkpoint_root_receipt_hashes": all_roots,
                "checkpoint_set_hash": HASH_A,
            }
        }


@pytest.fixture
def orchestration(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "validate_prior_release_manifest", lambda value: dict(value)
    )
    monkeypatch.setattr(
        bootstrap,
        "prior_role_expectation",
        lambda _release, _role: {
            "commit_sha": COMMIT,
            "pcr0": PCR0,
            "build_manifest_hash": HASH_B,
            "dependency_lock_hash": HASH_C,
        },
    )
    monkeypatch.setattr(bootstrap, "_lineage_id", lambda: LINEAGE)
    monkeypatch.setattr(
        bootstrap.ResearchLabGatewayConfig, "from_env", lambda: _Config()
    )

    def select_frontier(*, durable_compact_proofs, **_kwargs):
        values = list(durable_compact_proofs)
        harness = getattr(select_frontier, "harness", None)
        if harness is not None:
            harness.selector_durable.append(
                [
                    proof["certificate"]["claim"]["output_root_receipt_hash"]
                    for proof in values
                ]
            )
        return values[-1:] if values else []

    monkeypatch.setattr(
        bootstrap,
        "select_ancestry_checkpoint_resume_frontier_v2",
        select_frontier,
    )
    monkeypatch.setattr(
        bootstrap,
        "validate_ancestry_checkpoint_bootstrap_result_v2",
        lambda value, **_kwargs: deepcopy(value),
    )
    monkeypatch.setattr(
        bootstrap,
        "build_checkpointed_receipt_graph_from_full_graph_v2",
        lambda _graph, proof, **_kwargs: _bounded_graph(
            proof["certificate"]["claim"]["output_root_receipt_hash"],
            proof,
        ),
    )
    return select_frontier


async def _run(harness: _Harness, **overrides):
    async def resolve_epoch(_value):
        return 24000

    async def ensure_allocation_frontier(**_kwargs):
        return {
            "status": "already_initialized",
            "frontier_hash": HASH_A,
        }

    kwargs = {
        "netuid": 71,
        "release_manifest": {"commit_sha": COMMIT},
        "client": _Client(),
        "boot_verifier": lambda identity: identity,
        "resolve_epoch": resolve_epoch,
        "load_allocation_graphs": harness.load_allocation,
        "load_sourcing_graphs": harness.load_sourcing,
        "load_proofs": harness.load_proofs,
        "load_checkpointed_graphs": harness.load_bounded,
        "persist_checkpoint": harness.persist,
        "execute": harness.execute,
        "ensure_allocation_frontier": ensure_allocation_frontier,
    }
    kwargs.update(overrides)
    return await bootstrap.bootstrap_active_ancestry_checkpoints_v2(**kwargs)


@pytest.mark.asyncio
async def test_bootstrap_accepts_fresh_database_frontier_state(orchestration) -> None:
    harness = _Harness([], {})

    async def awaiting_first_allocation(**_kwargs):
        return {"status": "awaiting_first_allocation"}

    result = await _run(
        harness,
        ensure_allocation_frontier=awaiting_first_allocation,
    )

    assert result["status"] == "complete"
    assert result["active_root_count"] == 0
    assert harness.execute_calls == []


@pytest.mark.asyncio
async def test_no_legacy_active_roots_are_reselected_without_execution(orchestration):
    proof = _proof(HASH_A, "a")
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    harness.proofs[HASH_A] = proof
    harness.bounded[HASH_A] = _bounded_graph(HASH_A, proof)
    result = await _run(harness)
    assert result["status"] == "complete"
    assert result["legacy_graphs_processed"] == 0
    assert result["new_proof_count"] == 0
    assert harness.execute_calls == []
    assert harness.persisted == []


@pytest.mark.asyncio
async def test_valid_coordinator_without_advertised_operation_is_unsupported(
    orchestration,
):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Unsupported,
        match="does not advertise",
    ):
        await _run(harness, client=_Client(supported=False))
    assert harness.selection_count == 0


@pytest.mark.asyncio
async def test_invalid_health_is_not_misclassified_as_unsupported(orchestration):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match="health is invalid",
    ):
        await _run(
            harness,
            client=_Client(supported=False, corrupt_health=True),
        )


@pytest.mark.asyncio
async def test_success_persists_only_selected_root_proof_and_reads_back(orchestration):
    harness = _Harness(
        [HASH_B],
        {HASH_B: _full_graph(HASH_B, (HASH_A, HASH_B))},
    )
    orchestration.harness = harness
    result = await _run(harness)
    assert result["new_proof_count"] == 1
    assert harness.persisted == [HASH_B]
    assert harness.execute_calls[0]["require_egress_proxy"] is False
    assert harness.execute_calls[0]["payload"] == {
        "schema_version": (
            bootstrap.ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION
        ),
        "selected_root_receipt_hashes": [HASH_B],
    }


@pytest.mark.asyncio
async def test_compact_persistence_is_accepted_during_stability_reselection(
    orchestration,
):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    harness.compact_persistence = True

    result = await _run(harness)

    assert result["status"] == "complete"
    assert result["stability_rounds"] == 1
    assert result["new_proof_count"] == 1
    assert harness.persisted == [HASH_A]
    assert len(harness.execute_calls) == 1
    assert harness.bounded[HASH_A]["schema_version"] == (
        COMPACT_CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
    )


@pytest.mark.asyncio
async def test_multiple_active_roots_run_measured_jobs_sequentially(orchestration):
    harness = _Harness(
        [HASH_B, HASH_A],
        {
            HASH_A: _full_graph(HASH_A),
            HASH_B: _full_graph(HASH_B),
        },
    )
    active = 0
    maximum_active = 0
    order = []
    original_execute = harness.execute

    async def observed_execute(**kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(kwargs["parent_graphs"][0]["root_receipt_hash"])
        try:
            await asyncio.sleep(0)
            return await original_execute(**kwargs)
        finally:
            active -= 1

    harness.execute = observed_execute
    result = await _run(harness)
    assert result["active_root_count"] == 2
    assert order == [HASH_A, HASH_B]
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_all_active_graph_selections_are_concurrent_and_exact():
    both_started = asyncio.Event()
    started = set()
    graph = _full_graph(HASH_A)

    async def arrive(name):
        started.add(name)
        if len(started) == 3:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return [deepcopy(graph)]

    selected = await bootstrap._select_active_graphs(
        epoch_id=24000,
        netuid=71,
        policy={"enabled": True},
        load_allocation_graphs=lambda **_kwargs: arrive("allocation"),
        load_sourcing_graphs=lambda **_kwargs: arrive("sourcing"),
        load_source_add_graphs=lambda **_kwargs: arrive("source_add"),
    )
    assert set(started) == {"allocation", "sourcing", "source_add"}
    assert selected == {HASH_A: graph}

    conflicting = _full_graph(HASH_A)
    conflicting["receipts"].append({"receipt_hash": HASH_B})

    async def allocation(**_kwargs):
        return [deepcopy(graph)]

    async def sourcing(**_kwargs):
        return [conflicting]

    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match="conflicts for one immutable root",
    ):
        await bootstrap._select_active_graphs(
            epoch_id=24000,
            netuid=71,
            policy={"enabled": True},
            load_allocation_graphs=allocation,
            load_sourcing_graphs=sourcing,
        )


@pytest.mark.asyncio
async def test_allocation_selection_bootstraps_history_only_without_frontier():
    calls = []

    async def load_frontier(**kwargs):
        assert kwargs == {"netuid": 71, "before_epoch": 24001}
        return None

    async def load_parents(**kwargs):
        calls.append(kwargs)
        return [_full_graph(HASH_A)]

    graphs = await bootstrap._load_frontier_bounded_allocation_graphs(
        epoch_id=24000,
        netuid=71,
        policy={"enabled": True},
        load_frontier_context=load_frontier,
        load_parent_graphs=load_parents,
        load_graphs=lambda _roots: {},
    )

    assert [graph["root_receipt_hash"] for graph in graphs] == [HASH_A]
    assert calls == [{"epoch_id": 24000, "netuid": 71, "policy": {"enabled": True}}]


@pytest.mark.asyncio
async def test_allocation_selection_uses_prior_frontier_for_bounded_delta():
    context = {"frontier": {"allocation_epoch": 23999}}
    calls = []

    async def load_frontier(**_kwargs):
        return context

    async def load_parents(**kwargs):
        calls.append(kwargs)
        return [_full_graph(HASH_B)]

    graphs = await bootstrap._load_frontier_bounded_allocation_graphs(
        epoch_id=24000,
        netuid=71,
        policy={"enabled": True},
        load_frontier_context=load_frontier,
        load_parent_graphs=load_parents,
        load_graphs=lambda _roots: {},
    )

    assert [graph["root_receipt_hash"] for graph in graphs] == [HASH_B]
    assert calls == [
        {
            "epoch_id": 24000,
            "netuid": 71,
            "policy": {"enabled": True},
            "settlement_frontier_context": context,
        }
    ]


@pytest.mark.asyncio
async def test_allocation_selection_recovers_exact_current_frontier_parents():
    calls = []
    context = {
        "frontier": {"allocation_epoch": 24000},
        "source": {
            "receipt": {
                "parent_receipt_hashes": [HASH_B, HASH_A],
            }
        },
    }

    async def load_frontier(**_kwargs):
        return context

    async def load_parents(**_kwargs):
        raise AssertionError("current frontier replayed historical selection")

    async def load_graphs(roots):
        calls.append(roots)
        return {root: _full_graph(root) for root in roots}

    graphs = await bootstrap._load_frontier_bounded_allocation_graphs(
        epoch_id=24000,
        netuid=71,
        policy={"enabled": True},
        load_frontier_context=load_frontier,
        load_parent_graphs=load_parents,
        load_graphs=load_graphs,
    )

    assert calls == [[HASH_A, HASH_B]]
    assert [graph["root_receipt_hash"] for graph in graphs] == [HASH_A, HASH_B]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context,error",
    [
        ({"frontier": {"allocation_epoch": 24001}}, "outside the active epoch"),
        (
            {"frontier": {"allocation_epoch": 24000}, "source": {}},
            "parents are invalid",
        ),
        (
            {
                "frontier": {"allocation_epoch": 24000},
                "source": {"receipt": {"parent_receipt_hashes": [HASH_A, HASH_A]}},
            },
            "parent set is invalid",
        ),
    ],
)
async def test_allocation_selection_rejects_invalid_frontier_state(context, error):
    async def load_frontier(**_kwargs):
        return context

    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match=error,
    ):
        await bootstrap._load_frontier_bounded_allocation_graphs(
            epoch_id=24000,
            netuid=71,
            policy={},
            load_frontier_context=load_frontier,
            load_parent_graphs=lambda **_kwargs: [],
            load_graphs=lambda _roots: {},
        )


@pytest.mark.asyncio
async def test_partial_resume_loads_only_root_and_direct_parent_proofs(
    orchestration,
):
    graph = _full_graph(HASH_B, (HASH_A, HASH_B, HASH_C))
    graph["receipts"][1]["parent_receipt_hashes"] = [HASH_A]
    harness = _Harness(
        [HASH_B],
        {HASH_B: graph},
    )
    harness.proofs[HASH_A] = _proof(HASH_A, "a")
    harness.bounded[HASH_A] = _bounded_graph(HASH_A, harness.proofs[HASH_A])
    orchestration.harness = harness
    result = await _run(harness)
    assert harness.proof_queries[0] == [HASH_A, HASH_B]
    assert HASH_C not in harness.proof_queries[0]
    assert harness.selector_durable == [[HASH_A]]
    assert [
        item["certificate"]["claim"]["output_root_receipt_hash"]
        for item in harness.execute_calls[0]["parent_ancestry_proofs"]
    ] == [HASH_A]
    assert harness.persisted == [HASH_B]
    assert result["resume_proof_count"] == 1


@pytest.mark.asyncio
async def test_tampered_measured_result_fails_before_persistence(
    orchestration, monkeypatch
):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})

    def reject(*_args, **_kwargs):
        raise ValueError("checkpoint bootstrap result set hash differs")

    monkeypatch.setattr(
        bootstrap,
        "validate_ancestry_checkpoint_bootstrap_result_v2",
        reject,
    )
    with pytest.raises(ValueError, match="set hash differs"):
        await _run(harness)
    assert harness.persisted == []


@pytest.mark.asyncio
async def test_extra_nonselected_proof_fails_before_persistence(orchestration):
    harness = _Harness(
        [HASH_B],
        {HASH_B: _full_graph(HASH_B, (HASH_A, HASH_B))},
    )
    original_execute = harness.execute

    async def execute_with_extra(**kwargs):
        outcome = await original_execute(**kwargs)
        outcome["result"]["checkpoint_proofs"].insert(0, _proof(HASH_A, "a"))
        outcome["result"]["checkpoint_root_receipt_hashes"].insert(0, HASH_A)
        return outcome

    harness.execute = execute_with_extra
    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match="one selected-root proof",
    ):
        await _run(harness)
    assert harness.persisted == []


@pytest.mark.asyncio
async def test_active_root_change_is_processed_in_second_bounded_round(orchestration):
    harness = _Harness(
        [HASH_A],
        {
            HASH_A: _full_graph(HASH_A),
            HASH_B: _full_graph(HASH_B),
        },
    )
    harness.selection_roots = [HASH_A, HASH_B]
    result = await _run(harness)
    assert result["stability_rounds"] == 2
    assert harness.persisted == [HASH_A, HASH_B]


@pytest.mark.asyncio
async def test_epoch_selection_stays_frozen_across_long_bootstrap(orchestration):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    resolved_epochs = iter((24000, 24001))
    resolution_count = 0
    selected_epochs = []

    async def resolve_epoch(_value):
        nonlocal resolution_count
        resolution_count += 1
        return next(resolved_epochs)

    async def load_allocation(*, epoch_id, netuid, policy):
        selected_epochs.append(("allocation", epoch_id))
        return await harness.load_allocation(
            epoch_id=epoch_id,
            netuid=netuid,
            policy=policy,
        )

    async def load_sourcing(*, current_epoch, window):
        selected_epochs.append(("sourcing", current_epoch))
        return await harness.load_sourcing(
            current_epoch=current_epoch,
            window=window,
        )

    result = await _run(
        harness,
        resolve_epoch=resolve_epoch,
        load_allocation_graphs=load_allocation,
        load_sourcing_graphs=load_sourcing,
    )

    assert result["status"] == "complete"
    assert result["epoch_id"] == 24000
    assert resolution_count == 1
    assert selected_epochs == [
        ("allocation", 24000),
        ("sourcing", 24000),
        ("allocation", 24000),
        ("sourcing", 24000),
    ]


@pytest.mark.asyncio
async def test_continuously_changing_roots_exhaust_bounded_rounds(orchestration):
    harness = _Harness(
        [HASH_A],
        {
            HASH_A: _full_graph(HASH_A),
            HASH_B: _full_graph(HASH_B),
            HASH_C: _full_graph(HASH_C),
        },
    )
    harness.selection_roots = [HASH_A, HASH_B, HASH_B, HASH_C]
    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match="did not stabilize",
    ):
        await _run(harness, max_stability_rounds=2)


@pytest.mark.asyncio
async def test_missing_exact_proof_readback_fails_closed(orchestration):
    harness = _Harness([HASH_A], {HASH_A: _full_graph(HASH_A)})
    harness.tamper_readback = True
    with pytest.raises(
        bootstrap.ActiveAncestryCheckpointBootstrapV2Error,
        match="proof durable readback differs",
    ):
        await _run(harness)


def test_lazy_verifier_loads_historic_and_current_validator_release(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "validate_prior_release_manifest", lambda value: dict(value)
    )
    loaded = []

    def lineage_loader(*, current_release, parent_graphs, **_kwargs):
        identity = parent_graphs[0]["boot_identities"][0]
        loaded.append((identity["commit_sha"], identity["physical_role"]))
        return {
            identity["commit_sha"]: {
                "gateway_release_manifest": current_release,
                "validator_release_manifest": {"release": {}},
            }
        }

    def verifier_builder(_releases):
        return lambda identity: dict(identity)

    verifier = bootstrap._LazyApprovedReleaseBootVerifier(
        current_release={"commit_sha": COMMIT},
        lineage_loader=lineage_loader,
        verifier_builder=verifier_builder,
    )
    historic = {
        "commit_sha": "3" * 40,
        "physical_role": "gateway_scoring",
    }
    current_validator = {
        "commit_sha": COMMIT,
        "physical_role": "validator_weights",
    }
    assert verifier(historic) == historic
    assert verifier(current_validator) == current_validator
    assert loaded == [
        ("3" * 40, "gateway_scoring"),
        (COMMIT, "validator_weights"),
    ]


def test_cli_uses_exit_three_only_for_verified_unsupported_operation(
    monkeypatch, capsys
):
    async def unsupported(**_kwargs):
        raise bootstrap.ActiveAncestryCheckpointBootstrapV2Unsupported(
            "operation absent"
        )

    monkeypatch.setattr(
        bootstrap, "bootstrap_active_ancestry_checkpoints_v2", unsupported
    )
    assert bootstrap.main([]) == 3
    assert '"status": "unsupported"' in capsys.readouterr().out

    async def integrity_failure(**_kwargs):
        raise bootstrap.ActiveAncestryCheckpointBootstrapV2Error("PCR0 differs")

    monkeypatch.setattr(
        bootstrap,
        "bootstrap_active_ancestry_checkpoints_v2",
        integrity_failure,
    )
    assert bootstrap.main([]) == 1
    captured = capsys.readouterr()
    assert '"status": "unsupported"' not in captured.out
    assert "PCR0 differs" in captured.err
