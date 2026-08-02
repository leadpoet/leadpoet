from __future__ import annotations

from copy import deepcopy

import pytest

from gateway.tee import bootstrap_allocation_settlement_frontier_v2 as host_bootstrap
from gateway.tee.coordinator_allocation_frontier_bootstrap_v2 import (
    CoordinatorAllocationFrontierBootstrapV2,
    CoordinatorAllocationFrontierBootstrapV2Error,
    select_latest_allocation_source_row_v2,
)
from gateway.tee.execution_job_manager_v2 import (
    ExecutionContextV2,
    MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES,
    MAX_EXTERNAL_RECEIPT_GRAPHS,
    _job_external_authority_limit,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    build_allocation_settlement_frontier_bootstrap_v2,
    validate_allocation_settlement_frontier_bootstrap_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    MAX_REWARD_CHECKPOINTS,
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import merkle_root, sha256_json


def _sha(value: int) -> str:
    return "sha256:%064x" % value


def _source_row(*, epoch: int, receipt: int, marker: str = "same") -> dict:
    source_state = {
        "epoch": epoch,
        "netuid": 71,
        "settlement_frontier": None,
        "marker": marker,
    }
    allocation = {"epoch": epoch, "marker": marker}
    source_state_hash = sha256_json(source_state)
    artifact_hashes = sorted({source_state_hash, sha256_json(allocation)})
    result = {
        "source_state": source_state,
        "source_state_hash": source_state_hash,
        "allocation": allocation,
    }
    return {
        "receipt_hash": _sha(receipt),
        "schema_version": "leadpoet.attested_execution_result.v2",
        "role": "gateway_coordinator",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "job_id": "allocation:%d" % epoch,
        "epoch_id": epoch,
        "sequence": 0,
        "release_hash": _sha(900),
        "input_root": _sha(901),
        "output_root": sha256_json({"allocation": allocation}),
        "artifact_root": merkle_root(
            artifact_hashes,
            domain="leadpoet-artifact-v2",
        ),
        "result_hash": sha256_json(result),
        "artifact_hashes": artifact_hashes,
        "result_doc": result,
    }


def _source(*, epoch: int, receipt: int, frontier: dict | None) -> dict:
    row = _source_row(epoch=epoch, receipt=receipt)
    source_state = dict(row["result_doc"]["source_state"])
    source_state["settlement_frontier"] = frontier
    result = dict(row["result_doc"])
    result["source_state"] = source_state
    result["source_state_hash"] = sha256_json(source_state)
    row["result_doc"] = result
    row["result_hash"] = sha256_json(result)
    row["artifact_hashes"] = sorted(
        {result["source_state_hash"], sha256_json(result["allocation"])}
    )
    row["artifact_root"] = merkle_root(
        row["artifact_hashes"],
        domain="leadpoet-artifact-v2",
    )
    return {
        "row": row,
        "result": result,
        "receipt": {"receipt_hash": row["receipt_hash"]},
        "receipt_graph": {"root_receipt_hash": row["receipt_hash"]},
        "artifact_hashes": list(row["artifact_hashes"]),
    }


def _champion_reward_id() -> str:
    return "champion_reward:" + _sha(70)


def _champion_reward_row() -> dict:
    return {
        "champion_reward_id": _champion_reward_id(),
        "score_bundle_id": "score-bundle:frontier",
        "candidate_id": "candidate:frontier",
        "run_id": "run:frontier",
        "miner_hotkey": "5FfrontierChampion",
        "miner_uid": 14,
        "island": "generalist",
        "policy_id": "policy:frontier",
        "evaluation_epoch": 179,
        "start_epoch": 180,
        "epoch_count": 20,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 7.3,
        "source_score_bundle_hash": _sha(71),
        "input_hash": _sha(72),
        "anchored_hash": _sha(73),
        "current_reward_status": "partially_paid",
    }


def _source_row_with_champion(*, epoch: int = 200) -> dict:
    row = _source_row(epoch=epoch, receipt=1)
    source_id = _champion_reward_id()
    obligation = {
        "uid": 14,
        "miner_uid": 14,
        "miner_hotkey": "5FfrontierChampion",
        "source_id": source_id,
        "champion_reward_id": source_id,
        "candidate_id": "candidate:frontier",
        "score_bundle_id": "score-bundle:frontier",
        "run_id": "run:frontier",
        "island": "generalist",
        "status": "active",
        "reward_kind": "champion",
        "start_epoch": 180,
        "epoch_count": 20,
        "nominal_end_epoch": 200,
        "improvement_points": 2.0,
        "threshold_points": 1.0,
        "desired_alpha_percent": 7.3,
        "total_due_alpha_percent": 146.0,
        "paid_alpha_percent_to_date": 30.0,
        "remaining_alpha_percent": 116.0,
        "current_epoch_desired_alpha_percent": 7.3,
        "champ_cap_enabled": True,
        "replay_status": "extended_replay",
    }
    result = deepcopy(row["result_doc"])
    result["source_state"] = {
        "epoch": epoch,
        "netuid": 71,
        "policy": {"enable_champ_cap": True},
        "settlement_frontier": None,
        "champion_obligation_count": 1,
        "champion_obligations": [obligation],
        "source_add_obligation_count": 0,
        "source_add_obligations": [],
        "skipped": {"champions": [], "source_add": []},
    }
    result["source_state_hash"] = sha256_json(result["source_state"])
    row["result_doc"] = result
    row["result_hash"] = sha256_json(result)
    row["artifact_hashes"] = sorted(
        {result["source_state_hash"], sha256_json(result["allocation"])}
    )
    row["artifact_root"] = merkle_root(
        row["artifact_hashes"],
        domain="leadpoet-artifact-v2",
    )
    return row


def _source_add_reward_id() -> str:
    return "source_add_reward:0123456789abcdef"


def _source_add_reward_row() -> dict:
    return {
        "reward_ref": _source_add_reward_id(),
        "adapter_id": "adapter:frontier",
        "miner_hotkey": "5FfrontierSourceAdd",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 180,
        "current_reward_status": "partially_paid",
        "trigger_evidence_doc": {"functional_probe_passed": True},
        "public_label": "Source acceptance reward",
        "desired_alpha_percent": 1.0,
        "epoch_count": 20,
    }


def _source_row_with_source_add(*, epoch: int = 200) -> dict:
    row = _source_row(epoch=epoch, receipt=1)
    source_id = _source_add_reward_id()
    obligation = {
        "uid": 15,
        "miner_uid": 15,
        "miner_hotkey": "5FfrontierSourceAdd",
        "source_id": source_id,
        "source_add_reward_id": source_id,
        "adapter_id": "adapter:frontier",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "status": "active",
        "start_epoch": 180,
        "epoch_count": 20,
        "nominal_end_epoch": 200,
        "improvement_points": 0.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 1.0,
        "total_due_alpha_percent": 20.0,
        "paid_alpha_percent_to_date": 4.0,
        "remaining_alpha_percent": 16.0,
        "current_epoch_desired_alpha_percent": 1.0,
        "champ_cap_enabled": True,
        "replay_status": "extended_replay",
    }
    result = deepcopy(row["result_doc"])
    result["source_state"] = {
        "epoch": epoch,
        "netuid": 71,
        "policy": {"enable_champ_cap": True},
        "settlement_frontier": None,
        "champion_obligation_count": 0,
        "champion_obligations": [],
        "source_add_obligation_count": 1,
        "source_add_obligations": [obligation],
        "skipped": {"champions": [], "source_add": []},
    }
    result["source_state_hash"] = sha256_json(result["source_state"])
    row["result_doc"] = result
    row["result_hash"] = sha256_json(result)
    row["artifact_hashes"] = sorted(
        {result["source_state_hash"], sha256_json(result["allocation"])}
    )
    row["artifact_root"] = merkle_root(
        row["artifact_hashes"],
        domain="leadpoet-artifact-v2",
    )
    return row


class _FakeReader:
    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [deepcopy(item) for item in self.rows.get(policy_id, [])]


def test_bootstrap_document_is_hash_bound_and_fail_closed() -> None:
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=100,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    document = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=71,
        bootstrap_epoch=101,
        allocation_source_receipt_hash=_sha(1),
        source_state_hash=_sha(2),
        frontier=frontier,
    )

    assert validate_allocation_settlement_frontier_bootstrap_v2(document) == document
    tampered = deepcopy(document)
    tampered["bootstrap_epoch"] = 102
    with pytest.raises(ValueError, match="hash differs"):
        validate_allocation_settlement_frontier_bootstrap_v2(tampered)


def test_latest_source_selector_accepts_a_full_page_with_older_epochs() -> None:
    rows = [_source_row(epoch=200, receipt=1)] + [
        _source_row(epoch=epoch, receipt=300 - epoch)
        for epoch in range(199, 100, -1)
    ]

    selected = select_latest_allocation_source_row_v2(rows, through_epoch=200)

    assert len(rows) == 100
    assert selected["epoch_id"] == 200


def test_latest_source_selector_rejects_truncated_latest_epoch() -> None:
    row = _source_row(epoch=200, receipt=1)
    rows = []
    for receipt in range(1, 101):
        candidate = deepcopy(row)
        candidate["receipt_hash"] = _sha(receipt)
        rows.append(candidate)

    with pytest.raises(
        CoordinatorAllocationFrontierBootstrapV2Error,
        match="truncated",
    ):
        select_latest_allocation_source_row_v2(rows, through_epoch=200)


def test_latest_source_selector_rejects_conflicting_same_epoch_results() -> None:
    rows = [
        _source_row(epoch=200, receipt=1, marker="a"),
        _source_row(epoch=200, receipt=2, marker="b"),
    ]

    with pytest.raises(
        CoordinatorAllocationFrontierBootstrapV2Error,
        match="ambiguous",
    ):
        select_latest_allocation_source_row_v2(rows, through_epoch=200)


def test_measured_bootstrap_uses_exact_signed_reward_identities(monkeypatch) -> None:
    source_row = _source_row_with_champion()
    reward_row = _champion_reward_row()
    reward_receipt_hash = _sha(2)
    reader = _FakeReader(
        {
            "allocation_settlement_frontier_activation": [],
            "allocation_settlement_frontiers": [],
            "latest_attested_allocation_execution_results": [source_row],
            "champion_reward_by_id": [reward_row],
        }
    )
    resolver = CoordinatorAllocationFrontierBootstrapV2(reader)
    monkeypatch.setattr(
        resolver,
        "_validate_source_receipt",
        lambda **_kwargs: None,
    )

    def require_reward_receipt(**kwargs) -> None:
        assert kwargs["artifact_kind"] == "champion_reward_decision"
        assert kwargs["artifact_ref"] == _champion_reward_id()
        kwargs["required_parents"].add(reward_receipt_hash)

    monkeypatch.setattr(resolver, "_require_reward_receipt", require_reward_receipt)
    context = ExecutionContextV2(
        job_id="allocation-frontier-bootstrap:200",
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        epoch_id=200,
        parent_receipt_hashes=(source_row["receipt_hash"], reward_receipt_hash),
    )

    result = resolver.resolve(
        payload={
            "schema_version": (
                ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
            ),
            "netuid": 71,
            "through_epoch": 200,
            "allocation_source_receipt_hash": source_row["receipt_hash"],
        },
        context=context,
    )

    checkpoint = result["frontier"]["reward_checkpoints"][0]
    assert checkpoint["source_id"] == _champion_reward_id()
    assert checkpoint["applied_alpha_percent"] == "30.000000"
    assert (
        "champion_reward_by_id",
        {"champion_reward_id": _champion_reward_id()},
    ) in reader.calls
    assert all(
        policy_id not in {"allocation_champion_rewards", "allocation_source_add_rewards"}
        for policy_id, _parameters in reader.calls
    )


def test_measured_bootstrap_uses_exact_signed_source_add_identity(monkeypatch) -> None:
    source_row = _source_row_with_source_add()
    reward_row = _source_add_reward_row()
    reward_receipt_hash = _sha(2)
    reader = _FakeReader(
        {
            "allocation_settlement_frontier_activation": [],
            "allocation_settlement_frontiers": [],
            "latest_attested_allocation_execution_results": [source_row],
            "source_add_reward_by_ref": [reward_row],
        }
    )
    resolver = CoordinatorAllocationFrontierBootstrapV2(reader)
    monkeypatch.setattr(
        resolver,
        "_validate_source_receipt",
        lambda **_kwargs: None,
    )

    def require_reward_receipt(**kwargs) -> None:
        assert kwargs["artifact_kind"] == "source_add_reward_decision"
        assert kwargs["artifact_ref"] == _source_add_reward_id()
        kwargs["required_parents"].add(reward_receipt_hash)

    monkeypatch.setattr(resolver, "_require_reward_receipt", require_reward_receipt)
    context = ExecutionContextV2(
        job_id="allocation-frontier-bootstrap:200",
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        epoch_id=200,
        parent_receipt_hashes=(source_row["receipt_hash"], reward_receipt_hash),
    )

    result = resolver.resolve(
        payload={
            "schema_version": (
                ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
            ),
            "netuid": 71,
            "through_epoch": 200,
            "allocation_source_receipt_hash": source_row["receipt_hash"],
        },
        context=context,
    )

    checkpoint = result["frontier"]["reward_checkpoints"][0]
    assert checkpoint["reward_kind"] == "source_add"
    assert checkpoint["source_id"] == _source_add_reward_id()
    assert checkpoint["applied_alpha_percent"] == "4.000000"
    assert (
        "source_add_reward_by_ref",
        {"reward_ref": _source_add_reward_id()},
    ) in reader.calls


def test_measured_bootstrap_rejects_unrepresentable_skipped_rewards(
    monkeypatch,
) -> None:
    source_row = _source_row_with_champion()
    source_state = source_row["result_doc"]["source_state"]
    source_state["champion_obligation_count"] = 0
    source_state["champion_obligations"] = []
    source_state["skipped"]["champions"] = [
        {
            "champion_reward_id": _champion_reward_id(),
            "reason": "miner_hotkey_not_registered",
        }
    ]
    source_row["result_doc"]["source_state_hash"] = sha256_json(source_state)
    source_row["result_hash"] = sha256_json(source_row["result_doc"])
    source_row["artifact_hashes"] = sorted(
        {
            source_row["result_doc"]["source_state_hash"],
            sha256_json(source_row["result_doc"]["allocation"]),
        }
    )
    source_row["artifact_root"] = merkle_root(
        source_row["artifact_hashes"],
        domain="leadpoet-artifact-v2",
    )
    reader = _FakeReader(
        {
            "allocation_settlement_frontier_activation": [],
            "allocation_settlement_frontiers": [],
            "latest_attested_allocation_execution_results": [source_row],
        }
    )
    resolver = CoordinatorAllocationFrontierBootstrapV2(reader)
    monkeypatch.setattr(
        resolver,
        "_validate_source_receipt",
        lambda **_kwargs: None,
    )

    with pytest.raises(
        CoordinatorAllocationFrontierBootstrapV2Error,
        match="unrepresentable skipped reward",
    ):
        resolver.resolve(
            payload={
                "schema_version": (
                    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
                ),
                "netuid": 71,
                "through_epoch": 200,
                "allocation_source_receipt_hash": source_row["receipt_hash"],
            },
            context=ExecutionContextV2(
                job_id="allocation-frontier-bootstrap:200",
                purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
                epoch_id=200,
                parent_receipt_hashes=(source_row["receipt_hash"],),
            ),
        )


@pytest.mark.parametrize("rows", [[], [_champion_reward_row(), _champion_reward_row()]])
def test_measured_bootstrap_rejects_missing_or_ambiguous_exact_reward(
    rows,
    monkeypatch,
) -> None:
    source_row = _source_row_with_champion()
    reader = _FakeReader(
        {
            "allocation_settlement_frontier_activation": [],
            "allocation_settlement_frontiers": [],
            "latest_attested_allocation_execution_results": [source_row],
            "champion_reward_by_id": rows,
        }
    )
    resolver = CoordinatorAllocationFrontierBootstrapV2(reader)
    monkeypatch.setattr(
        resolver,
        "_validate_source_receipt",
        lambda **_kwargs: None,
    )

    with pytest.raises(
        CoordinatorAllocationFrontierBootstrapV2Error,
        match="immutable reward source is missing or ambiguous",
    ):
        resolver.resolve(
            payload={
                "schema_version": (
                    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
                ),
                "netuid": 71,
                "through_epoch": 200,
                "allocation_source_receipt_hash": source_row["receipt_hash"],
            },
            context=ExecutionContextV2(
                job_id="allocation-frontier-bootstrap:200",
                purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
                epoch_id=200,
                parent_receipt_hashes=(source_row["receipt_hash"],),
            ),
        )


@pytest.mark.asyncio
async def test_host_graph_selection_is_derived_only_from_signed_source(
    monkeypatch,
) -> None:
    champion_id = _champion_reward_id()
    source_add_id = "source_add_reward:0123456789abcdef"
    source_graph = {"root_receipt_hash": _sha(1)}
    source = {
        "result": {
            "source_state": {
                "champion_obligation_count": 1,
                "champion_obligations": [
                    {
                        "source_id": champion_id,
                        "champion_reward_id": champion_id,
                    }
                ],
                "source_add_obligation_count": 1,
                "source_add_obligations": [
                    {
                        "source_id": source_add_id,
                        "source_add_reward_id": source_add_id,
                    }
                ],
            }
        },
        "receipt_graph": source_graph,
    }
    requested = []

    async def load_business_graphs(refs):
        requested.append(set(refs))
        return {
            ("champion_reward_decision", champion_id): {
                "root_receipt_hash": _sha(2)
            },
            ("source_add_reward_decision", source_add_id): {
                "root_receipt_hash": _sha(3)
            },
        }

    monkeypatch.setattr(host_bootstrap, "validate_receipt_graph", lambda _value: None)
    graphs = await host_bootstrap._load_candidate_reward_graphs_v2(
        source=source,
        load_business_graphs=load_business_graphs,
    )

    assert requested == [
        {
            ("champion_reward_decision", champion_id),
            ("source_add_reward_decision", source_add_id),
        }
    ]
    assert [graph["root_receipt_hash"] for graph in graphs] == [
        _sha(1),
        _sha(2),
        _sha(3),
    ]


@pytest.mark.asyncio
async def test_latest_checkpointed_source_requires_bounded_graph() -> None:
    row = _source_row(epoch=200, receipt=1)
    calls = []

    async def select_results(*_args, **kwargs):
        assert kwargs["order_by"] == (("epoch_id", True), ("receipt_hash", False))
        assert kwargs["limit"] == 100
        return [row]

    async def load_result(receipt_hash, **kwargs):
        calls.append((receipt_hash, kwargs))
        return {"row": deepcopy(row)}

    result = await host_bootstrap.load_latest_checkpointed_allocation_source_v2(
        through_epoch=200,
        select_results=select_results,
        load_result=load_result,
    )

    assert result["row"] == row
    assert calls == [
        (
            row["receipt_hash"],
            {
                "expected_operation": "research_lab_allocation",
                "expected_purpose": "research_lab.allocation.v2",
                "require_checkpointed_graph": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_latest_checkpointed_source_distinguishes_a_fresh_database() -> None:
    async def select_results(*_args, **_kwargs):
        return []

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("an absent source must not load a graph")

    assert (
        await host_bootstrap.load_latest_checkpointed_allocation_source_v2(
            through_epoch=200,
            select_results=select_results,
            load_result=forbidden,
        )
        is None
    )


@pytest.mark.asyncio
async def test_startup_allows_only_an_unambiguous_fresh_database() -> None:
    async def load_context(**_kwargs):
        return None

    async def load_source(**_kwargs):
        return None

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("fresh startup must wait for its first allocation")

    result = await host_bootstrap.ensure_allocation_settlement_frontier_v2(
        netuid=71,
        through_epoch=200,
        release_manifest={},
        supported_operations=(ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,),
        client=object(),
        boot_verifier=object(),
        execute=forbidden,
        load_context=load_context,
        load_source=load_source,
        load_reward_graphs=forbidden,
        persist_frontier=forbidden,
    )

    assert result == {"status": "awaiting_first_allocation"}


@pytest.mark.asyncio
async def test_startup_recovers_a_signed_frontier_after_rpc_interruption() -> None:
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=200,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    source = _source(epoch=200, receipt=1, frontier=frontier)
    context = {"frontier": frontier}
    context_reads = []
    persisted = []

    async def load_context(**_kwargs):
        context_reads.append(1)
        return None if len(context_reads) == 1 else context

    async def load_source(**_kwargs):
        return source

    async def persist_frontier(**kwargs):
        persisted.append(kwargs)
        return {"status": "persisted"}

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("measured bootstrap must not run during recovery")

    result = await host_bootstrap.ensure_allocation_settlement_frontier_v2(
        netuid=71,
        through_epoch=201,
        release_manifest={},
        supported_operations=(),
        client=object(),
        boot_verifier=object(),
        execute=forbidden,
        load_context=load_context,
        load_source=load_source,
        load_reward_graphs=forbidden,
        persist_frontier=persist_frontier,
    )

    assert result["status"] == "recovered_signed_frontier"
    assert result["frontier_hash"] == frontier["frontier_hash"]
    assert persisted == [
        {
            "frontier": frontier,
            "source_receipt_hash": source["receipt"]["receipt_hash"],
            "source_state_hash": source["result"]["source_state_hash"],
        }
    ]


@pytest.mark.asyncio
async def test_startup_bootstraps_only_when_signed_source_has_no_frontier(
    monkeypatch,
) -> None:
    source = _source(epoch=200, receipt=1, frontier=None)
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=200,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    bootstrap = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=71,
        bootstrap_epoch=201,
        allocation_source_receipt_hash=source["receipt"]["receipt_hash"],
        source_state_hash=source["result"]["source_state_hash"],
        frontier=frontier,
    )
    receipt_hash = _sha(2)
    context_reads = []
    persisted = []

    async def load_context(**_kwargs):
        context_reads.append(1)
        return None if len(context_reads) == 1 else {"frontier": frontier}

    async def execute(**kwargs):
        assert kwargs["operation"] == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION
        assert kwargs["purpose"] == ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
        return {
            "result": bootstrap,
            "receipt": {
                "receipt_hash": receipt_hash,
                "purpose": ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
                "status": "succeeded",
                "output_root": sha256_json(bootstrap),
            },
            "receipt_graph": {"root_receipt_hash": receipt_hash},
        }

    async def persist_frontier(**kwargs):
        persisted.append(kwargs)
        return {"status": "persisted"}

    async def load_source(**_kwargs):
        return source

    async def load_reward_graphs(**_kwargs):
        return [source["receipt_graph"]]

    monkeypatch.setattr(host_bootstrap, "validate_receipt_graph", lambda _value: None)
    result = await host_bootstrap.ensure_allocation_settlement_frontier_v2(
        netuid=71,
        through_epoch=201,
        release_manifest={},
        supported_operations=(ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,),
        client=object(),
        boot_verifier=object(),
        execute=execute,
        load_context=load_context,
        load_source=load_source,
        load_reward_graphs=load_reward_graphs,
        persist_frontier=persist_frontier,
    )

    assert result["status"] == "initialized"
    assert persisted[0]["source_receipt_hash"] == receipt_hash


def test_frontier_bootstrap_authority_limit_matches_canonical_bound() -> None:
    assert MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES == (
        MAX_REWARD_CHECKPOINTS + 1
    )
    assert _job_external_authority_limit(
        operation=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    ) == MAX_ALLOCATION_FRONTIER_BOOTSTRAP_AUTHORITIES
    assert _job_external_authority_limit(
        operation="unrelated",
        purpose="research_lab.ranking.v2",
    ) == MAX_EXTERNAL_RECEIPT_GRAPHS
