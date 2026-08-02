"""Create the first bounded allocation frontier from measured state.

The host selects only bounded candidate inputs.  The coordinator enclave
independently repeats every database read, validates the signed allocation and
reward authorities, derives exact cumulative balances, and signs the bootstrap
document.  No historical allocation graph is materialized by this path.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_v2_store import (
    EXECUTION_RESULT_TABLE,
    load_allocation_settlement_frontier_context_v2,
    load_business_artifact_graphs_by_ref_v2,
    load_execution_result_by_receipt_v2,
    persist_allocation_settlement_frontier_v2,
)
from gateway.research_lab.store import select_many
from gateway.tee.coordinator_allocation_frontier_bootstrap_v2 import (
    select_latest_allocation_source_row_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    validate_allocation_settlement_frontier_bootstrap_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    validate_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_receipt_graph,
)


_EXECUTION_COLUMNS = (
    "receipt_hash,schema_version,role,operation,purpose,job_id,epoch_id,"
    "sequence,release_hash,input_root,output_root,artifact_root,result_hash,"
    "artifact_hashes,result_doc"
)


class AllocationSettlementFrontierBootstrapV2Error(RuntimeError):
    """The first bounded allocation frontier could not be installed."""


class AllocationSettlementFrontierBootstrapV2Unsupported(
    AllocationSettlementFrontierBootstrapV2Error
):
    """The running coordinator predates measured frontier bootstrapping."""


async def load_latest_checkpointed_allocation_source_v2(
    *,
    through_epoch: int,
    select_results: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = (
        select_many
    ),
    load_result: Callable[..., Awaitable[Mapping[str, Any]]] = (
        load_execution_result_by_receipt_v2
    ),
) -> Optional[dict[str, Any]]:
    """Load one latest source only through its bounded ancestry certificate."""

    rows = await select_results(
        EXECUTION_RESULT_TABLE,
        columns=_EXECUTION_COLUMNS,
        filters=(
            ("role", "gateway_coordinator"),
            ("operation", "research_lab_allocation"),
            ("purpose", "research_lab.allocation.v2"),
            ("epoch_id", "lte", int(through_epoch)),
        ),
        order_by=(("epoch_id", True), ("receipt_hash", False)),
        limit=100,
    )
    if not rows:
        return None
    selected = select_latest_allocation_source_row_v2(
        rows,
        through_epoch=int(through_epoch),
    )
    source = dict(
        await load_result(
            str(selected["receipt_hash"]),
            expected_operation="research_lab_allocation",
            expected_purpose="research_lab.allocation.v2",
            require_checkpointed_graph=True,
        )
    )
    if canonical_json(source.get("row")) != canonical_json(selected):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "latest checkpointed allocation source differs"
        )
    return source


async def _load_candidate_reward_graphs_v2(
    *,
    source: Mapping[str, Any],
    load_business_graphs: Callable[..., Awaitable[Mapping[Any, Any]]] = (
        load_business_artifact_graphs_by_ref_v2
    ),
) -> list[dict[str, Any]]:
    result = source.get("result")
    source_graph = source.get("receipt_graph")
    if not isinstance(result, Mapping) or not isinstance(source_graph, Mapping):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "latest allocation source is incomplete"
        )
    source_state = result.get("source_state")
    if not isinstance(source_state, Mapping):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "latest allocation source state is unavailable"
        )
    def source_ids(field: str, count_field: str, *identity_fields: str) -> list[str]:
        rows = source_state.get(field, [])
        count = source_state.get(count_field, 0)
        if (
            not isinstance(rows, list)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(rows)
        ):
            raise AllocationSettlementFrontierBootstrapV2Error(
                "latest allocation reward identity set is invalid"
            )
        identities: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise AllocationSettlementFrontierBootstrapV2Error(
                    "latest allocation reward identity is invalid"
                )
            values = {str(row.get(name) or "") for name in identity_fields}
            if len(values) != 1 or "" in values:
                raise AllocationSettlementFrontierBootstrapV2Error(
                    "latest allocation reward identity differs"
                )
            identities.append(next(iter(values)))
        if len(identities) != len(set(identities)):
            raise AllocationSettlementFrontierBootstrapV2Error(
                "latest allocation reward identity is duplicated"
            )
        return sorted(identities)

    champion_ids = source_ids(
        "champion_obligations",
        "champion_obligation_count",
        "source_id",
        "champion_reward_id",
    )
    source_add_ids = source_ids(
        "source_add_obligations",
        "source_add_obligation_count",
        "source_id",
        "source_add_reward_id",
    )
    refs = {
        ("champion_reward_decision", source_id)
        for source_id in champion_ids
    } | {
        ("source_add_reward_decision", source_id)
        for source_id in source_add_ids
    }
    if any(not kind or not ref for kind, ref in refs):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "allocation reward graph reference is invalid"
        )
    loaded = await load_business_graphs(refs)
    if not isinstance(loaded, Mapping) or set(loaded) != refs:
        raise AllocationSettlementFrontierBootstrapV2Error(
            "allocation reward graph authority is incomplete"
        )
    graphs = {str(source_graph.get("root_receipt_hash") or ""): dict(source_graph)}
    for key in sorted(refs):
        graph = loaded[key]
        if not isinstance(graph, Mapping):
            raise AllocationSettlementFrontierBootstrapV2Error(
                "allocation reward graph authority is invalid"
            )
        root = str(graph.get("root_receipt_hash") or "")
        if not root or root in graphs:
            raise AllocationSettlementFrontierBootstrapV2Error(
                "allocation bootstrap graph root is duplicated"
            )
        graphs[root] = dict(graph)
    for graph in graphs.values():
        validate_receipt_graph(graph)
    return [graphs[root] for root in sorted(graphs)]


async def ensure_allocation_settlement_frontier_v2(
    *,
    netuid: int,
    through_epoch: int,
    release_manifest: Mapping[str, Any],
    supported_operations: Sequence[str],
    client: Any,
    boot_verifier: Any,
    execute: Callable[..., Awaitable[Mapping[str, Any]]] = (
        execute_coordinator_v2
    ),
    load_context: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = (
        load_allocation_settlement_frontier_context_v2
    ),
    load_source: Callable[..., Awaitable[Mapping[str, Any]]] = (
        load_latest_checkpointed_allocation_source_v2
    ),
    load_reward_graphs: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = (
        _load_candidate_reward_graphs_v2
    ),
    persist_frontier: Callable[..., Awaitable[Mapping[str, Any]]] = (
        persist_allocation_settlement_frontier_v2
    ),
) -> dict[str, Any]:
    """Install or reread one first frontier without historical replay."""

    existing = await load_context(
        netuid=int(netuid),
        before_epoch=int(through_epoch) + 1,
    )
    if existing is not None:
        return {"status": "already_initialized", "context": dict(existing)}
    raw_source = await load_source(through_epoch=int(through_epoch))
    if raw_source is None:
        return {"status": "awaiting_first_allocation"}
    if not isinstance(raw_source, Mapping):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "checkpointed allocation source is invalid"
        )
    source = dict(raw_source)
    source_row = source.get("row")
    source_receipt = source.get("receipt")
    source_result = source.get("result")
    if (
        not isinstance(source_row, Mapping)
        or not isinstance(source_receipt, Mapping)
        or not isinstance(source_result, Mapping)
    ):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "checkpointed allocation source is incomplete"
        )
    source_hash = str(source_receipt.get("receipt_hash") or "")
    source_state = source_result.get("source_state")
    source_state_hash = str(source_result.get("source_state_hash") or "")
    if not isinstance(source_state, Mapping):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "checkpointed allocation source state is unavailable"
        )
    signed_frontier = source_state.get("settlement_frontier")
    if isinstance(signed_frontier, Mapping):
        frontier = validate_allocation_settlement_frontier_v2(signed_frontier)
        if (
            frontier.get("mode") != "legacy_full_history_bootstrap"
            or frontier.get("predecessor_frontier_hash") is not None
            or int(frontier.get("netuid", -1)) != int(netuid)
            or int(frontier.get("allocation_epoch", -1))
            != int(source_row.get("epoch_id", -2))
            or int(source_state.get("netuid", -1)) != int(netuid)
            or int(source_state.get("epoch", -1))
            != int(frontier["allocation_epoch"])
            or source_state_hash != sha256_json(dict(source_state))
        ):
            raise AllocationSettlementFrontierBootstrapV2Error(
                "signed allocation frontier recovery authority differs"
            )
        await persist_frontier(
            frontier=frontier,
            source_receipt_hash=source_hash,
            source_state_hash=source_state_hash,
        )
        context = await load_context(
            netuid=int(netuid),
            before_epoch=int(through_epoch) + 1,
        )
        if (
            not isinstance(context, Mapping)
            or context.get("frontier") != frontier
        ):
            raise AllocationSettlementFrontierBootstrapV2Error(
                "recovered allocation frontier durable readback differs"
            )
        return {
            "status": "recovered_signed_frontier",
            "frontier_hash": str(frontier["frontier_hash"]),
            "allocation_epoch": int(frontier["allocation_epoch"]),
            "source_receipt_hash": source_hash,
            "context": dict(context),
        }
    if signed_frontier is not None:
        raise AllocationSettlementFrontierBootstrapV2Error(
            "checkpointed allocation source frontier is invalid"
        )
    if ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION not in set(
        supported_operations
    ):
        raise AllocationSettlementFrontierBootstrapV2Unsupported(
            "running coordinator does not advertise allocation frontier bootstrap"
        )
    parent_graphs = list(await load_reward_graphs(source=source))
    parent_roots = {
        str(graph.get("root_receipt_hash") or "") for graph in parent_graphs
    }
    if source_hash not in parent_roots:
        raise AllocationSettlementFrontierBootstrapV2Error(
            "checkpointed allocation source graph is absent"
        )
    outcome = await execute(
        operation=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        epoch_id=int(through_epoch),
        sequence=0,
        payload={
            "schema_version": (
                ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION
            ),
            "netuid": int(netuid),
            "through_epoch": int(through_epoch),
            "allocation_source_receipt_hash": source_hash,
        },
        parent_graphs=tuple(parent_graphs),
        require_egress_proxy=False,
        release_manifest=release_manifest,
        client=client,
        boot_verifier=boot_verifier,
    )
    result = outcome.get("result")
    receipt = outcome.get("execution_receipt") or outcome.get("receipt")
    graph = outcome.get("execution_receipt_graph") or outcome.get("receipt_graph")
    if (
        not isinstance(result, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(graph, Mapping)
    ):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "measured allocation frontier bootstrap is incomplete"
        )
    bootstrap = validate_allocation_settlement_frontier_bootstrap_v2(result)
    if (
        bootstrap["allocation_source_receipt_hash"] != source_hash
        or bootstrap["source_state_hash"]
        != str(source_result.get("source_state_hash") or "")
        or receipt.get("purpose")
        != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
        or receipt.get("status") != "succeeded"
        or receipt.get("output_root") != sha256_json(dict(bootstrap))
        or graph.get("root_receipt_hash") != receipt.get("receipt_hash")
    ):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "measured allocation frontier bootstrap differs"
        )
    validate_receipt_graph(graph)
    await persist_frontier(
        frontier=bootstrap["frontier"],
        source_receipt_hash=str(receipt["receipt_hash"]),
        source_state_hash=str(bootstrap["source_state_hash"]),
    )
    context = await load_context(
        netuid=int(netuid),
        before_epoch=int(through_epoch) + 1,
    )
    if (
        not isinstance(context, Mapping)
        or context.get("frontier") != bootstrap["frontier"]
    ):
        raise AllocationSettlementFrontierBootstrapV2Error(
            "allocation frontier bootstrap durable readback differs"
        )
    return {
        "status": "initialized",
        "bootstrap_hash": str(bootstrap["bootstrap_hash"]),
        "frontier_hash": str(bootstrap["frontier"]["frontier_hash"]),
        "allocation_epoch": int(bootstrap["allocation_epoch"]),
        "source_receipt_hash": str(receipt["receipt_hash"]),
        "context": dict(context),
    }
