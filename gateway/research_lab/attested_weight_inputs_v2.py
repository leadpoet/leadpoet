"""Build the complete gateway-owned input set for validator V2 authority."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_v2_store import (
    load_sourcing_epoch_graphs_v2,
)
from gateway.tee.coordinator_executor_v2 import OP_ATTEST_WEIGHT_INPUT
from gateway.utils.tee_client import TEEClient, coordinator_tee_client
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_receipt_graph,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    gateway_weight_input_value_documents_v2,
)


_ALLOCATION_CATEGORIES = frozenset(
    {
        "research_lab_allocation",
        "champions",
        "reimbursements",
        "source_add_rewards",
    }
)
_ANOMALY_SOURCE_CATEGORIES = (
    "research_lab_allocation",
    "fulfillment_rewards",
    "leaderboard",
    "bans",
    "sourcing_history",
)


class AttestedWeightInputsV2Error(RuntimeError):
    """Gateway weight inputs are incomplete, conflicting, or unverifiable."""


def _coordinator_client() -> TEEClient:
    """Return one connection owner for one concurrent coordinator job."""

    return TEEClient(
        cid=coordinator_tee_client.cid,
        port=coordinator_tee_client.port,
    )


def _union_receipt_sets(
    graphs: Sequence[Mapping[str, Any]],
) -> Dict[str, list[dict[str, Any]]]:
    collections = {
        "boot_identities": ("boot_identity_hash", {}),
        "receipts": ("receipt_hash", {}),
        "transport_attempts": ("attempt_hash", {}),
        "host_operations": ("request", {}),
    }
    for graph in graphs:
        validate_receipt_graph(graph)
        for field, (key_field, values) in collections.items():
            for item in graph[field]:
                key = (
                    str(item["request"]["request_hash"])
                    if key_field == "request"
                    else str(item[key_field])
                )
                normalized = dict(item)
                if key in values and values[key] != normalized:
                    raise AttestedWeightInputsV2Error(
                        "V2 receipt graph hash conflicts across weight inputs"
                    )
                values[key] = normalized
    return {
        field: [values[key] for key in sorted(values)]
        for field, (_key_field, values) in collections.items()
    }


async def build_gateway_weight_inputs_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    allocation_graph: Mapping[str, Any],
    leaderboard_window_start: str,
    leaderboard_window_end: str,
    execute: Any = execute_coordinator_v2,
    load_sourcing_graphs: Any = load_sourcing_epoch_graphs_v2,
    execution_options: Mapping[str, Any] | None = None,
    coordinator_client_factory: Any = _coordinator_client,
) -> dict[str, Any]:
    """Produce every coordinator input receipt from measured source reads."""

    calculation = dict(calculation_snapshot)
    epoch_id = calculation.get("epoch_id")
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
        raise AttestedWeightInputsV2Error("weight input epoch is invalid")
    validate_receipt_graph(
        allocation_graph,
        required_purposes={"research_lab.allocation.v2"},
    )
    allocation_receipts = {
        str(receipt["receipt_hash"]): receipt
        for receipt in allocation_graph["receipts"]
    }
    allocation_hash = str(allocation_graph["root_receipt_hash"])
    allocation_receipt = allocation_receipts.get(allocation_hash)
    if (
        not isinstance(allocation_receipt, Mapping)
        or allocation_receipt.get("role") != "gateway_coordinator"
        or allocation_receipt.get("purpose") != "research_lab.allocation.v2"
        or int(allocation_receipt.get("epoch_id", -1)) != epoch_id
    ):
        raise AttestedWeightInputsV2Error(
            "allocation graph root is not the epoch authority receipt"
        )
    expected_documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=calculation,
        gateway_authority_event_hash=allocation_hash,
    )
    sourcing_graphs = await load_sourcing_graphs(current_epoch=epoch_id)
    options = dict(execution_options or {})
    executions: dict[str, dict[str, Any]] = {}
    independent_categories = sorted(
        category
        for category in GATEWAY_WEIGHT_INPUT_CATEGORIES
        if category != "anomaly_adjustments"
    )
    ordered_categories = independent_categories + ["anomaly_adjustments"]

    async def execute_category(category: str, sequence: int) -> None:
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        if role != "gateway_coordinator":
            raise AttestedWeightInputsV2Error(
                "gateway input category has a non-coordinator role"
            )
        if category in _ALLOCATION_CATEGORIES:
            parents = (allocation_graph,)
        elif category == "sourcing_history":
            parents = tuple(sourcing_graphs)
        elif category == "anomaly_adjustments":
            missing = [
                source
                for source in _ANOMALY_SOURCE_CATEGORIES
                if source not in executions
            ]
            if missing:
                raise AttestedWeightInputsV2Error(
                    "anomaly sources were not executed first"
                )
            parents = tuple(
                executions[source].get("execution_receipt_graph")
                or executions[source]["receipt_graph"]
                for source in _ANOMALY_SOURCE_CATEGORIES
            )
        else:
            parents = ()
        payload = {
            "category": category,
            "calculation_snapshot": calculation,
            "gateway_authority_event_hash": allocation_hash,
            "allocation_receipt": dict(allocation_receipt),
            "leaderboard_window_start": str(leaderboard_window_start),
            "leaderboard_window_end": str(leaderboard_window_end),
        }
        if category == "anomaly_adjustments":
            payload["upstream_documents"] = {
                source: dict(executions[source]["result"])
                for source in _ANOMALY_SOURCE_CATEGORIES
            }
        category_options = dict(options)
        category_client = coordinator_client_factory()
        category_options.update(
            {
                "client": category_client,
                "credential_coordinator_client": category_client,
                "artifact_coordinator_client": category_client,
            }
        )
        value = await execute(
            operation=OP_ATTEST_WEIGHT_INPUT,
            purpose=purpose,
            epoch_id=epoch_id,
            sequence=sequence,
            payload=payload,
            parent_graphs=parents,
            **category_options,
        )
        if not isinstance(value, Mapping) or value.get("status") != "succeeded":
            raise AttestedWeightInputsV2Error(
                "%s measured input failed" % category
            )
        graph = value.get("receipt_graph")
        root_receipt = value.get("receipt")
        receipt = value.get("execution_receipt") or root_receipt
        document = value.get("result")
        if (
            not isinstance(graph, Mapping)
            or not isinstance(root_receipt, Mapping)
            or not isinstance(receipt, Mapping)
            or not isinstance(document, Mapping)
        ):
            raise AttestedWeightInputsV2Error(
                "%s measured input is incomplete" % category
            )
        validate_receipt_graph(graph, required_purposes={purpose})
        receipts_by_hash = {
            str(item.get("receipt_hash") or ""): item
            for item in graph.get("receipts") or ()
            if isinstance(item, Mapping)
        }
        root_hash = str(graph.get("root_receipt_hash") or "")
        receipt_hash = str(receipt.get("receipt_hash") or "")
        expected_role, _expected_purpose = WEIGHT_INPUT_PURPOSES[category]
        if (
            root_receipt.get("receipt_hash") != root_hash
            or receipts_by_hash.get(receipt_hash) != receipt
            or receipt.get("role") != expected_role
            or receipt.get("purpose") != purpose
            or receipt.get("output_root") != sha256_json(document)
            or canonical_json(document)
            != canonical_json(expected_documents[category])
        ):
            raise AttestedWeightInputsV2Error(
                "%s measured input differs from calculation" % category
            )
        if root_hash != receipt_hash:
            if (
                root_receipt.get("role") != "gateway_coordinator"
                or root_receipt.get("purpose")
                != "leadpoet.artifact_persistence.v2"
                or receipt_hash
                not in (root_receipt.get("parent_receipt_hashes") or ())
            ):
                raise AttestedWeightInputsV2Error(
                    "%s persistence lineage differs from measured input"
                    % category
                )
        executions[category] = dict(value)

    # Every category except anomaly_adjustments has independent measured
    # inputs. Run those jobs together so their mandatory encrypted artifact
    # persistence does not consume the block-345 submission window serially.
    # The anomaly job remains ordered after all of its receipt parents exist.
    await asyncio.gather(
        *(
            execute_category(category, sequence)
            for sequence, category in enumerate(independent_categories)
        )
    )
    await execute_category("anomaly_adjustments", len(independent_categories))

    all_graphs = [
        executions[category]["receipt_graph"]
        for category in sorted(executions)
    ]
    receipt_set = _union_receipt_sets(all_graphs)
    input_hashes = {
        category: str(
            (
                executions[category].get("execution_receipt")
                or executions[category]["receipt"]
            )["receipt_hash"]
        )
        for category in sorted(executions)
    }
    if set(input_hashes) != set(GATEWAY_WEIGHT_INPUT_CATEGORIES):
        raise AttestedWeightInputsV2Error(
            "gateway V2 weight input categories are incomplete"
        )
    return {
        "input_receipt_hashes": input_hashes,
        "gateway_authority_event_hash": allocation_hash,
        "upstream_receipt_set": receipt_set,
        "executions": executions,
    }
