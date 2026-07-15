"""Canonical gateway-to-validator handoff for one attested Lab allocation."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from leadpoet_canonical.attested_v2 import sha256_json, validate_receipt_graph


ALLOCATION_HANDOFF_SCHEMA_VERSION = "leadpoet.attested_allocation_handoff.v2"


class AllocationHandoffV2Error(ValueError):
    """The allocation handoff is incomplete or differs from its receipt graph."""


def build_allocation_handoff_v2(
    *,
    bundle: Mapping[str, Any],
    receipt_graph: Mapping[str, Any],
    lineage_bindings: Any,
    lineage_complete: bool,
    persistence: Mapping[str, Any],
) -> Dict[str, Any]:
    document = {
        "schema_version": ALLOCATION_HANDOFF_SCHEMA_VERSION,
        "bundle": dict(bundle),
        "receipt_graph": dict(receipt_graph),
        "root_receipt_hash": str(receipt_graph.get("root_receipt_hash") or ""),
        "lineage_bindings": [dict(item) for item in lineage_bindings],
        "lineage_complete": bool(lineage_complete),
        "persistence": dict(persistence),
    }
    return validate_allocation_handoff_v2(document)


def validate_allocation_handoff_v2(
    value: Mapping[str, Any],
    *,
    expected_epoch_id: int | None = None,
    expected_netuid: int | None = None,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "bundle",
        "receipt_graph",
        "root_receipt_hash",
        "lineage_bindings",
        "lineage_complete",
        "persistence",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AllocationHandoffV2Error("allocation handoff fields are invalid")
    if value.get("schema_version") != ALLOCATION_HANDOFF_SCHEMA_VERSION:
        raise AllocationHandoffV2Error("allocation handoff schema is invalid")
    bundle = value.get("bundle")
    graph = value.get("receipt_graph")
    bindings = value.get("lineage_bindings")
    persistence = value.get("persistence")
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(graph, Mapping)
        or not isinstance(bindings, list)
        or not isinstance(persistence, Mapping)
        or value.get("lineage_complete") is not True
    ):
        raise AllocationHandoffV2Error("allocation handoff components are invalid")
    validate_receipt_graph(
        graph,
        required_purposes={"research_lab.allocation.v2"},
    )
    root_hash = str(graph.get("root_receipt_hash") or "")
    if value.get("root_receipt_hash") != root_hash:
        raise AllocationHandoffV2Error("allocation handoff root differs from graph")
    receipts = {
        str(receipt.get("receipt_hash") or ""): receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
    }
    root = receipts.get(root_hash)
    if not isinstance(root, Mapping):
        raise AllocationHandoffV2Error("allocation handoff root receipt is missing")
    try:
        epoch_id = int(bundle["epoch"])
        netuid = int(bundle["netuid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AllocationHandoffV2Error("allocation handoff scope is invalid") from exc
    allocation = bundle.get("allocation_doc")
    if not isinstance(allocation, Mapping):
        raise AllocationHandoffV2Error("allocation handoff document is missing")
    if (
        root.get("role") != "gateway_coordinator"
        or root.get("purpose") != "research_lab.allocation.v2"
        or int(root.get("epoch_id", -1)) != epoch_id
        or root.get("output_root") != sha256_json({"allocation": dict(allocation)})
    ):
        raise AllocationHandoffV2Error(
            "allocation handoff root does not bind the allocation"
        )
    if expected_epoch_id is not None and epoch_id != int(expected_epoch_id):
        raise AllocationHandoffV2Error("allocation handoff epoch differs")
    if expected_netuid is not None and netuid != int(expected_netuid):
        raise AllocationHandoffV2Error("allocation handoff netuid differs")
    parent_hashes = sorted(str(item) for item in root["parent_receipt_hashes"])
    normalized_bindings = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "receipt_hash",
            "receipt_purpose",
            "receipt_role",
        }:
            raise AllocationHandoffV2Error(
                "allocation lineage binding fields are invalid"
            )
        receipt_hash = str(binding.get("receipt_hash") or "")
        receipt = receipts.get(receipt_hash)
        if (
            receipt_hash not in parent_hashes
            or not isinstance(receipt, Mapping)
            or receipt.get("purpose") != binding.get("receipt_purpose")
            or receipt.get("role") != binding.get("receipt_role")
        ):
            raise AllocationHandoffV2Error(
                "allocation lineage binding differs from graph"
            )
        normalized_bindings.append(dict(binding))
    if sorted(item["receipt_hash"] for item in normalized_bindings) != parent_hashes:
        raise AllocationHandoffV2Error(
            "allocation lineage bindings do not cover every direct parent"
        )
    if persistence.get("root_receipt_hash") != root_hash:
        raise AllocationHandoffV2Error("allocation handoff is not durably persisted")
    return {
        "schema_version": ALLOCATION_HANDOFF_SCHEMA_VERSION,
        "bundle": dict(bundle),
        "receipt_graph": dict(graph),
        "root_receipt_hash": root_hash,
        "lineage_bindings": normalized_bindings,
        "lineage_complete": True,
        "persistence": dict(persistence),
    }
