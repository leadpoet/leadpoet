"""Measured coordinator authority for the one stateful subnet-epoch cutover.

The host is allowed to select a proposed mapping, a finalized boundary
snapshot, and the last legacy finalization.  This module does not trust those
selections: it recomputes the mapping and event commitments, validates both
complete receipt graphs, and requires the two graph roots to be the only
parents of the coordinator operation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    WEIGHT_ROLE,
    sha256_json,
    validate_receipt_graph,
)
from leadpoet_canonical.weight_authority_v2 import (
    WEIGHT_FINALIZATION_SUBMISSION_V2_SCHEMA_VERSION,
    validate_weight_finalization_submission_v2,
)


OP_ATTEST_SUBNET_EPOCH_CUTOVER_V2 = "attest_subnet_epoch_cutover_v2"
CUTOVER_REQUEST_SCHEMA_VERSION = "leadpoet.subnet_epoch_cutover_request.v2"
CUTOVER_AUTHORITY_SCHEMA_VERSION = (
    "leadpoet.subnet_epoch_cutover_authority.v1"
)
CUTOVER_PURPOSE = "research_lab.subnet_epoch_cutover.v2"
SNAPSHOT_PURPOSE = "validator.subnet_epoch_snapshot.v2"
FINALIZATION_PURPOSE = "validator.weights.finalized.v2"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest",
        "first_snapshot",
        "last_legacy_bundle_hash",
        "last_legacy_finalization",
    }
)


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise ValueError("%s is invalid" % field)
    return normalized


def _root_receipt(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_receipt_graph(graph)
    root_hash = str(graph.get("root_receipt_hash") or "")
    matches = [
        receipt
        for receipt in graph.get("receipts") or ()
        if isinstance(receipt, Mapping)
        and receipt.get("receipt_hash") == root_hash
    ]
    if len(matches) != 1:
        raise ValueError("cutover parent graph root is unavailable")
    return matches[0]


def _classify_parent_graphs(
    context: ExecutionContextV2,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    graphs = list(context.external_receipt_graphs)
    if len(graphs) != 2:
        raise ValueError("subnet epoch cutover requires exactly two parent graphs")
    roots = [_root_receipt(graph) for graph in graphs]
    root_hashes = sorted(str(root["receipt_hash"]) for root in roots)
    if root_hashes != sorted(str(value) for value in context.parent_receipt_hashes):
        raise ValueError("subnet epoch cutover parent roots differ")
    if len(context.parent_receipt_hashes) != 2 or len(set(root_hashes)) != 2:
        raise ValueError("subnet epoch cutover parents are not exact")

    by_purpose = {}
    for graph, root in zip(graphs, roots):
        purpose = str(root.get("purpose") or "")
        if purpose in by_purpose:
            raise ValueError("subnet epoch cutover parent purpose is duplicated")
        by_purpose[purpose] = (graph, root)
    if set(by_purpose) != {SNAPSHOT_PURPOSE, FINALIZATION_PURPOSE}:
        raise ValueError("subnet epoch cutover parent purposes are invalid")
    return (
        by_purpose[SNAPSHOT_PURPOSE][0],
        by_purpose[SNAPSHOT_PURPOSE][1],
        by_purpose[FINALIZATION_PURPOSE][0],
        by_purpose[FINALIZATION_PURPOSE][1],
    )


def _validate_first_snapshot(
    value: Any,
    *,
    cutover: SubnetEpochCutover,
) -> Tuple[SubnetEpochSnapshot, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("first subnet epoch snapshot is invalid")
    supplied = dict(value)
    try:
        snapshot = SubnetEpochSnapshot.from_mapping(supplied)
        expected = snapshot.to_dict(cutover=cutover)
    except (KeyError, TypeError, SubnetEpochError) as exc:
        raise ValueError("first subnet epoch snapshot is invalid") from exc
    if supplied != expected:
        raise ValueError("first subnet epoch snapshot fields or derivations differ")
    if (
        snapshot.head_kind != "finalized"
        or snapshot.network_genesis_hash != cutover.network_genesis_hash
        or snapshot.netuid != cutover.netuid
        or snapshot.current_block != cutover.cutover_block
        or snapshot.last_epoch_block != cutover.cutover_block
        or snapshot.block_hash != cutover.cutover_block_hash
        or snapshot.subnet_epoch_index != cutover.first_subnet_epoch_index
        or snapshot.settlement_epoch_id(cutover)
        != cutover.first_settlement_epoch_id
    ):
        raise ValueError("first subnet epoch snapshot differs from cutover")
    return snapshot, expected


def attest_subnet_epoch_cutover_v2(
    payload: Mapping[str, Any],
    context: ExecutionContextV2,
) -> Dict[str, Any]:
    """Return one canonical authority document for an exact two-parent cutover."""

    if not isinstance(payload, Mapping) or set(payload) != _REQUEST_FIELDS:
        raise ValueError("subnet epoch cutover request fields are invalid")
    if payload.get("schema_version") != CUTOVER_REQUEST_SCHEMA_VERSION:
        raise ValueError("subnet epoch cutover request schema is invalid")
    if context.purpose != CUTOVER_PURPOSE:
        raise ValueError("subnet epoch cutover purpose is invalid")

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("subnet epoch cutover manifest is invalid")
    try:
        cutover = SubnetEpochCutover.from_mapping(manifest)
    except (TypeError, SubnetEpochError) as exc:
        raise ValueError("subnet epoch cutover manifest is invalid") from exc
    if dict(manifest) != cutover.to_dict():
        raise ValueError("subnet epoch cutover manifest is not canonical")
    if int(context.epoch_id) != cutover.first_settlement_epoch_id:
        raise ValueError("subnet epoch cutover execution epoch differs")

    _, snapshot_doc = _validate_first_snapshot(
        payload.get("first_snapshot"),
        cutover=cutover,
    )
    first_snapshot_hash = sha256_json(snapshot_doc)
    (
        _snapshot_graph,
        snapshot_root,
        finalization_graph,
        finalization_root,
    ) = _classify_parent_graphs(context)
    if (
        snapshot_root.get("role") != WEIGHT_ROLE
        or int(snapshot_root.get("epoch_id", -1))
        != cutover.first_settlement_epoch_id
        or snapshot_root.get("output_root") != first_snapshot_hash
    ):
        raise ValueError("first subnet epoch snapshot receipt is invalid")

    finalization = payload.get("last_legacy_finalization")
    if not isinstance(finalization, Mapping):
        raise ValueError("last legacy finalization is invalid")
    try:
        verified_finalization = validate_weight_finalization_submission_v2(
            {
                "schema_version": WEIGHT_FINALIZATION_SUBMISSION_V2_SCHEMA_VERSION,
                "validator_hotkey": finalization.get("validator_hotkey"),
                "weight_submission_event_hash": finalization.get(
                    "weight_submission_event_hash"
                ),
                "finalization": dict(finalization),
                "receipt_graph": dict(finalization_graph),
            }
        )
    except Exception as exc:
        raise ValueError("last legacy finalization proof is invalid") from exc
    if (
        finalization_root.get("role") != WEIGHT_ROLE
        or finalization_root.get("receipt_hash")
        != verified_finalization["finalization_receipt_hash"]
        or int(verified_finalization["netuid"]) != cutover.netuid
        or int(verified_finalization["epoch_id"]) != cutover.last_legacy_epoch_id
        or int(verified_finalization["finalized_block"]) >= cutover.cutover_block
    ):
        raise ValueError("last legacy finalization differs from cutover")

    bundle_hash = _hash(
        payload.get("last_legacy_bundle_hash"),
        "last legacy bundle hash",
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": verified_finalization[
                "weight_submission_event_hash"
            ],
            "bundle_hash": bundle_hash,
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": verified_finalization[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": verified_finalization["extrinsic_hash"],
            "finalized_block": verified_finalization["finalized_block"],
            "finalized_block_hash": verified_finalization[
                "finalized_block_hash"
            ],
            "state_transition_hash": verified_finalization[
                "state_transition_hash"
            ],
        }
    )
    return {
        "schema_version": CUTOVER_AUTHORITY_SCHEMA_VERSION,
        "mapping_hash": str(cutover.mapping_hash),
        "first_epoch_ref": snapshot_doc["epoch_ref"],
        "first_snapshot_hash": first_snapshot_hash,
        "first_snapshot_receipt_hash": snapshot_root["receipt_hash"],
        "last_legacy_bundle_hash": bundle_hash,
        "last_legacy_weight_finalization_event_hash": finalization_event_hash,
        "last_legacy_finalization_receipt_hash": finalization_root[
            "receipt_hash"
        ],
        "manifest": cutover.to_dict(),
    }
