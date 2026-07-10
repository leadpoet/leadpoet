"""Canonical verification for enclave-computed final-weight bundles.

The verifier is Python 3.7 compatible and contains no gateway or chain I/O. PCR0
and hotkey authorization remain caller responsibilities because each verifier
must establish those from its own trusted runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leadpoet_canonical.attested_receipts import (
    SCORING_ROLE,
    SCORING_PURPOSES,
    WEIGHT_PURPOSE,
    WEIGHT_ROLE,
    validate_signed_receipt,
    verify_receipt_lineage,
)
from leadpoet_canonical.weight_computation import compute_final_weights, sha256_json


WEIGHT_BUNDLE_V2_SCHEMA_VERSION = "leadpoet.published_weight_bundle.v2"
ALLOCATION_PURPOSE = "research_lab.allocation.v1"

_FIELDS = {
    "schema_version",
    "validator_hotkey",
    "binding_message",
    "validator_hotkey_signature",
    "weight_snapshot",
    "weight_result",
    "weights_signature",
    "weight_receipt",
    "parent_receipts",
}


class WeightBundleV2Error(ValueError):
    """Raised when a v2 bundle is incomplete, inconsistent, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WeightBundleV2Error(message)


def validate_weight_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    require_allocation_ancestry: bool,
) -> Dict[str, Any]:
    """Validate the complete computation and signed receipt graph."""

    _require(isinstance(bundle, Mapping) and set(bundle) == _FIELDS, "v2 bundle fields do not match schema")
    _require(bundle.get("schema_version") == WEIGHT_BUNDLE_V2_SCHEMA_VERSION, "unsupported v2 bundle schema")
    snapshot = bundle.get("weight_snapshot")
    claimed_result = bundle.get("weight_result")
    receipt = bundle.get("weight_receipt")
    parent_receipts_value = bundle.get("parent_receipts")
    _require(isinstance(snapshot, Mapping), "weight_snapshot is missing")
    _require(isinstance(claimed_result, Mapping), "weight_result is missing")
    _require(isinstance(receipt, Mapping), "weight_receipt is missing")
    _require(isinstance(parent_receipts_value, list), "parent_receipts must be a list")

    computed_result = compute_final_weights(snapshot)
    _require(dict(claimed_result) == computed_result, "weight_result differs from canonical snapshot computation")

    validate_signed_receipt(receipt)
    _require(receipt.get("role") == WEIGHT_ROLE, "weight receipt role mismatch")
    _require(receipt.get("purpose") == WEIGHT_PURPOSE, "weight receipt purpose mismatch")
    _require(receipt.get("status") == "succeeded", "weight receipt is not successful")
    _require(receipt.get("epoch_id") == snapshot.get("epoch_id"), "weight receipt epoch mismatch")
    _require(receipt.get("commit_sha") == snapshot.get("commit_sha"), "weight receipt commit mismatch")
    _require(receipt.get("config_hash") == snapshot.get("config_hash"), "weight receipt config mismatch")
    _require(receipt.get("input_root") == computed_result["snapshot_hash"], "weight receipt input mismatch")
    _require(receipt.get("output_root") == sha256_json(computed_result), "weight receipt output mismatch")

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(receipt["enclave_pubkey"]))).verify(
            bytes.fromhex(str(bundle.get("weights_signature") or "")),
            bytes.fromhex(computed_result["weights_hash"]),
        )
    except Exception as exc:
        raise WeightBundleV2Error("invalid enclave-computed weight signature") from exc

    parent_receipts = {}  # type: Dict[str, Mapping[str, Any]]
    for parent in parent_receipts_value:
        _require(isinstance(parent, Mapping), "parent receipt is not an object")
        validate_signed_receipt(parent)
        parent_hash = str(parent.get("receipt_hash") or "")
        _require(parent_hash not in parent_receipts, "parent receipt is duplicated")
        parent_receipts[parent_hash] = parent
    ordered_lineage = verify_receipt_lineage(receipt, parent_receipts)
    _require(
        set(ordered_lineage) == set(parent_receipts) | {str(receipt["receipt_hash"])},
        "v2 bundle contains disconnected parent receipts",
    )
    expected_direct_parents = sorted(set(snapshot.get("parent_receipt_hashes") or []))
    _require(
        receipt.get("parent_receipt_hashes") == expected_direct_parents,
        "weight receipt parents differ from immutable snapshot",
    )
    for parent in parent_receipts.values():
        _require(parent.get("role") == SCORING_ROLE, "weight ancestry receipt role mismatch")
        _require(
            parent.get("purpose") in SCORING_PURPOSES,
            "weight ancestry receipt purpose mismatch",
        )
        _require(parent.get("status") == "succeeded", "weight ancestry receipt is not successful")
        _require(
            int(parent.get("epoch_id")) <= int(snapshot.get("epoch_id")),
            "weight ancestry receipt epoch is newer than weight epoch",
        )

    allocation_hash = str(snapshot.get("research_lab_allocation_receipt_hash") or "")
    if require_allocation_ancestry:
        _require(bool(allocation_hash), "Research Lab allocation receipt is required")
    if allocation_hash:
        _require(
            allocation_hash in expected_direct_parents,
            "Research Lab allocation receipt is not a direct weight parent",
        )
        _require(allocation_hash in parent_receipts, "Research Lab allocation receipt is missing")
        allocation_receipt = parent_receipts[allocation_hash]
        _require(allocation_receipt.get("role") == SCORING_ROLE, "allocation receipt role mismatch")
        _require(allocation_receipt.get("purpose") == ALLOCATION_PURPOSE, "allocation receipt purpose mismatch")
        _require(allocation_receipt.get("status") == "succeeded", "allocation receipt is not successful")
        _require(
            int(allocation_receipt.get("epoch_id")) <= int(snapshot.get("epoch_id")),
            "allocation receipt epoch is newer than weight epoch",
        )
        _require(
            receipt.get("evidence_roots", {}).get("research_lab_allocation_receipt") == allocation_hash,
            "weight receipt does not commit its allocation parent",
        )
    else:
        _require(
            "research_lab_allocation_receipt" not in receipt.get("evidence_roots", {}),
            "weight receipt contains an undeclared allocation parent",
        )

    return {
        "netuid": computed_result["netuid"],
        "epoch_id": computed_result["epoch_id"],
        "block": computed_result["block"],
        "uids": list(computed_result["sparse_uids"]),
        "weights_u16": list(computed_result["sparse_weights_u16"]),
        "weights_hash": computed_result["weights_hash"],
        "validator_enclave_pubkey": receipt["enclave_pubkey"],
        "validator_attestation_b64": receipt["attestation_document_b64"],
        "weight_receipt_hash": receipt["receipt_hash"],
        "weight_snapshot_hash": computed_result["snapshot_hash"],
        "validator_hotkey": str(bundle.get("validator_hotkey") or ""),
        "binding_message": str(bundle.get("binding_message") or ""),
        "validator_hotkey_signature": str(bundle.get("validator_hotkey_signature") or ""),
    }
