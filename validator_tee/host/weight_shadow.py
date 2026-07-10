"""Host verification for validator-enclave weight computation.

This module never computes an alternate production vector. It independently
checks the enclave response against the shared canonical core and the exact
host vector that the unchanged validator path is about to submit.
"""

import asyncio
import struct
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leadpoet_canonical.attested_receipts import (
    WEIGHT_PURPOSE,
    WEIGHT_ROLE,
    validate_signed_receipt,
)
from leadpoet_canonical.weight_computation import (
    compute_final_weights,
    normalize_to_u16_with_uids_pure,
    sha256_json,
)
from leadpoet_canonical.weight_bundle_v2 import (
    WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    validate_weight_bundle_v2,
)


class AttestedWeightVerificationError(ValueError):
    """Raised when an enclave result is not exactly the requested weight result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestedWeightVerificationError(message)


def _float_bits(values: Sequence[float]) -> List[str]:
    return [struct.pack("!d", float(value)).hex() for value in values]


def verify_enclave_weight_response(
    *,
    snapshot: Mapping[str, Any],
    response: Mapping[str, Any],
    host_uids: Sequence[int],
    host_weights: Sequence[float],
    require_real_attestation: bool,
) -> Dict[str, Any]:
    """Validate computation, receipt, signatures, and exact host equivalence."""

    _require(isinstance(response, Mapping), "enclave weight response is not an object")
    result = response.get("weight_result")
    receipt = response.get("receipt")
    user_data = response.get("attestation_user_data")
    _require(isinstance(result, Mapping), "enclave weight result is missing")
    _require(isinstance(receipt, Mapping), "enclave weight receipt is missing")
    _require(isinstance(user_data, Mapping), "enclave attestation user_data is missing")

    expected_result = compute_final_weights(snapshot)
    _require(dict(result) == expected_result, "enclave result differs from canonical recomputation")

    validate_signed_receipt(receipt)
    _require(receipt.get("role") == WEIGHT_ROLE, "weight receipt role mismatch")
    _require(receipt.get("purpose") == WEIGHT_PURPOSE, "weight receipt purpose mismatch")
    _require(receipt.get("status") == "succeeded", "weight receipt is not successful")
    _require(int(receipt.get("epoch_id")) == int(snapshot["epoch_id"]), "weight receipt epoch mismatch")
    _require(receipt.get("commit_sha") == snapshot["commit_sha"], "weight receipt commit mismatch")
    _require(receipt.get("config_hash") == snapshot["config_hash"], "weight receipt config mismatch")
    _require(receipt.get("input_root") == expected_result["snapshot_hash"], "weight receipt input mismatch")
    _require(receipt.get("output_root") == sha256_json(expected_result), "weight receipt output mismatch")
    _require(
        receipt.get("parent_receipt_hashes") == sorted(set(snapshot["parent_receipt_hashes"])),
        "weight receipt ancestry mismatch",
    )

    allocation_receipt_hash = str(snapshot.get("research_lab_allocation_receipt_hash") or "")
    expected_evidence = (
        {"research_lab_allocation_receipt": allocation_receipt_hash}
        if allocation_receipt_hash
        else {}
    )
    _require(receipt.get("evidence_roots") == expected_evidence, "weight receipt evidence mismatch")

    enclave_pubkey = str(receipt.get("enclave_pubkey") or "")
    weights_signature = str(response.get("weights_signature") or "")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(enclave_pubkey)).verify(
            bytes.fromhex(weights_signature),
            bytes.fromhex(str(result["weights_hash"])),
        )
    except Exception as exc:
        raise AttestedWeightVerificationError("invalid enclave weight signature") from exc

    _require(user_data.get("purpose") == WEIGHT_PURPOSE, "attestation purpose mismatch")
    _require(int(user_data.get("epoch_id")) == int(snapshot["epoch_id"]), "attestation epoch mismatch")
    _require(user_data.get("enclave_pubkey") == enclave_pubkey, "attestation public key mismatch")
    if require_real_attestation:
        _require(not bool(response.get("attestation_is_mock")), "mock attestation is forbidden")

    normalized_host_uids = [int(uid) for uid in host_uids]
    normalized_host_weights = [float(weight) for weight in host_weights]
    _require(normalized_host_uids == result["uids"], "host and enclave UID order differ")
    _require(
        _float_bits(normalized_host_weights) == result["weight_float_bits"],
        "host and enclave float weights differ",
    )
    host_pairs = sorted(
        ((uid, weight) for uid, weight in zip(normalized_host_uids, normalized_host_weights) if weight > 0),
        key=lambda pair: pair[0],
    )
    host_sparse_uids, host_sparse_u16 = normalize_to_u16_with_uids_pure(
        [pair[0] for pair in host_pairs],
        [pair[1] for pair in host_pairs],
    )
    _require(host_sparse_uids == result["sparse_uids"], "host and enclave sparse UIDs differ")
    _require(host_sparse_u16 == result["sparse_weights_u16"], "host and enclave u16 weights differ")
    return dict(result)


async def execute_attested_weight_mode(
    *,
    mode: str,
    snapshot: Mapping[str, Any],
    host_uids: Sequence[int],
    host_weights: Sequence[float],
    compute_weights: Callable[[Dict[str, Any]], Mapping[str, Any]],
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Run `off`, `shadow`, or fail-closed `required` mode."""

    if mode == "off":
        return None
    if mode not in {"shadow", "required"}:
        raise AttestedWeightVerificationError("unknown attested weight mode")
    try:
        response = await asyncio.to_thread(compute_weights, dict(snapshot))
        verify_enclave_weight_response(
            snapshot=snapshot,
            response=response,
            host_uids=host_uids,
            host_weights=host_weights,
            require_real_attestation=(mode == "required"),
        )
        return dict(response)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        if mode == "required":
            raise AttestedWeightVerificationError(
                "required validator enclave weight computation failed"
            ) from exc
        return None


def build_weight_bundle_v2(
    *,
    snapshot: Mapping[str, Any],
    enclave_response: Mapping[str, Any],
    validator_hotkey: str,
    binding_message: str,
    validator_hotkey_signature: str,
    parent_receipts: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Construct and self-verify the exact v2 gateway payload."""

    bundle = {
        "schema_version": WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
        "validator_hotkey": str(validator_hotkey),
        "binding_message": str(binding_message),
        "validator_hotkey_signature": str(validator_hotkey_signature),
        "weight_snapshot": dict(snapshot),
        "weight_result": dict(enclave_response["weight_result"]),
        "weights_signature": str(enclave_response["weights_signature"]),
        "weight_receipt": dict(enclave_response["receipt"]),
        "parent_receipts": [dict(receipt) for receipt in parent_receipts],
    }
    validate_weight_bundle_v2(bundle, require_allocation_ancestry=False)
    return bundle
