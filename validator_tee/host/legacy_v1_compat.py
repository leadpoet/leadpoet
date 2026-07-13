"""Explicit compatibility client for the retired V1 weight protocol.

This module exists only to bridge a validator host running current code to the
known legacy Nitro enclave and gateway while authoritative V2 is staged.  It
does not add V1 commands to the new enclave image; it can only call commands
that an already-running legacy enclave exposes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.weights import (
    bundle_weights_hash,
    normalize_to_u16_with_uids,
    u16_to_emit_floats,
)
from validator_tee.host.vsock_client import ValidatorEnclaveClient


LEGACY_V1_COMPAT_PROTOCOL = "legacy_v1_compat"
AUTHORITATIVE_V2_PROTOCOL = "authoritative_v2"
SUPPORTED_WEIGHT_PROTOCOLS = {
    AUTHORITATIVE_V2_PROTOCOL,
    LEGACY_V1_COMPAT_PROTOCOL,
}

# This is the live, independently allowlisted pre-authoritative-V2 validator
# enclave measurement. The restart wrapper refuses compatibility mode if the
# running enclave differs.
APPROVED_LEGACY_V1_PCR0 = (
    "8b56c0be4cfc55131ce299a6c7f9f2dde56d0cc2e75ffcef5558af5efc60ee3b"
    "d0c5e8b3db822659b50c175ed906a009"
)


def normalize_weight_protocol(value: Optional[str]) -> str:
    # V2 remains an explicit release action until both independent builders,
    # manifests, KMS envelopes, and production rollout evidence are ready.
    protocol = str(value or LEGACY_V1_COMPAT_PROTOCOL).strip().lower()
    if protocol not in SUPPORTED_WEIGHT_PROTOCOLS:
        allowed = ", ".join(sorted(SUPPORTED_WEIGHT_PROTOCOLS))
        raise RuntimeError(
            f"VALIDATOR_WEIGHT_PROTOCOL must be one of: {allowed}"
        )
    return protocol


class LegacyV1EnclaveClient:
    """Narrow host adapter for the legacy enclave's constrained RPCs."""

    def __init__(self, client: Optional[ValidatorEnclaveClient] = None):
        self._client = client or ValidatorEnclaveClient()
        self._public_key: Optional[str] = None
        self._code_hash: Optional[str] = None

    def health_check(self) -> Dict[str, Any]:
        return dict(self._client._send_request({"command": "health"}))

    def get_public_key(self) -> str:
        if self._public_key is None:
            response = self._client._send_request({"command": "get_public_key"})
            self._public_key = str(response["public_key"])
            self._code_hash = str(response["code_hash"])
        return self._public_key

    def get_code_hash(self) -> str:
        if self._code_hash is None:
            self.get_public_key()
        return str(self._code_hash)

    def sign_weights_hash(self, weights_hash: str) -> str:
        response = self._client._send_request(
            {"command": "sign_weights", "weights_hash": str(weights_hash)}
        )
        return str(response["signature"])

    def get_attestation(self, epoch_id: int) -> str:
        response = self._client._send_request(
            {"command": "get_attestation", "epoch_id": int(epoch_id)}
        )
        return str(response["attestation_b64"])


def build_legacy_v1_submission(
    *,
    client: LegacyV1EnclaveClient,
    netuid: int,
    epoch_id: int,
    block: int,
    uids: Sequence[int],
    weights: Sequence[float],
    validator_hotkey: str,
    sign_binding_message: Callable[[bytes], bytes],
    expected_chain: str,
    validator_version: str,
) -> Dict[str, Any]:
    """Build one V1 payload and the exact chain vector it represents."""

    if len(uids) != len(weights):
        raise ValueError("legacy V1 UID/weight length mismatch")
    positive_pairs = sorted(
        (int(uid), float(weight))
        for uid, weight in zip(uids, weights)
        if float(weight) > 0
    )
    if not positive_pairs:
        raise ValueError("legacy V1 submission has no positive weights")
    if len({uid for uid, _weight in positive_pairs}) != len(positive_pairs):
        raise ValueError("legacy V1 submission contains duplicate UIDs")

    sparse_uids, sparse_weights_u16 = normalize_to_u16_with_uids(
        [uid for uid, _weight in positive_pairs],
        [weight for _uid, weight in positive_pairs],
    )
    sparse_uids = [int(uid) for uid in sparse_uids]
    sparse_weights_u16 = [int(weight) for weight in sparse_weights_u16]
    if not sparse_uids or len(sparse_uids) != len(sparse_weights_u16):
        raise ValueError("legacy V1 normalization produced an invalid vector")

    weights_hash = bundle_weights_hash(
        int(netuid),
        int(epoch_id),
        int(block),
        list(zip(sparse_uids, sparse_weights_u16)),
    )
    validator_signature = client.sign_weights_hash(weights_hash)
    enclave_pubkey = client.get_public_key()
    code_hash = client.get_code_hash()
    binding_message = create_binding_message(
        netuid=int(netuid),
        chain=str(expected_chain),
        enclave_pubkey=enclave_pubkey,
        validator_code_hash=code_hash,
        version=str(validator_version),
    )
    hotkey_signature = sign_binding_message(binding_message.encode("utf-8"))
    if not isinstance(hotkey_signature, bytes):
        raise TypeError("validator hotkey signature must be bytes")

    payload = {
        "netuid": int(netuid),
        "epoch_id": int(epoch_id),
        "block": int(block),
        "uids": list(sparse_uids),
        "weights_u16": list(sparse_weights_u16),
        "weights_hash": weights_hash,
        "validator_hotkey": str(validator_hotkey),
        "validator_enclave_pubkey": enclave_pubkey,
        "validator_signature": validator_signature,
        "validator_attestation_b64": client.get_attestation(int(epoch_id)),
        "validator_code_hash": code_hash,
        "binding_message": binding_message,
        "validator_hotkey_signature": hotkey_signature.hex(),
    }
    return {
        "payload": payload,
        "uids": list(sparse_uids),
        "weights_u16": list(sparse_weights_u16),
        "chain_weights": u16_to_emit_floats(
            list(sparse_uids), list(sparse_weights_u16)
        ),
    }


def verify_existing_legacy_v1_bundle(
    existing: Dict[str, Any],
    expected: Dict[str, Any],
) -> str:
    """Accept duplicate recovery only when the stored signed bundle is exact."""

    compared_fields: List[str] = [
        "netuid",
        "epoch_id",
        "block",
        "uids",
        "weights_u16",
        "weights_hash",
        "validator_hotkey",
        "validator_enclave_pubkey",
        "validator_signature",
        "validator_code_hash",
    ]
    mismatched = [
        field for field in compared_fields if existing.get(field) != expected.get(field)
    ]
    if mismatched:
        raise RuntimeError(
            "legacy V1 duplicate differs from the prepared bundle: "
            + ", ".join(mismatched)
        )
    event_hash = str(existing.get("weight_submission_event_hash") or "")
    if not event_hash:
        raise RuntimeError("legacy V1 duplicate is missing its event hash")
    return event_hash
