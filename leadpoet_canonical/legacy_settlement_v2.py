"""Canonical verification for pre-V2 champion settlement migration.

Historical allocation rows describe intended payments but are not settlement
authority.  A migration is accepted only when the allocation is bound to a
signed legacy validator bundle, the bundle matches finalized epoch-end chain
state, and the same allocation/weight commitment appears in a signed audit
event included in its immutable Arweave checkpoint.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.chain_source_v2 import validate_arweave_checkpoint_event
from leadpoet_canonical.events import verify_log_entry
from leadpoet_canonical.weights import (
    AUDITOR_WEIGHT_TOLERANCE,
    bundle_weights_hash,
    compare_weights_hash,
    weights_within_tolerance,
)


LEGACY_SETTLEMENT_SCHEMA_VERSION = "leadpoet.legacy_finalized_allocation.v2"
LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION = (
    "leadpoet.legacy_finalized_allocation_request.v2"
)
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")


class LegacySettlementV2Error(ValueError):
    """Historical settlement evidence is incomplete, malformed, or divergent."""


_SETTLEMENT_FIELDS = {
    "schema_version",
    "netuid",
    "epoch_id",
    "allocation_hash",
    "allocation_doc",
    "validator_hotkey",
    "legacy_bundle_weights_hash",
    "legacy_bundle_block",
    "chain_compare_hash",
    "chain_vector_tolerance_u16",
    "chain_target_block",
    "chain_target_block_hash",
    "chain_finalized_head_block",
    "validator_uid",
    "weights_storage_key_hash",
    "audit_event_hash",
    "audit_payload_hash",
    "checkpoint_merkle_root",
    "checkpoint_number",
    "checkpoint_event_sequence",
    "arweave_tx_id",
    "settlement_hash",
}


def _raw_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]
    if not _RAW_HASH_RE.fullmatch(normalized):
        raise LegacySettlementV2Error("%s is invalid" % field)
    return normalized


def _sha256_ref(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if _RAW_HASH_RE.fullmatch(normalized):
        normalized = "sha256:" + normalized
    if not _SHA256_RE.fullmatch(normalized):
        raise LegacySettlementV2Error("%s is invalid" % field)
    return normalized


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise LegacySettlementV2Error("%s is invalid" % field)
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise LegacySettlementV2Error("%s is invalid" % field) from exc
    if normalized < (1 if positive else 0):
        raise LegacySettlementV2Error("%s is invalid" % field)
    return normalized


def validate_legacy_weight_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    expected_netuid: int,
    expected_epoch_id: int,
) -> Dict[str, Any]:
    """Validate the exact sparse V1 vector and its enclave signature."""

    if not isinstance(bundle, Mapping):
        raise LegacySettlementV2Error("legacy weight bundle is missing")
    netuid = _integer(bundle.get("netuid"), "bundle netuid", positive=True)
    epoch_id = _integer(bundle.get("epoch_id"), "bundle epoch")
    block = _integer(bundle.get("block"), "bundle block")
    if netuid != int(expected_netuid) or epoch_id != int(expected_epoch_id):
        raise LegacySettlementV2Error("legacy weight bundle scope differs")
    if block // 360 != epoch_id:
        raise LegacySettlementV2Error("legacy weight bundle block differs from epoch")

    uids = bundle.get("uids")
    weights = bundle.get("weights_u16")
    if (
        not isinstance(uids, list)
        or not isinstance(weights, list)
        or not uids
        or len(uids) != len(weights)
        or uids != sorted(uids)
        or len(uids) != len(set(uids))
    ):
        raise LegacySettlementV2Error("legacy weight vector is invalid")
    pairs: list[Tuple[int, int]] = []
    for uid, weight in zip(uids, weights):
        normalized_uid = _integer(uid, "bundle UID")
        normalized_weight = _integer(weight, "bundle weight", positive=True)
        if normalized_uid > 0xFFFF or normalized_weight > 0xFFFF:
            raise LegacySettlementV2Error("legacy weight vector is out of range")
        pairs.append((normalized_uid, normalized_weight))

    weights_hash = _raw_hash(bundle.get("weights_hash"), "bundle weights hash")
    if bundle_weights_hash(netuid, epoch_id, block, pairs) != weights_hash:
        raise LegacySettlementV2Error("legacy weight bundle hash differs")
    public_key = _raw_hash(
        bundle.get("validator_enclave_pubkey"), "validator enclave public key"
    )
    signature = str(bundle.get("validator_signature") or "").strip().lower()
    if not _SIGNATURE_RE.fullmatch(signature):
        raise LegacySettlementV2Error("validator enclave signature is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            bytes.fromhex(signature), bytes.fromhex(weights_hash)
        )
    except Exception as exc:
        raise LegacySettlementV2Error(
            "validator enclave signature verification failed"
        ) from exc
    validator_hotkey = str(bundle.get("validator_hotkey") or "").strip()
    if not validator_hotkey:
        raise LegacySettlementV2Error("validator hotkey is missing")
    return {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "block": block,
        "validator_hotkey": validator_hotkey,
        "validator_enclave_pubkey": public_key,
        "weights_hash": weights_hash,
        "weights": [[uid, weight] for uid, weight in pairs],
    }


def validate_legacy_audit_event_v2(
    *,
    log_row: Mapping[str, Any],
    anchor: Mapping[str, Any],
    expected_netuid: int,
    expected_epoch_id: int,
    expected_allocation_hash: str,
    expected_weights_hash: str,
    expected_validator_hotkey: str,
) -> Dict[str, Any]:
    """Validate the signed compact audit event and all durable anchor fields."""

    signed_log_entry = log_row.get("signed_log_entry")
    if not isinstance(signed_log_entry, Mapping):
        raise LegacySettlementV2Error("signed audit log entry is missing")
    expected_pubkey = _raw_hash(
        log_row.get("enclave_pubkey")
        or signed_log_entry.get("enclave_pubkey"),
        "audit enclave public key",
    )
    if not verify_log_entry(dict(signed_log_entry), expected_pubkey=expected_pubkey):
        raise LegacySettlementV2Error("signed audit log entry verification failed")
    event_hash = _raw_hash(signed_log_entry.get("event_hash"), "audit event hash")
    if _raw_hash(log_row.get("event_hash"), "stored audit event hash") != event_hash:
        raise LegacySettlementV2Error("stored audit event hash differs")
    anchor_event_hash = anchor.get("current_transparency_event_hash") or anchor.get(
        "transparency_event_hash"
    )
    if _raw_hash(anchor_event_hash, "anchor audit event hash") != event_hash:
        raise LegacySettlementV2Error("audit anchor event hash differs")

    signed_event = signed_log_entry.get("signed_event")
    payload = signed_event.get("payload") if isinstance(signed_event, Mapping) else None
    if (
        not isinstance(signed_event, Mapping)
        or signed_event.get("event_type") != "RESEARCH_LAB_EPOCH_AUDIT"
        or not isinstance(payload, Mapping)
        or payload.get("event_type") != "RESEARCH_LAB_EPOCH_AUDIT"
    ):
        raise LegacySettlementV2Error("signed audit event type is invalid")
    if (
        _integer(payload.get("epoch"), "audit epoch") != int(expected_epoch_id)
        or _integer(payload.get("netuid"), "audit netuid", positive=True)
        != int(expected_netuid)
        or payload.get("audit_kind") != "active"
        or anchor.get("audit_kind") != "active"
    ):
        raise LegacySettlementV2Error("signed audit event scope differs")
    if str(payload.get("actor_hotkey") or "") != str(expected_validator_hotkey):
        raise LegacySettlementV2Error("signed audit validator differs")

    allocation_ref = payload.get("lab_allocation")
    weights_ref = payload.get("weights")
    if not isinstance(allocation_ref, Mapping) or not isinstance(weights_ref, Mapping):
        raise LegacySettlementV2Error("signed audit commitments are missing")
    allocation_hash = _sha256_ref(
        allocation_ref.get("allocation_hash"), "audit allocation hash"
    )
    weights_hash = _raw_hash(weights_ref.get("weights_hash"), "audit weights hash")
    if allocation_hash != _sha256_ref(
        expected_allocation_hash, "expected allocation hash"
    ) or weights_hash != _raw_hash(expected_weights_hash, "expected weights hash"):
        raise LegacySettlementV2Error("signed audit commitments differ")
    if _sha256_ref(anchor.get("allocation_hash"), "anchor allocation hash") != allocation_hash:
        raise LegacySettlementV2Error("audit anchor allocation differs")
    if _raw_hash(anchor.get("weights_hash"), "anchor weights hash") != weights_hash:
        raise LegacySettlementV2Error("audit anchor weights differ")

    payload_document = dict(payload)
    internal_payload_hash = _sha256_ref(
        payload_document.pop("payload_hash", ""), "audit payload self-hash"
    )
    if sha256_json(payload_document) != internal_payload_hash:
        raise LegacySettlementV2Error("audit payload self-hash differs")
    outer_payload_hash = sha256_json(dict(payload))
    if _sha256_ref(anchor.get("payload_hash"), "anchor payload hash") != outer_payload_hash:
        raise LegacySettlementV2Error("audit anchor payload hash differs")
    stored_payload_hash = log_row.get("payload_hash")
    if stored_payload_hash and _sha256_ref(
        stored_payload_hash, "stored payload hash"
    ) != outer_payload_hash:
        raise LegacySettlementV2Error("stored audit payload hash differs")
    return {
        "event_hash": event_hash,
        "signed_log_entry": dict(signed_log_entry),
        "event_sequence": _integer(
            anchor.get("current_tee_sequence"), "checkpoint event sequence"
        ),
        "payload_hash": outer_payload_hash,
    }


def validate_legacy_finalized_settlement_v2(
    *,
    netuid: int,
    epoch_id: int,
    allocation_doc: Mapping[str, Any],
    weight_bundle: Mapping[str, Any],
    audit_anchor: Mapping[str, Any],
    transparency_log_row: Mapping[str, Any],
    chain_evidence: Mapping[str, Any],
    arweave_checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Produce one content-addressed V2 settlement migration document."""

    normalized_netuid = _integer(netuid, "netuid", positive=True)
    normalized_epoch = _integer(epoch_id, "epoch_id")
    allocation = dict(allocation_doc)
    allocation_hash = _sha256_ref(
        allocation.get("allocation_hash"), "allocation hash"
    )
    if sha256_json(
        {key: value for key, value in allocation.items() if key != "allocation_hash"}
    ) != allocation_hash:
        raise LegacySettlementV2Error("allocation document hash differs")
    if (
        _integer(allocation.get("epoch"), "allocation epoch") != normalized_epoch
        or _integer(allocation.get("netuid"), "allocation netuid", positive=True)
        != normalized_netuid
    ):
        raise LegacySettlementV2Error("allocation document scope differs")

    bundle = validate_legacy_weight_bundle_v2(
        weight_bundle,
        expected_netuid=normalized_netuid,
        expected_epoch_id=normalized_epoch,
    )
    audit = validate_legacy_audit_event_v2(
        log_row=transparency_log_row,
        anchor=audit_anchor,
        expected_netuid=normalized_netuid,
        expected_epoch_id=normalized_epoch,
        expected_allocation_hash=allocation_hash,
        expected_weights_hash=bundle["weights_hash"],
        expected_validator_hotkey=bundle["validator_hotkey"],
    )

    if not isinstance(chain_evidence, Mapping):
        raise LegacySettlementV2Error("historical chain evidence is missing")
    if (
        _integer(chain_evidence.get("epoch_id"), "chain epoch") != normalized_epoch
        or _integer(chain_evidence.get("netuid"), "chain netuid", positive=True)
        != normalized_netuid
        or str(chain_evidence.get("validator_hotkey") or "")
        != bundle["validator_hotkey"]
    ):
        raise LegacySettlementV2Error("historical chain evidence scope differs")
    chain_weights_value = chain_evidence.get("weights")
    if not isinstance(chain_weights_value, list):
        raise LegacySettlementV2Error("historical chain vector is missing")
    try:
        chain_weights = [
            (int(item[0]), int(item[1]))
            for item in chain_weights_value
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
    except (TypeError, ValueError) as exc:
        raise LegacySettlementV2Error("historical chain vector is invalid") from exc
    if len(chain_weights) != len(chain_weights_value) or not weights_within_tolerance(
        [(int(uid), int(weight)) for uid, weight in bundle["weights"]],
        chain_weights,
        tolerance=AUDITOR_WEIGHT_TOLERANCE,
    ):
        raise LegacySettlementV2Error("historical chain vector differs from bundle")

    target_block = (normalized_epoch + 1) * 360 - 1
    if (
        _integer(chain_evidence.get("target_block"), "settlement block")
        != target_block
        or _integer(
            chain_evidence.get("finalized_head_block"), "finalized head block"
        )
        < target_block
    ):
        raise LegacySettlementV2Error("historical settlement is not finalized")
    checkpoint_root = _raw_hash(
        audit_anchor.get("current_checkpoint_merkle_root"),
        "anchor checkpoint root",
    )
    checkpoint = validate_arweave_checkpoint_event(
        arweave_checkpoint,
        expected_event_hash=audit["event_hash"],
        expected_signed_log_entry=audit["signed_log_entry"],
        expected_sequence=audit["event_sequence"],
        expected_merkle_root=checkpoint_root,
    )
    arweave_tx_id = str(audit_anchor.get("current_arweave_tx_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", arweave_tx_id):
        raise LegacySettlementV2Error("Arweave transaction id is invalid")

    chain_compare_hash = compare_weights_hash(
        normalized_netuid, normalized_epoch, chain_weights
    )
    body = {
        "schema_version": LEGACY_SETTLEMENT_SCHEMA_VERSION,
        "netuid": normalized_netuid,
        "epoch_id": normalized_epoch,
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
        "validator_hotkey": bundle["validator_hotkey"],
        "legacy_bundle_weights_hash": bundle["weights_hash"],
        "legacy_bundle_block": bundle["block"],
        "chain_compare_hash": "sha256:" + chain_compare_hash,
        "chain_vector_tolerance_u16": int(AUDITOR_WEIGHT_TOLERANCE),
        "chain_target_block": target_block,
        "chain_target_block_hash": _sha256_ref(
            chain_evidence.get("target_block_hash"), "chain target block hash"
        ),
        "chain_finalized_head_block": _integer(
            chain_evidence.get("finalized_head_block"), "finalized head block"
        ),
        "validator_uid": _integer(
            chain_evidence.get("validator_uid"), "validator UID"
        ),
        "weights_storage_key_hash": sha256_json(
            {"storage_key": str(chain_evidence.get("weights_storage_key") or "")}
        ),
        "audit_event_hash": "sha256:" + audit["event_hash"],
        "audit_payload_hash": audit["payload_hash"],
        "checkpoint_merkle_root": "sha256:" + checkpoint_root,
        "checkpoint_number": checkpoint["checkpoint_number"],
        "checkpoint_event_sequence": checkpoint["event_sequence"],
        "arweave_tx_id": arweave_tx_id,
    }
    return validate_legacy_settlement_document_v2(
        {**body, "settlement_hash": sha256_json(body)}
    )


def validate_legacy_settlement_document_v2(
    document: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a persisted/output settlement without external proof inputs."""

    if not isinstance(document, Mapping) or set(document) != _SETTLEMENT_FIELDS:
        raise LegacySettlementV2Error("legacy settlement fields are invalid")
    normalized = dict(document)
    if normalized.get("schema_version") != LEGACY_SETTLEMENT_SCHEMA_VERSION:
        raise LegacySettlementV2Error("legacy settlement schema is invalid")
    netuid = _integer(normalized.get("netuid"), "settlement netuid", positive=True)
    epoch_id = _integer(normalized.get("epoch_id"), "settlement epoch")
    allocation = normalized.get("allocation_doc")
    if not isinstance(allocation, Mapping):
        raise LegacySettlementV2Error("settlement allocation is missing")
    allocation_hash = _sha256_ref(
        normalized.get("allocation_hash"), "settlement allocation hash"
    )
    if (
        allocation.get("allocation_hash") != allocation_hash
        or _integer(allocation.get("epoch"), "settlement allocation epoch")
        != epoch_id
        or _integer(
            allocation.get("netuid"), "settlement allocation netuid", positive=True
        )
        != netuid
        or sha256_json(
            {key: value for key, value in allocation.items() if key != "allocation_hash"}
        )
        != allocation_hash
    ):
        raise LegacySettlementV2Error("settlement allocation differs")
    _raw_hash(
        normalized.get("legacy_bundle_weights_hash"), "legacy bundle weights hash"
    )
    for field in (
        "chain_compare_hash",
        "chain_target_block_hash",
        "weights_storage_key_hash",
        "audit_event_hash",
        "audit_payload_hash",
        "checkpoint_merkle_root",
    ):
        _sha256_ref(normalized.get(field), field)
    if not str(normalized.get("validator_hotkey") or "").strip():
        raise LegacySettlementV2Error("settlement validator hotkey is missing")
    if not re.fullmatch(
        r"[A-Za-z0-9_-]{43}", str(normalized.get("arweave_tx_id") or "")
    ):
        raise LegacySettlementV2Error("settlement Arweave transaction is invalid")
    for field in (
        "legacy_bundle_block",
        "chain_vector_tolerance_u16",
        "chain_target_block",
        "chain_finalized_head_block",
        "validator_uid",
        "checkpoint_number",
        "checkpoint_event_sequence",
    ):
        _integer(normalized.get(field), field)
    body = {key: value for key, value in normalized.items() if key != "settlement_hash"}
    if _sha256_ref(normalized.get("settlement_hash"), "settlement hash") != sha256_json(
        body
    ):
        raise LegacySettlementV2Error("legacy settlement hash differs")
    return normalized
