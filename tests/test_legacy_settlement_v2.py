from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.events import compute_event_hash
from leadpoet_canonical.legacy_settlement_v2 import (
    LegacySettlementV2Error,
    validate_legacy_allocation_nonfinalization_v2,
    validate_legacy_finalized_settlement_v2,
    validate_legacy_nonfinalization_document_v2,
    validate_legacy_settlement_document_v2,
)
from leadpoet_canonical.weights import bundle_weights_hash


def _fixture(*, allocation_netuid=71) -> dict:
    netuid = 71
    epoch = 100
    block = epoch * 360 + 20
    validator_hotkey = "validator-hotkey"
    vector = [(1, 1000), (4, 2000)]
    bundle_key = Ed25519PrivateKey.generate()
    bundle_pubkey = bundle_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    weights_hash = bundle_weights_hash(netuid, epoch, block, vector)
    bundle = {
        "netuid": netuid,
        "epoch_id": epoch,
        "block": block,
        "uids": [uid for uid, _weight in vector],
        "weights_u16": [weight for _uid, weight in vector],
        "weights_hash": weights_hash,
        "validator_hotkey": validator_hotkey,
        "validator_enclave_pubkey": bundle_pubkey,
        "validator_signature": bundle_key.sign(
            bytes.fromhex(weights_hash)
        ).hex(),
    }
    allocation_body = {
        "schema_version": "1.0",
        "epoch": epoch,
        "champion_allocations": [],
        "queued_champion_allocations": [],
    }
    if allocation_netuid is not None:
        allocation_body["netuid"] = allocation_netuid
    allocation = {
        **allocation_body,
        "allocation_hash": sha256_json(allocation_body),
    }
    payload_body = {
        "schema_version": "1.0",
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "actor_hotkey": validator_hotkey,
        "epoch": epoch,
        "netuid": netuid,
        "audit_kind": "active",
        "lab_allocation": {"allocation_hash": allocation["allocation_hash"]},
        "weights": {"weights_hash": weights_hash},
    }
    payload = {**payload_body, "payload_hash": sha256_json(payload_body)}
    audit_key = Ed25519PrivateKey.generate()
    audit_pubkey = audit_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    signed_event = {
        "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
        "timestamp": "2026-07-10T00:00:00Z",
        "boot_id": "boot-1",
        "monotonic_seq": 12,
        "prev_event_hash": "0" * 64,
        "payload": payload,
    }
    event_hash = compute_event_hash(signed_event)
    signed_log_entry = {
        "signed_event": signed_event,
        "event_hash": event_hash,
        "enclave_pubkey": audit_pubkey,
        "enclave_signature": audit_key.sign(bytes.fromhex(event_hash)).hex(),
    }
    event = {
        "sequence": 77,
        "event_hash": event_hash,
        "signed_log_entry": signed_log_entry,
    }
    leaf = hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint = {
        "header": {
            "checkpoint_number": 8,
            "event_count": 1,
            "merkle_root": leaf,
            "sequence_range": {"first": 77, "last": 77},
        },
        "signature": "historical-checkpoint-signature",
        "events_compressed": base64.b64encode(
            gzip.compress(json.dumps([event]).encode())
        ).decode(),
        "tree_levels": [[leaf]],
    }
    anchor = {
        "audit_kind": "active",
        "allocation_hash": allocation["allocation_hash"],
        "weights_hash": weights_hash,
        "payload_hash": sha256_json(payload),
        "current_transparency_event_hash": event_hash,
        "current_tee_sequence": 77,
        "current_checkpoint_merkle_root": leaf,
        "current_arweave_tx_id": "A" * 43,
    }
    log_row = {
        "event_hash": event_hash,
        "payload_hash": sha256_json(payload),
        "enclave_pubkey": audit_pubkey,
        "signed_log_entry": signed_log_entry,
    }
    chain = {
        "epoch_id": epoch,
        "netuid": netuid,
        "target_block": (epoch + 1) * 360 - 1,
        "target_block_hash": "1" * 64,
        "finalized_head_block": (epoch + 1) * 360 + 10,
        "validator_hotkey": validator_hotkey,
        "validator_uid": 3,
        "weights_storage_key": "0x1234",
        "weights": [[uid, weight] for uid, weight in vector],
    }
    return {
        "netuid": netuid,
        "epoch_id": epoch,
        "allocation_doc": allocation,
        "weight_bundle": bundle,
        "audit_anchor": anchor,
        "transparency_log_row": log_row,
        "chain_evidence": chain,
        "arweave_checkpoint": checkpoint,
    }


def test_historical_settlement_binds_allocation_bundle_chain_and_checkpoint():
    inputs = _fixture()
    document = validate_legacy_finalized_settlement_v2(**inputs)
    assert document["epoch_id"] == 100
    assert document["allocation_hash"] == inputs["allocation_doc"][
        "allocation_hash"
    ]
    assert document["chain_vector_tolerance_u16"] == 1
    assert validate_legacy_settlement_document_v2(document) == document


def test_historical_settlement_accepts_hash_bound_allocation_without_netuid():
    inputs = _fixture(allocation_netuid=None)
    document = validate_legacy_finalized_settlement_v2(**inputs)
    assert "netuid" not in document["allocation_doc"]
    assert validate_legacy_settlement_document_v2(document) == document


def test_historical_settlement_rejects_mismatched_embedded_netuid():
    with pytest.raises(LegacySettlementV2Error, match="allocation document scope"):
        validate_legacy_finalized_settlement_v2(
            **_fixture(allocation_netuid=72)
        )


@pytest.mark.parametrize("tamper", ("chain", "checkpoint", "allocation"))
def test_historical_settlement_fails_closed_on_tampered_evidence(tamper):
    inputs = _fixture()
    if tamper == "chain":
        inputs["chain_evidence"]["weights"][0][1] += 2
        match = "chain vector differs"
    elif tamper == "checkpoint":
        inputs["arweave_checkpoint"]["tree_levels"] = [["f" * 64]]
        match = "Merkle tree differs"
    else:
        inputs["allocation_doc"]["champion_allocations"].append(
            {"source_id": "champion:tampered", "paid_alpha_percent": 1}
        )
        match = "allocation document hash differs"
    with pytest.raises((LegacySettlementV2Error, ValueError), match=match):
        validate_legacy_finalized_settlement_v2(**inputs)


def test_persisted_settlement_hash_prevents_partial_restart_substitution():
    document = validate_legacy_finalized_settlement_v2(**_fixture())
    tampered = copy.deepcopy(document)
    tampered["validator_uid"] += 1
    with pytest.raises(LegacySettlementV2Error, match="hash differs"):
        validate_legacy_settlement_document_v2(tampered)


def test_historical_nonfinalization_preserves_unpaid_allocation():
    inputs = _fixture()
    inputs.pop("arweave_checkpoint")
    inputs["chain_evidence"]["weights"][0][1] += 5

    finding = validate_legacy_allocation_nonfinalization_v2(**inputs)

    assert finding["epoch_id"] == 100
    assert finding["differing_uid_count"] == 1
    assert finding["allocation_hash"] == inputs["allocation_doc"][
        "allocation_hash"
    ]
    assert validate_legacy_nonfinalization_document_v2(finding) == finding


def test_historical_nonfinalization_rejects_matching_chain_vector():
    inputs = _fixture()
    inputs.pop("arweave_checkpoint")

    with pytest.raises(LegacySettlementV2Error, match="matches"):
        validate_legacy_allocation_nonfinalization_v2(**inputs)


def test_persisted_nonfinalization_hash_rejects_tampering():
    inputs = _fixture()
    inputs.pop("arweave_checkpoint")
    inputs["chain_evidence"]["weights"][0][1] += 5
    finding = validate_legacy_allocation_nonfinalization_v2(**inputs)
    tampered = copy.deepcopy(finding)
    tampered["differing_uid_count"] += 1

    with pytest.raises(LegacySettlementV2Error, match="hash differs"):
        validate_legacy_nonfinalization_document_v2(tampered)
