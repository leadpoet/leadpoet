from __future__ import annotations

from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_receipts import (
    SCORING_ROLE,
    WEIGHT_PURPOSE,
    WEIGHT_ROLE,
    ReceiptError,
    artifact_commitment,
    artifact_merkle_root,
    build_receipt_body,
    create_signed_receipt,
    sha256_json,
    validate_signed_receipt,
    verify_receipt_lineage,
)


COMMIT = "1" * 40
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def _keypair():
    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    return private, public_hex


def _receipt(*, purpose="research_lab.candidate_score.v1", role=SCORING_ROLE, parents=()):
    private, public_hex = _keypair()
    body = build_receipt_body(
        role=role,
        purpose=purpose,
        job_id="job:test-1",
        epoch_id=123,
        commit_sha=COMMIT,
        build_manifest_hash=HASH_A,
        config_hash=HASH_B,
        input_root=HASH_A,
        output_root=HASH_C,
        evidence_roots={"provider": HASH_B},
        parent_receipt_hashes=parents,
        status="succeeded",
        issued_at="2026-07-10T12:00:00Z",
    )
    return create_signed_receipt(
        body=body,
        enclave_pubkey=public_hex,
        attestation_document_b64="attestation",
        sign_digest=private.sign,
    )


def test_receipt_round_trip_and_signature_verification():
    receipt = _receipt()
    validate_signed_receipt(receipt)
    assert receipt["receipt_hash"].startswith("sha256:")


def test_receipt_rejects_tampered_output():
    receipt = _receipt()
    tampered = deepcopy(receipt)
    tampered["output_root"] = HASH_A
    with pytest.raises(ReceiptError, match="receipt_hash"):
        validate_signed_receipt(tampered)


def test_receipt_lineage_requires_every_parent():
    parent = _receipt()
    child = _receipt(purpose="research_lab.allocation.v1", parents=(parent["receipt_hash"],))
    assert verify_receipt_lineage(child, {parent["receipt_hash"]: parent}) == (
        parent["receipt_hash"],
        child["receipt_hash"],
    )
    with pytest.raises(ReceiptError, match="missing parent"):
        verify_receipt_lineage(child, {})


def test_weight_purpose_cannot_be_signed_by_scoring_role():
    with pytest.raises(ReceiptError, match="purpose"):
        build_receipt_body(
            role=SCORING_ROLE,
            purpose=WEIGHT_PURPOSE,
            job_id="job:weights",
            epoch_id=123,
            commit_sha=COMMIT,
            build_manifest_hash=HASH_A,
            config_hash=HASH_B,
            input_root=HASH_A,
            output_root=HASH_C,
            evidence_roots={},
            parent_receipt_hashes=(),
            status="succeeded",
            issued_at="2026-07-10T12:00:00Z",
        )


def test_artifact_merkle_root_is_order_independent_and_content_sensitive():
    first = artifact_commitment("model_output", b"one")
    second = artifact_commitment("provider_evidence", b"two")
    assert artifact_merkle_root([first, second]) == artifact_merkle_root([second, first])
    changed = artifact_commitment("provider_evidence", b"changed")
    assert artifact_merkle_root([first, second]) != artifact_merkle_root([first, changed])


def test_weight_receipt_role_is_supported():
    receipt = _receipt(role=WEIGHT_ROLE, purpose=WEIGHT_PURPOSE)
    validate_signed_receipt(receipt)
