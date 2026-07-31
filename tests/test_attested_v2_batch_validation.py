from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical import attested_v2


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = "2026-07-31T20:00:00Z"


def _overlapping_graphs():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = attested_v2.create_boot_identity(
        body=attested_v2.build_boot_identity_body(
            role="gateway_coordinator",
            physical_role="gateway_coordinator",
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_A,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH_A,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )

    def receipt(job_id, purpose, parents=()):
        return attested_v2.create_signed_execution_receipt(
            body=attested_v2.build_execution_receipt_body(
                role="gateway_coordinator",
                purpose=purpose,
                job_id=job_id,
                epoch_id=24_279,
                sequence=0,
                commit_sha="c" * 40,
                pcr0="d" * 96,
                build_manifest_hash=HASH_A,
                dependency_lock_hash=HASH_B,
                config_hash=HASH_A,
                boot_identity_hash=boot["boot_identity_hash"],
                input_root=HASH_A,
                output_root=HASH_B,
                transport_root_hash=attested_v2.merkle_root(
                    (), domain="leadpoet-transport-v2"
                ),
                host_operation_root_hash=attested_v2.merkle_root(
                    (), domain="leadpoet-host-operation-v2"
                ),
                artifact_root=attested_v2.merkle_root(
                    (), domain="leadpoet-artifact-v2"
                ),
                parent_receipt_hashes=parents,
                status="succeeded",
                failure_code=None,
                issued_at=NOW,
            ),
            enclave_pubkey=pubkey,
            sign_digest=key.sign,
        )

    shared = receipt("shared-job", "research_lab.provider_evidence.v2")
    child_a = receipt(
        "child-a",
        "research_lab.allocation.v2",
        (shared["receipt_hash"],),
    )
    child_b = receipt(
        "child-b",
        "research_lab.reward_decision.v2",
        (shared["receipt_hash"],),
    )
    return (
        attested_v2.build_receipt_graph(
            root_receipt_hash=child_a["receipt_hash"],
            boot_identities=(boot,),
            receipts=(shared, child_a),
            transport_attempts=(),
        ),
        attested_v2.build_receipt_graph(
            root_receipt_hash=child_b["receipt_hash"],
            boot_identities=(boot,),
            receipts=(shared, child_b),
            transport_attempts=(),
        ),
    )


def test_batch_verifies_each_exact_signed_object_once(monkeypatch):
    graphs = _overlapping_graphs()
    counts = {"boot": 0, "receipt": 0}
    validate_boot = attested_v2.validate_boot_identity
    validate_receipt = attested_v2.validate_signed_execution_receipt

    def counted_boot(value):
        counts["boot"] += 1
        return validate_boot(value)

    def counted_receipt(value, *, verify_signature=True):
        counts["receipt"] += 1
        return validate_receipt(value, verify_signature=verify_signature)

    monkeypatch.setattr(attested_v2, "validate_boot_identity", counted_boot)
    monkeypatch.setattr(
        attested_v2,
        "validate_signed_execution_receipt",
        counted_receipt,
    )

    ordered = attested_v2.validate_receipt_graphs(graphs)

    assert len(ordered) == 2
    assert counts == {"boot": 1, "receipt": 3}


def test_batch_rejects_conflicting_object_with_a_verified_hash():
    graph_a, graph_b = _overlapping_graphs()
    conflicting = dict(graph_b)
    conflicting["receipts"] = [dict(item) for item in graph_b["receipts"]]
    conflicting["receipts"][0]["output_root"] = HASH_A

    with pytest.raises(
        attested_v2.AttestedV2Error,
        match="receipt hash conflicts across receipt graphs",
    ):
        attested_v2.validate_receipt_graphs((graph_a, conflicting))


def test_batch_rejects_failure_policy_count_mismatch():
    graphs = _overlapping_graphs()
    with pytest.raises(
        attested_v2.AttestedV2Error,
        match="failure policies differ from graph count",
    ):
        attested_v2.validate_receipt_graphs(
            graphs,
            allowed_failed_receipt_hashes_by_graph=((),),
        )
