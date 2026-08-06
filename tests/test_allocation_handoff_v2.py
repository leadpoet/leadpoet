import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import allocation_handoff_disk_cache
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    build_full_graph_parent_v2,
    issue_ancestry_certificate_v2,
)
from leadpoet_canonical.allocation_handoff_v2 import (
    AllocationHandoffV2Error,
    build_allocation_handoff_v2,
    validate_allocation_handoff_v2,
)
from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_checkpointed_receipt_graph,
    build_execution_receipt_body,
    build_receipt_graph,
    compact_checkpointed_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_json,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
COMMIT = "d" * 40
PCR0 = "e" * 96
NOW = "2026-07-12T00:00:00Z"


def _document():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=COORDINATOR_ROLE,
            physical_role="gateway_coordinator",
            commit_sha=COMMIT,
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="1" * 32,
            signing_pubkey=public_key,
            transport_pubkey="2" * 64,
            transport_certificate_hash=HASH_A,
            attestation_user_data_hash=HASH_B,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode("ascii"),
    )

    def receipt(*, purpose, job_id, output_root, parents=(), sequence=0):
        return create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role=COORDINATOR_ROLE,
                purpose=purpose,
                job_id=job_id,
                epoch_id=23,
                sequence=sequence,
                commit_sha=COMMIT,
                pcr0=PCR0,
                build_manifest_hash=HASH_A,
                dependency_lock_hash=HASH_B,
                config_hash=HASH_C,
                boot_identity_hash=boot["boot_identity_hash"],
                input_root=HASH_A,
                output_root=output_root,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=EMPTY_ARTIFACT_ROOT,
                parent_receipt_hashes=parents,
                status="succeeded",
                failure_code=None,
                issued_at=NOW,
            ),
            enclave_pubkey=public_key,
            sign_digest=private_key.sign,
        )

    parent = receipt(
        purpose="research_lab.reward_decision.v2",
        job_id="reward:1",
        output_root=HASH_C,
    )
    allocation = {"allocation_hash": HASH_B, "lab_cap_percent": 20.0}
    root = receipt(
        purpose="research_lab.allocation.v2",
        job_id="allocation:23",
        output_root=sha256_json({"allocation": allocation}),
        parents=(parent["receipt_hash"],),
        sequence=1,
    )
    graph = build_receipt_graph(
        root_receipt_hash=root["receipt_hash"],
        boot_identities=[boot],
        receipts=[parent, root],
        transport_attempts=[],
        host_operations=[],
    )
    return build_allocation_handoff_v2(
        bundle={"epoch": 23, "netuid": 71, "allocation_doc": allocation},
        receipt_graph=graph,
        lineage_bindings=[
            {
                "receipt_hash": parent["receipt_hash"],
                "receipt_purpose": parent["purpose"],
                "receipt_role": parent["role"],
            }
        ],
        lineage_complete=True,
        persistence={"root_receipt_hash": root["receipt_hash"]},
    )


def _checkpoint_document():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=COORDINATOR_ROLE,
            physical_role="gateway_coordinator",
            commit_sha=COMMIT,
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="4" * 32,
            signing_pubkey=public_key,
            transport_pubkey="5" * 64,
            transport_certificate_hash=HASH_A,
            attestation_user_data_hash=HASH_B,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"attestation").decode("ascii"),
    )

    def receipt(*, purpose, job_id, output_root, parents=(), sequence=0):
        return create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role=COORDINATOR_ROLE,
                purpose=purpose,
                job_id=job_id,
                epoch_id=23,
                sequence=sequence,
                commit_sha=COMMIT,
                pcr0=PCR0,
                build_manifest_hash=HASH_A,
                dependency_lock_hash=HASH_B,
                config_hash=HASH_C,
                boot_identity_hash=boot["boot_identity_hash"],
                input_root=HASH_A,
                output_root=output_root,
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=EMPTY_ARTIFACT_ROOT,
                parent_receipt_hashes=parents,
                status="succeeded",
                failure_code=None,
                issued_at=NOW,
            ),
            enclave_pubkey=public_key,
            sign_digest=private_key.sign,
        )

    parent = receipt(
        purpose="research_lab.reward_decision.v2",
        job_id="reward:checkpoint",
        output_root=HASH_C,
    )
    parent_graph = build_receipt_graph(
        root_receipt_hash=parent["receipt_hash"],
        boot_identities=[boot],
        receipts=[parent],
        transport_attempts=[],
        host_operations=[],
    )
    allocation = {"allocation_hash": HASH_B, "lab_cap_percent": 20.0}
    root = receipt(
        purpose="research_lab.allocation.v2",
        job_id="allocation:checkpoint:23",
        output_root=sha256_json({"allocation": allocation}),
        parents=(parent["receipt_hash"],),
        sequence=1,
    )
    lineage_id = "sha256:" + "f" * 64

    def verify_boot(identity):
        assert identity["commit_sha"] == COMMIT
        assert identity["pcr0"] == PCR0
        return identity

    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": root["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [root],
        "transport_attempts": [],
        "host_operations": [],
    }
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=boot,
        issued_at=NOW,
        sign_digest=private_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(COORDINATOR_ROLE,),
        parent_full_graphs=(build_full_graph_parent_v2(parent_graph),),
        required_purposes=("research_lab.allocation.v2",),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(COORDINATOR_ROLE,),
    )
    graph = build_checkpointed_receipt_graph(
        root_receipt_hash=root["receipt_hash"],
        boot_identities=[boot],
        receipts=[root],
        transport_attempts=[],
        host_operations=[],
        ancestry_lineage_id=lineage_id,
        ancestry_proof=proof,
        boot_attestation_verifier=verify_boot,
        require_boot_attestation_verification=True,
    )
    handoff = build_allocation_handoff_v2(
        bundle={"epoch": 23, "netuid": 71, "allocation_doc": allocation},
        receipt_graph=graph,
        lineage_bindings=[
            {
                "receipt_hash": parent["receipt_hash"],
                "receipt_purpose": parent["purpose"],
                "receipt_role": parent["role"],
            }
        ],
        lineage_complete=True,
        persistence={"root_receipt_hash": root["receipt_hash"]},
    )
    return handoff


def test_allocation_handoff_binds_complete_graph_and_scope():
    document = _document()
    normalized = validate_allocation_handoff_v2(
        document,
        expected_epoch_id=23,
        expected_netuid=71,
    )
    assert normalized == document


def test_allocation_handoff_binds_checkpoint_parent_metadata_without_expansion():
    document = _checkpoint_document()

    assert len(document["receipt_graph"]["receipts"]) == 1
    assert validate_allocation_handoff_v2(document) == document


def test_allocation_handoff_binds_compact_checkpoint_parent_metadata():
    document = _checkpoint_document()
    document["receipt_graph"] = compact_checkpointed_receipt_graph(
        document["receipt_graph"]
    )

    assert document["receipt_graph"]["transport_attempts"] == []
    assert document["receipt_graph"]["host_operations"] == []
    assert validate_allocation_handoff_v2(document) == document


@pytest.mark.parametrize("field", ["parent_role", "parent_purpose"])
def test_allocation_handoff_rejects_tampered_checkpoint_parent_metadata(field):
    document = deepcopy(_checkpoint_document())
    authority = document["receipt_graph"]["ancestry_proof"]["certificate"][
        "claim"
    ]["parent_authorities"][0]
    authority[field] = "research_lab.tampered.v2"

    with pytest.raises((AllocationHandoffV2Error, ValueError)):
        validate_allocation_handoff_v2(document)


def test_disk_cache_accepts_only_matching_attested_release(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "RESEARCH_LAB_ALLOCATION_HANDOFF_DIR",
        str(tmp_path),
    )
    handoff = _document()
    allocation_handoff_disk_cache.store_handoff(
        71,
        23,
        True,
        COMMIT,
        handoff,
        ttl_seconds=60,
    )
    assert allocation_handoff_disk_cache.load_handoff(
        71,
        23,
        True,
        COMMIT,
    ) == handoff
    assert (
        allocation_handoff_disk_cache.load_handoff(
            71,
            23,
            True,
            "f" * 40,
        )
        is None
    )


@pytest.mark.parametrize("mutation", ["allocation", "binding", "persistence"])
def test_allocation_handoff_rejects_incomplete_or_tampered_authority(mutation):
    document = deepcopy(_document())
    if mutation == "allocation":
        document["bundle"]["allocation_doc"]["lab_cap_percent"] = 19.0
    elif mutation == "binding":
        document["lineage_bindings"] = []
    else:
        document["persistence"]["root_receipt_hash"] = HASH_A
    with pytest.raises(AllocationHandoffV2Error):
        validate_allocation_handoff_v2(document)
