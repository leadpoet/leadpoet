import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
    build_execution_receipt_body,
    build_receipt_graph,
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


def test_allocation_handoff_binds_complete_graph_and_scope():
    document = _document()
    normalized = validate_allocation_handoff_v2(
        document,
        expected_epoch_id=23,
        expected_netuid=71,
    )
    assert normalized == document


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
