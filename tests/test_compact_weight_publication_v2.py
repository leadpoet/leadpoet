from __future__ import annotations

import copy
from copy import deepcopy

import pytest

from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    derive_ancestry_lineage_id_v2,
    issue_ancestry_certificate_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_execution_receipt_body,
    build_receipt_graph,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.binding import create_binding_message

import leadpoet_canonical.compact_auditor_authority_v2 as compact_auditor
from leadpoet_canonical.compact_auditor_authority_v2 import (
    build_compact_published_weight_authority_v2,
    verify_compact_published_weight_authority_v2,
)
from leadpoet_canonical.compact_weight_authority_v2 import (
    CompactWeightAuthorityV2Error,
    compact_weight_bundle_hash_v2,
    reconstruct_published_weight_bundle_from_compact_v2,
    validate_compact_weight_ancestry_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    validate_published_weight_bundle_v2,
    weight_input_output_roots_v2,
    weight_input_value_documents_v2,
)
from tests.test_validator_weight_authority_v2 import (
    NOW,
    VALIDATOR_HOTKEY,
    _calculation_snapshot,
    _fixture,
    _receipt,
    _source_attempt,
)
from tests.test_validator_hotkey_authority_v2 import (
    HOTKEY_PUBLIC,
    _Drand,
    _Sr25519,
    _profile,
)
from validator_tee.enclave.hotkey_authority_v2 import ValidatorHotkeyAuthorityV2
from validator_tee.host.weight_authority_v2 import (
    build_compact_weight_submission_v2,
)


def test_compact_weight_publication_reconstructs_exact_canonical_bundle(monkeypatch):
    fixture = _fixture(stateful=True)
    authority = fixture["authority"]
    validator_boot = fixture["validator_boot"]
    gateway_boot = fixture["gateway_boot"]
    gateway_key = fixture["gateway_key"]
    chain_snapshot = fixture["chain_source"].read_finalized_snapshot(
        netuid=71, epoch_id=100
    )
    epoch_authority = chain_snapshot["epoch_authority"]
    lineage_id = derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=epoch_authority["cutover_mapping_hash"],
        network_genesis_hash=epoch_authority["network_genesis_hash"],
        netuid=epoch_authority["netuid"],
    )
    verify_boot = lambda identity: authority._verify_one_boot(  # noqa: SLF001
        identity, validator_boot
    )

    event_receipt = _receipt(
        boot=gateway_boot,
        private_key=gateway_key,
        purpose="research_lab.allocation.v2",
        job_id="weight-allocation-authority:100",
        sequence=100,
        output_root=sha256_json({"epoch_id": 100, "authority": "gateway"}),
    )
    event_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": event_receipt["receipt_hash"],
        "boot_identities": [gateway_boot],
        "receipts": [event_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    event_certificate = issue_ancestry_certificate_v2(
        local_delta=event_delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=gateway_boot,
        issued_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sign_digest=gateway_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles={"gateway_coordinator"},
        required_purposes={"research_lab.allocation.v2"},
    )

    preliminary = _calculation_snapshot([], "")
    gateway_event_hash = event_receipt["receipt_hash"]
    expected_roots = weight_input_output_roots_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=fixture["finalized_chain_state_root"],
        gateway_authority_event_hash=gateway_event_hash,
    )
    documents = weight_input_value_documents_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=fixture["finalized_chain_state_root"],
        gateway_authority_event_hash=gateway_event_hash,
    )
    proofs = {}
    full_graphs = {}
    input_hashes = {}
    direct_attempts = []
    for sequence, category in enumerate(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)):
        _role, purpose = WEIGHT_INPUT_PURPOSES[category]
        job_id = "compact-weight-input-%s" % category
        attempt = None
        if category != "anomaly_adjustments":
            attempt = _source_attempt(
                category=category,
                job_id=job_id,
                purpose=purpose,
                sequence=sequence + 200,
                provider_id="supabase",
                host="qplwoislplkcegvdmbim.supabase.co",
                method="GET",
            )
            direct_attempts.append(attempt)
        artifacts = [sha256_json(documents[category]["value"])]
        if attempt is not None:
            artifacts.extend(
                [attempt["request_artifact_hash"], attempt["response_artifact_hash"]]
            )
        receipt = _receipt(
            boot=gateway_boot,
            private_key=gateway_key,
            purpose=purpose,
            job_id=job_id,
            sequence=sequence,
            output_root=expected_roots[category],
            transport_root=(
                merkle_root([attempt["attempt_hash"]], domain="leadpoet-transport-v2")
                if attempt is not None
                else EMPTY_TRANSPORT_ROOT
            ),
            artifact_root=merkle_root(artifacts, domain="leadpoet-artifact-v2"),
            parent_receipt_hashes=(
                (gateway_event_hash,) if category == "research_lab_allocation" else ()
            ),
        )
        delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": receipt["receipt_hash"],
            "boot_identities": [gateway_boot],
            "receipts": [receipt],
            "transport_attempts": ([attempt] if attempt is not None else []),
            "host_operations": [],
        }
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=lineage_id,
            certificate_sequence=(1 if category == "research_lab_allocation" else 0),
            issuer_boot_identity=gateway_boot,
            issued_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sign_digest=gateway_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
            parent_certificates=(
                [event_certificate]
                if category == "research_lab_allocation"
                else []
            ),
            required_purposes={purpose},
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
        )
        graph_receipts = (
            [event_receipt, receipt]
            if category == "research_lab_allocation"
            else [receipt]
        )
        full_graph = build_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=[gateway_boot],
            receipts=graph_receipts,
            transport_attempts=([attempt] if attempt is not None else []),
            host_operations=[],
        )
        proofs[category] = proof
        full_graphs[receipt["receipt_hash"]] = full_graph
        input_hashes[category] = receipt["receipt_hash"]

    calculation = _calculation_snapshot(
        input_hashes.values(), input_hashes["research_lab_allocation"]
    )
    enclave_response = authority.compute(
        {
            "validator_hotkey": VALIDATOR_HOTKEY,
            "calculation_snapshot": calculation,
            "input_receipt_hashes": input_hashes,
            "gateway_authority_event_hash": gateway_event_hash,
            "upstream_ancestry_proofs": proofs,
            "upstream_transport_attempts": direct_attempts,
        }
    )
    hotkey_authority = ValidatorHotkeyAuthorityV2(
        boot_identity_supplier=lambda: fixture["validator_boot"],
        validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
        chain_profile=_profile(),
        sign_receipt_digest=fixture["validator_key"].sign,
        attestation_supplier=lambda **_kwargs: b"attestation",
        drand_backend=_Drand(),
        chain_source=fixture["chain_source"],
        sr25519_backend=_Sr25519(),
        clock=lambda: NOW,
    )
    enclave_response["weight_authorization_id"] = (
        hotkey_authority.register_weight_result(
            {
                field: enclave_response[field]
                for field in (
                    "weight_snapshot",
                    "weight_result",
                    "weights_signature",
                    "receipt_graph_delta",
                    "ancestry_commitment",
                    "boot_identity",
                )
            }
        )
    )
    boot = enclave_response["boot_identity"]
    binding_message = create_binding_message(
        netuid=71,
        chain="wss://entrypoint-finney.opentensor.ai:443",
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    application_request = build_application_signature_request_v2(
        message=binding_message.encode("utf-8"),
        validator_hotkey=VALIDATOR_HOTKEY,
        boot_identity_hash=boot["boot_identity_hash"],
    )
    signature_output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": application_request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "signature": "f" * 128,
    }
    binding_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="validator_weights",
            purpose="validator.hotkey_signature.v2",
            job_id="hotkey-signature:%s" % application_request["request_hash"].split(":", 1)[1],
            epoch_id=100,
            sequence=0,
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=application_request["request_hash"],
            output_root=sha256_json(signature_output),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(enclave_response["receipt_graph_delta"]["root_receipt_hash"],),
            status="succeeded",
            failure_code=None,
            issued_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=fixture["validator_key"].sign,
    )
    validator_proof = authority.issue_validator_publication_ancestry_proof(
        validator_receipt_delta=enclave_response["receipt_graph_delta"],
        upstream_ancestry_proofs=proofs,
        binding_receipt=binding_receipt,
        epoch_authority=enclave_response["epoch_authority"],
    )
    compact = build_compact_weight_submission_v2(
        enclave_response=enclave_response,
        validator_hotkey=VALIDATOR_HOTKEY,
        binding_message=binding_message,
        binding_signature_result={
            **signature_output,
            "receipt": binding_receipt,
            "validator_ancestry_proof": validator_proof,
        },
    )
    bundle = reconstruct_published_weight_bundle_from_compact_v2(
        compact,
        expected_lineage_id=lineage_id,
        full_graphs_by_root=full_graphs,
        boot_attestation_verifier=verify_boot,
    )

    verified = validate_published_weight_bundle_v2(bundle)
    assert bundle["weight_result"] == enclave_response["weight_result"]
    assert verified["weights_hash"] == enclave_response["weight_result"]["weights_hash"]
    assert verified["uids"] == enclave_response["weight_result"]["uids"]
    assert bundle["weight_result"]["weights"] == enclave_response["weight_result"][
        "weights"
    ]

    def assert_compact_rejected(tampered):
        tampered["compact_submission_hash"] = sha256_json(
            {
                key: value
                for key, value in tampered.items()
                if key != "compact_submission_hash"
            }
        )
        with pytest.raises(CompactWeightAuthorityV2Error):
            reconstruct_published_weight_bundle_from_compact_v2(
                tampered,
                expected_lineage_id=lineage_id,
                full_graphs_by_root=full_graphs,
                boot_attestation_verifier=verify_boot,
            )

    dropped_receipt = deepcopy(compact)
    dropped_receipt["validator_receipt_delta"]["receipts"].pop(0)
    assert_compact_rejected(dropped_receipt)

    dropped_attempt = deepcopy(compact)
    assert dropped_attempt["validator_receipt_delta"]["transport_attempts"]
    dropped_attempt["validator_receipt_delta"]["transport_attempts"].pop(0)
    assert_compact_rejected(dropped_attempt)

    extra_receipt = deepcopy(compact)
    extra_receipt["validator_receipt_delta"]["receipts"].append(event_receipt)
    assert_compact_rejected(extra_receipt)

    extra_boot = deepcopy(compact)
    extra_boot["validator_receipt_delta"]["boot_identities"].append(gateway_boot)
    assert_compact_rejected(extra_boot)

    swapped_proof = deepcopy(compact)
    swapped_proof["validator_ancestry_proof"] = next(
        iter(swapped_proof["upstream_ancestry_proofs"].values())
    )
    assert_compact_rejected(swapped_proof)

    compact_bundle_hash = compact_weight_bundle_hash_v2(compact)
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": compact_bundle_hash,
        "root_receipt_hash": verified["root_receipt_hash"],
        "durable_readback_hash": "sha256:" + "d" * 64,
        "transparency_event_hash": "sha256:" + "e" * 64,
    }
    publication_receipt = _receipt(
        boot=gateway_boot,
        private_key=gateway_key,
        purpose="gateway.weights.publication.v2",
        job_id="compact-publication:100",
        sequence=0,
        output_root=sha256_json(publication_doc),
        parent_receipt_hashes=(verified["root_receipt_hash"],),
    )
    publication_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": publication_receipt["receipt_hash"],
        "boot_identities": [gateway_boot],
        "receipts": [publication_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    publication_certificate = issue_ancestry_certificate_v2(
        local_delta=publication_delta,
        lineage_id=lineage_id,
        certificate_sequence=(
            int(validator_proof["certificate"]["claim"]["certificate_sequence"])
            + 1
        ),
        issuer_boot_identity=gateway_boot,
        issued_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sign_digest=gateway_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles={"gateway_coordinator", "validator_weights"},
        parent_certificates=[validator_proof["certificate"]],
        required_purposes={"gateway.weights.publication.v2"},
    )
    publication_proof = build_compact_ancestry_proof_from_delta_v2(
        publication_delta,
        publication_certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles={"gateway_coordinator", "validator_weights"},
    )
    event_hash = sha256_json(
        {
            "bundle_hash": compact_bundle_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": publication_doc["durable_readback_hash"],
        }
    )
    public_authority = build_compact_published_weight_authority_v2(
        authority_stage="published",
        lineage_id=lineage_id,
        bundle_hash=compact_bundle_hash,
        compact_submission=compact,
        publication={
            "weight_submission_event_hash": event_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "publication_doc": publication_doc,
            "ancestry_proof": publication_proof,
        },
        finalization=None,
    )
    identity_cache = {
        "schema_version": "leadpoet.independent_pcr0_identities.v2",
        "entries": [
            {
                "physical_role": item["physical_role"],
                "role": item["role"],
                "commit_sha": item["commit_sha"],
                "pcr0": item["pcr0"],
                "build_manifest_hash": item["build_manifest_hash"],
                "dependency_lock_hash": item["dependency_lock_hash"],
                "verified_build_count": 3,
            }
            for item in (gateway_boot, validator_boot)
        ],
    }
    monkeypatch.setattr(
        compact_auditor,
        "verify_binding_message",
        lambda *_args, **_kwargs: True,
    )
    audit = verify_compact_published_weight_authority_v2(
        public_authority,
        identity_cache=identity_cache,
        chain_signing_profile={},
        expected_lineage_id=lineage_id,
        expected_chain="wss://entrypoint-finney.opentensor.ai:443",
        boot_verifier=lambda identity, **_kwargs: verify_boot(identity),
    )
    assert audit["bundle_hash"] == compact_bundle_hash
    assert audit["uids"] == verified["uids"]
    assert audit["weights_u16"] == verified["weights_u16"]
    assert audit["weights_hash"] == verified["weights_hash"]

    # The independently consumed sidecar must bind the separately carried
    # validator bodies to the enclave-signed local-delta projection, rather
    # than trusting service-role persistence to preserve that equivalence.
    validator_receipts = compact["validator_receipt_delta"]["receipts"]
    assert len(validator_receipts) > 1
    receipt_mutations = []
    dropped = deepcopy(compact)
    dropped["validator_receipt_delta"]["receipts"].pop(0)
    receipt_mutations.append(dropped)
    extra = deepcopy(compact)
    extra["validator_receipt_delta"]["receipts"].append(
        deepcopy(extra["binding_receipt"])
    )
    receipt_mutations.append(extra)
    swapped = deepcopy(compact)
    swapped["validator_receipt_delta"]["receipts"][0] = deepcopy(
        swapped["validator_receipt_delta"]["receipts"][1]
    )
    receipt_mutations.append(swapped)
    for mutated in receipt_mutations:
        mutated["compact_submission_hash"] = sha256_json(
            {
                key: value
                for key, value in mutated.items()
                if key != "compact_submission_hash"
            }
        )
        with pytest.raises(
            CompactWeightAuthorityV2Error,
            match="validator receipt delta differs from its ancestry certificate",
        ):
            validate_compact_weight_ancestry_v2(
                mutated,
                expected_lineage_id=lineage_id,
                boot_attestation_verifier=verify_boot,
            )

    tampered = copy.deepcopy(public_authority)
    tampered["compact_submission"]["weight_result"]["sparse_weights_u16"][0] ^= 1
    with pytest.raises(Exception):
        verify_compact_published_weight_authority_v2(
            tampered,
            identity_cache=identity_cache,
            chain_signing_profile={},
            expected_lineage_id=lineage_id,
            expected_chain="wss://entrypoint-finney.opentensor.ai:443",
            boot_verifier=lambda identity, **_kwargs: verify_boot(identity),
        )

    tampered = deepcopy(compact)
    tampered["validator_ancestry_proof"]["certificate"][
        "enclave_signature"
    ] = "0" * 128
    tampered["compact_submission_hash"] = sha256_json(
        {
            key: value
            for key, value in tampered.items()
            if key != "compact_submission_hash"
        }
    )
    with pytest.raises(CompactWeightAuthorityV2Error):
        reconstruct_published_weight_bundle_from_compact_v2(
            tampered,
            expected_lineage_id=lineage_id,
            full_graphs_by_root=full_graphs,
            boot_attestation_verifier=verify_boot,
        )

    omitted = deepcopy(compact)
    omitted["upstream_ancestry_proofs"].pop(
        next(iter(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)))
    )
    with pytest.raises(CompactWeightAuthorityV2Error):
        reconstruct_published_weight_bundle_from_compact_v2(
            omitted,
            expected_lineage_id=lineage_id,
            full_graphs_by_root=full_graphs,
            boot_attestation_verifier=verify_boot,
        )
