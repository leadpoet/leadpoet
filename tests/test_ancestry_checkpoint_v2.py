from __future__ import annotations

import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    MAX_DELTA_RECEIPTS,
    MAX_PARENT_AUTHORITIES,
    AncestryCheckpointV2Error,
    build_ancestry_policy_v2,
    build_certificate_parent_authority_v2,
    build_compact_ancestry_proof_from_delta_v2,
    build_compact_ancestry_proof_v2,
    build_full_graph_parent_v2,
    build_full_graph_parent_authority_v2,
    derive_ancestry_lineage_id_v2,
    issue_ancestry_certificate_v2,
    project_receipt_graph_v2,
    validate_ancestry_certificate_v2,
    validate_ancestry_delta_v2,
    validate_ancestry_lineage_id_v2,
    validate_ancestry_projection_v2,
    validate_compact_ancestry_proof_v2,
    validate_full_graph_parent_v2,
    validate_local_delta_against_certificate_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    ROLE_PURPOSES,
    SCORING_ROLE,
    build_boot_identity_body,
    build_checkpointed_receipt_graph,
    build_execution_receipt_body,
    build_host_operation_request_body,
    build_host_operation_terminal_body,
    build_receipt_graph,
    build_transport_attempt,
    canonical_json,
    create_boot_identity,
    create_signed_execution_receipt,
    create_signed_host_operation_request,
    create_signed_host_operation_terminal,
    host_operation_root,
    sha256_bytes,
    transport_root,
)


COMMIT = "d" * 40
PCR0 = "e" * 96
NOW = "2026-07-31T12:00:00Z"
LATER = "2026-07-31T12:01:00Z"
CUTOVER_HASH = "sha256:" + "a" * 64
GENESIS_HASH = "0x" + "b" * 64
HASH_A = "sha256:" + "1" * 64
HASH_B = "sha256:" + "2" * 64
HASH_C = "sha256:" + "3" * 64
HASH_D = "sha256:" + "4" * 64
LINEAGE_ID = derive_ancestry_lineage_id_v2(
    cutover_mapping_hash=CUTOVER_HASH,
    network_genesis_hash=GENESIS_HASH,
    netuid=71,
)


def _keypair():
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key().public_bytes_raw().hex()


def _boot(private_key, public_key, *, nonce, pcr0=PCR0):
    del private_key
    body = build_boot_identity_body(
        role=SCORING_ROLE,
        physical_role="gateway_scoring",
        commit_sha=COMMIT,
        pcr0=pcr0,
        build_manifest_hash=HASH_A,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_nonce=nonce,
        signing_pubkey=public_key,
        transport_pubkey="5" * 64,
        transport_certificate_hash=HASH_D,
        attestation_user_data_hash=HASH_A,
        issued_at=NOW,
    )
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(b"nitro-attestation").decode(
            "ascii"
        ),
    )


def _receipt(
    *,
    private_key,
    public_key,
    boot,
    purpose,
    job_id,
    parents=(),
    epoch_id=24_300,
    sequence=1,
    attempt_root=EMPTY_TRANSPORT_ROOT,
    host_root=EMPTY_HOST_OPERATION_ROOT,
    status="succeeded",
    failure_code=None,
):
    body = build_execution_receipt_body(
        role=SCORING_ROLE,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=sequence,
        commit_sha=COMMIT,
        pcr0=boot["pcr0"],
        build_manifest_hash=HASH_A,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=HASH_A,
        output_root=HASH_B,
        transport_root_hash=attempt_root,
        host_operation_root_hash=host_root,
        artifact_root=EMPTY_ARTIFACT_ROOT,
        parent_receipt_hashes=parents,
        status=status,
        failure_code=failure_code,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _attempt(*, job_id, purpose):
    return build_transport_attempt(
        request_id="6" * 32,
        logical_operation_id="provider-call-1",
        job_id=job_id,
        purpose=purpose,
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=HASH_A,
        nonsecret_headers_hash=HASH_B,
        body_hash=HASH_C,
        credential_ref_hash=HASH_D,
        retry_policy_hash=HASH_A,
        timeout_ms=900_000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH_B,
        request_artifact_hash=HASH_C,
        response_artifact_hash=HASH_D,
        tls_peer_chain_hash=HASH_A,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=LATER,
    )


def _host_record(*, private_key, public_key, boot, job_id, purpose):
    request_body = build_host_operation_request_body(
        job_id=job_id,
        purpose=purpose,
        operation="persist_result",
        sequence=0,
        payload_hash=HASH_A,
        expected_state_hash=HASH_B,
        boot_identity_hash=boot["boot_identity_hash"],
        request_nonce="a" * 32,
        issued_at=NOW,
        expires_at="2026-07-31T12:02:00Z",
    )
    request = create_signed_host_operation_request(
        body=request_body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    terminal_body = build_host_operation_terminal_body(
        request_hash=request["request_hash"],
        job_id=job_id,
        purpose=purpose,
        operation="persist_result",
        sequence=0,
        terminal_status="succeeded",
        response_hash=HASH_C,
        failure_code=None,
        completed_at=LATER,
    )
    terminal = create_signed_host_operation_terminal(
        body=terminal_body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    return {"request": request, "terminal": terminal}


def _delta(*, root, receipts, boots, attempts=(), hosts=()):
    return {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": root["receipt_hash"],
        "boot_identities": [dict(item) for item in boots],
        "receipts": [dict(item) for item in receipts],
        "transport_attempts": [dict(item) for item in attempts],
        "host_operations": [dict(item) for item in hosts],
    }


def _boot_verifier(identity):
    if identity["commit_sha"] != COMMIT or identity["pcr0"] != PCR0:
        raise ValueError("release/PCR0 is not approved")
    return identity


def _trusted_parents(certificate):
    return {
        item["parent_receipt_hash"]: item["authority_hash"]
        for item in certificate["claim"]["parent_authorities"]
    }


@pytest.fixture
def ancestry_fixture():
    legacy_key, legacy_pub = _keypair()
    current_key, current_pub = _keypair()
    legacy_boot = _boot(legacy_key, legacy_pub, nonce="7" * 32)
    current_boot = _boot(current_key, current_pub, nonce="8" * 32)
    legacy = _receipt(
        private_key=legacy_key,
        public_key=legacy_pub,
        boot=legacy_boot,
        purpose="research_lab.baseline_score.v2",
        job_id="legacy-baseline",
        epoch_id=24_299,
    )
    legacy_graph = build_receipt_graph(
        root_receipt_hash=legacy["receipt_hash"],
        boot_identities=[legacy_boot],
        receipts=[legacy],
        transport_attempts=[],
    )
    attempt = _attempt(
        job_id="local-candidate", purpose="research_lab.candidate_score.v2"
    )
    host = _host_record(
        private_key=current_key,
        public_key=current_pub,
        boot=current_boot,
        job_id="local-company",
        purpose="research_lab.company_score.v2",
    )
    intermediate = _receipt(
        private_key=current_key,
        public_key=current_pub,
        boot=current_boot,
        purpose="research_lab.candidate_score.v2",
        job_id="local-candidate",
        parents=(legacy["receipt_hash"],),
        attempt_root=transport_root([attempt]),
    )
    output = _receipt(
        private_key=current_key,
        public_key=current_pub,
        boot=current_boot,
        purpose="research_lab.company_score.v2",
        job_id="local-company",
        parents=(intermediate["receipt_hash"],),
        sequence=2,
        host_root=host_operation_root([host]),
    )
    delta = _delta(
        root=output,
        receipts=[output, intermediate],
        boots=[current_boot],
        attempts=[attempt],
        hosts=[host],
    )
    parent = build_full_graph_parent_v2(
        legacy_graph,
        required_purposes=("research_lab.baseline_score.v2",),
    )
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=LINEAGE_ID,
        certificate_sequence=0,
        issuer_boot_identity=current_boot,
        issued_at=NOW,
        sign_digest=current_key.sign,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
        parent_full_graphs=(parent,),
        required_purposes=(
            "research_lab.candidate_score.v2",
            "research_lab.company_score.v2",
        ),
    )
    combined_graph = build_receipt_graph(
        root_receipt_hash=output["receipt_hash"],
        boot_identities=[current_boot, legacy_boot],
        receipts=[output, legacy, intermediate],
        transport_attempts=[attempt],
        host_operations=[host],
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
    )
    return {
        "legacy_key": legacy_key,
        "legacy_pub": legacy_pub,
        "legacy_boot": legacy_boot,
        "legacy": legacy,
        "legacy_graph": legacy_graph,
        "current_key": current_key,
        "current_pub": current_pub,
        "current_boot": current_boot,
        "intermediate": intermediate,
        "output": output,
        "attempt": attempt,
        "host": host,
        "delta": delta,
        "parent": parent,
        "certificate": certificate,
        "combined_graph": combined_graph,
        "proof": proof,
    }


def test_stable_lineage_uses_cutover_identity_not_release_identity():
    assert (
        validate_ancestry_lineage_id_v2(
            LINEAGE_ID,
            cutover_mapping_hash=CUTOVER_HASH,
            network_genesis_hash=GENESIS_HASH,
            netuid=71,
        )
        == LINEAGE_ID
    )
    assert "release" not in derive_ancestry_lineage_id_v2.__annotations__
    with pytest.raises(AncestryCheckpointV2Error, match="genesis hash"):
        derive_ancestry_lineage_id_v2(
            cutover_mapping_hash=CUTOVER_HASH,
            network_genesis_hash=GENESIS_HASH.upper(),
            netuid=71,
        )
    with pytest.raises(AncestryCheckpointV2Error, match="differs"):
        validate_ancestry_lineage_id_v2(
            LINEAGE_ID,
            cutover_mapping_hash=CUTOVER_HASH,
            network_genesis_hash=GENESIS_HASH,
            netuid=72,
        )


def test_projection_is_deterministic_and_does_not_mutate_full_graph(
    ancestry_fixture,
):
    graph = ancestry_fixture["combined_graph"]
    before = deepcopy(graph)
    projection = project_receipt_graph_v2(
        graph, boot_attestation_verifier=_boot_verifier
    )
    reordered = deepcopy(graph)
    for field in (
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    ):
        reordered[field].reverse()
    assert (
        project_receipt_graph_v2(
            reordered, boot_attestation_verifier=_boot_verifier
        )
        == projection
    )
    assert graph == before
    assert validate_ancestry_projection_v2(projection) == projection


def test_projection_requires_exact_attestation_and_rejects_tampered_graph(
    ancestry_fixture,
):
    graph = ancestry_fixture["combined_graph"]
    with pytest.raises(AncestryCheckpointV2Error, match="boot verifier"):
        project_receipt_graph_v2(graph, boot_attestation_verifier=None)
    tampered = deepcopy(graph)
    tampered["transport_attempts"][0]["response_hash"] = HASH_C
    with pytest.raises(AncestryCheckpointV2Error, match="attested validation"):
        project_receipt_graph_v2(
            tampered, boot_attestation_verifier=_boot_verifier
        )
    disconnected = deepcopy(graph)
    disconnected["root_receipt_hash"] = ancestry_fixture["legacy"]["receipt_hash"]
    with pytest.raises(AncestryCheckpointV2Error, match="attested validation"):
        project_receipt_graph_v2(
            disconnected, boot_attestation_verifier=_boot_verifier
        )


def test_detached_certificate_and_local_body_proof_round_trip(ancestry_fixture):
    certificate = ancestry_fixture["certificate"]
    assert certificate["schema_version"].endswith("ancestry_certificate.v2")
    assert certificate["claim"]["output_root_receipt_hash"] == ancestry_fixture[
        "output"
    ]["receipt_hash"]
    assert "research_lab.ancestry" not in {
        purpose for purposes in ROLE_PURPOSES.values() for purpose in purposes
    }
    parent_descriptor = build_full_graph_parent_authority_v2(
        ancestry_fixture["parent"],
        boot_attestation_verifier=_boot_verifier,
    )
    assert parent_descriptor == certificate["claim"]["parent_authorities"][0]
    assert (
        validate_full_graph_parent_v2(
            ancestry_fixture["parent"],
            boot_attestation_verifier=_boot_verifier,
        )
        == ancestry_fixture["parent"]
    )
    projection = validate_local_delta_against_certificate_v2(
        ancestry_fixture["delta"],
        certificate,
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
        trusted_parent_authorities=_trusted_parents(certificate),
    )
    assert projection == certificate["claim"]["local_delta_projection"]
    assert projection["transport_attempt_count"] == 1
    assert projection["host_operation_count"] == 1
    proof = ancestry_fixture["proof"]
    assert "transport_attempts" not in proof
    assert "host_operations" not in proof
    assert (
        validate_compact_ancestry_proof_v2(
            proof,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
            trusted_parent_authorities=_trusted_parents(certificate),
            required_receipt_hashes=(ancestry_fixture["output"]["receipt_hash"],),
            required_purposes=("research_lab.company_score.v2",),
        )["proof_hash"]
        == proof["proof_hash"]
    )


def test_full_graph_builder_matches_bounded_delta_builder_without_mutation(
    ancestry_fixture,
):
    graph = ancestry_fixture["combined_graph"]
    before = deepcopy(graph)
    proof = build_compact_ancestry_proof_v2(
        graph,
        ancestry_fixture["certificate"],
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
    )
    assert proof == ancestry_fixture["proof"]
    assert graph == before


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["claim"]["local_delta_projection"].update(
            {"evidence_commitment": HASH_D}
        ),
        lambda value: value["claim"]["parent_authorities"][0].update(
            {"authority_hash": HASH_D}
        ),
        lambda value: value.update({"enclave_signature": "0" * 128}),
        lambda value: value.update({"unexpected": True}),
    ),
)
def test_certificate_tamper_and_extra_fields_fail_closed(
    ancestry_fixture, mutator
):
    certificate = deepcopy(ancestry_fixture["certificate"])
    mutator(certificate)
    with pytest.raises(AncestryCheckpointV2Error):
        validate_ancestry_certificate_v2(
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )


def test_local_attempt_omission_tamper_and_parent_boundary_fail_closed(
    ancestry_fixture,
):
    certificate = ancestry_fixture["certificate"]
    omitted = deepcopy(ancestry_fixture["delta"])
    omitted["transport_attempts"] = []
    with pytest.raises(AncestryCheckpointV2Error, match="transport root"):
        validate_local_delta_against_certificate_v2(
            omitted,
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
    tampered = deepcopy(ancestry_fixture["delta"])
    tampered["transport_attempts"][0]["response_hash"] = HASH_C
    with pytest.raises(AncestryCheckpointV2Error, match="transport attempt"):
        validate_local_delta_against_certificate_v2(
            tampered,
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
    omitted_host = deepcopy(ancestry_fixture["delta"])
    omitted_host["host_operations"] = []
    with pytest.raises(AncestryCheckpointV2Error, match="host root"):
        validate_local_delta_against_certificate_v2(
            omitted_host,
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
    tampered_host = deepcopy(ancestry_fixture["delta"])
    tampered_host["host_operations"][0]["terminal"]["response_hash"] = HASH_D
    with pytest.raises(AncestryCheckpointV2Error, match="host operation"):
        validate_local_delta_against_certificate_v2(
            tampered_host,
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
    with pytest.raises(AncestryCheckpointV2Error, match="parent authority"):
        validate_ancestry_delta_v2(
            ancestry_fixture["delta"],
            parent_authorities=(),
            policy=certificate["claim"]["policy"],
            boot_attestation_verifier=_boot_verifier,
        )


def test_proof_omission_reordering_body_tamper_and_hash_tamper_fail_closed(
    ancestry_fixture,
):
    base = ancestry_fixture["proof"]
    variants = []
    omitted = deepcopy(base)
    omitted["disclosed_receipts"].pop()
    variants.append(omitted)
    reordered = deepcopy(base)
    reordered["disclosed_receipts"].reverse()
    variants.append(reordered)
    body_tampered = deepcopy(base)
    body_tampered["disclosed_receipts"][0]["parent_receipt_hashes"] = []
    variants.append(body_tampered)
    hash_tampered = deepcopy(base)
    hash_tampered["proof_hash"] = HASH_D
    variants.append(hash_tampered)
    extra = deepcopy(base)
    extra["unexpected"] = True
    variants.append(extra)
    for proof in variants:
        with pytest.raises(AncestryCheckpointV2Error):
            validate_compact_ancestry_proof_v2(
                proof,
                expected_lineage_id=LINEAGE_ID,
                boot_attestation_verifier=_boot_verifier,
                allowed_issuer_roles=(SCORING_ROLE,),
            )


def test_signed_failed_policy_is_exact_in_delta_and_disclosure(ancestry_fixture):
    policy = build_ancestry_policy_v2(
        allowed_failed_receipt_hashes=(
            ancestry_fixture["intermediate"]["receipt_hash"],
        )
    )
    with pytest.raises(AncestryCheckpointV2Error, match="failed receipt policy"):
        validate_ancestry_delta_v2(
            ancestry_fixture["delta"],
            parent_authorities=ancestry_fixture["certificate"]["claim"][
                "parent_authorities"
            ],
            policy=policy,
            boot_attestation_verifier=_boot_verifier,
        )


def test_recursive_certificate_proof_and_sequence_are_bounded(ancestry_fixture):
    key = ancestry_fixture["current_key"]
    pub = ancestry_fixture["current_pub"]
    boot = ancestry_fixture["current_boot"]
    successor = _receipt(
        private_key=key,
        public_key=pub,
        boot=boot,
        purpose="research_lab.confirmation_score.v2",
        job_id="next-confirmation",
        parents=(ancestry_fixture["output"]["receipt_hash"],),
        epoch_id=24_301,
        sequence=3,
    )
    delta = _delta(root=successor, receipts=[successor], boots=[boot])
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=LINEAGE_ID,
        certificate_sequence=1,
        issuer_boot_identity=boot,
        issued_at=LATER,
        sign_digest=key.sign,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
        parent_certificates=(ancestry_fixture["certificate"],),
        required_purposes=("research_lab.confirmation_score.v2",),
    )
    assert build_certificate_parent_authority_v2(
        ancestry_fixture["certificate"],
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
    ) == certificate["claim"]["parent_authorities"][0]
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
    )
    assert len(proof["disclosed_receipts"]) == 1
    assert ancestry_fixture["intermediate"]["receipt_hash"] not in canonical_json(
        proof
    )
    assert validate_compact_ancestry_proof_v2(
        proof,
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
        trusted_parent_authorities=_trusted_parents(certificate),
        minimum_certificate_sequence=1,
    )["certificate"]["claim"]["certificate_sequence"] == 1
    with pytest.raises(AncestryCheckpointV2Error, match="does not extend"):
        issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=LINEAGE_ID,
            certificate_sequence=2,
            issuer_boot_identity=boot,
            issued_at=LATER,
            sign_digest=key.sign,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
            parent_certificates=(ancestry_fixture["certificate"],),
        )


def test_one_hundred_recursive_checkpoints_remain_constant_size(ancestry_fixture):
    key = ancestry_fixture["current_key"]
    boot = ancestry_fixture["current_boot"]
    parent_certificate = ancestry_fixture["certificate"]
    parent_root = ancestry_fixture["output"]["receipt_hash"]
    encoded_sizes = []
    checkpointed_graph_sizes = []

    for offset in range(1, 101):
        receipt = _receipt(
            private_key=key,
            public_key=ancestry_fixture["current_pub"],
            boot=boot,
            purpose="research_lab.confirmation_score.v2",
            job_id="recursive-confirmation-%03d" % offset,
            parents=(parent_root,),
            epoch_id=24_300 + offset,
            sequence=2 + offset,
        )
        delta = _delta(root=receipt, receipts=[receipt], boots=[boot])
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=LINEAGE_ID,
            certificate_sequence=offset,
            issuer_boot_identity=boot,
            issued_at=LATER,
            sign_digest=key.sign,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
            parent_certificates=(parent_certificate,),
            required_purposes=("research_lab.confirmation_score.v2",),
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=LINEAGE_ID,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
        assert len(proof["disclosed_receipts"]) == 1
        assert len(proof["disclosed_boot_identities"]) == 1
        assert parent_certificate["certificate_hash"] not in canonical_json(
            proof["certificate"]["claim"]["local_delta_projection"]
        )
        encoded_sizes.append(len(canonical_json(proof).encode("utf-8")))
        checkpointed_graph = build_checkpointed_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(receipt,),
            transport_attempts=(),
            host_operations=(),
            ancestry_lineage_id=LINEAGE_ID,
            ancestry_proof=proof,
            boot_attestation_verifier=_boot_verifier,
            require_boot_attestation_verification=True,
        )
        checkpointed_graph_sizes.append(
            len(canonical_json(checkpointed_graph).encode("utf-8"))
        )
        assert len(checkpointed_graph["receipts"]) == 1
        assert parent_root not in {
            item["receipt_hash"] for item in checkpointed_graph["receipts"]
        }
        parent_certificate = certificate
        parent_root = receipt["receipt_hash"]

    # The first child adds one previously unseen purpose to the bounded
    # ancestry-purpose set. After that one-time addition, decimal sequence
    # growth may add bytes but no prior certificate or body is embedded.
    assert max(encoded_sizes) < 9_000
    assert max(encoded_sizes[1:]) - min(encoded_sizes[1:]) < 32
    assert max(checkpointed_graph_sizes) < 12_000
    assert (
        max(checkpointed_graph_sizes[1:])
        - min(checkpointed_graph_sizes[1:])
        < 64
    )


def test_fork_rewind_lineage_and_parent_substitution_fail_closed(
    ancestry_fixture,
):
    certificate = ancestry_fixture["certificate"]
    validation_args = {
        "expected_lineage_id": LINEAGE_ID,
        "boot_attestation_verifier": _boot_verifier,
        "allowed_issuer_roles": (SCORING_ROLE,),
    }
    with pytest.raises(AncestryCheckpointV2Error, match="known fork"):
        validate_ancestry_certificate_v2(
            certificate,
            known_certificate_hashes_by_root={
                certificate["claim"]["output_root_receipt_hash"]: HASH_D
            },
            **validation_args
        )
    with pytest.raises(AncestryCheckpointV2Error, match="rewinds"):
        validate_ancestry_certificate_v2(
            certificate, minimum_certificate_sequence=1, **validation_args
        )
    with pytest.raises(AncestryCheckpointV2Error, match="lineage differs"):
        validate_ancestry_certificate_v2(
            certificate,
            expected_lineage_id=sha256_bytes(b"different-cutover"),
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
        )
    with pytest.raises(AncestryCheckpointV2Error, match="trusted state"):
        validate_ancestry_certificate_v2(
            certificate,
            trusted_parent_authorities={
                ancestry_fixture["legacy"]["receipt_hash"]: HASH_D
            },
            **validation_args
        )


def test_legacy_failed_parent_policy_is_local_and_cryptographically_bound():
    key, pub = _keypair()
    boot = _boot(key, pub, nonce="9" * 32)
    failed = _receipt(
        private_key=key,
        public_key=pub,
        boot=boot,
        purpose="research_lab.candidate_score.v2",
        job_id="legacy-failed",
        status="failed",
        failure_code="execution_runtimeerror",
    )
    graph = build_receipt_graph(
        root_receipt_hash=failed["receipt_hash"],
        boot_identities=[boot],
        receipts=[failed],
        transport_attempts=[],
        allowed_failed_receipt_hashes=(failed["receipt_hash"],),
    )
    accepted_parent = build_full_graph_parent_v2(
        graph,
        allowed_failed_receipt_hashes=(failed["receipt_hash"],),
    )
    rejected_parent = build_full_graph_parent_v2(graph)
    child = _receipt(
        private_key=key,
        public_key=pub,
        boot=boot,
        purpose="research_lab.confirmation_score.v2",
        job_id="post-failure-confirmation",
        parents=(failed["receipt_hash"],),
        epoch_id=24_301,
    )
    delta = _delta(root=child, receipts=[child], boots=[boot])
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=LINEAGE_ID,
        certificate_sequence=0,
        issuer_boot_identity=boot,
        issued_at=LATER,
        sign_digest=key.sign,
        boot_attestation_verifier=_boot_verifier,
        allowed_issuer_roles=(SCORING_ROLE,),
        parent_full_graphs=(accepted_parent,),
    )
    descriptor = certificate["claim"]["parent_authorities"][0]
    assert descriptor["authority_policy_hash"] == accepted_parent["policy"][
        "policy_hash"
    ]
    with pytest.raises(AncestryCheckpointV2Error, match="attested validation"):
        issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=LINEAGE_ID,
            certificate_sequence=0,
            issuer_boot_identity=boot,
            issued_at=LATER,
            sign_digest=key.sign,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
            parent_full_graphs=(rejected_parent,),
        )


def test_caps_and_canonical_order_are_enforced_before_expensive_validation(
    ancestry_fixture,
):
    too_many_receipts = deepcopy(ancestry_fixture["delta"])
    too_many_receipts["receipts"] = [
        deepcopy(ancestry_fixture["output"])
        for _ in range(MAX_DELTA_RECEIPTS + 1)
    ]
    with pytest.raises(AncestryCheckpointV2Error, match="canonical bound"):
        validate_ancestry_delta_v2(
            too_many_receipts,
            parent_authorities=ancestry_fixture["certificate"]["claim"][
                "parent_authorities"
            ],
            policy=ancestry_fixture["certificate"]["claim"]["policy"],
            boot_attestation_verifier=_boot_verifier,
        )
    with pytest.raises(AncestryCheckpointV2Error, match="authority count"):
        issue_ancestry_certificate_v2(
            local_delta=ancestry_fixture["delta"],
            lineage_id=LINEAGE_ID,
            certificate_sequence=0,
            issuer_boot_identity=ancestry_fixture["current_boot"],
            issued_at=NOW,
            sign_digest=ancestry_fixture["current_key"].sign,
            boot_attestation_verifier=_boot_verifier,
            allowed_issuer_roles=(SCORING_ROLE,),
            parent_full_graphs=tuple(
                ancestry_fixture["parent"]
                for _ in range(MAX_PARENT_AUTHORITIES + 1)
            ),
        )
    projection = deepcopy(
        ancestry_fixture["certificate"]["claim"]["local_delta_projection"]
    )
    projection["receipt_hashes"].reverse()
    with pytest.raises(AncestryCheckpointV2Error, match="not canonical"):
        validate_ancestry_projection_v2(projection)
