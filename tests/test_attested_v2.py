from __future__ import annotations

import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from leadpoet_canonical.attested_v2 import (
    AUTORESEARCH_ROLE,
    COORDINATOR_ROLE,
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    SCORING_ROLE,
    WEIGHT_ROLE,
    AttestedV2Error,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_host_operation_request_body,
    build_host_operation_terminal_body,
    build_receipt_graph,
    build_transition_command_body,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    create_signed_host_operation_request,
    create_signed_host_operation_terminal,
    create_signed_transition_command,
    sha256_bytes,
    validate_receipt_graph,
    validate_signed_transition_command,
    validate_signed_host_operation_request,
    validate_signed_host_operation_terminal,
    validate_transport_attempt,
)


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
COMMIT = "d" * 40
PCR0 = "e" * 96
NOW = "2026-07-10T20:00:00Z"
LATER = "2026-07-10T20:05:00Z"


def _keypair():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    return private_key, public_key


def _boot(role: str, private_key, public_key: str):
    physical_role = {
        COORDINATOR_ROLE: "gateway_coordinator",
        SCORING_ROLE: "gateway_scoring_a",
        AUTORESEARCH_ROLE: "gateway_autoresearch",
        WEIGHT_ROLE: "validator_weights",
    }[role]
    body = build_boot_identity_body(
        role=role,
        physical_role=physical_role,
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_nonce="1" * 32,
        signing_pubkey=public_key,
        transport_pubkey="2" * 64,
        transport_certificate_hash=HASH_B,
        attestation_user_data_hash=HASH,
        issued_at=NOW,
    )
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(b"attestation").decode("ascii"),
    )


def _receipt(
    *,
    role: str,
    purpose: str,
    job_id: str,
    boot,
    private_key,
    public_key: str,
    parents=(),
    transport_root_hash=EMPTY_TRANSPORT_ROOT,
    status="succeeded",
    failure_code=None,
):
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=24000,
        sequence=1,
        commit_sha=COMMIT,
        pcr0=PCR0,
        build_manifest_hash=HASH,
        dependency_lock_hash=HASH_B,
        config_hash=HASH_C,
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=HASH,
        output_root=HASH_B,
        transport_root_hash=transport_root_hash,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
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


def _authenticated_attempt(*, job_id="score-1", purpose="research_lab.company_score.v2", status=200):
    return build_transport_attempt(
        request_id="3" * 32,
        logical_operation_id="provider-call-1",
        job_id=job_id,
        purpose=purpose,
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH_B,
        body_hash=HASH_C,
        credential_ref_hash=HASH,
        retry_policy_hash=HASH_B,
        timeout_ms=900000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=status,
        response_hash=HASH_C,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=LATER,
    )


def test_complete_receipt_graph_validates_terminal_transport():
    auto_key, auto_pub = _keypair()
    score_key, score_pub = _keypair()
    auto_boot = _boot(AUTORESEARCH_ROLE, auto_key, auto_pub)
    score_boot = _boot(SCORING_ROLE, score_key, score_pub)
    source_receipt = _receipt(
        role=AUTORESEARCH_ROLE,
        purpose="research_lab.source_inspection.v2",
        job_id="loop-1",
        boot=auto_boot,
        private_key=auto_key,
        public_key=auto_pub,
    )
    attempt = _authenticated_attempt()
    from leadpoet_canonical.attested_v2 import transport_root

    score_receipt = _receipt(
        role=SCORING_ROLE,
        purpose="research_lab.company_score.v2",
        job_id="score-1",
        boot=score_boot,
        private_key=score_key,
        public_key=score_pub,
        parents=(source_receipt["receipt_hash"],),
        transport_root_hash=transport_root([attempt]),
    )
    graph = build_receipt_graph(
        root_receipt_hash=score_receipt["receipt_hash"],
        boot_identities=[auto_boot, score_boot],
        receipts=[source_receipt, score_receipt],
        transport_attempts=[attempt],
    )
    assert validate_receipt_graph(
        graph,
        required_purposes={
            "research_lab.source_inspection.v2",
            "research_lab.company_score.v2",
        },
    )[-1] == score_receipt["receipt_hash"]


def test_failed_receipt_requires_exact_explicit_graph_authorization():
    key, pub = _keypair()
    boot = _boot(AUTORESEARCH_ROLE, key, pub)
    receipt = _receipt(
        role=AUTORESEARCH_ROLE,
        purpose="research_lab.candidate_decision.v2",
        job_id="failed-loop-1",
        boot=boot,
        private_key=key,
        public_key=pub,
        status="failed",
        failure_code="execution_runtimeerror",
    )
    with pytest.raises(AttestedV2Error, match="unauthorized failed"):
        build_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=[boot],
            receipts=[receipt],
            transport_attempts=[],
        )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=[boot],
        receipts=[receipt],
        transport_attempts=[],
        allowed_failed_receipt_hashes=(receipt["receipt_hash"],),
    )
    assert validate_receipt_graph(
        graph,
        allowed_failed_receipt_hashes=(receipt["receipt_hash"],),
    ) == (receipt["receipt_hash"],)
    with pytest.raises(AttestedV2Error, match="unauthorized failed"):
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=(HASH_C,),
        )


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_authenticated_provider_errors_remain_authenticated_responses(status):
    attempt = _authenticated_attempt(status=status)
    validate_transport_attempt(attempt)
    assert attempt["terminal_status"] == "authenticated_response"
    assert attempt["http_status"] == status
    assert attempt["failure_code"] is None


def test_host_failure_cannot_masquerade_as_provider_http_error():
    kwargs = dict(
        request_id="4" * 32,
        logical_operation_id="provider-call-2",
        job_id="score-2",
        purpose="research_lab.company_score.v2",
        provider_id="exa",
        attempt_number=0,
        method="POST",
        destination_host="api.exa.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH,
        credential_ref_hash=HASH,
        retry_policy_hash=HASH,
        timeout_ms=30000,
        started_at=NOW,
        terminal_status="transport_failure",
        http_status=502,
        response_hash=None,
        request_artifact_hash=HASH,
        response_artifact_hash=None,
        tls_peer_chain_hash=None,
        tls_protocol=None,
        failure_code="proxy_failure",
        completed_at=LATER,
    )
    with pytest.raises(AttestedV2Error, match="cannot claim an HTTP status"):
        build_transport_attempt(**kwargs)


def test_external_plaintext_port_is_rejected():
    kwargs = dict(
        request_id="5" * 32,
        logical_operation_id="provider-call-3",
        job_id="score-3",
        purpose="research_lab.company_score.v2",
        provider_id="wayback",
        attempt_number=0,
        method="GET",
        destination_host="archive.org",
        destination_port=80,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=sha256_bytes(b""),
        credential_ref_hash=HASH,
        retry_policy_hash=HASH,
        timeout_ms=30000,
        started_at=NOW,
        terminal_status="transport_failure",
        http_status=None,
        response_hash=None,
        request_artifact_hash=HASH,
        response_artifact_hash=None,
        tls_peer_chain_hash=None,
        tls_protocol=None,
        failure_code="plaintext_forbidden",
        completed_at=LATER,
    )
    with pytest.raises(AttestedV2Error, match="requires port 443"):
        build_transport_attempt(**kwargs)


def test_graph_rejects_missing_terminal_transport_record():
    key, pub = _keypair()
    boot = _boot(SCORING_ROLE, key, pub)
    attempt = _authenticated_attempt()
    from leadpoet_canonical.attested_v2 import transport_root

    receipt = _receipt(
        role=SCORING_ROLE,
        purpose="research_lab.company_score.v2",
        job_id="score-1",
        boot=boot,
        private_key=key,
        public_key=pub,
        transport_root_hash=transport_root([attempt]),
    )
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": receipt["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    with pytest.raises(AttestedV2Error, match="transport root"):
        validate_receipt_graph(graph)


def test_graph_rejects_wrong_boot_signing_key():
    key, pub = _keypair()
    other_key, other_pub = _keypair()
    boot = _boot(SCORING_ROLE, key, pub)
    receipt = _receipt(
        role=SCORING_ROLE,
        purpose="research_lab.company_score.v2",
        job_id="score-1",
        boot=boot,
        private_key=other_key,
        public_key=other_pub,
    )
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": receipt["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    with pytest.raises(AttestedV2Error, match="attested boot key"):
        validate_receipt_graph(graph)


def test_graph_rejects_disconnected_valid_receipt():
    key, pub = _keypair()
    boot = _boot(COORDINATOR_ROLE, key, pub)
    first = _receipt(
        role=COORDINATOR_ROLE,
        purpose="research_lab.admission.v2",
        job_id="run-1",
        boot=boot,
        private_key=key,
        public_key=pub,
    )
    second = _receipt(
        role=COORDINATOR_ROLE,
        purpose="research_lab.allocation.v2",
        job_id="run-2",
        boot=boot,
        private_key=key,
        public_key=pub,
    )
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": first["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [first, second],
        "transport_attempts": [],
        "host_operations": [],
    }
    with pytest.raises(AttestedV2Error, match="disconnected"):
        validate_receipt_graph(graph)


def test_tampered_receipt_signature_is_rejected():
    key, pub = _keypair()
    boot = _boot(COORDINATOR_ROLE, key, pub)
    receipt = _receipt(
        role=COORDINATOR_ROLE,
        purpose="research_lab.allocation.v2",
        job_id="allocation-1",
        boot=boot,
        private_key=key,
        public_key=pub,
    )
    tampered = deepcopy(receipt)
    tampered["enclave_signature"] = "0" * 128
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": tampered["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [tampered],
        "transport_attempts": [],
        "host_operations": [],
    }
    with pytest.raises(AttestedV2Error, match="signature"):
        validate_receipt_graph(graph)


def test_signed_transition_command_binds_precondition_and_payload():
    key, pub = _keypair()
    body = build_transition_command_body(
        operation="insert",
        target="research_lab_candidate_events",
        idempotency_key="candidate-1-event-5",
        expected_state_hash=HASH,
        payload_hash=HASH_B,
        receipt_hash=HASH_C,
        issued_at=NOW,
        expires_at=LATER,
    )
    command = create_signed_transition_command(
        body=body,
        enclave_pubkey=pub,
        sign_digest=key.sign,
    )
    validate_signed_transition_command(command)
    tampered = dict(command)
    tampered["payload_hash"] = HASH
    with pytest.raises(AttestedV2Error, match="command_hash"):
        validate_signed_transition_command(tampered)


def test_signed_host_operation_request_and_terminal_bind_exact_exchange():
    key, pub = _keypair()
    request = create_signed_host_operation_request(
        body=build_host_operation_request_body(
            job_id="autoresearch-job-1",
            purpose="research_lab.candidate_build.v2",
            operation="build_candidate_image",
            sequence=0,
            payload_hash=HASH,
            expected_state_hash=HASH_B,
            boot_identity_hash=HASH_C,
            request_nonce="6" * 32,
            issued_at=NOW,
            expires_at=LATER,
        ),
        enclave_pubkey=pub,
        sign_digest=key.sign,
    )
    validate_signed_host_operation_request(request)
    terminal = create_signed_host_operation_terminal(
        body=build_host_operation_terminal_body(
            request_hash=request["request_hash"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            operation=request["operation"],
            sequence=request["sequence"],
            terminal_status="succeeded",
            response_hash=HASH_C,
            failure_code=None,
            completed_at=LATER,
        ),
        enclave_pubkey=pub,
        sign_digest=key.sign,
    )
    validate_signed_host_operation_terminal(terminal)

    tampered = {**terminal, "response_hash": HASH}
    with pytest.raises(AttestedV2Error, match="terminal hash"):
        validate_signed_host_operation_terminal(tampered)


def test_failed_host_operation_cannot_claim_success_without_response_hash():
    with pytest.raises(AttestedV2Error, match="response_hash"):
        build_host_operation_terminal_body(
            request_hash=HASH,
            job_id="autoresearch-job-1",
            purpose="research_lab.candidate_build.v2",
            operation="build_candidate_image",
            sequence=0,
            terminal_status="succeeded",
            response_hash=None,
            failure_code=None,
            completed_at=LATER,
        )
