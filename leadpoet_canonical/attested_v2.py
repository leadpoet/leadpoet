"""Canonical V2 contracts for attested Research Lab execution.

This module is intentionally compatible with Python 3.7. It contains no
gateway, database, AWS, or chain I/O so the same validation code can run in
gateway enclaves, the validator enclave, auditors, and offline verifiers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BOOT_IDENTITY_SCHEMA_VERSION = "leadpoet.attested_boot_identity.v2"
TRANSPORT_ATTEMPT_SCHEMA_VERSION = "leadpoet.attested_transport_attempt.v2"
EXECUTION_RECEIPT_SCHEMA_VERSION = "leadpoet.attested_execution_receipt.v2"
RECEIPT_GRAPH_SCHEMA_VERSION = "leadpoet.attested_receipt_graph.v2"
TRANSITION_COMMAND_SCHEMA_VERSION = "leadpoet.signed_transition_command.v2"
HOST_OPERATION_REQUEST_SCHEMA_VERSION = "leadpoet.host_operation_request.v2"
HOST_OPERATION_TERMINAL_SCHEMA_VERSION = "leadpoet.host_operation_terminal.v2"
BOOT_ATTESTATION_CLAIM_SCHEMA_VERSION = "leadpoet.attested_boot_claim.v2"
BOOT_ATTESTATION_PURPOSE = "leadpoet.boot_identity.v2"

COORDINATOR_ROLE = "gateway_coordinator"
SCORING_ROLE = "gateway_scoring"
AUTORESEARCH_ROLE = "gateway_autoresearch"
WEIGHT_ROLE = "validator_weights"

ROLE_PURPOSES = {
    COORDINATOR_ROLE: frozenset(
        {
            "research_lab.admission.v2",
            "research_lab.provider_evidence.v2",
            "research_lab.provider_outcome_snapshot.v2",
            "research_lab.provider_outcome_state.v2",
            "research_lab.active_private_model.v2",
            "leadpoet.artifact_persistence.v2",
            "research_lab.ranking.v2",
            "research_lab.promotion_decision.v2",
            "research_lab.reward_decision.v2",
            "research_lab.source_add_provenance.v2",
            "research_lab.source_add_functional_probe.v2",
            "research_lab.source_add_catalog_snapshot.v2",
            "research_lab.source_add_credential.v2",
            "research_lab.openrouter_credential.v2",
            "research_lab.openrouter_credit_preflight.v2",
            "research_lab.allocation.v2",
            "research_lab.champion_input.v2",
            "research_lab.reimbursement_input.v2",
            "research_lab.source_add_reward_input.v2",
            "research_lab.sourcing_input.v2",
            "research_lab.fulfillment_input.v2",
            "research_lab.leaderboard_input.v2",
            "research_lab.ban_input.v2",
            "research_lab.anomaly_adjustment_input.v2",
            "gateway.weights.publication.v2",
        }
    ),
    SCORING_ROLE: frozenset(
        {
            "research_lab.private_model_run.v2",
            "research_lab.candidate_model_run.v2",
            "research_lab.provider_evidence_tape.v2",
            "research_lab.candidate_test.v2",
            "research_lab.company_score.v2",
            "research_lab.provider_preflight.v2",
            "research_lab.candidate_score.v2",
            "research_lab.baseline_score.v2",
            "research_lab.benchmark.v2",
            "research_lab.rebenchmark.v2",
            "research_lab.confirmation_score.v2",
            "research_lab.source_add_judge.v2",
            "qualification.lead_decision.v2",
            "qualification.email_evidence.v2",
            "qualification.sourcing_epoch.v2",
        }
    ),
    AUTORESEARCH_ROLE: frozenset(
        {
            "research_lab.source_inspection.v2",
            "research_lab.research_plan.v2",
            "research_lab.patch_draft.v2",
            "research_lab.patch_validation.v2",
            "research_lab.candidate_test.v2",
            "research_lab.candidate_build.v2",
            "research_lab.candidate_decision.v2",
            "research_lab.stale_parent_repair.v2",
            "research_lab.checkpoint.v2",
            "research_lab.openrouter_guard.v2",
        }
    ),
    WEIGHT_ROLE: frozenset(
        {
            "validator.weight_snapshot.v2",
            "validator.weights.computed.v2",
            "validator.chain_state.v2",
            "validator.metagraph_state.v2",
            "validator.burn_ownership.v2",
            "validator.feature_flags.v2",
            "validator.constants.v2",
            "validator.hotkey_signature.v2",
            "validator.serve_axon_extrinsic.v2",
            "validator.set_weights_extrinsic.v2",
            "validator.weights.finalized.v2",
        }
    ),
}

PHYSICAL_ROLES_BY_SERVICE_ROLE = {
    COORDINATOR_ROLE: frozenset({"gateway_coordinator"}),
    # The A/B roles remain valid only for already-persisted V2 history. New
    # releases use the single shared gateway_scoring role.
    SCORING_ROLE: frozenset(
        {"gateway_scoring", "gateway_scoring_a", "gateway_scoring_b"}
    ),
    AUTORESEARCH_ROLE: frozenset({"gateway_autoresearch"}),
    WEIGHT_ROLE: frozenset({"validator_weights"}),
}

RECEIPT_STATUSES = frozenset({"succeeded", "failed"})
TRANSPORT_TERMINAL_STATUSES = frozenset(
    {"authenticated_response", "attested_local_response", "transport_failure"}
)
TRANSPORT_FAILURE_CODES = frozenset(
    {
        "cancelled",
        "certificate_invalid",
        "connection_refused",
        "connection_reset",
        "dns_failure",
        "host_dropped",
        "malformed_reply",
        "plaintext_forbidden",
        "policy_denied",
        "proxy_failure",
        "tls_failure",
        "timeout",
        "unexpected_eof",
    }
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_PUBKEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32,64}$")
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_METHOD_RE = re.compile(r"^[A-Z]{3,12}$")
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class AttestedV2Error(ValueError):
    """An attested V2 object is missing, non-canonical, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestedV2Error(message)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AttestedV2Error("value is not canonical JSON: %s" % exc) from exc


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


DIRECT_EGRESS_REF_HASH = sha256_json(
    {
        "schema_version": "leadpoet.egress_route.v2",
        "route": "direct",
    }
)


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_HASH_RE.fullmatch(normalized)), "%s must be sha256:<64 lowercase hex>" % field)
    return normalized


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    _require(bool(_IDENTIFIER_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _timestamp(value: Any, field: str) -> str:
    normalized = str(value or "")
    _require(bool(_TIMESTAMP_RE.fullmatch(normalized)), "%s must be RFC3339 UTC seconds" % field)
    return normalized


def _full_commit(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_COMMIT_RE.fullmatch(normalized)), "commit_sha must be a full Git object id")
    return normalized


def _signature_hex(value: Any) -> str:
    normalized = value.hex() if isinstance(value, bytes) else str(value or "").lower()
    _require(bool(_SIGNATURE_RE.fullmatch(normalized)), "signature must be 64-byte lowercase hex")
    return normalized


def _pubkey_hex(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_PUBKEY_RE.fullmatch(normalized)), "%s must be 32-byte lowercase hex" % field)
    return normalized


def _validate_exact_fields(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    _require(set(value) == expected, "%s fields do not match canonical schema" % label)


def _verify_ed25519(pubkey: str, signature: str, digest_hash: str) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(
            bytes.fromhex(signature),
            bytes.fromhex(digest_hash.split(":", 1)[1]),
        )
    except Exception as exc:
        raise AttestedV2Error("invalid Ed25519 signature") from exc


def merkle_root(hashes: Iterable[str], *, domain: str) -> str:
    domain_bytes = _identifier(domain, "merkle domain").encode("ascii")
    normalized = sorted({_hash(item, "merkle item") for item in hashes})
    if not normalized:
        return sha256_bytes(b"\x02" + domain_bytes + b":empty")
    nodes = [
        hashlib.sha256(b"\x00" + domain_bytes + bytes.fromhex(item.split(":", 1)[1])).digest()
        for item in normalized
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + domain_bytes + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return "sha256:" + nodes[0].hex()


EMPTY_TRANSPORT_ROOT = merkle_root((), domain="leadpoet-transport-v2")
EMPTY_ARTIFACT_ROOT = merkle_root((), domain="leadpoet-artifact-v2")
EMPTY_HOST_OPERATION_ROOT = merkle_root((), domain="leadpoet-host-operation-v2")


_BOOT_BODY_FIELDS = {
    "schema_version",
    "role",
    "physical_role",
    "commit_sha",
    "pcr0",
    "build_manifest_hash",
    "dependency_lock_hash",
    "config_hash",
    "boot_nonce",
    "signing_pubkey",
    "transport_pubkey",
    "transport_certificate_hash",
    "attestation_user_data_hash",
    "issued_at",
}
_BOOT_FIELDS = _BOOT_BODY_FIELDS | {
    "boot_identity_hash",
    "attestation_document_b64",
}


def build_boot_identity_body(
    *,
    role: str,
    physical_role: str,
    commit_sha: str,
    pcr0: str,
    build_manifest_hash: str,
    dependency_lock_hash: str,
    config_hash: str,
    boot_nonce: str,
    signing_pubkey: str,
    transport_pubkey: str,
    transport_certificate_hash: str,
    attestation_user_data_hash: str,
    issued_at: str,
) -> Dict[str, Any]:
    role = str(role or "")
    _require(role in ROLE_PURPOSES, "unsupported boot identity role")
    normalized_physical_role = _identifier(physical_role, "physical_role")
    _require(
        normalized_physical_role in PHYSICAL_ROLES_BY_SERVICE_ROLE[role],
        "physical_role is invalid for boot identity role",
    )
    normalized_pcr0 = str(pcr0 or "").strip().lower()
    _require(bool(_PCR0_RE.fullmatch(normalized_pcr0)), "pcr0 must be 48-byte lowercase hex")
    normalized_nonce = str(boot_nonce or "").strip().lower()
    _require(bool(_NONCE_RE.fullmatch(normalized_nonce)), "boot_nonce must be 16-32 bytes lowercase hex")
    return {
        "schema_version": BOOT_IDENTITY_SCHEMA_VERSION,
        "role": role,
        "physical_role": normalized_physical_role,
        "commit_sha": _full_commit(commit_sha),
        "pcr0": normalized_pcr0,
        "build_manifest_hash": _hash(build_manifest_hash, "build_manifest_hash"),
        "dependency_lock_hash": _hash(dependency_lock_hash, "dependency_lock_hash"),
        "config_hash": _hash(config_hash, "config_hash"),
        "boot_nonce": normalized_nonce,
        "signing_pubkey": _pubkey_hex(signing_pubkey, "signing_pubkey"),
        "transport_pubkey": _pubkey_hex(transport_pubkey, "transport_pubkey"),
        "transport_certificate_hash": _hash(
            transport_certificate_hash, "transport_certificate_hash"
        ),
        "attestation_user_data_hash": _hash(
            attestation_user_data_hash, "attestation_user_data_hash"
        ),
        "issued_at": _timestamp(issued_at, "issued_at"),
    }


def create_boot_identity(
    *, body: Mapping[str, Any], attestation_document_b64: str
) -> Dict[str, Any]:
    validate_boot_identity_body(body)
    try:
        base64.b64decode(str(attestation_document_b64 or ""), validate=True)
    except Exception as exc:
        raise AttestedV2Error("attestation_document_b64 is invalid") from exc
    _require(bool(attestation_document_b64), "attestation_document_b64 is required")
    identity_hash = sha256_json(dict(body))
    identity = dict(body)
    identity.update(
        {
            "boot_identity_hash": identity_hash,
            "attestation_document_b64": str(attestation_document_b64),
        }
    )
    validate_boot_identity(identity)
    return identity


def validate_boot_identity_body(body: Mapping[str, Any]) -> None:
    _validate_exact_fields(body, _BOOT_BODY_FIELDS, "boot identity body")
    rebuilt = build_boot_identity_body(
        role=body["role"],
        physical_role=body["physical_role"],
        commit_sha=body["commit_sha"],
        pcr0=body["pcr0"],
        build_manifest_hash=body["build_manifest_hash"],
        dependency_lock_hash=body["dependency_lock_hash"],
        config_hash=body["config_hash"],
        boot_nonce=body["boot_nonce"],
        signing_pubkey=body["signing_pubkey"],
        transport_pubkey=body["transport_pubkey"],
        transport_certificate_hash=body["transport_certificate_hash"],
        attestation_user_data_hash=body["attestation_user_data_hash"],
        issued_at=body["issued_at"],
    )
    _require(dict(body) == rebuilt, "boot identity body is not canonical")


def validate_boot_identity(identity: Mapping[str, Any]) -> None:
    _validate_exact_fields(identity, _BOOT_FIELDS, "boot identity")
    body = {key: identity[key] for key in _BOOT_BODY_FIELDS}
    validate_boot_identity_body(body)
    _require(
        identity["boot_identity_hash"] == sha256_json(body),
        "boot_identity_hash does not match body",
    )
    try:
        decoded = base64.b64decode(str(identity["attestation_document_b64"]), validate=True)
    except Exception as exc:
        raise AttestedV2Error("attestation_document_b64 is invalid") from exc
    _require(bool(decoded), "attestation document is empty")


def build_boot_attestation_claim(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the compact claim committed by Nitro ``user_data``.

    The claim deliberately excludes the attestation document and
    ``attestation_user_data_hash`` so it has no circular dependency. Every
    execution-relevant boot field remains committed by ``claim_hash``.
    """

    fields = (
        "role",
        "physical_role",
        "commit_sha",
        "pcr0",
        "build_manifest_hash",
        "dependency_lock_hash",
        "config_hash",
        "boot_nonce",
        "signing_pubkey",
        "transport_pubkey",
        "transport_certificate_hash",
        "issued_at",
    )
    missing = [field for field in fields if field not in identity]
    _require(not missing, "boot attestation claim fields are missing")
    return {
        "schema_version": BOOT_ATTESTATION_CLAIM_SCHEMA_VERSION,
        **{field: identity[field] for field in fields},
    }


def build_boot_attestation_user_data(identity: Mapping[str, Any]) -> Dict[str, Any]:
    claim = build_boot_attestation_claim(identity)
    return {
        "schema_version": BOOT_ATTESTATION_CLAIM_SCHEMA_VERSION,
        "purpose": BOOT_ATTESTATION_PURPOSE,
        "claim_hash": sha256_json(claim),
        "enclave_pubkey": _pubkey_hex(identity["signing_pubkey"], "signing_pubkey"),
    }


def verify_boot_identity_nitro(
    identity: Mapping[str, Any],
    *,
    expected_pcr0: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify AWS Nitro authenticity and the exact V2 boot claim.

    Callers must supply the dynamically rebuilt PCR0. V2 never consults a
    static allowlist here.
    """

    validate_boot_identity(identity)
    normalized_expected = str(expected_pcr0 or identity["pcr0"]).strip().lower()
    _require(normalized_expected == identity["pcr0"], "boot expected PCR0 mismatch")
    from leadpoet_canonical.nitro import verify_nitro_attestation_full

    valid, extracted = verify_nitro_attestation_full(
        attestation_b64=str(identity["attestation_document_b64"]),
        expected_pcr0=normalized_expected,
        expected_pubkey=str(identity["signing_pubkey"]),
        expected_purpose=BOOT_ATTESTATION_PURPOSE,
        role="gateway" if str(identity["role"]).startswith("gateway_") else "validator",
    )
    _require(bool(valid), "Nitro boot attestation failed: %s" % extracted.get("error", "unknown"))
    expected_user_data = build_boot_attestation_user_data(identity)
    _require(extracted.get("user_data") == expected_user_data, "Nitro boot claim mismatch")
    _require(extracted.get("pcr0") == identity["pcr0"], "Nitro boot PCR0 mismatch")
    _require(
        extracted.get("enclave_pubkey") == identity["signing_pubkey"],
        "Nitro boot signing key mismatch",
    )
    return dict(extracted)


_TRANSPORT_FIELDS = {
    "schema_version",
    "attempt_hash",
    "request_id",
    "logical_operation_id",
    "job_id",
    "purpose",
    "provider_id",
    "attempt_number",
    "method",
    "destination_host",
    "destination_port",
    "path_hash",
    "nonsecret_headers_hash",
    "body_hash",
    "credential_ref_hash",
    "egress_proxy_ref_hash",
    "retry_policy_hash",
    "timeout_ms",
    "request_hash",
    "started_at",
    "terminal_status",
    "http_status",
    "response_hash",
    "request_artifact_hash",
    "response_artifact_hash",
    "tls_peer_chain_hash",
    "tls_protocol",
    "failure_code",
    "completed_at",
}


def build_transport_attempt(
    *,
    request_id: str,
    logical_operation_id: str,
    job_id: str,
    purpose: str,
    provider_id: str,
    attempt_number: int,
    method: str,
    destination_host: str,
    destination_port: int,
    path_hash: str,
    nonsecret_headers_hash: str,
    body_hash: str,
    credential_ref_hash: str,
    retry_policy_hash: str,
    timeout_ms: int,
    started_at: str,
    terminal_status: str,
    http_status: Optional[int],
    response_hash: Optional[str],
    request_artifact_hash: str,
    response_artifact_hash: Optional[str],
    tls_peer_chain_hash: Optional[str],
    tls_protocol: Optional[str],
    failure_code: Optional[str],
    completed_at: str,
    egress_proxy_ref_hash: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_request_id = str(request_id or "").strip().lower()
    _require(bool(_REQUEST_ID_RE.fullmatch(normalized_request_id)), "request_id must be 16-byte lowercase hex")
    normalized_purpose = _identifier(purpose, "purpose")
    _require(
        any(normalized_purpose in purposes for purposes in ROLE_PURPOSES.values()),
        "transport purpose is unsupported",
    )
    _require(isinstance(attempt_number, int) and attempt_number >= 0, "attempt_number must be non-negative")
    normalized_method = str(method or "").strip().upper()
    _require(bool(_METHOD_RE.fullmatch(normalized_method)), "transport method is invalid")
    normalized_host = str(destination_host or "").strip().rstrip(".").lower()
    _require(bool(_HOST_RE.fullmatch(normalized_host)), "destination_host must be a public DNS name")
    _require(destination_port == 443, "external attested transport requires port 443")
    _require(isinstance(timeout_ms, int) and timeout_ms > 0, "timeout_ms must be positive")
    _require(terminal_status in TRANSPORT_TERMINAL_STATUSES, "transport terminal_status is invalid")

    request_descriptor = {
        "request_id": normalized_request_id,
        "logical_operation_id": _identifier(logical_operation_id, "logical_operation_id"),
        "job_id": _identifier(job_id, "job_id"),
        "purpose": normalized_purpose,
        "provider_id": _identifier(provider_id, "provider_id"),
        "attempt_number": attempt_number,
        "method": normalized_method,
        "destination_host": normalized_host,
        "destination_port": destination_port,
        "path_hash": _hash(path_hash, "path_hash"),
        "nonsecret_headers_hash": _hash(nonsecret_headers_hash, "nonsecret_headers_hash"),
        "body_hash": _hash(body_hash, "body_hash"),
        "credential_ref_hash": _hash(credential_ref_hash, "credential_ref_hash"),
        "egress_proxy_ref_hash": _hash(
            egress_proxy_ref_hash or DIRECT_EGRESS_REF_HASH,
            "egress_proxy_ref_hash",
        ),
        "retry_policy_hash": _hash(retry_policy_hash, "retry_policy_hash"),
        "timeout_ms": timeout_ms,
        "started_at": _timestamp(started_at, "started_at"),
    }
    request_hash = sha256_json(request_descriptor)
    normalized_request_artifact_hash = _hash(
        request_artifact_hash,
        "request_artifact_hash",
    )

    if terminal_status in {"authenticated_response", "attested_local_response"}:
        _require(isinstance(http_status, int) and 100 <= http_status <= 599, "authenticated response needs HTTP status")
        normalized_response_hash = _hash(response_hash, "response_hash")
        normalized_artifact_hash = _hash(response_artifact_hash, "response_artifact_hash")
        if terminal_status == "authenticated_response":
            normalized_tls_hash = _hash(tls_peer_chain_hash, "tls_peer_chain_hash")
            normalized_tls_protocol = _identifier(tls_protocol, "tls_protocol")
        else:
            _require(
                tls_peer_chain_hash in (None, "") and tls_protocol in (None, ""),
                "attested local response cannot claim provider TLS",
            )
            normalized_tls_hash = None
            normalized_tls_protocol = None
        _require(failure_code in (None, ""), "authenticated response cannot have failure_code")
        normalized_failure = None
    else:
        _require(http_status is None, "transport failure cannot claim an HTTP status")
        _require(response_hash in (None, ""), "transport failure cannot claim response_hash")
        _require(
            response_artifact_hash in (None, ""),
            "transport failure cannot claim response artifact",
        )
        normalized_response_hash = None
        normalized_artifact_hash = None
        normalized_tls_hash = (
            _hash(tls_peer_chain_hash, "tls_peer_chain_hash")
            if tls_peer_chain_hash not in (None, "")
            else None
        )
        normalized_tls_protocol = (
            _identifier(tls_protocol, "tls_protocol")
            if tls_protocol not in (None, "")
            else None
        )
        normalized_failure = str(failure_code or "")
        _require(
            normalized_failure in TRANSPORT_FAILURE_CODES,
            "transport failure_code is invalid",
        )

    attempt_body = {
        "schema_version": TRANSPORT_ATTEMPT_SCHEMA_VERSION,
        **request_descriptor,
        "request_hash": request_hash,
        "terminal_status": terminal_status,
        "http_status": http_status,
        "response_hash": normalized_response_hash,
        "request_artifact_hash": normalized_request_artifact_hash,
        "response_artifact_hash": normalized_artifact_hash,
        "tls_peer_chain_hash": normalized_tls_hash,
        "tls_protocol": normalized_tls_protocol,
        "failure_code": normalized_failure,
        "completed_at": _timestamp(completed_at, "completed_at"),
    }
    attempt_hash = sha256_json(attempt_body)
    return {**attempt_body, "attempt_hash": attempt_hash}


def validate_transport_attempt(attempt: Mapping[str, Any]) -> None:
    _validate_exact_fields(attempt, _TRANSPORT_FIELDS, "transport attempt")
    rebuilt = build_transport_attempt(
        request_id=attempt["request_id"],
        logical_operation_id=attempt["logical_operation_id"],
        job_id=attempt["job_id"],
        purpose=attempt["purpose"],
        provider_id=attempt["provider_id"],
        attempt_number=attempt["attempt_number"],
        method=attempt["method"],
        destination_host=attempt["destination_host"],
        destination_port=attempt["destination_port"],
        path_hash=attempt["path_hash"],
        nonsecret_headers_hash=attempt["nonsecret_headers_hash"],
        body_hash=attempt["body_hash"],
        credential_ref_hash=attempt["credential_ref_hash"],
        egress_proxy_ref_hash=attempt["egress_proxy_ref_hash"],
        retry_policy_hash=attempt["retry_policy_hash"],
        timeout_ms=attempt["timeout_ms"],
        started_at=attempt["started_at"],
        terminal_status=attempt["terminal_status"],
        http_status=attempt["http_status"],
        response_hash=attempt["response_hash"],
        request_artifact_hash=attempt["request_artifact_hash"],
        response_artifact_hash=attempt["response_artifact_hash"],
        tls_peer_chain_hash=attempt["tls_peer_chain_hash"],
        tls_protocol=attempt["tls_protocol"],
        failure_code=attempt["failure_code"],
        completed_at=attempt["completed_at"],
    )
    _require(dict(attempt) == rebuilt, "transport attempt is not canonical")


def transport_root(attempts: Sequence[Mapping[str, Any]]) -> str:
    hashes = []
    for attempt in attempts:
        validate_transport_attempt(attempt)
        hashes.append(str(attempt["attempt_hash"]))
    return merkle_root(hashes, domain="leadpoet-transport-v2")


_RECEIPT_BODY_FIELDS = {
    "schema_version",
    "role",
    "purpose",
    "job_id",
    "epoch_id",
    "sequence",
    "commit_sha",
    "pcr0",
    "build_manifest_hash",
    "dependency_lock_hash",
    "config_hash",
    "boot_identity_hash",
    "input_root",
    "output_root",
    "transport_root",
    "host_operation_root",
    "artifact_root",
    "parent_receipt_hashes",
    "status",
    "failure_code",
    "issued_at",
}
_SIGNED_RECEIPT_FIELDS = _RECEIPT_BODY_FIELDS | {
    "receipt_hash",
    "enclave_pubkey",
    "enclave_signature",
}


def build_execution_receipt_body(
    *,
    role: str,
    purpose: str,
    job_id: str,
    epoch_id: int,
    sequence: int,
    commit_sha: str,
    pcr0: str,
    build_manifest_hash: str,
    dependency_lock_hash: str,
    config_hash: str,
    boot_identity_hash: str,
    input_root: str,
    output_root: str,
    transport_root_hash: str,
    host_operation_root_hash: str,
    artifact_root: str,
    parent_receipt_hashes: Iterable[str],
    status: str,
    failure_code: Optional[str],
    issued_at: str,
) -> Dict[str, Any]:
    normalized_role = str(role or "")
    normalized_purpose = str(purpose or "")
    _require(normalized_role in ROLE_PURPOSES, "unsupported execution role")
    _require(
        normalized_purpose in ROLE_PURPOSES[normalized_role],
        "purpose is invalid for execution role",
    )
    _require(isinstance(epoch_id, int) and epoch_id >= 0, "epoch_id must be non-negative")
    _require(isinstance(sequence, int) and sequence >= 0, "sequence must be non-negative")
    normalized_pcr0 = str(pcr0 or "").strip().lower()
    _require(bool(_PCR0_RE.fullmatch(normalized_pcr0)), "pcr0 must be 48-byte lowercase hex")
    _require(status in RECEIPT_STATUSES, "execution receipt status is invalid")
    if status == "failed":
        normalized_failure = _identifier(failure_code, "failure_code")
    else:
        _require(failure_code in (None, ""), "successful receipt cannot have failure_code")
        normalized_failure = None
    return {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "role": normalized_role,
        "purpose": normalized_purpose,
        "job_id": _identifier(job_id, "job_id"),
        "epoch_id": epoch_id,
        "sequence": sequence,
        "commit_sha": _full_commit(commit_sha),
        "pcr0": normalized_pcr0,
        "build_manifest_hash": _hash(build_manifest_hash, "build_manifest_hash"),
        "dependency_lock_hash": _hash(dependency_lock_hash, "dependency_lock_hash"),
        "config_hash": _hash(config_hash, "config_hash"),
        "boot_identity_hash": _hash(boot_identity_hash, "boot_identity_hash"),
        "input_root": _hash(input_root, "input_root"),
        "output_root": _hash(output_root, "output_root"),
        "transport_root": _hash(transport_root_hash, "transport_root"),
        "host_operation_root": _hash(
            host_operation_root_hash, "host_operation_root"
        ),
        "artifact_root": _hash(artifact_root, "artifact_root"),
        "parent_receipt_hashes": sorted(
            {_hash(item, "parent_receipt_hash") for item in parent_receipt_hashes}
        ),
        "status": status,
        "failure_code": normalized_failure,
        "issued_at": _timestamp(issued_at, "issued_at"),
    }


def validate_execution_receipt_body(body: Mapping[str, Any]) -> None:
    _validate_exact_fields(body, _RECEIPT_BODY_FIELDS, "execution receipt body")
    rebuilt = build_execution_receipt_body(
        role=body["role"],
        purpose=body["purpose"],
        job_id=body["job_id"],
        epoch_id=body["epoch_id"],
        sequence=body["sequence"],
        commit_sha=body["commit_sha"],
        pcr0=body["pcr0"],
        build_manifest_hash=body["build_manifest_hash"],
        dependency_lock_hash=body["dependency_lock_hash"],
        config_hash=body["config_hash"],
        boot_identity_hash=body["boot_identity_hash"],
        input_root=body["input_root"],
        output_root=body["output_root"],
        transport_root_hash=body["transport_root"],
        host_operation_root_hash=body["host_operation_root"],
        artifact_root=body["artifact_root"],
        parent_receipt_hashes=body["parent_receipt_hashes"],
        status=body["status"],
        failure_code=body["failure_code"],
        issued_at=body["issued_at"],
    )
    _require(dict(body) == rebuilt, "execution receipt body is not canonical")


def create_signed_execution_receipt(
    *,
    body: Mapping[str, Any],
    enclave_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    validate_execution_receipt_body(body)
    digest_hash = sha256_json(dict(body))
    signature = _signature_hex(
        sign_digest(bytes.fromhex(digest_hash.split(":", 1)[1]))
    )
    receipt = dict(body)
    receipt.update(
        {
            "receipt_hash": digest_hash,
            "enclave_pubkey": _pubkey_hex(enclave_pubkey, "enclave_pubkey"),
            "enclave_signature": signature,
        }
    )
    validate_signed_execution_receipt(receipt, verify_signature=False)
    return receipt


def validate_signed_execution_receipt(
    receipt: Mapping[str, Any], *, verify_signature: bool = True
) -> None:
    _validate_exact_fields(receipt, _SIGNED_RECEIPT_FIELDS, "signed execution receipt")
    body = {key: receipt[key] for key in _RECEIPT_BODY_FIELDS}
    validate_execution_receipt_body(body)
    expected_hash = sha256_json(body)
    _require(receipt["receipt_hash"] == expected_hash, "receipt_hash does not match body")
    pubkey = _pubkey_hex(receipt["enclave_pubkey"], "enclave_pubkey")
    signature = _signature_hex(receipt["enclave_signature"])
    if verify_signature:
        _verify_ed25519(pubkey, signature, expected_hash)


_TRANSITION_BODY_FIELDS = {
    "schema_version",
    "operation",
    "target",
    "idempotency_key",
    "expected_state_hash",
    "payload_hash",
    "receipt_hash",
    "issued_at",
    "expires_at",
}
_SIGNED_TRANSITION_FIELDS = _TRANSITION_BODY_FIELDS | {
    "command_hash",
    "enclave_pubkey",
    "enclave_signature",
}


def build_transition_command_body(
    *,
    operation: str,
    target: str,
    idempotency_key: str,
    expected_state_hash: str,
    payload_hash: str,
    receipt_hash: str,
    issued_at: str,
    expires_at: str,
) -> Dict[str, Any]:
    normalized_issued = _timestamp(issued_at, "issued_at")
    normalized_expires = _timestamp(expires_at, "expires_at")
    _require(normalized_expires > normalized_issued, "transition command must expire after issue")
    return {
        "schema_version": TRANSITION_COMMAND_SCHEMA_VERSION,
        "operation": _identifier(operation, "operation"),
        "target": _identifier(target, "target"),
        "idempotency_key": _identifier(idempotency_key, "idempotency_key"),
        "expected_state_hash": _hash(expected_state_hash, "expected_state_hash"),
        "payload_hash": _hash(payload_hash, "payload_hash"),
        "receipt_hash": _hash(receipt_hash, "receipt_hash"),
        "issued_at": normalized_issued,
        "expires_at": normalized_expires,
    }


def create_signed_transition_command(
    *,
    body: Mapping[str, Any],
    enclave_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    validate_transition_command_body(body)
    digest_hash = sha256_json(dict(body))
    command = dict(body)
    command.update(
        {
            "command_hash": digest_hash,
            "enclave_pubkey": _pubkey_hex(enclave_pubkey, "enclave_pubkey"),
            "enclave_signature": _signature_hex(
                sign_digest(bytes.fromhex(digest_hash.split(":", 1)[1]))
            ),
        }
    )
    validate_signed_transition_command(command, verify_signature=False)
    return command


def validate_transition_command_body(body: Mapping[str, Any]) -> None:
    _validate_exact_fields(body, _TRANSITION_BODY_FIELDS, "transition command body")
    rebuilt = build_transition_command_body(
        operation=body["operation"],
        target=body["target"],
        idempotency_key=body["idempotency_key"],
        expected_state_hash=body["expected_state_hash"],
        payload_hash=body["payload_hash"],
        receipt_hash=body["receipt_hash"],
        issued_at=body["issued_at"],
        expires_at=body["expires_at"],
    )
    _require(dict(body) == rebuilt, "transition command body is not canonical")


def validate_signed_transition_command(
    command: Mapping[str, Any], *, verify_signature: bool = True
) -> None:
    _validate_exact_fields(command, _SIGNED_TRANSITION_FIELDS, "signed transition command")
    body = {key: command[key] for key in _TRANSITION_BODY_FIELDS}
    validate_transition_command_body(body)
    expected_hash = sha256_json(body)
    _require(command["command_hash"] == expected_hash, "command_hash does not match body")
    pubkey = _pubkey_hex(command["enclave_pubkey"], "enclave_pubkey")
    signature = _signature_hex(command["enclave_signature"])
    if verify_signature:
        _verify_ed25519(pubkey, signature, expected_hash)


_HOST_OPERATION_REQUEST_BODY_FIELDS = {
    "schema_version",
    "job_id",
    "purpose",
    "operation",
    "sequence",
    "payload_hash",
    "expected_state_hash",
    "boot_identity_hash",
    "request_nonce",
    "issued_at",
    "expires_at",
}
_SIGNED_HOST_OPERATION_REQUEST_FIELDS = _HOST_OPERATION_REQUEST_BODY_FIELDS | {
    "request_hash",
    "enclave_pubkey",
    "enclave_signature",
}
_HOST_OPERATION_TERMINAL_BODY_FIELDS = {
    "schema_version",
    "request_hash",
    "job_id",
    "purpose",
    "operation",
    "sequence",
    "terminal_status",
    "response_hash",
    "failure_code",
    "completed_at",
}
_SIGNED_HOST_OPERATION_TERMINAL_FIELDS = _HOST_OPERATION_TERMINAL_BODY_FIELDS | {
    "terminal_hash",
    "enclave_pubkey",
    "enclave_signature",
}


def build_host_operation_request_body(
    *,
    job_id: str,
    purpose: str,
    operation: str,
    sequence: int,
    payload_hash: str,
    expected_state_hash: str,
    boot_identity_hash: str,
    request_nonce: str,
    issued_at: str,
    expires_at: str,
) -> Dict[str, Any]:
    normalized_purpose = _identifier(purpose, "purpose")
    _require(
        any(normalized_purpose in purposes for purposes in ROLE_PURPOSES.values()),
        "host operation purpose is unsupported",
    )
    _require(isinstance(sequence, int) and sequence >= 0, "host operation sequence is invalid")
    normalized_nonce = str(request_nonce or "").strip().lower()
    _require(bool(_NONCE_RE.fullmatch(normalized_nonce)), "host operation nonce is invalid")
    normalized_issued = _timestamp(issued_at, "issued_at")
    normalized_expires = _timestamp(expires_at, "expires_at")
    _require(normalized_expires > normalized_issued, "host operation request must expire")
    return {
        "schema_version": HOST_OPERATION_REQUEST_SCHEMA_VERSION,
        "job_id": _identifier(job_id, "job_id"),
        "purpose": normalized_purpose,
        "operation": _identifier(operation, "operation"),
        "sequence": sequence,
        "payload_hash": _hash(payload_hash, "payload_hash"),
        "expected_state_hash": _hash(expected_state_hash, "expected_state_hash"),
        "boot_identity_hash": _hash(boot_identity_hash, "boot_identity_hash"),
        "request_nonce": normalized_nonce,
        "issued_at": normalized_issued,
        "expires_at": normalized_expires,
    }


def create_signed_host_operation_request(
    *,
    body: Mapping[str, Any],
    enclave_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    validate_host_operation_request_body(body)
    digest = sha256_json(dict(body))
    value = dict(body)
    value.update(
        {
            "request_hash": digest,
            "enclave_pubkey": _pubkey_hex(enclave_pubkey, "enclave_pubkey"),
            "enclave_signature": _signature_hex(
                sign_digest(bytes.fromhex(digest.split(":", 1)[1]))
            ),
        }
    )
    validate_signed_host_operation_request(value, verify_signature=False)
    return value


def validate_host_operation_request_body(body: Mapping[str, Any]) -> None:
    _validate_exact_fields(body, _HOST_OPERATION_REQUEST_BODY_FIELDS, "host operation request body")
    rebuilt = build_host_operation_request_body(
        job_id=body["job_id"],
        purpose=body["purpose"],
        operation=body["operation"],
        sequence=body["sequence"],
        payload_hash=body["payload_hash"],
        expected_state_hash=body["expected_state_hash"],
        boot_identity_hash=body["boot_identity_hash"],
        request_nonce=body["request_nonce"],
        issued_at=body["issued_at"],
        expires_at=body["expires_at"],
    )
    _require(dict(body) == rebuilt, "host operation request body is not canonical")


def validate_signed_host_operation_request(
    request: Mapping[str, Any], *, verify_signature: bool = True
) -> None:
    _validate_exact_fields(request, _SIGNED_HOST_OPERATION_REQUEST_FIELDS, "signed host operation request")
    body = {key: request[key] for key in _HOST_OPERATION_REQUEST_BODY_FIELDS}
    validate_host_operation_request_body(body)
    expected_hash = sha256_json(body)
    _require(request["request_hash"] == expected_hash, "host operation request hash mismatch")
    pubkey = _pubkey_hex(request["enclave_pubkey"], "enclave_pubkey")
    signature = _signature_hex(request["enclave_signature"])
    if verify_signature:
        _verify_ed25519(pubkey, signature, expected_hash)


def build_host_operation_terminal_body(
    *,
    request_hash: str,
    job_id: str,
    purpose: str,
    operation: str,
    sequence: int,
    terminal_status: str,
    response_hash: Optional[str],
    failure_code: Optional[str],
    completed_at: str,
) -> Dict[str, Any]:
    normalized_purpose = _identifier(purpose, "purpose")
    _require(
        any(normalized_purpose in purposes for purposes in ROLE_PURPOSES.values()),
        "host operation purpose is unsupported",
    )
    _require(isinstance(sequence, int) and sequence >= 0, "host operation sequence is invalid")
    _require(terminal_status in {"succeeded", "failed"}, "host operation terminal status is invalid")
    if terminal_status == "succeeded":
        normalized_response_hash = _hash(response_hash, "response_hash")
        _require(failure_code in (None, ""), "successful host operation cannot have failure")
        normalized_failure = None
    else:
        normalized_response_hash = (
            _hash(response_hash, "response_hash")
            if response_hash not in (None, "")
            else None
        )
        normalized_failure = _identifier(failure_code, "failure_code")
    return {
        "schema_version": HOST_OPERATION_TERMINAL_SCHEMA_VERSION,
        "request_hash": _hash(request_hash, "request_hash"),
        "job_id": _identifier(job_id, "job_id"),
        "purpose": normalized_purpose,
        "operation": _identifier(operation, "operation"),
        "sequence": sequence,
        "terminal_status": terminal_status,
        "response_hash": normalized_response_hash,
        "failure_code": normalized_failure,
        "completed_at": _timestamp(completed_at, "completed_at"),
    }


def create_signed_host_operation_terminal(
    *,
    body: Mapping[str, Any],
    enclave_pubkey: str,
    sign_digest: Callable[[bytes], Any],
) -> Dict[str, Any]:
    validate_host_operation_terminal_body(body)
    digest = sha256_json(dict(body))
    value = dict(body)
    value.update(
        {
            "terminal_hash": digest,
            "enclave_pubkey": _pubkey_hex(enclave_pubkey, "enclave_pubkey"),
            "enclave_signature": _signature_hex(
                sign_digest(bytes.fromhex(digest.split(":", 1)[1]))
            ),
        }
    )
    validate_signed_host_operation_terminal(value, verify_signature=False)
    return value


def validate_host_operation_terminal_body(body: Mapping[str, Any]) -> None:
    _validate_exact_fields(body, _HOST_OPERATION_TERMINAL_BODY_FIELDS, "host operation terminal body")
    rebuilt = build_host_operation_terminal_body(
        request_hash=body["request_hash"],
        job_id=body["job_id"],
        purpose=body["purpose"],
        operation=body["operation"],
        sequence=body["sequence"],
        terminal_status=body["terminal_status"],
        response_hash=body["response_hash"],
        failure_code=body["failure_code"],
        completed_at=body["completed_at"],
    )
    _require(dict(body) == rebuilt, "host operation terminal body is not canonical")


def validate_signed_host_operation_terminal(
    terminal: Mapping[str, Any], *, verify_signature: bool = True
) -> None:
    _validate_exact_fields(terminal, _SIGNED_HOST_OPERATION_TERMINAL_FIELDS, "signed host operation terminal")
    body = {key: terminal[key] for key in _HOST_OPERATION_TERMINAL_BODY_FIELDS}
    validate_host_operation_terminal_body(body)
    expected_hash = sha256_json(body)
    _require(terminal["terminal_hash"] == expected_hash, "host operation terminal hash mismatch")
    pubkey = _pubkey_hex(terminal["enclave_pubkey"], "enclave_pubkey")
    signature = _signature_hex(terminal["enclave_signature"])
    if verify_signature:
        _verify_ed25519(pubkey, signature, expected_hash)


_HOST_OPERATION_RECORD_FIELDS = {"request", "terminal"}


def validate_host_operation_record(record: Mapping[str, Any]) -> None:
    """Validate one complete enclave-requested host operation.

    Responses remain private artifacts. Their canonical hash is bound by the
    signed terminal, while the signed request binds the exact payload hash and
    expected host/database state.
    """

    _validate_exact_fields(
        record,
        _HOST_OPERATION_RECORD_FIELDS,
        "host operation record",
    )
    request = record.get("request")
    terminal = record.get("terminal")
    _require(isinstance(request, Mapping), "host operation request is missing")
    _require(isinstance(terminal, Mapping), "host operation terminal is missing")
    validate_signed_host_operation_request(request)
    validate_signed_host_operation_terminal(terminal)
    for field in ("job_id", "purpose", "operation", "sequence"):
        _require(
            request[field] == terminal[field],
            "host operation terminal differs from request at %s" % field,
        )
    _require(
        terminal["request_hash"] == request["request_hash"],
        "host operation terminal request hash mismatch",
    )
    _require(
        terminal["enclave_pubkey"] == request["enclave_pubkey"],
        "host operation terminal key mismatch",
    )


def host_operation_root(records: Sequence[Mapping[str, Any]]) -> str:
    terminal_hashes = []
    request_hashes = set()
    scopes = set()
    for record in records:
        validate_host_operation_record(record)
        request = record["request"]
        terminal = record["terminal"]
        request_hash = str(request["request_hash"])
        scope = (
            str(request["job_id"]),
            str(request["purpose"]),
            int(request["sequence"]),
        )
        _require(request_hash not in request_hashes, "host operation request is duplicated")
        _require(scope not in scopes, "host operation sequence is duplicated")
        request_hashes.add(request_hash)
        scopes.add(scope)
        terminal_hashes.append(str(terminal["terminal_hash"]))
    return merkle_root(terminal_hashes, domain="leadpoet-host-operation-v2")


_GRAPH_FIELDS = {
    "schema_version",
    "root_receipt_hash",
    "boot_identities",
    "receipts",
    "transport_attempts",
    "host_operations",
}


def build_receipt_graph(
    *,
    root_receipt_hash: str,
    boot_identities: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    transport_attempts: Sequence[Mapping[str, Any]],
    host_operations: Sequence[Mapping[str, Any]] = (),
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> Dict[str, Any]:
    graph = {
        "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
        "root_receipt_hash": _hash(root_receipt_hash, "root_receipt_hash"),
        "boot_identities": [dict(item) for item in boot_identities],
        "receipts": [dict(item) for item in receipts],
        "transport_attempts": [dict(item) for item in transport_attempts],
        "host_operations": [dict(item) for item in host_operations],
    }
    validate_receipt_graph(
        graph,
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )
    return graph


def validate_receipt_graph(
    graph: Mapping[str, Any],
    *,
    required_purposes: Iterable[str] = (),
    require_success: bool = True,
    allowed_failed_receipt_hashes: Iterable[str] = (),
    boot_attestation_verifier: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    require_boot_attestation_verification: bool = False,
) -> Tuple[str, ...]:
    _validate_exact_fields(graph, _GRAPH_FIELDS, "receipt graph")
    _require(graph["schema_version"] == RECEIPT_GRAPH_SCHEMA_VERSION, "unsupported receipt graph schema")
    _require(isinstance(graph["boot_identities"], list), "boot_identities must be a list")
    _require(isinstance(graph["receipts"], list), "receipts must be a list")
    _require(isinstance(graph["transport_attempts"], list), "transport_attempts must be a list")
    _require(isinstance(graph["host_operations"], list), "host_operations must be a list")

    boots = {}  # type: Dict[str, Mapping[str, Any]]
    for identity in graph["boot_identities"]:
        _require(isinstance(identity, Mapping), "boot identity is not an object")
        validate_boot_identity(identity)
        if boot_attestation_verifier is not None:
            boot_attestation_verifier(identity)
        elif require_boot_attestation_verification:
            raise AttestedV2Error("boot attestation verifier is required")
        identity_hash = str(identity["boot_identity_hash"])
        _require(identity_hash not in boots, "boot identity is duplicated")
        boots[identity_hash] = identity

    allowed_failed = {
        _hash(value, "allowed_failed_receipt_hash")
        for value in allowed_failed_receipt_hashes
    }
    receipts = {}  # type: Dict[str, Mapping[str, Any]]
    scope_receipts = set()
    for receipt in graph["receipts"]:
        _require(isinstance(receipt, Mapping), "receipt is not an object")
        validate_signed_execution_receipt(receipt)
        receipt_hash = str(receipt["receipt_hash"])
        _require(receipt_hash not in receipts, "receipt is duplicated")
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        _require(scope not in scope_receipts, "job/purpose receipt scope is duplicated")
        scope_receipts.add(scope)
        boot = boots.get(str(receipt["boot_identity_hash"]))
        _require(boot is not None, "receipt boot identity is missing")
        _require(boot["role"] == receipt["role"], "receipt role differs from boot role")
        _require(boot["commit_sha"] == receipt["commit_sha"], "receipt commit differs from boot commit")
        _require(boot["pcr0"] == receipt["pcr0"], "receipt PCR0 differs from boot PCR0")
        _require(
            boot["build_manifest_hash"] == receipt["build_manifest_hash"],
            "receipt build manifest differs from boot identity",
        )
        _require(
            boot["dependency_lock_hash"] == receipt["dependency_lock_hash"],
            "receipt dependency lock differs from boot identity",
        )
        _require(
            boot["config_hash"] == receipt["config_hash"],
            "receipt config differs from boot identity",
        )
        _require(
            boot["signing_pubkey"] == receipt["enclave_pubkey"],
            "receipt key differs from attested boot key",
        )
        if require_success and receipt["status"] != "succeeded":
            _require(
                receipt_hash in allowed_failed,
                "receipt graph contains an unauthorized failed receipt",
            )
        receipts[receipt_hash] = receipt

    _require(
        allowed_failed.issubset(receipts),
        "allowed failed receipt is absent from graph",
    )
    for receipt_hash in allowed_failed:
        _require(
            receipts[receipt_hash]["status"] == "failed",
            "allowed failed receipt is not failed",
        )

    attempts_by_scope = {}  # type: Dict[Tuple[str, str], List[Mapping[str, Any]]]
    attempt_hashes = set()
    for attempt in graph["transport_attempts"]:
        _require(isinstance(attempt, Mapping), "transport attempt is not an object")
        validate_transport_attempt(attempt)
        attempt_hash = str(attempt["attempt_hash"])
        _require(attempt_hash not in attempt_hashes, "transport attempt is duplicated")
        attempt_hashes.add(attempt_hash)
        scope = (str(attempt["job_id"]), str(attempt["purpose"]))
        attempts_by_scope.setdefault(scope, []).append(attempt)

    for receipt in receipts.values():
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        observed_root = transport_root(attempts_by_scope.pop(scope, []))
        _require(
            observed_root == receipt["transport_root"],
            "receipt transport root does not match terminal attempts",
        )
    _require(not attempts_by_scope, "receipt graph contains unclaimed transport attempts")

    host_operations_by_scope = {}  # type: Dict[Tuple[str, str], List[Mapping[str, Any]]]
    host_request_hashes = set()
    for record in graph["host_operations"]:
        _require(isinstance(record, Mapping), "host operation record is not an object")
        validate_host_operation_record(record)
        request = record["request"]
        request_hash = str(request["request_hash"])
        _require(
            request_hash not in host_request_hashes,
            "host operation request is duplicated",
        )
        host_request_hashes.add(request_hash)
        scope = (str(request["job_id"]), str(request["purpose"]))
        host_operations_by_scope.setdefault(scope, []).append(record)

    for receipt in receipts.values():
        scope = (str(receipt["job_id"]), str(receipt["purpose"]))
        records = host_operations_by_scope.pop(scope, [])
        for record in records:
            request = record["request"]
            _require(
                request["enclave_pubkey"] == receipt["enclave_pubkey"],
                "host operation key differs from receipt key",
            )
            _require(
                request["boot_identity_hash"] == receipt["boot_identity_hash"],
                "host operation boot differs from receipt boot",
            )
        observed_root = host_operation_root(records)
        _require(
            observed_root == receipt["host_operation_root"],
            "receipt host operation root does not match signed terminals",
        )
    _require(
        not host_operations_by_scope,
        "receipt graph contains unclaimed host operations",
    )

    root_hash = _hash(graph["root_receipt_hash"], "root_receipt_hash")
    _require(root_hash in receipts, "root receipt is missing")
    visiting = set()
    visited = set()
    ordered = []  # type: List[str]

    def visit(receipt_hash: str) -> None:
        if receipt_hash in visited:
            return
        _require(receipt_hash not in visiting, "receipt graph contains a cycle")
        visiting.add(receipt_hash)
        receipt = receipts[receipt_hash]
        for parent_hash in receipt["parent_receipt_hashes"]:
            _require(parent_hash in receipts, "receipt graph parent is missing")
            parent = receipts[parent_hash]
            _require(
                int(parent["epoch_id"]) <= int(receipt["epoch_id"]),
                "receipt parent epoch is newer than child",
            )
            visit(parent_hash)
        visiting.remove(receipt_hash)
        visited.add(receipt_hash)
        ordered.append(receipt_hash)

    visit(root_hash)
    _require(set(visited) == set(receipts), "receipt graph contains disconnected receipts")
    observed_purposes = {str(receipt["purpose"]) for receipt in receipts.values()}
    for purpose in required_purposes:
        _require(
            _identifier(purpose, "required purpose") in observed_purposes,
            "receipt graph is missing required purpose %s" % purpose,
        )
    return tuple(ordered)
