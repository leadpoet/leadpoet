"""Canonical measured gateway boot identity contracts.

This module contains no gateway, database, AWS, or chain I/O so the same
validation code can run in gateway enclaves and offline verifiers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


BOOT_IDENTITY_SCHEMA_VERSION = "leadpoet.attested_boot_identity.v2"
TRANSPORT_ATTEMPT_SCHEMA_VERSION = "leadpoet.attested_transport_attempt.v2"
BOOT_ATTESTATION_CLAIM_SCHEMA_VERSION = "leadpoet.attested_boot_claim.v2"
BOOT_ATTESTATION_PURPOSE = "leadpoet.boot_identity.v2"

COORDINATOR_ROLE = "gateway_coordinator"
SCORING_ROLE = "gateway_scoring"

SUPPORTED_BOOT_ROLES = frozenset(
    {COORDINATOR_ROLE, SCORING_ROLE}
)

ARENA_CHAIN_TRANSPORT_PURPOSES = frozenset(
    {
        "validator.chain_state.v2",
        "validator.subnet_epoch_snapshot.v2",
        "validator.metagraph_state.v2",
        "validator.weights.finalized.v2",
    }
)


PHYSICAL_ROLES_BY_SERVICE_ROLE = {
    COORDINATOR_ROLE: frozenset({"gateway_coordinator"}),
    SCORING_ROLE: frozenset({"gateway_scoring"}),
}

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
        "response_too_large",
        "tls_failure",
        "timeout",
        "unexpected_eof",
    }
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_PUBKEY_RE = re.compile(r"^[0-9a-f]{64}$")
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
    {"schema_version": "leadpoet.egress_route.v2", "route": "direct"}
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


def _pubkey_hex(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_PUBKEY_RE.fullmatch(normalized)), "%s must be 32-byte lowercase hex" % field)
    return normalized


def _validate_exact_fields(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    _require(set(value) == expected, "%s fields do not match canonical schema" % label)


def merkle_root(hashes: Iterable[str], *, domain: str) -> str:
    domain_bytes = _identifier(domain, "merkle domain").encode("ascii")
    normalized = sorted({_hash(item, "merkle item") for item in hashes})
    if not normalized:
        return sha256_bytes(b"\x02" + domain_bytes + b":empty")
    nodes = [
        hashlib.sha256(
            b"\x00" + domain_bytes + bytes.fromhex(item.split(":", 1)[1])
        ).digest()
        for item in normalized
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(
                b"\x01" + domain_bytes + nodes[index] + nodes[index + 1]
            ).digest()
            for index in range(0, len(nodes), 2)
        ]
    return "sha256:" + nodes[0].hex()


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
    _require(role in SUPPORTED_BOOT_ROLES, "unsupported boot identity role")
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
    certificate_validity_at_attestation_time: bool = False,
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
        certificate_validity_at_attestation_time=(
            certificate_validity_at_attestation_time
        ),
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
        normalized_purpose in ARENA_CHAIN_TRANSPORT_PURPOSES,
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
