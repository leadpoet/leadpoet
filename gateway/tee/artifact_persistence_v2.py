"""Enclave-authenticated verification of immutable artifact persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import secrets
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_broker_v2 import HTTPXProviderTransport
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
    transport_root,
)


ARTIFACT_POLICY_SCHEMA_VERSION = "leadpoet.encrypted_artifact_policy.v2"
ARTIFACT_PERSISTENCE_PURPOSE = "leadpoet.artifact_persistence.v2"
ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS = 3
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DNS_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class ArtifactPersistenceV2Error(RuntimeError):
    """An artifact policy or authenticated persistence result is invalid."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_artifact_policy(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "bucket_host",
        "key_prefix",
        "minimum_retention_days",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ArtifactPersistenceV2Error("encrypted artifact policy fields are invalid")
    if value.get("schema_version") != ARTIFACT_POLICY_SCHEMA_VERSION:
        raise ArtifactPersistenceV2Error("encrypted artifact policy schema is invalid")
    bucket_host = str(value.get("bucket_host") or "").strip().lower().rstrip(".")
    if (
        not _DNS_RE.fullmatch(bucket_host)
        or ".s3" not in bucket_host
        or not bucket_host.endswith(".amazonaws.com")
    ):
        raise ArtifactPersistenceV2Error("encrypted artifact bucket host is invalid")
    key_prefix = str(value.get("key_prefix") or "")
    if (
        not key_prefix.startswith("/")
        or not key_prefix.endswith("/")
        or ".." in key_prefix
        or "?" in key_prefix
        or "#" in key_prefix
    ):
        raise ArtifactPersistenceV2Error("encrypted artifact key prefix is invalid")
    retention = value.get("minimum_retention_days")
    if not isinstance(retention, int) or isinstance(retention, bool) or not 1 <= retention <= 3650:
        raise ArtifactPersistenceV2Error("encrypted artifact retention is invalid")
    return {
        "schema_version": ARTIFACT_POLICY_SCHEMA_VERSION,
        "bucket_host": bucket_host,
        "key_prefix": key_prefix,
        "minimum_retention_days": retention,
    }


def _validate_presigned_url(
    value: str,
    *,
    policy: Mapping[str, Any],
) -> str:
    parsed = urlsplit(str(value or ""))
    if (
        parsed.scheme != "https"
        or parsed.hostname != policy["bucket_host"]
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.path.startswith(str(policy["key_prefix"]))
    ):
        raise ArtifactPersistenceV2Error("artifact verification URL violates policy")
    query = {name.lower(): values for name, values in parse_qs(parsed.query).items()}
    required = {
        "x-amz-algorithm",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-signedheaders",
        "x-amz-signature",
    }
    if not required.issubset(query):
        raise ArtifactPersistenceV2Error("artifact verification URL is not SigV4 signed")
    if query["x-amz-algorithm"] != ["AWS4-HMAC-SHA256"]:
        raise ArtifactPersistenceV2Error("artifact verification URL algorithm is invalid")
    return parsed.geturl()


def _transport_failure_code(exc: BaseException) -> str:
    text = (type(exc).__name__ + " " + str(exc)).lower()
    for token, code in (
        ("timeout", "timeout"),
        ("certificate", "tls_failure"),
        ("tls", "tls_failure"),
        ("dns", "dns_failure"),
        ("reset", "connection_reset"),
        ("refused", "connection_refused"),
        ("malformed", "malformed_reply"),
    ):
        if token in text:
            return code
    return "unexpected_eof"


class ArtifactPersistenceVerifierV2:
    """Fetch ciphertext back through enclave-verified TLS before acceptance."""

    def __init__(
        self,
        *,
        vault: EncryptedArtifactVaultV2,
        policy: Mapping[str, Any],
        transport: Callable[..., Mapping[str, Any]] = HTTPXProviderTransport(),
        clock: Callable[[], str] = _timestamp,
    ) -> None:
        self._vault = vault
        self._policy = validate_artifact_policy(policy)
        self._policy_hash = sha256_json(self._policy)
        self._transport = transport
        self._clock = clock

    def _request(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        method: str,
        url: str,
        ordinal: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        parsed = urlsplit(url)
        started_at = self._clock()
        request_artifact_hash = sha256_json(
            {
                "schema_version": "leadpoet.storage_verification_request.v2",
                "artifact_id": artifact_id,
                "attestation_job_id": attestation_job_id,
                "method": method,
                "destination_host": str(parsed.hostname or ""),
                "path_hash": sha256_bytes(parsed.path.encode("utf-8")),
            }
        )
        terminal = {}
        response = {}
        try:
            response = dict(
                self._transport(
                    method=method,
                    url=url,
                    headers={"accept": "application/json"},
                    body=b"",
                    timeout_ms=30000,
                )
            )
            body = bytes(response.get("body") or b"")
            terminal = {
                "terminal_status": "authenticated_response",
                "http_status": int(response["http_status"]),
                "response_hash": sha256_bytes(body),
                "response_artifact_hash": sha256_bytes(body),
                "tls_peer_chain_hash": str(response["tls_peer_chain_hash"]),
                "tls_protocol": str(response["tls_protocol"]),
                "failure_code": None,
            }
        except Exception as exc:
            terminal = {
                "terminal_status": "transport_failure",
                "http_status": None,
                "response_hash": None,
                "response_artifact_hash": None,
                "tls_peer_chain_hash": None,
                "tls_protocol": None,
                "failure_code": _transport_failure_code(exc),
            }
        attempt = build_transport_attempt(
            request_id=secrets.token_hex(16),
            logical_operation_id="%s:%s" % (artifact_id, method.lower()),
            job_id=attestation_job_id,
            purpose=ARTIFACT_PERSISTENCE_PURPOSE,
            provider_id="aws_s3_object_lock",
            attempt_number=ordinal,
            method=method,
            destination_host=str(parsed.hostname or ""),
            destination_port=parsed.port or 443,
            path_hash=sha256_bytes(parsed.path.encode("utf-8")),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=sha256_bytes(b""),
            credential_ref_hash=sha256_json(
                {"policy_hash": self._policy_hash, "sigv4_query_present": True}
            ),
            retry_policy_hash=self._policy_hash,
            timeout_ms=30000,
            started_at=started_at,
            request_artifact_hash=request_artifact_hash,
            completed_at=self._clock(),
            **terminal,
        )
        return response, attempt

    def verify(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        artifact_ref: str,
        get_url: str,
        head_url: str,
    ) -> Dict[str, Any]:
        normalized_get = _validate_presigned_url(get_url, policy=self._policy)
        normalized_head = _validate_presigned_url(head_url, policy=self._policy)
        if urlsplit(normalized_get).path != urlsplit(normalized_head).path:
            raise ArtifactPersistenceV2Error("artifact verification URLs differ")

        attempts = []
        get_response = {}
        get_attempt = {}
        for ordinal in range(ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS):
            get_response, get_attempt = self._request(
                artifact_id=artifact_id,
                attestation_job_id=attestation_job_id,
                method="GET",
                url=normalized_get,
                ordinal=ordinal,
            )
            attempts.append(get_attempt)
            if get_attempt["terminal_status"] == "authenticated_response":
                break
        if get_attempt["terminal_status"] != "authenticated_response":
            return self._failure(attempts, get_attempt["failure_code"])
        if get_attempt["http_status"] != 200:
            return self._failure(
                attempts, "authenticated_http_%s" % get_attempt["http_status"]
            )
        body = bytes(get_response.get("body") or b"")
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._failure(attempts, "malformed_storage_document")
        if not isinstance(document, Mapping) or canonical_json(dict(document)).encode(
            "utf-8"
        ) != body:
            return self._failure(attempts, "noncanonical_storage_document")

        head_response = {}
        head_attempt = {}
        for offset in range(ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS):
            head_response, head_attempt = self._request(
                artifact_id=artifact_id,
                attestation_job_id=attestation_job_id,
                method="HEAD",
                url=normalized_head,
                ordinal=len(attempts),
            )
            attempts.append(head_attempt)
            if head_attempt["terminal_status"] == "authenticated_response":
                break
        if head_attempt["terminal_status"] != "authenticated_response":
            return self._failure(attempts, head_attempt["failure_code"])
        if head_attempt["http_status"] != 200:
            return self._failure(
                attempts, "authenticated_http_%s" % head_attempt["http_status"]
            )
        try:
            persistence_transport_root = transport_root(attempts)
            descriptor = self._vault.confirm_persistence(
                artifact_id=artifact_id,
                artifact_ref=artifact_ref,
                observed_storage_document=document,
                response_headers=head_response.get("headers") or {},
                transport_attempts=attempts,
            )
        except Exception:
            return self._failure(attempts, "object_lock_verification_failed")
        return {
            "status": "persisted",
            "artifact": descriptor,
            "transport_attempts": attempts,
            "transport_root": persistence_transport_root,
        }

    @staticmethod
    def _failure(attempts: Sequence[Mapping[str, Any]], code: str) -> Dict[str, Any]:
        normalized = [dict(item) for item in attempts]
        return {
            "status": "failed",
            "failure_code": str(code),
            "transport_attempts": normalized,
            "transport_root": transport_root(normalized),
        }
