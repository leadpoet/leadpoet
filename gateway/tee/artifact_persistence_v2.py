"""Enclave-authenticated verification of immutable artifact persistence."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import json
import re
import secrets
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

from gateway.tee.artifact_vault_v2 import (
    ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD,
    ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES,
    EncryptedArtifactVaultV2,
)
from gateway.tee.egress_proxy import (
    DEFAULT_IDLE_TIMEOUT_SECONDS as EGRESS_TUNNEL_IDLE_TIMEOUT_SECONDS,
)
from gateway.tee.egress_framing import TUNNEL_FRAMING_MODE
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
ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS = (
    ARTIFACT_PERSISTENCE_MAX_ATTEMPTS_PER_METHOD
)
ARTIFACT_PERSISTENCE_TRANSPORT_TIMEOUT_MS = 30000
ARTIFACT_PERSISTENCE_RETRY_DELAYS_SECONDS = (0.0, 0.25, 1.0, 2.0)
MAX_ARTIFACT_STORAGE_DOCUMENT_BYTES = 96 * 1024 * 1024
MAX_ARTIFACT_VERIFICATION_TRANSPORTS = 32
ARTIFACT_VERIFICATION_TRANSPORT_WAIT_SECONDS = 30.0
ARTIFACT_VERIFICATION_TRANSPORT_MAX_IDLE_SECONDS = min(
    120.0,
    EGRESS_TUNNEL_IDLE_TIMEOUT_SECONDS / 2.0,
)
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
        ("cannot assign requested address", "proxy_failure"),
        ("address already in use", "proxy_failure"),
        ("too many open files", "proxy_failure"),
        ("no buffer space available", "proxy_failure"),
        ("cannot allocate memory", "proxy_failure"),
        ("transport pool exhausted", "proxy_failure"),
        ("timeout", "timeout"),
        ("certificate", "tls_failure"),
        ("tls", "tls_failure"),
        ("dns", "dns_failure"),
        ("reset", "connection_reset"),
        ("refused", "connection_refused"),
        ("proxy", "proxy_failure"),
        ("malformed", "malformed_reply"),
        ("protocol", "malformed_reply"),
    ):
        if token in text:
            return code
    return "unexpected_eof"


class _ArtifactVerificationTransportPool:
    """Lease bounded clients exclusively across concurrent verifications."""

    def __init__(
        self,
        *,
        maximum_transports: int = MAX_ARTIFACT_VERIFICATION_TRANSPORTS,
        wait_seconds: float = ARTIFACT_VERIFICATION_TRANSPORT_WAIT_SECONDS,
        maximum_idle_seconds: float = ARTIFACT_VERIFICATION_TRANSPORT_MAX_IDLE_SECONDS,
        idle_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(maximum_transports, bool)
            or not isinstance(maximum_transports, int)
            or maximum_transports < 1
            or maximum_transports > 128
            or float(wait_seconds) <= 0
            or float(maximum_idle_seconds) <= 0
            or float(maximum_idle_seconds) >= EGRESS_TUNNEL_IDLE_TIMEOUT_SECONDS
        ):
            raise ArtifactPersistenceV2Error(
                "artifact verification transport pool configuration is invalid"
            )
        self._condition = threading.Condition()
        self._maximum_transports = maximum_transports
        self._wait_seconds = float(wait_seconds)
        self._maximum_idle_seconds = float(maximum_idle_seconds)
        self._idle_clock = idle_clock
        self._generation = 0
        self._idle = []  # type: list[tuple[float, int, Any]]
        self._leased_generations = {}  # type: dict[int, int]
        self._transport_count = 0

    @staticmethod
    def _new_transport() -> HTTPXProviderTransport:
        return HTTPXProviderTransport(
            response_body_ceiling_bytes=MAX_ARTIFACT_STORAGE_DOCUMENT_BYTES,
            allow_authenticated_complete_body_eof=True,
            parent_tunnel_framing=TUNNEL_FRAMING_MODE,
        )

    def _expire_idle_locked(self) -> list[Any]:
        now = self._idle_clock()
        reusable = []
        expired = []
        for released_at, generation, transport in self._idle:
            if (
                generation != self._generation
                or max(0.0, now - released_at) >= self._maximum_idle_seconds
            ):
                expired.append(transport)
                self._transport_count -= 1
            else:
                reusable.append((released_at, generation, transport))
        self._idle = reusable
        if expired:
            self._condition.notify_all()
        return expired

    @staticmethod
    def _close_all(transports: Sequence[Any]) -> None:
        for transport in transports:
            with suppress(Exception):
                transport.close()

    def acquire(self) -> Callable[..., Mapping[str, Any]]:
        deadline = time.monotonic() + self._wait_seconds
        expired = []
        selected = None
        create_new = False
        timed_out = False
        with self._condition:
            while True:
                expired.extend(self._expire_idle_locked())
                if self._idle:
                    _released_at, generation, selected = self._idle.pop()
                    self._leased_generations[id(selected)] = generation
                    break
                if self._transport_count < self._maximum_transports:
                    self._transport_count += 1
                    create_new = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                self._condition.wait(timeout=remaining)
        self._close_all(expired)
        if timed_out:
            raise ArtifactPersistenceV2Error(
                "artifact verification transport pool exhausted"
            )
        if selected is not None:
            return selected
        if not create_new:
            raise RuntimeError("artifact verification transport lease is invalid")
        try:
            selected = self._new_transport()
        except BaseException:
            with self._condition:
                self._transport_count -= 1
                self._condition.notify()
            raise
        with self._condition:
            # Construction performs no network I/O. If another lease failed
            # while this client was being built, it belongs to the new healthy
            # generation rather than the retired one.
            self._leased_generations[id(selected)] = self._generation
        return selected

    def release(self, transport: Any, *, failed: bool = False) -> None:
        to_close = []
        with self._condition:
            generation = self._leased_generations.pop(id(transport), None)
            if generation is None:
                raise ArtifactPersistenceV2Error(
                    "artifact verification transport lease is invalid"
                )
            if failed and generation == self._generation:
                # Retire the complete generation, including siblings that were
                # leased when this failure occurred. Those siblings may finish
                # their current authenticated request, but they must not return
                # later and repopulate the pool with dead framed connections.
                self._generation += 1
                to_close = [transport, *(item[2] for item in self._idle)]
                self._transport_count -= len(to_close)
                self._idle = []
            elif failed or generation != self._generation:
                to_close = [transport]
                self._transport_count -= 1
            else:
                self._idle.append(
                    (self._idle_clock(), self._generation, transport)
                )
            self._condition.notify_all()
        self._close_all(to_close)


class _ArtifactVerificationTransportSession:
    """Lease one reusable transport generation for an artifact verification."""

    def __init__(
        self,
        transport: Optional[Callable[..., Mapping[str, Any]]],
        transport_pool: Optional[_ArtifactVerificationTransportPool],
    ) -> None:
        self._transport = transport
        self._transport_pool = transport_pool
        self._leased_transport = None

    def get(self) -> Callable[..., Mapping[str, Any]]:
        if self._transport is None:
            if self._transport_pool is None:
                raise RuntimeError("artifact verification transport pool is unavailable")
            self._transport = self._transport_pool.acquire()
            self._leased_transport = self._transport
        return self._transport

    def rotate_after_failure(self) -> None:
        self._release(failed=True)

    def close(self) -> None:
        self._release(failed=False)

    def _release(self, *, failed: bool) -> None:
        leased_transport = self._leased_transport
        if leased_transport is None:
            return
        self._transport = None
        self._leased_transport = None
        if self._transport_pool is not None:
            self._transport_pool.release(leased_transport, failed=failed)

    def __enter__(self) -> "_ArtifactVerificationTransportSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class ArtifactPersistenceVerifierV2:
    """Fetch ciphertext back through enclave-verified TLS before acceptance."""

    def __init__(
        self,
        *,
        vault: EncryptedArtifactVaultV2,
        policy: Mapping[str, Any],
        transport: Optional[Callable[..., Mapping[str, Any]]] = None,
        clock: Callable[[], str] = _timestamp,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._vault = vault
        self._policy = validate_artifact_policy(policy)
        self._policy_hash = sha256_json(self._policy)
        # Artifact readbacks can approach the relay's per-tunnel byte budget.
        # Each verification exclusively leases one bounded session. Healthy
        # clients are reused sequentially to avoid exhausting loopback TCP
        # endpoints, while a failed connection cannot poison unrelated
        # concurrent persistence jobs.
        self._transport = transport
        self._transport_pool = (
            None if transport is not None else _ArtifactVerificationTransportPool()
        )
        self._clock = clock
        self._sleeper = sleeper
        self._retry_policy_hash = sha256_json(
            {
                "schema_version": "leadpoet.artifact_persistence_retry_policy.v2",
                "attempts_per_method": ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS,
                "retry_delays_seconds": list(
                    ARTIFACT_PERSISTENCE_RETRY_DELAYS_SECONDS
                ),
                "retryable_http_statuses": sorted(
                    ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES
                ),
                "timeout_ms": ARTIFACT_PERSISTENCE_TRANSPORT_TIMEOUT_MS,
                "storage_policy_hash": self._policy_hash,
                "maximum_storage_document_bytes": (
                    MAX_ARTIFACT_STORAGE_DOCUMENT_BYTES
                ),
            }
        )

    def _request(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        method: str,
        url: str,
        ordinal: int,
        transport_session: _ArtifactVerificationTransportSession,
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
            transport_kwargs = {}
            if method == "GET":
                transport_kwargs["max_response_bytes"] = (
                    MAX_ARTIFACT_STORAGE_DOCUMENT_BYTES
                )
            response = dict(
                transport_session.get()(
                    method=method,
                    url=url,
                    headers={"accept": "application/json"},
                    body=b"",
                    timeout_ms=ARTIFACT_PERSISTENCE_TRANSPORT_TIMEOUT_MS,
                    **transport_kwargs,
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
            retry_policy_hash=self._retry_policy_hash,
            timeout_ms=ARTIFACT_PERSISTENCE_TRANSPORT_TIMEOUT_MS,
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

        with _ArtifactVerificationTransportSession(
            self._transport,
            self._transport_pool,
        ) as transport_session:
            return self._verify_with_transport_session(
                artifact_id=artifact_id,
                attestation_job_id=attestation_job_id,
                artifact_ref=artifact_ref,
                normalized_get=normalized_get,
                normalized_head=normalized_head,
                transport_session=transport_session,
            )

    def _verify_with_transport_session(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        artifact_ref: str,
        normalized_get: str,
        normalized_head: str,
        transport_session: _ArtifactVerificationTransportSession,
    ) -> Dict[str, Any]:
        attempts = []
        get_response = {}
        get_attempt = {}
        for ordinal in range(ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS):
            if ordinal:
                self._sleeper(
                    ARTIFACT_PERSISTENCE_RETRY_DELAYS_SECONDS[ordinal]
                )
            get_response, get_attempt = self._request(
                artifact_id=artifact_id,
                attestation_job_id=attestation_job_id,
                method="GET",
                url=normalized_get,
                ordinal=ordinal,
                transport_session=transport_session,
            )
            attempts.append(get_attempt)
            if get_attempt["terminal_status"] == "authenticated_response":
                if (
                    get_attempt["http_status"]
                    in ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES
                    and ordinal + 1 < ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS
                ):
                    continue
                break
            transport_session.rotate_after_failure()
        if get_attempt["terminal_status"] != "authenticated_response":
            return self._failure(attempts, get_attempt["failure_code"])
        if get_attempt["http_status"] != 200:
            return self._failure(
                attempts, "authenticated_http_%s" % get_attempt["http_status"]
            )
        body = bytes(get_response.get("body") or b"")
        if len(body) > MAX_ARTIFACT_STORAGE_DOCUMENT_BYTES:
            return self._failure(attempts, "storage_document_too_large")
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
            if offset:
                self._sleeper(
                    ARTIFACT_PERSISTENCE_RETRY_DELAYS_SECONDS[offset]
                )
            head_response, head_attempt = self._request(
                artifact_id=artifact_id,
                attestation_job_id=attestation_job_id,
                method="HEAD",
                url=normalized_head,
                ordinal=len(attempts),
                transport_session=transport_session,
            )
            attempts.append(head_attempt)
            if head_attempt["terminal_status"] == "authenticated_response":
                if (
                    head_attempt["http_status"]
                    in ARTIFACT_PERSISTENCE_RETRYABLE_HTTP_STATUSES
                    and offset + 1 < ARTIFACT_PERSISTENCE_TRANSPORT_ATTEMPTS
                ):
                    continue
                break
            transport_session.rotate_after_failure()
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
