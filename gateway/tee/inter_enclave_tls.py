"""Mutually attested TLS 1.3 RPC carried by the opaque parent relay."""

from __future__ import annotations

import base64
from collections import OrderedDict
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import ssl
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from gateway.tee.mtls_identity import (
    create_mutual_tls_context,
    verify_peer_certificate_binding,
    write_identity_to_tmpfs,
)
from gateway.tee.topology import ROLE_SPECS, role_spec, topology_hash
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    validate_boot_identity,
    verify_boot_identity_nitro,
)


AF_VSOCK = 40
PARENT_CID = 3
RELAY_PORT = 5002
TLS_SERVICE_PORT = 5003
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_REPLAY_CACHE_BYTES = 128 * 1024 * 1024
MAX_REPLAY_CACHE_ENTRIES = 128
REPLAY_CACHE_TTL_SECONDS = 300.0
REPLAY_WAIT_SECONDS = 1800.0
MAX_RPC_DELIVERY_ATTEMPTS = 6
RPC_DELIVERY_BACKOFF_SECONDS = 0.05
RPC_DELIVERY_ATTEMPT_TIMEOUT_SECONDS = 300.0
SCHEMA_VERSION = "leadpoet.inter_enclave_rpc.v2"
TRANSPORT_HEALTH_SCHEMA_VERSION = "leadpoet.inter_enclave_transport_health.v2"
MAX_TRANSPORT_CLEANUP_EVENT_COUNT = (1 << 63) - 1
TRANSPORT_SUPERVISOR_POLL_SECONDS = 0.25
TRANSPORT_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE = 1
MAX_TRANSPORT_CLEANUP_ATTEMPT_COUNT = (1 << 63) - 1
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)


class InterEnclaveTLSError(RuntimeError):
    """A peer, certificate, frame, or topology binding is invalid."""


class _RetryableInterEnclaveTransportError(InterEnclaveTLSError):
    """A connection ended without an authenticated terminal response."""


class InterEnclaveTransportCleanupError(InterEnclaveTLSError):
    """An owned RPC transport could not prove descriptor release."""

    def __init__(
        self,
        *,
        stage: str,
        primary_error: BaseException,
        cleanup_error: BaseException,
        resource: Any,
    ) -> None:
        super().__init__("inter-enclave RPC transport cleanup failed")
        self.stage = str(stage)
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        self._cleanup_attempt_count = 1
        # The resource is private and never crosses an RPC or health boundary.
        # Retaining it avoids declaring ownership released without proof.
        self._resource = resource


class _ExplicitCloseFailure(RuntimeError):
    """A transport adapter explicitly reported retained ownership."""


def _close_transport_required(candidate: Any) -> Optional[BaseException]:
    """Attempt full-duplex shutdown and return any close-proof failure."""

    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # TLS and relay peers can already be half-closed. close() is the
        # descriptor ownership boundary; shutdown() only terminates live I/O.
        pass
    try:
        if candidate.close() is False:
            return _ExplicitCloseFailure(
                "inter-enclave transport close was not confirmed"
            )
    except BaseException as exc:
        return exc
    return None


def _recv_exact(connection: Any, size: int) -> bytes:
    """Read one complete bounded frame without importing the host TEE client."""

    output = bytearray()
    while len(output) < size:
        try:
            chunk = connection.recv(min(64 * 1024, size - len(output)))
        except OSError as exc:
            raise _RetryableInterEnclaveTransportError(
                "inter-enclave connection read failed"
            ) from exc
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _certificate_der(certificate_pem: bytes) -> bytes:
    try:
        certificate = x509.load_pem_x509_certificate(bytes(certificate_pem))
    except Exception as exc:
        raise InterEnclaveTLSError("peer certificate is invalid") from exc
    return certificate.public_bytes(serialization.Encoding.DER)


def _send_frame(connection: Any, value: Mapping[str, Any]) -> None:
    encoded = canonical_json(dict(value)).encode("utf-8")
    if len(encoded) < 2 or len(encoded) > MAX_FRAME_BYTES:
        raise InterEnclaveTLSError("inter-enclave frame is outside size limit")
    try:
        connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
    except OSError as exc:
        raise _RetryableInterEnclaveTransportError(
            "inter-enclave connection write failed"
        ) from exc


def _read_frame(connection: Any) -> Dict[str, Any]:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise _RetryableInterEnclaveTransportError(
            "inter-enclave frame prefix is incomplete"
        )
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_FRAME_BYTES:
        raise InterEnclaveTLSError("inter-enclave frame is outside size limit")
    encoded = _recv_exact(connection, size)
    if len(encoded) != size:
        raise _RetryableInterEnclaveTransportError(
            "inter-enclave frame body is incomplete"
        )
    try:
        value = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise InterEnclaveTLSError("inter-enclave frame JSON is invalid") from exc
    if not isinstance(value, dict) or canonical_json(value).encode("utf-8") != encoded:
        raise InterEnclaveTLSError("inter-enclave frame is not canonical")
    return value


def _pair_allowed(source_role: str, target_role: str) -> bool:
    if source_role == "gateway_coordinator":
        return target_role == "gateway_scoring"
    if target_role == "gateway_coordinator":
        return source_role == "gateway_scoring"
    return False


class AttestedPeerRegistry:
    """Trust exact peer certificates only after full Nitro verification."""

    def __init__(
        self,
        *,
        local_physical_role: str,
        boot_verifier: Callable[..., Mapping[str, Any]] = verify_boot_identity_nitro,
    ) -> None:
        role_spec(local_physical_role)
        self.local_physical_role = local_physical_role
        self._boot_verifier = boot_verifier
        self._peers = {}  # type: Dict[str, Dict[str, Any]]
        self._cert_to_role = {}  # type: Dict[str, str]
        self._lock = threading.Lock()

    def register(
        self,
        *,
        boot_identity: Mapping[str, Any],
        certificate_pem: bytes,
        expected_pcr0: str,
        expected_commit_sha: str,
        expected_build_manifest_hash: str,
        expected_config_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        validate_boot_identity(boot_identity)
        physical_role = str(boot_identity["physical_role"])
        if not _pair_allowed(self.local_physical_role, physical_role):
            raise InterEnclaveTLSError("peer role pair is not authorized")
        peer_spec = role_spec(physical_role)
        if boot_identity["role"] != peer_spec["service_role"]:
            raise InterEnclaveTLSError("peer service role differs from topology")
        if boot_identity["commit_sha"] != expected_commit_sha:
            raise InterEnclaveTLSError("peer commit differs from expected release")
        if boot_identity["build_manifest_hash"] != expected_build_manifest_hash:
            raise InterEnclaveTLSError("peer build manifest differs from expected release")
        if expected_config_hash is not None and (
            boot_identity["config_hash"] != expected_config_hash
        ):
            raise InterEnclaveTLSError("peer config differs from expected release")
        self._boot_verifier(boot_identity, expected_pcr0=expected_pcr0)
        certificate_der = _certificate_der(certificate_pem)
        certificate = verify_peer_certificate_binding(
            boot_identity=boot_identity,
            certificate_der=certificate_der,
            expected_service_role=str(peer_spec["service_role"]),
        )
        certificate_hash = str(certificate["certificate_sha256"])
        peer = {
            "physical_role": physical_role,
            "service_role": boot_identity["role"],
            "boot_identity": dict(boot_identity),
            "certificate_pem": bytes(certificate_pem),
            "certificate_hash": certificate_hash,
        }
        with self._lock:
            existing = self._peers.get(physical_role)
            if existing is not None and existing["boot_identity"] != peer["boot_identity"]:
                raise InterEnclaveTLSError("peer role already has another boot identity")
            other_role = self._cert_to_role.get(certificate_hash)
            if other_role is not None and other_role != physical_role:
                raise InterEnclaveTLSError("peer certificate is reused across roles")
            self._peers[physical_role] = peer
            self._cert_to_role[certificate_hash] = physical_role
        return {
            "physical_role": physical_role,
            "service_role": peer["service_role"],
            "boot_identity_hash": boot_identity["boot_identity_hash"],
            "certificate_hash": certificate_hash,
        }

    def peer(self, physical_role: str) -> Dict[str, Any]:
        with self._lock:
            peer = self._peers.get(str(physical_role or ""))
            if peer is None:
                raise InterEnclaveTLSError("attested peer is not registered")
            return dict(peer)

    def peer_for_certificate(self, certificate_der: bytes) -> Dict[str, Any]:
        certificate_hash = "sha256:" + hashlib.sha256(bytes(certificate_der)).hexdigest()
        with self._lock:
            physical_role = self._cert_to_role.get(certificate_hash)
            if physical_role is None:
                raise InterEnclaveTLSError("TLS peer certificate is not attested")
            return dict(self._peers[physical_role])

    def trusted_certificates(self) -> Sequence[bytes]:
        with self._lock:
            return tuple(
                peer["certificate_pem"]
                for _, peer in sorted(self._peers.items())
            )

    def registered_roles(self) -> Sequence[str]:
        with self._lock:
            return tuple(sorted(self._peers))


def build_rpc_request(
    *,
    method: str,
    params: Mapping[str, Any],
    channel_id: str,
    source_boot_identity_hash: str,
    target_boot_identity_hash: str,
) -> Dict[str, Any]:
    normalized_channel = str(channel_id or "").lower()
    if not _CHANNEL_ID_RE.fullmatch(normalized_channel):
        raise InterEnclaveTLSError("inter-enclave channel ID is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "method": str(method or ""),
        "params": dict(params),
        "channel_id": normalized_channel,
        "source_boot_identity_hash": str(source_boot_identity_hash or ""),
        "target_boot_identity_hash": str(target_boot_identity_hash or ""),
        "topology_hash": topology_hash(),
    }


def validate_rpc_request(
    request: Mapping[str, Any],
    *,
    source_boot_identity_hash: str,
    target_boot_identity_hash: str,
) -> Dict[str, Any]:
    required = {
        "schema_version",
        "method",
        "params",
        "channel_id",
        "source_boot_identity_hash",
        "target_boot_identity_hash",
        "topology_hash",
    }
    if not isinstance(request, Mapping) or set(request) != required:
        raise InterEnclaveTLSError("inter-enclave request fields are invalid")
    if request["schema_version"] != SCHEMA_VERSION:
        raise InterEnclaveTLSError("inter-enclave request schema is invalid")
    if request["topology_hash"] != topology_hash():
        raise InterEnclaveTLSError("inter-enclave request topology mismatch")
    if request["source_boot_identity_hash"] != source_boot_identity_hash:
        raise InterEnclaveTLSError("inter-enclave request source boot mismatch")
    if request["target_boot_identity_hash"] != target_boot_identity_hash:
        raise InterEnclaveTLSError("inter-enclave request target boot mismatch")
    if not _CHANNEL_ID_RE.fullmatch(str(request["channel_id"])):
        raise InterEnclaveTLSError("inter-enclave request channel ID is invalid")
    if not isinstance(request["params"], Mapping) or not str(request["method"]):
        raise InterEnclaveTLSError("inter-enclave request method or params are invalid")
    return dict(request)


class AttestedTLSRPCClient:
    def __init__(
        self,
        *,
        local_physical_role: str,
        local_boot_identity: Mapping[str, Any],
        local_tls_identity: Mapping[str, Any],
        peer_registry: AttestedPeerRegistry,
        tmpfs_root: Path = Path("/run/leadpoet-v2"),
        connector: Optional[Callable[[], Any]] = None,
        delivery_attempt_timeout_seconds: float = (
            RPC_DELIVERY_ATTEMPT_TIMEOUT_SECONDS
        ),
    ) -> None:
        self.local_physical_role = local_physical_role
        self.local_boot_identity = dict(local_boot_identity)
        self.local_tls_identity = dict(local_tls_identity)
        self.peer_registry = peer_registry
        self.identity_paths = write_identity_to_tmpfs(
            self.local_tls_identity,
            directory=tmpfs_root / local_physical_role,
        )
        self._connector = connector
        self._delivery_attempt_timeout_seconds = float(
            delivery_attempt_timeout_seconds
        )
        if (
            not math.isfinite(self._delivery_attempt_timeout_seconds)
            or self._delivery_attempt_timeout_seconds <= 0
            or self._delivery_attempt_timeout_seconds > REPLAY_WAIT_SECONDS
        ):
            raise InterEnclaveTLSError(
                "inter-enclave delivery attempt timeout is invalid"
            )
        self._transport_health_lock = threading.Lock()
        self._transport_recovery_lock = threading.Lock()
        self._pending_cleanup_failures = []
        self._cleanup_recovery_count = 0

    def _retain_transport_cleanup_failure(
        self,
        failure: InterEnclaveTransportCleanupError,
    ) -> None:
        with self._transport_health_lock:
            self._pending_cleanup_failures.append(failure)

    def _recover_transport_cleanup_failures(self) -> None:
        with self._transport_recovery_lock:
            with self._transport_health_lock:
                snapshot = tuple(self._pending_cleanup_failures)
            resolved = []
            for failure in snapshot:
                cleanup_error = None  # type: Optional[BaseException]
                for _attempt in range(
                    TRANSPORT_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
                ):
                    failure._cleanup_attempt_count = min(
                        MAX_TRANSPORT_CLEANUP_ATTEMPT_COUNT,
                        failure._cleanup_attempt_count + 1,
                    )
                    cleanup_error = _close_transport_required(
                        failure._resource
                    )
                    if cleanup_error is None:
                        resolved.append(failure)
                        break
                    failure.cleanup_error_type = type(cleanup_error).__name__
            with self._transport_health_lock:
                resolved_ids = {id(failure) for failure in resolved}
                self._pending_cleanup_failures[:] = [
                    failure
                    for failure in self._pending_cleanup_failures
                    if id(failure) not in resolved_ids
                ]
                self._cleanup_recovery_count += len(resolved)
                failure = (
                    self._pending_cleanup_failures[0]
                    if self._pending_cleanup_failures
                    else None
                )
        if failure is not None:
            raise InterEnclaveTLSError(
                "inter-enclave client transport cleanup retry failed"
            ) from failure

    def transport_health(self) -> Dict[str, Any]:
        with self._transport_health_lock:
            failures = tuple(self._pending_cleanup_failures)
            failure = failures[0] if failures else None
            return {
                "schema_version": TRANSPORT_HEALTH_SCHEMA_VERSION,
                "status": "error" if failure is not None else "healthy",
                "terminal_failure_latched": failure is not None,
                "retained_resource_count": len(failures),
                "cleanup_recovery_count": self._cleanup_recovery_count,
                "last_primary_error_type": (
                    failure.primary_error_type if failure is not None else ""
                ),
                "last_cleanup_error_type": (
                    failure.cleanup_error_type if failure is not None else ""
                ),
            }

    def _require_transport_healthy(self) -> None:
        self._recover_transport_cleanup_failures()

    def _bound_delivery_attempt(self, connection: Any) -> Any:
        try:
            connection.settimeout(self._delivery_attempt_timeout_seconds)
        except (AttributeError, OSError) as exc:
            cleanup_error = _close_transport_required(connection)
            if cleanup_error is not None:
                failure = InterEnclaveTransportCleanupError(
                    stage="client_timeout_setup_cleanup",
                    primary_error=exc,
                    cleanup_error=cleanup_error,
                    resource=connection,
                )
                self._retain_transport_cleanup_failure(failure)
                raise failure from exc
            raise _RetryableInterEnclaveTransportError(
                "inter-enclave delivery timeout could not be applied"
            ) from exc
        return connection

    def _connect_relay(self) -> Any:
        if self._connector is not None:
            return self._bound_delivery_attempt(self._connector())
        connection = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        self._bound_delivery_attempt(connection)
        try:
            connection.connect((PARENT_CID, RELAY_PORT))
        except OSError as exc:
            cleanup_error = _close_transport_required(connection)
            if cleanup_error is not None:
                failure = InterEnclaveTransportCleanupError(
                    stage="client_connect_cleanup",
                    primary_error=exc,
                    cleanup_error=cleanup_error,
                    resource=connection,
                )
                self._retain_transport_cleanup_failure(failure)
                raise failure from exc
            raise _RetryableInterEnclaveTransportError(
                "inter-enclave parent relay connection failed"
            ) from exc
        return connection

    def call(
        self,
        *,
        target_physical_role: str,
        method: str,
        params: Mapping[str, Any],
        channel_id: str,
    ) -> Dict[str, Any]:
        last_error = None
        for attempt in range(MAX_RPC_DELIVERY_ATTEMPTS):
            try:
                return self._call_once(
                    target_physical_role=target_physical_role,
                    method=method,
                    params=params,
                    channel_id=channel_id,
                )
            except _RetryableInterEnclaveTransportError as exc:
                last_error = exc
                if attempt + 1 < MAX_RPC_DELIVERY_ATTEMPTS:
                    time.sleep(
                        min(
                            RPC_DELIVERY_BACKOFF_SECONDS * (2**attempt),
                            0.4,
                        )
                    )
                    continue
        raise InterEnclaveTLSError(
            "inter-enclave transport failed after bounded replay"
        ) from last_error

    def _call_once(
        self,
        *,
        target_physical_role: str,
        method: str,
        params: Mapping[str, Any],
        channel_id: str,
    ) -> Dict[str, Any]:
        self._require_transport_healthy()
        peer = self.peer_registry.peer(target_physical_role)
        target_spec = role_spec(target_physical_role)
        connection = self._connect_relay()
        tls = None
        result = None  # type: Optional[Dict[str, Any]]
        primary_error = None  # type: Optional[BaseException]
        try:
            _send_frame(
                connection,
                {
                    "schema_version": "leadpoet.inter_enclave_relay.v2",
                    "channel_id": channel_id,
                    "source_cid": int(ROLE_SPECS[self.local_physical_role]["cid"]),
                    "target_cid": int(target_spec["cid"]),
                    "target_port": TLS_SERVICE_PORT,
                    "topology_hash": topology_hash(),
                },
            )
            connected = _read_frame(connection)
            if connected.get("result", {}).get("status") != "connected":
                raise _RetryableInterEnclaveTransportError(
                    "parent relay did not connect target"
                )
            context = create_mutual_tls_context(
                identity_paths=self.identity_paths,
                trusted_peer_certificate_pem=peer["certificate_pem"],
                server_side=False,
            )
            tls = context.wrap_socket(connection, server_hostname=None)
            peer_from_tls = self.peer_registry.peer_for_certificate(
                tls.getpeercert(binary_form=True)
            )
            if peer_from_tls["physical_role"] != target_physical_role:
                raise InterEnclaveTLSError("TLS target role mismatch")
            request = build_rpc_request(
                method=method,
                params=params,
                channel_id=channel_id,
                source_boot_identity_hash=self.local_boot_identity[
                    "boot_identity_hash"
                ],
                target_boot_identity_hash=peer["boot_identity"][
                    "boot_identity_hash"
                ],
            )
            _send_frame(tls, request)
            response = _read_frame(tls)
            if set(response) not in (
                {"result", "channel_id"},
                {"error", "channel_id"},
            ):
                raise InterEnclaveTLSError("inter-enclave response fields are invalid")
            if response["channel_id"] != channel_id:
                raise InterEnclaveTLSError("inter-enclave response channel mismatch")
            if "error" in response:
                error = response["error"]
                if (
                    not isinstance(error, Mapping)
                    or set(error) != {"code", "error_type"}
                    or error["code"] != "remote_handler_failed"
                    or not _ERROR_TYPE_RE.fullmatch(str(error["error_type"]))
                ):
                    raise InterEnclaveTLSError(
                        "inter-enclave response error is invalid"
                    )
                raise InterEnclaveTLSError(
                    "inter-enclave remote handler failed: "
                    + str(error["error_type"])
                )
            result = response["result"]
            if not isinstance(result, Mapping):
                raise InterEnclaveTLSError("inter-enclave response result is invalid")
            result = dict(result)
        except BaseException as exc:
            primary_error = exc
        # Once wrap_socket succeeds the TLS object owns the underlying socket;
        # closing both objects would manufacture a second cleanup boundary.
        owned_transport = tls if tls is not None else connection
        cleanup_error = _close_transport_required(owned_transport)
        if cleanup_error is not None:
            cleanup_primary = primary_error or cleanup_error
            failure = InterEnclaveTransportCleanupError(
                stage="client_rpc_cleanup",
                primary_error=cleanup_primary,
                cleanup_error=cleanup_error,
                resource=owned_transport,
            )
            self._retain_transport_cleanup_failure(failure)
            raise failure from cleanup_primary
        if primary_error is not None:
            raise primary_error
        if result is None:
            raise InterEnclaveTLSError("inter-enclave response result is unavailable")
        return result


class AttestedTLSRPCServer:
    def __init__(
        self,
        *,
        local_physical_role: str,
        local_boot_identity: Mapping[str, Any],
        local_tls_identity: Mapping[str, Any],
        peer_registry: AttestedPeerRegistry,
        handler: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        tmpfs_root: Path = Path("/run/leadpoet-v2"),
    ) -> None:
        self.local_physical_role = local_physical_role
        self.local_boot_identity = dict(local_boot_identity)
        self.peer_registry = peer_registry
        self.handler = handler
        self.identity_paths = write_identity_to_tmpfs(
            local_tls_identity,
            directory=tmpfs_root / local_physical_role,
        )
        self._replay_cache = OrderedDict()
        self._replay_cache_bytes = 0
        self._replay_inflight = {}
        self._replay_lock = threading.Lock()
        self._transport_health_lock = threading.Lock()
        self._transport_recovery_lock = threading.Lock()
        self._transport_cleanup_attempt_count = 0
        self._transport_cleanup_failure_count = 0
        self._last_cleanup_primary_error_type = ""
        self._last_cleanup_error_type = ""
        self._pending_cleanup_failures = []
        self._cleanup_recovery_count = 0
        self._terminal_cleanup_failure_event = threading.Event()

    def _retain_transport_cleanup_failure(
        self,
        failure: InterEnclaveTransportCleanupError,
    ) -> None:
        with self._transport_health_lock:
            self._pending_cleanup_failures.append(failure)
            self._terminal_cleanup_failure_event.set()

    def _recover_transport_cleanup_failures(self) -> bool:
        with self._transport_recovery_lock:
            with self._transport_health_lock:
                snapshot = tuple(self._pending_cleanup_failures)
            resolved = []
            for failure in snapshot:
                cleanup_error = None  # type: Optional[BaseException]
                for _attempt in range(
                    TRANSPORT_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
                ):
                    failure._cleanup_attempt_count = min(
                        MAX_TRANSPORT_CLEANUP_ATTEMPT_COUNT,
                        failure._cleanup_attempt_count + 1,
                    )
                    cleanup_error = _close_transport_required(
                        failure._resource
                    )
                    if cleanup_error is None:
                        resolved.append(failure)
                        break
                    failure.cleanup_error_type = type(cleanup_error).__name__
            with self._transport_health_lock:
                resolved_ids = {id(failure) for failure in resolved}
                self._pending_cleanup_failures[:] = [
                    failure
                    for failure in self._pending_cleanup_failures
                    if id(failure) not in resolved_ids
                ]
                self._cleanup_recovery_count += len(resolved)
                pending = bool(self._pending_cleanup_failures)
                if pending:
                    self._terminal_cleanup_failure_event.set()
                else:
                    self._terminal_cleanup_failure_event.clear()
                return not pending

    def _record_transport_cleanup(
        self,
        *,
        primary_error: Optional[BaseException],
        cleanup_error: Optional[BaseException],
    ) -> None:
        with self._transport_health_lock:
            self._transport_cleanup_attempt_count = min(
                MAX_TRANSPORT_CLEANUP_EVENT_COUNT,
                self._transport_cleanup_attempt_count + 1,
            )
            if cleanup_error is not None:
                self._transport_cleanup_failure_count = min(
                    MAX_TRANSPORT_CLEANUP_EVENT_COUNT,
                    self._transport_cleanup_failure_count + 1,
                )
                self._last_cleanup_primary_error_type = (
                    type(primary_error).__name__
                    if primary_error is not None
                    else type(cleanup_error).__name__
                )
                self._last_cleanup_error_type = type(cleanup_error).__name__

    def transport_health(self) -> Dict[str, Any]:
        """Return a bounded, text-free projection of accepted-RPC cleanup."""

        with self._transport_health_lock:
            failures = tuple(self._pending_cleanup_failures)
            return {
                "schema_version": TRANSPORT_HEALTH_SCHEMA_VERSION,
                "status": (
                    "error"
                    if failures
                    else "healthy"
                ),
                "cleanup_attempt_count": self._transport_cleanup_attempt_count,
                "cleanup_failure_count": self._transport_cleanup_failure_count,
                "last_primary_error_type": self._last_cleanup_primary_error_type,
                "last_cleanup_error_type": self._last_cleanup_error_type,
                "terminal_failure_latched": (
                    bool(failures)
                ),
                "retained_resource_count": len(failures),
                "cleanup_recovery_count": self._cleanup_recovery_count,
            }

    @staticmethod
    def _request_hash(request: Mapping[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json(dict(request)).encode("utf-8")
        ).hexdigest()

    def _purge_replay_cache_locked(self, now: float) -> None:
        expired = [
            key
            for key, record in self._replay_cache.items()
            if now - float(record["completed_at"]) >= REPLAY_CACHE_TTL_SECONDS
        ]
        for key in expired:
            record = self._replay_cache.pop(key)
            self._replay_cache_bytes -= int(record["size"])

    def _cached_response(
        self,
        *,
        request: Mapping[str, Any],
        peer: Mapping[str, Any],
    ) -> Dict[str, Any]:
        request_hash = self._request_hash(request)
        key = (
            str(peer["boot_identity"]["boot_identity_hash"]),
            str(request["channel_id"]),
        )
        with self._replay_lock:
            self._purge_replay_cache_locked(time.monotonic())
            cached = self._replay_cache.get(key)
            if cached is not None:
                if cached["request_hash"] != request_hash:
                    raise InterEnclaveTLSError(
                        "inter-enclave channel was reused with another request"
                    )
                self._replay_cache.move_to_end(key)
                return dict(cached["response"])
            inflight = self._replay_inflight.get(key)
            if inflight is None:
                event = threading.Event()
                self._replay_inflight[key] = {
                    "request_hash": request_hash,
                    "event": event,
                }
                owns_request = True
            else:
                if inflight["request_hash"] != request_hash:
                    raise InterEnclaveTLSError(
                        "inter-enclave channel was reused with another request"
                    )
                event = inflight["event"]
                owns_request = False

        if not owns_request:
            if not event.wait(REPLAY_WAIT_SECONDS):
                raise InterEnclaveTLSError(
                    "inter-enclave replay wait timed out"
                )
            with self._replay_lock:
                cached = self._replay_cache.get(key)
                if cached is None or cached["request_hash"] != request_hash:
                    raise InterEnclaveTLSError(
                        "inter-enclave replay result is unavailable"
                    )
                self._replay_cache.move_to_end(key)
                return dict(cached["response"])

        try:
            try:
                result = self.handler(request["method"], request["params"], peer)
                if not isinstance(result, Mapping):
                    raise InterEnclaveTLSError(
                        "inter-enclave handler result is invalid"
                    )
                response = {
                    "result": dict(result),
                    "channel_id": request["channel_id"],
                }
            except Exception as exc:
                error_type = type(exc).__name__
                if not _ERROR_TYPE_RE.fullmatch(error_type):
                    error_type = "Exception"
                response = {
                    "error": {
                        "code": "remote_handler_failed",
                        "error_type": error_type,
                    },
                    "channel_id": request["channel_id"],
                }
            response_size = len(canonical_json(response).encode("utf-8"))
            if response_size < 2 or response_size > MAX_FRAME_BYTES:
                response = {
                    "error": {
                        "code": "remote_handler_failed",
                        "error_type": "InterEnclaveTLSError",
                    },
                    "channel_id": request["channel_id"],
                }
                response_size = len(canonical_json(response).encode("utf-8"))
            record = {
                "request_hash": request_hash,
                "response": dict(response),
                "size": response_size,
                "completed_at": time.monotonic(),
            }
            with self._replay_lock:
                self._replay_cache[key] = record
                self._replay_cache.move_to_end(key)
                self._replay_cache_bytes += response_size
                while (
                    len(self._replay_cache) > MAX_REPLAY_CACHE_ENTRIES
                    or self._replay_cache_bytes > MAX_REPLAY_CACHE_BYTES
                ):
                    old_key, old_record = self._replay_cache.popitem(last=False)
                    self._replay_cache_bytes -= int(old_record["size"])
                    if old_key == key:
                        raise InterEnclaveTLSError(
                            "inter-enclave replay response exceeds cache limit"
                        )
                self._replay_inflight.pop(key, None)
                event.set()
            return dict(response)
        except BaseException:
            with self._replay_lock:
                self._replay_inflight.pop(key, None)
                event.set()
            raise

    def handle_connection(self, connection: Any) -> None:
        tls = None
        primary_error = None  # type: Optional[BaseException]
        try:
            context = create_mutual_tls_context(
                identity_paths=self.identity_paths,
                trusted_peer_certificate_pem=self.peer_registry.trusted_certificates(),
                server_side=True,
            )
            tls = context.wrap_socket(connection, server_side=True)
            peer = self.peer_registry.peer_for_certificate(
                tls.getpeercert(binary_form=True)
            )
            request = _read_frame(tls)
            validate_rpc_request(
                request,
                source_boot_identity_hash=peer["boot_identity"][
                    "boot_identity_hash"
                ],
                target_boot_identity_hash=self.local_boot_identity[
                    "boot_identity_hash"
                ],
            )
            _send_frame(
                tls,
                self._cached_response(request=request, peer=peer),
            )
        except BaseException as exc:
            primary_error = exc
        owned_transport = tls if tls is not None else connection
        cleanup_error = _close_transport_required(owned_transport)
        self._record_transport_cleanup(
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )
        if cleanup_error is not None:
            cleanup_primary = primary_error or cleanup_error
            failure = InterEnclaveTransportCleanupError(
                stage="server_rpc_cleanup",
                primary_error=cleanup_primary,
                cleanup_error=cleanup_error,
                resource=owned_transport,
            )
            self._retain_transport_cleanup_failure(failure)
            raise failure from cleanup_primary
        if primary_error is not None:
            raise primary_error

    def serve_forever(self, *, listener: Optional[Any] = None) -> None:
        owns_listener = listener is None
        primary_error = None  # type: Optional[BaseException]
        try:
            if listener is None:
                listener = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
                listener.bind((0xFFFFFFFF, TLS_SERVICE_PORT))
                listener.listen(64)
            listener.settimeout(TRANSPORT_SUPERVISOR_POLL_SECONDS)
            while True:
                if self._terminal_cleanup_failure_event.is_set():
                    if not self._recover_transport_cleanup_failures():
                        time.sleep(TRANSPORT_SUPERVISOR_POLL_SECONDS)
                    continue
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if exc.errno in _TRANSIENT_ACCEPT_ERRNOS:
                        continue
                    raise
                if self._terminal_cleanup_failure_event.is_set():
                    cleanup_error = _close_transport_required(connection)
                    self._record_transport_cleanup(
                        primary_error=None,
                        cleanup_error=cleanup_error,
                    )
                    if cleanup_error is not None:
                        self._retain_transport_cleanup_failure(
                            InterEnclaveTransportCleanupError(
                                stage="server_admission_cleanup",
                                primary_error=cleanup_error,
                                cleanup_error=cleanup_error,
                                resource=connection,
                            )
                        )
                    if not self._recover_transport_cleanup_failures():
                        time.sleep(TRANSPORT_SUPERVISOR_POLL_SECONDS)
                    continue

                def _handle_owned_connection(
                    owned_connection: Any = connection,
                ) -> None:
                    try:
                        self.handle_connection(owned_connection)
                    except InterEnclaveTransportCleanupError:
                        # handle_connection retained the failed owner and set
                        # the process-fatal supervisor latch.
                        return

                try:
                    worker = threading.Thread(
                        target=_handle_owned_connection,
                        name="attested-inter-enclave-tls",
                        daemon=True,
                    )
                    worker.start()
                except BaseException as exc:
                    cleanup_error = _close_transport_required(connection)
                    if cleanup_error is not None:
                        failure = InterEnclaveTransportCleanupError(
                            stage="server_thread_start_cleanup",
                            primary_error=exc,
                            cleanup_error=cleanup_error,
                            resource=connection,
                        )
                        self._record_transport_cleanup(
                            primary_error=exc,
                            cleanup_error=cleanup_error,
                        )
                        self._retain_transport_cleanup_failure(failure)
                        if not self._recover_transport_cleanup_failures():
                            raise failure from exc
                    raise
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if owns_listener and listener is not None:
                cleanup_error = _close_transport_required(listener)
                if cleanup_error is not None:
                    cleanup_primary = primary_error or cleanup_error
                    failure = InterEnclaveTransportCleanupError(
                        stage="server_listener_cleanup",
                        primary_error=cleanup_primary,
                        cleanup_error=cleanup_error,
                        resource=listener,
                    )
                    self._retain_transport_cleanup_failure(failure)
                    if not self._recover_transport_cleanup_failures():
                        raise failure from cleanup_primary
