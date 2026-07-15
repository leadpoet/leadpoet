"""Mutually attested TLS 1.3 RPC carried by the opaque parent relay."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import socket
import ssl
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from gateway.tee.mtls_identity import (
    create_mutual_tls_context,
    verify_peer_certificate_binding,
    write_identity_to_tmpfs,
)
from gateway.tee.topology import ROLE_SPECS, role_spec, topology_hash
from gateway.utils.tee_client import AF_VSOCK, _recv_exact
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    validate_boot_identity,
    verify_boot_identity_nitro,
)


PARENT_CID = 3
RELAY_PORT = 5002
TLS_SERVICE_PORT = 5003
MAX_FRAME_BYTES = 64 * 1024 * 1024
SCHEMA_VERSION = "leadpoet.inter_enclave_rpc.v2"
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class InterEnclaveTLSError(RuntimeError):
    """A peer, certificate, frame, or topology binding is invalid."""


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
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _read_frame(connection: Any) -> Dict[str, Any]:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise InterEnclaveTLSError("inter-enclave frame prefix is incomplete")
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_FRAME_BYTES:
        raise InterEnclaveTLSError("inter-enclave frame is outside size limit")
    encoded = _recv_exact(connection, size)
    if len(encoded) != size:
        raise InterEnclaveTLSError("inter-enclave frame body is incomplete")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise InterEnclaveTLSError("inter-enclave frame JSON is invalid") from exc
    if not isinstance(value, dict) or canonical_json(value).encode("utf-8") != encoded:
        raise InterEnclaveTLSError("inter-enclave frame is not canonical")
    return value


def _pair_allowed(source_role: str, target_role: str) -> bool:
    if source_role == "gateway_coordinator":
        return target_role in {
            "gateway_scoring",
            "gateway_autoresearch",
        }
    if target_role == "gateway_coordinator":
        return source_role in {
            "gateway_scoring",
            "gateway_autoresearch",
        }
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

    def _connect_relay(self) -> Any:
        if self._connector is not None:
            return self._connector()
        connection = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        connection.connect((PARENT_CID, RELAY_PORT))
        return connection

    def call(
        self,
        *,
        target_physical_role: str,
        method: str,
        params: Mapping[str, Any],
        channel_id: str,
    ) -> Dict[str, Any]:
        peer = self.peer_registry.peer(target_physical_role)
        target_spec = role_spec(target_physical_role)
        connection = self._connect_relay()
        tls = None
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
                raise InterEnclaveTLSError("parent relay did not connect target")
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
            if set(response) != {"result", "channel_id"}:
                raise InterEnclaveTLSError("inter-enclave response fields are invalid")
            if response["channel_id"] != channel_id:
                raise InterEnclaveTLSError("inter-enclave response channel mismatch")
            result = response["result"]
            if not isinstance(result, Mapping):
                raise InterEnclaveTLSError("inter-enclave response result is invalid")
            return dict(result)
        finally:
            for candidate in (tls, connection):
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass


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

    def handle_connection(self, connection: Any) -> None:
        context = create_mutual_tls_context(
            identity_paths=self.identity_paths,
            trusted_peer_certificate_pem=self.peer_registry.trusted_certificates(),
            server_side=True,
        )
        tls = context.wrap_socket(connection, server_side=True)
        try:
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
            result = self.handler(request["method"], request["params"], peer)
            if not isinstance(result, Mapping):
                raise InterEnclaveTLSError("inter-enclave handler result is invalid")
            _send_frame(
                tls,
                {"result": dict(result), "channel_id": request["channel_id"]},
            )
        finally:
            try:
                tls.close()
            except Exception:
                pass

    def serve_forever(self, *, listener: Optional[Any] = None) -> None:
        if listener is None:
            listener = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
            listener.bind((0xFFFFFFFF, TLS_SERVICE_PORT))
            listener.listen(64)
        while True:
            connection, _ = listener.accept()
            threading.Thread(
                target=self.handle_connection,
                args=(connection,),
                name="attested-inter-enclave-tls",
                daemon=True,
            ).start()
