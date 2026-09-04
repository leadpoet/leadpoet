"""
Client for communicating with TEE enclave via vsock.

This module provides an async interface for the parent EC2 instance to
communicate with the Nitro Enclave running the TEE service.

vsock (Virtual Socket) is a socket protocol designed for VM-to-host communication,
providing a secure channel between the parent EC2 and the enclave.
"""

import socket
import json
import asyncio
import subprocess
import base64
import hashlib
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime


# vsock address family constant (Linux)
AF_VSOCK = 40  # socket.AF_VSOCK on Linux systems

# Parent EC2 CID (reserved)
PARENT_CID = 3

# RPC port for TEE communication
RPC_PORT = 5000
MAX_RPC_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 256 * 1024 * 1024
TEE_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION = "leadpoet.tee_rpc_transport_health.v2"
TEE_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE = 1
MAX_TEE_RPC_CLEANUP_ATTEMPT_COUNT = (1 << 63) - 1
_TEE_RPC_TRANSPORT_LOCK = threading.Lock()
_TEE_RPC_RECOVERY_LOCK = threading.Lock()
_tee_rpc_pending_cleanup_failures: List[Any] = []
_tee_rpc_cleanup_recovery_count = 0


class TEEEnclaveRPCError(RuntimeError):
    """A structured error returned by the enclave RPC handler."""

    def __init__(self, message: str, *, error_type: str = "") -> None:
        super().__init__(f"Enclave error: {message}")
        self.error_type = str(error_type or "")


class TEETransportCleanupError(RuntimeError):
    """A one-shot enclave RPC socket could not prove descriptor release."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
        resource: Any,
    ) -> None:
        super().__init__("enclave RPC transport cleanup failed")
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        self._cleanup_attempt_count = 1
        # Keep the still-owned descriptor reachable without serializing it.
        self._resource = resource


class TEETransportUnavailableError(RuntimeError):
    """The host process retained an unresolved one-shot RPC transport."""


class _ExplicitCloseFailure(RuntimeError):
    """A socket adapter explicitly reported retained ownership."""


def _close_rpc_socket_required(candidate: Any) -> Optional[BaseException]:
    """Attempt full-duplex shutdown and return any close-proof failure."""

    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # The enclave normally closes its write side after the response.
        # close() remains the descriptor-release boundary.
        pass
    try:
        if candidate.close() is False:
            return _ExplicitCloseFailure("enclave RPC close was not confirmed")
    except BaseException as exc:
        return exc
    return None


def _retain_tee_rpc_cleanup_failure(
    failure: TEETransportCleanupError,
) -> None:
    with _TEE_RPC_TRANSPORT_LOCK:
        _tee_rpc_pending_cleanup_failures.append(failure)


def _recover_tee_rpc_cleanup_failures() -> None:
    global _tee_rpc_cleanup_recovery_count
    with _TEE_RPC_RECOVERY_LOCK:
        with _TEE_RPC_TRANSPORT_LOCK:
            snapshot = tuple(_tee_rpc_pending_cleanup_failures)
        resolved = []
        for failure in snapshot:
            cleanup_error = None  # type: Optional[BaseException]
            for _attempt in range(
                TEE_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
            ):
                failure._cleanup_attempt_count = min(
                    MAX_TEE_RPC_CLEANUP_ATTEMPT_COUNT,
                    failure._cleanup_attempt_count + 1,
                )
                cleanup_error = _close_rpc_socket_required(failure._resource)
                if cleanup_error is None:
                    resolved.append(failure)
                    break
                failure.cleanup_error_type = type(cleanup_error).__name__
        with _TEE_RPC_TRANSPORT_LOCK:
            resolved_ids = {id(failure) for failure in resolved}
            _tee_rpc_pending_cleanup_failures[:] = [
                failure
                for failure in _tee_rpc_pending_cleanup_failures
                if id(failure) not in resolved_ids
            ]
            _tee_rpc_cleanup_recovery_count += len(resolved)
            failure = (
                _tee_rpc_pending_cleanup_failures[0]
                if _tee_rpc_pending_cleanup_failures
                else None
            )
    if failure is not None:
        raise TEETransportUnavailableError(
            "enclave RPC transport cleanup retry failed"
        ) from failure


def tee_rpc_transport_health() -> Dict[str, Any]:
    """Return the process-wide, text-free one-shot RPC cleanup latch."""

    with _TEE_RPC_TRANSPORT_LOCK:
        failures = tuple(_tee_rpc_pending_cleanup_failures)
        failure = failures[0] if failures else None
        return {
            "schema_version": TEE_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION,
            "status": "error" if failure is not None else "healthy",
            "terminal_failure_latched": failure is not None,
            "retained_resource_count": len(failures),
            "cleanup_recovery_count": _tee_rpc_cleanup_recovery_count,
            "last_primary_error_type": (
                failure.primary_error_type if failure is not None else ""
            ),
            "last_cleanup_error_type": (
                failure.cleanup_error_type if failure is not None else ""
            ),
        }


def _require_tee_rpc_transport_healthy() -> None:
    _recover_tee_rpc_cleanup_failures()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


class TEEClient:
    """
    Async client for vsock RPC communication with TEE enclave.
    
    The enclave's CID (Context ID) is dynamically assigned by AWS and can be
    retrieved using `nitro-cli describe-enclaves`.
    """
    
    def __init__(self, cid: Optional[int] = None, port: int = RPC_PORT):
        """
        Initialize TEE client.
        
        Args:
            cid: Enclave CID (if None, will be auto-detected)
            port: vsock port number (default: 5000)
        """
        self.cid = cid
        self.port = port
        # Only CID discovery is shared mutable state. Locks are created lazily
        # for the active loop so importing this host module from an enclave RPC
        # worker never requires an implicit event loop.
        self._cid_locks: Dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._cid_locks_guard = threading.Lock()
    
    async def _get_enclave_cid(self) -> Optional[int]:
        """
        Auto-detect enclave CID from nitro-cli.
        
        Returns:
            Enclave CID or None if no enclave running
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "sudo", "nitro-cli", "describe-enclaves",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                print(f"⚠️ nitro-cli error: {stderr.decode()}")
                return None
            
            enclaves = json.loads(stdout.decode())
            
            if not enclaves:
                print("⚠️ No enclaves running")
                return None
            
            cid = enclaves[0].get("EnclaveCID")
            print(f"✅ Detected enclave CID: {cid}")
            return cid
        
        except Exception as e:
            print(f"❌ Failed to get enclave CID: {e}")
            return None
    
    async def _resolved_cid(self) -> int:
        """
        Resolve the enclave CID once without serializing independent RPCs.
        """
        if self.cid is not None:
            return int(self.cid)
        loop = asyncio.get_running_loop()
        with self._cid_locks_guard:
            lock = self._cid_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._cid_locks[loop] = lock
        async with lock:
            if self.cid is None:
                self.cid = await self._get_enclave_cid()
                if self.cid is None:
                    raise RuntimeError("No enclave running - cannot connect")
            return int(self.cid)

    def _send_rpc_blocking(
        self,
        *,
        cid: int,
        request_bytes: bytes,
    ) -> Dict:
        """Perform one complete RPC on one socket owned by this call."""

        _require_tee_rpc_transport_healthy()
        rpc_socket: Optional[socket.socket] = None
        missing_result = object()
        result: Any = missing_result
        primary_error = None  # type: Optional[BaseException]
        try:
            try:
                rpc_socket = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
                rpc_socket.settimeout(30.0)
                rpc_socket.connect((cid, self.port))
            except Exception as exc:
                raise RuntimeError(f"Failed to connect to enclave: {exc}") from exc

            request_length = len(request_bytes)
            rpc_socket.sendall(
                request_length.to_bytes(4, byteorder="big") + request_bytes
            )

            response_length_bytes = _recv_exact(rpc_socket, 4)
            if len(response_length_bytes) != 4:
                raise RuntimeError("Failed to read response length")

            response_length = int.from_bytes(
                response_length_bytes, byteorder="big"
            )
            if response_length < 2 or response_length > MAX_RPC_RESPONSE_BYTES:
                raise RuntimeError("RPC response size is outside the allowed range")

            response_bytes = _recv_exact(rpc_socket, response_length)
            if len(response_bytes) != response_length:
                raise RuntimeError("Connection closed by enclave")

            response = json.loads(response_bytes.decode("utf-8"))
            if response.get("status") == "error" or "error" in response:
                raise TEEEnclaveRPCError(
                    str(response.get("error") or "unknown enclave error"),
                    error_type=str(response.get("error_type") or ""),
                )
            result = response.get("result", {})
        except BaseException as exc:
            primary_error = exc
        if rpc_socket is not None:
            cleanup_error = _close_rpc_socket_required(rpc_socket)
            if cleanup_error is not None:
                cleanup_primary = primary_error or cleanup_error
                failure = TEETransportCleanupError(
                    primary_error=cleanup_primary,
                    cleanup_error=cleanup_error,
                    resource=rpc_socket,
                )
                _retain_tee_rpc_cleanup_failure(failure)
                raise failure from cleanup_primary
        if primary_error is not None:
            if isinstance(primary_error, RuntimeError):
                raise primary_error
            if isinstance(primary_error, Exception):
                raise RuntimeError(f"RPC failed: {primary_error}") from primary_error
            raise primary_error
        if result is missing_result:
            raise RuntimeError("RPC result is unavailable")
        return result
    
    async def _send_rpc(self, method: str, params: Optional[Dict] = None) -> Dict:
        """
        Send RPC request to enclave and wait for response.
        
        Protocol:
        - Send: {"method": "method_name", "params": {...}}
        - Receive: {"status": "success", "result": ...} or {"status": "error", "error": "..."}
        
        Args:
            method: RPC method name
            params: Optional parameters dict
        
        Returns:
            Result dict from enclave
        
        Raises:
            RuntimeError: If RPC fails or enclave returns error
        """
        # Build RPC request
        request = {
            "method": method,
            "params": params or {}
        }
        
        # Serialize to JSON
        request_json = json.dumps(request)
        request_bytes = request_json.encode('utf-8')
        
        # Send request (with length prefix)
        request_length = len(request_bytes)
        if request_length < 2 or request_length > MAX_RPC_REQUEST_BYTES:
            raise RuntimeError("RPC request size is outside the allowed range")
        cid = await self._resolved_cid()
        # Socket calls are blocking. Keeping them off the event loop allows
        # maintenance-lease heartbeats and cancellation logic to keep running
        # while an enclave request is in flight.
        return await asyncio.to_thread(
            self._send_rpc_blocking,
            cid=cid,
            request_bytes=request_bytes,
        )
    
    async def append_event(self, event: Dict) -> Dict:
        """
        Append event to TEE buffer.
        
        Args:
            event: Event dict (without sequence number)
        
        Returns:
            {"status": "buffered", "sequence": N}
        """
        return await self._send_rpc("append_event", {"event": event})

    async def initialize_event_signer(
        self, prev_log_tip_hash: Optional[str]
    ) -> Dict:
        """Initialize the coordinator-enclave transparency signer once per boot."""
        return await self._send_rpc(
            "initialize_event_signer",
            {"prev_log_tip_hash": prev_log_tip_hash},
        )

    async def sign_transparency_event(
        self,
        *,
        event_type: str,
        payload: Dict,
        payload_hash: str,
    ) -> Dict:
        """Create, sign, and buffer one transparency event in the enclave."""
        return await self._send_rpc(
            "sign_transparency_event",
            {
                "event_type": event_type,
                "payload": payload,
                "payload_hash": payload_hash,
            },
        )

    async def get_event_signing_identity(self) -> Dict:
        """Return the Nitro-bound public identity for transparency signatures."""
        return await self._send_rpc("get_event_signing_identity", {})
    
    async def get_buffer(self) -> List[Dict]:
        """
        Retrieve all buffered events from TEE.
        
        Returns:
            List of event dicts
        """
        result = await self._send_rpc("get_buffer", {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            events = result.get("events", [])
            return events if isinstance(events, list) else []
        return []
    
    async def get_buffer_size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of events in buffer
        """
        result = await self._send_rpc("get_buffer_size", {})
        # Result is directly an int, not a dict
        return result if isinstance(result, int) else 0
    
    async def clear_buffer(self) -> Dict:
        """
        Clear buffer after successful Arweave upload.
        
        Returns:
            {"status": "cleared", "cleared_events": N, "next_checkpoint_at": "ISO8601"}
        """
        return await self._send_rpc("clear_buffer", {})

    async def acknowledge_checkpoint(
        self,
        *,
        checkpoint_number: int,
        merkle_root: str,
        sequence_range: Dict,
    ) -> Dict:
        """Commit one confirmed checkpoint and remove its exact event prefix."""

        return await self._send_rpc(
            "acknowledge_checkpoint",
            {
                "checkpoint_number": int(checkpoint_number),
                "merkle_root": str(merkle_root),
                "sequence_range": dict(sequence_range),
            },
        )
    
    async def get_public_key(self) -> bytes:
        """
        Get enclave's Ed25519 public key.
        
        Returns:
            32-byte public key (raw bytes)
        """
        result = await self._send_rpc("get_public_key", {})
        public_key_hex = result.get("public_key")
        return bytes.fromhex(public_key_hex)
    
    async def get_attestation(self) -> Dict:
        """
        Get attestation document from enclave.
        
        Returns:
            {
                "attestation_document": "hex",
                "public_key": "hex",
                "code_hash": "hex",
                "pcr0": "hex",
                "pcr1": "hex",
                "pcr2": "hex"
            }
        """
        return await self._send_rpc("get_attestation", {})

    async def role_health(self) -> Dict:
        """Return the measured physical role and build identity for this CID."""
        return await self._send_rpc("role_health", {})

    async def v2_configure_runtime(
        self,
        *,
        configuration: Dict,
        configuration_hash: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_configure_runtime",
            {
                "schema_version": "leadpoet.enclave_runtime_config.v2",
                "configuration": configuration,
                "configuration_hash": configuration_hash,
            },
        )

    async def v2_get_boot_identity(self) -> Dict:
        return await self._send_rpc("v2_get_boot_identity", {})

    async def v2_get_transport_certificate(self) -> bytes:
        result = await self._send_rpc("v2_get_transport_certificate", {})
        try:
            return base64.b64decode(
                str(result["certificate_pem_b64"]),
                validate=True,
            )
        except Exception as exc:
            raise RuntimeError("enclave V2 transport certificate is invalid") from exc

    async def v2_register_peer(
        self,
        *,
        boot_identity: Dict,
        certificate_pem: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "v2_register_peer",
            {
                "boot_identity": boot_identity,
                "certificate_pem_b64": base64.b64encode(certificate_pem).decode(
                    "ascii"
                ),
            },
        )

    async def v2_start_tls_service(self) -> Dict:
        return await self._send_rpc("v2_start_tls_service", {})

    async def v2_peer_status(self) -> Dict:
        return await self._send_rpc("v2_peer_status", {})

    async def v2_call_peer_health(self, physical_role: str) -> Dict:
        return await self._send_rpc(
            "v2_call_peer_health",
            {"physical_role": physical_role},
        )

    async def v2_provider_broker_health(self) -> Dict:
        return await self._send_rpc("v2_provider_broker_health", {})

    async def v2_provider_semantics_health(self) -> Dict:
        return await self._send_rpc("v2_provider_semantics_health", {})

    async def v2_get_kms_recipient(self, credential_slot: str) -> Dict:
        return await self._send_rpc(
            "v2_get_kms_recipient",
            {
                "credential_slot": credential_slot,
            },
        )

    async def v2_get_source_add_ingress_recipient(
        self,
        *,
        miner_hotkey: str,
        adapter_ref: str,
        credential_ref: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_get_source_add_ingress_recipient",
            {
                "miner_hotkey": miner_hotkey,
                "adapter_ref": adapter_ref,
                "credential_ref": credential_ref,
            },
        )

    async def v2_seal_source_add_ingress_credential(
        self,
        *,
        request_id: str,
        ciphertext_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_seal_source_add_ingress_credential",
            {
                "request_id": request_id,
                "ciphertext_b64": ciphertext_b64,
            },
        )

    async def v2_provision_encrypted_secret(
        self,
        *,
        credential_slot: str,
        ciphertext_for_recipient_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_encrypted_secret",
            {
                "credential_slot": credential_slot,
                "ciphertext_for_recipient_b64": ciphertext_for_recipient_b64,
            },
        )

    async def v2_get_job_kms_recipient(
        self,
        *,
        job_id: str,
        credential_slot: str,
        credential_value_hash: str,
        key_ref_hash: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_get_job_kms_recipient",
            {
                "job_id": job_id,
                "credential_slot": credential_slot,
                "credential_value_hash": credential_value_hash,
                "key_ref_hash": key_ref_hash,
            },
        )

    async def v2_provision_job_encrypted_secret(
        self,
        *,
        request_id: str,
        ciphertext_for_recipient_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_job_encrypted_secret",
            {
                "request_id": request_id,
                "ciphertext_for_recipient_b64": ciphertext_for_recipient_b64,
            },
        )

    async def v2_provision_job_sealed_source_add_secret(
        self,
        *,
        envelope: Dict,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_job_sealed_source_add_secret",
            {"envelope": envelope},
        )

    async def v2_release_job_credentials(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "v2_release_job_credentials",
            {"job_id": job_id},
        )

    async def v2_list_encrypted_artifacts(self, *, job_id: str, purpose: str) -> Dict:
        return await self._send_rpc(
            "v2_list_encrypted_artifacts",
            {"job_id": job_id, "purpose": purpose},
        )

    async def v2_export_encrypted_artifact(self, artifact_id: str) -> Dict:
        return await self._send_rpc(
            "v2_export_encrypted_artifact",
            {"artifact_id": artifact_id},
        )

    async def v2_verify_encrypted_artifact_persistence(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        artifact_ref: str,
        get_url: str,
        head_url: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_verify_encrypted_artifact_persistence",
            {
                "artifact_id": artifact_id,
                "attestation_job_id": attestation_job_id,
                "artifact_ref": artifact_ref,
                "get_url": get_url,
                "head_url": head_url,
            },
        )
    
    async def get_buffer_stats(self) -> Dict:
        """
        Get buffer statistics from TEE.
        
        Returns:
            {
                "size": int,
                "sequence_range": {"first": int, "last": int},
                "age_seconds": float,
                ...
            }
        """
        return await self._send_rpc("get_buffer_stats", {})
    
    async def build_checkpoint(self) -> Dict:
        """
        Request checkpoint from TEE (for hourly batching).
        
        Returns:
            {
                "header": {...},
                "signature": "hex",
                "events": [...],
                "tree_levels": [...]
            }
        """
        return await self._send_rpc("build_checkpoint", {})

    async def scoring_v2_health(self) -> Dict:
        return await self._send_rpc("scoring_v2_health", {})

    async def scoring_v2_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc("scoring_v2_submit_job", {"manifest": manifest})

    async def scoring_v2_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_v2_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def scoring_v2_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_seal_job", {"job_id": job_id})

    async def scoring_v2_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_get_status", {"job_id": job_id})

    async def scoring_v2_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_cancel_job", {"job_id": job_id})

    async def scoring_v2_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_v2_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def scoring_v2_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_get_receipt", {"job_id": job_id})

    async def scoring_v2_get_receipts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc("scoring_v2_get_receipts", {"job_id": job_id})

    async def scoring_v2_get_ancestry_compact_proof(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "scoring_v2_get_ancestry_compact_proof", {"job_id": job_id}
        )

    async def scoring_v2_get_transitions(self, job_id: str) -> List[Dict]:
        return await self._send_rpc("scoring_v2_get_transitions", {"job_id": job_id})

    async def scoring_v2_get_transport_attempts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "scoring_v2_get_transport_attempts",
            {"job_id": job_id},
        )

    async def scoring_v2_get_artifact_hashes(self, job_id: str) -> List[str]:
        return await self._send_rpc(
            "scoring_v2_get_artifact_hashes", {"job_id": job_id}
        )

    async def scoring_v2_get_host_operations(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "scoring_v2_get_host_operations", {"job_id": job_id}
        )

    async def coordinator_v2_health(self) -> Dict:
        return await self._send_rpc("coordinator_v2_health", {})

    async def coordinator_v2_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_submit_job", {"manifest": manifest}
        )

    async def coordinator_v2_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def coordinator_v2_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_seal_job", {"job_id": job_id})

    async def coordinator_v2_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_get_status", {"job_id": job_id})

    async def coordinator_v2_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_cancel_job", {"job_id": job_id})

    async def coordinator_v2_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def coordinator_v2_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_get_receipt", {"job_id": job_id})

    async def coordinator_v2_get_receipts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_receipts", {"job_id": job_id}
        )

    async def coordinator_v2_get_ancestry_compact_proof(
        self, job_id: str
    ) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_get_ancestry_compact_proof", {"job_id": job_id}
        )

    async def coordinator_v2_get_transitions(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_transitions", {"job_id": job_id}
        )

    async def coordinator_v2_get_transport_attempts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_transport_attempts",
            {"job_id": job_id},
        )

    async def coordinator_v2_get_artifact_hashes(self, job_id: str) -> List[str]:
        return await self._send_rpc(
            "coordinator_v2_get_artifact_hashes", {"job_id": job_id}
        )

    async def coordinator_v2_get_host_operations(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_host_operations", {"job_id": job_id}
        )

    def close(self):
        """Compatibility no-op: RPC sockets are call-scoped and self-closing."""


# Fixed CIDs are part of the measured V2 topology. Existing event/checkpoint
# callers retain ``tee_client`` as a coordinator alias.
coordinator_tee_client = TEEClient(cid=16)
scoring_tee_client = TEEClient(cid=17)
scoring_tee_clients = (scoring_tee_client,)
tee_client = coordinator_tee_client
