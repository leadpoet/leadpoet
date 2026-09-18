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
    
    async def get_event_signing_identity(self) -> Dict:
        """Return the Nitro-bound coordinator runtime identity."""
        return await self._send_rpc("get_event_signing_identity", {})

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

    def close(self):
        """Compatibility no-op: RPC sockets are call-scoped and self-closing."""


# The fixed CID is part of the measured coordinator topology. Public
# attestation callers retain ``tee_client`` as the coordinator alias.
coordinator_tee_client = TEEClient(cid=16)
tee_client = coordinator_tee_client
