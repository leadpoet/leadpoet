#!/usr/bin/env python3.11
"""
Nitro Enclave TEE Service
=========================

This service runs inside an AWS Nitro Enclave (hardware-isolated trusted execution environment).
It exposes measured identity, attestation, and TLS bootstrap over vsock.

KEY CONCEPTS:
- vsock: Virtual socket for secure parent ↔ enclave communication (no network access)
- CID 16: Enclave's Context Identifier (fixed by AWS Nitro)
- CID 3: Parent EC2's Context Identifier (fixed by AWS Nitro)
- Port 5000: Application-defined port for RPC communication

SECURITY GUARANTEES:
- Enclave has NO network access (cannot reach internet or other VMs)
- Enclave memory is hardware-isolated (parent EC2 cannot read it)
- Private key generated inside enclave NEVER leaves (no export mechanism)
- Attestation document cryptographically proves code integrity
"""

# ULTRA-EARLY DEBUG: Print before ANY imports
print("=" * 80, flush=True)
print("🐛 DEBUG: tee_service.py STARTING (before imports)", flush=True)
print("=" * 80, flush=True)

print("🐛 DEBUG: Importing standard library modules...", flush=True)
import socket
import json
import sys
import os
import errno
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Mapping, Optional
from threading import BoundedSemaphore, Lock
print("🐛 DEBUG: Standard library imports OK", flush=True)

# Cryptography for Ed25519 keypair generation
print("🐛 DEBUG: Importing cryptography...", flush=True)
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
print("🐛 DEBUG: Cryptography imports OK", flush=True)

print("🐛 DEBUG: Importing code hash module...", flush=True)
try:
    from gateway.tee.code_hash import compute_gateway_code_hash
except Exception:
    from code_hash import compute_gateway_code_hash
print("🐛 DEBUG: Code hash module imports OK", flush=True)


# ============================================================================
# VSOCK CONFIGURATION (AWS Nitro Enclaves)
# ============================================================================

# vsock address family constant (Linux)
# See: https://man7.org/linux/man-pages/man7/vsock.7.html
AF_VSOCK = 40  # Address family for vsock

# VMADDR_CID_ANY: Special CID for binding to any address (inside enclave)
# When running INSIDE the enclave, bind to this (not a specific CID)
VMADDR_CID_ANY = 0xFFFFFFFF  # 4294967295 or -1 (cast to u32)

# Parent EC2 CID - always 3 (for reference, not used in binding)
PARENT_CID = 3

# RPC port for communication
RPC_PORT = 5000
MAX_RPC_REQUEST_BYTES = 64 * 1024 * 1024
VSOCK_RPC_LISTEN_BACKLOG = 128
VSOCK_RPC_MAX_CONNECTIONS = 64
VSOCK_RPC_CONNECTION_TIMEOUT_SECONDS = 30.0
VSOCK_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION = (
    "leadpoet.gateway_vsock_rpc_transport_health.v2"
)
MAX_VSOCK_RPC_CLEANUP_EVENT_COUNT = (1 << 63) - 1
VSOCK_RPC_SUPERVISOR_POLL_SECONDS = 0.25
VSOCK_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE = 1
MAX_VSOCK_RPC_CLEANUP_ATTEMPT_COUNT = (1 << 63) - 1
_VSOCK_RPC_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)

# Note: The enclave's actual CID (e.g., 16, 26, 27) is assigned by AWS
# and visible to the parent EC2, but the enclave binds to VMADDR_CID_ANY


# ============================================================================
# GLOBAL STATE (In-Memory, Hardware-Protected)
# ============================================================================

# Ephemeral enclave signing key and measured runtime state.
private_key: Optional[ed25519.Ed25519PrivateKey] = None
public_key: Optional[ed25519.Ed25519PublicKey] = None
keypair_lock = Lock()

pcr_measurements: Dict[str, str] = {
    "PCR0": None,
    "PCR1": None,
    "PCR2": None,
}
pcr_measurements_lock = Lock()

v2_runtime_identity = None
v2_runtime_identity_lock = Lock()
vsock_rpc_transport_health_lock = Lock()
vsock_rpc_cleanup_attempt_count = 0
vsock_rpc_cleanup_failure_count = 0
vsock_rpc_last_primary_error_type = ""
vsock_rpc_last_cleanup_error_type = ""
vsock_rpc_pending_cleanup_failures = []
vsock_rpc_cleanup_recovery_count = 0
vsock_rpc_terminal_failure_event = threading.Event()
vsock_rpc_cleanup_recovery_lock = Lock()

print("=" * 80, flush=True)
print("🐛 DEBUG: All imports and global state OK", flush=True)
print("🐛 DEBUG: Defining functions...", flush=True)
print("=" * 80, flush=True)

# ============================================================================
# KEYPAIR GENERATION (Inside TEE, Never Exported)
# ============================================================================

def generate_keypair() -> None:
    """
    Generate Ed25519 keypair inside the enclave.
    
    CRITICAL SECURITY PROPERTIES:
    - Private key is generated using hardware RNG (/dev/urandom in enclave)
    - Private key is stored in enclave memory ONLY (never written to disk)
    - Private key CANNOT be exported or accessed by parent EC2
    - Public key can be retrieved for verification
    - Keypair is destroyed when enclave terminates (ephemeral)
    
    This ensures the signing key is ONLY accessible to verified enclave code.
    """
    global private_key, public_key
    
    with keypair_lock:
        if private_key is None:
            print("[TEE] Generating Ed25519 keypair inside enclave...", flush=True)
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            print("[TEE] ✅ Keypair generated (private key never leaves enclave)", flush=True)


def get_public_key_bytes() -> bytes:
    """
    Get public key bytes for sharing with verifiers.
    
    Returns:
        Public key in raw bytes (32 bytes for Ed25519)
    """
    if public_key is None:
        generate_keypair()
    
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )


def compute_code_hash() -> str:
    """Compute SHA256 hash of all attested gateway runtime code."""
    try:
        from pathlib import Path
        gateway_root = Path(__file__).parent.parent
        print(f"[TEE] 🔐 Computing code hash from gateway directory: {gateway_root}", flush=True)
        return compute_gateway_code_hash(gateway_root, log_prefix="[TEE]")
    except Exception as e:
        print(f"[TEE] ⚠️  Failed to compute code hash: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return zeros on error (will cause verification to fail, which is correct)
        return "0" * 64


def read_pcrs_from_hardware() -> Dict[str, str]:
    """
    🔴 CRITICAL: Read PCR measurements DIRECTLY from /dev/nsm hardware.
    
    This is the ONLY trustless way to get PCR values. Reading from parent EC2
    allows a malicious operator to lie about what code is running.
    
    Why this matters:
    - A malicious operator could modify tee_service.py
    - Rebuild Docker image → new PCR0_modified
    - Start enclave with modified code
    - Send FAKE PCR0_old via vsock (from legitimate build)
    - Enclave would include fake PCR0 in attestation
    - Miners would think code is legitimate ❌
    
    By reading PCRs from /dev/nsm hardware:
    - Hardware ALWAYS returns REAL PCR0 (cannot be faked)
    - If code modified → hardware returns PCR0_modified
    - Miners see PCR0_modified != PCR0_expected
    - Attack detected ✅
    
    Returns:
        Dict with PCR0, PCR1, PCR2 (hex strings, 96 chars each)
        Returns zeros if /dev/nsm not available (development mode)
    """
    NSM_DEVICE = "/dev/nsm"
    
    print(f"[TEE] DEBUG: Checking for {NSM_DEVICE}...", flush=True)
    
    # Check if NSM device exists (only available inside enclave)
    if not os.path.exists(NSM_DEVICE):
        print(f"[TEE] ⚠️  /dev/nsm not found - using development mode PCRs (zeros)", flush=True)
        return {
            "PCR0": "0" * 96,
            "PCR1": "0" * 96,
            "PCR2": "0" * 96
        }
    
    print(f"[TEE] DEBUG: {NSM_DEVICE} exists, proceeding...", flush=True)
    
    # Give NSM device time to fully initialize (important on first boot)
    print(f"[TEE] DEBUG: Waiting 2 seconds for NSM device to fully initialize...", flush=True)
    import time
    time.sleep(2)
    print(f"[TEE] DEBUG: NSM device should be ready now", flush=True)
    
    try:
        print(f"[TEE] DEBUG: Importing nsm_lib (AWS NSM Python wrapper)...", flush=True)
        import nsm_lib
        print(f"[TEE] DEBUG: nsm_lib imported successfully", flush=True)
        
        print(f"[TEE] 🔒 Reading PCR measurements from /dev/nsm hardware...", flush=True)
        
        # Use the proper NSM library to get PCR measurements
        # This uses the correct ioctl interface with proper request structures
        pcr_dict = nsm_lib.get_pcr_measurements()
        
        print(f"[TEE] ✅ PCRs read from hardware (unfakeable):", flush=True)
        print(f"[TEE]    PCR0: {pcr_dict['PCR0'][:32]}...{pcr_dict['PCR0'][-32:]}", flush=True)
        print(f"[TEE]    PCR1: {pcr_dict['PCR1'][:32]}...{pcr_dict['PCR1'][-32:]}", flush=True)
        print(f"[TEE]    PCR2: {pcr_dict['PCR2'][:32]}...{pcr_dict['PCR2'][-32:]}", flush=True)
        
        return pcr_dict
    
    except Exception as e:
        print(f"[TEE] ❌ Failed to read PCRs from hardware: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Fallback to zeros (development mode)
        print(f"[TEE] ⚠️  Using development mode PCRs (zeros)", flush=True)
        return {
            "PCR0": "0" * 96,
            "PCR1": "0" * 96,
            "PCR2": "0" * 96
        }


# ============================================================================
# RPC HANDLER (vsock Request/Response)
# ============================================================================

def get_v2_runtime_identity():
    global v2_runtime_identity
    with v2_runtime_identity_lock:
        if v2_runtime_identity is not None:
            return v2_runtime_identity
        from gateway.tee.rpc_authority import active_enclave_role
        from gateway.tee.runtime_identity_v2 import RuntimeIdentityV2

        physical_role = active_enclave_role()
        gateway_root = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        v2_runtime_identity = RuntimeIdentityV2(
            gateway_root=gateway_root,
            physical_role=physical_role,
            signing_pubkey_supplier=lambda: get_public_key_bytes().hex(),
            pcr0_supplier=lambda: str(pcr_measurements.get("PCR0") or ""),
        )
        return v2_runtime_identity


def handle_v2_runtime_rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    manager = get_v2_runtime_identity()
    if method == "v2_configure_runtime":
        if not isinstance(params, dict) or set(params) != {
            "schema_version",
            "configuration",
            "configuration_hash",
        }:
            raise ValueError("V2 runtime configuration fields are invalid")
        if params.get("schema_version") != "leadpoet.enclave_runtime_config.v2":
            raise ValueError("V2 runtime configuration schema is invalid")
        return {
            "result": manager.configure(
                configuration=params.get("configuration"),
                expected_config_hash=str(params.get("configuration_hash") or ""),
            )
        }
    if method == "v2_get_boot_identity":
        return {"result": manager.boot_identity()}
    raise ValueError("Unknown V2 runtime method")


def _event_signing_identity() -> Dict[str, Any]:
    """Return the coordinator's measured boot identity for public attestation."""

    boot = get_v2_runtime_identity().boot_identity()
    return {
        "purpose": "gateway_event_signing",
        "enclave_pubkey": boot["signing_pubkey"],
        "code_hash": compute_code_hash(),
        "attestation_document_b64": boot["attestation_document_b64"],
        "signer_state": {
            "status": "ready",
            "boot_identity_hash": boot["boot_identity_hash"],
            "physical_role": boot["physical_role"],
            "commit_sha": boot["commit_sha"],
        },
    }


def handle_rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle one authorized parent-to-enclave trust RPC."""

    try:
        from gateway.tee.rpc_authority import active_enclave_role, rpc_method_allowed

        enclave_role = active_enclave_role()
        if not rpc_method_allowed(enclave_role, method):
            return {
                "error": "RPC method is not authorized for enclave role %s"
                % enclave_role
            }
        if method == "get_event_signing_identity":
            return {"result": _event_signing_identity()}
        if method == "role_health":
            from gateway.tee.build_identity import load_identity
            from gateway.tee.topology import role_spec

            gateway_root = Path(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            identity = load_identity(
                gateway_root=gateway_root,
                expected_role=enclave_role,
            )
            v2_status = get_v2_runtime_identity().public_status()
            return {
                "result": {
                    "status": "healthy",
                    "role": enclave_role,
                    "service_role": role_spec(enclave_role)["service_role"],
                    "commit_sha": identity["commit_sha"],
                    "build_identity_hash": identity["identity_hash"],
                    "execution_manifest_hash": identity["execution_manifest_hash"],
                    "dependency_lock_hash": identity["dependency_lock_hash"],
                    "topology_hash": identity["topology_hash"],
                    "public_key": get_public_key_bytes().hex(),
                    "pcr0": pcr_measurements.get("PCR0"),
                    "v2_runtime": v2_status,
                    "parent_rpc_transport": vsock_rpc_transport_health(),
                }
            }
        if method.startswith("v2_"):
            return handle_v2_runtime_rpc(method, params)
        return {"error": "Unknown method: %s" % method}
    except Exception as exc:
        print("[TEE] RPC error type=%s" % type(exc).__name__, flush=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


# ============================================================================
# VSOCK SERVER (Parent EC2 ↔ Enclave Communication)
# ============================================================================

class VSOCKRPCCleanupError(RuntimeError):
    """An accepted parent RPC socket could not prove descriptor release."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
        resource: Any,
    ) -> None:
        super().__init__("gateway vsock RPC transport cleanup failed")
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        self._cleanup_attempt_count = 1
        # Never serialized. Preserve ownership until the worker failure has
        # been observed by the bounded transport-health latch.
        self._resource = resource


class _ExplicitVSOCKCloseFailure(RuntimeError):
    """A socket adapter explicitly reported retained ownership."""


def _close_vsock_rpc_required(candidate: Any) -> Optional[BaseException]:
    """Attempt full-duplex shutdown and return any close-proof failure."""

    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # The peer can already be half-closed after a complete response.
        # close() is the accepted descriptor's ownership boundary.
        pass
    try:
        if candidate.close() is False:
            return _ExplicitVSOCKCloseFailure(
                "gateway vsock RPC close was not confirmed"
            )
    except BaseException as exc:
        return exc
    return None


def _record_vsock_rpc_cleanup(
    *,
    primary_error: Optional[BaseException],
    cleanup_error: Optional[BaseException],
) -> None:
    global vsock_rpc_cleanup_attempt_count
    global vsock_rpc_cleanup_failure_count
    global vsock_rpc_last_primary_error_type
    global vsock_rpc_last_cleanup_error_type
    with vsock_rpc_transport_health_lock:
        vsock_rpc_cleanup_attempt_count = min(
            MAX_VSOCK_RPC_CLEANUP_EVENT_COUNT,
            vsock_rpc_cleanup_attempt_count + 1,
        )
        if cleanup_error is not None:
            vsock_rpc_cleanup_failure_count = min(
                MAX_VSOCK_RPC_CLEANUP_EVENT_COUNT,
                vsock_rpc_cleanup_failure_count + 1,
            )
            vsock_rpc_last_primary_error_type = (
                type(primary_error).__name__
                if primary_error is not None
                else type(cleanup_error).__name__
            )
            vsock_rpc_last_cleanup_error_type = type(cleanup_error).__name__


def _retain_vsock_rpc_cleanup_failure(
    failure: VSOCKRPCCleanupError,
) -> None:
    with vsock_rpc_transport_health_lock:
        vsock_rpc_pending_cleanup_failures.append(failure)
        vsock_rpc_terminal_failure_event.set()


def _recover_vsock_rpc_cleanup_failures() -> bool:
    global vsock_rpc_cleanup_recovery_count
    with vsock_rpc_cleanup_recovery_lock:
        with vsock_rpc_transport_health_lock:
            snapshot = tuple(vsock_rpc_pending_cleanup_failures)
        resolved = []
        for failure in snapshot:
            cleanup_error = None  # type: Optional[BaseException]
            for _attempt in range(
                VSOCK_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
            ):
                failure._cleanup_attempt_count = min(
                    MAX_VSOCK_RPC_CLEANUP_ATTEMPT_COUNT,
                    failure._cleanup_attempt_count + 1,
                )
                cleanup_error = _close_vsock_rpc_required(failure._resource)
                if cleanup_error is None:
                    resolved.append(failure)
                    break
                failure.cleanup_error_type = type(cleanup_error).__name__
        with vsock_rpc_transport_health_lock:
            resolved_ids = {id(failure) for failure in resolved}
            vsock_rpc_pending_cleanup_failures[:] = [
                failure
                for failure in vsock_rpc_pending_cleanup_failures
                if id(failure) not in resolved_ids
            ]
            vsock_rpc_cleanup_recovery_count += len(resolved)
            pending = bool(vsock_rpc_pending_cleanup_failures)
            if pending:
                vsock_rpc_terminal_failure_event.set()
            else:
                vsock_rpc_terminal_failure_event.clear()
            return not pending


def vsock_rpc_transport_health() -> Dict[str, Any]:
    """Return bounded, text-free accepted-RPC cleanup health."""

    with vsock_rpc_transport_health_lock:
        failures = tuple(vsock_rpc_pending_cleanup_failures)
        return {
            "schema_version": VSOCK_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION,
            "status": (
                "error" if failures else "healthy"
            ),
            "cleanup_attempt_count": vsock_rpc_cleanup_attempt_count,
            "cleanup_failure_count": vsock_rpc_cleanup_failure_count,
            "last_primary_error_type": vsock_rpc_last_primary_error_type,
            "last_cleanup_error_type": vsock_rpc_last_cleanup_error_type,
            "terminal_failure_latched": (
                bool(failures)
            ),
            "retained_resource_count": len(failures),
            "cleanup_recovery_count": vsock_rpc_cleanup_recovery_count,
        }


def _recv_exact(conn: Any, size: int) -> bytes:
    """Read an exact bounded frame segment or return the partial bytes."""

    output = bytearray()
    while len(output) < size:
        chunk = conn.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _handle_vsock_connection(conn: Any, addr: Any) -> None:
    """Handle one bounded RPC without allowing a dead peer to wedge the role."""

    primary_error = None  # type: Optional[BaseException]
    try:
        conn.settimeout(VSOCK_RPC_CONNECTION_TIMEOUT_SECONDS)
        print(f"[TEE] Connection from CID {addr[0]}, port {addr[1]}", flush=True)
        length_bytes = _recv_exact(conn, 4)
        if len(length_bytes) != 4:
            print("[TEE] ⚠️ Invalid request (no length prefix)", flush=True)
            return

        request_length = int.from_bytes(length_bytes, byteorder="big")
        if request_length < 2 or request_length > MAX_RPC_REQUEST_BYTES:
            print(
                f"[TEE] ⚠️ Request length outside limit: {request_length}",
                flush=True,
            )
            return

        request_data = _recv_exact(conn, request_length)
        if len(request_data) != request_length:
            print(
                "[TEE] ⚠️ Incomplete request "
                f"(expected {request_length}, got {len(request_data)})",
                flush=True,
            )
            return

        try:
            request = json.loads(request_data.decode("utf-8"))
            method = request.get("method")
            params = request.get("params", {})
            print(f"[TEE] RPC call: {method}", flush=True)
            response = handle_rpc(method, params)
        except json.JSONDecodeError:
            response = {
                "error": "Invalid JSON",
                "error_type": "JSONDecodeError",
            }

        response_bytes = json.dumps(response).encode("utf-8")
        conn.sendall(
            len(response_bytes).to_bytes(4, byteorder="big")
            + response_bytes
        )
        print(f"[TEE] ✅ Response sent ({len(response_bytes)} bytes)", flush=True)
    except Exception as exc:
        primary_error = exc
        print(
            "[TEE] ❌ Connection error type=%s" % type(exc).__name__,
            flush=True,
        )
    finally:
        cleanup_error = _close_vsock_rpc_required(conn)
        _record_vsock_rpc_cleanup(
            primary_error=primary_error,
            cleanup_error=cleanup_error,
        )
        if cleanup_error is not None:
            cleanup_primary = primary_error or cleanup_error
            failure = VSOCKRPCCleanupError(
                primary_error=cleanup_primary,
                cleanup_error=cleanup_error,
                resource=conn,
            )
            _retain_vsock_rpc_cleanup_failure(failure)
            raise failure from cleanup_primary


def _serve_vsock_connections(
    sock: Any,
    *,
    stop_event: Optional[Any] = None,
    max_connections: int = VSOCK_RPC_MAX_CONNECTIONS,
) -> None:
    """Accept RPCs with bounded concurrency and deterministic backpressure."""

    connection_slots = BoundedSemaphore(max(1, int(max_connections)))
    executor = ThreadPoolExecutor(
        max_workers=max(1, int(max_connections)),
        thread_name_prefix="gateway-vsock-rpc",
    )
    try:
        sock.settimeout(VSOCK_RPC_SUPERVISOR_POLL_SECONDS)
        while stop_event is None or not stop_event.is_set():
            connection_slots.acquire()
            if vsock_rpc_terminal_failure_event.is_set():
                connection_slots.release()
                if not _recover_vsock_rpc_cleanup_failures():
                    time.sleep(VSOCK_RPC_SUPERVISOR_POLL_SECONDS)
                continue
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                connection_slots.release()
                continue
            except OSError as exc:
                connection_slots.release()
                if stop_event is not None and stop_event.is_set():
                    break
                if exc.errno in _VSOCK_RPC_TRANSIENT_ACCEPT_ERRNOS:
                    continue
                raise
            except Exception:
                connection_slots.release()
                if stop_event is not None and stop_event.is_set():
                    break
                raise
            if vsock_rpc_terminal_failure_event.is_set():
                connection_slots.release()
                cleanup_error = _close_vsock_rpc_required(conn)
                _record_vsock_rpc_cleanup(
                    primary_error=None,
                    cleanup_error=cleanup_error,
                )
                if cleanup_error is not None:
                    _retain_vsock_rpc_cleanup_failure(
                        VSOCKRPCCleanupError(
                            primary_error=cleanup_error,
                            cleanup_error=cleanup_error,
                            resource=conn,
                        )
                    )
                if not _recover_vsock_rpc_cleanup_failures():
                    time.sleep(VSOCK_RPC_SUPERVISOR_POLL_SECONDS)
                continue
            try:
                future = executor.submit(_handle_vsock_connection, conn, addr)
            except BaseException as exc:
                connection_slots.release()
                cleanup_error = _close_vsock_rpc_required(conn)
                _record_vsock_rpc_cleanup(
                    primary_error=exc,
                    cleanup_error=cleanup_error,
                )
                if cleanup_error is not None:
                    failure = VSOCKRPCCleanupError(
                        primary_error=exc,
                        cleanup_error=cleanup_error,
                        resource=conn,
                    )
                    _retain_vsock_rpc_cleanup_failure(failure)
                    _recover_vsock_rpc_cleanup_failures()
                raise

            def _release_connection_slot(completed_future: Any) -> None:
                try:
                    worker_error = completed_future.exception()
                except BaseException as exc:
                    worker_error = exc
                if worker_error is not None:
                    print(
                        "[TEE] vsock RPC worker terminal type=%s"
                        % type(worker_error).__name__,
                        flush=True,
                    )
                connection_slots.release()

            future.add_done_callback(_release_connection_slot)
    finally:
        executor.shutdown(wait=True)


def start_vsock_server():
    """Start the bounded parent-to-enclave RPC service."""

    print("[TEE] Starting vsock server...", flush=True)
    print(f"[TEE] Binding to VMADDR_CID_ANY (any CID), port {RPC_PORT}", flush=True)
    print("[TEE] DEBUG: Creating vsock socket...", flush=True)
    try:
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        print("[TEE] DEBUG: Socket created successfully", flush=True)
    except Exception as exc:
        print(f"[TEE] ❌ ERROR creating socket: {exc}", flush=True)
        raise

    print(f"[TEE] DEBUG: Binding to ({VMADDR_CID_ANY}, {RPC_PORT})...", flush=True)
    try:
        sock.bind((VMADDR_CID_ANY, RPC_PORT))
        print("[TEE] DEBUG: Bind successful", flush=True)
        print("[TEE] DEBUG: Starting to listen...", flush=True)
        sock.listen(VSOCK_RPC_LISTEN_BACKLOG)
        print("[TEE] DEBUG: Listen successful", flush=True)
    except Exception as exc:
        print(f"[TEE] ❌ ERROR starting socket: {exc}", flush=True)
        sock.close()
        raise

    print("[TEE] ✅ vsock server started", flush=True)
    print("[TEE] Ready to accept RPC calls from parent EC2", flush=True)
    try:
        _serve_vsock_connections(sock)
    finally:
        sock.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for TEE service.
    
    Steps:
    1. Generate Ed25519 keypair inside enclave
    2. 🔴 Read PCR measurements from /dev/nsm hardware (CRITICAL FOR TRUSTLESSNESS)
    3. Start vsock server to listen for RPC calls
    4. Handle measured identity, attestation, and TLS requests
    """
    global pcr_measurements
    
    print("=" * 80, flush=True)
    print("🔒 NITRO ENCLAVE TEE SERVICE STARTING", flush=True)
    print("=" * 80, flush=True)
    print(f"[TEE] DEBUG: Python version: {sys.version}", flush=True)
    print(f"[TEE] DEBUG: Current working directory: {os.getcwd()}", flush=True)
    print(f"[TEE] Binding to: VMADDR_CID_ANY (0xFFFFFFFF)", flush=True)
    print(f"[TEE] Parent CID: {PARENT_CID}", flush=True)
    print(f"[TEE] RPC Port: {RPC_PORT}", flush=True)
    print("=" * 80, flush=True)
    
    # Step 1: Generate keypair on startup
    print("[TEE] DEBUG: Starting keypair generation...", flush=True)
    try:
        generate_keypair()
        print("[TEE] DEBUG: Keypair generation completed successfully", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR in keypair generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: 🔴 CRITICAL - Read PCR measurements from hardware
    print("=" * 80, flush=True)
    print("🔴 READING PCR MEASUREMENTS FROM HARDWARE", flush=True)
    print("=" * 80, flush=True)
    print("[TEE] DEBUG: About to read PCRs from hardware...", flush=True)
    try:
        global pcr_measurements  # CRITICAL: Update global variable, not local!
        with pcr_measurements_lock:
            pcr_measurements = read_pcrs_from_hardware()
        print("[TEE] DEBUG: PCR reading completed successfully", flush=True)
        print(f"[TEE] DEBUG: PCRs: {list(pcr_measurements.keys())}", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR reading PCRs: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("=" * 80, flush=True)
    
    # Step 3: Start vsock server (blocks forever)
    print("[TEE] DEBUG: About to start vsock server...", flush=True)
    try:
        start_vsock_server()
    except Exception as e:
        print(f"[TEE] ❌ ERROR starting vsock server: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[TEE] Shutting down...", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n[TEE] ❌ Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
