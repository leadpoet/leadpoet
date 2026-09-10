"""
Validator TEE vsock Client
==========================

This module runs on the HOST (parent EC2) and communicates with the 
validator enclave via vsock.

Only Arena hotkey, weight, recovery, outcome, and health RPCs are exposed.
"""

import socket
import json
import os
import subprocess
import sys
import threading
import time
import zlib
from typing import Dict, Any, Optional
from leadpoet_observability import (
    capture_failure,
    failure_code_for_exception,
    record_retry,
    record_stage,
)

# vsock constants
AF_VSOCK = 40
PARENT_CID = 3
RPC_PORT = 5001  # Must match tee_service.py
# Larger logical responses use a versioned compressed frame with an explicit
# expanded-size ceiling.
MAX_RPC_REQUEST_FRAME_BYTES = 16 * 1024 * 1024
MAX_RPC_RESPONSE_FRAME_BYTES = 16 * 1024 * 1024
MAX_RPC_REQUEST_BYTES = 128 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 128 * 1024 * 1024
_COMPRESSED_FRAME_MAGIC = b"LPZ2"
_COMPRESSED_FRAME_HEADER_BYTES = 8


class ValidatorEnclaveTransportCleanupError(RuntimeError):
    """A validator host RPC could not prove required socket cleanup."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        resource: Any,
        response: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__("validator enclave RPC transport cleanup failed")
        self.primary_error = primary_error
        self._resource = resource
        self._response = response


def _shutdown_and_close_socket(candidate: Any) -> bool:
    if candidate is None:
        return True
    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        return candidate.close() is not False
    except Exception:
        return False


_RETIRED_CLEANUP_LOCK = threading.RLock()
_RETIRED_CLEANUP_RECOVERY_LOCK = threading.Lock()
_RETIRED_CLEANUP = {}


def _encode_rpc_payload(
    payload: bytes,
    *,
    logical_limit: int,
    frame_limit: int,
) -> bytes:
    if len(payload) < 2 or len(payload) > logical_limit:
        raise RuntimeError("Enclave message size is outside the allowed range")
    if len(payload) <= frame_limit:
        return payload
    compressed = zlib.compress(payload, level=1)
    framed = (
        _COMPRESSED_FRAME_MAGIC
        + len(payload).to_bytes(4, byteorder="big")
        + compressed
    )
    if len(framed) > frame_limit:
        raise RuntimeError("Enclave compressed frame exceeds the allowed range")
    return framed


def _decode_rpc_payload(payload: bytes, *, logical_limit: int) -> bytes:
    if not payload.startswith(_COMPRESSED_FRAME_MAGIC):
        if len(payload) < 2 or len(payload) > logical_limit:
            raise RuntimeError("Enclave message size is outside the allowed range")
        return payload
    if len(payload) <= _COMPRESSED_FRAME_HEADER_BYTES:
        raise RuntimeError("Enclave compressed frame is incomplete")
    expected_length = int.from_bytes(payload[4:8], byteorder="big")
    if expected_length < 2 or expected_length > logical_limit:
        raise RuntimeError("Enclave decoded message size is outside the allowed range")
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(
            payload[_COMPRESSED_FRAME_HEADER_BYTES:],
            expected_length + 1,
        )
    except zlib.error as exc:
        raise RuntimeError("Enclave compressed frame is invalid") from exc
    if (
        len(decoded) != expected_length
        or len(decoded) > logical_limit
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise RuntimeError("Enclave compressed frame failed validation")
    return decoded


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def get_enclave_cid() -> Optional[int]:
    """
    Get the CID of the running validator enclave.
    
    Priority:
    1. ENCLAVE_CID environment variable (for Docker containers)
    2. nitro-cli describe-enclaves (for host)
    
    Returns:
        Enclave CID or None if not running
    """
    # Check environment variable first (for Docker containers)
    env_cid = os.environ.get("ENCLAVE_CID")
    if env_cid:
        try:
            cid = int(env_cid)
            print(
                f"[vsock] Using ENCLAVE_CID from environment: {cid}",
                file=sys.stderr,
            )
            return cid
        except ValueError:
            print(f"[vsock] Invalid ENCLAVE_CID: {env_cid}", file=sys.stderr)
    
    # Fall back to nitro-cli (for host). Resolve an absolute path first:
    # coordinator/worker processes can run with a PATH that lacks /usr/bin.
    try:
        import shutil as _shutil

        nitro_cli = _shutil.which("nitro-cli") or "/usr/bin/nitro-cli"
        result = subprocess.run(
            [nitro_cli, "describe-enclaves"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None
        
        import json as json_mod
        enclaves = json_mod.loads(result.stdout)
        
        for enclave in enclaves:
            # Look for validator enclave (by name or just return first running one)
            if enclave.get("State") == "RUNNING":
                return enclave.get("EnclaveCID")
        
        return None
        
    except Exception as e:
        print(f"[vsock] Error getting enclave CID: {e}", file=sys.stderr)
        return None


class ValidatorEnclaveClient:
    """
    Client for communicating with the validator TEE enclave.
    """
    
    def __init__(self, enclave_cid: Optional[int] = None):
        """
        Initialize the enclave client.
        
        Args:
            enclave_cid: Enclave CID (auto-detected if not provided)
        """
        self.enclave_cid = enclave_cid
        self._cached_pubkey: Optional[str] = None
        self._cached_code_hash: Optional[str] = None

    def _retain_cleanup_failure(
        self,
        resource: Any,
        *,
        primary_error: BaseException,
        response: Optional[Dict[str, Any]],
    ) -> None:
        with _RETIRED_CLEANUP_LOCK:
            _RETIRED_CLEANUP[id(resource)] = (
                resource,
                primary_error,
                response,
            )

    def _require_retired_cleanup(self) -> None:
        with _RETIRED_CLEANUP_RECOVERY_LOCK:
            with _RETIRED_CLEANUP_LOCK:
                snapshot = tuple(_RETIRED_CLEANUP.items())
            resolved_entries = []
            for resource_id, entry in snapshot:
                resource, _primary_error, _response = entry
                if _shutdown_and_close_socket(resource):
                    resolved_entries.append((resource_id, entry))
            with _RETIRED_CLEANUP_LOCK:
                for resource_id, entry in resolved_entries:
                    if _RETIRED_CLEANUP.get(resource_id) is entry:
                        _RETIRED_CLEANUP.pop(resource_id, None)
                pending = tuple(_RETIRED_CLEANUP.values())
        if not pending:
            return
        resource, primary_error, response = pending[0]
        raise ValidatorEnclaveTransportCleanupError(
            primary_error=primary_error,
            resource=resource,
            response=response,
        ) from primary_error
    
    def _get_cid(self) -> int:
        """Get enclave CID, auto-detecting if needed."""
        if self.enclave_cid is not None:
            return self.enclave_cid
        
        cid = get_enclave_cid()
        if cid is None:
            raise RuntimeError("No running validator enclave found")
        
        self.enclave_cid = cid
        return cid
    
    def _send_request(
        self,
        request: Dict[str, Any],
        *,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Send request to enclave via vsock.
        
        Args:
            request: Request dict with 'command' and parameters
            
        Returns:
            Response dict from enclave
        """
        self._require_retired_cleanup()
        cid = self._get_cid()

        # Create vsock socket
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        started_at = time.monotonic()
        command = str(request.get("command") or "unknown")[:100]
        request_data = b""
        response_data = b""
        response = None
        primary_error = None
        try:
            sock.settimeout(timeout_seconds)

            # Connect to enclave
            sock.connect((cid, RPC_PORT))
            
            # Send request
            request_data = json.dumps(request).encode()
            request_frame = _encode_rpc_payload(
                request_data,
                logical_limit=MAX_RPC_REQUEST_BYTES,
                frame_limit=MAX_RPC_REQUEST_FRAME_BYTES,
            )
            sock.sendall(
                len(request_frame).to_bytes(4, byteorder="big") + request_frame
            )
            
            # Receive response
            prefix = _recv_exact(sock, 4)
            if len(prefix) != 4:
                raise RuntimeError("Failed to read enclave response length")
            response_length = int.from_bytes(prefix, byteorder="big")
            if (
                response_length < 2
                or response_length > MAX_RPC_RESPONSE_FRAME_BYTES
            ):
                raise RuntimeError("Enclave response size is outside the allowed range")
            response_frame = _recv_exact(sock, response_length)
            if len(response_frame) != response_length:
                raise RuntimeError("Enclave response body is incomplete")
            response_data = _decode_rpc_payload(
                response_frame,
                logical_limit=MAX_RPC_RESPONSE_BYTES,
            )
            
            response = json.loads(response_data.decode())
            
            if response.get("status") == "error":
                raise RuntimeError(f"Enclave error: {response.get('error')}")
            
        except Exception as exc:
            primary_error = exc
            code = failure_code_for_exception(
                exc,
                default="runtime.enclave_relay_unavailable",
            )
            deterministic = code in {
                "weight.ancestry_bounds_exceeded",
                "weight.authoritative_result_invalid",
                "weight.bundle_divergence",
            }
            if deterministic:
                capture_failure(
                    code,
                    component="validator",
                    stage="enclave_rpc",
                    exception=exc,
                    terminal=True,
                    retryable=False,
                    fail_closed=True,
                    operation=command,
                    frame_limit_bytes=MAX_RPC_RESPONSE_FRAME_BYTES,
                    runtime_sha=(
                        os.environ.get("GITHUB_SHA")
                        or os.environ.get("GIT_COMMIT")
                        or ""
                    ),
                )
            else:
                record_retry(
                    code,
                    component="validator",
                    stage="enclave_rpc",
                    attempt=1,
                    attempts=1,
                    operation=command,
                    exception_class=type(exc).__name__,
                    runtime_sha=(
                        os.environ.get("GITHUB_SHA")
                        or os.environ.get("GIT_COMMIT")
                        or ""
                    ),
                )
            raise

        finally:
            if not _shutdown_and_close_socket(sock):
                cleanup_primary = primary_error or RuntimeError(
                    "validator enclave RPC socket cleanup failed"
                )
                self._retain_cleanup_failure(
                    sock,
                    primary_error=cleanup_primary,
                    response=response,
                )
                raise ValidatorEnclaveTransportCleanupError(
                    primary_error=cleanup_primary,
                    resource=sock,
                    response=response,
                ) from cleanup_primary
        if not isinstance(response, dict):
            raise RuntimeError("Enclave response is invalid")
        record_stage(
            component="validator",
            stage="enclave_rpc",
            status="passed",
            duration_seconds=time.monotonic() - started_at,
            operation=command,
            logical_bytes=len(request_data) + len(response_data),
            frame_limit_bytes=MAX_RPC_RESPONSE_FRAME_BYTES,
            runtime_sha=(
                os.environ.get("GITHUB_SHA")
                or os.environ.get("GIT_COMMIT")
                or ""
            ),
        )
        return response
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check enclave health.
        
        Returns:
            Health status dict
        """
        return self._send_request({"command": "health"})

    def get_arena_hotkey_state_v1(self) -> Dict[str, Any]:
        response = self._send_request({"command": "get_arena_hotkey_state_v1"})
        return dict(response["arena_hotkey_state"])

    def get_arena_hotkey_recipient_v1(self) -> Dict[str, Any]:
        response = self._send_request({"command": "get_arena_hotkey_recipient_v1"})
        return dict(response["recipient_request"])

    def provision_arena_hotkey_v1(self, ciphertext_for_recipient_b64: str) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "provision_arena_hotkey_v1",
                "ciphertext_for_recipient_b64": str(ciphertext_for_recipient_b64),
            },
            timeout_seconds=120,
        )
        return dict(response["arena_hotkey_state"])

    def sign_arena_application_v1(self, message: bytes) -> Dict[str, Any]:
        if not isinstance(message, bytes):
            raise TypeError("Arena application message must be bytes")
        response = self._send_request(
            {"command": "sign_arena_application_v1", "message_hex": message.hex()},
            timeout_seconds=120,
        )
        return dict(response["signature_result"])

    def prepare_arena_weight_extrinsic_v1(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask the protected signer to verify and sign one Arena weight call."""

        response = self._send_request(
            {
                "command": "prepare_arena_weight_extrinsic_v1",
                "signature_request": dict(request),
            },
            timeout_seconds=600,
        )
        return dict(response["signature_result"])

    def confirm_arena_weight_extrinsic_v1(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return protected finalized-chain evidence for one signed call."""

        response = self._send_request(
            {
                "command": "confirm_arena_weight_extrinsic_v1",
                "confirmation_request": dict(request),
            },
            timeout_seconds=600,
        )
        return dict(response["confirmation_result"])

    def recover_arena_weight_extrinsic_v1(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "recover_arena_weight_extrinsic_v1",
                "recovery_request": dict(request),
            },
            timeout_seconds=600,
        )
        return dict(response["recovery_result"])

    def sign_arena_chain_outcome_v1(
        self, outcome_document: Dict[str, Any]
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "sign_arena_chain_outcome_v1",
                "outcome_document": dict(outcome_document),
            },
            timeout_seconds=120,
        )
        return dict(response["signature_result"])



def is_enclave_available() -> bool:
    """Check whether a validator enclave is running."""
    return get_enclave_cid() is not None
