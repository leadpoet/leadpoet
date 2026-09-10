#!/usr/bin/env python3
"""
Validator TEE Service (Runs Inside Nitro Enclave)
=================================================

This service runs inside an AWS Nitro Enclave and handles:
- Ed25519 keypair generation (ephemeral per boot)
- Weight hash signing
- Attestation document generation with epoch binding

SECURITY MODEL:
- Private key generated inside enclave, NEVER leaves
- Attestation document binds public key to enclave code (PCR0)
- epoch_id in attestation prevents replay attacks
- Signs only canonical weight hashes (not arbitrary data)

COMMUNICATION:
- Uses vsock (virtual socket) for parent <-> enclave communication
- No network access from inside the enclave
"""

# The enclave runs Python 3.7: annotations must stay lazy or modern builtin
# generics (tuple[...], dict[...]) raise TypeError at def time.
from __future__ import annotations

print("=" * 80, flush=True)
print("🔐 VALIDATOR TEE SERVICE STARTING", flush=True)
print("=" * 80, flush=True)

import socket
import json
import sys
import os
import errno
import zlib
from typing import Dict, Any, Optional

print("🐛 DEBUG: Standard library imports OK", flush=True)

print("🐛 DEBUG: Cryptography imports OK", flush=True)

# The enclave interpreter only has this script's directory on sys.path; the
# repo-level packages copied to /app (leadpoet_canonical, research_lab, ...)
# must be importable before the imports below or the enclave dies at boot.
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

print("🐛 DEBUG: Canonical runtime path configured", flush=True)

# ============================================================================
# VSOCK CONFIGURATION
# ============================================================================

AF_VSOCK = 40  # Address family for vsock
VMADDR_CID_ANY = 0xFFFFFFFF  # Bind to any CID (inside enclave)
PARENT_CID = 3  # Parent EC2's CID
RPC_PORT = 5001  # Use different port from gateway (5001 vs 5000)
VSOCK_RPC_LISTEN_BACKLOG = 32
VSOCK_RPC_RECEIVE_TIMEOUT_SECONDS = 30.0
VSOCK_RPC_RESPONSE_TIMEOUT_SECONDS = 600.0
# Keep the wire envelope bounded inside the memory-constrained enclave.
MAX_RPC_REQUEST_FRAME_BYTES = 16 * 1024 * 1024
MAX_RPC_RESPONSE_FRAME_BYTES = 16 * 1024 * 1024
MAX_RPC_REQUEST_BYTES = 128 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 128 * 1024 * 1024
_COMPRESSED_FRAME_MAGIC = b"LPZ2"
_COMPRESSED_FRAME_HEADER_BYTES = 8
MAX_VSOCK_RPC_CLEANUP_ATTEMPTS = 4
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)


def _encode_rpc_payload(
    payload: bytes,
    *,
    logical_limit: int,
    frame_limit: int,
) -> bytes:
    if len(payload) < 2 or len(payload) > logical_limit:
        raise ValueError("message size is outside the allowed range")
    if len(payload) <= frame_limit:
        return payload
    compressed = zlib.compress(payload, level=1)
    framed = (
        _COMPRESSED_FRAME_MAGIC
        + len(payload).to_bytes(4, byteorder="big")
        + compressed
    )
    if len(framed) > frame_limit:
        raise ValueError("compressed frame exceeds the allowed range")
    return framed


def _decode_rpc_payload(payload: bytes, *, logical_limit: int) -> bytes:
    if not payload.startswith(_COMPRESSED_FRAME_MAGIC):
        if len(payload) < 2 or len(payload) > logical_limit:
            raise ValueError("message size is outside the allowed range")
        return payload
    if len(payload) <= _COMPRESSED_FRAME_HEADER_BYTES:
        raise ValueError("compressed frame is incomplete")
    expected_length = int.from_bytes(payload[4:8], byteorder="big")
    if expected_length < 2 or expected_length > logical_limit:
        raise ValueError("decoded message size is outside the allowed range")
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(
            payload[_COMPRESSED_FRAME_HEADER_BYTES:],
            expected_length + 1,
        )
    except zlib.error as exc:
        raise ValueError("compressed frame is invalid") from exc
    if (
        len(decoded) != expected_length
        or len(decoded) > logical_limit
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("compressed frame failed validation")
    return decoded


# ============================================================================
# ARENA PROTECTED STATE
# ============================================================================

validator_chain_source_v2: Optional[Any] = None
validator_arena_weight_signer_v1: Optional[Any] = None
validator_arena_hotkey_authority_v1: Optional[Any] = None


def get_arena_hotkey_recipient_v1() -> Dict[str, Any]:
    global validator_arena_hotkey_authority_v1
    if validator_arena_hotkey_authority_v1 is None:
        from validator_tee.enclave.arena_hotkey import ArenaHotkeyAuthority
        validator_arena_hotkey_authority_v1 = ArenaHotkeyAuthority()
    return validator_arena_hotkey_authority_v1.recipient_request()


def provision_arena_hotkey_v1(ciphertext: str) -> Dict[str, Any]:
    global validator_arena_weight_signer_v1, validator_chain_source_v2
    if validator_arena_hotkey_authority_v1 is None:
        raise RuntimeError("Arena hotkey recipient was not configured")
    existing = validator_arena_hotkey_authority_v1.public_state()
    if existing.get("provisioned") is True and validator_arena_weight_signer_v1 is not None:
        return existing
    state = (
        existing
        if existing.get("provisioned") is True
        else validator_arena_hotkey_authority_v1.provision(ciphertext)
    )
    policy = validator_arena_hotkey_authority_v1.policy
    from urllib.parse import urlsplit
    from leadpoet_canonical.chain_source_v2 import configure_chain_source_boundary_v2
    configure_chain_source_boundary_v2(
        chain_host=str(urlsplit(policy["chain_profile"]["chain_endpoint"]).hostname),
        chain_archive_host=str(policy["chain_archive_host"]),
    )
    from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2
    validator_chain_source_v2 = ValidatorChainSourceV2(
        epoch_authority_supplier=lambda: policy["epoch_authority"]
    )
    from validator_tee.enclave.drand_v2 import CtypesDrandCommitBackendV2
    from validator_tee.enclave.arena_hotkey import MEASURED_DRAND_LIBRARY_PATH
    from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner
    from validator_tee.enclave.arena_state_source import ArenaStateSource
    drand = CtypesDrandCommitBackendV2(
        library_path=MEASURED_DRAND_LIBRARY_PATH,
        expected_sha256=policy["drand_library_sha256"],
    )
    import sr25519
    public_key = bytes.fromhex(policy["hotkey_public_key"])
    validator_arena_weight_signer_v1 = ArenaWeightSigner(
        validator_hotkey=policy["validator_hotkey"],
        hotkey_public_key_hex=policy["hotkey_public_key"],
        chain_profile=policy["chain_profile"], chain_source=validator_chain_source_v2,
        drand_backend=drand,
        sign_sr25519=validator_arena_hotkey_authority_v1.sign_weight_payload,
        verify_sr25519=lambda signature, payload: bool(sr25519.verify(signature, payload, public_key)),
        arena_public_key_der=__import__(
            "leadpoet_canonical.lab_arena_rewards", fromlist=["signing_key_from_document"]
        ).signing_key_from_document(policy["arena_signing_key"], policy["arena_signing_key_hash"]),
        arena_public_key_hash=policy["arena_signing_key_hash"],
        network=policy["network"], netuid=policy["netuid"],
        burn_hotkey=policy["burn_hotkey"],
        state_source=ArenaStateSource(policy),
    )
    return state


def provision_arena_legacy_hotkey_v1(ciphertext: str) -> Dict[str, Any]:
    """Install an existing KMS raw-seed envelope under measured Arena policy."""

    global validator_arena_weight_signer_v1, validator_chain_source_v2
    if validator_arena_hotkey_authority_v1 is None:
        raise RuntimeError("Arena hotkey recipient was not configured")
    existing = validator_arena_hotkey_authority_v1.public_state()
    if existing.get("provisioned") is True:
        if validator_arena_weight_signer_v1 is None:
            raise RuntimeError("Arena signer initialization is incomplete")
        return existing
    validator_arena_hotkey_authority_v1.provision_legacy_seed(ciphertext)
    # Reuse the single signer initialization path without decrypting twice.
    return provision_arena_hotkey_v1("")


def configure_arena_weight_signer_v1(configuration: Dict[str, Any]) -> Dict[str, Any]:
    """Confirm the signer configuration against its sealed policy."""

    if validator_arena_hotkey_authority_v1 is None:
        raise RuntimeError("sealed Arena hotkey policy is not provisioned")
    policy = validator_arena_hotkey_authority_v1.policy
    expected = {
        "network": policy["network"], "netuid": policy["netuid"],
        "signing_key": policy["arena_signing_key"],
        "expected_public_key_hash": policy["arena_signing_key_hash"],
    }
    if configuration != expected or validator_arena_weight_signer_v1 is None:
        raise RuntimeError("Arena signer configuration differs from sealed policy")
    return {
        "configured": True, "network": policy["network"],
        "netuid": policy["netuid"],
        "arena_public_key_hash": policy["arena_signing_key_hash"],
    }


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch only the measured Arena signer RPC surface."""
    command = request.get("command")
    allowed = {
        "health", "get_arena_hotkey_recipient_v1", "provision_arena_hotkey_v1",
        "provision_arena_legacy_hotkey_v1",
        "get_arena_hotkey_state_v1", "sign_arena_application_v1",
        "configure_arena_weight_signer_v1", "prepare_arena_weight_extrinsic_v1",
        "recover_arena_weight_extrinsic_v1", "confirm_arena_weight_extrinsic_v1",
        "sign_arena_chain_outcome_v1",
    }
    if command not in allowed:
        return {"status": "error", "error": "RPC is outside Arena enclave mode"}
    try:
        if command == "get_arena_hotkey_recipient_v1":
            return {"status": "ok", "recipient_request": get_arena_hotkey_recipient_v1()}
        if command == "provision_arena_hotkey_v1":
            ciphertext = request.get("ciphertext_for_recipient_b64")
            if not isinstance(ciphertext, str):
                return {"status": "error", "error": "Missing Arena recipient ciphertext"}
            return {"status": "ok", "arena_hotkey_state": provision_arena_hotkey_v1(ciphertext)}
        if command == "provision_arena_legacy_hotkey_v1":
            ciphertext = request.get("ciphertext_for_recipient_b64")
            if not isinstance(ciphertext, str):
                return {"status": "error", "error": "Missing Arena recipient ciphertext"}
            return {"status": "ok", "arena_hotkey_state": provision_arena_legacy_hotkey_v1(ciphertext)}
        if command == "get_arena_hotkey_state_v1":
            if validator_arena_hotkey_authority_v1 is None:
                state = {"provisioned": False}
            else:
                state = validator_arena_hotkey_authority_v1.public_state()
                if state.get("provisioned") is True and validator_arena_weight_signer_v1 is None:
                    state = {**state, "provisioned": False, "key_unsealed": True}
            return {"status": "ok", "arena_hotkey_state": state}
        if command == "sign_arena_application_v1":
            if validator_arena_hotkey_authority_v1 is None:
                raise RuntimeError("Arena hotkey authority is not configured")
            try:
                message = bytes.fromhex(str(request.get("message_hex") or ""))
            except ValueError:
                return {"status": "error", "error": "Arena application message is invalid hex"}
            return {"status": "ok", "signature_result": validator_arena_hotkey_authority_v1.sign_application(message)}
        if command == "configure_arena_weight_signer_v1":
            configuration = request.get("configuration")
            if not isinstance(configuration, dict):
                return {"status": "error", "error": "Missing Arena signer configuration"}
            return {"status": "ok", "arena_signer_state": configure_arena_weight_signer_v1(configuration)}
        if command == "health":
            return {
                "status": "ok", "service": "validator_arena_signer",
                "arena_weight_signer_v1_supported": True,
                "arena_weight_signer_v1_configured": validator_arena_weight_signer_v1 is not None,
                "arena_hotkey_v1_configured": (
                    validator_arena_hotkey_authority_v1 is not None
                    and validator_arena_hotkey_authority_v1.public_state().get("provisioned") is True
                ),
            }
        if validator_arena_weight_signer_v1 is None:
            raise RuntimeError("validator Arena weight signer is not configured")
        if command == "prepare_arena_weight_extrinsic_v1":
            value = request.get("signature_request")
            if not isinstance(value, dict):
                return {"status": "error", "error": "Missing Arena weight signature request"}
            return {"status": "ok", "signature_result": validator_arena_weight_signer_v1.prepare(value)}
        if command == "confirm_arena_weight_extrinsic_v1":
            value = request.get("confirmation_request")
            if not isinstance(value, dict):
                return {"status": "error", "error": "Missing Arena weight confirmation request"}
            return {"status": "ok", "confirmation_result": validator_arena_weight_signer_v1.confirm(value)}
        if command == "recover_arena_weight_extrinsic_v1":
            value = request.get("recovery_request")
            if not isinstance(value, dict):
                return {"status": "error", "error": "Missing Arena weight recovery request"}
            return {"status": "ok", "recovery_result": validator_arena_weight_signer_v1.recover(value)}
        if command == "sign_arena_chain_outcome_v1":
            value = request.get("outcome_document")
            if not isinstance(value, dict):
                return {"status": "error", "error": "Missing Arena chain outcome document"}
            return {"status": "ok", "signature_result": validator_arena_weight_signer_v1.sign_chain_outcome(value)}
    except Exception as exc:
        print("[TEE] Error handling %s: %s" % (command, exc), flush=True)
        return {"status": "error", "error": str(exc)}


# ============================================================================
# VSOCK SERVER
# ============================================================================

class ValidatorVSOCKRPCCleanupError(RuntimeError):
    """An accepted validator RPC socket could not prove descriptor release."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
        resource: Any,
    ) -> None:
        super().__init__("validator vsock RPC transport cleanup failed")
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        # Never serialized. The synchronous listener supervisor observes this
        # terminal error before ownership can leave scope.
        self._resource = resource


class _ExplicitVSOCKCloseFailure(RuntimeError):
    """A socket adapter explicitly reported retained ownership."""


def _close_vsock_rpc_required(candidate: Any) -> Optional[BaseException]:
    """Attempt full-duplex shutdown and return any close-proof failure."""

    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # A peer may have closed immediately after reading the response.
        # close() remains the accepted descriptor's ownership boundary.
        pass
    last_error = None  # type: Optional[BaseException]
    for _attempt in range(MAX_VSOCK_RPC_CLEANUP_ATTEMPTS):
        try:
            if candidate.close() is False:
                last_error = _ExplicitVSOCKCloseFailure(
                    "validator vsock RPC close was not confirmed"
                )
                continue
        except BaseException as exc:
            last_error = exc
            continue
        return None
    return last_error


def _recv_exact(client: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = client.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _receive_request(client: Any) -> tuple[Dict[str, Any], bool]:
    """Receive bounded length-prefixed JSON, with old EOF framing support."""

    prefix = _recv_exact(client, 4)
    if len(prefix) != 4:
        raise ValueError("incomplete request prefix")
    if prefix.startswith(b"{"):
        data = bytearray(prefix)
        while True:
            chunk = client.recv(64 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_RPC_REQUEST_FRAME_BYTES:
                raise ValueError("uncompressed request exceeds maximum size")
        request_data = bytes(data)
        length_prefixed = False
    else:
        request_length = int.from_bytes(prefix, byteorder="big")
        if (
            request_length < 2
            or request_length > MAX_RPC_REQUEST_FRAME_BYTES
        ):
            raise ValueError("request length is outside the allowed range")
        request_frame = _recv_exact(client, request_length)
        if len(request_frame) != request_length:
            raise ValueError("request body is incomplete")
        request_data = _decode_rpc_payload(
            request_frame,
            logical_limit=MAX_RPC_REQUEST_BYTES,
        )
        length_prefixed = True
    request = json.loads(request_data.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request, length_prefixed


def _handle_vsock_client(client: Any, addr: Any) -> None:
    """Handle one validator RPC with a bounded abandoned-client lifetime."""

    primary_error = None  # type: Optional[BaseException]
    try:
        client.settimeout(VSOCK_RPC_RECEIVE_TIMEOUT_SECONDS)
        print(f"[TEE] Connection from CID {addr[0]}", flush=True)

        request, length_prefixed = _receive_request(client)
        client.settimeout(VSOCK_RPC_RESPONSE_TIMEOUT_SECONDS)
        print(f"[TEE] Request: {request.get('command')}", flush=True)

        response = handle_request(request)
        response_data = json.dumps(response).encode()
        if length_prefixed:
            response_frame = _encode_rpc_payload(
                response_data,
                logical_limit=MAX_RPC_RESPONSE_BYTES,
                frame_limit=MAX_RPC_RESPONSE_FRAME_BYTES,
            )
            client.sendall(
                len(response_frame).to_bytes(4, byteorder="big")
                + response_frame
            )
        else:
            if len(response_data) > MAX_RPC_RESPONSE_FRAME_BYTES:
                raise ValueError("uncompressed response exceeds maximum size")
            client.sendall(response_data)
    except Exception as exc:
        primary_error = exc
        print(
            "[TEE] ❌ Server error type=%s" % type(exc).__name__,
            flush=True,
        )
    finally:
        cleanup_error = _close_vsock_rpc_required(client)
        if cleanup_error is not None:
            cleanup_primary = primary_error or cleanup_error
            raise ValidatorVSOCKRPCCleanupError(
                primary_error=cleanup_primary,
                cleanup_error=cleanup_error,
                resource=client,
            ) from cleanup_primary

def run_vsock_server():
    """
    Run vsock server to handle requests from parent EC2.
    """
    print(f"[TEE] Starting vsock server on port {RPC_PORT}...", flush=True)
    
    # Create vsock socket
    server = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to any CID on our port
    server.bind((VMADDR_CID_ANY, RPC_PORT))
    server.listen(VSOCK_RPC_LISTEN_BACKLOG)
    
    print(f"[TEE] ✅ Listening on vsock port {RPC_PORT}", flush=True)
    
    while True:
        try:
            client, addr = server.accept()
        except OSError as exc:
            if exc.errno in _TRANSIENT_ACCEPT_ERRNOS:
                print(
                    "[TEE] transient accept error type=%s"
                    % type(exc).__name__,
                    flush=True,
                )
                continue
            raise
        _handle_vsock_client(client, addr)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_vsock_server()
