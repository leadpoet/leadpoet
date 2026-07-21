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
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Lock

print("🐛 DEBUG: Standard library imports OK", flush=True)

# Cryptography for Ed25519
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

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
MAX_RPC_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 16 * 1024 * 1024


# ============================================================================
# GLOBAL STATE (In-Memory, Hardware-Protected)
# ============================================================================

private_key: Optional[ed25519.Ed25519PrivateKey] = None
public_key: Optional[ed25519.Ed25519PublicKey] = None
public_key_hex: Optional[str] = None
keypair_lock = Lock()

# Code hash (computed at startup)
code_hash: Optional[str] = None

# Boot ID for this enclave session
boot_id: Optional[str] = None

# Authoritative V2 runtime state. Configuration is immutable for one enclave
# boot; once enabled, legacy blind-signing RPCs stay disabled until restart.
validator_runtime_v2: Optional[Any] = None
validator_weight_authority_v2: Optional[Any] = None
validator_hotkey_authority_v2: Optional[Any] = None
validator_chain_source_v2: Optional[Any] = None


# ============================================================================
# KEYPAIR GENERATION
# ============================================================================

def generate_keypair() -> str:
    """
    Generate Ed25519 keypair inside the enclave.
    
    SECURITY: Private key NEVER leaves enclave memory.
    
    Returns:
        Public key as hex string
    """
    global private_key, public_key, public_key_hex, boot_id
    
    with keypair_lock:
        if private_key is not None:
            print("[TEE] Keypair already generated", flush=True)
            return public_key_hex
        
        print("[TEE] Generating Ed25519 keypair...", flush=True)
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Export public key as hex
        pubkey_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        public_key_hex = pubkey_bytes.hex()
        
        # Generate boot_id for this session
        boot_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{public_key_hex}".encode()
        ).hexdigest()[:16]
        
        print(f"[TEE] ✅ Keypair generated: {public_key_hex[:16]}...", flush=True)
        print(f"[TEE] ✅ Boot ID: {boot_id}", flush=True)
        
        return public_key_hex


def get_public_key() -> str:
    """Get the enclave's public key as hex."""
    if public_key_hex is None:
        generate_keypair()
    return public_key_hex


# ============================================================================
# CODE HASH
# ============================================================================

def compute_code_hash() -> str:
    """
    Compute SHA256 hash of validator code inside the enclave.
    
    This hashes the actual code files in the enclave image.
    """
    global code_hash
    
    if code_hash is not None:
        return code_hash
    
    hasher = hashlib.sha256()
    
    # Hash critical files in the enclave
    critical_files = [
        "/app/validator_tee/enclave/tee_service.py",
        "/app/leadpoet_canonical/weights.py",
        "/app/leadpoet_canonical/weight_computation.py",
        "/app/leadpoet_canonical/attested_receipts.py",
    ]
    
    for filepath in sorted(critical_files):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
            hasher.update(filepath.encode())
            print(f"[TEE] Hashed: {filepath}", flush=True)
    
    code_hash = hasher.hexdigest()
    print(f"[TEE] ✅ Code hash: {code_hash[:16]}...", flush=True)
    
    return code_hash


def configure_authoritative_v2(
    configuration: Dict[str, Any],
    expected_config_hash: str,
) -> Dict[str, Any]:
    """Configure the hardware-only V2 release and weight authority once."""

    global validator_runtime_v2, validator_weight_authority_v2, validator_chain_source_v2
    from validator_tee.enclave.runtime_v2 import ValidatorRuntimeIdentityV2
    from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2
    from validator_tee.enclave.weight_authority_v2 import ValidatorWeightAuthorityV2

    if validator_runtime_v2 is None:
        validator_runtime_v2 = ValidatorRuntimeIdentityV2(
            signing_pubkey_supplier=get_public_key,
        )
    boot_identity = validator_runtime_v2.configure(
        configuration,
        expected_config_hash=expected_config_hash,
    )
    if validator_chain_source_v2 is None:
        epoch_authority_supplier = getattr(
            validator_runtime_v2,
            "epoch_authority",
            lambda: None,
        )
        validator_chain_source_v2 = ValidatorChainSourceV2(
            epoch_authority_supplier=epoch_authority_supplier,
        )
    if validator_weight_authority_v2 is None:
        validator_weight_authority_v2 = ValidatorWeightAuthorityV2(
            boot_identity_supplier=validator_runtime_v2.boot_identity,
            gateway_release_lineage_supplier=(
                validator_runtime_v2.gateway_release_lineage
            ),
            sign_digest=lambda digest: private_key.sign(digest),
            chain_source=validator_chain_source_v2,
        )
    return boot_identity


def get_authoritative_v2_boot_identity() -> Dict[str, Any]:
    if validator_runtime_v2 is None:
        raise RuntimeError("validator authoritative V2 runtime is not configured")
    return validator_runtime_v2.boot_identity()


def configure_hotkey_authority_v2(
    configuration: Dict[str, Any],
    expected_config_hash: str,
) -> Dict[str, Any]:
    """Configure the measured drand backend and KMS-sealed sr25519 authority."""

    global validator_hotkey_authority_v2, validator_chain_source_v2
    if validator_runtime_v2 is None:
        raise RuntimeError("validator authoritative V2 runtime is not configured")
    from leadpoet_canonical.attested_v2 import sha256_json as sha256_json_v2
    from validator_tee.enclave.drand_v2 import CtypesDrandCommitBackendV2
    from validator_tee.enclave.hotkey_authority_v2 import (
        ValidatorHotkeyAuthorityV2,
        load_chain_signing_profile,
        validate_hotkey_authority_configuration,
    )
    from validator_tee.enclave.runtime_v2 import _nsm_attest

    normalized = validate_hotkey_authority_configuration(configuration)
    observed_hash = sha256_json_v2(normalized)
    if observed_hash != str(expected_config_hash or "").lower():
        raise RuntimeError("validator hotkey configuration hash mismatch")
    if observed_hash != validator_runtime_v2.hotkey_authority_config_hash():
        raise RuntimeError("validator hotkey configuration differs from boot release")
    profile = load_chain_signing_profile()
    if sha256_json_v2(profile) != normalized["chain_signing_profile_hash"]:
        raise RuntimeError("measured chain signing profile hash mismatch")
    if validator_hotkey_authority_v2 is None:
        drand_backend = CtypesDrandCommitBackendV2(
            library_path=normalized["drand_library_path"],
            expected_sha256=normalized["drand_library_sha256"],
        )
        validator_hotkey_authority_v2 = ValidatorHotkeyAuthorityV2(
            boot_identity_supplier=validator_runtime_v2.boot_identity,
            validator_hotkey=normalized["validator_hotkey"],
            hotkey_public_key_hex=normalized["hotkey_public_key"],
            chain_profile=profile,
            sign_receipt_digest=lambda digest: private_key.sign(digest),
            attestation_supplier=_nsm_attest,
            drand_backend=drand_backend,
            chain_source=validator_chain_source_v2,
        )
    return validator_hotkey_authority_v2.public_state()


def compute_authoritative_weights_v2(request: Dict[str, Any]) -> Dict[str, Any]:
    if validator_weight_authority_v2 is None:
        raise RuntimeError("validator authoritative V2 runtime is not configured")
    if validator_hotkey_authority_v2 is None:
        raise RuntimeError("validator V2 hotkey authority is not configured")
    result = validator_weight_authority_v2.compute(request)
    authorization_input = {
        field: result[field]
        for field in (
            "weight_snapshot",
            "weight_result",
            "weights_signature",
            "receipt_graph",
            "boot_identity",
        )
    }
    result["weight_authorization_id"] = (
        validator_hotkey_authority_v2.register_weight_result(
            authorization_input
        )
    )
    return result


def capture_subnet_epoch_boundary_v2(request: Dict[str, Any]) -> Dict[str, Any]:
    if validator_weight_authority_v2 is None:
        raise RuntimeError("validator authoritative V2 runtime is not configured")
    return validator_weight_authority_v2.capture_epoch_boundary(request)


def _authoritative_v2_enabled() -> bool:
    return validator_runtime_v2 is not None


# ============================================================================
# RPC HANDLER
# ============================================================================

def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle RPC request from parent EC2.
    
    Supported commands:
    - configure_authoritative_v2: Lock the measured V2 release for this boot
    - get_authoritative_v2_boot_identity: Return the hardware V2 boot identity
    - configure_hotkey_authority_v2: Lock hotkey, chain, and drand identity
    - get_hotkey_recipient_v2: Return an attested KMS recipient request
    - provision_hotkey_v2: Unseal the validator seed directly inside Nitro
    - get_hotkey_state_v2: Return non-secret hotkey authority state
    - sign_application_message_v2: Sign one recognized application domain
    - prepare_weight_commit_v2: Generate the exact timelocked weight commitment
    - sign_weight_extrinsic_v2: Sign one enclave-authorized SCALE payload
    - recover_weight_publication_v2: Validate and restore public signed state
    - sign_serve_axon_extrinsic_v2: Sign the exact measured serve_axon payload
    - compute_authoritative_weights_v2: Verify ancestry, compute, and sign weights
    - capture_subnet_epoch_boundary_v2: Prove a proposed stateful cutover boundary
    - health: Health check
    """
    command = request.get("command")

    if command in {
        "get_public_key",
        "sign_weights",
        "get_attestation",
        "compute_weights_v2",
    }:
        return {
            "status": "error",
            "error": "Legacy validator V1 RPC is permanently removed",
        }

    try:
        if command == "configure_authoritative_v2":
            configuration = request.get("configuration")
            expected_config_hash = request.get("expected_config_hash")
            if not isinstance(configuration, dict) or not isinstance(
                expected_config_hash, str
            ):
                return {
                    "status": "error",
                    "error": "Missing authoritative V2 configuration",
                }
            return {
                "status": "ok",
                "boot_identity": configure_authoritative_v2(
                    configuration,
                    expected_config_hash,
                ),
            }

        elif command == "get_authoritative_v2_boot_identity":
            return {
                "status": "ok",
                "boot_identity": get_authoritative_v2_boot_identity(),
            }

        elif command == "configure_hotkey_authority_v2":
            configuration = request.get("configuration")
            expected_config_hash = request.get("expected_config_hash")
            if not isinstance(configuration, dict) or not isinstance(
                expected_config_hash, str
            ):
                return {
                    "status": "error",
                    "error": "Missing validator hotkey V2 configuration",
                }
            return {
                "status": "ok",
                "hotkey_state": configure_hotkey_authority_v2(
                    configuration,
                    expected_config_hash,
                ),
            }

        elif command == "get_hotkey_recipient_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            return {
                "status": "ok",
                "recipient_request": (
                    validator_hotkey_authority_v2.recipient_request()
                ),
            }

        elif command == "provision_hotkey_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            ciphertext = request.get("ciphertext_for_recipient_b64")
            if not isinstance(ciphertext, str):
                return {
                    "status": "error",
                    "error": "Missing KMS CiphertextForRecipient",
                }
            return {
                "status": "ok",
                "hotkey_state": validator_hotkey_authority_v2.provision_seed(
                    ciphertext_for_recipient_b64=ciphertext,
                ),
            }

        elif command == "get_hotkey_state_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            return {
                "status": "ok",
                "hotkey_state": validator_hotkey_authority_v2.public_state(),
            }

        elif command == "sign_application_message_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            message_hex = request.get("message_hex")
            parent_receipt_hash = request.get("parent_receipt_hash")
            if not isinstance(message_hex, str) or (
                parent_receipt_hash is not None
                and not isinstance(parent_receipt_hash, str)
            ):
                return {
                    "status": "error",
                    "error": "Invalid application signature request",
                }
            return {
                "status": "ok",
                "signature_result": (
                    validator_hotkey_authority_v2.sign_application_message(
                        message_hex=message_hex,
                        parent_receipt_hash=parent_receipt_hash,
                    )
                ),
            }

        elif command == "prepare_weight_commit_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            commit_request = request.get("commit_request")
            if not isinstance(commit_request, dict):
                return {
                    "status": "error",
                    "error": "Missing weight commitment request",
                }
            return {
                "status": "ok",
                "commit_result": (
                    validator_hotkey_authority_v2.prepare_weight_commit(
                        **commit_request
                    )
                ),
            }

        elif command == "sign_weight_extrinsic_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            signature_request = request.get("signature_request")
            if not isinstance(signature_request, dict):
                return {
                    "status": "error",
                    "error": "Missing weight extrinsic signature request",
                }
            return {
                "status": "ok",
                "signature_result": (
                    validator_hotkey_authority_v2.sign_weight_extrinsic(
                        **signature_request
                    )
                ),
            }

        elif command == "confirm_weight_publication_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            weight_authorization_id = request.get("weight_authorization_id")
            if not isinstance(weight_authorization_id, str):
                return {
                    "status": "error",
                    "error": "Missing weight publication authorization",
                }
            return {
                "status": "ok",
                "finalization_result": (
                    validator_hotkey_authority_v2.confirm_weight_publication(
                        weight_authorization_id=weight_authorization_id
                    )
                ),
            }

        elif command == "recover_weight_publication_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            published_bundle = request.get("published_bundle")
            event_hash = request.get("weight_submission_event_hash")
            signed_results = request.get("extrinsic_signature_results")
            if (
                not isinstance(published_bundle, dict)
                or not isinstance(event_hash, str)
                or not isinstance(signed_results, list)
            ):
                return {
                    "status": "error",
                    "error": "Invalid weight publication recovery request",
                }
            return {
                "status": "ok",
                "recovery_result": (
                    validator_hotkey_authority_v2.recover_weight_publication(
                        published_bundle=published_bundle,
                        weight_submission_event_hash=event_hash,
                        extrinsic_signature_results=signed_results,
                    )
                ),
            }

        elif command == "sign_serve_axon_extrinsic_v2":
            if validator_hotkey_authority_v2 is None:
                raise RuntimeError("validator V2 hotkey authority is not configured")
            signature_request = request.get("signature_request")
            if not isinstance(signature_request, dict):
                return {
                    "status": "error",
                    "error": "Missing serve axon signature request",
                }
            return {
                "status": "ok",
                "signature_result": (
                    validator_hotkey_authority_v2.sign_serve_axon_extrinsic(
                        **signature_request
                    )
                ),
            }

        elif command == "compute_authoritative_weights_v2":
            weight_request = request.get("weight_request")
            if not isinstance(weight_request, dict):
                return {
                    "status": "error",
                    "error": "Missing authoritative V2 weight request",
                }
            return {
                "status": "ok",
                **compute_authoritative_weights_v2(weight_request),
            }

        elif command == "capture_subnet_epoch_boundary_v2":
            capture_request = request.get("capture_request")
            if not isinstance(capture_request, dict):
                return {
                    "status": "error",
                    "error": "Missing subnet epoch boundary capture request",
                }
            return {
                "status": "ok",
                "capture_result": capture_subnet_epoch_boundary_v2(
                    capture_request
                ),
            }

        elif command == "health":
            hotkey_state = None
            if validator_hotkey_authority_v2 is not None:
                hotkey_state = validator_hotkey_authority_v2.public_state()
            return {
                "status": "ok",
                "service": "validator_tee",
                "keypair_initialized": private_key is not None,
                "boot_id": boot_id,
                "authoritative_v2_configured": _authoritative_v2_enabled(),
                "hotkey_authority_v2_configured": (
                    validator_hotkey_authority_v2 is not None
                ),
                "hotkey_state_v2": hotkey_state,
            }
        
        else:
            return {"status": "error", "error": f"Unknown command: {command}"}
            
    except Exception as e:
        print(f"[TEE] ❌ Error handling {command}: {e}", flush=True)
        return {"status": "error", "error": str(e)}


# ============================================================================
# VSOCK SERVER
# ============================================================================

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
            if len(data) > MAX_RPC_REQUEST_BYTES:
                raise ValueError("legacy request exceeds maximum size")
        request_data = bytes(data)
        length_prefixed = False
    else:
        request_length = int.from_bytes(prefix, byteorder="big")
        if request_length < 2 or request_length > MAX_RPC_REQUEST_BYTES:
            raise ValueError("request length is outside the allowed range")
        request_data = _recv_exact(client, request_length)
        if len(request_data) != request_length:
            raise ValueError("request body is incomplete")
        length_prefixed = True
    request = json.loads(request_data.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request, length_prefixed

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
    server.listen(5)
    
    print(f"[TEE] ✅ Listening on vsock port {RPC_PORT}", flush=True)
    
    while True:
        try:
            client, addr = server.accept()
            print(f"[TEE] Connection from CID {addr[0]}", flush=True)
            
            request, length_prefixed = _receive_request(client)
            print(f"[TEE] Request: {request.get('command')}", flush=True)

            response = handle_request(request)
            response_data = json.dumps(response).encode()
            if len(response_data) > MAX_RPC_RESPONSE_BYTES:
                raise ValueError("response exceeds maximum size")
            if length_prefixed:
                client.sendall(len(response_data).to_bytes(4, byteorder="big") + response_data)
            else:
                client.sendall(response_data)
            
            client.close()
            
        except Exception as e:
            print(f"[TEE] ❌ Server error: {e}", flush=True)
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("🔐 VALIDATOR TEE SERVICE", flush=True)
    print("=" * 80, flush=True)
    
    # Initialize keypair on startup
    generate_keypair()
    
    # Compute code hash
    compute_code_hash()
    
    print("", flush=True)
    print("📊 Enclave State:", flush=True)
    print(f"   Public Key: {public_key_hex[:32]}...", flush=True)
    print(f"   Code Hash: {code_hash[:32]}...", flush=True)
    print(f"   Boot ID: {boot_id}", flush=True)
    print("", flush=True)
    
    # Start vsock server
    run_vsock_server()
