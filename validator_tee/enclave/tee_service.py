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

# CBOR for attestation documents
import cbor2

print("🐛 DEBUG: CBOR imports OK", flush=True)

# ============================================================================
# VSOCK CONFIGURATION
# ============================================================================

AF_VSOCK = 40  # Address family for vsock
VMADDR_CID_ANY = 0xFFFFFFFF  # Bind to any CID (inside enclave)
PARENT_CID = 3  # Parent EC2's CID
RPC_PORT = 5001  # Use different port from gateway (5001 vs 5000)


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
        "/app/validator_tee/tee_service.py",
        "/app/validator_tee/enclave_signer.py",
        "/app/leadpoet_canonical/weights.py",
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


# ============================================================================
# SIGNING (CONSTRAINED - Only Signs Weight Hashes)
# ============================================================================

def sign_weights_hash(weights_hash: str) -> str:
    """
    Sign a weights hash using the enclave's private key.
    
    SECURITY: This function ONLY signs weight hashes, not arbitrary data.
    The hash is verified to be a valid SHA256 hex string before signing.
    
    Args:
        weights_hash: SHA256 hex string (64 characters)
        
    Returns:
        Ed25519 signature as hex string
        
    Raises:
        ValueError: If hash format is invalid
    """
    if private_key is None:
        generate_keypair()
    
    # Validate hash format (must be 64 hex chars = 32 bytes)
    if not weights_hash or len(weights_hash) != 64:
        raise ValueError(f"Invalid weights_hash length: {len(weights_hash) if weights_hash else 0}, expected 64")
    
    try:
        hash_bytes = bytes.fromhex(weights_hash)
    except ValueError as e:
        raise ValueError(f"Invalid hex in weights_hash: {e}")
    
    if len(hash_bytes) != 32:
        raise ValueError(f"weights_hash must be 32 bytes, got {len(hash_bytes)}")
    
    # Sign the hash bytes directly (Ed25519 signs raw bytes)
    signature = private_key.sign(hash_bytes)
    signature_hex = signature.hex()
    
    print(f"[TEE] ✅ Signed weights hash: {weights_hash[:16]}...", flush=True)
    
    return signature_hex


# ============================================================================
# ATTESTATION DOCUMENT
# ============================================================================

def get_attestation_document(epoch_id: int) -> Dict[str, Any]:
    """
    Get attestation document with epoch_id binding.
    
    The attestation includes:
    - purpose: "validator_weights"
    - epoch_id: Bound to specific epoch (replay protection)
    - enclave_pubkey: The signing public key
    - code_hash: Hash of validator code
    
    Args:
        epoch_id: Epoch being attested
        
    Returns:
        Dict with attestation_b64 and user_data
    """
    if public_key_hex is None:
        generate_keypair()
    
    # Build user_data for attestation
    user_data = {
        "purpose": "validator_weights",
        "epoch_id": epoch_id,
        "enclave_pubkey": public_key_hex,
        "code_hash": compute_code_hash(),
    }
    
    # Encode user_data as CBOR
    user_data_cbor = cbor2.dumps(user_data)
    
    # Try to get real Nitro attestation
    try:
        from nsm_lib import get_attestation_document as get_nsm_attestation
        
        # Request attestation from NSM device
        attestation_response = get_nsm_attestation(
            user_data=user_data_cbor,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )
        
        # Extract the raw COSE_Sign1 attestation document from NSM response
        # NSM returns: {"Attestation": {"document": <bytes - COSE_Sign1>}}
        # The "document" is already the raw COSE_Sign1 structure (CBOR bytes)
        import base64
        attestation_bytes = attestation_response["Attestation"]["document"]
        attestation_b64 = base64.b64encode(attestation_bytes).decode()
        
        print(f"[TEE] ✅ Generated REAL Nitro attestation for epoch {epoch_id}", flush=True)
        
        return {
            "attestation_b64": attestation_b64,
            "user_data": user_data,
            "is_mock": False
        }
        
    except Exception as e:
        # Fall back to mock attestation (for development/testing)
        print(f"[TEE] ⚠️ NSM not available ({e}), using mock attestation", flush=True)
        
        import base64
        mock_attestation = {
            "mock": True,
            "user_data": user_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        attestation_b64 = base64.b64encode(cbor2.dumps(mock_attestation)).decode()
        
        return {
            "attestation_b64": attestation_b64,
            "user_data": user_data,
            "is_mock": True
        }


# ============================================================================
# RPC HANDLER
# ============================================================================

def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle RPC request from parent EC2.
    
    Supported commands:
    - get_public_key: Return enclave's public key
    - sign_weights: Sign a weights hash
    - get_attestation: Get attestation document for epoch
    - health: Health check
    """
    command = request.get("command")
    
    try:
        if command == "get_public_key":
            return {
                "status": "ok",
                "public_key": get_public_key(),
                "code_hash": compute_code_hash(),
                "boot_id": boot_id,
            }
        
        elif command == "sign_weights":
            weights_hash = request.get("weights_hash")
            if not weights_hash:
                return {"status": "error", "error": "Missing weights_hash"}
            
            signature = sign_weights_hash(weights_hash)
            return {
                "status": "ok",
                "signature": signature,
                "public_key": public_key_hex,
            }
        
        elif command == "get_attestation":
            epoch_id = request.get("epoch_id")
            if epoch_id is None:
                return {"status": "error", "error": "Missing epoch_id"}
            
            attestation = get_attestation_document(epoch_id)
            return {
                "status": "ok",
                **attestation
            }
        
        elif command == "health":
            return {
                "status": "ok",
                "service": "validator_tee",
                "keypair_initialized": private_key is not None,
                "boot_id": boot_id,
            }
        
        else:
            return {"status": "error", "error": f"Unknown command: {command}"}
            
    except Exception as e:
        print(f"[TEE] ❌ Error handling {command}: {e}", flush=True)
        return {"status": "error", "error": str(e)}


# ============================================================================
# VSOCK SERVER
# ============================================================================

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
            
            # Receive request
            data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Check for end of JSON
                try:
                    request = json.loads(data.decode())
                    break
                except json.JSONDecodeError:
                    continue
            
            if data:
                request = json.loads(data.decode())
                print(f"[TEE] Request: {request.get('command')}", flush=True)
                
                # Handle request
                response = handle_request(request)
                
                # Send response
                response_data = json.dumps(response).encode()
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

