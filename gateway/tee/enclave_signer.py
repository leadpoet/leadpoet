"""
Gateway TEE Enclave Signer
==========================

This module provides enclave key generation and event signing for the gateway TEE.

SECURITY MODEL:
- Ed25519 keypair generated at boot, private key never leaves memory
- All events are signed and hash-chained for tamper-evidence
- Hash-chain (prev_event_hash) is the PRIMARY integrity mechanism
- (boot_id, monotonic_seq) provides ordering within a boot session
- Thread-safe: _SIGN_LOCK protects hash-chain from concurrent modification

SINGLE-PROCESS REQUIREMENT:
The gateway MUST run as a single process. Multiple workers would create
divergent hash-chains with different _PREV_EVENT_HASH values, breaking
auditability. Fail fast if WEB_CONCURRENCY or UVICORN_WORKERS > 1.

CANONICAL ENVELOPE FORMAT:
- signed_event: {event_type, timestamp, boot_id, monotonic_seq, prev_event_hash, payload}
- log_entry: {signed_event, event_hash, enclave_pubkey, enclave_signature}

TIMESTAMP RULE:
The timestamp field lives ONLY in signed_event.timestamp, NOT in payload.
This module is the SINGLE SOURCE OF TRUTH for timestamps.
"""

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

# Use canonical timestamp from shared module (Issue 18)
from leadpoet_canonical.timestamps import canonical_timestamp
from leadpoet_canonical.constants import EVENT_TYPE_ENCLAVE_RESTART


# =============================================================================
# GLOBAL STATE (Module-level, initialized at boot)
# =============================================================================

# Enclave keypair (generated at boot, never exported)
_ENCLAVE_PRIVATE_KEY: Optional[Ed25519PrivateKey] = None
_ENCLAVE_PUBLIC_KEY_BYTES: Optional[bytes] = None

# Attestation document (generated at boot)
_ATTESTATION_DOCUMENT: Optional[bytes] = None

# Boot identification (generated once at module load)
# boot_id allows verifiers to detect enclave restarts
_BOOT_ID: str = str(uuid.uuid4())

# Monotonic sequence counter
# - Resets to 0 on each boot
# - Strictly increasing within boot
# - Used with boot_id to order events within a boot session
_MONOTONIC_COUNTER: int = 0

# Hash-chain state
# - Each event includes prev_event_hash linking to previous event
# - This is the PRIMARY integrity mechanism
# - Allows verifiers to detect missing/reordered events
_PREV_EVENT_HASH: Optional[str] = None

# Thread safety lock
# ALL hash-chain operations must be atomic to prevent forking
# under concurrent FastAPI requests
_SIGN_LOCK = threading.Lock()

# Cached code hash (set at boot by TEEService)
_CACHED_CODE_HASH: Optional[str] = None


# =============================================================================
# KEYPAIR MANAGEMENT
# =============================================================================

def initialize_enclave_keypair() -> str:
    """
    Generate enclave keypair at boot time.
    
    This MUST be called once at enclave startup.
    The private key is stored in memory only - never written to disk.
    
    Returns:
        Public key as hex string (64 characters)
        
    Raises:
        RuntimeError: If already initialized (cannot reinitialize)
        
    Security Notes:
        - Private key never leaves memory
        - Key material is lost on enclave restart (by design)
        - New keypair = new attestation required
    """
    global _ENCLAVE_PRIVATE_KEY, _ENCLAVE_PUBLIC_KEY_BYTES
    
    if _ENCLAVE_PRIVATE_KEY is not None:
        raise RuntimeError("Enclave keypair already initialized - cannot reinitialize")
    
    # Generate Ed25519 keypair
    _ENCLAVE_PRIVATE_KEY = Ed25519PrivateKey.generate()
    _ENCLAVE_PUBLIC_KEY_BYTES = _ENCLAVE_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    pubkey_hex = _ENCLAVE_PUBLIC_KEY_BYTES.hex()
    print(f"[TEE] 🔐 Enclave keypair generated", flush=True)
    print(f"[TEE]    Public key: {pubkey_hex[:16]}...{pubkey_hex[-16:]}", flush=True)
    
    return pubkey_hex


def get_enclave_public_key() -> bytes:
    """
    Get the enclave public key (raw 32 bytes).
    
    Raises:
        RuntimeError: If keypair not initialized
    """
    if _ENCLAVE_PUBLIC_KEY_BYTES is None:
        raise RuntimeError("Enclave keypair not initialized - call initialize_enclave_keypair() first")
    return _ENCLAVE_PUBLIC_KEY_BYTES


def get_enclave_public_key_hex() -> str:
    """
    Get the enclave public key as hex string (64 characters).
    
    Raises:
        RuntimeError: If keypair not initialized
    """
    return get_enclave_public_key().hex()


def is_keypair_initialized() -> bool:
    """Check if the enclave keypair has been initialized."""
    return _ENCLAVE_PRIVATE_KEY is not None


# =============================================================================
# CODE HASH MANAGEMENT
# =============================================================================

def set_cached_code_hash(code_hash: str):
    """
    Set the code hash at boot (called by TEEService).
    
    The code hash is computed ONCE at boot and cached.
    It should never be recomputed during runtime.
    
    Args:
        code_hash: SHA256 hash of gateway code
    """
    global _CACHED_CODE_HASH
    _CACHED_CODE_HASH = code_hash
    print(f"[TEE] 📋 Code hash cached: {code_hash[:16]}...{code_hash[-16:]}", flush=True)


def get_cached_code_hash() -> str:
    """
    Get the cached code hash (set at boot, never recomputed).
    
    Raises:
        RuntimeError: If code hash not cached (TEE not initialized)
    """
    if _CACHED_CODE_HASH is None:
        raise RuntimeError("Code hash not cached - TEE not initialized")
    return _CACHED_CODE_HASH


# =============================================================================
# EVENT SIGNING (CORE FUNCTION)
# =============================================================================

def sign_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sign an event with the enclave private key.
    
    This is the CORE function for creating signed log entries.
    All events pass through this function to ensure consistent
    signing, hash-chaining, and timestamp management.
    
    CANONICAL ENVELOPE FORMAT:
    - signed_event: {event_type, timestamp, boot_id, monotonic_seq, prev_event_hash, payload}
    - log_entry: {signed_event, event_hash, enclave_pubkey, enclave_signature}
    
    Hash-chaining: Each event includes prev_event_hash linking to previous event.
    This allows verifiers to detect missing/reordered events.
    On restart, emit ENCLAVE_RESTART event with prev_log_tip_event_hash.
    
    Args:
        event_type: Type of event (e.g., "WEIGHT_SUBMISSION", "ENCLAVE_RESTART")
        payload: Event-specific data (NO timestamp - this function adds it)
        
    Returns:
        log_entry dictionary - store as-is, never repack fields
        
    Raises:
        RuntimeError: If enclave keypair not initialized
        
    THREAD SAFETY:
        This function is protected by _SIGN_LOCK. The hash-chain
        MUST be modified atomically to prevent forking under concurrent requests.
        
    TIMESTAMP RULE:
        Do NOT include timestamp in payload - this function is the
        SINGLE SOURCE OF TRUTH for timestamps via canonical_timestamp().
    """
    global _MONOTONIC_COUNTER, _PREV_EVENT_HASH
    
    if _ENCLAVE_PRIVATE_KEY is None:
        raise RuntimeError("Enclave keypair not initialized")
    
    # Acquire lock before modifying hash-chain state
    # ALL hash-chain operations must be atomic to prevent forking
    with _SIGN_LOCK:
        _MONOTONIC_COUNTER += 1
        
        # Build signed_event (canonical structure)
        # - boot_id + monotonic_seq: ordering within a boot session
        # - prev_event_hash: hash-chain is the PRIMARY integrity mechanism
        # - timestamp: from canonical_timestamp() (single source of truth)
        
        ts = canonical_timestamp()
        
        signed_event = {
            "event_type": event_type,
            "timestamp": ts,
            "boot_id": _BOOT_ID,
            "monotonic_seq": _MONOTONIC_COUNTER,
            "prev_event_hash": _PREV_EVENT_HASH,
            "payload": payload,
        }
        
        # Hash the signed_event using canonical JSON serialization
        # sort_keys=True ensures deterministic ordering
        # separators=(',', ':') ensures no whitespace
        canonical_json = json.dumps(signed_event, sort_keys=True, separators=(',', ':'))
        event_hash_bytes = hashlib.sha256(canonical_json.encode('utf-8')).digest()
        event_hash_hex = event_hash_bytes.hex()
        
        # Ed25519 signature over the 32 raw hash bytes (NOT the hex string)
        signature = _ENCLAVE_PRIVATE_KEY.sign(event_hash_bytes)
        
        # Update hash chain for next event
        _PREV_EVENT_HASH = event_hash_hex
        
        # Return log_entry (store this exactly, never reshuffle fields)
        return {
            "signed_event": signed_event,
            "event_hash": event_hash_hex,
            "enclave_pubkey": _ENCLAVE_PUBLIC_KEY_BYTES.hex(),
            "enclave_signature": signature.hex(),
        }


# =============================================================================
# RESTART EVENT
# =============================================================================

def publish_restart_event(prev_log_tip_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Emit an ENCLAVE_RESTART event on boot to maintain hash-chain integrity.
    
    This event allows verifiers to distinguish "enclave restarted" from
    "events missing". It MUST be emitted BEFORE any other events on boot.
    
    Verifiers check:
    - Hash-chain continuity across the stream
    - (boot_id, monotonic_seq) is monotonic within each boot
    - ENCLAVE_RESTART events mark boot boundaries
    
    Args:
        prev_log_tip_hash: Last known event hash from previous boot session
                          (fetched from Supabase/Arweave). If None, this is
                          the first boot or we couldn't determine the tip.
                          
    Returns:
        log_entry for the ENCLAVE_RESTART event
        
    Usage:
        # On gateway startup:
        prev_tip = fetch_last_log_tip()  # Query Supabase
        restart_entry = publish_restart_event(prev_tip)
        store_restart_entry(restart_entry)  # Store before any other logs
    """
    global _PREV_EVENT_HASH
    
    # On restart, link to the last known event from previous boot
    _PREV_EVENT_HASH = prev_log_tip_hash
    
    # Emit restart event with boot information
    return sign_event(EVENT_TYPE_ENCLAVE_RESTART, {
        "new_boot_id": _BOOT_ID,
        "enclave_pubkey": _ENCLAVE_PUBLIC_KEY_BYTES.hex(),
        "prev_log_tip_event_hash": prev_log_tip_hash,
        "reason": "enclave_boot",
    })


def get_boot_id() -> str:
    """Get the current boot ID (UUID generated at module load)."""
    return _BOOT_ID


# =============================================================================
# ATTESTATION DOCUMENT
# =============================================================================

def generate_attestation_document(code_hash: str) -> bytes:
    """
    Generate AWS Nitro attestation document binding public key to code hash.
    
    This document is signed by AWS Nitro hardware and proves:
    - The public key was generated inside THIS specific enclave
    - The enclave is running code with THIS specific measurement (PCR0)
    
    Args:
        code_hash: SHA256 hash of gateway code
        
    Returns:
        Attestation document (CBOR-encoded, AWS-signed)
        Empty bytes if NSM is not available (development mode)
        
    Security Notes:
        - In production, attestation generation MUST succeed
        - The attestation binds our pubkey to the enclave measurement
        - user_data contains: purpose, enclave_pubkey, code_hash
        - NO timestamps in user_data (must be deterministic)
        - NO epoch_id (gateway signs events across many epochs)
    """
    global _ATTESTATION_DOCUMENT
    
    if _ENCLAVE_PUBLIC_KEY_BYTES is None:
        raise RuntimeError("Enclave keypair not initialized")
    
    try:
        # Import NSM library for attestation
        from gateway.tee.nsm_lib import get_attestation_document as nsm_get_attestation_doc
        import cbor2
        
        # Build user_data for attestation
        # Purpose distinguishes gateway from validator attestations
        # NO timestamps - must be deterministic for verification
        user_data = cbor2.dumps({
            "purpose": "gateway_event_signing",  # MUST match spec exactly
            "enclave_pubkey": _ENCLAVE_PUBLIC_KEY_BYTES.hex(),
            "code_hash": code_hash,
            # NO generated_at - non-deterministic, breaks verification
            # NO epoch_id - gateway signs events across many epochs
        })
        
        # Request attestation from NSM
        attestation_response = nsm_get_attestation_doc(
            user_data=user_data,
            public_key=_ENCLAVE_PUBLIC_KEY_BYTES,
        )
        
        # Extract the attestation document bytes
        _ATTESTATION_DOCUMENT = attestation_response.get("Attestation", {}).get("document", b"")
        
        print(f"[TEE] 📜 Attestation document generated ({len(_ATTESTATION_DOCUMENT)} bytes)", flush=True)
        
        return _ATTESTATION_DOCUMENT
        
    except Exception as e:
        print(f"[TEE] ⚠️  Failed to generate attestation: {e}", flush=True)
        # In development (no NSM), return a placeholder
        # CRITICAL: In production, this MUST succeed
        import traceback
        traceback.print_exc()
        _ATTESTATION_DOCUMENT = b""
        return b""


def get_attestation_document() -> bytes:
    """
    Get the cached attestation document.
    
    Raises:
        RuntimeError: If attestation not generated
    """
    if _ATTESTATION_DOCUMENT is None:
        raise RuntimeError("Attestation document not generated")
    return _ATTESTATION_DOCUMENT


def get_attestation_document_b64() -> str:
    """Get the attestation document as base64 string."""
    import base64
    return base64.b64encode(get_attestation_document()).decode('ascii')


# =============================================================================
# STATE ACCESSORS
# =============================================================================

def get_monotonic_counter() -> int:
    """Get current monotonic counter value."""
    return _MONOTONIC_COUNTER


def get_prev_event_hash() -> Optional[str]:
    """Get the current hash chain tip (last event hash)."""
    return _PREV_EVENT_HASH


def get_signer_state() -> Dict[str, Any]:
    """
    Get current signer state for debugging/monitoring.
    
    Returns:
        Dictionary with current state (safe to expose)
    """
    return {
        "initialized": is_keypair_initialized(),
        "boot_id": _BOOT_ID,
        "monotonic_counter": _MONOTONIC_COUNTER,
        "has_attestation": _ATTESTATION_DOCUMENT is not None and len(_ATTESTATION_DOCUMENT) > 0,
        "has_code_hash": _CACHED_CODE_HASH is not None,
        "prev_event_hash": _PREV_EVENT_HASH[:16] + "..." if _PREV_EVENT_HASH else None,
        "pubkey": _ENCLAVE_PUBLIC_KEY_BYTES.hex()[:16] + "..." if _ENCLAVE_PUBLIC_KEY_BYTES else None,
    }


# =============================================================================
# UNIT TESTS
# =============================================================================

def _reset_for_testing():
    """Reset global state for testing only. DO NOT USE IN PRODUCTION."""
    global _ENCLAVE_PRIVATE_KEY, _ENCLAVE_PUBLIC_KEY_BYTES, _ATTESTATION_DOCUMENT
    global _MONOTONIC_COUNTER, _PREV_EVENT_HASH, _CACHED_CODE_HASH, _BOOT_ID
    
    _ENCLAVE_PRIVATE_KEY = None
    _ENCLAVE_PUBLIC_KEY_BYTES = None
    _ATTESTATION_DOCUMENT = None
    _MONOTONIC_COUNTER = 0
    _PREV_EVENT_HASH = None
    _CACHED_CODE_HASH = None
    _BOOT_ID = str(uuid.uuid4())


def test_initialize_keypair():
    """Test keypair initialization."""
    _reset_for_testing()
    
    pubkey = initialize_enclave_keypair()
    assert len(pubkey) == 64, "Pubkey should be 64 hex chars"
    assert is_keypair_initialized(), "Should be initialized"
    
    # Should fail on reinitialize
    try:
        initialize_enclave_keypair()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "already initialized" in str(e)
    
    print("✅ Initialize keypair test passed")


def test_sign_event():
    """Test event signing."""
    _reset_for_testing()
    initialize_enclave_keypair()
    
    log_entry = sign_event("TEST", {"data": "hello"})
    
    # Check structure
    assert "signed_event" in log_entry
    assert "event_hash" in log_entry
    assert "enclave_pubkey" in log_entry
    assert "enclave_signature" in log_entry
    
    # Check signed_event fields
    signed_event = log_entry["signed_event"]
    assert signed_event["event_type"] == "TEST"
    assert "timestamp" in signed_event
    assert "boot_id" in signed_event
    assert signed_event["monotonic_seq"] == 1
    assert signed_event["prev_event_hash"] is None  # First event
    assert signed_event["payload"] == {"data": "hello"}
    
    print("✅ Sign event test passed")


def test_hash_chain():
    """Test hash-chain integrity."""
    _reset_for_testing()
    initialize_enclave_keypair()
    
    e1 = sign_event("TEST", {"seq": 1})
    e2 = sign_event("TEST", {"seq": 2})
    e3 = sign_event("TEST", {"seq": 3})
    
    # Verify chain
    assert e1["signed_event"]["prev_event_hash"] is None
    assert e2["signed_event"]["prev_event_hash"] == e1["event_hash"]
    assert e3["signed_event"]["prev_event_hash"] == e2["event_hash"]
    
    # Verify monotonic sequence
    assert e1["signed_event"]["monotonic_seq"] == 1
    assert e2["signed_event"]["monotonic_seq"] == 2
    assert e3["signed_event"]["monotonic_seq"] == 3
    
    print("✅ Hash chain test passed")


def test_signature_verification():
    """Test that signatures can be verified."""
    from leadpoet_canonical.events import verify_log_entry
    
    _reset_for_testing()
    pubkey = initialize_enclave_keypair()
    
    log_entry = sign_event("TEST", {"data": "hello"})
    
    # Verify signature
    assert verify_log_entry(log_entry, pubkey) == True, "Signature should verify"
    
    # Tamper and verify fails
    tampered = dict(log_entry)
    tampered["signed_event"] = dict(tampered["signed_event"])
    tampered["signed_event"]["payload"] = {"data": "tampered"}
    assert verify_log_entry(tampered, pubkey) == False, "Tampered should fail"
    
    print("✅ Signature verification test passed")


def test_restart_event():
    """Test restart event emission."""
    _reset_for_testing()
    initialize_enclave_keypair()
    
    # Emit some events
    e1 = sign_event("TEST", {"seq": 1})
    e2 = sign_event("TEST", {"seq": 2})
    
    # Simulate restart with known tip
    prev_tip = e2["event_hash"]
    _reset_for_testing()
    initialize_enclave_keypair()
    
    restart_entry = publish_restart_event(prev_tip)
    
    # Check restart event
    assert restart_entry["signed_event"]["event_type"] == EVENT_TYPE_ENCLAVE_RESTART
    assert restart_entry["signed_event"]["prev_event_hash"] == prev_tip
    assert restart_entry["signed_event"]["payload"]["prev_log_tip_event_hash"] == prev_tip
    assert restart_entry["signed_event"]["monotonic_seq"] == 1  # Reset on boot
    
    print("✅ Restart event test passed")


if __name__ == "__main__":
    print("Running gateway/tee/enclave_signer.py unit tests...\n")
    
    test_initialize_keypair()
    test_sign_event()
    test_hash_chain()
    test_signature_verification()
    test_restart_event()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

