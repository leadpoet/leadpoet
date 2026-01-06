"""
LeadPoet Canonical Event Verification Functions

This module provides canonical event verification for the TEE logging system.

CANONICAL LOG_ENTRY STRUCTURE:
{
    "signed_event": {
        "event_type": "...",
        "timestamp": "2024-01-01T00:00:00Z",  # ONLY timestamp location
        "boot_id": "uuid-of-this-boot-session",
        "monotonic_seq": 12345,
        "prev_event_hash": "abc123...",
        "payload": { ... }  # NO timestamp here!
    },
    "event_hash": "sha256(canonical_json(signed_event)).hex()",
    "enclave_pubkey": "hex",
    "enclave_signature": "hex"
}

Security Model:
- Hash chain (prev_event_hash) is the PRIMARY integrity mechanism
- (boot_id, monotonic_seq) provides ordering within a boot session
- ENCLAVE_RESTART events mark boot boundaries
- All components MUST use verify_log_entry() for signature verification
"""

import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List


def compute_event_hash(signed_event: Dict[str, Any]) -> str:
    """
    Compute the canonical hash of a signed_event.
    
    Uses JSON canonicalization with sorted keys and no extra whitespace.
    
    Args:
        signed_event: The signed_event dictionary to hash
        
    Returns:
        SHA256 hash as hex string (64 characters)
    """
    canonical_json = json.dumps(signed_event, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def verify_log_entry(log_entry: Dict[str, Any], expected_pubkey: Optional[str] = None) -> bool:
    """
    Verify enclave signature on a log_entry using CANONICAL format.
    
    This function:
    1. Extracts fields from the canonical log_entry structure
    2. Verifies pubkey matches expected (if provided)
    3. Recomputes hash from signed_event
    4. Verifies Ed25519 signature over hash bytes
    
    Args:
        log_entry: Log entry in canonical format (see module docstring)
        expected_pubkey: Expected enclave public key (hex). If None, uses
                        the pubkey from the log_entry (useful for initial verification)
        
    Returns:
        True if signature is valid and hash matches
        False on any error (fail-closed)
        
    Security Notes:
        - Recomputes hash from signed_event, not trusted from log_entry
        - Signature is over raw hash bytes (32 bytes), not hex string
        - If expected_pubkey is None, the pubkey from log_entry is used
          (only use this for initial bootstrap/discovery scenarios)
    """
    try:
        # Import here to avoid circular imports and make module usable without cryptography
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        
        # Extract fields from canonical log_entry structure
        signed_event = log_entry.get("signed_event")
        claimed_hash = log_entry.get("event_hash")
        signature = log_entry.get("enclave_signature")
        pubkey = log_entry.get("enclave_pubkey")
        
        # Check required fields
        if not all([signed_event, claimed_hash, signature, pubkey]):
            return False
        
        # Verify pubkey matches expected key (if provided)
        if expected_pubkey is not None and pubkey != expected_pubkey:
            return False
        
        # Recompute hash from signed_event (don't trust claimed_hash)
        computed_hash = compute_event_hash(signed_event)
        
        # Verify hash matches
        if computed_hash != claimed_hash:
            return False
        
        # Verify signature over hash BYTES (not hex string)
        verify_pubkey = expected_pubkey if expected_pubkey else pubkey
        pubkey_bytes = bytes.fromhex(verify_pubkey)
        pk = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        
        # Ed25519 verify: signature over the raw 32 bytes
        pk.verify(bytes.fromhex(signature), bytes.fromhex(claimed_hash))
        return True
        
    except Exception:
        # Fail-closed on any error
        return False


def verify_log_entry_detailed(
    log_entry: Dict[str, Any], 
    expected_pubkey: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Verify log entry with detailed error messages (for debugging/logging).
    
    Same as verify_log_entry but returns error description on failure.
    
    Args:
        log_entry: Log entry in canonical format
        expected_pubkey: Expected enclave public key (hex)
        
    Returns:
        Tuple of (success, error_message)
        - success: True if verification passed
        - error_message: Description of failure (or None on success)
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        
        # Extract fields
        signed_event = log_entry.get("signed_event")
        claimed_hash = log_entry.get("event_hash")
        signature = log_entry.get("enclave_signature")
        pubkey = log_entry.get("enclave_pubkey")
        
        # Check required fields
        if not signed_event:
            return False, "Missing signed_event"
        if not claimed_hash:
            return False, "Missing event_hash"
        if not signature:
            return False, "Missing enclave_signature"
        if not pubkey:
            return False, "Missing enclave_pubkey"
        
        # Verify pubkey matches expected
        if expected_pubkey is not None and pubkey != expected_pubkey:
            return False, f"Pubkey mismatch: expected {expected_pubkey[:16]}..., got {pubkey[:16]}..."
        
        # Recompute hash
        computed_hash = compute_event_hash(signed_event)
        
        # Verify hash matches
        if computed_hash != claimed_hash:
            return False, f"Hash mismatch: computed {computed_hash[:32]}..., claimed {claimed_hash[:32]}..."
        
        # Verify signature
        verify_pubkey = expected_pubkey if expected_pubkey else pubkey
        pubkey_bytes = bytes.fromhex(verify_pubkey)
        pk = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        pk.verify(bytes.fromhex(signature), bytes.fromhex(claimed_hash))
        
        return True, None
        
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def extract_event_chain_info(log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract hash-chain information from a log entry.
    
    Returns:
        Dictionary with chain info, or None if extraction fails:
        {
            "event_hash": "...",
            "prev_event_hash": "...",
            "boot_id": "...",
            "monotonic_seq": 123,
            "event_type": "...",
            "timestamp": "..."
        }
    """
    try:
        signed_event = log_entry.get("signed_event", {})
        
        return {
            "event_hash": log_entry.get("event_hash"),
            "prev_event_hash": signed_event.get("prev_event_hash"),
            "boot_id": signed_event.get("boot_id"),
            "monotonic_seq": signed_event.get("monotonic_seq"),
            "event_type": signed_event.get("event_type"),
            "timestamp": signed_event.get("timestamp"),
        }
    except Exception:
        return None


def verify_hash_chain_continuity(events: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Verify hash-chain continuity across a sequence of events.
    
    Each event's prev_event_hash should equal the previous event's event_hash.
    ENCLAVE_RESTART events are allowed to have different prev_event_hash
    (they link to the last known event before restart).
    
    Args:
        events: List of log_entry dictionaries, in chronological order
        
    Returns:
        Tuple of (success, error_message)
    """
    if not events:
        return True, None
    
    for i in range(1, len(events)):
        current = events[i]
        previous = events[i-1]
        
        current_info = extract_event_chain_info(current)
        previous_info = extract_event_chain_info(previous)
        
        if not current_info or not previous_info:
            return False, f"Failed to extract chain info at index {i}"
        
        # ENCLAVE_RESTART events may link to a different chain point
        if current_info["event_type"] == "ENCLAVE_RESTART":
            # Restart events are allowed - they indicate enclave reboot
            continue
        
        # Normal events must chain to previous
        if current_info["prev_event_hash"] != previous_info["event_hash"]:
            return False, (
                f"Hash chain broken at index {i}: "
                f"prev_event_hash={current_info['prev_event_hash'][:16]}... "
                f"!= previous event_hash={previous_info['event_hash'][:16]}..."
            )
    
    return True, None


def verify_monotonic_sequence(events: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Verify monotonic sequence numbers within each boot session.
    
    Within each boot_id, monotonic_seq must be strictly increasing.
    ENCLAVE_RESTART events start a new boot session with potentially reset sequence.
    
    Args:
        events: List of log_entry dictionaries, in chronological order
        
    Returns:
        Tuple of (success, error_message)
    """
    if not events:
        return True, None
    
    # Group events by boot_id
    boot_sequences: Dict[str, List[int]] = {}
    
    for i, event in enumerate(events):
        info = extract_event_chain_info(event)
        if not info:
            return False, f"Failed to extract chain info at index {i}"
        
        boot_id = info["boot_id"]
        seq = info["monotonic_seq"]
        
        if boot_id not in boot_sequences:
            boot_sequences[boot_id] = []
        
        boot_sequences[boot_id].append(seq)
    
    # Verify monotonicity within each boot
    for boot_id, sequences in boot_sequences.items():
        for i in range(1, len(sequences)):
            if sequences[i] <= sequences[i-1]:
                return False, (
                    f"Non-monotonic sequence in boot {boot_id[:8]}...: "
                    f"seq[{i-1}]={sequences[i-1]} >= seq[{i}]={sequences[i]}"
                )
    
    return True, None


# =============================================================================
# UNIT TESTS
# =============================================================================

def _create_test_log_entry(event_type: str, payload: Dict, prev_hash: Optional[str] = None) -> Dict:
    """Helper to create test log entries (for testing only)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from leadpoet_canonical.timestamps import canonical_timestamp
    import uuid
    
    # Generate test keypair
    private_key = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    # Build signed_event
    signed_event = {
        "event_type": event_type,
        "timestamp": canonical_timestamp(),
        "boot_id": str(uuid.uuid4()),
        "monotonic_seq": 1,
        "prev_event_hash": prev_hash,
        "payload": payload,
    }
    
    # Compute hash and sign
    event_hash = compute_event_hash(signed_event)
    signature = private_key.sign(bytes.fromhex(event_hash))
    
    return {
        "signed_event": signed_event,
        "event_hash": event_hash,
        "enclave_pubkey": public_key_bytes.hex(),
        "enclave_signature": signature.hex(),
    }


def test_verify_log_entry_valid():
    """Test verification of a valid log entry."""
    log_entry = _create_test_log_entry("TEST", {"data": "hello"})
    
    # Should pass with correct pubkey
    assert verify_log_entry(log_entry, log_entry["enclave_pubkey"]) == True
    
    # Should pass with no expected pubkey (uses entry's pubkey)
    assert verify_log_entry(log_entry) == True
    
    print("✅ Valid log entry verification test passed")


def test_verify_log_entry_wrong_pubkey():
    """Test rejection of log entry with wrong pubkey."""
    log_entry = _create_test_log_entry("TEST", {"data": "hello"})
    
    # Generate different pubkey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    other_key = Ed25519PrivateKey.generate()
    other_pubkey = other_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    ).hex()
    
    # Should fail with wrong pubkey
    assert verify_log_entry(log_entry, other_pubkey) == False
    
    print("✅ Wrong pubkey rejection test passed")


def test_verify_log_entry_tampered_payload():
    """Test rejection of tampered payload."""
    log_entry = _create_test_log_entry("TEST", {"data": "hello"})
    
    # Tamper with payload
    log_entry["signed_event"]["payload"]["data"] = "tampered"
    
    # Should fail (hash won't match)
    assert verify_log_entry(log_entry) == False
    
    print("✅ Tampered payload rejection test passed")


def test_verify_log_entry_tampered_hash():
    """Test rejection of tampered hash."""
    log_entry = _create_test_log_entry("TEST", {"data": "hello"})
    
    # Tamper with claimed hash
    log_entry["event_hash"] = "a" * 64
    
    # Should fail (signature won't match)
    assert verify_log_entry(log_entry) == False
    
    print("✅ Tampered hash rejection test passed")


def test_compute_event_hash_deterministic():
    """Test that hash computation is deterministic."""
    signed_event = {
        "event_type": "TEST",
        "timestamp": "2024-01-01T00:00:00Z",
        "boot_id": "test-boot",
        "monotonic_seq": 1,
        "prev_event_hash": None,
        "payload": {"data": "hello"},
    }
    
    hash1 = compute_event_hash(signed_event)
    hash2 = compute_event_hash(signed_event)
    
    assert hash1 == hash2, "Hash should be deterministic"
    assert len(hash1) == 64, "Hash should be 64 hex characters"
    
    print("✅ Hash determinism test passed")


def test_extract_event_chain_info():
    """Test extraction of chain info."""
    log_entry = _create_test_log_entry("TEST", {"data": "hello"}, prev_hash="abc123")
    
    info = extract_event_chain_info(log_entry)
    
    assert info is not None, "Should extract chain info"
    assert info["event_type"] == "TEST"
    assert info["prev_event_hash"] == "abc123"
    assert info["monotonic_seq"] == 1
    
    print("✅ Extract chain info test passed")


if __name__ == "__main__":
    print("Running leadpoet_canonical/events.py unit tests...\n")
    
    test_compute_event_hash_deterministic()
    test_verify_log_entry_valid()
    test_verify_log_entry_wrong_pubkey()
    test_verify_log_entry_tampered_payload()
    test_verify_log_entry_tampered_hash()
    test_extract_event_chain_info()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

