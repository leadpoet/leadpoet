"""
Signature Verification Utility
==============================

Wallet signature verification for trustless authentication.

Uses Bittensor Keypair for Ed25519 signature verification.
No JWT tokens, no server-issued credentials.
"""

import hashlib
import json
from typing import Any, Dict
from bittensor import Keypair


def verify_wallet_signature(
    message: str,
    signature: str,
    ss58_address: str
) -> bool:
    """
    Verify Ed25519 signature from Bittensor wallet.
    
    Args:
        message: Message that was signed
        signature: Hex-encoded signature
        ss58_address: SS58 address of signer
    
    Returns:
        True if signature is valid
    
    Example:
        >>> message = "SUBMISSION_REQUEST:5GNJq...:uuid:1234567890:hash:build-id"
        >>> signature = "0xabc123..."
        >>> ss58_address = "5GNJqR7T..."
        >>> verify_wallet_signature(message, signature, ss58_address)
        True
    """
    try:
        # Create keypair from SS58 address (public key only)
        keypair = Keypair(ss58_address=ss58_address)
        
        # Convert hex signature to bytes
        # Handle both '0x' prefixed and non-prefixed hex strings
        if signature.startswith('0x'):
            signature = signature[2:]
        
        signature_bytes = bytes.fromhex(signature)
        
        # Verify signature
        is_valid = keypair.verify(message, signature_bytes)
        
        return is_valid
    
    except ValueError as e:
        print(f"Signature format error: {e}")
        return False
    
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA256 hash of payload.
    
    Uses JSON canonicalization (sort_keys=True) to ensure
    deterministic hashing regardless of key order.
    
    Args:
        payload: Event payload (dict)
    
    Returns:
        SHA256 hex string (64 characters)
    
    Example:
        >>> payload = {"lead_id": "123", "cid": "abc"}
        >>> compute_payload_hash(payload)
        'a3b2c1d4e5f6...'
    """
    # Canonicalize JSON (sort keys for determinism)
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)  # Handle datetime objects
    
    # Compute SHA256 hash
    payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
    
    return payload_hash


def construct_signed_message(event) -> str:
    """
    Construct the message that was signed by the actor.
    
    Format: {event_type}:{actor_hotkey}:{nonce}:{ts}:{payload_hash}:{build_id}
    
    Args:
        event: Event object with required fields
    
    Returns:
        Message string that should be signed
    
    Example:
        >>> event = SubmissionRequestEvent(...)
        >>> construct_signed_message(event)
        'SUBMISSION_REQUEST:5GNJqR7T...:550e8400-e29b-...:2025-11-02T12:00:00:abc123...:commit-hash'
    """
    # Extract timestamp as ISO format string
    if hasattr(event.ts, 'isoformat'):
        ts_str = event.ts.isoformat()
    else:
        ts_str = str(event.ts)
    
    # Extract event type string value (not enum representation)
    event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
    
    # Construct message in exact format
    message = f"{event_type_str}:{event.actor_hotkey}:{event.nonce}:{ts_str}:{event.payload_hash}:{event.build_id}"
    
    return message


def verify_event_signature(event) -> bool:
    """
    Convenience function to verify an event's signature.
    
    Combines construct_signed_message() and verify_wallet_signature().
    
    Args:
        event: Event object with signature field
    
    Returns:
        True if signature is valid
    
    Example:
        >>> event = SubmissionRequestEvent(...)
        >>> verify_event_signature(event)
        True
    """
    try:
        message = construct_signed_message(event)
        return verify_wallet_signature(message, event.signature, event.actor_hotkey)
    except Exception as e:
        print(f"Event signature verification error: {e}")
        return False


def sign_message(message: str, keypair: Keypair) -> str:
    """
    Sign a message with a Bittensor wallet keypair.
    
    This function is for CLIENT-SIDE use (miners/validators signing requests).
    The gateway only VERIFIES signatures, it never signs.
    
    Args:
        message: Message to sign
        keypair: Bittensor Keypair with private key
    
    Returns:
        Hex-encoded signature (with '0x' prefix)
    
    Example:
        >>> from substrateinterface import Keypair
        >>> keypair = Keypair.create_from_mnemonic("your seed phrase...")
        >>> message = "SUBMISSION_REQUEST:5GNJq...:uuid:..."
        >>> signature = sign_message(message, keypair)
        >>> signature
        '0xabc123def456...'
    """
    signature_bytes = keypair.sign(message)
    signature_hex = '0x' + signature_bytes.hex()
    return signature_hex

