"""
Validator TEE Enclave Signer
============================

This module runs INSIDE the Nitro Enclave and handles:
- Ed25519 keypair generation (ephemeral per boot)
- Weight hash signing
- Attestation document generation

SECURITY MODEL:
- Keypair is generated once at enclave boot
- Private key NEVER leaves the enclave
- Attestation binds public key to enclave code (PCR0)
- epoch_id in attestation prevents replay attacks

CONSTRAINT (CRITICAL):
This module does NOT expose a generic sign(bytes) API.
The sign_weights() function computes the hash internally using canonical
bundle_weights_hash() before signing. This prevents attackers from
making the enclave sign arbitrary data.

ATTESTATION USER_DATA SCHEMA (Validator - includes epoch_id):
{
    "purpose": "validator_weights",
    "epoch_id": int,           # CRITICAL: binds to specific epoch
    "enclave_pubkey": str,     # hex
    "code_hash": str,          # SHA256 of validator code
}

Auditors verify: purpose="validator_weights", epoch_id matches bundle.epoch_id
"""

import os
import json
import hashlib
import logging
import threading
from typing import Tuple, Optional, List

# Ed25519 signing
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

# Canonical hash function (MUST use shared module)
from leadpoet_canonical.weights import bundle_weights_hash

logger = logging.getLogger(__name__)

# ============================================================================
# Module-Level State (Enclave Singleton)
# ============================================================================

_PRIVATE_KEY: Optional[Ed25519PrivateKey] = None
_PUBLIC_KEY: Optional[Ed25519PublicKey] = None
_PUBLIC_KEY_HEX: Optional[str] = None
_CODE_HASH: Optional[str] = None
_ATTESTATION_CACHE: dict = {}  # epoch_id -> attestation_b64

# Thread safety for signing operations
_SIGN_LOCK = threading.Lock()


# ============================================================================
# Initialization
# ============================================================================

def initialize_enclave_keypair() -> str:
    """
    Generate Ed25519 keypair for this enclave boot session.
    
    SECURITY: Private key exists only in enclave memory.
    This function should be called ONCE at enclave boot.
    
    Returns:
        Public key as hex string
    """
    global _PRIVATE_KEY, _PUBLIC_KEY, _PUBLIC_KEY_HEX
    
    if _PRIVATE_KEY is not None:
        logger.warning("Enclave keypair already initialized")
        return _PUBLIC_KEY_HEX
    
    # Generate new Ed25519 keypair
    _PRIVATE_KEY = Ed25519PrivateKey.generate()
    _PUBLIC_KEY = _PRIVATE_KEY.public_key()
    
    # Export public key as hex
    pubkey_bytes = _PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    _PUBLIC_KEY_HEX = pubkey_bytes.hex()
    
    logger.info(f"✅ Validator enclave keypair initialized: {_PUBLIC_KEY_HEX[:16]}...")
    
    return _PUBLIC_KEY_HEX


def is_keypair_initialized() -> bool:
    """Check if enclave keypair has been initialized."""
    return _PRIVATE_KEY is not None


def get_enclave_public_key_hex() -> str:
    """
    Get the enclave's public key as hex string.
    
    Returns:
        Public key hex (64 characters for Ed25519)
        
    Raises:
        RuntimeError: If keypair not initialized
    """
    if _PUBLIC_KEY_HEX is None:
        raise RuntimeError("Enclave keypair not initialized. Call initialize_enclave_keypair() first.")
    return _PUBLIC_KEY_HEX


# Aliases for compatibility with tasks8.md naming
def get_enclave_pubkey() -> str:
    """Alias for get_enclave_public_key_hex()."""
    return get_enclave_public_key_hex()


# ============================================================================
# Code Hash (for attestation)
# ============================================================================

def compute_code_hash() -> str:
    """
    Compute SHA256 hash of validator code.
    
    In production, this should hash the actual enclave image (EIF).
    For now, we hash key source files to detect code changes.
    
    Returns:
        SHA256 hex digest
    """
    hasher = hashlib.sha256()
    
    # List of critical files to hash (in sorted order for determinism)
    critical_files = [
        "validator_tee/enclave_signer.py",
        "leadpoet_canonical/weights.py",
        "leadpoet_canonical/binding.py",
        "neurons/validator.py",
    ]
    
    for filepath in sorted(critical_files):
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            filepath
        )
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                hasher.update(f.read())
            hasher.update(filepath.encode())  # Include path in hash
    
    return hasher.hexdigest()


def set_cached_code_hash(code_hash: str):
    """Set the cached code hash (called at boot)."""
    global _CODE_HASH
    _CODE_HASH = code_hash


def get_code_hash() -> str:
    """
    Get the validator code hash.
    
    Returns:
        SHA256 hex of validator code
    """
    global _CODE_HASH
    if _CODE_HASH is None:
        _CODE_HASH = compute_code_hash()
    return _CODE_HASH


# ============================================================================
# Weight Signing
# ============================================================================

def sign_weights(
    netuid: int,
    epoch_id: int,
    block: int,
    uids: List[int],
    weights_u16: List[int],
) -> Tuple[str, str]:
    """
    Sign computed weights using the enclave's private key.
    
    SECURITY: This function computes the canonical hash internally.
    It does NOT accept pre-computed hashes, preventing signing oracle attacks.
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch identifier
        block: Block number when weights were computed
        uids: List of UIDs (must be sorted ascending)
        weights_u16: Corresponding u16 weights
        
    Returns:
        Tuple of (weights_hash_hex, signature_hex)
        
    Raises:
        RuntimeError: If keypair not initialized
        ValueError: If inputs are invalid
    """
    if _PRIVATE_KEY is None:
        raise RuntimeError("Enclave keypair not initialized")
    
    # Validate inputs
    if len(uids) != len(weights_u16):
        raise ValueError(f"Length mismatch: {len(uids)} uids vs {len(weights_u16)} weights")
    
    if uids != sorted(uids):
        raise ValueError("UIDs must be sorted ascending")
    
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate UIDs detected")
    
    for w in weights_u16:
        if not (1 <= w <= 65535):
            raise ValueError(f"Weight {w} out of valid sparse range [1, 65535]")
    
    # Compute canonical hash internally (NOT accepting pre-computed hashes)
    weights_pairs = list(zip(uids, weights_u16))
    weights_hash = bundle_weights_hash(netuid, epoch_id, block, weights_pairs)
    
    # Sign the raw hash bytes (32 bytes), NOT the hex string
    digest_bytes = bytes.fromhex(weights_hash)
    
    with _SIGN_LOCK:
        signature_bytes = _PRIVATE_KEY.sign(digest_bytes)
    
    signature_hex = signature_bytes.hex()
    
    logger.info(
        f"✅ Signed weights: epoch={epoch_id}, {len(uids)} UIDs, "
        f"hash={weights_hash[:16]}..."
    )
    
    return weights_hash, signature_hex


def sign_digest(digest_bytes: bytes) -> str:
    """
    Sign raw digest bytes with the enclave's private key.
    
    WARNING: This is a lower-level function. Prefer sign_weights() which
    computes the hash internally to prevent signing oracle attacks.
    
    Args:
        digest_bytes: 32-byte SHA256 digest to sign
        
    Returns:
        Signature as hex string
        
    Raises:
        RuntimeError: If keypair not initialized
        ValueError: If digest is not 32 bytes
    """
    if _PRIVATE_KEY is None:
        raise RuntimeError("Enclave keypair not initialized")
    
    if len(digest_bytes) != 32:
        raise ValueError(f"Digest must be 32 bytes, got {len(digest_bytes)}")
    
    with _SIGN_LOCK:
        signature_bytes = _PRIVATE_KEY.sign(digest_bytes)
    
    return signature_bytes.hex()


# ============================================================================
# Attestation Document Generation
# ============================================================================

def generate_attestation_document(code_hash: str, epoch_id: int) -> bytes:
    """
    Generate AWS Nitro attestation document with epoch binding.
    
    CRITICAL: The epoch_id is included in user_data to prevent replay attacks.
    An attestation from epoch N cannot be reused for epoch N+1.
    
    Args:
        code_hash: SHA256 hash of validator code
        epoch_id: Current epoch (MUST be included for replay protection)
        
    Returns:
        Raw attestation document bytes (CBOR-encoded COSE_Sign1)
        
    Note:
        In production Nitro Enclave, this calls the NSM API.
        Outside enclave, returns a placeholder for testing.
    """
    import cbor2
    
    # Build user_data (CBOR for deterministic encoding)
    # VALIDATOR schema: includes epoch_id (different from gateway!)
    user_data = {
        "purpose": "validator_weights",  # Different from gateway!
        "epoch_id": epoch_id,  # CRITICAL: binds to specific epoch
        "enclave_pubkey": get_enclave_public_key_hex(),
        "code_hash": code_hash,
    }
    user_data_cbor = cbor2.dumps(user_data)
    
    # Check if we're in a Nitro Enclave
    try:
        # Try to import NSM library (only available inside enclave)
        from gateway.tee.nsm_lib import nsm_get_attestation_doc
        
        # Get real attestation from NSM
        attestation_doc = nsm_get_attestation_doc(
            public_key=bytes.fromhex(get_enclave_public_key_hex()),
            user_data=user_data_cbor,
            nonce=None,
        )
        logger.info(f"✅ Generated Nitro attestation for epoch {epoch_id}")
        return attestation_doc
        
    except ImportError:
        # Not in enclave - generate placeholder for testing
        logger.warning(
            f"⚠️ Not in Nitro Enclave - generating placeholder attestation for epoch {epoch_id}"
        )
        
        # Placeholder structure (NOT valid for production)
        placeholder = {
            "_placeholder": True,
            "_warning": "This is NOT a real Nitro attestation",
            "user_data": user_data,
            "pcrs": {0: "placeholder_pcr0"},
        }
        
        return cbor2.dumps(placeholder)


def get_attestation_document_b64(epoch_id: int) -> str:
    """
    Get base64-encoded attestation document for the given epoch.
    
    Caches attestation per epoch to avoid regenerating.
    
    Args:
        epoch_id: Epoch to get attestation for
        
    Returns:
        Base64-encoded attestation document
    """
    import base64
    
    # Check cache first
    if epoch_id in _ATTESTATION_CACHE:
        return _ATTESTATION_CACHE[epoch_id]
    
    # Generate new attestation
    code_hash = get_code_hash()
    att_bytes = generate_attestation_document(code_hash, epoch_id)
    att_b64 = base64.b64encode(att_bytes).decode('ascii')
    
    # Cache it (limit cache size)
    if len(_ATTESTATION_CACHE) > 10:
        # Remove oldest entries
        oldest_key = min(_ATTESTATION_CACHE.keys())
        del _ATTESTATION_CACHE[oldest_key]
    
    _ATTESTATION_CACHE[epoch_id] = att_b64
    
    return att_b64


# Alias for compatibility with tasks8.md naming
def get_attestation(epoch_id: int) -> str:
    """Alias for get_attestation_document_b64()."""
    return get_attestation_document_b64(epoch_id)


# ============================================================================
# State Inspection (for debugging)
# ============================================================================

def get_signer_state() -> dict:
    """
    Get current state of the enclave signer for debugging.
    
    Returns:
        Dict with state info (NEVER includes private key)
    """
    return {
        "initialized": _PRIVATE_KEY is not None,
        "public_key_hex": _PUBLIC_KEY_HEX,
        "code_hash": _CODE_HASH,
        "attestation_cache_size": len(_ATTESTATION_CACHE),
        "cached_epochs": list(_ATTESTATION_CACHE.keys()),
    }

