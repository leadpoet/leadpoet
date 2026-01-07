"""
Validator TEE Enclave Signer
============================

This module provides the HOST-SIDE interface for validator TEE operations.
All operations are delegated to the Nitro Enclave via vsock.

SECURITY MODEL:
- Private key NEVER leaves the enclave (hardware-protected)
- Attestation binds public key to enclave code (PCR0)
- epoch_id in attestation prevents replay attacks
- NO MOCK MODE: Enclave MUST be running, otherwise operations FAIL

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

FAIL-CLOSED: If enclave is not running, all operations raise RuntimeError.
This ensures the validator cannot accidentally run without TEE protection.
"""

import os
import json
import hashlib
import logging
import threading
from typing import Tuple, Optional, List, Dict, Any

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
# Enclave Connection (REQUIRED - No Mock Mode)
# ============================================================================

_ENCLAVE_CLIENT = None
_ENCLAVE_CONNECTED: bool = False


def _ensure_enclave_connected():
    """
    Ensure connection to Nitro Enclave via vsock.
    
    FAIL-CLOSED: Raises RuntimeError if enclave is not available.
    There is NO mock mode - the enclave MUST be running.
    """
    global _ENCLAVE_CONNECTED, _ENCLAVE_CLIENT
    
    if _ENCLAVE_CONNECTED and _ENCLAVE_CLIENT is not None:
        return _ENCLAVE_CLIENT
    
    try:
        from validator_tee.host.vsock_client import ValidatorEnclaveClient, is_enclave_available
        
        if not is_enclave_available():
            raise RuntimeError(
                "FAIL-CLOSED: Nitro Enclave is NOT running. "
                "Start the enclave with: bash validator_tee/start_enclave.sh"
            )
        
        _ENCLAVE_CLIENT = ValidatorEnclaveClient()
        # Test connection
        _ENCLAVE_CLIENT.health_check()
        _ENCLAVE_CONNECTED = True
        logger.info("✅ Connected to Nitro Enclave via vsock")
        return _ENCLAVE_CLIENT
        
    except ImportError as e:
        raise RuntimeError(f"FAIL-CLOSED: vsock_client not available: {e}")
    except Exception as e:
        raise RuntimeError(f"FAIL-CLOSED: Cannot connect to enclave: {e}")


def is_enclave_running() -> bool:
    """Check if Nitro Enclave is running and connected."""
    try:
        _ensure_enclave_connected()
        return True
    except RuntimeError:
        return False

# ============================================================================
# Module-Level State
# ============================================================================

# Cache for attestations (keyed by epoch_id)
_ATTESTATION_CACHE: dict = {}  # epoch_id -> attestation_b64

# Cache for public key (avoid repeated vsock calls)
_PUBLIC_KEY_HEX: Optional[str] = None

# Cache for code hash (avoid recomputing)
_CODE_HASH: Optional[str] = None


# ============================================================================
# Initialization
# ============================================================================

def initialize_enclave_keypair() -> str:
    """
    Connect to enclave and get public key.
    
    FAIL-CLOSED: Raises RuntimeError if enclave is not running.
    
    Returns:
        Public key as hex string
    """
    global _PUBLIC_KEY_HEX
    
    client = _ensure_enclave_connected()
    _PUBLIC_KEY_HEX = client.get_public_key()
    logger.info(f"✅ Connected to enclave: {_PUBLIC_KEY_HEX[:16]}...")
    return _PUBLIC_KEY_HEX


def is_keypair_initialized() -> bool:
    """Check if connected to enclave (enclave always has its keypair)."""
    return is_enclave_running()


def get_enclave_public_key_hex() -> str:
    """
    Get the enclave's public key as hex string.
    
    FAIL-CLOSED: Raises RuntimeError if enclave is not running.
    
    Returns:
        Public key hex (64 characters for Ed25519)
    """
    client = _ensure_enclave_connected()
    return client.get_public_key()


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
    
    FAIL-CLOSED: Enclave MUST be running, otherwise raises RuntimeError.
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch identifier
        block: Block number when weights were computed
        uids: List of UIDs (must be sorted ascending)
        weights_u16: Corresponding u16 weights
        
    Returns:
        Tuple of (weights_hash_hex, signature_hex)
        
    Raises:
        RuntimeError: If enclave not running
        ValueError: If inputs are invalid
    """
    # Validate inputs first (before sending to enclave)
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
    
    # Sign via enclave (FAIL-CLOSED - no mock mode)
    client = _ensure_enclave_connected()
    signature_hex = client.sign_weights(weights_hash)
    
    logger.info(
        f"✅ Signed weights: epoch={epoch_id}, {len(uids)} UIDs, "
        f"hash={weights_hash[:16]}..."
    )
    
    return weights_hash, signature_hex


# sign_digest removed - all signing must go through enclave


# ============================================================================
# Attestation Document (via Enclave)
# ============================================================================

# generate_attestation_document removed - attestation now comes from enclave
# The enclave's tee_service.py handles attestation generation via NSM


def get_attestation_document_b64(epoch_id: int) -> str:
    """
    Get base64-encoded attestation document for the given epoch.
    
    FAIL-CLOSED: Enclave MUST be running, otherwise raises RuntimeError.
    Caches attestation per epoch to avoid regenerating.
    
    Args:
        epoch_id: Epoch to get attestation for
        
    Returns:
        Base64-encoded attestation document
    """
    # Check cache first
    if epoch_id in _ATTESTATION_CACHE:
        return _ATTESTATION_CACHE[epoch_id]
    
    # Get from enclave (FAIL-CLOSED - no mock mode)
    client = _ensure_enclave_connected()
    attestation_data = client.get_attestation(epoch_id)
    attestation_b64 = attestation_data["attestation_b64"]
    
    # Cache it
    _ATTESTATION_CACHE[epoch_id] = attestation_b64
    
    is_mock = attestation_data.get("is_mock", False)
    if is_mock:
        logger.warning(f"⚠️ Enclave returned MOCK attestation for epoch {epoch_id} (NSM not available)")
    else:
        logger.info(f"✅ Got Nitro attestation for epoch {epoch_id}")
    
    return attestation_b64


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
        Dict with state info
    """
    try:
        client = _ensure_enclave_connected()
        pubkey = client.get_public_key()
        code_hash = client.get_code_hash()
        connected = True
    except RuntimeError:
        pubkey = None
        code_hash = None
        connected = False
    
    return {
        "enclave_connected": connected,
        "public_key_hex": pubkey,
        "code_hash": code_hash,
        "attestation_cache_size": len(_ATTESTATION_CACHE),
        "cached_epochs": list(_ATTESTATION_CACHE.keys()),
    }

