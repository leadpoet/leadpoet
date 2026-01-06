"""
TEE Attestation Endpoints (Canonical)
=====================================

These endpoints provide the gateway's TEE attestation document and public key
for auditors and validators to verify the gateway's identity and code integrity.

ENDPOINTS:
- GET /attestation/document - Full attestation document + pubkey + code_hash
- GET /attestation/pubkey - Enclave public key only

NO AUTHENTICATION REQUIRED:
Anyone can request the attestation - this is by design for transparency.
Auditors need to fetch the attestation to verify event signatures.

CANONICAL RESPONSE FORMAT:
{
    "attestation_document": "base64-encoded COSE_Sign1",  # From AWS Nitro
    "enclave_pubkey": "hex-encoded Ed25519 public key",
    "code_hash": "hex-encoded SHA256 of gateway code"
}

SECURITY NOTES:
- The attestation document is signed by AWS Nitro hardware
- The enclave pubkey is bound to the code hash via user_data in attestation
- Auditors MUST verify: cert chain → PCR0 matches allowlist → pubkey from user_data
- The returned code_hash is informational; trust comes from PCR0 in attestation
"""

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gateway.tee.enclave_signer import (
    get_enclave_public_key_hex,
    get_attestation_document,
    get_attestation_document_b64,
    get_cached_code_hash,
    is_keypair_initialized,
)


router = APIRouter(prefix="/attestation", tags=["attestation"])


class AttestationDocumentResponse(BaseModel):
    """
    Response model for /attestation/document endpoint.
    
    Fields:
        attestation_document: Base64-encoded COSE_Sign1 attestation from AWS Nitro.
                              Empty string if not running in Nitro Enclave (dev mode).
        enclave_pubkey: Hex-encoded Ed25519 public key (64 hex chars = 32 bytes).
        code_hash: Hex-encoded SHA256 hash of gateway code (64 hex chars).
        trust_level: "full_nitro" if attestation is real, "signature_only" if not.
        
    IMPORTANT:
        - In production, auditors MUST verify attestation_document against AWS Nitro root
        - The enclave_pubkey is bound to code_hash via user_data in the attestation
        - If trust_level is "signature_only", Nitro verification is not available
    """
    attestation_document: str  # Base64-encoded COSE_Sign1 (or empty if dev mode)
    enclave_pubkey: str  # Hex-encoded Ed25519 public key
    code_hash: str  # Hex-encoded SHA256 hash of gateway code
    trust_level: str  # "full_nitro" or "signature_only"


class PubkeyResponse(BaseModel):
    """
    Response model for /attestation/pubkey endpoint.
    
    Returns only the enclave public key - useful for lightweight clients
    that already have verified attestation and just need the pubkey.
    """
    enclave_pubkey: str  # Hex-encoded Ed25519 public key


@router.get("/document", response_model=AttestationDocumentResponse)
async def get_attestation_document_endpoint():
    """
    Get the gateway's TEE attestation document.
    
    This endpoint returns:
    - attestation_document: COSE_Sign1 document from AWS Nitro (base64)
    - enclave_pubkey: The gateway's Ed25519 public key used for event signing
    - code_hash: SHA256 hash of the gateway code
    - trust_level: Whether full Nitro verification is available
    
    **No authentication required** - this is a public endpoint for transparency.
    
    **Auditor verification flow:**
    1. Fetch this endpoint
    2. Decode attestation_document from base64
    3. Verify COSE_Sign1 signature against AWS Nitro root certificate
    4. Extract PCR0 from attestation and verify against pinned allowlist
    5. Extract user_data from attestation and verify:
       - purpose == "gateway_event_signing"
       - enclave_pubkey matches response
       - code_hash matches response
    6. If all pass: use enclave_pubkey to verify event signatures
    
    **Trust levels:**
    - "full_nitro": Attestation document is from real AWS Nitro hardware
    - "signature_only": No attestation (dev mode) - can only verify signatures
    
    **In signature-only mode:**
    - Event signatures are still valid and verifiable
    - But there's no hardware proof that the gateway code is authentic
    - Auditors should report this limitation in their output
    """
    try:
        # Check if TEE is initialized
        if not is_keypair_initialized():
            raise HTTPException(
                status_code=503,
                detail="TEE enclave not initialized. Gateway may be starting up."
            )
        
        # Get attestation document (base64)
        try:
            attestation_b64 = get_attestation_document_b64()
            trust_level = "full_nitro" if attestation_b64 else "signature_only"
        except Exception:
            # No attestation available (dev mode)
            attestation_b64 = ""
            trust_level = "signature_only"
        
        # Get enclave public key
        enclave_pubkey = get_enclave_public_key_hex()
        
        # Get code hash (cached at boot)
        try:
            code_hash = get_cached_code_hash()
        except RuntimeError:
            # Code hash not cached - compute it
            from gateway.tee.gateway_tee_service import compute_code_hash
            code_hash = compute_code_hash()
        
        return AttestationDocumentResponse(
            attestation_document=attestation_b64,
            enclave_pubkey=enclave_pubkey,
            code_hash=code_hash,
            trust_level=trust_level,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get attestation: {str(e)}"
        )


@router.get("/pubkey", response_model=PubkeyResponse)
async def get_pubkey_endpoint():
    """
    Get the gateway's enclave public key.
    
    This is a lightweight endpoint that returns only the public key.
    Useful for clients that have already verified the attestation and
    just need the pubkey for signature verification.
    
    **No authentication required.**
    
    Returns:
        enclave_pubkey: Hex-encoded Ed25519 public key (64 hex chars)
    """
    try:
        if not is_keypair_initialized():
            raise HTTPException(
                status_code=503,
                detail="TEE enclave not initialized. Gateway may be starting up."
            )
        
        return PubkeyResponse(
            enclave_pubkey=get_enclave_public_key_hex()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get public key: {str(e)}"
        )


@router.get("/health")
async def attestation_health():
    """
    Health check for TEE attestation capability.
    
    Returns the current state of the TEE subsystem.
    
    **Fields:**
    - tee_initialized: Whether the enclave keypair is generated
    - attestation_available: Whether a real Nitro attestation exists
    - trust_level: "full_nitro" or "signature_only"
    - pubkey_prefix: First 16 chars of public key (for identification)
    """
    try:
        initialized = is_keypair_initialized()
        
        if not initialized:
            return {
                "tee_initialized": False,
                "attestation_available": False,
                "trust_level": "unavailable",
                "pubkey_prefix": None,
                "message": "TEE not initialized"
            }
        
        # Get pubkey prefix for identification
        pubkey = get_enclave_public_key_hex()
        pubkey_prefix = pubkey[:16] if pubkey else None
        
        # Check attestation availability
        try:
            attestation_b64 = get_attestation_document_b64()
            attestation_available = bool(attestation_b64)
        except Exception:
            attestation_available = False
        
        trust_level = "full_nitro" if attestation_available else "signature_only"
        
        return {
            "tee_initialized": True,
            "attestation_available": attestation_available,
            "trust_level": trust_level,
            "pubkey_prefix": pubkey_prefix,
            "message": f"TEE operational ({trust_level})"
        }
        
    except Exception as e:
        return {
            "tee_initialized": False,
            "attestation_available": False,
            "trust_level": "error",
            "pubkey_prefix": None,
            "message": f"Error: {str(e)}"
        }

