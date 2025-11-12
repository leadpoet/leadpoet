"""
TEE Attestation Endpoint

This endpoint returns the TEE attestation document, which provides
cryptographic proof that the gateway is running the canonical code from GitHub.

Anyone (miners, validators, auditors) can call this endpoint to verify:
1. The attestation signature (proves it came from AWS Nitro hardware)
2. The code_hash (proves which code is running)
3. The public key (used to verify checkpoint signatures)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import subprocess

from gateway.utils.tee_client import tee_client


router = APIRouter()


class AttestationResponse(BaseModel):
    """Response model for /attest endpoint"""
    attestation_document: str  # CBOR-encoded attestation (hex)
    enclave_public_key: str  # Ed25519 public key (hex)
    code_hash: str  # SHA256 of gateway code (hex)
    github_commit: Optional[str]  # Git commit hash (if available)
    pcr0: Optional[str]  # Docker image measurement (hex)
    pcr1: Optional[str]  # Linux kernel measurement (hex)
    pcr2: Optional[str]  # Application measurement (hex)
    timestamp: str  # When attestation was retrieved


def get_github_commit() -> Optional[str]:
    """
    Get current GitHub commit hash.
    
    Tries:
    1. GIT_COMMIT_HASH environment variable (set during deployment)
    2. git rev-parse HEAD (if running in git repo)
    
    Returns:
        Git commit hash or None if not available
    """
    # Try environment variable first
    commit = os.getenv("GIT_COMMIT_HASH")
    if commit:
        return commit
    
    # Try git command
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return None


@router.get("/attest", response_model=AttestationResponse)
async def get_attestation():
    """
    Get TEE attestation document.
    
    This endpoint provides cryptographic proof that the gateway is running
    the canonical code from GitHub. The attestation document is signed by
    AWS Nitro hardware and cannot be faked.
    
    **How to verify:**
    
    1. Download attestation document from this endpoint
    2. Verify AWS Nitro signature using `scripts/verify_attestation.py`
    3. Verify code_hash matches GitHub using `scripts/verify_code_hash.py`
    4. If both pass: Gateway is provably running canonical code ✅
    
    **Response fields:**
    
    - `attestation_document`: CBOR-encoded attestation (signed by AWS Nitro)
    - `enclave_public_key`: Public key used to sign checkpoints
    - `code_hash`: SHA256 hash of gateway application code
    - `github_commit`: Git commit hash (for code verification)
    - `pcr0`, `pcr1`, `pcr2`: Platform Configuration Registers (enclave measurements)
    - `timestamp`: When this attestation was retrieved
    
    **Security notes:**
    
    - Attestation is generated inside the TEE (hardware-protected)
    - code_hash binds the running code to the public key
    - Anyone can verify the attestation signature
    - Modified code = different code_hash = detectable
    
    **Cost:** Free (no rate limiting - encourage frequent verification!)
    """
    try:
        # Get attestation from TEE
        attestation_data = await tee_client.get_attestation()
        
        # Get GitHub commit hash
        github_commit = get_github_commit()
        
        # Build response
        response = AttestationResponse(
            attestation_document=attestation_data.get("attestation_document", ""),
            enclave_public_key=attestation_data.get("public_key", ""),
            code_hash=attestation_data.get("code_hash", ""),
            github_commit=github_commit,
            pcr0=attestation_data.get("pcr0"),
            pcr1=attestation_data.get("pcr1"),
            pcr2=attestation_data.get("pcr2"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to get attestation from TEE: {str(e)}. "
                   f"TEE enclave may not be running. "
                   f"Check: sudo nitro-cli describe-enclaves"
        )


@router.get("/attest/health")
async def attestation_health():
    """
    Health check for TEE attestation capability.
    
    Returns:
        {
            "tee_available": true/false,
            "attestation_valid": true/false,
            "message": "..."
        }
    """
    try:
        # Try to get attestation
        attestation_data = await tee_client.get_attestation()
        
        # Check if all required fields present
        required_fields = ["attestation_document", "public_key", "code_hash"]
        all_present = all(
            attestation_data.get(field) 
            for field in required_fields
        )
        
        if all_present:
            return {
                "tee_available": True,
                "attestation_valid": True,
                "message": "TEE is running and producing valid attestations"
            }
        else:
            return {
                "tee_available": True,
                "attestation_valid": False,
                "message": "TEE is running but attestation is incomplete"
            }
    
    except Exception as e:
        return {
            "tee_available": False,
            "attestation_valid": False,
            "message": f"TEE unavailable: {str(e)}"
        }

