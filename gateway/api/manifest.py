"""
Epoch Manifest API

Endpoint for validators to submit epoch completion manifests.

Manifests prove that a validator completed their validation work for an epoch.
The manifest is a Merkle root of all evidence_ids the validator submitted.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json
from uuid import uuid4

from gateway.utils.signature import verify_wallet_signature
from gateway.utils.epoch import is_epoch_closed
from gateway.utils.merkle import compute_merkle_root
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BUILD_ID
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/manifest", tags=["Manifest"])


class ManifestPayload(BaseModel):
    """
    Payload for epoch manifest submission.
    
    Validators submit a manifest at the end of each epoch proving
    they completed their validation work.
    """
    epoch_id: int = Field(..., description="Epoch number")
    validation_count: int = Field(..., ge=0, description="Number of validations submitted")
    manifest_root: str = Field(..., description="Merkle root of all evidence_ids (hex string)")
    validator_hotkey: str = Field(..., description="Validator's SS58 address")


@router.post("/")
async def submit_epoch_manifest(
    payload: ManifestPayload,
    signature: str = Body(..., description="Ed25519 signature over JSON(payload)")
):
    """
    Submit epoch manifest proving validator work completion.
    
    Manifest is a Merkle root of all evidence_ids the validator submitted during the epoch.
    This proves the validator completed their assigned work.
    
    Flow:
    1. Verify signature over JSON(payload)
    2. Verify epoch is closed
    3. Query validator's evidence from validation_evidence_private
    4. Verify validation_count matches
    5. Compute Merkle root of evidence_ids
    6. Verify manifest_root matches computed root
    7. Log EPOCH_MANIFEST to transparency_log
    8. Return success
    
    Args:
        payload: ManifestPayload with epoch_id, validation_count, manifest_root
        signature: Ed25519 signature over JSON(payload)
    
    Returns:
        {
            "status": "manifest_recorded",
            "epoch_id": int,
            "validation_count": int,
            "manifest_root": str,
            "message": "Epoch manifest recorded successfully"
        }
    
    Raises:
        400: Bad request (epoch not closed, count mismatch, root mismatch)
        403: Forbidden (invalid signature)
        500: Server error
    
    Security:
        - Ed25519 signature verification
        - Epoch phase enforcement (closed epoch only)
        - Work verification (compute and compare Merkle root)
    
    Purpose:
        - Proves validator completed assigned work
        - Enables detection of non-participating validators
        - Provides verifiable work completion record
    """
    
    # ========================================
    # Step 1: Verify signature
    # ========================================
    message = json.dumps(payload.model_dump(), sort_keys=True)
    if not verify_wallet_signature(message, signature, payload.validator_hotkey):
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # ========================================
    # Step 2: Verify epoch is closed
    # ========================================
    if not is_epoch_closed(payload.epoch_id):
        raise HTTPException(
            status_code=400,
            detail=f"Epoch {payload.epoch_id} is still active. Wait for epoch to close to submit manifest."
        )
    
    # ========================================
    # Step 3: Query validator's evidence from Private DB
    # ========================================
    try:
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id") \
            .eq("validator_hotkey", payload.validator_hotkey) \
            .eq("epoch_id", payload.epoch_id) \
            .order("evidence_id") \
            .execute()
        
        evidence_ids = [row["evidence_id"] for row in result.data]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query validator evidence: {str(e)}"
        )
    
    # ========================================
    # Step 4: Verify validation_count matches
    # ========================================
    actual_count = len(evidence_ids)
    
    if actual_count != payload.validation_count:
        raise HTTPException(
            status_code=400,
            detail=f"Validation count mismatch: claimed {payload.validation_count}, found {actual_count}"
        )
    
    # ========================================
    # Step 5: Compute Merkle root of evidence_ids
    # ========================================
    if evidence_ids:
        computed_root = compute_merkle_root(evidence_ids)
    else:
        computed_root = "0" * 64  # Empty root for no validations
    
    # ========================================
    # Step 6: Verify manifest_root matches computed root
    # ========================================
    if computed_root != payload.manifest_root:
        print(f"❌ Manifest root mismatch for {payload.validator_hotkey}")
        print(f"   Claimed: {payload.manifest_root}")
        print(f"   Computed: {computed_root}")
        print(f"   Evidence IDs: {evidence_ids[:5]}..." if len(evidence_ids) > 5 else f"   Evidence IDs: {evidence_ids}")
        
        raise HTTPException(
            status_code=400,
            detail="Manifest root mismatch. Computed root does not match claimed root."
        )
    
    # ========================================
    # Step 7: Log EPOCH_MANIFEST to transparency_log
    # ========================================
    try:
        payload_hash = hashlib.sha256(message.encode()).hexdigest()
        
        log_entry = {
            "event_type": "EPOCH_MANIFEST",
            "actor_hotkey": payload.validator_hotkey,
            "nonce": str(uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "payload_hash": payload_hash,
            "build_id": BUILD_ID,
            "signature": signature,
            "payload": payload.model_dump()
        }
        
        supabase.table("transparency_log").insert(log_entry).execute()
        
        print(f"✅ EPOCH_MANIFEST logged")
        print(f"   Epoch: {payload.epoch_id}")
        print(f"   Validator: {payload.validator_hotkey[:20]}...")
        print(f"   Validations: {payload.validation_count}")
        print(f"   Manifest root: {payload.manifest_root[:16]}...")
    
    except Exception as e:
        print(f"❌ Error logging to transparency_log: {e}")
        # Continue anyway - manifest verification already succeeded
    
    # ========================================
    # Step 8: Return success
    # ========================================
    return {
        "status": "manifest_recorded",
        "epoch_id": payload.epoch_id,
        "validation_count": payload.validation_count,
        "manifest_root": computed_root,
        "message": "Epoch manifest recorded successfully"
    }


@router.get("/stats")
async def get_manifest_stats(epoch_id: int):
    """
    Get manifest submission statistics for an epoch.
    
    Public endpoint for monitoring manifest submission progress.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        {
            "epoch_id": int,
            "total_validators": int,
            "manifests_submitted": int,
            "submission_percentage": float,
            "missing_validators": List[str]
        }
    """
    try:
        # Query all manifests for this epoch from transparency_log
        manifest_result = supabase.table("transparency_log") \
            .select("actor_hotkey") \
            .eq("event_type", "EPOCH_MANIFEST") \
            .filter("payload->>epoch_id", "eq", str(epoch_id)) \
            .execute()
        
        submitted_validators = list(set([row["actor_hotkey"] for row in manifest_result.data]))
        manifests_submitted = len(submitted_validators)
        
        # Query all validators who submitted evidence for this epoch
        evidence_result = supabase.table("validation_evidence_private") \
            .select("validator_hotkey", count="exact") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        all_validators = list(set([row["validator_hotkey"] for row in evidence_result.data]))
        total_validators = len(all_validators)
        
        # Find missing validators
        missing_validators = [v for v in all_validators if v not in submitted_validators]
        
        # Calculate percentage
        submission_percentage = (manifests_submitted / total_validators * 100) if total_validators > 0 else 0
        
        return {
            "epoch_id": epoch_id,
            "total_validators": total_validators,
            "manifests_submitted": manifests_submitted,
            "submission_percentage": round(submission_percentage, 2),
            "missing_count": len(missing_validators),
            "missing_validators": missing_validators
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get manifest stats: {str(e)}"
        )


@router.get("/validator/{validator_hotkey}")
async def get_validator_manifests(
    validator_hotkey: str,
    limit: int = 10
):
    """
    Get recent manifests submitted by a specific validator.
    
    Public endpoint for checking validator participation history.
    
    Args:
        validator_hotkey: Validator's SS58 address
        limit: Number of recent manifests to return (default: 10)
    
    Returns:
        {
            "validator_hotkey": str,
            "manifests": List[{
                "epoch_id": int,
                "validation_count": int,
                "manifest_root": str,
                "timestamp": str
            }]
        }
    """
    try:
        result = supabase.table("transparency_log") \
            .select("payload, ts") \
            .eq("event_type", "EPOCH_MANIFEST") \
            .eq("actor_hotkey", validator_hotkey) \
            .order("ts", desc=True) \
            .limit(limit) \
            .execute()
        
        manifests = []
        for row in result.data:
            payload = row["payload"]
            manifests.append({
                "epoch_id": payload["epoch_id"],
                "validation_count": payload["validation_count"],
                "manifest_root": payload["manifest_root"],
                "timestamp": row["ts"]
            })
        
        return {
            "validator_hotkey": validator_hotkey,
            "manifest_count": len(manifests),
            "manifests": manifests
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validator manifests: {str(e)}"
        )

