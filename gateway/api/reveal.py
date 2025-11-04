"""
Reveal Phase API (Post-Epoch Reveal)

Endpoint for validators to reveal validation decisions after epoch closes.

This implements the REVEAL phase of commit-reveal:
- Validators submitted hashes during epoch (commit phase)
- After epoch closes, validators reveal actual values
- Gateway verifies hashes match revealed values
- Only decision + rep_score revealed (evidence_blob stays private)
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import hashlib

from gateway.utils.signature import verify_wallet_signature
from gateway.utils.epoch import is_epoch_closed
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/reveal", tags=["Reveal"])


class RevealPayload(BaseModel):
    """
    Payload for revealing validation decision.
    
    Validators reveal the actual values that were hashed during commit phase.
    """
    evidence_id: str = Field(..., description="Evidence ID from commit phase")
    epoch_id: int = Field(..., description="Epoch number")
    decision: Literal["approve", "deny"] = Field(..., description="Validation decision: 'approve' or 'deny'")
    rep_score: float = Field(..., ge=0.0, le=1.0, description="Reputation score (0.0 to 1.0)")
    rejection_reason: str = Field(..., description="Rejection reason: 'pass' if approved, or specific failure reason if denied")
    salt: str = Field(..., description="Hex-encoded salt used in commitment")


@router.post("/")
async def reveal_validation_result(
    payload: RevealPayload,
    validator_hotkey: str = Body(..., description="Validator's SS58 address"),
    signature: str = Body(..., description="Ed25519 signature over JSON(payload)")
):
    """
    Reveal validation decision + rep_score + rejection_reason after epoch closes.
    
    Validators committed to their validation by submitting hashes during the epoch.
    After the epoch closes, they reveal the actual values.
    
    The gateway verifies:
    - decision_hash matches H(decision + salt)
    - rep_score_hash matches H(rep_score + salt)
    - rejection_reason_hash matches H(rejection_reason + salt)
    
    Evidence blob is NEVER revealed publicly - it stays in Private DB.
    
    Flow:
    1. Verify signature over JSON(payload)
    2. Verify epoch is closed
    3. Fetch evidence from validation_evidence_private
    4. Verify validator owns this evidence
    5. Verify decision_hash matches H(decision + salt)
    6. Verify rep_score_hash matches H(rep_score + salt)
    7. Verify rejection_reason_hash matches H(rejection_reason + salt)
    8. Validate rejection_reason logic (must be "pass" if approved)
    9. Update Private DB with revealed values
    10. Set revealed_ts timestamp
    11. Return success
    
    Args:
        payload: RevealPayload with decision, rep_score, rejection_reason, salt
        validator_hotkey: Validator's SS58 address
        signature: Ed25519 signature over JSON(payload)
    
    Returns:
        {
            "status": "revealed",
            "evidence_id": "uuid",
            "epoch_id": int,
            "decision": "approve" or "deny",
            "rep_score": float,
            "rejection_reason": str
        }
    
    Raises:
        400: Bad request (epoch not closed, hash mismatch, invalid rejection_reason)
        403: Forbidden (invalid signature)
        404: Evidence not found or not owned by validator
        500: Server error
    
    Security:
        - Ed25519 signature verification
        - Epoch boundary enforcement (must be closed)
        - Ownership verification (only validator who committed can reveal)
        - Hash verification (prevents cheating)
        - Logic validation (rejection_reason must be "pass" if approved)
    
    Privacy:
        - evidence_blob is NEVER revealed
        - Only decision, rep_score, and rejection_reason are revealed
        - Evidence stays in Private DB with RLS
    """
    
    # ========================================
    # Step 1: Verify signature
    # ========================================
    import json
    
    message = json.dumps(payload.model_dump(), sort_keys=True)
    if not verify_wallet_signature(message, signature, validator_hotkey):
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
            detail=f"Epoch {payload.epoch_id} is not closed yet. Wait until epoch closes to reveal."
        )
    
    # ========================================
    # Step 3: Fetch evidence from Private DB
    # ========================================
    try:
        result = supabase.table("validation_evidence_private") \
            .select("*") \
            .eq("evidence_id", payload.evidence_id) \
            .eq("validator_hotkey", validator_hotkey) \
            .limit(1) \
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Evidence {payload.evidence_id} not found or not owned by validator"
            )
        
        evidence = result.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch evidence: {str(e)}"
        )
    
    # ========================================
    # Step 4: Verify evidence hasn't been revealed yet
    # ========================================
    if evidence["decision"] is not None or evidence["rep_score"] is not None:
        # Already revealed
        print(f"⚠️  Evidence {payload.evidence_id} already revealed")
        # Return success anyway (idempotent)
        return {
            "status": "already_revealed",
            "evidence_id": payload.evidence_id,
            "epoch_id": payload.epoch_id,
            "decision": evidence["decision"],
            "rep_score": evidence["rep_score"],
            "rejection_reason": evidence.get("rejection_reason", "unknown"),
            "revealed_ts": evidence["revealed_ts"]
        }
    
    # ========================================
    # Step 5: Verify decision_hash
    # ========================================
    computed_decision_hash = hashlib.sha256((payload.decision + payload.salt).encode()).hexdigest()
    
    if computed_decision_hash != evidence["decision_hash"]:
        print(f"❌ Decision hash mismatch for {payload.evidence_id}")
        print(f"   Expected: {evidence['decision_hash']}")
        print(f"   Computed: {computed_decision_hash}")
        raise HTTPException(
            status_code=400,
            detail="Decision hash mismatch - invalid reveal"
        )
    
    # ========================================
    # Step 6: Verify rep_score_hash
    # ========================================
    computed_rep_score_hash = hashlib.sha256((str(payload.rep_score) + payload.salt).encode()).hexdigest()
    
    if computed_rep_score_hash != evidence["rep_score_hash"]:
        print(f"❌ Rep score hash mismatch for {payload.evidence_id}")
        print(f"   Expected: {evidence['rep_score_hash']}")
        print(f"   Computed: {computed_rep_score_hash}")
        raise HTTPException(
            status_code=400,
            detail="Rep score hash mismatch - invalid reveal"
        )
    
    # ========================================
    # Step 7: Verify rejection_reason_hash
    # ========================================
    computed_rejection_reason_hash = hashlib.sha256((payload.rejection_reason + payload.salt).encode()).hexdigest()
    
    if computed_rejection_reason_hash != evidence["rejection_reason_hash"]:
        print(f"❌ Rejection reason hash mismatch for {payload.evidence_id}")
        print(f"   Expected: {evidence['rejection_reason_hash']}")
        print(f"   Computed: {computed_rejection_reason_hash}")
        raise HTTPException(
            status_code=400,
            detail="Rejection reason hash mismatch - invalid reveal"
        )
    
    # ========================================
    # Step 8: Validate rejection_reason logic
    # ========================================
    if payload.decision == "approve" and payload.rejection_reason != "pass":
        raise HTTPException(
            status_code=400,
            detail=f"If decision is 'approve', rejection_reason must be 'pass' (got: '{payload.rejection_reason}')"
        )
    
    # ========================================
    # Step 9: Update Private DB with revealed values
    # ========================================
    try:
        supabase.table("validation_evidence_private") \
            .update({
                "decision": payload.decision,
                "rep_score": payload.rep_score,
                "rejection_reason": payload.rejection_reason,
                "revealed_ts": datetime.utcnow().isoformat()
            }) \
            .eq("evidence_id", payload.evidence_id) \
            .execute()
        
        print(f"✅ Evidence revealed: {payload.evidence_id}")
        print(f"   Epoch: {payload.epoch_id}")
        print(f"   Decision: {payload.decision}")
        print(f"   Rep score: {payload.rep_score}")
        print(f"   Rejection reason: {payload.rejection_reason}")
        print(f"   Validator: {validator_hotkey[:20]}...")
    
    except Exception as e:
        print(f"❌ Error updating evidence: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update evidence with revealed values"
        )
    
    # ========================================
    # Step 10: Return success
    # ========================================
    return {
        "status": "revealed",
        "evidence_id": payload.evidence_id,
        "epoch_id": payload.epoch_id,
        "decision": payload.decision,
        "rep_score": payload.rep_score,
        "rejection_reason": payload.rejection_reason,
        "message": "Validation revealed successfully"
    }


@router.get("/stats")
async def get_reveal_stats(epoch_id: int):
    """
    Get reveal statistics for an epoch.
    
    Public endpoint for monitoring reveal progress.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        {
            "epoch_id": int,
            "total_commits": int,
            "total_reveals": int,
            "reveal_percentage": float,
            "unrevealed_validators": List[str]
        }
    """
    try:
        # Query all evidence for this epoch
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id, validator_hotkey, decision, revealed_ts") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        evidences = result.data
        total_commits = len(evidences)
        
        # Count reveals
        revealed = [e for e in evidences if e["decision"] is not None]
        total_reveals = len(revealed)
        
        # Calculate percentage
        reveal_percentage = (total_reveals / total_commits * 100) if total_commits > 0 else 0
        
        # Get unrevealed validators
        unrevealed = [e for e in evidences if e["decision"] is None]
        unrevealed_validators = list(set([e["validator_hotkey"] for e in unrevealed]))
        
        return {
            "epoch_id": epoch_id,
            "total_commits": total_commits,
            "total_reveals": total_reveals,
            "reveal_percentage": round(reveal_percentage, 2),
            "unrevealed_count": len(unrevealed),
            "unrevealed_validators": unrevealed_validators
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get reveal stats: {str(e)}"
        )


@router.get("/evidence/{evidence_id}")
async def get_evidence_status(evidence_id: str):
    """
    Get status of a specific evidence (revealed or not).
    
    Public endpoint for checking if evidence has been revealed.
    
    Note: This only returns status, NOT the actual evidence_blob or values.
    
    Args:
        evidence_id: Evidence UUID
    
    Returns:
        {
            "evidence_id": str,
            "epoch_id": int,
            "revealed": bool,
            "revealed_ts": str or null
        }
    """
    try:
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id, epoch_id, decision, revealed_ts") \
            .eq("evidence_id", evidence_id) \
            .limit(1) \
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Evidence {evidence_id} not found"
            )
        
        evidence = result.data[0]
        
        return {
            "evidence_id": evidence["evidence_id"],
            "epoch_id": evidence["epoch_id"],
            "revealed": evidence["decision"] is not None,
            "revealed_ts": evidence["revealed_ts"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evidence status: {str(e)}"
        )

