"""
Validation Submission API (Commit Phase)

Endpoint for validators to submit validation results during epoch.

This implements the COMMIT phase of commit-reveal:
- Validators submit hashes (decision_hash, rep_score_hash, evidence_hash)
- Evidence blob stored privately (NOT publicly logged)
- Actual values (decision, rep_score) revealed after epoch closes
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict
from datetime import datetime
from uuid import uuid4

from gateway.models.events import ValidationResultEvent
from gateway.utils.signature import verify_wallet_signature, compute_payload_hash, construct_signed_message
from gateway.utils.registry import is_registered_hotkey, get_metagraph
from gateway.utils.nonce import check_and_store_nonce, validate_nonce_format
from gateway.utils.epoch import is_epoch_active, get_current_epoch_id
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BUILD_ID, TIMESTAMP_TOLERANCE_SECONDS
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/validate", tags=["Validation"])


@router.post("/")
async def submit_validation_result(
    event: ValidationResultEvent,
    evidence_blob: Dict = Body(..., description="Full evidence data (stored privately, NOT logged)")
):
    """
    Submit validation result (commit phase).
    
    Validators commit to their validation decision by submitting:
    - decision_hash = H(salt || "approve" or "reject")
    - rep_score_hash = H(salt || rep_score_value)
    - evidence_hash = H(evidence_blob)
    
    The evidence_blob is stored in Private DB (access-controlled).
    Only the hashes are logged to transparency log.
    
    After epoch closes, validators reveal actual values via POST /reveal.
    
    Flow:
    1. Verify payload hash
    2. Verify wallet signature
    3. Verify validator is registered
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    6. Verify epoch is active
    7. Fetch validator V-score from metagraph
    8. Store evidence in validation_evidence_private (PII, access-controlled)
    9. Log VALIDATION_RESULT to transparency_log (hashes only)
    10. Return {status, evidence_id, epoch_id, merkle_proof}
    
    Args:
        event: ValidationResultEvent with decision_hash, rep_score_hash, evidence_hash
        evidence_blob: Full evidence data (stored privately)
    
    Returns:
        {
            "status": "recorded",
            "evidence_id": "uuid",
            "lead_id": "uuid",
            "epoch_id": int,
            "merkle_proof": ["hash1", "hash2", ...]
        }
    
    Raises:
        400: Bad request (payload hash, nonce, timestamp, epoch)
        403: Forbidden (invalid signature, not registered, not validator)
        404: Lead not found
        500: Server error
    
    Security:
        - Ed25519 signature verification
        - Nonce replay protection
        - Timestamp validation (±2 minutes)
        - Registry check (on-chain metagraph)
        - Epoch boundary enforcement
        - Evidence stored with RLS (only service_role can read)
    
    Privacy:
        - evidence_blob NEVER logged publicly
        - Only hashes appear in transparency_log
        - Evidence revealed after epoch closes
    """
    
    # ========================================
    # Step 1: Verify payload hash
    # ========================================
    computed_hash = compute_payload_hash(event.payload.model_dump())
    if computed_hash != event.payload_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Payload hash mismatch: expected {event.payload_hash[:16]}..., got {computed_hash[:16]}..."
        )
    
    # ========================================
    # Step 2: Verify wallet signature
    # ========================================
    message = construct_signed_message(event)
    if not verify_wallet_signature(message, event.signature, event.actor_hotkey):
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # ========================================
    # Step 3: Verify actor is registered validator
    # ========================================
    is_registered, role = is_registered_hotkey(event.actor_hotkey)
    
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    if role != "validator":
        raise HTTPException(
            status_code=403,
            detail="Only validators can submit validation results"
        )
    
    # ========================================
    # Step 4: Verify nonce format and freshness
    # ========================================
    if not validate_nonce_format(event.nonce):
        raise HTTPException(
            status_code=400,
            detail="Invalid nonce format (must be UUID v4)"
        )
    
    if not check_and_store_nonce(event.nonce, event.actor_hotkey):
        raise HTTPException(
            status_code=400,
            detail="Nonce already used (replay attack detected)"
        )
    
    # ========================================
    # Step 5: Verify timestamp
    # ========================================
    now = datetime.utcnow()
    time_diff = abs((now - event.ts).total_seconds())
    
    if time_diff > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp out of range: {time_diff:.0f}s (max: {TIMESTAMP_TOLERANCE_SECONDS}s)"
        )
    
    # ========================================
    # Step 6: Determine epoch_id and verify it's active
    # ========================================
    current_epoch = get_current_epoch_id()
    
    if not is_epoch_active(current_epoch):
        raise HTTPException(
            status_code=400,
            detail=f"Epoch {current_epoch} is not active for validation"
        )
    
    epoch_id = current_epoch
    
    # ========================================
    # Step 7: Verify lead exists in leads_private
    # ========================================
    try:
        lead_result = supabase.table("leads_private") \
            .select("lead_id, created_ts") \
            .eq("lead_id", event.payload.lead_id) \
            .limit(1) \
            .execute()
        
        if not lead_result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Lead {event.payload.lead_id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify lead existence: {str(e)}"
        )
    
    # ========================================
    # Step 8: Fetch validator V-score from metagraph
    # ========================================
    try:
        metagraph = get_metagraph()
        
        # Find validator's UID
        validator_uid = None
        for i, hotkey in enumerate(metagraph.hotkeys):
            if hotkey == event.actor_hotkey:
                validator_uid = i
                break
        
        if validator_uid is None:
            # Should not happen (already checked registry)
            v_score = 1.0
        else:
            # Get stake as V-score proxy
            # In production, fetch from Bittensor tStats API
            stake = float(metagraph.S[validator_uid])
            
            # Normalize stake to V-score (simplified)
            # V-score should be in range [0, 1] with validators having higher scores
            # For now, use stake directly (will be normalized in consensus)
            v_score = max(0.1, min(1.0, stake / 100.0))  # Simple normalization
    
    except Exception as e:
        print(f"⚠️  Failed to fetch V-score for {event.actor_hotkey}: {e}")
        v_score = 1.0  # Fallback to neutral score
    
    # ========================================
    # Step 9: Store evidence in Private DB
    # ========================================
    evidence_id = str(uuid4())
    
    evidence_entry = {
        "evidence_id": evidence_id,
        "lead_id": event.payload.lead_id,
        "validator_hotkey": event.actor_hotkey,
        "epoch_id": epoch_id,
        "decision_hash": event.payload.decision_hash,
        "rep_score_hash": event.payload.rep_score_hash,
        "evidence_hash": event.payload.evidence_hash,
        "decision": None,  # Will be revealed after epoch close
        "rep_score": None,  # Will be revealed after epoch close
        "evidence_blob": evidence_blob,  # Stored privately, NEVER publicly logged
        "v_score": v_score,
        "created_ts": event.ts.isoformat(),
        "revealed_ts": None
    }
    
    try:
        supabase.table("validation_evidence_private").insert(evidence_entry).execute()
        
        print(f"✅ Evidence stored: {evidence_id} for lead {event.payload.lead_id}")
        print(f"   Validator: {event.actor_hotkey[:20]}...")
        print(f"   Epoch: {epoch_id}")
        print(f"   V-score: {v_score}")
    
    except Exception as e:
        print(f"❌ Error storing evidence: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to store validation evidence"
        )
    
    # ========================================
    # Step 10: Log VALIDATION_RESULT to transparency log
    # ========================================
    # CRITICAL: Only log hashes, NOT evidence_blob
    try:
        log_entry = {
            "event_type": event.event_type.value,  # Convert enum to string
            "actor_hotkey": event.actor_hotkey,
            "nonce": event.nonce,
            "ts": event.ts.isoformat(),
            "payload_hash": event.payload_hash,
            "build_id": event.build_id,
            "signature": event.signature,
            "payload": {
                "lead_id": event.payload.lead_id,
                "decision_hash": event.payload.decision_hash,
                "rep_score_hash": event.payload.rep_score_hash,
                "evidence_hash": event.payload.evidence_hash
                # NOTE: evidence_blob is NOT included here
            }
        }
        
        log_result = supabase.table("transparency_log").insert(log_entry).execute()
        
        print(f"✅ VALIDATION_RESULT logged to transparency_log")
        print(f"   ⚠️  Only hashes logged (evidence_blob kept private)")
    
    except Exception as e:
        print(f"❌ Error logging to transparency_log: {e}")
        # Continue anyway - evidence is already stored
    
    # ========================================
    # Step 11: Generate Merkle proof (placeholder)
    # ========================================
    # In production, compute proof after batch of validations
    # For now, return placeholder
    merkle_proof = []  # Will be computed in batch later
    
    # ========================================
    # Step 12: Return response
    # ========================================
    return {
        "status": "recorded",
        "evidence_id": evidence_id,
        "lead_id": event.payload.lead_id,
        "epoch_id": epoch_id,
        "v_score": v_score,
        "merkle_proof": merkle_proof,
        "message": "Validation committed. Reveal after epoch closes."
    }


@router.get("/stats")
async def get_validation_stats(epoch_id: int = None):
    """
    Get validation statistics for an epoch.
    
    Public endpoint for monitoring validation activity.
    
    Args:
        epoch_id: Epoch number (defaults to current epoch)
    
    Returns:
        {
            "epoch_id": int,
            "total_validations": int,
            "validators_active": int,
            "leads_validated": int,
            "unrevealed_count": int
        }
    """
    if epoch_id is None:
        epoch_id = get_current_epoch_id()
    
    try:
        # Query validation_evidence_private for this epoch
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id, validator_hotkey, lead_id, decision", count="exact") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        validations = result.data
        total_validations = len(validations)
        
        # Count unique validators
        unique_validators = len(set([v["validator_hotkey"] for v in validations]))
        
        # Count unique leads
        unique_leads = len(set([v["lead_id"] for v in validations]))
        
        # Count unrevealed
        unrevealed = len([v for v in validations if v["decision"] is None])
        
        return {
            "epoch_id": epoch_id,
            "total_validations": total_validations,
            "validators_active": unique_validators,
            "leads_validated": unique_leads,
            "unrevealed_count": unrevealed
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validation stats: {str(e)}"
        )

