"""
Validation Submission API (Commit Phase)

Endpoint for validators to submit validation results during epoch.

This implements the COMMIT phase of commit-reveal:
- Validators submit hashes (decision_hash, rep_score_hash, evidence_hash)
- Evidence blob stored privately (NOT publicly logged)
- Actual values (decision, rep_score) revealed after epoch closes

Validators submit all validations for an epoch in a single batch request.
Works dynamically with any MAX_LEADS_PER_EPOCH (10, 20, 50, etc.).

Timing Windows:
- Blocks 0-350: Lead distribution (gateway sends leads to validators)
- Blocks 351-355: Validation submission (validators submit commit hashes)
- Blocks 356-359: Buffer period (no new submissions)
- Block 360+: Epoch closed (reveal phase begins in next epoch)

Security Safeguards:
- Epoch verification: Only current epoch accepted (no past/future submissions)
- Block cutoff: Submissions only during blocks 0-355 (closes 5 blocks early)
- Lead assignment verification: Only assigned lead_ids accepted
- Duplicate prevention: One submission per validator per epoch
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from gateway.utils.signature import verify_wallet_signature, compute_payload_hash, construct_signed_message
from gateway.utils.registry import is_registered_hotkey
from gateway.utils.nonce import check_and_store_nonce, validate_nonce_format
from gateway.utils.logger import log_event
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client, Client

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/validate", tags=["Validation"])


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationItem(BaseModel):
    """Single validation result within a batch"""
    lead_id: str
    decision_hash: str = Field(..., description="H(decision + salt)")
    rep_score_hash: str = Field(..., description="H(rep_score + salt)")
    rejection_reason_hash: str = Field(..., description="H(rejection_reason + salt)")
    evidence_hash: str = Field(..., description="H(evidence_blob)")
    evidence_blob: Dict = Field(..., description="Full evidence data (stored privately)")


class ValidationPayload(BaseModel):
    """Payload for validation submission"""
    epoch_id: int
    validations: List[ValidationItem]


class ValidationEvent(BaseModel):
    """Validation event (signed by validator)"""
    event_type: str = "VALIDATION_RESULT_BATCH"
    actor_hotkey: str = Field(..., description="Validator's SS58 address")
    nonce: str = Field(..., description="UUID v4")
    ts: datetime
    payload_hash: str = Field(..., description="SHA256 of payload")
    build_id: str
    signature: str = Field(..., description="Ed25519 signature")
    payload: ValidationPayload


# ============================================================================
# VALIDATION ENDPOINT
# ============================================================================

@router.post("/")
async def submit_validation(event: ValidationEvent):
    """
    Submit validation results for all leads in an epoch.
    
    Validators submit all their validations in a single request:
    - 1 HTTP request (not N individual requests)
    - 1 signature verification
    - Atomic operation (all succeed or all fail)
    - Efficient TEE logging (one event per epoch)
    - Works dynamically with any MAX_LEADS_PER_EPOCH (10, 20, 50, etc.)
    
    Flow:
    1. Verify payload hash
    2. Verify wallet signature
    3. Verify validator is registered
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    5.1. Verify epoch is current (not past/future) - SECURITY
    5.2. Verify within validation submission window (blocks 0-355) - SECURITY
    5.3. Verify lead_ids were assigned to this epoch - SECURITY
    5.4. Verify no duplicate submission for this epoch - SECURITY
    6. Fetch validator weights (stake + v_trust)
    7. Store all evidence blobs in validation_evidence_private
    8. Log single VALIDATION_RESULT_BATCH event to TEE
    9. Return success
    
    Args:
        event: BatchValidationEvent with epoch_id and list of validations
    
    Returns:
        {
            "status": "recorded",
            "epoch_id": int,
            "validation_count": int,
            "timestamp": str
        }
    
    Raises:
        400: Bad request (payload hash, nonce, timestamp, epoch mismatch, 
             epoch closed, invalid lead_ids, duplicate submission)
        403: Forbidden (invalid signature, not registered, not validator)
        500: Server error
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
    # CRITICAL: Must run in thread to avoid blocking event loop during metagraph fetch
    import asyncio
    try:
        is_registered, role = await asyncio.wait_for(
            asyncio.to_thread(is_registered_hotkey, event.actor_hotkey),
            timeout=90.0  # 90 second timeout for metagraph query (matches validator timeout)
        )
    except asyncio.TimeoutError:
        print(f"❌ Metagraph query timed out after 90s for {event.actor_hotkey[:20]}...")
        raise HTTPException(
            status_code=504,
            detail="Metagraph query timeout - please retry in a moment (cache warming)"
        )
    
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
    now = datetime.now(timezone.utc)
    time_diff = abs((now - event.ts).total_seconds())
    
    if time_diff > 120:  # 2 minutes tolerance
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp too old or in future (diff: {time_diff:.0f}s)"
        )
    
    # ========================================
    # Step 5.1: Verify epoch is current (not past or future)
    # ========================================
    # CRITICAL SECURITY: Prevent validators from submitting old/stale validations
    # or pre-computing validations for future epochs
    from gateway.utils.epoch import get_current_epoch_id
    
    current_epoch = get_current_epoch_id()
    
    if event.payload.epoch_id != current_epoch:
        raise HTTPException(
            status_code=400,
            detail=f"Epoch mismatch: submitted epoch {event.payload.epoch_id}, current epoch is {current_epoch}. Cannot submit validations for past or future epochs."
        )
    
    print(f"✅ Step 5.1: Epoch verification passed (epoch {event.payload.epoch_id} is current)")
    
    # ========================================
    # Step 5.2: Verify within validation submission window (blocks 0-355)
    # ========================================
    # CRITICAL SECURITY: Prevent validators from submitting after block 355
    # This gives validators:
    # - Blocks 0-350: Fetch leads from gateway
    # - Blocks 351-355: Complete validation and submit results
    # - Blocks 356-359: Buffer period (no new submissions)
    # - Block 360+: Epoch closed (next epoch begins, reveal phase starts)
    from gateway.utils.epoch import get_block_within_epoch
    
    block_within_epoch = get_block_within_epoch()
    if block_within_epoch > 355:
        raise HTTPException(
            status_code=400,
            detail=f"Validation submission window closed at block 355. Current block within epoch: {block_within_epoch}. Validators must submit before block 356."
        )
    
    print(f"✅ Step 5.2: Within validation submission window (block {block_within_epoch}/355)")
    
    # ========================================
    # Step 5.3: Verify lead_ids were assigned to this epoch
    # ========================================
    # CRITICAL SECURITY: Prevent validators from submitting validations for
    # leads that were never assigned to this epoch (could be from old epochs)
    try:
        from gateway.utils.assignment import deterministic_lead_assignment, get_validator_set
        from gateway.config import MAX_LEADS_PER_EPOCH
        
        # Get queue state for this epoch from transparency log
        # We query the leads_private table for the current pending leads
        # (same logic as /epoch/{epoch_id}/leads endpoint)
        result = supabase.table("leads_private") \
            .select("lead_id, created_ts") \
            .eq("status", "pending_validation") \
            .order("created_ts", desc=False) \
            .limit(MAX_LEADS_PER_EPOCH) \
            .execute()
        
        if not result.data:
            # No leads in queue - this is actually OK (validators can submit empty batches)
            assigned_lead_ids = []
        else:
            assigned_lead_ids = [row["lead_id"] for row in result.data]
        
        # Verify all submitted lead_ids are in the assigned set
        submitted_lead_ids = {v.lead_id for v in event.payload.validations}
        assigned_lead_ids_set = set(assigned_lead_ids)
        
        invalid_leads = submitted_lead_ids - assigned_lead_ids_set
        if invalid_leads:
            # Show first 3 invalid lead_ids for debugging
            invalid_sample = list(invalid_leads)[:3]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid lead_ids: {invalid_sample} (showing first 3) were not assigned to epoch {event.payload.epoch_id}. Cannot submit validations for unassigned leads."
            )
        
        print(f"✅ Step 5.3: Lead assignment verification passed ({len(submitted_lead_ids)} lead_ids valid for epoch {event.payload.epoch_id})")
    
    except HTTPException:
        # Re-raise HTTPException (validation errors)
        raise
    except Exception as e:
        # Log error but don't fail - we don't want this check to break the workflow
        # if there's a transient DB issue
        print(f"⚠️  Warning: Failed to verify lead assignment (continuing anyway): {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Step 5.4: Verify no duplicate submission for this epoch
    # ========================================
    # CRITICAL SECURITY: Prevent validators from submitting multiple times
    # for the same epoch (double-dipping)
    try:
        existing_submission = supabase.table("validation_evidence_private") \
            .select("evidence_id") \
            .eq("validator_hotkey", event.actor_hotkey) \
            .eq("epoch_id", event.payload.epoch_id) \
            .limit(1) \
            .execute()
        
        if existing_submission.data:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate submission detected: validator {event.actor_hotkey[:20]}... already submitted validations for epoch {event.payload.epoch_id}. Cannot submit twice."
            )
        
        print(f"✅ Step 5.4: Duplicate submission check passed (first submission for epoch {event.payload.epoch_id})")
    
    except HTTPException:
        # Re-raise HTTPException (duplicate submission error)
        raise
    except Exception as e:
        # Log error but don't fail - we don't want this check to break the workflow
        print(f"⚠️  Warning: Failed to check duplicate submission (continuing anyway): {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Step 6: Fetch validator weights (stake + v_trust)
    # ========================================
    # CRITICAL: Must snapshot stake and v_trust at COMMIT time (not REVEAL time)
    # This prevents validators from gaming the system by unstaking after seeing
    # other validators' decisions but before revealing their own.
    
    from gateway.utils.registry import get_validator_weights
    stake, v_trust = get_validator_weights(event.actor_hotkey)
    
    # ========================================
    # Step 7: Store evidence blobs (private)
    # ========================================
    # Store full evidence blobs in Supabase for reveal phase verification
    # Evidence is stored privately (NOT logged to public Arweave)
    # Only evidence_hash goes to Arweave (for tamper detection)
    
    print(f"✅ Batch validation received: {len(event.payload.validations)} validations from {event.actor_hotkey[:20]}...")
    print(f"   Epoch: {event.payload.epoch_id}")
    
    # Store evidence blobs in validation_evidence_private table
    try:
        from uuid import uuid4
        evidence_records = []
        for v in event.payload.validations:
            evidence_records.append({
                "evidence_id": str(uuid4()),  # Generate unique evidence ID
                "lead_id": v.lead_id,
                "epoch_id": event.payload.epoch_id,
                "validator_hotkey": event.actor_hotkey,
                "evidence_blob": v.evidence_blob,
                "evidence_hash": v.evidence_hash,
                "decision_hash": v.decision_hash,
                "rep_score_hash": v.rep_score_hash,
                "rejection_reason_hash": v.rejection_reason_hash,
                "stake": stake,  # Snapshot validator stake at COMMIT time
                "v_trust": v_trust,  # Snapshot validator trust score at COMMIT time
                "created_ts": event.ts.isoformat()  # Use created_ts (matches Supabase schema)
            })
        
        # Insert all evidence records in batch (with timeout to prevent hanging)
        import asyncio
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: supabase.table("validation_evidence_private").insert(evidence_records).execute()
                ),
                timeout=30.0  # 30 second timeout for Supabase insert
            )
            print(f"✅ Stored {len(evidence_records)} evidence blobs in private DB")
            print(f"   Validator stake: {stake:.6f} τ, V-Trust: {v_trust:.6f}")
        except asyncio.TimeoutError:
            # FAIL the entire request - validator will retry
            # This preserves atomicity: either everything succeeds or nothing succeeds
            print(f"❌ Supabase insert timed out after 30s")
            raise HTTPException(
                status_code=504,
                detail="Database timeout while storing evidence blobs - please retry"
            )
    
    except HTTPException:
        # Re-raise HTTPException (timeout or other HTTP errors) to fail the request
        # This preserves atomicity: if evidence storage fails, the entire request fails
        raise
    except Exception as e:
        # For non-HTTP exceptions, also fail the request to maintain atomicity
        print(f"⚠️  Failed to store evidence blobs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store evidence blobs: {str(e)}"
        )
    
    # ========================================
    # Step 8: Log to TEE transparency log
    # ========================================
    # Create log event with validation hashes (NO evidence_blob - that's private)
    validation_hashes = []
    for v in event.payload.validations:
        validation_hashes.append({
            "lead_id": v.lead_id,
            "decision_hash": v.decision_hash,
            "rep_score_hash": v.rep_score_hash,
            "rejection_reason_hash": v.rejection_reason_hash,
            "evidence_hash": v.evidence_hash
        })
    
    log_payload = {
        "epoch_id": event.payload.epoch_id,
        "validator_hotkey": event.actor_hotkey,
        "validation_count": len(event.payload.validations),
        "validations": validation_hashes  # Hashes only, no evidence_blob
    }
    
    # Log to TEE buffer (will be batched to Arweave hourly)
    log_entry = {
        "event_type": "VALIDATION_RESULT_BATCH",
        "actor_hotkey": event.actor_hotkey,
        "nonce": event.nonce,
        "ts": event.ts.isoformat(),
        "payload": log_payload,
        "signature": event.signature,
        "build_id": event.build_id
    }
    await log_event(log_entry)
    
    print(f"✅ Batch validation logged to TEE buffer")
    
    # ========================================
    # Step 9: Return success
    # ========================================
    return {
        "status": "recorded",
        "epoch_id": event.payload.epoch_id,
        "validation_count": len(event.payload.validations),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "message": f"Validation recorded in TEE. Will be logged to Arweave in next hourly checkpoint."
    }

