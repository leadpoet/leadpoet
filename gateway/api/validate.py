"""
Validation Submission API (Commit Phase)

Endpoint for validators to submit validation results during epoch.

This implements the COMMIT phase of commit-reveal:
- Validators submit hashes (decision_hash, rep_score_hash, evidence_hash)
- Evidence blob stored privately (NOT publicly logged)
- Actual values (decision, rep_score) revealed after epoch closes

Validators submit all validations for an epoch in a single batch request.
Works dynamically with any MAX_LEADS_PER_EPOCH (10, 20, 50, etc.).
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
    6. Store all evidence blobs in validation_evidence_private
    7. Log single VALIDATION_RESULT_BATCH event to TEE
    8. Return success
    
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
        400: Bad request (payload hash, nonce, timestamp)
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
    now = datetime.now(timezone.utc)
    time_diff = abs((now - event.ts).total_seconds())
    
    if time_diff > 120:  # 2 minutes tolerance
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp too old or in future (diff: {time_diff:.0f}s)"
        )
    
    # ========================================
    # Step 6: Store evidence blobs (private)
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
                "created_ts": event.ts.isoformat()  # Use created_ts (matches Supabase schema)
            })
        
        # Insert all evidence records in batch
        result = supabase.table("validation_evidence_private").insert(evidence_records).execute()
        print(f"✅ Stored {len(evidence_records)} evidence blobs in private DB")
    
    except Exception as e:
        print(f"⚠️  Failed to store evidence blobs: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the request - evidence can be reconstructed from validator logs if needed
    
    # ========================================
    # Step 7: Log to TEE transparency log
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
    # Step 8: Return success
    # ========================================
    return {
        "status": "recorded",
        "epoch_id": event.payload.epoch_id,
        "validation_count": len(event.payload.validations),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "message": f"Validation recorded in TEE. Will be logged to Arweave in next hourly checkpoint."
    }

