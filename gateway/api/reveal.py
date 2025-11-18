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
from uuid import uuid4
import hashlib
import json

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
    from gateway.utils.epoch import is_epoch_closed_async
    if not await is_epoch_closed_async(payload.epoch_id):
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
                "salt": payload.salt,
                "revealed_ts": datetime.utcnow().isoformat()
            }) \
            .eq("evidence_id", payload.evidence_id) \
            .execute()
        
        print(f"✅ Evidence revealed: {payload.evidence_id}")
        print(f"   Epoch: {payload.epoch_id}")
        print(f"   Decision: {payload.decision}")
        print(f"   Rep score: {payload.rep_score}")
        print(f"   Rejection reason: {payload.rejection_reason}")
        print(f"   Salt: {payload.salt[:8]}...")
        print(f"   Validator: {validator_hotkey[:20]}...")
    
    except Exception as e:
        print(f"❌ Error updating evidence: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update evidence with revealed values"
        )
    
    # ========================================
    # Step 9.5: Update leads_private.validators_responded array
    # ========================================
    # CRITICAL FIX: Ensure late reveals are reflected in the lead record
    try:
        lead_id = evidence["lead_id"]
        
        # Fetch current lead record
        lead_result = supabase.table("leads_private")\
            .select("validators_responded, validator_responses")\
            .eq("lead_id", lead_id)\
            .limit(1)\
            .execute()
        
        if lead_result.data and len(lead_result.data) > 0:
            lead_data = lead_result.data[0]
            
            # Get current arrays (or empty if null)
            validators_responded = lead_data.get("validators_responded") or []
            validator_responses = lead_data.get("validator_responses") or []
            
            # Only append if validator not already in list (idempotent)
            if validator_hotkey not in validators_responded:
                validators_responded.append(validator_hotkey)
                
                # Append full response object
                validator_responses.append({
                    "validator": validator_hotkey,
                    "decision": payload.decision,
                    "rep_score": payload.rep_score,
                    "rejection_reason": payload.rejection_reason,
                    "submitted_at": datetime.utcnow().isoformat(),
                    "v_trust": None,  # Will be populated by next consensus run
                    "stake": None     # Will be populated by next consensus run
                })
                
                # Update leads_private with appended arrays
                supabase.table("leads_private")\
                    .update({
                        "validators_responded": validators_responded,
                        "validator_responses": validator_responses
                    })\
                    .eq("lead_id", lead_id)\
                    .execute()
                
                print(f"✅ Appended {validator_hotkey[:10]}... to validators_responded ({len(validators_responded)} total)")
            else:
                print(f"ℹ️  Validator {validator_hotkey[:10]}... already in validators_responded (idempotent)")
    except Exception as e:
        # Don't fail the entire reveal if leads_private update fails
        # The evidence is already updated, which is the source of truth
        print(f"⚠️  Warning: Failed to update leads_private.validators_responded: {e}")
    
    # ========================================
    # Step 10: Log REVEAL event to TEE Buffer
    # ========================================
    reveal_timestamp = datetime.utcnow().isoformat()
    
    print(f"🔍 Step 10: Logging REVEAL to TEE buffer...")
    try:
        from gateway.utils.logger import log_event
        import hashlib
        
        # Construct REVEAL event for TEE buffer
        reveal_log_entry = {
            "event_type": "REVEAL",
            "actor_hotkey": validator_hotkey,
            "nonce": str(uuid4()),  # Generate fresh nonce for this event
            "ts": reveal_timestamp,
            "payload_hash": hashlib.sha256(
                json.dumps({
                    "evidence_id": payload.evidence_id,
                    "epoch_id": payload.epoch_id,
                    "decision": payload.decision,
                    "rep_score": payload.rep_score,
                    "rejection_reason": payload.rejection_reason,
                    "salt": payload.salt
                }, sort_keys=True).encode()
            ).hexdigest(),
            "build_id": "gateway",  # Gateway-generated event
            "signature": signature,  # Validator's signature over reveal payload
            "payload": {
                "evidence_id": payload.evidence_id,
                "lead_id": evidence["lead_id"],
                "epoch_id": payload.epoch_id,
                "decision": payload.decision,
                "rep_score": payload.rep_score,
                "rejection_reason": payload.rejection_reason,
                "salt": payload.salt,  # CRITICAL for public audit (verify hashes on Arweave)
                "validator_hotkey": validator_hotkey
                # NOTE: evidence_blob is NEVER revealed publicly (kept private)
            }
        }
        
        # Write to TEE buffer (hardware-protected)
        result = await log_event(reveal_log_entry)
        
        reveal_tee_seq = result.get("sequence")
        print(f"✅ Step 10 complete: REVEAL buffered in TEE: seq={reveal_tee_seq}")
    
    except Exception as e:
        print(f"❌ Error logging REVEAL: {e}")
        import traceback
        traceback.print_exc()
        print(f"   ⚠️  CONTINUING despite logging error - reveal is already stored in DB")
    
    # ========================================
    # Step 11: Return Response
    # ========================================
    # NOTE (Phase 4): Receipts deprecated - TEE attestation provides trust
    # - Event is buffered in TEE (hardware-protected memory)
    # - Will be included in next hourly Arweave checkpoint (signed by TEE)
    # - Verify gateway code integrity: GET /attest
    reveal_timestamp = datetime.now(tz.utc).isoformat()
    
    return {
        "status": "revealed",
        "evidence_id": payload.evidence_id,
        "epoch_id": payload.epoch_id,
        "decision": payload.decision,
        "rep_score": payload.rep_score,
        "rejection_reason": payload.rejection_reason,
        "timestamp": reveal_timestamp,
        "message": "Validation revealed successfully. Proof available in next hourly Arweave checkpoint."
    }


@router.post("/batch")
async def reveal_validation_batch(
    epoch_id: int = Body(..., description="Epoch ID"),
    validator_hotkey: str = Body(..., description="Validator's SS58 address"),
    signature: str = Body(..., description="Ed25519 signature over message"),
    nonce: str = Body(..., description="Nonce used in signature"),
    reveals: list = Body(..., description="List of reveal objects")
):
    """
    Batch reveal endpoint for validators to reveal multiple validations at once.
    
    This is more efficient than calling /reveal once per lead.
    
    Args:
        epoch_id: Epoch ID
        validator_hotkey: Validator's SS58 address
        signature: Ed25519 signature over "reveal:{hotkey}:{epoch_id}:{nonce}"
        nonce: Nonce (timestamp)
        reveals: List of {lead_id, decision, rep_score, rejection_reason, salt}
    
    Returns:
        {
            "status": "revealed",
            "epoch_id": int,
            "revealed_count": int,
            "failed_count": int,
            "errors": List[str]
        }
    """
    # Verify signature
    message = f"reveal:{validator_hotkey}:{epoch_id}:{nonce}"
    if not verify_wallet_signature(message, signature, validator_hotkey):
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # Verify epoch is closed
    from gateway.utils.epoch import is_epoch_closed_async
    if not await is_epoch_closed_async(epoch_id):
        raise HTTPException(
            status_code=400,
            detail=f"Epoch {epoch_id} is not closed yet. Wait until epoch closes to reveal."
        )
    
    print(f"\n{'='*80}")
    print(f"📥 BATCH REVEAL: {len(reveals)} reveals from {validator_hotkey[:20]}...")
    print(f"{'='*80}")
    
    import time
    start_time = time.time()
    
    revealed_count = 0
    failed_count = 0
    errors = []
    
    # ============================================================================
    # OPTIMIZATION: Bulk database operations (1 SELECT + 1 UPDATE for all leads)
    # Instead of N queries per lead, do 2 queries total
    # ============================================================================
    
    # Step 1: Build lead_id list for bulk SELECT
    lead_ids = [reveal["lead_id"] for reveal in reveals]
    
    print(f"   📊 Fetching evidence for {len(lead_ids)} leads in bulk...")
    
    # Step 2: Bulk SELECT all evidence for this validator + epoch
    import asyncio
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: supabase.table("validation_evidence_private")
                    .select("*")
                    .eq("validator_hotkey", validator_hotkey)
                    .eq("epoch_id", epoch_id)
                    .in_("lead_id", lead_ids)
                    .execute()
            ),
            timeout=90.0  # 90 second timeout for bulk query (Supabase can be very slow on testnet)
        )
        evidence_records = result.data
        print(f"   ✅ Fetched {len(evidence_records)} evidence records from database")
    except asyncio.TimeoutError:
        error_msg = f"Database timeout during bulk SELECT (90s)"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)
        return {
            "status": "error",
            "epoch_id": epoch_id,
            "revealed_count": 0,
            "failed_count": len(reveals),
            "errors": [error_msg]
        }
    except Exception as e:
        error_msg = f"Database error during bulk SELECT: {str(e)}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)
        return {
            "status": "error",
            "epoch_id": epoch_id,
            "revealed_count": 0,
            "failed_count": len(reveals),
            "errors": [error_msg]
        }
    
    # Step 3: Build evidence lookup map (lead_id -> evidence)
    evidence_map = {evidence["lead_id"]: evidence for evidence in evidence_records}
    
    # Step 4: Verify hashes and build bulk update list
    valid_updates = []
    
    for idx, reveal in enumerate(reveals, 1):
        try:
            lead_id = reveal["lead_id"]
            decision = reveal["decision"]
            rep_score = reveal["rep_score"]
            rejection_reason = reveal["rejection_reason"]
            salt = reveal["salt"]
            
            print(f"   [{idx}/{len(reveals)}] Verifying lead {lead_id[:8]}... - Decision: {decision}")
            
            # Check if evidence exists in bulk query results
            if lead_id not in evidence_map:
                error_msg = f"Evidence not found for lead {lead_id[:8]}..."
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue
            
            evidence = evidence_map[lead_id]
            
            # Verify hashes
            computed_decision_hash = hashlib.sha256((decision + salt).encode()).hexdigest()
            computed_rep_score_hash = hashlib.sha256((str(rep_score) + salt).encode()).hexdigest()
            computed_rejection_reason_hash = hashlib.sha256((json.dumps(rejection_reason) + salt).encode()).hexdigest()
            
            if computed_decision_hash != evidence["decision_hash"]:
                error_msg = f"Decision hash mismatch for lead {lead_id[:8]}..."
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue
            
            if computed_rep_score_hash != evidence["rep_score_hash"]:
                error_msg = f"Rep score hash mismatch for lead {lead_id[:8]}..."
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue
            
            if computed_rejection_reason_hash != evidence["rejection_reason_hash"]:
                error_msg = f"Rejection reason hash mismatch for lead {lead_id[:8]}..."
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
                continue
            
            # Hash verification passed - add to bulk update list
            valid_updates.append({
                "evidence_id": evidence["evidence_id"],
                "lead_id": lead_id,
                "decision": decision,
                "rep_score": rep_score,
                "rejection_reason": json.dumps(rejection_reason),
                "salt": salt
            })
            
            print(f"      ✅ Hash verified: {decision} (rep_score={rep_score})")
            
        except Exception as e:
            error_msg = f"Error verifying lead {reveal.get('lead_id', 'unknown')[:8]}...: {str(e)}"
            print(f"      ❌ {error_msg}")
            errors.append(error_msg)
            failed_count += 1
    
    # Step 5: Bulk UPDATE all valid reveals
    if valid_updates:
        print(f"   📊 Performing bulk UPDATE for {len(valid_updates)} validated reveals...")
        
        revealed_ts = datetime.utcnow().isoformat()
        
        # Update each evidence record
        # Note: Supabase Python client doesn't support true bulk updates with different values per row,
        # so we need to update individually but in a tight loop (still faster than original N queries)
        for idx, update in enumerate(valid_updates, 1):
            try:
                # Extract values explicitly (avoid lambda closure issues)
                evidence_id = update["evidence_id"]
                decision = update["decision"]
                rep_score = update["rep_score"]
                rejection_reason_json = update["rejection_reason"]
                salt = update["salt"]
                lead_id = update["lead_id"]
                
                # Define update function
                def do_update():
                    return supabase.table("validation_evidence_private")\
                        .update({
                            "decision": decision,
                            "rep_score": rep_score,
                            "rejection_reason": rejection_reason_json,
                            "salt": salt,
                            "revealed_ts": revealed_ts
                        })\
                        .eq("evidence_id", evidence_id)\
                        .execute()
                
                # Execute update with timeout
                await asyncio.wait_for(
                    asyncio.to_thread(do_update),
                    timeout=15.0  # 15 second timeout per update (Supabase can be slow)
                )
                
                # ════════════════════════════════════════════════════════════════════
                # CRITICAL FIX: Also update leads_private.validators_responded array
                # This ensures late reveals are reflected in the lead record
                # ════════════════════════════════════════════════════════════════════
                try:
                    # Fetch current lead record
                    def fetch_lead():
                        return supabase.table("leads_private")\
                            .select("validators_responded, validator_responses")\
                            .eq("lead_id", lead_id)\
                            .limit(1)\
                            .execute()
                    
                    lead_result = await asyncio.wait_for(
                        asyncio.to_thread(fetch_lead),
                        timeout=10.0
                    )
                    
                    if lead_result.data and len(lead_result.data) > 0:
                        lead_data = lead_result.data[0]
                        
                        # Get current arrays (or empty if null)
                        validators_responded = lead_data.get("validators_responded") or []
                        validator_responses = lead_data.get("validator_responses") or []
                        
                        # Only append if validator not already in list (idempotent)
                        if validator_hotkey not in validators_responded:
                            validators_responded.append(validator_hotkey)
                            
                            # Append full response object
                            validator_responses.append({
                                "validator": validator_hotkey,
                                "decision": decision,
                                "rep_score": rep_score,
                                "rejection_reason": rejection_reason_json,
                                "submitted_at": revealed_ts,
                                "v_trust": None,  # Will be populated by next consensus run
                                "stake": None     # Will be populated by next consensus run
                            })
                            
                            # Update leads_private with appended arrays
                            def update_lead():
                                return supabase.table("leads_private")\
                                    .update({
                                        "validators_responded": validators_responded,
                                        "validator_responses": validator_responses
                                    })\
                                    .eq("lead_id", lead_id)\
                                    .execute()
                            
                            await asyncio.wait_for(
                                asyncio.to_thread(update_lead),
                                timeout=10.0
                            )
                            
                            print(f"         ✅ Appended {validator_hotkey[:10]}... to validators_responded ({len(validators_responded)} total)")
                        else:
                            print(f"         ℹ️  Validator {validator_hotkey[:10]}... already in validators_responded (idempotent)")
                except Exception as e:
                    # Don't fail the entire reveal if leads_private update fails
                    # The evidence is already updated, which is the source of truth
                    print(f"         ⚠️  Warning: Failed to update leads_private.validators_responded: {e}")
                
                print(f"      [{idx}/{len(valid_updates)}] ✅ Updated lead {lead_id[:8]}...")
                revealed_count += 1
                
            except asyncio.TimeoutError:
                error_msg = f"Database timeout updating lead {update['lead_id'][:8]}..."
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
            except Exception as e:
                error_msg = f"Database error updating lead {update['lead_id'][:8]}...: {str(e)}"
                print(f"      ❌ {error_msg}")
                errors.append(error_msg)
                failed_count += 1
        
        print(f"   ✅ Bulk UPDATE complete: {revealed_count} successful, {failed_count} failed")
    else:
        print(f"   ⚠️  No valid reveals to update (all failed hash verification)")
    
    elapsed_time = time.time() - start_time
    print(f"{'='*80}")
    print(f"✅ Batch reveal complete: {revealed_count} revealed, {failed_count} failed")
    print(f"   ⏱️  Total time: {elapsed_time:.2f}s")
    print(f"{'='*80}\n")
    
    return {
        "status": "revealed",
        "epoch_id": epoch_id,
        "revealed_count": revealed_count,
        "failed_count": failed_count,
        "errors": errors if errors else None
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

