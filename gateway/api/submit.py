"""
POST /submit - Verify lead upload and finalize submission
=========================================================

After miner uploads lead blob to S3/MinIO via presigned URLs,
they call this endpoint to trigger verification.

Flow per BRD Section 4.1:
1. Gateway fetches uploaded blob from each mirror
2. Recomputes SHA256 hash
3. Verifies hash matches committed lead_blob_hash from SUBMISSION_REQUEST
4. If verification succeeds:
   - Logs STORAGE_PROOF event per mirror
   - Stores lead in leads_private table
   - Logs SUBMISSION event
5. If verification fails:
   - Logs UPLOAD_FAILED event
   - Returns error

This prevents blob substitution attacks (BRD Section 5.2).
"""

import sys
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

# Import configuration
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Import utilities
from gateway.utils.signature import verify_wallet_signature, construct_signed_message, compute_payload_hash
from gateway.utils.registry import is_registered_hotkey
from gateway.utils.nonce import check_and_store_nonce, validate_nonce_format
from gateway.utils.storage import verify_storage_proof

# Import Supabase
from supabase import create_client, Client

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/submit", tags=["Submission"])


# ============================================================
# Request Models
# ============================================================

class SubmitLeadPayload(BaseModel):
    """Payload for submit request"""
    lead_id: str = Field(..., description="UUID of lead")


class SubmitLeadEvent(BaseModel):
    """
    Event for finalizing lead submission after upload.
    
    Miner signs this event after uploading to S3/MinIO.
    Gateway verifies uploaded blobs match committed hash.
    """
    event_type: str = "SUBMIT_LEAD"
    actor_hotkey: str = Field(..., description="Miner's SS58 address")
    nonce: str = Field(..., description="UUID v4 nonce")
    ts: datetime = Field(..., description="ISO timestamp")
    payload_hash: str = Field(..., description="SHA256 of payload")
    build_id: str = Field(default="miner-client", description="Client build ID")
    signature: str = Field(..., description="Ed25519 signature")
    payload: SubmitLeadPayload


# ============================================================
# POST /submit - Verify and finalize lead submission
# ============================================================

@router.post("/")
async def submit_lead(event: SubmitLeadEvent):
    """
    Verify uploaded lead blobs and finalize submission.
    
    Called by miner after uploading lead blob to S3/MinIO via presigned URLs.
    
    Flow (BRD Section 4.1, Steps 5-6):
    1. Verify payload hash
    2. Verify wallet signature
    3. Check actor is registered miner
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    6. Fetch SUBMISSION_REQUEST event to get committed lead_blob_hash
    7. Verify uploaded blob from S3 matches lead_blob_hash
    8. Verify uploaded blob from MinIO matches lead_blob_hash
    9a. SUCCESS PATH (if both mirrors verify):
        - Log STORAGE_PROOF event for S3
        - Log STORAGE_PROOF event for MinIO
        - Store lead in leads_private table
        - Log SUBMISSION event
        - Return {status: "accepted", lead_id, merkle_proof}
    9b. FAILURE PATH (if verification fails):
        - Log UPLOAD_FAILED event
        - Return HTTPException 400
    
    Args:
        event: SubmitLeadEvent with lead_id and miner signature
    
    Returns:
        {
            "status": "accepted",
            "lead_id": "uuid",
            "storage_backends": ["s3", "minio"],
            "merkle_proof": ["hash1", "hash2", ...],
            "submission_ts": "ISO timestamp"
        }
    
    Raises:
        400: Bad request (payload hash, nonce, timestamp, verification failed)
        403: Forbidden (invalid signature, not registered, not miner)
        404: SUBMISSION_REQUEST not found
        500: Server error
    
    Security:
        - Ed25519 signature verification
        - Nonce replay protection
        - Hash verification (prevents blob substitution)
        - Only registered miners can submit
    """
    
    import uuid  # For generating nonces for transparency log events
    
    print(f"\n🔍 POST /submit called - lead_id={event.payload.lead_id}")
    
    # ========================================
    # Step 0: Check rate limits (BEFORE expensive operations)
    # ========================================
    # This is a DoS protection mechanism - we check rate limits using only
    # the actor_hotkey field from the JSON payload BEFORE doing any expensive
    # crypto operations (signature verification) or I/O (DB queries, S3 fetches).
    # 
    # An attacker can spam fake requests, but we reject them in <1ms.
    print("🔍 Step 0: Checking rate limits...")
    from gateway.utils.rate_limiter import check_rate_limit
    
    allowed, reason, stats = check_rate_limit(event.actor_hotkey)
    if not allowed:
        print(f"❌ Rate limit exceeded for {event.actor_hotkey[:20]}...")
        print(f"   Reason: {reason}")
        print(f"   Stats: {stats}")
        
        # Log RATE_LIMIT_HIT event to TEE buffer (for transparency)
        try:
            from gateway.utils.logger import log_event
            
            rate_limit_event = {
                "event_type": "RATE_LIMIT_HIT",
                "actor_hotkey": event.actor_hotkey,
                "nonce": str(uuid.uuid4()),
                "ts": datetime.utcnow().isoformat(),
                "payload_hash": hashlib.sha256(json.dumps({
                    "lead_id": event.payload.lead_id,
                    "reason": reason,
                    "stats": stats
                }, sort_keys=True).encode()).hexdigest(),
                "build_id": "gateway",
                "signature": "rate_limit_check",  # No signature needed (gateway-generated)
                "payload": {
                    "lead_id": event.payload.lead_id,
                    "reason": reason,
                    "stats": stats
                }
            }
            
            await log_event(rate_limit_event)
            print(f"   ✅ Logged RATE_LIMIT_HIT to TEE buffer")
        except Exception as e:
            print(f"   ⚠️  Failed to log RATE_LIMIT_HIT: {e}")
        
        # Return 429 Too Many Requests
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": reason,
                "stats": stats
            }
        )
    
    print(f"🔍 Step 0 complete: Rate limit OK (submissions={stats['submissions']}, rejections={stats['rejections']})")
    
    # ========================================
    # Step 1: Verify payload hash
    # ========================================
    print("🔍 Step 1: Verifying payload hash...")
    computed_hash = compute_payload_hash(event.payload.model_dump())
    if computed_hash != event.payload_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Payload hash mismatch: expected {event.payload_hash[:16]}..., got {computed_hash[:16]}..."
        )
    print("🔍 Step 1 complete: Payload hash valid")
    
    # ========================================
    # Step 2: Verify wallet signature
    # ========================================
    print("🔍 Step 2: Verifying signature...")
    message = construct_signed_message(event)
    is_valid = verify_wallet_signature(message, event.signature, event.actor_hotkey)
    
    if not is_valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    print("🔍 Step 2 complete: Signature valid")
    
    # ========================================
    # Step 3: Check actor is registered miner
    # ========================================
    print("🔍 Step 3: Checking registration...")
    is_registered, role = is_registered_hotkey(event.actor_hotkey)
    
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    if role != "miner":
        raise HTTPException(
            status_code=403,
            detail="Only miners can submit leads"
        )
    print(f"🔍 Step 3 complete: Miner registered (hotkey={event.actor_hotkey[:10]}...)")
    
    # ========================================
    # Step 4: Verify nonce format and freshness
    # ========================================
    print("🔍 Step 4: Verifying nonce...")
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
    print("🔍 Step 4 complete: Nonce valid")
    
    # ========================================
    # Step 5: Verify timestamp
    # ========================================
    print("🔍 Step 5: Verifying timestamp...")
    from datetime import timezone as tz
    from gateway.config import TIMESTAMP_TOLERANCE_SECONDS
    
    now = datetime.now(tz.utc)
    event_ts = event.ts if event.ts.tzinfo else event.ts.replace(tzinfo=tz.utc)
    time_diff = abs((now - event_ts).total_seconds())
    
    if time_diff > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp out of range: {time_diff:.0f}s (max: {TIMESTAMP_TOLERANCE_SECONDS}s)"
        )
    print(f"🔍 Step 5 complete: Timestamp valid (diff={time_diff:.2f}s)")
    
    # ========================================
    # Step 6: Fetch SUBMISSION_REQUEST event
    # ========================================
    print(f"🔍 Step 6: Fetching SUBMISSION_REQUEST for lead_id={event.payload.lead_id}...")
    try:
        result = supabase.table("transparency_log") \
            .select("*") \
            .eq("event_type", "SUBMISSION_REQUEST") \
            .eq("actor_hotkey", event.actor_hotkey) \
            .execute()
        
        print(f"🔍 Found {len(result.data) if result.data else 0} SUBMISSION_REQUEST events for actor {event.actor_hotkey[:8]}...")
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No SUBMISSION_REQUEST found for actor {event.actor_hotkey[:8]}... (lead_id={event.payload.lead_id[:8]}...)"
            )
        
        # Find the specific SUBMISSION_REQUEST for this lead_id
        submission_request = None
        for log in result.data:
            payload = log.get("payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)
            if payload.get("lead_id") == event.payload.lead_id:
                submission_request = log
                break
        
        if not submission_request:
            raise HTTPException(
                status_code=404,
                detail=f"SUBMISSION_REQUEST not found for lead_id={event.payload.lead_id}"
            )
        
        # Extract committed lead_blob_hash and email_hash
        payload = submission_request.get("payload", {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        committed_lead_blob_hash = payload.get("lead_blob_hash")
        committed_email_hash = payload.get("email_hash")
        
        if not committed_lead_blob_hash:
            raise HTTPException(
                status_code=500,
                detail="SUBMISSION_REQUEST missing lead_blob_hash"
            )
        
        if not committed_email_hash:
            raise HTTPException(
                status_code=500,
                detail="SUBMISSION_REQUEST missing email_hash"
            )
        
        print(f"🔍 Step 6 complete: Found SUBMISSION_REQUEST")
        print(f"   Committed lead_blob_hash: {committed_lead_blob_hash[:32]}...{committed_lead_blob_hash[-8:]}")
        print(f"   Committed email_hash: {committed_email_hash[:32]}...{committed_email_hash[-8:]}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching SUBMISSION_REQUEST: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch SUBMISSION_REQUEST: {str(e)}"
        )
    
    # ========================================
    # Step 7: Verify S3 upload
    # ========================================
    print(f"🔍 Step 7: Verifying S3 upload...")
    s3_verified = verify_storage_proof(committed_lead_blob_hash, "s3")
    
    if s3_verified:
        print(f"✅ S3 verification successful")
    else:
        print(f"❌ S3 verification failed")
    
    # ========================================
    # Step 8: Mirror S3 to MinIO (gateway-side)
    # ========================================
    minio_verified = False
    if s3_verified:
        print(f"🔍 Step 8: Mirroring S3 to MinIO (gateway-side)...")
        from gateway.utils.storage import mirror_s3_to_minio
        mirror_success = mirror_s3_to_minio(committed_lead_blob_hash)
        
        if mirror_success:
            # Verify MinIO mirror
            print(f"🔍 Step 8b: Verifying MinIO mirror...")
            minio_verified = verify_storage_proof(committed_lead_blob_hash, "minio")
            if minio_verified:
                print(f"✅ MinIO mirror verified")
            else:
                print(f"⚠️  MinIO mirror verification failed (but S3 is OK)")
        else:
            print(f"⚠️  MinIO mirroring failed (but S3 is OK)")
    
    # ========================================
    # Step 9a: SUCCESS PATH - S3 verified (MinIO is optional backup)
    # ========================================
    if s3_verified:
        print(f"🔍 Step 9a: SUCCESS PATH - S3 verified (MinIO {'verified' if minio_verified else 'optional'})")
        
        try:
            # Log STORAGE_PROOF events to TEE buffer (hardware-protected)
            from gateway.utils.logger import log_event
            import asyncio
            
            storage_proof_events = []
            storage_proof_tee_seqs = {}
            
            # Only log for verified mirrors
            verified_mirrors = ["s3"]
            if minio_verified:
                verified_mirrors.append("minio")
            
            for mirror in verified_mirrors:
                storage_proof_payload = {
                    "lead_id": event.payload.lead_id,
                    "lead_blob_hash": committed_lead_blob_hash,
                    "email_hash": committed_email_hash,
                    "mirror": mirror,
                    "verified": True
                }
                
                storage_proof_log_entry = {
                    "event_type": "STORAGE_PROOF",
                    "actor_hotkey": "gateway",
                    "nonce": str(uuid.uuid4()),  # Generate fresh UUID for each event
                    "ts": datetime.now(tz.utc).isoformat(),
                    "payload_hash": hashlib.sha256(
                        json.dumps(storage_proof_payload, sort_keys=True).encode()
                    ).hexdigest(),
                    "build_id": "gateway",
                    "signature": "gateway_internal",
                    "payload": storage_proof_payload
                }
                
                print(f"   🔍 Logging STORAGE_PROOF for {mirror} to TEE buffer...")
                result = await log_event(storage_proof_log_entry)
                
                tee_sequence = result.get("sequence")
                storage_proof_tee_seqs[mirror] = tee_sequence
                print(f"   ✅ STORAGE_PROOF buffered in TEE for {mirror}: seq={tee_sequence}")
                
                storage_proof_events.append(mirror)
                
        except Exception as e:
            print(f"❌ Error logging STORAGE_PROOF: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL: If TEE write fails, request MUST fail
            print(f"🚨 CRITICAL: TEE buffer unavailable - failing request")
            raise HTTPException(
                status_code=503,
                detail=f"TEE buffer unavailable: {str(e)}"
            )
        
        # Fetch the lead blob from S3 to store in leads_private
        from gateway.utils.storage import s3_client
        from gateway.config import AWS_S3_BUCKET
        
        print(f"   🔍 Fetching lead blob from S3 for database storage...")
        object_key = f"leads/{committed_lead_blob_hash}.json"
        try:
            response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=object_key)
            lead_blob = json.loads(response['Body'].read().decode('utf-8'))
            print(f"   ✅ Lead blob fetched from S3")
        except Exception as e:
            print(f"❌ Failed to fetch lead blob from S3: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch lead blob: {str(e)}"
            )
        
        # ========================================
        # CRITICAL: Validate Required Fields (README.md lines 239-258)
        # ========================================
        print(f"   🔍 Validating required fields...")
        
        REQUIRED_FIELDS = [
            "business",      # Company name
            "full_name",     # Contact full name
            "first",         # First name
            "last",          # Last name
            "email",         # Email address
            "role",          # Job title
            "website",       # Company website
            "industry",      # Primary industry
            "sub_industry",  # Sub-industry/niche
            "region",        # Location
            "linkedin"       # LinkedIn URL
        ]
        
        missing_fields = []
        for field in REQUIRED_FIELDS:
            value = lead_blob.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Required fields validation failed: Missing {len(missing_fields)} field(s)")
            print(f"   Missing: {', '.join(missing_fields)}")
            
            # Increment rate limit counter (FAILURE - missing required fields)
            from gateway.utils.rate_limiter import increment_submission
            updated_stats = increment_submission(event.actor_hotkey, success=False)
            print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/10, rejections={updated_stats['rejections']}/10")
            
            # Log VALIDATION_FAILED event to TEE buffer (for transparency)
            try:
                from gateway.utils.logger import log_event
                
                validation_failed_event = {
                    "event_type": "VALIDATION_FAILED",
                    "actor_hotkey": event.actor_hotkey,
                    "nonce": str(uuid.uuid4()),
                    "ts": datetime.utcnow().isoformat(),
                    "payload_hash": hashlib.sha256(json.dumps({
                        "lead_id": event.payload.lead_id,
                        "reason": "missing_required_fields",
                        "missing_fields": missing_fields
                    }, sort_keys=True).encode()).hexdigest(),
                    "build_id": "gateway",
                    "signature": "required_fields_check",  # Gateway-generated
                    "payload": {
                        "lead_id": event.payload.lead_id,
                        "reason": "missing_required_fields",
                        "missing_fields": missing_fields,
                        "miner_hotkey": event.actor_hotkey
                    }
                }
                
                await log_event(validation_failed_event)
                print(f"   ✅ Logged VALIDATION_FAILED to TEE buffer")
            except Exception as e:
                print(f"   ⚠️  Failed to log VALIDATION_FAILED: {e}")
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_required_fields",
                    "message": f"Lead is missing {len(missing_fields)} required field(s)",
                    "missing_fields": missing_fields,
                    "required_fields": REQUIRED_FIELDS
                }
            )
        
        print(f"   ✅ All required fields present")
        
        # ========================================
        # CRITICAL: Verify Miner Attestation
        # ========================================
        print(f"   🔍 Verifying miner attestation...")
        try:
            wallet_ss58 = lead_blob.get("wallet_ss58")
            terms_version_hash = lead_blob.get("terms_version_hash")
            
            if not wallet_ss58 or not terms_version_hash:
                print(f"❌ Attestation check failed: Missing wallet_ss58 or terms_version_hash in lead")
                raise HTTPException(
                    status_code=400,
                    detail="Lead missing required attestation fields (wallet_ss58, terms_version_hash)"
                )
            
            # Query contributor_attestations table
            attestation_result = supabase.table("contributor_attestations")\
                .select("*")\
                .eq("wallet_ss58", wallet_ss58)\
                .eq("terms_version_hash", terms_version_hash)\
                .eq("accepted", True)\
                .execute()
            
            if not attestation_result.data or len(attestation_result.data) == 0:
                print(f"❌ Attestation check failed: No valid attestation for wallet {wallet_ss58[:20]}...")
                raise HTTPException(
                    status_code=403,
                    detail=f"Miner has not accepted terms (wallet: {wallet_ss58[:20]}...)"
                )
            
            # Verify attestation metadata matches lead
            attestation = attestation_result.data[0]
            expected_lawful = lead_blob.get("lawful_collection")
            expected_no_restricted = lead_blob.get("no_restricted_sources")
            expected_licensed = lead_blob.get("license_granted")
            
            actual_lawful = attestation.get("lawful_collection")
            actual_no_restricted = attestation.get("no_restricted_sources")
            actual_licensed = attestation.get("license_granted")
            
            if (expected_lawful != actual_lawful or 
                expected_no_restricted != actual_no_restricted or 
                expected_licensed != actual_licensed):
                print(f"❌ Attestation mismatch: Lead metadata doesn't match attestation record")
                raise HTTPException(
                    status_code=400,
                    detail="Lead attestation metadata doesn't match miner's attestation record"
                )
            
            print(f"   ✅ Attestation verified for wallet {wallet_ss58[:20]}...")
            
        except HTTPException:
            # Increment rate limit counter (FAILURE - attestation check)
            from gateway.utils.rate_limiter import increment_submission
            updated_stats = increment_submission(event.actor_hotkey, success=False)
            print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/10, rejections={updated_stats['rejections']}/10")
            
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            print(f"❌ Attestation verification error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Attestation verification failed: {str(e)}"
            )
        
        # Store lead in leads_private table
        print(f"   🔍 Storing lead in leads_private database...")
        try:
            # NOTE: miner_hotkey column doesn't exist yet in Supabase
            # TODO: Add migration to add miner_hotkey column for optimization
            lead_private_entry = {
                "lead_id": event.payload.lead_id,
                "lead_blob_hash": committed_lead_blob_hash,
                # "miner_hotkey": event.actor_hotkey,  # TODO: Uncomment after column is added
                "lead_blob": lead_blob,
                "created_ts": datetime.now(tz.utc).isoformat()
            }
            
            supabase.table("leads_private").insert(lead_private_entry).execute()
            print(f"   ✅ Lead stored in leads_private")
            
        except Exception as e:
            print(f"❌ Failed to store lead in leads_private: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store lead: {str(e)}"
            )
        
        # Log SUBMISSION event to Arweave FIRST
        print(f"   🔍 Logging SUBMISSION event to TEE buffer...")
        try:
            submission_payload = {
                "lead_id": event.payload.lead_id,
                "lead_blob_hash": committed_lead_blob_hash,
                "email_hash": committed_email_hash,
                "miner_hotkey": event.actor_hotkey,
                "submission_timestamp": datetime.now(tz.utc).isoformat(),
                "s3_proof_tee_seq": storage_proof_tee_seqs.get("s3"),
                "minio_proof_tee_seq": storage_proof_tee_seqs.get("minio")
            }
            
            submission_log_entry = {
                "event_type": "SUBMISSION",
                "actor_hotkey": event.actor_hotkey,
                "nonce": str(uuid.uuid4()),  # Generate fresh UUID for this event
                "ts": datetime.now(tz.utc).isoformat(),
                "payload_hash": hashlib.sha256(
                    json.dumps(submission_payload, sort_keys=True).encode()
                ).hexdigest(),
                "build_id": event.build_id,
                "signature": event.signature,
                "payload": submission_payload
            }
            
            result = await log_event(submission_log_entry)
            
            submission_tee_seq = result.get("sequence")
            print(f"   ✅ SUBMISSION event buffered in TEE: seq={submission_tee_seq}")
                
        except Exception as e:
            print(f"❌ Error logging SUBMISSION event: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL: If TEE write fails, request MUST fail
            print(f"🚨 CRITICAL: TEE buffer unavailable - failing request")
            raise HTTPException(
                status_code=503,
                detail=f"TEE buffer unavailable: {str(e)}"
            )
        
        # Compute queue_position (simplified - just count total submissions)
        submission_timestamp = datetime.now(tz.utc).isoformat()
        try:
            queue_count_result = supabase.table("leads_private").select("lead_id", count="exact").execute()
            queue_position = queue_count_result.count if hasattr(queue_count_result, 'count') else None
        except Exception as e:
            print(f"⚠️  Could not compute queue_position: {e}")
            queue_position = None
        
        # Return success with simple acknowledgment
        # NOTE (Phase 4): TEE-based trust model
        # - Events buffered in TEE (hardware-protected memory)
        # - Will be included in next hourly Arweave checkpoint (signed by TEE)
        # - Verify gateway code integrity: GET /attest
        
        # Increment rate limit counter (SUCCESS)
        from gateway.utils.rate_limiter import increment_submission
        updated_stats = increment_submission(event.actor_hotkey, success=True)
        print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/10, rejections={updated_stats['rejections']}/10")
        
        print(f"✅ /submit complete - lead accepted")
        return {
            "status": "accepted",
            "lead_id": event.payload.lead_id,
            "storage_backends": storage_proof_events,
            "submission_timestamp": submission_timestamp,
            "queue_position": queue_position,
            "message": "Lead accepted. Proof available in next hourly Arweave checkpoint."
        }
    
    # ========================================
    # Step 9b: FAILURE PATH - Verification failed
    # ========================================
    else:
        print(f"🔍 Step 9b: FAILURE PATH - Verification failed")
        
        failed_mirrors = []
        if not s3_verified:
            failed_mirrors.append("s3")
        if not minio_verified:
            failed_mirrors.append("minio")
        
        # Log UPLOAD_FAILED event to Arweave FIRST
        upload_failed_payload = {
            "lead_id": event.payload.lead_id,
            "lead_blob_hash": committed_lead_blob_hash,
            "email_hash": committed_email_hash,
            "miner_hotkey": event.actor_hotkey,
            "failed_mirrors": failed_mirrors,
            "reason": "Hash mismatch or blob not found",
            "timestamp": datetime.now(tz.utc).isoformat()
        }
        
        upload_failed_log_entry = {
            "event_type": "UPLOAD_FAILED",
            "actor_hotkey": event.actor_hotkey,
            "nonce": str(uuid.uuid4()),  # Generate fresh UUID for this event
            "ts": datetime.now(tz.utc).isoformat(),
            "payload_hash": hashlib.sha256(
                json.dumps(upload_failed_payload, sort_keys=True).encode()
            ).hexdigest(),
            "build_id": event.build_id,
            "signature": event.signature,
            "payload": upload_failed_payload
        }
        
        try:
            from gateway.utils.logger import log_event
            result = await log_event(upload_failed_log_entry)
            
            upload_failed_tee_seq = result.get("sequence")
            print(f"   ❌ UPLOAD_FAILED event buffered in TEE: seq={upload_failed_tee_seq}")
        except Exception as e:
            print(f"   ⚠️  Error logging UPLOAD_FAILED: {e} (continuing with error response)")
        
        # Increment rate limit counter (FAILURE - verification failed)
        from gateway.utils.rate_limiter import increment_submission
        updated_stats = increment_submission(event.actor_hotkey, success=False)
        print(f"   📊 Rate limit updated: submissions={updated_stats['submissions']}/10, rejections={updated_stats['rejections']}/10")
        
        raise HTTPException(
            status_code=400,
            detail=f"Upload verification failed for mirrors: {', '.join(failed_mirrors)}"
        )

