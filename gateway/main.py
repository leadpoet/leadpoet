"""
LeadPoet Trustless Gateway
=========================

Open-source FastAPI gateway for trustless lead validation.

Endpoints:
- GET /: Health check + build info
- POST /presign: Generate presigned URLs for miner submission
- GET /health: Kubernetes health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import configuration
from gateway.config import BUILD_ID, GITHUB_COMMIT, TIMESTAMP_TOLERANCE_SECONDS
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Import models
from gateway.models.events import SubmissionRequestEvent, EventType
from gateway.models.responses import PresignedURLResponse, ErrorResponse, HealthResponse

# Import utilities
from gateway.utils.signature import verify_wallet_signature, compute_payload_hash, construct_signed_message
from gateway.utils.registry import is_registered_hotkey
from gateway.utils.nonce import check_and_store_nonce, validate_nonce_format
from gateway.utils.storage import generate_presigned_put_urls

# Import Supabase
from supabase import create_client, Client

# Import API routers
from gateway.api import epoch, validate, reveal, manifest, submit, attest

# Import background tasks
from gateway.tasks.epoch_lifecycle import epoch_lifecycle_task
from gateway.tasks.reveal_collector import reveal_collector_task
from gateway.tasks.checkpoints import checkpoint_task
from gateway.tasks.anchor import daily_anchor_task
from gateway.tasks.mirror_monitor import mirror_integrity_task
from gateway.tasks.hourly_batch import start_hourly_batch_task

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Lifespan Context Manager (for background tasks)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    
    Initializes MinIO bucket and starts background tasks on startup.
    """
    # Initialize MinIO bucket (must happen before background tasks)
    print("\n" + "="*80)
    print("🔧 INITIALIZING MINIO")
    print("="*80)
    from gateway.init_minio import init_minio_bucket
    # Run synchronous MinIO init in thread pool to avoid blocking event loop
    import asyncio
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, init_minio_bucket)
    if not success:
        print("⚠️  WARNING: MinIO bucket initialization failed")
        print("   Presigned URL generation may fail until bucket is created")
    print("="*80 + "\n")
    
    # Load gateway keypair for signed receipts
    print("="*80)
    print("🔐 LOADING GATEWAY KEYPAIR")
    print("="*80)
    try:
        from gateway.utils.keys import load_gateway_keypair
        success = load_gateway_keypair()
        if success:
            print("✅ Gateway keypair loaded successfully")
            print("✅ Signed receipts ENABLED")
        else:
            print("⚠️  WARNING: Gateway keypair failed to load")
            print("   Signed receipts will be DISABLED")
            print("   Receipts will not include gateway_signature field")
    except Exception as e:
        print(f"❌ ERROR loading gateway keypair: {e}")
        print("   Signed receipts will be DISABLED")
        print("   Gateway will continue but cannot sign receipts")
    print("="*80 + "\n")
    
    print("="*80)
    print("🚀 STARTING BACKGROUND TASKS")
    print("="*80)
    
    # Start all background tasks
    epoch_task = asyncio.create_task(epoch_lifecycle_task())
    print("✅ Epoch lifecycle task started")
    
    reveal_task = asyncio.create_task(reveal_collector_task())
    print("✅ Reveal collector task started")
    
    checkpoint_task_handle = asyncio.create_task(checkpoint_task())
    print("✅ Checkpoint task started")
    
    anchor_task = asyncio.create_task(daily_anchor_task())
    print("✅ Anchor task started")
    
    mirror_task = asyncio.create_task(mirror_integrity_task())
    print("✅ Mirror monitor task started")
    
    hourly_batch_task_handle = asyncio.create_task(start_hourly_batch_task())
    print("✅ Hourly Arweave batch task started")
    
    # Start rate limiter cleanup task
    from gateway.utils.rate_limiter import rate_limiter_cleanup_task
    rate_limiter_task = asyncio.create_task(rate_limiter_cleanup_task())
    print("✅ Rate limiter cleanup task started")
    
    print("="*80 + "\n")
    
    # Yield control back to FastAPI (app runs here)
    yield
    
    # Cleanup on shutdown (cancel all background tasks)
    print("\n🛑 Shutting down background tasks...")
    tasks = [epoch_task, reveal_task, checkpoint_task_handle, anchor_task, mirror_task, hourly_batch_task_handle, rate_limiter_task]
    for task in tasks:
        task.cancel()
    
    # Wait for all tasks to finish
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    print("✅ Background tasks stopped\n")

# ============================================================
# Create FastAPI App
# ============================================================

app = FastAPI(
    title="LeadPoet Trustless Gateway",
    description="Open-source, reproducible gateway for lead validation",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan context manager
)

# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Debug middleware to log all requests
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"🔍 INCOMING REQUEST: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"🔍 RESPONSE STATUS: {response.status_code}")
    return response

# ============================================================
# Include API Routers
# ============================================================

app.include_router(epoch.router)
app.include_router(validate.router)  # Individual + Batch validation
app.include_router(reveal.router)
app.include_router(manifest.router)
app.include_router(submit.router)
app.include_router(attest.router)  # TEE attestation endpoint

# ============================================================
# Health Check Endpoints
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Health check + build info.
    
    Returns gateway status, build ID, and commit hash for reproducibility.
    """
    return HealthResponse(
        service="leadpoet-gateway",
        status="ok",
        build_id=BUILD_ID,
        github_commit=GITHUB_COMMIT,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health")
async def health():
    """
    Kubernetes health check.
    
    Simple endpoint for container orchestration health probes.
    """
    return {"status": "healthy"}


# ============================================================
# Miner Submission Flow
# ============================================================

@app.post("/presign", response_model=PresignedURLResponse)
async def presign_urls(event: SubmissionRequestEvent):
    """
    Generate presigned PUT URL for miner submission (S3 only).
    
    Gateway automatically mirrors to MinIO after S3 upload verification.
    
    Flow:
    1. Verify payload hash
    2. Verify wallet signature
    3. Check actor is registered miner
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    6. Generate presigned URL for S3
    7. Log SUBMISSION_REQUEST to transparency log
    8. Return S3 URL
    
    Args:
        event: SubmissionRequestEvent with signature
    
    Returns:
        PresignedURLResponse with S3 URL (MinIO mirroring happens gateway-side)
    
    Raises:
        HTTPException: 400 (bad request), 403 (forbidden)
    """
    print("🔍 /presign called - START")
    
    # ========================================
    # Step 0: Verify Proof-of-Work (Anti-DDoS)
    # ========================================
    # This is the FIRST line of defense against DDoS attacks.
    # PoW verification is O(1) (single SHA256 hash) but REQUIRES the attacker
    # to compute ~65k hashes per request, making billion-request attacks
    # economically impossible (~$50k/hour for 1M req/sec).
    #
    # This runs BEFORE signature verification or any I/O.
    print("🔍 Step 0: Verifying Proof-of-Work...")
    from gateway.utils.pow import verify_pow
    import uuid
    import hashlib
    import json
    
    # Extract PoW fields from request body
    pow_challenge = event.payload.lead_id  # Challenge = lead ID
    pow_timestamp = event.pow_timestamp
    pow_nonce = event.pow_nonce
    
    pow_valid, pow_error = verify_pow(pow_challenge, pow_timestamp, pow_nonce)
    if not pow_valid:
        print(f"❌ PoW invalid: {pow_error}")
        
        # Log POW_FAILED event to TEE buffer (for transparency)
        try:
            from gateway.utils.logger import log_event
            
            pow_failed_event = {
                "event_type": "POW_FAILED",
                "actor_hotkey": event.actor_hotkey,
                "nonce": str(uuid.uuid4()),
                "ts": datetime.utcnow().isoformat(),
                "payload_hash": hashlib.sha256(json.dumps({
                    "lead_id": event.payload.lead_id,
                    "reason": pow_error,
                    "pow_timestamp": pow_timestamp,
                    "pow_nonce": pow_nonce
                }, sort_keys=True).encode()).hexdigest(),
                "build_id": "gateway",
                "signature": "pow_check",
                "payload": {
                    "lead_id": event.payload.lead_id,
                    "reason": pow_error,
                    "miner_hotkey": event.actor_hotkey
                }
            }
            
            await log_event(pow_failed_event)
            print(f"   ✅ Logged POW_FAILED to TEE buffer")
        except Exception as e:
            print(f"   ⚠️  Failed to log POW_FAILED: {e}")
        
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_proof_of_work",
                "message": pow_error,
                "hint": "Compute SHA256(lead_id:timestamp:nonce) with 4 leading zeros"
            }
        )
    
    print(f"✅ PoW valid (timestamp: {pow_timestamp}, nonce: {pow_nonce})")
    
    # ========================================
    # Step 1: Verify payload hash
    # ========================================
    print("🔍 Step 1: Computing payload hash...")
    computed_hash = compute_payload_hash(event.payload.model_dump())
    if computed_hash != event.payload_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Payload hash mismatch: expected {event.payload_hash[:16]}..., got {computed_hash[:16]}..."
        )
    
    # ========================================
    # Step 2: Verify wallet signature
    # ========================================
    print("🔍 Step 2: Verifying signature...")
    message = construct_signed_message(event)
    print(f"🔍 Message constructed for verification: {message[:150]}...")
    print(f"🔍 Signature received: {event.signature[:64]}...")
    print(f"🔍 Actor hotkey: {event.actor_hotkey}")
    
    is_valid = verify_wallet_signature(message, event.signature, event.actor_hotkey)
    print(f"🔍 Signature valid: {is_valid}")
    
    if not is_valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # ========================================
    # Step 3: Check actor is registered miner
    # ========================================
    # Run blocking Bittensor call in thread to avoid blocking event loop
    print("🔍 Step 3: Checking registration (in thread)...")
    is_registered, role = await asyncio.to_thread(is_registered_hotkey, event.actor_hotkey)
    print(f"🔍 Step 3 complete: is_registered={is_registered}, role={role}")
    
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
    try:
        # Use timezone-aware datetime for comparison
        from datetime import timezone as tz
        now = datetime.now(tz.utc)
        
        # Make event.ts timezone-aware if it's naive
        event_ts = event.ts if event.ts.tzinfo else event.ts.replace(tzinfo=tz.utc)
        
        time_diff = abs((now - event_ts).total_seconds())
        print(f"🔍 Timestamp check: now={now.isoformat()}, event={event_ts.isoformat()}, diff={time_diff:.2f}s")
        
        if time_diff > TIMESTAMP_TOLERANCE_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Timestamp out of range: {time_diff:.0f}s (max: {TIMESTAMP_TOLERANCE_SECONDS}s)"
            )
        print(f"🔍 Step 5 complete: Timestamp valid (diff={time_diff:.2f}s)")
    except Exception as e:
        print(f"❌ Timestamp verification error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Timestamp verification failed: {str(e)}"
        )
    
    # ========================================
    # Step 6: Generate presigned URLs
    # ========================================
    print(f"🔍 Step 6: Generating presigned URLs for lead_id={event.payload.lead_id}...")
    print(f"   Using lead_blob_hash as S3 key: {event.payload.lead_blob_hash[:16]}...")
    try:
        # Use lead_blob_hash as the S3 object key (content-addressed storage)
        urls = generate_presigned_put_urls(event.payload.lead_blob_hash)
    except Exception as e:
        print(f"❌ Error generating presigned URLs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate presigned URLs"
        )
    print(f"🔍 Step 6 complete: URLs generated")
    
    # ========================================
    # Step 7: Log SUBMISSION_REQUEST to TEE Buffer (CRITICAL: Hardware-Protected)
    # ========================================
    print("🔍 Step 7: Logging SUBMISSION_REQUEST to TEE buffer...")
    arweave_tx_id = None  # Will be available after hourly Arweave batch
    tee_sequence = None
    
    try:
        from gateway.utils.logger import log_event
        
        log_entry = {
            "event_type": event.event_type.value,  # Convert enum to string
            "actor_hotkey": event.actor_hotkey,
            "nonce": event.nonce,
            "ts": event.ts.isoformat(),
            "payload_hash": event.payload_hash,
            "build_id": event.build_id,
            "signature": event.signature,
            "payload": event.payload.model_dump()
        }
        
        # Write to TEE buffer (authoritative, hardware-protected)
        # TEE will batch to Arweave hourly
        result = await log_event(log_entry)
        
        tee_sequence = result.get("sequence")
        buffer_size = result.get("buffer_size", 0)
        
        print(f"✅ Step 7 complete: SUBMISSION_REQUEST buffered in TEE")
        print(f"   TEE sequence: {tee_sequence}")
        print(f"   Buffer size: {buffer_size} events")
        print(f"   ⏰ Will batch to Arweave in next hourly checkpoint")
    
    except Exception as e:
        # CRITICAL: If TEE buffer write fails, request MUST fail
        # This prevents censorship (cannot accept event and then drop it)
        print(f"❌ Error logging to TEE buffer: {e}")
        import traceback
        traceback.print_exc()
        print(f"🚨 CRITICAL: TEE buffer unavailable - failing request")
        print(f"   Operator: Check TEE enclave health: sudo nitro-cli describe-enclaves")
        raise HTTPException(
            status_code=503,
            detail=f"TEE buffer unavailable: {str(e)}. Gateway cannot accept events."
        )
    
    # ========================================
    # Step 8: Return presigned URL + Acknowledgment
    # ========================================
    # NOTE (Phase 4): TEE-based trust model
    # - Event is buffered in TEE (hardware-protected, sequence={tee_sequence})
    # - Will be included in next hourly Arweave checkpoint (signed by TEE)
    # - Verify gateway code integrity: GET /attest
    request_timestamp = datetime.now(tz.utc).isoformat()
    
    print("✅ /presign SUCCESS - returning S3 presigned URL (MinIO will be mirrored by gateway)")
    return PresignedURLResponse(
        lead_id=event.payload.lead_id,
        presigned_url=urls["s3_url"],  # Miner uploads to S3
        s3_url=urls["s3_url"],  # Alias for backward compatibility
        expires_in=urls["expires_in"],
        timestamp=request_timestamp  # ISO 8601 timestamp
    )


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting LeadPoet Trustless Gateway")
    print("=" * 60)
    print(f"Build ID: {BUILD_ID}")
    print(f"GitHub Commit: {GITHUB_COMMIT}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

