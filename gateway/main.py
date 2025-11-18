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
from gateway.tasks.reveal_collector import reveal_collector_task
from gateway.tasks.checkpoints import checkpoint_task
from gateway.tasks.anchor import daily_anchor_task
from gateway.tasks.mirror_monitor import mirror_integrity_task
from gateway.tasks.hourly_batch import start_hourly_batch_task

# Import new event-driven monitors (replace polling tasks)
from gateway.tasks.epoch_monitor import EpochMonitor
from gateway.tasks.metagraph_monitor import MetagraphMonitor
from gateway.utils.block_publisher import ChainBlockPublisher

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Lifespan Context Manager (for background tasks)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    
    ASYNC ARCHITECTURE:
    - Creates single AsyncSubtensor instance for entire gateway lifecycle
    - Subscribes to block notifications (push-based, not polling)
    - Event-driven epoch management (no background polling loops)
    - Zero memory leaks (async context manager handles cleanup)
    - Zero HTTP 429 errors (single WebSocket stays alive)
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
    
    # ════════════════════════════════════════════════════════════════
    # ASYNC SUBTENSOR: Create single instance for entire lifecycle
    # ════════════════════════════════════════════════════════════════
    from gateway.config import BITTENSOR_NETWORK
    import bittensor as bt
    
    print("="*80)
    print("🔗 INITIALIZING ASYNC SUBTENSOR")
    print("="*80)
    print(f"   Network: {BITTENSOR_NETWORK}")
    print(f"   Architecture: Single WebSocket for entire lifecycle")
    print(f"   Benefits: Zero memory leaks, zero HTTP 429 errors")
    print("="*80 + "\n")
    
    # Create async subtensor with context manager (auto-cleanup on exit)
    async with bt.AsyncSubtensor(network=BITTENSOR_NETWORK) as async_subtensor:
        print("✅ AsyncSubtensor created (WebSocket active)")
        print(f"   Endpoint: {async_subtensor.chain_endpoint}")
        print("")
        
        # ════════════════════════════════════════════════════════════════
        # BLOCK PUBLISHER: Subscribe to chain blocks (push-based)
        # ════════════════════════════════════════════════════════════════
        print("="*80)
        print("🔔 INITIALIZING BLOCK SUBSCRIPTION")
        print("="*80)
        
        stop_event = asyncio.Event()
        block_publisher = ChainBlockPublisher(
            substrate=async_subtensor.substrate,  # Underlying substrate interface
            stop_event=stop_event
        )
        print("✅ ChainBlockPublisher created")
        
        # Create epoch monitor (replaces epoch_lifecycle_task polling loop)
        epoch_monitor = EpochMonitor()
        block_publisher.add_subscriber(epoch_monitor)
        print("✅ EpochMonitor registered (replaces epoch_lifecycle polling)")
        
        # Create metagraph monitor (replaces metagraph_warmer_task polling loop)
        metagraph_monitor = MetagraphMonitor(async_subtensor)
        block_publisher.add_subscriber(metagraph_monitor)
        print("✅ MetagraphMonitor registered (replaces metagraph_warmer polling)")
        
        print("="*80 + "\n")
        
        # ════════════════════════════════════════════════════════════════
        # DEPENDENCY INJECTION: Inject async_subtensor into modules
        # ════════════════════════════════════════════════════════════════
        print("="*80)
        print("💉 INJECTING ASYNC SUBTENSOR INTO MODULES")
        print("="*80)
        
        from gateway.utils import epoch as epoch_utils
        from gateway.utils import registry as registry_utils
        
        epoch_utils.inject_async_subtensor(async_subtensor)
        print("✅ Injected into gateway.utils.epoch")
        
        registry_utils.inject_async_subtensor(async_subtensor)
        print("✅ Injected into gateway.utils.registry")
        
        print("="*80 + "\n")
        
        # ════════════════════════════════════════════════════════════════
        # APP STATE: Store for request handlers
        # ════════════════════════════════════════════════════════════════
        app.state.async_subtensor = async_subtensor
        app.state.block_publisher = block_publisher
        app.state.stop_event = stop_event
        print("✅ Async subtensor stored in app.state")
        print("")
        
        # ════════════════════════════════════════════════════════════════
        # BACKGROUND TASKS: Start services
        # ════════════════════════════════════════════════════════════════
        print("="*80)
        print("🚀 STARTING BACKGROUND TASKS")
        print("="*80)
        
        # Start block subscription (keeps WebSocket alive, triggers monitors)
        subscription_task = asyncio.create_task(block_publisher.start())
        print("✅ Block subscription started (push-based, replaces polling)")
        
        # Start other background tasks (existing services)
        reveal_task = asyncio.create_task(reveal_collector_task())
        print("✅ Reveal collector task started")
        
        checkpoint_task_handle = asyncio.create_task(checkpoint_task())
        print("✅ Checkpoint task started")
        
        anchor_task = asyncio.create_task(daily_anchor_task())
        print("✅ Anchor task started")
        
        # TEMPORARILY DISABLED: mirror_integrity_task causes event loop deadlock
        # TODO: Fix async/sync event loop conflict in mirror_monitor.py
        # mirror_task = asyncio.create_task(mirror_integrity_task())
        print("⚠️  Mirror monitor task DISABLED (causes deadlock)")
        
        hourly_batch_task_handle = asyncio.create_task(start_hourly_batch_task())
        print("✅ Hourly Arweave batch task started")
        
        # Start rate limiter cleanup task
        from gateway.utils.rate_limiter import rate_limiter_cleanup_task
        rate_limiter_task = asyncio.create_task(rate_limiter_cleanup_task())
        print("✅ Rate limiter cleanup task started")
        
        print("")
        print("🎯 ARCHITECTURE SUMMARY:")
        print("   • Single AsyncSubtensor (no memory leaks)")
        print("   • Block subscription (push-based, no polling)")
        print("   • Event-driven epoch management")
        print("   • Zero HTTP 429 errors (WebSocket stays alive)")
        print("="*80 + "\n")
        
        # Yield control back to FastAPI (app runs here)
        yield
        
        # ════════════════════════════════════════════════════════════════
        # CLEANUP: Graceful shutdown
        # ════════════════════════════════════════════════════════════════
        print("\n" + "="*80)
        print("🛑 SHUTTING DOWN GATEWAY")
        print("="*80)
        
        # Signal stop to block publisher
        print("   🛑 Signaling stop event...")
        stop_event.set()
        
        # Cancel all background tasks
        print("   🛑 Cancelling background tasks...")
        tasks = [
            subscription_task,
            reveal_task,
            checkpoint_task_handle,
            anchor_task,
            # mirror_task,  # DISABLED (see line 203-206)
            hourly_batch_task_handle,
            rate_limiter_task
        ]
        
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to finish gracefully
        print("   ⏳ Waiting for tasks to finish...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors during shutdown
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                print(f"   ⚠️  Task {i} error during shutdown: {result}")
        
        print("   ✅ All background tasks stopped")
        print("")
        
        # AsyncSubtensor will be closed by context manager (async with)
        print("   🔌 Closing AsyncSubtensor WebSocket...")
        # (automatic cleanup by async with context manager)
        
        print("="*80)
        print("✅ GATEWAY SHUTDOWN COMPLETE")
        print("="*80 + "\n")

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

# Production middleware: Only log errors and critical paths
# Comment out request logging to reduce overhead in production
# @app.middleware("http")
# async def log_requests(request, call_next):
#     print(f"🔍 INCOMING REQUEST: {request.method} {request.url.path}")
#     response = await call_next(request)
#     print(f"🔍 RESPONSE STATUS: {response.status_code}")
#     return response

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
    1. Verify wallet signature (MUST be first to prove identity)
    2. Check rate limits (uses verified hotkey, blocks before expensive ops)
    3. Verify payload hash (skip for rate-limited requests)
    4. Check actor is registered miner
    5. Verify nonce is fresh
    6. Verify timestamp within tolerance
    7. Generate presigned URL for S3
    8. Log SUBMISSION_REQUEST to transparency log
    9. Return S3 URL
    
    Args:
        event: SubmissionRequestEvent with signature
    
    Returns:
        PresignedURLResponse with S3 URL (MinIO mirroring happens gateway-side)
    
    Raises:
        HTTPException: 400 (bad request), 403 (forbidden), 429 (rate limited)
    """
    print("🔍 /presign called - START")
    
    # ========================================
    # Step 1: Verify wallet signature (REQUIRED to verify identity)
    # ========================================
    print("🔍 Step 1: Verifying signature...")
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
    print("🔍 Step 1 complete: Signature verified")
    
    # ========================================
    # Step 2: Check rate limits (NOW we have verified identity)
    # ========================================
    print("🔍 Step 2: Checking rate limits...")
    from gateway.utils.rate_limiter import check_rate_limit
    
    allowed, rate_limit_message, _ = check_rate_limit(event.actor_hotkey)
    if not allowed:
        print(f"⚠️  Rate limit exceeded for {event.actor_hotkey[:20]}...")
        print(f"   {rate_limit_message}")
        raise HTTPException(
            status_code=429,
            detail=rate_limit_message
        )
    print("🔍 Step 2 complete: Rate limit OK")
    
    # ========================================
    # Step 3: Verify payload hash
    # ========================================
    print("🔍 Step 3: Computing payload hash...")
    computed_hash = compute_payload_hash(event.payload.model_dump())
    if computed_hash != event.payload_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Payload hash mismatch: expected {event.payload_hash[:16]}..., got {computed_hash[:16]}..."
        )
    print("🔍 Step 3 complete: Payload hash verified")
    
    # ========================================
    # Step 4: Check actor is registered miner
    # ========================================
    # Run blocking Bittensor call in thread to avoid blocking event loop
    print("🔍 Step 4: Checking registration (in thread)...")
    try:
        is_registered, role = await asyncio.wait_for(
            asyncio.to_thread(is_registered_hotkey, event.actor_hotkey),
            timeout=45.0  # 45 second timeout for metagraph query (cache refresh can be slow under load)
        )
        print(f"🔍 Step 4 complete: is_registered={is_registered}, role={role}")
    except asyncio.TimeoutError:
        print(f"❌ Metagraph query timed out after 45s for {event.actor_hotkey[:20]}...")
        raise HTTPException(
            status_code=504,
            detail="Metagraph query timeout - please retry in a moment (cache warming)"
        )
    
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
    # Step 5: Verify nonce format and freshness
    # ========================================
    print("🔍 Step 5: Verifying nonce...")
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
    print("🔍 Step 5 complete: Nonce valid")
    
    # ========================================
    # Step 6: Verify timestamp
    # ========================================
    print("🔍 Step 6: Verifying timestamp...")
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
        print(f"🔍 Step 6 complete: Timestamp valid (diff={time_diff:.2f}s)")
    except Exception as e:
        print(f"❌ Timestamp verification error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Timestamp verification failed: {str(e)}"
        )
    
    # ========================================
    # Step 7: Generate presigned URLs
    # ========================================
    print(f"🔍 Step 7: Generating presigned URLs for lead_id={event.payload.lead_id}...")
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
    print(f"🔍 Step 7 complete: URLs generated")
    
    # ========================================
    # Step 8: Log SUBMISSION_REQUEST to TEE Buffer (CRITICAL: Hardware-Protected)
    # ========================================
    print("🔍 Step 8: Logging SUBMISSION_REQUEST to TEE buffer...")
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
    # Step 9: Return presigned URL + Acknowledgment
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

