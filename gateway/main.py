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
from gateway.api import epoch, validate, reveal, manifest, submit, attest, weights, attestation

# Import background tasks
from gateway.tasks.reveal_collector import reveal_collector_task
from gateway.tasks.checkpoints import checkpoint_task
from gateway.tasks.anchor import daily_anchor_task
from gateway.tasks.hourly_batch import start_hourly_batch_task

# Import epoch monitor (polling-based, like validator)
from gateway.tasks.epoch_monitor import EpochMonitor

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
    - Polling-based epoch management (proven stable, like validator)
    - Zero memory leaks (async context manager handles cleanup)
    - Bulletproof: No WebSocket subscriptions = No WebSocket failures
    """
    
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
    
    # Create async subtensor with timeout + retry (handles network delays)
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30
    async_subtensor = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🔄 Attempt {attempt}/{MAX_RETRIES}: Connecting to {BITTENSOR_NETWORK}...")
            
            # Wrap AsyncSubtensor creation in timeout (prevents infinite hang)
            async_subtensor = await asyncio.wait_for(
                asyncio.create_task(bt.AsyncSubtensor(network=BITTENSOR_NETWORK).__aenter__()),
                timeout=TIMEOUT_SECONDS
            )
            
            print("✅ AsyncSubtensor created (WebSocket active)")
            print(f"   Endpoint: {async_subtensor.chain_endpoint}")
            print(f"   Connected on attempt {attempt}")
            print("")
            break  # Success - exit retry loop
            
        except asyncio.TimeoutError:
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES}: Connection timeout after {TIMEOUT_SECONDS}s")
            if attempt < MAX_RETRIES:
                wait_time = 5 * attempt  # Progressive backoff: 5s, 10s, 15s
                print(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ FATAL: Failed to connect to {BITTENSOR_NETWORK} after {MAX_RETRIES} attempts")
                print(f"   Check network connectivity and Bittensor chain status")
                raise RuntimeError(f"AsyncSubtensor connection failed after {MAX_RETRIES} attempts")
                
        except Exception as e:
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES}: Connection error: {e}")
            if attempt < MAX_RETRIES:
                wait_time = 5 * attempt
                print(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ FATAL: Failed to initialize AsyncSubtensor: {e}")
                raise
    
    # Now use async_subtensor in a try/finally to ensure cleanup
    try:
        
        # ════════════════════════════════════════════════════════════════
        # EPOCH MONITOR: Polling-based (like validator - proven stable)
        # ════════════════════════════════════════════════════════════════
        print("="*80)
        print("🔄 INITIALIZING EPOCH MONITOR")
        print("="*80)
        print(f"   Architecture: Polling (same as validator)")
        print(f"   Benefits: Bulletproof - no WebSocket failures")
        print("="*80 + "\n")
        
        # Create epoch monitor
        from gateway.config import BITTENSOR_NETWORK
        epoch_monitor = EpochMonitor(network=BITTENSOR_NETWORK)
        print("✅ EpochMonitor created (replaces event-driven version)")
        print("")
        
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
        print("✅ Async subtensor stored in app.state")
        print("")
        
        # ════════════════════════════════════════════════════════════════
        # BACKGROUND TASKS: Start all services
        # ════════════════════════════════════════════════════════════════
        print("="*80)
        print("🚀 STARTING BACKGROUND TASKS")
        print("="*80)
        
        # Start epoch monitor (polling loop - bulletproof)
        epoch_monitor_task = asyncio.create_task(epoch_monitor.start())
        print("✅ Epoch monitor started (polling mode)")
        
        # Start other background tasks
        reveal_task = asyncio.create_task(reveal_collector_task())
        print("✅ Reveal collector task started")
        
        checkpoint_task_handle = asyncio.create_task(checkpoint_task())
        print("✅ Checkpoint task started")
        
        anchor_task = asyncio.create_task(daily_anchor_task())
        print("✅ Anchor task started")
        
        hourly_batch_task_handle = asyncio.create_task(start_hourly_batch_task())
        print("✅ Hourly Arweave batch task started")
        
        from gateway.utils.rate_limiter import rate_limiter_cleanup_task
        rate_limiter_task = asyncio.create_task(rate_limiter_cleanup_task())
        print("✅ Rate limiter cleanup task started")
        
        # Start PCR0 builder for trustless verification
        from gateway.utils.pcr0_builder import start_pcr0_builder
        start_pcr0_builder()
        print("✅ PCR0 builder started (trustless validator verification)")
        
        print("")
        print("🎯 ARCHITECTURE SUMMARY:")
        print("   • Single AsyncSubtensor (no memory leaks)")
        print("   • Polling-based epoch monitor (same as validator)")
        print("   • Bulletproof: No WebSocket = No WebSocket failures")
        print("   • Proven stable: Validator uses polling for months")
        print("   • PCR0 builder: Computes expected PCR0 from GitHub (trustless)")
        print("="*80 + "\n")
        
        # Yield control back to FastAPI (app runs here)
        yield
    
    finally:
        # ════════════════════════════════════════════════════════════════
        # CLEANUP: Graceful shutdown
        # ════════════════════════════════════════════════════════════════
        print("\n" + "="*80)
        print("🛑 SHUTTING DOWN GATEWAY")
        print("="*80)
        
        # Cancel all background tasks
        print("   🛑 Cancelling background tasks...")
        tasks = [
            epoch_monitor_task,
            reveal_task,
            checkpoint_task_handle,
            anchor_task,
            hourly_batch_task_handle,
            rate_limiter_task
        ]
        
        for task in tasks:
            if task:  # Check if task exists
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
        
        # Close AsyncSubtensor WebSocket manually (since we used __aenter__ with timeout)
        print("   🔌 Closing AsyncSubtensor WebSocket...")
        if async_subtensor:
            try:
                await async_subtensor.__aexit__(None, None, None)
                print("   ✅ AsyncSubtensor closed")
            except Exception as e:
                print(f"   ⚠️  Error closing AsyncSubtensor: {e}")
        
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

# ============================================================
# Request Priority Middleware (Validator > Miner)
# ============================================================
# Prioritize validator requests (/epoch/, /validate) over miner requests (/presign, /submit)
# This prevents validators from timing out during high miner submission traffic.
# 
# Configuration:
# - max_concurrent_miners: Max concurrent miner requests (default: 20)
#   * Lower = more aggressive throttling (better validator protection)
#   * Higher = less throttling (more miner throughput)
#   * Recommended: 15-25 based on your Supabase pool size (15) and max connections (200)
#
# Safe to deploy: Only adds async waiting, no logic changes.

from gateway.middleware.priority import PriorityMiddleware

app.add_middleware(
    PriorityMiddleware,
    max_concurrent_miners=40  # Micro compute can handle higher throughput
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
app.include_router(attest.router)  # TEE attestation endpoint (legacy /attest)
app.include_router(attestation.router)  # TEE attestation endpoint (/attestation/document, /attestation/pubkey)
app.include_router(weights.router)  # Weights submission for auditor validators

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
    Generate presigned PUT URL for miner submission to S3.
    
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
        PresignedURLResponse with S3 URL
    
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
    
    print("✅ /presign SUCCESS - returning S3 presigned URL")
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

