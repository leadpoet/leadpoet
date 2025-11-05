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
from gateway.api import epoch, validate, reveal, manifest

# Import background tasks
from gateway.tasks.epoch_lifecycle import epoch_lifecycle_task
from gateway.tasks.reveal_collector import reveal_collector_task
from gateway.tasks.checkpoints import checkpoint_task
from gateway.tasks.anchor import daily_anchor_task
from gateway.tasks.mirror_monitor import mirror_integrity_task

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Lifespan Context Manager (for background tasks)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    
    Starts background tasks on startup and ensures they run
    for the lifetime of the application.
    """
    print("\n" + "="*80)
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
    
    print("="*80 + "\n")
    
    # Yield control back to FastAPI (app runs here)
    yield
    
    # Cleanup on shutdown (cancel all background tasks)
    print("\n🛑 Shutting down background tasks...")
    tasks = [epoch_task, reveal_task, checkpoint_task_handle, anchor_task, mirror_task]
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

# ============================================================
# Include API Routers
# ============================================================

app.include_router(epoch.router)
app.include_router(validate.router)
app.include_router(reveal.router)
app.include_router(manifest.router)

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
    Generate presigned PUT URLs for miner submission.
    
    Flow:
    1. Verify payload hash
    2. Verify wallet signature
    3. Check actor is registered miner
    4. Verify nonce is fresh
    5. Verify timestamp within tolerance
    6. Generate presigned URLs for S3 + MinIO
    7. Log SUBMISSION_REQUEST to transparency log
    8. Return URLs
    
    Args:
        event: SubmissionRequestEvent with signature
    
    Returns:
        PresignedURLResponse with S3 and MinIO URLs
    
    Raises:
        HTTPException: 400 (bad request), 403 (forbidden)
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
    # Step 3: Check actor is registered miner
    # ========================================
    # Run blocking Bittensor call in thread to avoid blocking event loop
    is_registered, role = await asyncio.to_thread(is_registered_hotkey, event.actor_hotkey)
    
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
    # Step 6: Generate presigned URLs
    # ========================================
    try:
        urls = generate_presigned_put_urls(event.payload.cid)
    except Exception as e:
        print(f"❌ Error generating presigned URLs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate presigned URLs"
        )
    
    # ========================================
    # Step 7: Log SUBMISSION_REQUEST to transparency log
    # ========================================
    try:
        log_entry = {
            "event_type": event.event_type,
            "actor_hotkey": event.actor_hotkey,
            "nonce": event.nonce,
            "ts": event.ts.isoformat(),
            "payload_hash": event.payload_hash,
            "build_id": event.build_id,
            "signature": event.signature,
            "payload": event.payload.model_dump()
        }
        
        supabase.table("transparency_log").insert(log_entry).execute()
        
        print(f"✅ SUBMISSION_REQUEST logged: {event.payload.lead_id} from {event.actor_hotkey[:20]}...")
    
    except Exception as e:
        print(f"❌ Error logging to transparency_log: {e}")
        # Continue anyway - presigned URLs are valid
    
    # ========================================
    # Step 8: Return presigned URLs
    # ========================================
    return PresignedURLResponse(
        s3_url=urls["s3_url"],
        minio_url=urls["minio_url"],
        expires_in=urls["expires_in"]
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

