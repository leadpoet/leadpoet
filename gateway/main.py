"""
LeadPoet Arena Gateway
======================

FastAPI gateway for Arena proxying, epoch reads, and measured runtime identity.

Endpoints:
- GET /: Health check + build info
- GET /health: Kubernetes health check
- GET /attestation/document: Measured Nitro runtime identity
- /arena/*: Arena service proxy
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import os
import asyncio

# Add import roots for the gateway package and attested enclave dependencies.
_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(_GATEWAY_DIR)
_ATTESTED_RUNTIME_DIR = os.path.join(_GATEWAY_DIR, "_attested_runtime")
for _path in (_ATTESTED_RUNTIME_DIR, _PACKAGE_PARENT):
    if not os.path.isdir(_path):
        continue
    while _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

# Opt-in, fail-closed error monitoring (docs/sentry_error_monitoring.md).
# Wired before the config/service imports so an import-time crash of the
# gateway host process is still captured. Complete no-op unless the
# LEADPOET_SENTRY_* environment gate is satisfied.
try:
    from leadpoet_observability import (
        capture_failure as _capture_sentry_failure,
        configure_sentry_context as _configure_sentry_context,
        init_sentry,
        record_retry as _record_sentry_retry,
        record_stage as _record_sentry_stage,
    )

    init_sentry(component="gateway")
except Exception as _sentry_exc:  # error monitoring must never break startup
    _capture_sentry_failure = lambda *args, **kwargs: False
    _configure_sentry_context = lambda *args, **kwargs: {}
    _record_sentry_retry = lambda *args, **kwargs: None
    _record_sentry_stage = lambda *args, **kwargs: None
    print(
        "leadpoet_sentry_wiring_skipped error=%s" % type(_sentry_exc).__name__,
        flush=True,
    )

# Import configuration
from gateway.build_info import get_build_info
from gateway.config import BUILD_ID, GITHUB_COMMIT

_configure_sentry_context(
    component="gateway",
    physical_role="gateway-coordinator",
    runtime_sha=GITHUB_COMMIT,
    restart_invocation_id=os.environ.get("LEADPOET_RESTART_INVOCATION_ID"),
)

# Import models
from gateway.models.responses import HealthResponse

# Import API routers
from gateway.api import epoch, attestation
from gateway.api.arena_proxy import router as arena_proxy_router
from gateway.api.arena_proxy import testnet_router as arena_testnet_proxy_router
from gateway.api import metrics as metrics_api

# Import background tasks
from gateway.tasks.icp_generator import icp_rotation_task, ensure_icp_set_exists

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
    
    # ════════════════════════════════════════════════════════════════
    # COORDINATOR-ENCLAVE IDENTITY INITIALIZATION
    # ════════════════════════════════════════════════════════════════
    print("="*80)
    print("INITIALIZING COORDINATOR-ENCLAVE IDENTITY")
    print("="*80)
    try:
        from gateway.utils.logger import initialize_enclave_identity

        enclave_identity = await initialize_enclave_identity()
        enclave_pubkey = str(enclave_identity["enclave_pubkey"])
        print("Coordinator-enclave identity initialized")
        print(f"   Pubkey: {enclave_pubkey[:32]}...")
        print("Nitro attestation is available for runtime identity verification")
    except Exception as e:
        _capture_sentry_failure(
            "runtime.enclave_relay_unavailable",
            component="gateway",
            stage="coordinator_event_signer_startup",
            exception=e,
            terminal=True,
            retryable=False,
            fail_closed=True,
            runtime_sha=GITHUB_COMMIT,
            restart_invocation_id=os.environ.get(
                "LEADPOET_RESTART_INVOCATION_ID"
            ),
        )
        print(f"CRITICAL ERROR initializing coordinator identity: {e}")
        print("   Refusing gateway startup without Nitro-backed identity")
        raise RuntimeError("coordinator-enclave identity initialization failed") from e
    print("="*80 + "\n")
    
    # Start the event-loop stall watchdog first, so even a hang later in
    # startup (or any future loop-blocking bug) logs CRITICAL + full thread
    # stacks instead of silently freezing every endpoint.
    from gateway.utils.loop_watchdog import start_loop_watchdog
    start_loop_watchdog(asyncio.get_running_loop())

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
            _record_sentry_retry(
                "authority.dependency_unreadable",
                component="gateway",
                stage="chain_runtime_startup",
                attempt=attempt,
                attempts=MAX_RETRIES,
                dependency="finney",
                runtime_sha=GITHUB_COMMIT,
            )
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES}: Connection timeout after {TIMEOUT_SECONDS}s")
            if attempt < MAX_RETRIES:
                wait_time = 5 * attempt  # Progressive backoff: 5s, 10s, 15s
                print(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                _capture_sentry_failure(
                    "authority.dependency_unreadable",
                    component="gateway",
                    stage="chain_runtime_startup",
                    terminal=True,
                    retryable=True,
                    fail_closed=True,
                    attempts=MAX_RETRIES,
                    dependency="finney",
                    timed_out=True,
                    runtime_sha=GITHUB_COMMIT,
                )
                print(f"❌ FATAL: Failed to connect to {BITTENSOR_NETWORK} after {MAX_RETRIES} attempts")
                print(f"   Check network connectivity and Bittensor chain status")
                raise RuntimeError(f"AsyncSubtensor connection failed after {MAX_RETRIES} attempts")
                
        except Exception as e:
            _record_sentry_retry(
                "authority.dependency_unreadable",
                component="gateway",
                stage="chain_runtime_startup",
                attempt=attempt,
                attempts=MAX_RETRIES,
                dependency="finney",
                exception_class=type(e).__name__,
                runtime_sha=GITHUB_COMMIT,
            )
            print(f"⚠️  Attempt {attempt}/{MAX_RETRIES}: Connection error: {e}")
            if attempt < MAX_RETRIES:
                wait_time = 5 * attempt
                print(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                _capture_sentry_failure(
                    "authority.dependency_unreadable",
                    component="gateway",
                    stage="chain_runtime_startup",
                    exception=e,
                    terminal=True,
                    retryable=True,
                    fail_closed=True,
                    attempts=MAX_RETRIES,
                    dependency="finney",
                    runtime_sha=GITHUB_COMMIT,
                )
                print(f"❌ FATAL: Failed to initialize AsyncSubtensor: {e}")
                raise
    
    # ════════════════════════════════════════════════════════════════
    # CRYPTO PREFLIGHT: prove the lazily-imported verification stack
    # works in THIS runtime before serving. The binding verifier fails
    # closed on ImportError, so a missing dependency otherwise surfaces
    # only as 403 "invalid hotkey binding" at the weight-submission
    # window (epoch 23929). Fail startup loudly instead.
    # ════════════════════════════════════════════════════════════════
    try:
        from bittensor_wallet import Keypair as _PreflightKeypair
        _probe_kp = _PreflightKeypair.create_from_uri("//gateway-startup-preflight")
        if not _probe_kp.verify(b"preflight", _probe_kp.sign(b"preflight")):
            raise RuntimeError("sr25519 sign/verify round-trip returned False")
        from leadpoet_canonical.nitro import verify_nitro_attestation_full as _preflight_nitro  # noqa: F401
        print("✅ Crypto preflight passed (sr25519 + nitro verifier importable)")
    except Exception as exc:
        print(f"❌ FATAL: crypto verification preflight failed: {exc}")
        print("   Weight submissions would be silently rejected — refusing to start.")
        raise

    # Initialize all task handles before try block to prevent NameError in finally
    icp_task = None

    # Now use async_subtensor in a try/finally to ensure cleanup
    try:
        
        # ════════════════════════════════════════════════════════════════
        # EPOCH MONITOR: Polling-based (like validator - proven stable)
        # ════════════════════════════════════════════════════════════════
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
        
        # ════════════════════════════════════════════════════════════════
        # ARENA ICP SET INITIALIZATION
        # This private daily set is the current Arena benchmark input.
        # TESTNET GUARD: Skip on testnet to prevent writing to production
        # qualification_private_icp_sets (testnet and mainnet share same Supabase)
        # ════════════════════════════════════════════════════════════════
        from gateway.config import BITTENSOR_NETWORK
        if BITTENSOR_NETWORK == "test":
            print("⚠️  TESTNET MODE: Skipping ICP set initialization (protect production qualification_private_icp_sets)")
            print("⚠️  TESTNET MODE: Skipping ICP rotation task")
        else:
            try:
                await ensure_icp_set_exists()
                print("✅ ICP set initialized (benchmark ICPs ready)")
            except Exception as e:
                print(f"⚠️  Failed to initialize ICP set: {e}")
                print("   Arena benchmark creation may not work!")
            icp_task = asyncio.create_task(icp_rotation_task())
            print("✅ Arena ICP rotation task started")

        app.state.enclave_identity = dict(enclave_identity)
        
        print("")
        print("🎯 ARCHITECTURE SUMMARY:")
        print("   • Single AsyncSubtensor (no memory leaks)")
        print("   • Bulletproof: No WebSocket = No WebSocket failures")
        print("   • Proven stable: Validator uses polling for months")
        print("   • Agent competition: operated by the independent Arena service")
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
        tasks = [icp_task]
        
        # The task is absent on testnet.
        active_tasks = [t for t in tasks if t is not None]
        
        for task in active_tasks:
                task.cancel()
        
        # Wait for all tasks to finish gracefully
        print("   ⏳ Waiting for tasks to finish...")
        if active_tasks:
            results = await asyncio.gather(*active_tasks, return_exceptions=True)
        else:
            results = []
            print("   (No background tasks were running)")
        
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
    title="LeadPoet Arena Gateway",
    description="Gateway for Arena proxying, epoch reads, and runtime identity",
    version="1.0.0",
    lifespan=lifespan,  # Use lifespan context manager
    redirect_slashes=False,  # Prevent 307 redirects from consuming semaphore slots
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
# Request Body Guard
# ============================================================

from gateway.middleware.body_size import BodySizeLimitMiddleware

app.add_middleware(BodySizeLimitMiddleware)

# ============================================================
# Request Priority Middleware
# ============================================================
# Prioritize epoch and Arena result traffic during concurrent Arena submissions.
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
    max_concurrent_miners=75  # Pool=150, miners=75, leaves 75 for validators/consensus (doubled miners: 128→256 UIDs)
)

# ============================================================
# Optional OpenTelemetry (infra-only, off unless configured)
# ============================================================
# Emits request method/route-template/status/duration only when
# GATEWAY_OTEL_ENABLED, GATEWAY_OTEL_ENDPOINT, and GATEWAY_OTEL_TOKEN are
# all set. Never
# captures bodies, query strings, DB statements, or LLM/training content, and
# never touches the enclaves. No-op by default.
from gateway.observability.otel_bootstrap import configure_gateway_otel

configure_gateway_otel(app)

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
# NOTE: reveal.router REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
# Legacy lead intake and validation are retired. Arena owns current work.
app.include_router(attestation.router)  # TEE attestation endpoint (/attestation/document, /attestation/pubkey)
app.include_router(metrics_api.router)

app.include_router(arena_proxy_router)
app.include_router(arena_testnet_proxy_router)

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
        timestamp=datetime.utcnow().isoformat(),
        build_info=get_build_info(),
    )


@app.get("/build-info")
async def build_info():
    """
    Runtime build provenance.

    This is the canonical operator endpoint for checking which source commit is
    live. Deployments should ship gateway/BUILD_INFO.json; env and local git are
    fallback sources for CI and development.
    """
    return get_build_info()


@app.get("/health")
async def health():
    """
    Kubernetes health check.
    
    Simple endpoint for container orchestration health probes.
    """
    return {"status": "healthy"}


@app.get("/health/v2-authority")
async def v2_authority_health():
    """Fail-closed readiness for the retained live V2 enclave authority."""
    try:
        from gateway.api.attestation import _runtime_identity
        from gateway.tee.verify_v2_runtime_ready import verify_v2_runtime_ready

        event_identity, enclave_health = await asyncio.gather(
            _runtime_identity(),
            verify_v2_runtime_ready(),
        )
    except Exception as exc:
        _record_sentry_retry(
            "runtime.enclave_relay_unavailable",
            component="gateway",
            stage="gateway_v2_authority_health",
            attempt=1,
            attempts=1,
            exception_class=type(exc).__name__,
            runtime_sha=GITHUB_COMMIT,
            restart_invocation_id=os.environ.get(
                "LEADPOET_RESTART_INVOCATION_ID"
            ),
        )
        raise HTTPException(
            status_code=503,
            detail=f"authoritative V2 runtime is not ready: {type(exc).__name__}",
        ) from exc
    return {
        "schema_version": "leadpoet.gateway_v2_authority_health.v2",
        "status": "ready",
        "commit_sha": GITHUB_COMMIT,
        "event_signer": {
            "purpose": event_identity["purpose"],
            "enclave_pubkey": event_identity["enclave_pubkey"],
            "code_hash": event_identity["code_hash"],
        },
        "enclaves": enclave_health,
    }


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting LeadPoet Trustless Gateway")
    print("=" * 60)
    _build_info = get_build_info()
    print(f"Build ID: {BUILD_ID}")
    print(f"GitHub Commit: {GITHUB_COMMIT}")
    print(f"Commit Source: {_build_info.get('commit_source')}")
    print(f"Build Time UTC: {_build_info.get('build_time_utc')}")
    print(f"Git Branch: {_build_info.get('git_branch')}")
    print(f"Git Dirty: {_build_info.get('git_dirty')}")
    print(f"Build Info File: {_build_info.get('build_info_path') or 'not found'}")
    if not _build_info.get("is_commit_known"):
        print("⚠️  WARNING: gateway commit is unknown. Generate and deploy BUILD_INFO.json.")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        limit_concurrency=int(os.getenv("GATEWAY_UVICORN_LIMIT_CONCURRENCY", "300")),
        backlog=int(os.getenv("GATEWAY_UVICORN_BACKLOG", "2048")),
        timeout_keep_alive=int(os.getenv("GATEWAY_UVICORN_KEEPALIVE_SECONDS", "5")),
    )
