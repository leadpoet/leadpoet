"""
Request Priority Middleware for Gateway

Prioritizes validator requests over miner requests to prevent validators
from timing out during high miner submission traffic.

Design:
- Validator paths get immediate processing (no throttling)
- Qualification model paths get immediate processing (rare, shouldn't compete with sourcing)
- Miner sourcing paths are throttled (max concurrent limit)
- Simple, safe, no changes to database logic

Validator Priority Paths (time-sensitive, can fail if delayed):
- GET /epoch/{id}/leads (validators fetching leads for validation)
- POST /validate (validators submitting decision hashes)
- POST /reveal/ and /reveal/batch (validators revealing decisions)
- POST /weights/submit (auditor validators submitting weights)
- /qualification/validator/* (register, request-evaluation, report-results, etc.)
- /qualification/proxy (API proxy for model evaluation)

Qualification Model Paths (priority - rare, bypasses sourcing throttle):
- /qualification/model/presign (miners uploading qualification models)
- /qualification/model/submit (miners submitting qualification models)

Miner Throttled Paths (sourcing only):
- POST /presign (miners requesting lead presigned URLs)
- POST /submit (miners submitting leads)
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import asyncio
import os
import time

# Longest a miner request may queue for a semaphore slot before being shed
# with a 503.  Miner clients stop reading after ~15s, so a slot granted
# later than that is spent rendering a response nobody receives — while
# blocking a slot that could serve a live request.  Shedding early keeps
# the queue bounded: under overload miners get a fast 503 and retry on
# their normal polling cadence instead of stacking up behind dead sockets
# until the queue can never drain.
MINER_SLOT_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("MINER_SLOT_WAIT_TIMEOUT_SECONDS", "8")
)


class PriorityMiddleware(BaseHTTPMiddleware):
    """
    Prioritize validator and qualification model requests over miner sourcing.
    
    Architecture:
    - Validators bypass throttling (immediate processing)
    - Qualification model submissions bypass throttling (rare, immediate)
    - Sourcing miners use a semaphore (max N concurrent)
    - No changes to request processing logic
    - Safe: Only adds async waiting, never blocks
    
    Args:
        max_concurrent_miners: Max concurrent miner sourcing requests (default: 20)
        
    Example:
        from gateway.middleware.priority import PriorityMiddleware
        app.add_middleware(PriorityMiddleware, max_concurrent_miners=20)
    """
    
    def __init__(self, app, max_concurrent_miners: int = 20):
        super().__init__(app)
        self.max_concurrent_miners = max_concurrent_miners
        self.miner_semaphore = asyncio.Semaphore(max_concurrent_miners)
        
        # Track metrics
        self.validator_requests = 0
        self.qualification_model_requests = 0
        self.miner_requests = 0
        self.throttled_miners = 0
        self.shed_miners = 0
    
    def _is_validator_request(self, path: str) -> bool:
        """Check if request is from a validator (high priority)."""
        validator_paths = [
            "/epoch/",                      # GET /epoch/{id}/leads
            "/validate",                    # POST /validate
            "/reveal",                      # POST /reveal/ and /reveal/batch
            "/weights",                     # POST /weights/submit
            "/qualification/validator/",    # All qualification validator endpoints
            "/qualification/proxy",         # API proxy for model evaluation
            "/fulfillment/score",           # Validator fulfillment scoring
            "/fulfillment/scoring",         # Validator fetching reveals to score
            "/fulfillment/ban",             # Validator fulfillment ban requests
            "/fulfillment/rewards/",        # Validator fetching active rewards
                                            # for weight calculation (time-
                                            # sensitive; one failed fetch ==
                                            # one missed epoch of emission).
            "/fulfillment/results/",        # Validator/dashboard queries
        ]
        return any(vpath in path for vpath in validator_paths)
    
    def _is_qualification_model_request(self, path: str) -> bool:
        """Check if request is a qualification model submission (priority)."""
        return "/qualification/model/" in path
    
    def _is_miner_request(self, path: str) -> bool:
        """Check if request is a sourcing miner submission (throttled)."""
        miner_paths = [
            "/presign",                 # POST /presign
            "/submit",                  # POST /submit
            "/fulfillment/commit",      # Miner fulfillment commit
            "/fulfillment/reveal",      # Miner fulfillment reveal
            "/fulfillment/requests",    # Miner polling active requests
            "/fulfillment/excluded-now",
        ]
        return any(mpath in path for mpath in miner_paths)
    
    async def dispatch(self, request: Request, call_next):
        """
        Dispatch request with priority handling.
        
        Flow:
        1. Validator requests → immediate processing
        2. Qualification model requests → immediate processing
        3. Sourcing miner requests → wait for semaphore (throttled)
        4. Other requests → immediate processing (health checks, etc.)
        """
        path = request.url.path
        
        print(f"🔍 MIDDLEWARE: {request.method} {path}")
        
        is_validator = self._is_validator_request(path)
        is_qual_model = self._is_qualification_model_request(path)
        is_miner = self._is_miner_request(path) and not is_qual_model
        
        print(f"   → Validator={is_validator}, QualModel={is_qual_model}, Miner={is_miner}")
        
        # PRIORITY 1: Validators bypass throttling
        if is_validator:
            self.validator_requests += 1
            print(f"🔵 VALIDATOR REQUEST (priority): {request.method} {path}")
            return await call_next(request)
        
        # PRIORITY 2: Qualification model submissions bypass throttling
        if is_qual_model:
            self.qualification_model_requests += 1
            print(f"🟣 QUALIFICATION MODEL REQUEST (priority): {request.method} {path}")
            return await call_next(request)
        
        # PRIORITY 3: Sourcing miners are throttled (max N concurrent)
        if is_miner:
            self.miner_requests += 1
            
            if self.miner_semaphore.locked():
                self.throttled_miners += 1
                print(f"⏸️  MINER THROTTLED (queue full): {request.method} {path}")
                print(f"   📊 Stats: Validators={self.validator_requests}, "
                      f"QualModels={self.qualification_model_requests}, "
                      f"Miners={self.miner_requests}, Throttled={self.throttled_miners}")
            
            start_wait = time.time()
            try:
                await asyncio.wait_for(
                    self.miner_semaphore.acquire(),
                    timeout=MINER_SLOT_WAIT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                self.shed_miners += 1
                print(f"🛑 MINER SHED (no slot within "
                      f"{MINER_SLOT_WAIT_TIMEOUT_SECONDS:.0f}s): "
                      f"{request.method} {path} (shed={self.shed_miners})")
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Gateway at miner capacity — retry shortly"},
                    headers={"Retry-After": "15"},
                )
            try:
                wait_time = time.time() - start_wait
                if wait_time > 0.1:
                    print(f"⏳ MINER WAITED {wait_time:.2f}s for slot: {request.method} {path}")

                return await call_next(request)
            finally:
                self.miner_semaphore.release()
        
        # PRIORITY 4: Other requests (health checks, etc.) - immediate
        return await call_next(request)

