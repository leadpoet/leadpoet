"""
Request Priority Middleware for Gateway

Prioritizes validator requests over miner requests to prevent validators
from timing out during high miner submission traffic.

Design:
- Validator paths get immediate processing (no throttling)
- Miner paths are throttled (max concurrent limit)
- Simple, safe, no changes to database logic

Validator Priority Paths:
- GET /epoch/{id}/leads (validators fetching leads for validation)
- POST /validate (validators submitting decision hashes/reveals)

Miner Throttled Paths:
- POST /presign (miners requesting presigned URLs)
- POST /submit (miners submitting leads)
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import time


class PriorityMiddleware(BaseHTTPMiddleware):
    """
    Prioritize validator requests over miner requests.
    
    Architecture:
    - Validators bypass throttling (immediate processing)
    - Miners use a semaphore (max N concurrent)
    - No changes to request processing logic
    - Safe: Only adds async waiting, never blocks
    
    Args:
        max_concurrent_miners: Max concurrent miner requests (default: 20)
        
    Example:
        from gateway.middleware.priority import PriorityMiddleware
        app.add_middleware(PriorityMiddleware, max_concurrent_miners=20)
    """
    
    def __init__(self, app, max_concurrent_miners: int = 20):
        super().__init__(app)
        self.max_concurrent_miners = max_concurrent_miners
        self.miner_semaphore = asyncio.Semaphore(max_concurrent_miners)
        
        # Track metrics (optional)
        self.validator_requests = 0
        self.miner_requests = 0
        self.throttled_miners = 0
    
    def _is_validator_request(self, path: str) -> bool:
        """Check if request is from a validator (high priority)."""
        # Validator paths get priority
        validator_paths = [
            "/epoch/",      # GET /epoch/{id}/leads
            "/validate",    # POST /validate (hashes/reveals)
        ]
        return any(vpath in path for vpath in validator_paths)
    
    def _is_miner_request(self, path: str) -> bool:
        """Check if request is from a miner (throttled)."""
        # Miner paths are throttled
        miner_paths = [
            "/presign",     # POST /presign
            "/submit",      # POST /submit
        ]
        return any(mpath in path for mpath in miner_paths)
    
    async def dispatch(self, request: Request, call_next):
        """
        Dispatch request with priority handling.
        
        Flow:
        1. Validator requests → immediate processing
        2. Miner requests → wait for semaphore (throttled)
        3. Other requests → immediate processing (health checks, etc.)
        """
        path = request.url.path
        
        # DEBUG: Log EVERY request to diagnose validator detection
        print(f"🔍 MIDDLEWARE: {request.method} {path}")
        
        # Check request type
        is_validator = self._is_validator_request(path)
        is_miner = self._is_miner_request(path)
        
        # DEBUG: Show classification
        print(f"   → Validator={is_validator}, Miner={is_miner}")
        
        # PRIORITY 1: Validators bypass throttling
        if is_validator:
            self.validator_requests += 1
            print(f"🔵 VALIDATOR REQUEST (priority): {request.method} {path}")
            return await call_next(request)
        
        # PRIORITY 2: Miners are throttled (max N concurrent)
        if is_miner:
            self.miner_requests += 1
            
            # Try to acquire semaphore (non-blocking check)
            if self.miner_semaphore.locked():
                self.throttled_miners += 1
                print(f"⏸️  MINER THROTTLED (queue full): {request.method} {path}")
                print(f"   📊 Stats: Validators={self.validator_requests}, "
                      f"Miners={self.miner_requests}, Throttled={self.throttled_miners}")
            
            # Wait for semaphore slot
            start_wait = time.time()
            async with self.miner_semaphore:
                wait_time = time.time() - start_wait
                if wait_time > 0.1:  # Log if waited > 100ms
                    print(f"⏳ MINER WAITED {wait_time:.2f}s for slot: {request.method} {path}")
                
                # Process request
                return await call_next(request)
        
        # PRIORITY 3: Other requests (health checks, etc.) - immediate
        return await call_next(request)

