"""
Supabase Client Management
==========================

Provides centralized read/write client separation for Supabase.

SECURITY PRINCIPLE (Least Privilege):
- Read operations use ANON key (respects RLS, public access only)
- Write operations use SERVICE_ROLE key (bypasses RLS for gateway authority)

ASYNC CLIENTS:
- Async variants (get_async_read_client, get_async_write_client) release the
  event loop while waiting for Supabase HTTP responses, preventing one slow
  query from blocking all other requests.
- Use async clients in FastAPI endpoint handlers (presign, submit, validate).
- Sync clients remain available for background tasks and non-async code paths.
"""

import asyncio
import logging
import os
from typing import Optional

import httpx
from supabase import create_client, Client
from supabase import create_async_client, AsyncClient

from gateway.config import (
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
)

logger = logging.getLogger(__name__)

# Bounded HTTP read timeout (seconds) for SYNCHRONOUS Supabase clients.
# The synchronous postgrest client defaults to a 120s timeout, and the
# fulfillment lifecycle tick makes these sync calls directly on the asyncio
# event loop. When the PostgREST layer intermittently stalls (Postgres itself
# stays idle/fast), the tick's sequential calls each block up to 120s, wedging
# the ENTIRE gateway — every endpoint returns 000 and miners get read-timeouts
# for minutes until the connections finally die. Capping at 30s means the first
# stalled call raises at 30s; the lifecycle loop's try/except then aborts that
# tick and retries next interval, so a stall becomes a ~30s blip instead of a
# multi-minute outage. 30s is far above normal PostgREST latency (sub-second to
# a few seconds), so legitimate queries never trip it.
_SYNC_HTTP_TIMEOUT_SECONDS = int(
    os.getenv(
        "SUPABASE_TIMEOUT_SECONDS",
        os.getenv("SUPABASE_SYNC_TIMEOUT_SECONDS", "8"),
    )
)


def _apply_sync_timeout(client: Client) -> Client:
    """Bound the sync postgrest HTTP timeout. Best-effort: wrapped in try/except
    so it can never break client creation — worst case the cap isn't applied and
    behaviour falls back to the library's 120s default."""
    try:
        client.postgrest.session.timeout = httpx.Timeout(float(_SYNC_HTTP_TIMEOUT_SECONDS))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not apply Supabase sync HTTP timeout: %s", e)
    return client


# Errors that mean the pooled connection died under us (Supabase's edge sends
# HTTP/2 GOAWAY / resets idle connections; httpx surfaces the reuse attempt as
# one of these). The request was terminated at the transport layer, so one
# retry on a fresh connection is safe and turns an intermittent API 500 into
# a served request. Timeouts and HTTP status codes are deliberately excluded:
# those can mean the server already did the work, and re-firing non-idempotent
# writes there would be worse than failing.
_POOL_TERMINATION_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectError,
)


def _install_sync_send_retry(client: Client) -> Client:
    """Retry a sync postgrest request once when the pooled connection died."""
    try:
        session = client.postgrest.session
        original_send = session.send

        def send_with_retry(request, **kwargs):
            try:
                return original_send(request, **kwargs)
            except _POOL_TERMINATION_ERRORS as exc:
                logger.warning(
                    "supabase_http_send_retry transport=sync error=%s path=%s",
                    type(exc).__name__,
                    request.url.path,
                )
                return original_send(request, **kwargs)

        session.send = send_with_retry
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not install Supabase sync send retry: %s", e)
    return client


def _install_async_send_retry(client: AsyncClient) -> AsyncClient:
    """Retry an async postgrest request once when the pooled connection died."""
    try:
        session = client.postgrest.session
        original_send = session.send

        async def send_with_retry(request, **kwargs):
            try:
                return await original_send(request, **kwargs)
            except _POOL_TERMINATION_ERRORS as exc:
                logger.warning(
                    "supabase_http_send_retry transport=async error=%s path=%s",
                    type(exc).__name__,
                    request.url.path,
                )
                return await original_send(request, **kwargs)

        session.send = send_with_retry
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not install Supabase async send retry: %s", e)
    return client


# ============================================================
# Sync Singleton Clients (lazily initialized)
# ============================================================
_read_client: Optional[Client] = None
_write_client: Optional[Client] = None


def get_read_client() -> Client:
    """
    Get Supabase client for READ operations (uses ANON key).
    """
    global _read_client
    
    if _read_client is not None:
        return _read_client
    
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL not configured")
    
    if not SUPABASE_ANON_KEY:
        logger.warning("⚠️ SUPABASE_ANON_KEY not configured - using SERVICE_ROLE_KEY for reads")
        if not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("No Supabase key configured")
        _read_client = _install_sync_send_retry(_apply_sync_timeout(create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)))
    else:
        _read_client = _install_sync_send_retry(_apply_sync_timeout(create_client(SUPABASE_URL, SUPABASE_ANON_KEY)))
        logger.info("✅ Supabase READ client initialized (ANON_KEY)")
    
    return _read_client


def get_write_client() -> Client:
    """
    Get Supabase client for WRITE operations (uses SERVICE_ROLE key).
    """
    global _write_client
    
    if _write_client is not None:
        return _write_client
    
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL not configured")
    
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not configured")
    
    _write_client = _install_sync_send_retry(_apply_sync_timeout(create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)))
    logger.info("✅ Supabase WRITE client initialized (SERVICE_ROLE_KEY)")
    
    return _write_client


# ============================================================
# Async Singleton Clients (lazily initialized)
# ============================================================
_async_read_client: Optional[AsyncClient] = None
_async_write_client: Optional[AsyncClient] = None
_async_lock = asyncio.Lock()


async def get_async_read_client() -> AsyncClient:
    """
    Get async Supabase client for READ operations (uses ANON key).
    Non-blocking — releases the event loop during HTTP I/O.
    """
    global _async_read_client

    if _async_read_client is not None:
        return _async_read_client

    async with _async_lock:
        if _async_read_client is not None:
            return _async_read_client

        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL not configured")

        if not SUPABASE_ANON_KEY:
            logger.warning("⚠️ SUPABASE_ANON_KEY not configured - using SERVICE_ROLE_KEY for async reads")
            if not SUPABASE_SERVICE_ROLE_KEY:
                raise RuntimeError("No Supabase key configured")
            _async_read_client = _install_async_send_retry(await create_async_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY))
        else:
            _async_read_client = _install_async_send_retry(await create_async_client(SUPABASE_URL, SUPABASE_ANON_KEY))
            logger.info("✅ Async Supabase READ client initialized (ANON_KEY)")

        return _async_read_client


async def get_async_write_client() -> AsyncClient:
    """
    Get async Supabase client for WRITE operations (uses SERVICE_ROLE key).
    Non-blocking — releases the event loop during HTTP I/O.
    """
    global _async_write_client

    if _async_write_client is not None:
        return _async_write_client

    async with _async_lock:
        if _async_write_client is not None:
            return _async_write_client

        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL not configured")

        if not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY not configured")

        _async_write_client = _install_async_send_retry(await create_async_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY))
        logger.info("✅ Async Supabase WRITE client initialized (SERVICE_ROLE_KEY)")

        return _async_write_client
