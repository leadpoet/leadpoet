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
from typing import Optional

from supabase import create_client, Client
from supabase import create_async_client, AsyncClient

from gateway.config import (
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
)

logger = logging.getLogger(__name__)

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
        _read_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    else:
        _read_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
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
    
    _write_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
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
            _async_read_client = await create_async_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        else:
            _async_read_client = await create_async_client(SUPABASE_URL, SUPABASE_ANON_KEY)
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

        _async_write_client = await create_async_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logger.info("✅ Async Supabase WRITE client initialized (SERVICE_ROLE_KEY)")

        return _async_write_client

