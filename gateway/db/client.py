"""
Supabase Client Management
==========================

Provides centralized read/write client separation for Supabase.

SECURITY PRINCIPLE (Least Privilege):
- Read operations use ANON key (respects RLS, public access only)
- Write operations use SERVICE_ROLE key (bypasses RLS for gateway authority)
"""

import logging
from typing import Optional

from supabase import create_client, Client

from gateway.config import (
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
)

logger = logging.getLogger(__name__)

# Singleton Clients (lazily initialized)
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
        # Fall back to service role key if anon key not available
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

