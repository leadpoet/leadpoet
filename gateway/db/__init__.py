"""
Gateway Database Client Module
==============================

Provides centralized Supabase client management with proper read/write separation.
"""

from gateway.db.client import get_read_client, get_write_client

__all__ = ["get_read_client", "get_write_client"]

