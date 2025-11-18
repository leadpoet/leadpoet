"""
Nonce Tracking Utility (Anti-Replay Protection)
==============================================

Prevents replay attacks by tracking used nonces.

Uses the transparency_log table's UNIQUE nonce constraint to ensure
each nonce is only used once. Nonces are UUID v4 strings.
"""

from typing import Optional
from datetime import datetime, timedelta
import sys
import os

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gateway.config import NONCE_EXPIRY_SECONDS, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Import Supabase
from supabase import create_client, Client

# Create Supabase client (using service_role key for full access)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def check_and_store_nonce(nonce: str, actor_hotkey: str) -> bool:
    """
    Check if nonce has been used before.
    
    This function only CHECKS for nonce existence. The nonce is actually
    stored when the full event is inserted into transparency_log.
    
    Args:
        nonce: UUID v4 nonce from the request
        actor_hotkey: Actor's hotkey (for logging purposes)
    
    Returns:
        True if nonce is fresh (not used before)
        False if nonce has been used (replay attack detected)
    
    Example:
        >>> nonce = "550e8400-e29b-41d4-a716-446655440000"
        >>> if check_and_store_nonce(nonce, "5GNJq..."):
        >>>     # Process request
        >>>     pass
        >>> else:
        >>>     # Reject replay attack
        >>>     raise HTTPException(400, "Nonce already used")
    
    Notes:
        - Uses transparency_log table's UNIQUE constraint on nonce column
        - If nonce exists, it means this request is a replay
        - The nonce will be stored when the full event is logged
        - This prevents the "check-then-store" race condition
    """
    try:
        # Query transparency_log for this nonce
        result = supabase.table("transparency_log").select("id").eq("nonce", nonce).execute()
        
        if result.data:
            # Nonce exists - this is a replay attack
            print(f"⚠️  Replay attack detected: nonce {nonce} already used by {actor_hotkey[:20]}...")
            return False
        
        # Nonce doesn't exist - it's fresh
        return True
    
    except Exception as e:
        print(f"❌ Nonce check error: {e}")
        # On error, fail closed (reject the request)
        return False


def is_nonce_expired(timestamp: datetime) -> bool:
    """
    Check if a nonce timestamp is expired.
    
    Args:
        timestamp: The timestamp from the request
    
    Returns:
        True if timestamp is older than NONCE_EXPIRY_SECONDS
    
    Example:
        >>> from datetime import datetime, timedelta
        >>> old_timestamp = datetime.utcnow() - timedelta(seconds=400)
        >>> is_nonce_expired(old_timestamp)
        True
    """
    now = datetime.utcnow()
    age = (now - timestamp).total_seconds()
    return age > NONCE_EXPIRY_SECONDS


def cleanup_expired_nonces():
    """
    Log information about expired nonces.
    
    Note: The transparency_log is append-only, so we don't actually delete nonces.
    The UNIQUE constraint on the nonce column prevents reuse regardless of age.
    
    This function is primarily informational - it logs how many old nonces exist.
    
    Example:
        >>> cleanup_expired_nonces()
        🧹 Found 1234 nonces older than 300s (5 minutes)
        ℹ️  Transparency log is append-only - no cleanup performed
    """
    try:
        # Calculate expiry time
        expiry_time = datetime.utcnow() - timedelta(seconds=NONCE_EXPIRY_SECONDS)
        
        # Count old nonces (for informational purposes)
        result = supabase.table("transparency_log").select("id", count="exact").lt("ts", expiry_time.isoformat()).execute()
        
        count = result.count if result.count else 0
        
        print(f"🧹 Found {count} events older than {NONCE_EXPIRY_SECONDS}s")
        print(f"ℹ️  Transparency log is append-only - nonces remain for audit trail")
        
        return count
    
    except Exception as e:
        print(f"❌ Cleanup check error: {e}")
        return 0


def get_nonce_stats() -> dict:
    """
    Get statistics about nonces in the transparency log.
    
    Returns:
        Dictionary with nonce statistics:
        - total_events: Total number of events in log
        - recent_events: Events within NONCE_EXPIRY_SECONDS
        - old_events: Events older than NONCE_EXPIRY_SECONDS
    
    Example:
        >>> stats = get_nonce_stats()
        >>> print(f"Total events: {stats['total_events']}")
    """
    try:
        # Get total count
        total_result = supabase.table("transparency_log").select("id", count="exact").execute()
        total_events = total_result.count if total_result.count else 0
        
        # Get recent count (within expiry window)
        expiry_time = datetime.utcnow() - timedelta(seconds=NONCE_EXPIRY_SECONDS)
        recent_result = supabase.table("transparency_log").select("id", count="exact").gte("ts", expiry_time.isoformat()).execute()
        recent_events = recent_result.count if recent_result.count else 0
        
        # Calculate old count
        old_events = total_events - recent_events
        
        return {
            "total_events": total_events,
            "recent_events": recent_events,
            "old_events": old_events,
            "expiry_seconds": NONCE_EXPIRY_SECONDS
        }
    
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return {
            "total_events": 0,
            "recent_events": 0,
            "old_events": 0,
            "expiry_seconds": NONCE_EXPIRY_SECONDS
        }


def print_nonce_stats():
    """
    Print nonce statistics in a readable format.
    
    Example:
        >>> print_nonce_stats()
        ============================================================
        Nonce Statistics
        ============================================================
        Total Events: 1234
        Recent Events (< 5 min): 123
        Old Events (> 5 min): 1111
        Expiry Window: 300s (5 minutes)
        ============================================================
    """
    stats = get_nonce_stats()
    
    print("=" * 60)
    print("Nonce Statistics")
    print("=" * 60)
    print(f"Total Events: {stats['total_events']}")
    print(f"Recent Events (< {stats['expiry_seconds']}s): {stats['recent_events']}")
    print(f"Old Events (> {stats['expiry_seconds']}s): {stats['old_events']}")
    print(f"Expiry Window: {stats['expiry_seconds']}s ({stats['expiry_seconds']//60} minutes)")
    print("=" * 60)


def validate_nonce_format(nonce: str) -> bool:
    """
    Validate that nonce is a valid UUID v4 format.
    
    Args:
        nonce: Nonce string to validate
    
    Returns:
        True if valid UUID v4 format
    
    Example:
        >>> validate_nonce_format("550e8400-e29b-41d4-a716-446655440000")
        True
        >>> validate_nonce_format("not-a-uuid")
        False
    """
    import uuid
    
    try:
        # Try to parse as UUID
        uuid_obj = uuid.UUID(nonce)
        
        # Check if it's version 4
        return uuid_obj.version == 4
    
    except (ValueError, AttributeError):
        return False

