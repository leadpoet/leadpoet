"""
Trustless Rate Limiter for Gateway

Prevents DoS attacks by rate-limiting miner submissions:
- 10 submissions max per miner per day
- 5 rejections max per miner per day
- Daily reset at midnight EST (05:00 UTC)

Design:
- In-memory cache for fast lookups (O(1))
- Check BEFORE expensive operations (signature verification, DB queries)
- Log all rate limit decisions to TEE for transparency
- Early rejection with HTTP 429 (minimal resource usage)

Security:
- Rate limits checked before signature verification (DoS protection)
- All decisions logged to TEE buffer (auditability)
- Daily reset prevents indefinite blocks
- Per-hotkey limits (no metagraph dependency)
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple
import threading
import asyncio

# In-memory rate limit cache
# Structure: {miner_hotkey: {submissions: int, rejections: int, reset_at: datetime}}
_rate_limit_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()

# Rate limit constants
# TEMPORARY: Increased to 100 for testing (TODO: Revert to 10/5 for production)
MAX_SUBMISSIONS_PER_DAY = 100
MAX_REJECTIONS_PER_DAY = 100

# EST timezone offset (UTC-5, or UTC-4 during DST)
# For simplicity, we'll use UTC-5 (EST) year-round
EST_OFFSET = -5


def get_next_midnight_est() -> datetime:
    """
    Calculate the next midnight EST (05:00 UTC).
    
    Returns:
        datetime: Next midnight EST in UTC
    """
    now_utc = datetime.now(timezone.utc)
    
    # Convert to EST (UTC-5)
    est_now = now_utc + timedelta(hours=EST_OFFSET)
    
    # Get next midnight EST
    next_midnight_est = est_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    # Convert back to UTC
    next_midnight_utc = next_midnight_est - timedelta(hours=EST_OFFSET)
    
    return next_midnight_utc


def check_rate_limit(miner_hotkey: str) -> Tuple[bool, str, Dict]:
    """
    Check if miner has exceeded rate limits.
    
    This is called BEFORE signature verification to prevent DoS attacks.
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Tuple[bool, str, Dict]: (allowed, reason, stats)
            - allowed: True if miner can submit, False if rate limited
            - reason: Human-readable reason for rejection (empty if allowed)
            - stats: Current rate limit stats {submissions, rejections, reset_at}
    """
    with _cache_lock:
        now_utc = datetime.now(timezone.utc)
        
        # Get or create rate limit entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est()
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            # Reset counters
            entry["submissions"] = 0
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Check submission limit
        if entry["submissions"] >= MAX_SUBMISSIONS_PER_DAY:
            time_until_reset = entry["reset_at"] - now_utc
            hours_left = int(time_until_reset.total_seconds() / 3600)
            return (
                False,
                f"Daily submission limit reached ({MAX_SUBMISSIONS_PER_DAY}/day). Resets in {hours_left}h at midnight EST.",
                {
                    "submissions": entry["submissions"],
                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                    "rejections": entry["rejections"],
                    "max_rejections": MAX_REJECTIONS_PER_DAY,
                    "reset_at": entry["reset_at"].isoformat(),
                    "limit_type": "submissions"
                }
            )
        
        # Check rejection limit
        if entry["rejections"] >= MAX_REJECTIONS_PER_DAY:
            time_until_reset = entry["reset_at"] - now_utc
            hours_left = int(time_until_reset.total_seconds() / 3600)
            return (
                False,
                f"Daily rejection limit reached ({MAX_REJECTIONS_PER_DAY}/day). Resets in {hours_left}h at midnight EST.",
                {
                    "submissions": entry["submissions"],
                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                    "rejections": entry["rejections"],
                    "max_rejections": MAX_REJECTIONS_PER_DAY,
                    "reset_at": entry["reset_at"].isoformat(),
                    "limit_type": "rejections"
                }
            )
        
        # Allowed - return current stats
        return (
            True,
            "",
            {
                "submissions": entry["submissions"],
                "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                "rejections": entry["rejections"],
                "max_rejections": MAX_REJECTIONS_PER_DAY,
                "reset_at": entry["reset_at"].isoformat()
            }
        )


def increment_submission(miner_hotkey: str, success: bool) -> Dict:
    """
    Increment submission counters after processing a lead.
    
    Called AFTER signature verification and processing.
    
    Args:
        miner_hotkey: Miner's SS58 address
        success: True if lead was accepted, False if rejected
        
    Returns:
        Dict: Updated stats {submissions, rejections, reset_at}
    """
    with _cache_lock:
        now_utc = datetime.now(timezone.utc)
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est()
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            entry["submissions"] = 0
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Increment counters
        entry["submissions"] += 1
        if not success:
            entry["rejections"] += 1
        
        return {
            "submissions": entry["submissions"],
            "rejections": entry["rejections"],
            "reset_at": entry["reset_at"].isoformat()
        }


def get_rate_limit_stats(miner_hotkey: str) -> Dict:
    """
    Get current rate limit stats for a miner (read-only).
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Dict: {submissions, rejections, reset_at, limits}
    """
    with _cache_lock:
        if miner_hotkey not in _rate_limit_cache:
            return {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est().isoformat(),
                "limits": {
                    "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                    "max_rejections": MAX_REJECTIONS_PER_DAY
                }
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        return {
            "submissions": entry["submissions"],
            "rejections": entry["rejections"],
            "reset_at": entry["reset_at"],
            "limits": {
                "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                "max_rejections": MAX_REJECTIONS_PER_DAY
            }
        }


def cleanup_old_entries():
    """
    Clean up old entries from cache (called by background task).
    
    Removes entries that are past their reset time and have 0 submissions.
    This prevents memory leaks from inactive miners.
    """
    with _cache_lock:
        now_utc = datetime.now(timezone.utc)
        
        # Find entries to remove
        to_remove = []
        for hotkey, entry in _rate_limit_cache.items():
            # If reset time passed and no recent activity, remove
            if now_utc >= entry["reset_at"] and entry["submissions"] == 0:
                to_remove.append(hotkey)
        
        # Remove entries
        for hotkey in to_remove:
            del _rate_limit_cache[hotkey]
        
        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} inactive rate limit entries")


async def rate_limiter_cleanup_task():
    """
    Background task to clean up old rate limit entries.
    
    Runs every hour to prevent memory leaks.
    """
    print("🚀 Rate limiter cleanup task started")
    
    while True:
        try:
            # Clean up every hour
            await asyncio.sleep(3600)
            cleanup_old_entries()
            
        except Exception as e:
            print(f"❌ Rate limiter cleanup error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait 1 minute before retry


def get_all_rate_limit_stats() -> Dict[str, Dict]:
    """
    Get rate limit stats for all miners (admin endpoint).
    
    Returns:
        Dict[str, Dict]: {miner_hotkey: stats}
    """
    with _cache_lock:
        return {
            hotkey: {
                "submissions": entry["submissions"],
                "rejections": entry["rejections"],
                "reset_at": entry["reset_at"].isoformat()
            }
            for hotkey, entry in _rate_limit_cache.items()
        }

