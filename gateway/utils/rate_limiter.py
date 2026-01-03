"""
Trustless Rate Limiter for Gateway

Prevents DoS attacks by rate-limiting miner submissions:
- 250 submissions max per miner per day
- 50 rejections max per miner per day
- Daily reset at midnight EST (05:00 UTC)

Design:
- In-memory cache for fast lookups (O(1))
- Supabase persistence (survives gateway restarts)
- Public read-only table (transparent rate limits)
- Async writes to Supabase (non-blocking)
- Check BEFORE expensive operations (signature verification, DB queries)

Security:
- Rate limits checked before signature verification (DoS protection)
- Persisted to Supabase (can't be bypassed by restarting gateway)
- Public transparency (miners can verify their limits)
- Daily reset prevents indefinite blocks
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional
import threading
import asyncio
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client for rate limit persistence
_supabase_client = None

def _get_supabase():
    """Get or create Supabase client (lazy initialization)."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_client

# In-memory rate limit cache (fast lookups)
# Structure: {miner_hotkey: {submissions: int, rejections: int, reset_at: datetime, last_submission_time: datetime}}
# Cache is loaded from Supabase on first use, then kept in sync
_rate_limit_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
_cache_loaded = False  # Track if we've loaded from Supabase yet

# Rate limit constants
# Production limits to maintain lead quality and prevent spam
MAX_SUBMISSIONS_PER_DAY = 500
MAX_REJECTIONS_PER_DAY = 100
MIN_SECONDS_BETWEEN_SUBMISSIONS = 45  # Cooldown between submissions (anti-spam)

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


def _load_cache_from_supabase():
    """
    Load rate limit cache from Supabase on first use.
    
    Called once on first rate limit check to hydrate in-memory cache.
    This ensures rate limits persist across gateway restarts.
    """
    global _cache_loaded
    
    if _cache_loaded:
        return  # Already loaded
    
    try:
        supabase = _get_supabase()
        
        # Fetch all rate limit entries
        result = supabase.table("miner_rate_limits").select("*").execute()
        
        if result.data:
            # Populate cache from database
            for row in result.data:
                last_sub = row.get("last_submission_time")
                
                # Parse timestamps flexibly (handles varying decimal precision)
                def parse_timestamp(ts_str):
                    if not ts_str:
                        return None
                    ts_str = ts_str.replace("Z", "+00:00")
                    try:
                        return datetime.fromisoformat(ts_str)
                    except ValueError:
                        # Handle non-standard decimal precision (e.g., 5 digits instead of 3/6)
                        # Strip fractional seconds and parse, then add them back
                        from dateutil import parser
                        return parser.isoparse(ts_str)
                
                _rate_limit_cache[row["miner_hotkey"]] = {
                    "submissions": row["submissions"],
                    "rejections": row["rejections"],
                    "reset_at": parse_timestamp(row["reset_at"]),
                    "last_submission_time": parse_timestamp(last_sub)
                }
            
            print(f"✅ Loaded {len(result.data)} miner rate limits from Supabase")
        else:
            print(f"ℹ️  No existing rate limits in Supabase (starting fresh)")
        
        _cache_loaded = True
        
    except Exception as e:
        print(f"⚠️  Failed to load rate limits from Supabase: {e}")
        print(f"   Will start with empty cache and sync on first update")
        _cache_loaded = True  # Don't keep trying to load


async def _sync_to_supabase_async(miner_hotkey: str, entry: Dict):
    """
    Sync rate limit entry to Supabase (async, non-blocking).
    
    This is called after EVERY submission increment to ensure persistence.
    Uses upsert (insert or update) for simplicity.
    
    Args:
        miner_hotkey: Miner's SS58 address
        entry: Rate limit entry dict {submissions, rejections, reset_at}
    """
    try:
        supabase = _get_supabase()
        
        # Upsert to Supabase (insert or update)
        last_sub_time = entry.get("last_submission_time")
        supabase.table("miner_rate_limits").upsert({
            "miner_hotkey": miner_hotkey,
            "submissions": entry["submissions"],
            "rejections": entry["rejections"],
            "max_submissions": MAX_SUBMISSIONS_PER_DAY,
            "max_rejections": MAX_REJECTIONS_PER_DAY,
            "reset_at": entry["reset_at"].isoformat(),
            "last_submission_time": last_sub_time.isoformat() if last_sub_time else None,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }).execute()
        
    except Exception as e:
        # Don't fail the submission if Supabase sync fails
        # The in-memory cache is still updated
        print(f"⚠️  Failed to sync rate limit to Supabase for {miner_hotkey[:10]}...: {e}")


def check_rate_limit(miner_hotkey: str) -> Tuple[bool, str, Dict]:
    """
    Check if miner has exceeded rate limits.
    
    This is called BEFORE signature verification to prevent DoS attacks.
    Loads cache from Supabase on first use.
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Tuple[bool, str, Dict]: (allowed, reason, stats)
            - allowed: True if miner can submit, False if rate limited
            - reason: Human-readable reason for rejection (empty if allowed)
            - stats: Current rate limit stats {submissions, rejections, reset_at}
    """
    with _cache_lock:
        # Load cache from Supabase on first use
        if not _cache_loaded:
            _load_cache_from_supabase()
        now_utc = datetime.now(timezone.utc)
        
        # Get or create rate limit entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            # Reset counters
            entry["submissions"] = 0
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Check cooldown (anti-spam: minimum time between submissions)
        last_time = entry.get("last_submission_time")
        if last_time is not None:
            seconds_since_last = (now_utc - last_time).total_seconds()
            if seconds_since_last < MIN_SECONDS_BETWEEN_SUBMISSIONS:
                wait_seconds = int(MIN_SECONDS_BETWEEN_SUBMISSIONS - seconds_since_last)
                return (
                    False,
                    f"Please wait {wait_seconds} seconds before submitting another lead (anti-spam cooldown).",
                    {
                        "submissions": entry["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": entry["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": entry["reset_at"].isoformat(),
                        "limit_type": "cooldown",
                        "cooldown_seconds": MIN_SECONDS_BETWEEN_SUBMISSIONS,
                        "wait_seconds": wait_seconds
                    }
                )
        
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
        
        # NOTE: last_submission_time is updated in increment_submission() AFTER successful submission
        # We do NOT update it here because:
        # 1. /presign calls check_rate_limit() first
        # 2. Then /submit calls check_rate_limit() again
        # 3. If we set the time here, /submit would be blocked after /presign succeeds!
        # The 30-second cooldown is enforced AFTER a lead is fully submitted.
        
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
    Updates both in-memory cache AND Supabase for persistence.
    
    Args:
        miner_hotkey: Miner's SS58 address
        success: True if lead was accepted, False if rejected
        
    Returns:
        Dict: Updated stats {submissions, rejections, reset_at}
    """
    with _cache_lock:
        now_utc = datetime.now(timezone.utc)
        
        # Load cache from Supabase on first use
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            entry["submissions"] = 0
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Increment counters (in-memory)
        entry["submissions"] += 1
        if not success:
            entry["rejections"] += 1
        
        # Record last submission time (for cooldown check)
        entry["last_submission_time"] = now_utc
        
        # Sync to Supabase (async, non-blocking)
        # Fire-and-forget - don't wait for DB write
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            # No event loop running (shouldn't happen in FastAPI, but be defensive)
            # Skip Supabase sync but in-memory cache is still updated
            print(f"⚠️  No event loop - skipping Supabase sync for {miner_hotkey[:10]}...")
        
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


# ============================================================
# ATOMIC RATE LIMITING (Fixes race condition)
# ============================================================
# These functions atomically check AND reserve/update counters
# to prevent the race condition where multiple simultaneous
# requests all pass check_rate_limit() before any increment.
# ============================================================

def reserve_submission_slot(miner_hotkey: str) -> Tuple[bool, str, Dict]:
    """
    ATOMICALLY check rate limits AND reserve a submission slot.
    
    This fixes the race condition in the original check_rate_limit() + increment_submission()
    pattern where multiple simultaneous requests could all pass the check before any incremented.
    
    If allowed:
      - Increments submissions counter IMMEDIATELY (atomically)
      - Sets last_submission_time (starts cooldown)
      - Syncs to Supabase
      - Returns (True, "", stats)
    
    If not allowed:
      - Does NOT increment anything
      - Returns (False, reason, stats)
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Tuple[bool, str, Dict]: (allowed, reason, stats)
    
    Usage in /submit endpoint:
        allowed, reason, stats = reserve_submission_slot(hotkey)
        if not allowed:
            return 429 error
        # Process submission...
        if failed:
            mark_submission_failed(hotkey)  # Increment rejections
        # If success: slot already consumed, nothing more to do
    """
    with _cache_lock:
        # Load cache from Supabase on first use
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        now_utc = datetime.now(timezone.utc)
        
        # Get or create rate limit entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 0,
                "rejections": 0,
                "reset_at": get_next_midnight_est(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            # Reset counters
            entry["submissions"] = 0
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Check cooldown (anti-spam: minimum time between submissions)
        last_time = entry.get("last_submission_time")
        if last_time is not None:
            seconds_since_last = (now_utc - last_time).total_seconds()
            if seconds_since_last < MIN_SECONDS_BETWEEN_SUBMISSIONS:
                wait_seconds = int(MIN_SECONDS_BETWEEN_SUBMISSIONS - seconds_since_last)
                return (
                    False,
                    f"Please wait {wait_seconds} seconds before submitting another lead (anti-spam cooldown).",
                    {
                        "submissions": entry["submissions"],
                        "max_submissions": MAX_SUBMISSIONS_PER_DAY,
                        "rejections": entry["rejections"],
                        "max_rejections": MAX_REJECTIONS_PER_DAY,
                        "reset_at": entry["reset_at"].isoformat(),
                        "limit_type": "cooldown",
                        "cooldown_seconds": MIN_SECONDS_BETWEEN_SUBMISSIONS,
                        "wait_seconds": wait_seconds
                    }
                )
        
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
        
        # ============================================================
        # ATOMIC RESERVATION: Increment submissions counter NOW
        # ============================================================
        # This is the key fix: we increment INSIDE the lock, before
        # releasing it. Any other request that arrives will see the
        # updated counter and may be blocked.
        entry["submissions"] += 1
        entry["last_submission_time"] = now_utc
        
        # Prepare stats to return
        stats = {
            "submissions": entry["submissions"],
            "max_submissions": MAX_SUBMISSIONS_PER_DAY,
            "rejections": entry["rejections"],
            "max_rejections": MAX_REJECTIONS_PER_DAY,
            "reset_at": entry["reset_at"].isoformat()
        }
        
        # Sync to Supabase (async, non-blocking)
        # Fire-and-forget - don't wait for DB write
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            # No event loop running - sync will happen on next update
            print(f"⚠️  No event loop - Supabase sync deferred for {miner_hotkey[:10]}...")
        
        return (True, "", stats)


def mark_submission_failed(miner_hotkey: str) -> Dict:
    """
    Mark a reserved submission slot as failed (increment rejections).
    
    Called when a submission that was reserved via reserve_submission_slot()
    fails during processing. This increments the rejections counter.
    
    NOTE: The submissions counter was already incremented in reserve_submission_slot().
    This function ONLY increments rejections.
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Dict: Updated stats {submissions, rejections, reset_at}
    """
    with _cache_lock:
        now_utc = datetime.now(timezone.utc)
        
        # Load cache from Supabase on first use (should already be loaded)
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        # Entry should exist (reserved earlier), but be defensive
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "submissions": 1,  # Already had one submission (the failed one)
                "rejections": 0,
                "reset_at": get_next_midnight_est(),
                "last_submission_time": now_utc
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            entry["submissions"] = 1  # The current failed one
            entry["rejections"] = 0
            entry["reset_at"] = get_next_midnight_est()
        
        # Increment ONLY rejections (submissions was already incremented in reserve_submission_slot)
        entry["rejections"] += 1
        
        # Sync to Supabase (async, non-blocking)
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            print(f"⚠️  No event loop - Supabase sync deferred for {miner_hotkey[:10]}...")
        
        return {
            "submissions": entry["submissions"],
            "rejections": entry["rejections"],
            "reset_at": entry["reset_at"].isoformat()
        }

