"""
Rate Limiter for Qualification Model Submissions

Prevents spam and manages submission quotas:
- 2 submissions per miner per day MAXIMUM (resets at midnight UTC)
- Credits are a SAFETY MECHANISM for failed submissions, not extra submissions

Design:
- In-memory cache for fast lookups (O(1))
- Supabase persistence (survives gateway restarts)
- Async writes to Supabase (non-blocking)
- Check BEFORE expensive operations

Flow:
1. Miner pays $5 TAO → credit is added (submission_credits += 1)
2. Miner submits model → gateway checks:
   a. If daily_submissions >= MAX_DAILY → reject (429) - even with credits!
   b. If submission_credits > 0 → allow (will consume credit + increment daily)
   c. Else → reject (payment required)
3. On successful submission:
   - Credit consumed (if had credit)
   - Daily incremented (ALWAYS)
4. If payment succeeds but submission fails:
   - Credit remains (can retry without paying again)
   - Daily NOT incremented

KEY POINT: Credits do NOT bypass daily limit. They only ensure miners
don't lose money if submission fails after payment.
"""

import os
import asyncio
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional

from supabase import create_client

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Maximum FREE submissions per day (resets at midnight UTC)
MAX_FREE_SUBMISSIONS_PER_DAY = 2

# Cost per submission credit in USD (each $5 TAO = 1 credit)
USD_PER_CREDIT = 5.0

# Minimum time between submissions (anti-spam cooldown)
MIN_SECONDS_BETWEEN_SUBMISSIONS = 60  # 1 minute cooldown

# =============================================================================
# Supabase Client
# =============================================================================

_supabase_client = None


def _get_supabase():
    """Get or create Supabase client (lazy initialization)."""
    global _supabase_client
    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_url and supabase_key:
            _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client


# =============================================================================
# In-Memory Cache
# =============================================================================

# Structure: {
#     miner_hotkey: {
#         "daily_submissions": int,      # resets at midnight UTC
#         "submission_credits": int,     # never resets (paid credits)
#         "reset_at": datetime,           # next midnight UTC
#         "last_submission_time": datetime,
#     }
# }
_rate_limit_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
_cache_loaded = False


def get_next_midnight_utc() -> datetime:
    """Calculate the next midnight UTC (00:00 UTC)."""
    now_utc = datetime.now(timezone.utc)
    next_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return next_midnight


def _load_cache_from_supabase():
    """Load rate limit cache from Supabase on first use."""
    global _cache_loaded
    
    if _cache_loaded:
        return
    
    try:
        supabase = _get_supabase()
        if not supabase:
            print("⚠️  Supabase not configured - model rate limits will be in-memory only")
            _cache_loaded = True
            return
        
        result = supabase.table("qualification_model_rate_limits").select("*").execute()
        
        if result.data:
            for row in result.data:
                # Parse timestamps
                def parse_timestamp(ts_str):
                    if not ts_str:
                        return None
                    ts_str = ts_str.replace("Z", "+00:00")
                    try:
                        return datetime.fromisoformat(ts_str)
                    except ValueError:
                        from dateutil import parser
                        return parser.isoparse(ts_str)
                
                _rate_limit_cache[row["miner_hotkey"]] = {
                    "daily_submissions": row["daily_submissions"],
                    "submission_credits": row["submission_credits"],
                    "reset_at": parse_timestamp(row["reset_at"]),
                    "last_submission_time": parse_timestamp(row.get("last_submission_time"))
                }
            
            print(f"✅ Loaded {len(result.data)} model rate limits from Supabase")
        else:
            print(f"ℹ️  No existing model rate limits in Supabase (starting fresh)")
        
        _cache_loaded = True
        
    except Exception as e:
        print(f"⚠️  Failed to load model rate limits from Supabase: {e}")
        _cache_loaded = True  # Don't keep retrying


async def _sync_to_supabase_async(miner_hotkey: str, entry: Dict):
    """Sync rate limit entry to Supabase (async, non-blocking)."""
    try:
        supabase = _get_supabase()
        if not supabase:
            return
        
        last_sub_time = entry.get("last_submission_time")
        supabase.table("qualification_model_rate_limits").upsert({
            "miner_hotkey": miner_hotkey,
            "daily_submissions": entry["daily_submissions"],
            "submission_credits": entry["submission_credits"],
            "max_daily_submissions": MAX_FREE_SUBMISSIONS_PER_DAY,
            "reset_at": entry["reset_at"].isoformat(),
            "last_submission_time": last_sub_time.isoformat() if last_sub_time else None,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
    except Exception as e:
        print(f"⚠️  Failed to sync model rate limit to Supabase for {miner_hotkey[:10]}...: {e}")


# =============================================================================
# Main Rate Limit Functions
# =============================================================================

def check_model_rate_limit(miner_hotkey: str) -> Tuple[bool, str, Dict]:
    """
    Check if miner can submit a model.
    
    Rules:
    1. If daily_submissions >= MAX_DAILY → reject (even with credits!)
    2. If submission_credits > 0 → allowed (has prepaid credit)
    3. Otherwise → reject (need to pay first)
    
    Credits do NOT bypass daily limit - they only ensure miners don't
    lose money if submission fails after payment.
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Tuple[bool, str, Dict]: (allowed, reason, stats)
    """
    with _cache_lock:
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        now_utc = datetime.now(timezone.utc)
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "daily_submissions": 0,
                "submission_credits": 0,
                "reset_at": get_next_midnight_utc(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed (daily reset at midnight UTC)
        if now_utc >= entry["reset_at"]:
            entry["daily_submissions"] = 0  # Reset daily counter
            entry["reset_at"] = get_next_midnight_utc()
            # Note: submission_credits is NOT reset (persists indefinitely)
        
        # Check cooldown (anti-spam)
        last_time = entry.get("last_submission_time")
        if last_time is not None:
            seconds_since_last = (now_utc - last_time).total_seconds()
            if seconds_since_last < MIN_SECONDS_BETWEEN_SUBMISSIONS:
                wait_seconds = int(MIN_SECONDS_BETWEEN_SUBMISSIONS - seconds_since_last)
                return (
                    False,
                    f"Please wait {wait_seconds} seconds before submitting another model (anti-spam cooldown).",
                    _build_stats(entry, "cooldown", wait_seconds)
                )
        
        # Build stats
        stats = _build_stats(entry)
        
        # RULE 1: Check daily limit FIRST (credits don't bypass this!)
        if entry["daily_submissions"] >= MAX_FREE_SUBMISSIONS_PER_DAY:
            time_until_reset = entry["reset_at"] - now_utc
            hours_left = int(time_until_reset.total_seconds() / 3600)
            
            return (
                False,
                f"Daily submission limit reached ({MAX_FREE_SUBMISSIONS_PER_DAY}/day). "
                f"Resets in {hours_left}h at midnight UTC. "
                f"You have {entry['submission_credits']} unused credit(s) that will be available tomorrow.",
                _build_stats(entry, "daily_limit")
            )
        
        # RULE 2: Check if has credit (prepaid from previous payment)
        if entry["submission_credits"] > 0:
            stats["will_use"] = "credit"
            stats["payment_required"] = False
            return (True, "", stats)
        
        # RULE 3: No credits - payment required
        stats["will_use"] = "payment_required"
        stats["payment_required"] = True
        return (True, "", stats)  # Allowed but needs payment


def reserve_model_submission(miner_hotkey: str) -> Tuple[bool, str, Dict]:
    """
    ATOMICALLY reserve a submission slot after model is created.
    
    This is called AFTER the model is successfully recorded in the database.
    It:
    1. Checks daily limit (reject if >= MAX)
    2. Decrements credit (must have one)
    3. Increments daily_submissions (ALWAYS)
    4. Updates last_submission_time
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Tuple[bool, str, Dict]: (allowed, reason, stats)
    """
    with _cache_lock:
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        now_utc = datetime.now(timezone.utc)
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "daily_submissions": 0,
                "submission_credits": 0,
                "reset_at": get_next_midnight_utc(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time has passed
        if now_utc >= entry["reset_at"]:
            entry["daily_submissions"] = 0
            entry["reset_at"] = get_next_midnight_utc()
        
        # Check cooldown
        last_time = entry.get("last_submission_time")
        if last_time is not None:
            seconds_since_last = (now_utc - last_time).total_seconds()
            if seconds_since_last < MIN_SECONDS_BETWEEN_SUBMISSIONS:
                wait_seconds = int(MIN_SECONDS_BETWEEN_SUBMISSIONS - seconds_since_last)
                return (
                    False,
                    f"Please wait {wait_seconds} seconds before submitting another model.",
                    _build_stats(entry, "cooldown", wait_seconds)
                )
        
        # Check daily limit (credits do NOT bypass this)
        if entry["daily_submissions"] >= MAX_FREE_SUBMISSIONS_PER_DAY:
            time_until_reset = entry["reset_at"] - now_utc
            hours_left = int(time_until_reset.total_seconds() / 3600)
            return (
                False,
                f"Daily limit reached ({MAX_FREE_SUBMISSIONS_PER_DAY}/day). Resets in {hours_left}h.",
                _build_stats(entry, "daily_limit")
            )
        
        # Must have credit to submit
        if entry["submission_credits"] <= 0:
            return (
                False,
                "No submission credit available. Payment is required.",
                _build_stats(entry, "no_credit")
            )
        
        # Consume credit AND increment daily (BOTH happen)
        entry["submission_credits"] -= 1
        entry["daily_submissions"] += 1
        entry["last_submission_time"] = now_utc
        
        # Build stats
        stats = _build_stats(entry)
        stats["used"] = "credit"
        
        print(f"📊 Submission reserved: Daily {entry['daily_submissions']}/{MAX_FREE_SUBMISSIONS_PER_DAY}, "
              f"Credits remaining: {entry['submission_credits']}")
        
        # Sync to Supabase
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            print(f"⚠️  No event loop - Supabase sync deferred for {miner_hotkey[:10]}...")
        
        return (True, "", stats)


def add_submission_credit(miner_hotkey: str, credits_to_add: int = 1) -> Dict:
    """
    Add submission credits to a miner's account.
    
    Called when payment is verified but BEFORE submission.
    This ensures miners don't lose credits if submission fails.
    
    Args:
        miner_hotkey: Miner's SS58 address
        credits_to_add: Number of credits to add (default: 1)
        
    Returns:
        Dict: Updated stats
    """
    with _cache_lock:
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        now_utc = datetime.now(timezone.utc)
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "daily_submissions": 0,
                "submission_credits": 0,
                "reset_at": get_next_midnight_utc(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time passed (for daily counter)
        if now_utc >= entry["reset_at"]:
            entry["daily_submissions"] = 0
            entry["reset_at"] = get_next_midnight_utc()
        
        # Add credits
        entry["submission_credits"] += credits_to_add
        
        print(f"💳 Added {credits_to_add} submission credit(s) for {miner_hotkey[:16]}... "
              f"(total credits: {entry['submission_credits']})")
        
        # Sync to Supabase
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            print(f"⚠️  No event loop - Supabase sync deferred for {miner_hotkey[:10]}...")
        
        return _build_stats(entry)


def increment_daily_only(miner_hotkey: str) -> Dict:
    """
    Increment daily submission count WITHOUT consuming credit.
    
    Called when:
    - Payment verified but security scan failed
    - We want to prevent spam (count toward daily limit)
    - But preserve credit so miner can fix and retry
    
    Args:
        miner_hotkey: Miner's SS58 address
        
    Returns:
        Dict: Updated stats
    """
    with _cache_lock:
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        now_utc = datetime.now(timezone.utc)
        
        # Get or create entry
        if miner_hotkey not in _rate_limit_cache:
            _rate_limit_cache[miner_hotkey] = {
                "daily_submissions": 0,
                "submission_credits": 0,
                "reset_at": get_next_midnight_utc(),
                "last_submission_time": None
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Check if reset time passed
        if now_utc >= entry["reset_at"]:
            entry["daily_submissions"] = 0
            entry["reset_at"] = get_next_midnight_utc()
        
        # Increment daily ONLY (not credit)
        entry["daily_submissions"] += 1
        entry["last_submission_time"] = now_utc
        
        print(f"📊 Daily count incremented (no credit consumed) for {miner_hotkey[:16]}... "
              f"(daily: {entry['daily_submissions']}/{MAX_FREE_SUBMISSIONS_PER_DAY}, "
              f"credits preserved: {entry['submission_credits']})")
        
        # Sync to Supabase
        try:
            asyncio.create_task(_sync_to_supabase_async(miner_hotkey, entry))
        except RuntimeError:
            print(f"⚠️  No event loop - Supabase sync deferred for {miner_hotkey[:10]}...")
        
        return _build_stats(entry)


def get_model_rate_limit_stats(miner_hotkey: str) -> Dict:
    """Get current rate limit stats for a miner."""
    with _cache_lock:
        if not _cache_loaded:
            _load_cache_from_supabase()
        
        if miner_hotkey not in _rate_limit_cache:
            return {
                "daily_submissions": 0,
                "max_daily_submissions": MAX_FREE_SUBMISSIONS_PER_DAY,
                "submission_credits": 0,
                "reset_at": get_next_midnight_utc().isoformat(),
                "can_submit": True,
                "reason": ""
            }
        
        entry = _rate_limit_cache[miner_hotkey]
        
        # Reset expired daily counters (same logic as check_model_rate_limit)
        now_utc = datetime.now(timezone.utc)
        if entry["reset_at"] and now_utc >= entry["reset_at"]:
            entry["daily_submissions"] = 0
            entry["reset_at"] = get_next_midnight_utc()
        
        return _build_stats(entry)


def _build_stats(entry: Dict, limit_type: str = None, wait_seconds: int = None) -> Dict:
    """Build stats dictionary from entry."""
    # Can submit if: daily < MAX AND has credit
    daily_ok = entry["daily_submissions"] < MAX_FREE_SUBMISSIONS_PER_DAY
    has_credit = entry["submission_credits"] > 0
    
    stats = {
        "daily_submissions": entry["daily_submissions"],
        "max_daily_submissions": MAX_FREE_SUBMISSIONS_PER_DAY,
        "submission_credits": entry["submission_credits"],
        "reset_at": entry["reset_at"].isoformat() if entry["reset_at"] else None,
        # Can submit only if: daily limit not reached AND has credit (or payment will be required)
        "can_submit": daily_ok,  # Daily check - payment/credit checked separately
        "daily_limit_reached": not daily_ok,
        "has_credit": has_credit,
    }
    
    if limit_type:
        stats["limit_type"] = limit_type
    if wait_seconds:
        stats["wait_seconds"] = wait_seconds
    
    return stats


# =============================================================================
# Cleanup Task
# =============================================================================

async def model_rate_limiter_cleanup_task():
    """Background task to clean up old entries (runs every hour)."""
    print("🚀 Model rate limiter cleanup task started")
    
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            
            with _cache_lock:
                now_utc = datetime.now(timezone.utc)
                to_remove = []
                
                for hotkey, entry in _rate_limit_cache.items():
                    # Remove entries with no credits and past reset time
                    if (now_utc >= entry["reset_at"] and 
                        entry["daily_submissions"] == 0 and 
                        entry["submission_credits"] == 0):
                        to_remove.append(hotkey)
                
                for hotkey in to_remove:
                    del _rate_limit_cache[hotkey]
                
                if to_remove:
                    print(f"🧹 Cleaned up {len(to_remove)} inactive model rate limit entries")
                    
        except Exception as e:
            print(f"❌ Model rate limiter cleanup error: {e}")
            await asyncio.sleep(60)
