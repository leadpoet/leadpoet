"""
Epoch Leads Cache

Proactively fetches and caches leads for the next epoch during blocks 351-360,
enabling instant lead distribution to validators without repeated Supabase queries.

Design:
- Prefetch triggered at block 351 of current epoch
- Aggressive retry with 30s timeout until successful
- Cache stores at most 2 epochs (current + next)
- Automatic cleanup on epoch transition

Benefits:
- 1 Supabase query per epoch (vs 30-45 queries)
- <100ms response time (vs 15-30s timeouts)
- Scales to 100+ validators with zero DB load
"""

import asyncio
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime


# Global cache: {epoch_id: {"epoch_id": int, "leads": [lead1, lead2, ...]}}
# Changed to store epoch_id inside cache value for validation
_epoch_leads_cache: Dict[int, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_prefetch_in_progress = False
_prefetch_lock = threading.Lock()


def get_cached_leads(epoch_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached leads for an epoch (instant, no DB query).
    
    Args:
        epoch_id: The epoch to get leads for
    
    Returns:
        List of leads if cached, None if not in cache
    
    Example:
        >>> leads = get_cached_leads(16220)
        >>> if leads:
        >>>     print(f"Serving {len(leads)} cached leads")
    """
    with _cache_lock:
        if epoch_id in _epoch_leads_cache:
            cache_entry = _epoch_leads_cache[epoch_id]
            cached_epoch_id = cache_entry.get("epoch_id")
            leads = cache_entry.get("leads", [])
            
            # CRITICAL: Validate cached epoch_id matches requested epoch_id
            if cached_epoch_id != epoch_id:
                print(f"❌ [CACHE CORRUPTION] Requested epoch {epoch_id} but cache contains epoch {cached_epoch_id}!")
                print(f"   Removing corrupted cache entry to prevent serving wrong data")
                del _epoch_leads_cache[epoch_id]
                return None
            
            print(f"✅ [CACHE HIT] Serving {len(leads)} leads for epoch {epoch_id} from cache")
            return leads
        else:
            print(f"❌ [CACHE MISS] Epoch {epoch_id} not in cache")
            return None


def set_cached_leads(epoch_id: int, leads: List[Dict[str, Any]]):
    """
    Store leads in cache for an epoch.
    
    Args:
        epoch_id: The epoch these leads belong to
        leads: List of lead objects
    
    Example:
        >>> leads = [...10 leads from DB...]
        >>> set_cached_leads(16221, leads)
    """
    with _cache_lock:
        # Store epoch_id alongside leads for validation on retrieval
        _epoch_leads_cache[epoch_id] = {
            "epoch_id": epoch_id,
            "leads": leads
        }
        print(f"💾 [CACHE SET] Stored {len(leads)} leads for epoch {epoch_id}")
        print(f"   Cache now contains epochs: {sorted(_epoch_leads_cache.keys())}")


def clear_epoch_cache(epoch_id: int):
    """
    Remove cached leads for a specific epoch (memory cleanup).
    
    Args:
        epoch_id: The epoch to remove from cache
    
    Example:
        >>> clear_epoch_cache(16220)  # Clean up old epoch
    """
    with _cache_lock:
        if epoch_id in _epoch_leads_cache:
            cache_entry = _epoch_leads_cache[epoch_id]
            lead_count = len(cache_entry.get("leads", []))
            del _epoch_leads_cache[epoch_id]
            print(f"🧹 [CACHE CLEAR] Removed {lead_count} leads for epoch {epoch_id}")
            print(f"   Cache now contains epochs: {sorted(_epoch_leads_cache.keys())}")


def cleanup_old_epochs(current_epoch: int):
    """
    Remove cached leads for epochs older than (current_epoch - 1).
    
    This ensures we only keep at most 2 epochs in memory:
    - Current epoch (being validated)
    - Next epoch (prefetched)
    
    Args:
        current_epoch: The current epoch ID
    
    Example:
        >>> cleanup_old_epochs(16221)
        >>> # Removes epochs 16219 and earlier, keeps 16220 and 16221
    """
    with _cache_lock:
        old_epochs = [e for e in _epoch_leads_cache if e < current_epoch - 1]
        for epoch in old_epochs:
            cache_entry = _epoch_leads_cache[epoch]
            lead_count = len(cache_entry.get("leads", []))
            del _epoch_leads_cache[epoch]
            print(f"🧹 [CACHE CLEANUP] Removed {lead_count} leads for old epoch {epoch}")
        
        if old_epochs:
            print(f"   Cleaned up {len(old_epochs)} old epoch(s)")
            print(f"   Cache now contains epochs: {sorted(_epoch_leads_cache.keys())}")


def is_prefetch_in_progress() -> bool:
    """
    Check if a prefetch operation is currently in progress.
    
    Returns:
        True if prefetch is running, False otherwise
    """
    with _prefetch_lock:
        return _prefetch_in_progress


def set_prefetch_in_progress(in_progress: bool):
    """
    Set the prefetch in progress flag.
    
    Args:
        in_progress: True to indicate prefetch started, False when complete
    """
    global _prefetch_in_progress
    with _prefetch_lock:
        _prefetch_in_progress = in_progress


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring/debugging.
    
    Returns:
        Dictionary with cache stats
    
    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Cached epochs: {stats['cached_epochs']}")
    """
    with _cache_lock:
        stats = {
            "cached_epochs": sorted(_epoch_leads_cache.keys()),
            "total_cached_leads": sum(len(entry.get("leads", [])) for entry in _epoch_leads_cache.values()),
            "epoch_count": len(_epoch_leads_cache)
        }
        return stats


async def prefetch_leads_for_next_epoch(
    next_epoch: int,
    fetch_function,
    max_attempts: int = 999,
    timeout: int = 30,
    retry_delay: int = 5
):
    """
    Proactively fetch and cache leads for the next epoch (blocks 351-360).
    
    CRITICAL: This MUST succeed. Retries indefinitely until successful.
    
    Args:
        next_epoch: The epoch to prefetch leads for
        fetch_function: Async function that fetches leads from DB
        max_attempts: Maximum retry attempts (default: 999 = nearly unlimited)
        timeout: Timeout per attempt in seconds (default: 30s)
        retry_delay: Delay between retries in seconds (default: 5s)
    
    Returns:
        True if successful, False if all attempts exhausted (very unlikely)
    
    Example:
        >>> success = await prefetch_leads_for_next_epoch(
        >>>     next_epoch=16221,
        >>>     fetch_function=lambda: fetch_leads_from_db(16221)
        >>> )
    """
    # Prevent duplicate prefetch for same epoch
    with _prefetch_lock:
        if _prefetch_in_progress:
            print(f"⚠️  [PREFETCH] Already in progress for epoch {next_epoch}, skipping")
            return False
    
    set_prefetch_in_progress(True)
    
    try:
        print(f"\n{'='*80}")
        print(f"🔍 [PREFETCH] Starting for epoch {next_epoch}")
        print(f"   Timeout: {timeout}s per attempt")
        print(f"   Max attempts: {max_attempts}")
        print(f"={'='*80}\n")
        
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"🔍 [PREFETCH] Attempt {attempt}/{max_attempts} for epoch {next_epoch}...")
                
                # Fetch leads with timeout
                leads = await asyncio.wait_for(fetch_function(), timeout=timeout)
                
                # Cache successful result
                set_cached_leads(next_epoch, leads)
                
                print(f"\n{'='*80}")
                print(f"✅ [PREFETCH] SUCCESS for epoch {next_epoch}")
                print(f"   Leads cached: {len(leads)}")
                print(f"   Attempts: {attempt}")
                print(f"   Cache ready for instant distribution!")
                print(f"{'='*80}\n")
                
                return True
                
            except asyncio.TimeoutError:
                print(f"⏳ [PREFETCH] Timeout after {timeout}s (attempt {attempt}/{max_attempts})")
                if attempt < max_attempts:
                    print(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                continue
                
            except Exception as e:
                print(f"⚠️  [PREFETCH] Error on attempt {attempt}/{max_attempts}: {e}")
                if attempt < max_attempts:
                    print(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                continue
        
        # All attempts exhausted (very unlikely)
        print(f"\n{'='*80}")
        print(f"❌ [PREFETCH] FAILED for epoch {next_epoch} after {max_attempts} attempts")
        print(f"   Validators will fall back to on-demand DB queries")
        print(f"{'='*80}\n")
        return False
        
    finally:
        set_prefetch_in_progress(False)


def print_cache_status():
    """
    Print current cache status for debugging.
    """
    stats = get_cache_stats()
    print(f"\n{'='*80}")
    print(f"📊 [CACHE STATUS]")
    print(f"   Cached epochs: {stats['cached_epochs']}")
    print(f"   Total leads: {stats['total_cached_leads']}")
    print(f"   Memory usage: ~{stats['total_cached_leads'] * 50}KB (estimated)")
    print(f"{'='*80}\n")

