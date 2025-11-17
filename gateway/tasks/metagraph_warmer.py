"""
Background Metagraph Warmer Task

Proactively fetches the metagraph at epoch boundaries to ensure zero-downtime
during epoch transitions.

Design:
- Polls for epoch changes every 30 seconds
- When new epoch detected, immediately starts async metagraph fetch
- During fetch, requests use old epoch's cache (graceful degradation)
- Once fetched, seamlessly switches to new cache

This prevents miners/validators from experiencing timeouts during epoch transitions.
"""

import asyncio
import threading
from datetime import datetime


# Flag to track if warming is in progress
_warming_in_progress = False
_warming_lock = threading.Lock()


async def metagraph_warmer_task():
    """
    Background task that proactively warms the metagraph cache at epoch boundaries.
    
    This ensures that when a new epoch starts, the metagraph is already being fetched
    in the background, allowing requests to continue using the old epoch's cache
    until the new one is ready.
    """
    print("\n" + "="*80)
    print("🔥 METAGRAPH WARMER: Starting background task")
    print("="*80 + "\n")
    
    # Import here to avoid circular dependency
    from gateway.utils.epoch import get_current_epoch_id
    from gateway.utils.registry import warm_metagraph_cache
    
    # CRITICAL: Declare global to avoid UnboundLocalError when checking/modifying
    global _warming_in_progress
    
    last_checked_epoch = None
    
    while True:
        try:
            # Sleep first to give the gateway time to start
            await asyncio.sleep(30)
            
            # Get current epoch
            try:
                current_epoch = get_current_epoch_id()
            except Exception as e:
                print(f"🔥 Metagraph warmer: Failed to get current epoch: {e}")
                print(f"   Will retry in 30s...")
                continue
            
            # Initialize on first run
            if last_checked_epoch is None:
                last_checked_epoch = current_epoch
                print(f"🔥 Metagraph warmer initialized at epoch {current_epoch}")
                continue
            
            # Check if epoch changed
            if current_epoch != last_checked_epoch:
                print(f"\n{'='*80}")
                print(f"🔥 EPOCH TRANSITION DETECTED: {last_checked_epoch} → {current_epoch}")
                print(f"{'='*80}")
                print(f"🔥 Starting proactive metagraph fetch for epoch {current_epoch}...")
                print(f"   • Will attempt up to 8 times with 60s timeout per attempt")
                print(f"   • Meanwhile, requests will use epoch {last_checked_epoch} cache")
                print(f"   • Workflow continues even if all warming attempts fail")
                print(f"{'='*80}\n")
                
                # Start warming the cache in a separate thread to avoid blocking
                # This allows the main event loop to continue serving requests
                with _warming_lock:
                    if not _warming_in_progress:
                        _warming_in_progress = True
                        
                        # Run warming in executor (separate thread)
                        loop = asyncio.get_event_loop()
                        loop.run_in_executor(None, _warm_cache_sync, current_epoch)
                
                # Update last checked epoch
                last_checked_epoch = current_epoch
            
        except Exception as e:
            print(f"🔥 Metagraph warmer error: {e}")
            import traceback
            traceback.print_exc()
            # Continue running despite errors


def _warm_cache_sync(target_epoch: int):
    """
    Synchronous wrapper for warming cache (runs in thread pool).
    
    Args:
        target_epoch: The epoch to fetch metagraph for
    """
    global _warming_in_progress
    
    try:
        from gateway.utils.registry import warm_metagraph_cache
        
        print(f"🔥 [Thread] Starting metagraph fetch for epoch {target_epoch}...")
        start_time = datetime.now()
        
        success = warm_metagraph_cache(target_epoch)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if success:
            print(f"🔥 [Thread] ✅ Metagraph warming complete for epoch {target_epoch} ({elapsed:.1f}s)")
            print(f"🔥 [Thread] All requests will now seamlessly switch to epoch {target_epoch} cache")
        else:
            print(f"🔥 [Thread] ⚠️  All 8 warming attempts failed for epoch {target_epoch} ({elapsed:.1f}s)")
            print(f"🔥 [Thread] Workflow continues: requests use epoch {target_epoch - 1} cache as fallback")
            print(f"🔥 [Thread] This is safe - miners/validators can still operate normally")
            
    except Exception as e:
        print(f"🔥 [Thread] ❌ Metagraph warming error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Reset flag
        with _warming_lock:
            _warming_in_progress = False

