"""
Metagraph Monitor (Block Listener)
===================================

Proactively warms metagraph cache at epoch boundaries.
Replaces the polling-based metagraph_warmer.py.

This is an event-driven architecture:
- Receives block notifications from ChainBlockPublisher
- Detects epoch transitions automatically  
- Warms metagraph cache in background (non-blocking)
- Uses single AsyncSubtensor instance (no memory leaks)

Benefits over polling:
- Instant detection of epoch changes (triggered by block event)
- No wasted CPU cycles (only runs when epoch changes)
- Uses async subtensor (no new WebSocket connections)
- Guaranteed to run at epoch boundary (can't miss transitions)
"""

import asyncio
import logging
import time

from gateway.utils.block_publisher import BlockListener, BlockInfo

logger = logging.getLogger(__name__)


class MetagraphMonitor(BlockListener):
    """
    Monitors blocks for epoch transitions and warms metagraph cache.
    
    Implements BlockListener protocol to receive block notifications.
    When epoch changes, fetches new metagraph using the shared AsyncSubtensor
    instance and updates the global cache in registry.py.
    
    Design:
    - Injected with async_subtensor at initialization (dependency injection)
    - Triggered by block events (no polling)
    - Warms cache asynchronously (doesn't block gateway)
    - Updates global cache atomically (thread-safe)
    
    Cache Strategy:
    - Fetch metagraph when epoch transitions
    - Retry up to 8 times with exponential backoff
    - Update global cache in registry.py (_metagraph_cache, _cache_epoch)
    - Requests can use old cache while new one is fetching (graceful degradation)
    """
    
    def __init__(self, async_subtensor):
        """
        Initialize metagraph monitor.
        
        Args:
            async_subtensor: Shared AsyncSubtensor instance from main.py lifespan
                            This is the ONLY subtensor instance - reused for all queries
        """
        self.async_subtensor = async_subtensor
        self.last_warmed_epoch = None
        
        logger.info("🔥 MetagraphMonitor initialized (event-driven)")
    
    async def on_block(self, block_info: BlockInfo):
        """
        Called when a new finalized block arrives.
        
        Args:
            block_info: Information about the new block
        
        Checks if epoch has changed and triggers cache warming if needed.
        Warming happens in background (non-blocking).
        """
        try:
            current_epoch = block_info.epoch_id
            
            # Initialize on first block
            if self.last_warmed_epoch is None:
                self.last_warmed_epoch = current_epoch
                logger.info(f"🔥 Metagraph monitor initialized at epoch {current_epoch}")
                return
            
            # Check if epoch changed
            if current_epoch > self.last_warmed_epoch:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔥 EPOCH TRANSITION DETECTED: {self.last_warmed_epoch} → {current_epoch}")
                logger.info(f"{'='*80}")
                logger.info(f"🔥 Triggering metagraph cache warming for epoch {current_epoch}...")
                logger.info(f"   • Using shared AsyncSubtensor (no new instances)")
                logger.info(f"   • Will retry up to 8 times if needed")
                logger.info(f"   • Meanwhile, requests will use epoch {self.last_warmed_epoch} cache")
                logger.info(f"{'='*80}\n")
                
                # Warm cache in background (don't block block processing)
                asyncio.create_task(self._warm_cache(current_epoch))
                
                self.last_warmed_epoch = current_epoch
        
        except Exception as e:
            logger.error(f"❌ Error in MetagraphMonitor.on_block: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - log and continue
    
    async def _warm_cache(self, epoch_id: int):
        """
        Fetch and cache metagraph for new epoch.
        
        This runs in the background (via asyncio.create_task) so it doesn't block
        the block publisher or other monitors.
        
        Strategy:
        - Attempts 1-4: Use shared AsyncSubtensor (fast path)
        - Attempts 5-8: Fall back to fresh sync Subtensor in thread pool (reliable path)
        
        The sync fallback runs in asyncio.to_thread() so it does NOT block the event loop.
        Gateway continues processing requests while this runs in background.
        
        Args:
            epoch_id: The new epoch to fetch metagraph for
        
        Returns:
            bool: True if successful, False if all retries failed
        """
        try:
            # Import cache globals from registry.py
            # We'll update these directly (atomic with lock)
            from gateway.utils.registry import _metagraph_cache, _cache_epoch, _cache_epoch_timestamp, _cache_lock
            from gateway.config import BITTENSOR_NETUID, BITTENSOR_NETWORK
            
            max_retries = 8
            switch_to_sync_after = 4  # Switch to sync fallback after 4 async failures
            retry_delay = 10  # Initial delay (seconds)
            timeout_per_attempt = 60  # 60 second timeout per attempt
            
            for attempt in range(1, max_retries + 1):
                try:
                    use_sync_fallback = attempt > switch_to_sync_after
                    
                    if use_sync_fallback:
                        # ════════════════════════════════════════════════════════
                        # SYNC FALLBACK: Fresh Subtensor in thread pool
                        # Runs in separate thread - does NOT block event loop
                        # ════════════════════════════════════════════════════════
                        logger.warning(f"🔥 Attempt {attempt}/{max_retries}: Using SYNC FALLBACK (thread pool)")
                        logger.warning(f"   AsyncSubtensor may be stale, using fresh Subtensor...")
                        
                        def fetch_metagraph_sync():
                            """Runs in thread pool - does not block event loop"""
                            import bittensor as bt
                            subtensor = bt.Subtensor(network=BITTENSOR_NETWORK)
                            return subtensor.metagraph(BITTENSOR_NETUID)
                        
                        # Run sync code in thread pool with timeout
                        metagraph = await asyncio.wait_for(
                            asyncio.to_thread(fetch_metagraph_sync),
                            timeout=timeout_per_attempt
                        )
                        logger.info(f"🔥 ✅ Sync fallback succeeded!")
                    else:
                        # ════════════════════════════════════════════════════════
                        # ASYNC PATH: Use shared AsyncSubtensor (fast when healthy)
                        # ════════════════════════════════════════════════════════
                        logger.info(f"🔥 Attempt {attempt}/{max_retries} for epoch {epoch_id}...")
                        logger.info(f"   Network: {self.async_subtensor.network}")
                        logger.info(f"   NetUID: {BITTENSOR_NETUID}")
                        logger.info(f"   Timeout: {timeout_per_attempt}s")
                        
                        metagraph = await asyncio.wait_for(
                            self.async_subtensor.metagraph(BITTENSOR_NETUID),
                            timeout=timeout_per_attempt
                        )
                    
                    # ════════════════════════════════════════════════════════
                    # Update global cache atomically (thread-safe)
                    # ════════════════════════════════════════════════════════
                    import gateway.utils.registry as registry_module
                    
                    with _cache_lock:
                        registry_module._metagraph_cache = metagraph
                        registry_module._cache_epoch = epoch_id
                        registry_module._cache_epoch_timestamp = time.time()
                    
                    logger.info(f"🔥 ✅ Cache warmed for epoch {epoch_id}")
                    logger.info(f"   Neurons: {len(metagraph.hotkeys)}")
                    logger.info(f"   Validators: {sum(1 for i in range(len(metagraph.validator_permit)) if metagraph.validator_permit[i])}")
                    logger.info(f"   Method: {'sync fallback' if use_sync_fallback else 'async'}")
                    
                    return True
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        method = "sync fallback" if attempt > switch_to_sync_after else "async"
                        logger.warning(f"🔥 ⚠️  Attempt {attempt}/{max_retries} ({method}) timed out after {timeout_per_attempt}s")
                        if attempt == switch_to_sync_after:
                            logger.warning(f"   → Next attempt will use SYNC FALLBACK (fresh Subtensor)")
                        else:
                            logger.warning(f"   Retrying in {retry_delay}s...")
                        
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 30)
                        continue
                    else:
                        logger.error(f"🔥 ❌ All {max_retries} attempts failed for epoch {epoch_id}")
                        logger.error(f"   Workflow will continue using epoch {epoch_id - 1} cache")
                        return False
                    
                except Exception as e:
                    if attempt < max_retries:
                        method = "sync fallback" if attempt > switch_to_sync_after else "async"
                        logger.warning(f"🔥 ⚠️  Attempt {attempt}/{max_retries} ({method}) failed: {e}")
                        if attempt == switch_to_sync_after:
                            logger.warning(f"   → Next attempt will use SYNC FALLBACK (fresh Subtensor)")
                        else:
                            logger.warning(f"   Retrying in {retry_delay}s...")
                        
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 30)
                        continue
                    else:
                        logger.error(f"🔥 ❌ All {max_retries} attempts failed for epoch {epoch_id}")
                        logger.error(f"   Last error: {e}")
                        logger.error(f"   Workflow will continue using epoch {epoch_id - 1} cache")
                        
                        import traceback
                        traceback.print_exc()
                        
                        return False
            
            # Should never reach here (loop always returns)
            return False
            
        except Exception as e:
            logger.error(f"🔥 ❌ Unexpected error in _warm_cache for epoch {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_stats(self) -> dict:
        """
        Get monitor statistics for debugging.
        
        Returns:
            Dict with monitor state
        """
        return {
            "last_warmed_epoch": self.last_warmed_epoch,
            "async_subtensor_network": self.async_subtensor.network if self.async_subtensor else None,
        }
