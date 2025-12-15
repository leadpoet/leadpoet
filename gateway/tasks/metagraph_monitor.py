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
            import bittensor as bt
            
            max_retries = 8
            retry_delay = 10  # Initial delay (seconds)
            timeout_per_attempt = 60  # 60 second timeout per attempt (matches old sync version)
            reconnect_after_failures = 3  # Reconnect WebSocket after this many failures
            consecutive_failures = 0
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"🔥 Warming attempt {attempt}/{max_retries} for epoch {epoch_id}...")
                    logger.info(f"   Network: {self.async_subtensor.network}")
                    logger.info(f"   NetUID: {BITTENSOR_NETUID}")
                    logger.info(f"   Timeout: {timeout_per_attempt}s per attempt")
                    
                    # ════════════════════════════════════════════════════════
                    # Use async_subtensor to fetch metagraph
                    # Wrap with 60-second timeout to prevent hanging
                    # ════════════════════════════════════════════════════════
                    metagraph = await asyncio.wait_for(
                        self.async_subtensor.metagraph(BITTENSOR_NETUID),
                        timeout=timeout_per_attempt
                    )
                    
                    # ════════════════════════════════════════════════════════
                    # Update global cache atomically (thread-safe)
                    # ════════════════════════════════════════════════════════
                    # Import as globals to modify them
                    import gateway.utils.registry as registry_module
                    
                    with _cache_lock:
                        registry_module._metagraph_cache = metagraph
                        registry_module._cache_epoch = epoch_id
                        registry_module._cache_epoch_timestamp = time.time()
                    
                    logger.info(f"🔥 ✅ Cache warmed for epoch {epoch_id}")
                    logger.info(f"   Neurons: {len(metagraph.hotkeys)}")
                    logger.info(f"   Validators: {sum(1 for i in range(len(metagraph.validator_permit)) if metagraph.validator_permit[i])}")
                    logger.info(f"   Cache updated atomically (thread-safe)")
                    
                    consecutive_failures = 0  # Reset on success
                    return True
                    
                except asyncio.TimeoutError:
                    consecutive_failures += 1
                    logger.warning(f"🔥 ⚠️  Warming attempt {attempt}/{max_retries} timed out after {timeout_per_attempt}s")
                    
                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(f"🔥 ⚠️  Warming attempt {attempt}/{max_retries} failed: {e}")
                
                # ════════════════════════════════════════════════════════
                # RECONNECT: If multiple consecutive failures, WebSocket may be stale
                # Create a fresh AsyncSubtensor instance
                # ════════════════════════════════════════════════════════
                if consecutive_failures >= reconnect_after_failures:
                    logger.warning(f"🔥 🔄 {consecutive_failures} consecutive failures - reconnecting WebSocket...")
                    try:
                        # Close old connection
                        try:
                            await self.async_subtensor.__aexit__(None, None, None)
                        except Exception:
                            pass  # Ignore close errors
                        
                        # Create fresh connection
                        self.async_subtensor = await asyncio.wait_for(
                            bt.AsyncSubtensor(network=BITTENSOR_NETWORK).__aenter__(),
                            timeout=30
                        )
                        logger.info(f"🔥 ✅ WebSocket reconnected to {BITTENSOR_NETWORK}")
                        consecutive_failures = 0  # Reset after reconnect
                        
                        # Also update the injected reference in registry
                        import gateway.utils.registry as registry_module
                        registry_module._async_subtensor = self.async_subtensor
                        
                    except Exception as reconnect_error:
                        logger.error(f"🔥 ❌ Reconnection failed: {reconnect_error}")
                        # Continue with retries anyway
                
                if attempt < max_retries:
                    logger.warning(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 30)  # Cap at 30s
                else:
                    # Last attempt failed
                    logger.error(f"🔥 ❌ All {max_retries} warming attempts failed for epoch {epoch_id}")
                    logger.error(f"   Workflow will continue using epoch {epoch_id - 1} cache as fallback")
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
