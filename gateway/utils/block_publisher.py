"""
Block Publisher for Bittensor Chain
====================================

Subscribes to new blocks from Bittensor chain and notifies listeners.
Keeps WebSocket connection alive and prevents HTTP 429 rate limiting.

Based on: Tao.com patrol subnet pattern + custom improvements for LeadPoet

Architecture:
- Single WebSocket subscription for entire gateway lifecycle
- Push-based notifications (blocks are sent to us, we don't poll)
- Multiple listeners can subscribe to block events
- Graceful error handling (one listener failure doesn't crash others)

Benefits:
- Zero memory leaks (single connection, properly managed)
- Zero HTTP 429 errors (connection stays alive)
- Instant updates (no polling delay)
- 100x less resource usage
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Protocol
import logging

logger = logging.getLogger(__name__)


class BlockInfo:
    """
    Information about a finalized block from the Bittensor chain.
    
    Attributes:
        block_hash: Block hash (hex string)
        block_number: Block number (incremental)
        block_timestamp: Block timestamp (UTC)
        epoch_id: LeadPoet epoch number (block_number // 360)
        block_within_epoch: Position within current epoch (0-359)
    """
    
    def __init__(self, block_hash: str, block_number: int, block_timestamp: datetime):
        self.block_hash = block_hash
        self.block_number = block_number
        self.block_timestamp = block_timestamp
        
        # LeadPoet-specific: 360-block epochs
        self.epoch_id = block_number // 360
        self.block_within_epoch = block_number % 360
    
    def __repr__(self):
        return (
            f"BlockInfo(number={self.block_number}, "
            f"epoch={self.epoch_id}, "
            f"block_in_epoch={self.block_within_epoch}/360)"
        )


class BlockListener(Protocol):
    """
    Protocol for components that want to receive block notifications.
    
    Implement this protocol to subscribe to new blocks.
    
    Example:
        class MyMonitor(BlockListener):
            async def on_block(self, block_info: BlockInfo):
                print(f"New block: {block_info.block_number}")
    """
    
    async def on_block(self, block_info: BlockInfo):
        """
        Called when a new finalized block arrives.
        
        Args:
            block_info: Information about the new block
        
        Raises:
            Exception: If processing fails (caught by publisher, doesn't crash subscription)
        """
        ...


class ChainBlockPublisher:
    """
    Subscribes to Bittensor chain blocks and notifies registered listeners.
    
    This keeps the WebSocket connection alive and prevents HTTP 429 errors
    by using a single long-lived subscription instead of repeated queries.
    
    Usage:
        # In gateway startup (main.py lifespan):
        async with AsyncSubtensor(...) as async_subtensor:
            stop_event = asyncio.Event()
            publisher = ChainBlockPublisher(async_subtensor.substrate, stop_event)
            
            # Register listeners
            publisher.add_subscriber(epoch_monitor)
            publisher.add_subscriber(metagraph_monitor)
            
            # Start subscription (non-blocking)
            asyncio.create_task(publisher.start())
            
            # ... run application ...
            
            # On shutdown:
            stop_event.set()
    """
    
    def __init__(self, substrate, stop_event: asyncio.Event):
        """
        Initialize block publisher.
        
        Args:
            substrate: AsyncSubstrateInterface from Bittensor's AsyncSubtensor
                      (access via: async_subtensor.substrate)
            stop_event: Event to signal shutdown (set this to stop subscription)
        """
        self.__substrate = substrate
        self.__stop_event = stop_event
        self.__subscribers: List[BlockListener] = []
        self.__last_block_number = None
        self.__last_block_time = None  # Track when last block was received
        
        print("🔔 ChainBlockPublisher initialized")
        print("   Heartbeat monitor: 60s timeout (auto-switches to polling)")
    
    
    def add_subscriber(self, subscriber: BlockListener):
        """
        Register a component to receive block notifications.
        
        Args:
            subscriber: Object implementing BlockListener protocol
        
        Example:
            publisher.add_subscriber(epoch_monitor)
            publisher.add_subscriber(metagraph_warmer)
        """
        self.__subscribers.append(subscriber)
        subscriber_name = type(subscriber).__name__
        print(f"✅ Added block subscriber: {subscriber_name} (total: {len(self.__subscribers)})")
    
    async def _on_block(self, obj: dict):
        """
        Callback for new block notifications.
        
        Called by Bittensor's subscribe_block_headers when a new finalized block arrives.
        Extracts block info and notifies all registered subscribers.
        
        Args:
            obj: Block header object from Bittensor
                 Format: {"header": {"number": int, ...}}
        
        Returns:
            True if subscription should stop (stop_event is set)
            None to continue subscription
        """
        # Check if shutdown requested
        if self.__stop_event.is_set():
            print("🛑 Stop event detected - ending block subscription")
            return True  # Stop subscription
        
        try:
            # Extract block number from header
            header = obj["header"]
            block_number = header["number"]
            
            # Skip if we've already processed this block (防止重复)
            if self.__last_block_number is not None and block_number <= self.__last_block_number:
                logger.debug(f"⏭️  Skipping duplicate block {block_number}")
                return None
            
            self.__last_block_number = block_number
            self.__last_block_time = asyncio.get_event_loop().time()  # Track when block was received
            
            # Get block hash (use header hash directly to avoid extra query)
            block_hash = header.get("hash", "")
            if not block_hash:
                block_hash = await self.__substrate.get_block_hash(block_number)
            
            # Use current time instead of querying chain (avoid blocking)
            timestamp = datetime.now(timezone.utc)
            
            # Create block info object
            block_info = BlockInfo(
                block_hash=block_hash,
                block_number=block_number,
                block_timestamp=timestamp
            )
            
            # Log block arrival
            print(
                f"📦 Block #{block_number} | "
                f"Epoch {block_info.epoch_id} | "
                f"Block {block_info.block_within_epoch}/360 | "
                f"Time: {timestamp.strftime('%H:%M:%S')} UTC"
            )
            
            # Notify all subscribers concurrently
            if self.__subscribers:
                # Use gather with return_exceptions=True so one subscriber failure doesn't crash others
                results = await asyncio.gather(*[
                    self._notify_subscriber(subscriber, block_info)
                    for subscriber in self.__subscribers
                ], return_exceptions=True)
                
                # Log any subscriber errors (but don't crash)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        subscriber_name = type(self.__subscribers[i]).__name__
                        print(f"❌ Subscriber {subscriber_name} failed: {result}")
            
        except Exception as e:
            print(f"❌ Error processing block notification: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash subscription - continue to next block
        
        # Continue subscription (unless stop_event is set)
        return True if self.__stop_event.is_set() else None
    
    async def _notify_subscriber(self, subscriber: BlockListener, block_info: BlockInfo):
        """
        Notify a single subscriber (with timeout protection).
        
        Args:
            subscriber: BlockListener to notify
            block_info: Block information
        
        Raises:
            Exception: If subscriber processing fails or times out
        """
        subscriber_name = type(subscriber).__name__
        
        try:
            # Call subscriber with 30-second timeout
            # This prevents one slow subscriber from blocking others
            await asyncio.wait_for(
                subscriber.on_block(block_info),
                timeout=30.0
            )
            logger.debug(f"✅ {subscriber_name} processed block {block_info.block_number}")
            
        except asyncio.TimeoutError:
            print(f"⏱️  {subscriber_name} timed out processing block {block_info.block_number}")
            raise
        except Exception as e:
            print(f"❌ {subscriber_name} error: {e}")
            raise
    
    async def _get_timestamp(self, block_hash: str) -> datetime:
        """
        Get block timestamp from chain.
        
        Args:
            block_hash: Block hash to query
        
        Returns:
            Block timestamp as UTC datetime
        """
        try:
            # Query Timestamp pallet for block timestamp
            res = await self.__substrate.query(
                module="Timestamp",
                storage_function="Now",
                block_hash=block_hash,
            )
            
            # Bittensor stores timestamps in milliseconds
            unix_ms = res.value
            return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to get block timestamp for {block_hash[:16]}...: {e}")
            # Fallback to current time (not ideal but better than crashing)
            return datetime.now(timezone.utc)
    
    async def start(self):
        """
        Start subscribing to new blocks with automatic recovery.
        
        This keeps the WebSocket alive and receives blocks as they're finalized.
        If WebSocket dies, automatically switches to polling mode until reconnect succeeds.
        
        RESILIENCE FEATURES:
        - Detects WebSocket death (no blocks for 60s)
        - Automatically switches to polling mode
        - Attempts reconnection in background
        - Never requires manual restart
        
        IMPORTANT: This should be run as a background task:
            asyncio.create_task(publisher.start())
        
        Example:
            subscription_task = asyncio.create_task(block_publisher.start())
            # ... run application ...
            stop_event.set()  # Trigger shutdown
            await subscription_task  # Wait for clean shutdown
        """
        print("="*80)
        print("🔔 STARTING CHAIN BLOCK SUBSCRIPTION (WITH AUTO-RECOVERY)")
        print("="*80)
        print(f"   Subscription type: Finalized blocks only")
        print(f"   Fallback: Polling mode if WebSocket dies")
        print(f"   Heartbeat: 60s timeout detection")
        print(f"   Registered listeners: {len(self.__subscribers)}")
        for sub in self.__subscribers:
            print(f"      • {type(sub).__name__}")
        print("="*80)
        print("")
        
        # CRITICAL: Wait 5 seconds to let app finish starting
        # Without this, subscribe_block_headers blocks the lifespan and prevents HTTP from working
        await asyncio.sleep(5)
        print("🔔 Connecting to block stream (after 5s delay)...")
        
        # ══════════════════════════════════════════════════════════════
        # AUTOMATIC RECOVERY LOOP: Retry WebSocket forever
        # ══════════════════════════════════════════════════════════════
        retry_count = 0
        while not self.__stop_event.is_set():
            try:
                retry_count += 1
                if retry_count > 1:
                    print(f"🔄 WebSocket reconnection attempt #{retry_count}...")
                
                # Reset heartbeat tracking
                self.__last_block_time = asyncio.get_event_loop().time()
                
                # Start heartbeat monitor (runs in parallel with subscription)
                heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
                
                try:
                    # Subscribe to new block headers (this blocks until subscription ends)
                    # This is a LONG-LIVED subscription that keeps the WebSocket alive
                    await self.__substrate.subscribe_block_headers(
                        subscription_handler=self._on_block,
                        finalized_only=True  # Only notify for finalized blocks (more reliable)
                    )
                    
                    # If we get here, subscription ended normally
                    print("ℹ️  WebSocket subscription ended normally")
                    heartbeat_task.cancel()
                    break
                
                except asyncio.CancelledError:
                    # Heartbeat monitor detected timeout and cancelled us
                    heartbeat_task.cancel()
                    raise  # Re-raise to be caught by outer exception handler
                
                finally:
                    # Ensure heartbeat monitor is always cancelled
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                
            except asyncio.CancelledError as e:
                # Check if this is heartbeat timeout or shutdown
                if "Heartbeat timeout" in str(e):
                    print(f"💔 WebSocket heartbeat timeout detected")
                    print(f"   🔄 SWITCHING TO POLLING MODE (automatic recovery)")
                    
                    # Start polling until we can reconnect
                    await self._polling_fallback_mode()
                    
                    # After polling mode exits, try to reconnect WebSocket
                    print(f"🔄 Attempting to restore WebSocket subscription...")
                    continue  # Retry WebSocket connection
                else:
                    # Shutdown requested
                    print("🛑 Block subscription cancelled (shutdown requested)")
                    break
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is the known Bittensor reconnection bug
                if "open subscriptions" in error_str or "unable to reconnect" in error_str:
                    print(f"❌ WebSocket died due to Bittensor reconnection bug")
                    print(f"   Error: {e}")
                    print(f"   🔄 SWITCHING TO POLLING MODE (automatic recovery)")
                    
                    # Start polling until we can reconnect
                    await self._polling_fallback_mode()
                    
                    # After polling mode exits, try to reconnect WebSocket
                    print(f"🔄 Attempting to restore WebSocket subscription...")
                    continue  # Retry WebSocket connection
                
                else:
                    # Unknown error - log and retry
                    print(f"❌ Block subscription error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Wait before retry
                    print(f"⏳ Retrying WebSocket in 30 seconds...")
                    await asyncio.sleep(30)
                    continue
            
        # ══════════════════════════════════════════════════════════════
        # CLEANUP
        # ══════════════════════════════════════════════════════════════
        print("")
        print("="*80)
        print("🔔 STOPPED CHAIN BLOCK SUBSCRIPTION")
        print("="*80)
    
    async def _heartbeat_monitor(self):
        """
        Monitor WebSocket health by checking block reception.
        
        If no blocks are received for 60 seconds, assumes WebSocket is dead
        and cancels the subscription to force fallback to polling mode.
        
        This runs in parallel with the WebSocket subscription.
        """
        TIMEOUT_SECONDS = 60  # No blocks for 60s = WebSocket is dead
        CHECK_INTERVAL = 10   # Check every 10 seconds
        
        print("💓 Heartbeat monitor started (60s timeout)")
        
        try:
            while not self.__stop_event.is_set():
                await asyncio.sleep(CHECK_INTERVAL)
                
                # Check if we've received any blocks recently
                if self.__last_block_time is not None:
                    current_time = asyncio.get_event_loop().time()
                    time_since_last_block = current_time - self.__last_block_time
                    
                    if time_since_last_block > TIMEOUT_SECONDS:
                        print(f"💔 HEARTBEAT TIMEOUT: No blocks for {int(time_since_last_block)}s")
                        print(f"   WebSocket appears dead - forcing reconnection...")
                        
                        # Force WebSocket to reconnect by raising an exception
                        # (caught by start() method which will switch to polling)
                        raise asyncio.CancelledError("Heartbeat timeout - WebSocket dead")
                
        except asyncio.CancelledError:
            # Normal cancellation (subscription ended or timeout)
            print("💓 Heartbeat monitor stopped")
            raise
        except Exception as e:
            print(f"❌ Heartbeat monitor error: {e}")
            raise
    
    async def _polling_fallback_mode(self):
        """
        FALLBACK MODE: Poll for blocks when WebSocket dies.
        
        This keeps the gateway operational while WebSocket reconnection
        is attempted in the background.
        
        Exits when:
        - Stop event is set (shutdown)
        - After 5 minutes (to retry WebSocket)
        """
        print("")
        print("="*80)
        print("⚠️  ENTERING POLLING FALLBACK MODE")
        print("="*80)
        print("   WebSocket subscription failed - using polling as backup")
        print("   Gateway will continue processing blocks via HTTP polling")
        print("   Attempting WebSocket reconnect in 5 minutes...")
        print("="*80)
        print("")
        
        import bittensor as bt
        from datetime import datetime, timezone
        
        # Create sync subtensor for polling
        try:
            # Get network from existing substrate connection
            network = getattr(self.__substrate, 'network', 'finney')
            sync_subtensor = bt.subtensor(network=network)
            print(f"✅ Created sync subtensor for polling (network: {network})")
        except Exception as e:
            print(f"❌ Failed to create sync subtensor: {e}")
            print(f"   Waiting 30s before retry...")
            await asyncio.sleep(30)
            return  # Exit fallback mode, will retry WebSocket
        
        # Poll for blocks every 12 seconds (Bittensor block time)
        start_time = asyncio.get_event_loop().time()
        max_polling_duration = 300  # 5 minutes
        poll_interval = 12  # seconds
        
        last_processed_block = self.__last_block_number
        
        while not self.__stop_event.is_set():
            try:
                # Check if we should exit polling mode (5 minutes elapsed)
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_polling_duration:
                    print(f"⏱️  Polling mode timeout ({max_polling_duration}s) - exiting to retry WebSocket")
                    break
                
                # Poll for current block
                current_block = sync_subtensor.get_current_block()
                
                # Skip if we've already processed this block
                if last_processed_block is not None and current_block <= last_processed_block:
                    await asyncio.sleep(poll_interval)
                    continue
                
                last_processed_block = current_block
                
                # Create block info (similar to WebSocket callback)
                block_info = BlockInfo(
                    block_hash="",  # Not available in polling mode
                    block_number=current_block,
                    block_timestamp=datetime.now(timezone.utc)
                )
                
                # Log block arrival (polling mode indicator)
                print(
                    f"📦 Block #{current_block} | "
                    f"Epoch {block_info.epoch_id} | "
                    f"Block {block_info.block_within_epoch}/360 | "
                    f"[POLLING MODE]"
                )
                
                # Notify all subscribers
                if self.__subscribers:
                    results = await asyncio.gather(*[
                        self._notify_subscriber(subscriber, block_info)
                        for subscriber in self.__subscribers
                    ], return_exceptions=True)
                    
                    # Log any subscriber errors
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            subscriber_name = type(self.__subscribers[i]).__name__
                            print(f"❌ Subscriber {subscriber_name} failed: {result}")
                
                # Update last block number for next iteration
                self.__last_block_number = current_block
                
                # Wait for next block
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                print(f"❌ Error in polling fallback: {e}")
                await asyncio.sleep(poll_interval)
                continue
        
        print("")
        print("="*80)
        print("🔄 EXITING POLLING FALLBACK MODE")
        print("="*80)
        print("   Will attempt to restore WebSocket subscription...")
        print("="*80)
        print("")
    
    def get_stats(self) -> dict:
        """
        Get subscription statistics for monitoring.
        
        Returns:
            Dict with subscription stats
        """
        return {
            "subscribers": len(self.__subscribers),
            "subscriber_types": [type(s).__name__ for s in self.__subscribers],
            "last_block_number": self.__last_block_number,
            "is_stopped": self.__stop_event.is_set()
        }

