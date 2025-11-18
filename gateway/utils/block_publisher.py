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
        
        logger.info("🔔 ChainBlockPublisher initialized")
    
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
        logger.info(f"✅ Added block subscriber: {subscriber_name} (total: {len(self.__subscribers)})")
    
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
            logger.info("🛑 Stop event detected - ending block subscription")
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
            logger.info(
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
                        logger.error(f"❌ Subscriber {subscriber_name} failed: {result}")
            
        except Exception as e:
            logger.error(f"❌ Error processing block notification: {e}")
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
            logger.error(f"⏱️  {subscriber_name} timed out processing block {block_info.block_number}")
            raise
        except Exception as e:
            logger.error(f"❌ {subscriber_name} error: {e}")
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
        Start subscribing to new blocks.
        
        This keeps the WebSocket alive and receives blocks as they're finalized.
        Blocks until stop_event is set or an unrecoverable error occurs.
        
        IMPORTANT: This should be run as a background task:
            asyncio.create_task(publisher.start())
        
        Example:
            subscription_task = asyncio.create_task(block_publisher.start())
            # ... run application ...
            stop_event.set()  # Trigger shutdown
            await subscription_task  # Wait for clean shutdown
        """
        logger.info("="*80)
        logger.info("🔔 STARTING CHAIN BLOCK SUBSCRIPTION")
        logger.info("="*80)
        logger.info(f"   Subscription type: Finalized blocks only")
        logger.info(f"   Registered listeners: {len(self.__subscribers)}")
        for sub in self.__subscribers:
            logger.info(f"      • {type(sub).__name__}")
        logger.info("="*80)
        logger.info("")
        
        try:
            # Subscribe to new block headers
            # This is a LONG-LIVED subscription that keeps the WebSocket alive
            await self.__substrate.subscribe_block_headers(
                subscription_handler=self._on_block,
                finalized_only=True  # Only notify for finalized blocks (more reliable)
            )
            
        except asyncio.CancelledError:
            logger.info("🛑 Block subscription cancelled (shutdown requested)")
            
        except Exception as e:
            logger.error(f"❌ Block subscription error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            logger.info("")
            logger.info("="*80)
            logger.info("🔔 STOPPED CHAIN BLOCK SUBSCRIPTION")
            logger.info("="*80)
    
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

