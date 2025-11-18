"""
Epoch Monitor (Block Listener)
===============================

Subscribes to new blocks and triggers epoch lifecycle events.
Replaces the polling-based epoch_lifecycle.py background task.

This is an event-driven architecture:
- Receives block notifications from ChainBlockPublisher (PUSH, not POLL)
- Detects epoch transitions automatically
- Triggers epoch start, validation end, and consensus phases
- Zero CPU overhead when blocks aren't arriving (vs polling every 30s)

Benefits over polling:
- Instant detection of epoch transitions (no 30s delay)
- Zero CPU usage between blocks
- No risk of missed epochs (guaranteed notification for every block)
"""

import asyncio
import logging
from datetime import datetime

from gateway.utils.block_publisher import BlockListener, BlockInfo

logger = logging.getLogger(__name__)


class EpochMonitor(BlockListener):
    """
    Monitors blocks for epoch transitions and triggers lifecycle events.
    
    Implements BlockListener protocol to receive block notifications from
    ChainBlockPublisher. Triggered by new blocks (push-based), not polling.
    
    Responsibilities:
    - Detect new epochs (block 0 of new epoch)
    - Log EPOCH_INITIALIZATION event
    - Detect validation end (block 360 = epoch close)
    - Log EPOCH_END and EPOCH_INPUTS events
    - Trigger reveal phase for closed epochs
    - Compute consensus for revealed epochs
    
    State Management:
    - last_epoch: Last epoch we've seen (for transition detection)
    - validation_ended_epochs: Set of epochs we've logged EPOCH_END for
    - closed_epochs: Set of epochs we've processed consensus for
    """
    
    def __init__(self):
        """Initialize epoch monitor with empty state."""
        self.last_epoch = None
        self.validation_ended_epochs = set()
        self.closed_epochs = set()
        
        logger.info("🔄 EpochMonitor initialized (event-driven)")
    
    async def on_block(self, block_info: BlockInfo):
        """
        Called when a new finalized block arrives.
        
        Args:
            block_info: Information about the new block
        
        This method:
        1. Checks if epoch has changed (new epoch start)
        2. Checks if validation phase ended (block 0 of next epoch)
        3. Checks if epoch is closed and needs reveals/consensus
        
        All checks are non-blocking and independent.
        """
        try:
            current_epoch = block_info.epoch_id
            block_within_epoch = block_info.block_within_epoch
            
            # ════════════════════════════════════════════════════════════════
            # Check 1: New epoch started (block 0, or first time seeing this epoch)
            # ════════════════════════════════════════════════════════════════
            if self.last_epoch is None or current_epoch > self.last_epoch:
                logger.info(f"\n{'='*80}")
                logger.info(f"🚀 EPOCH TRANSITION DETECTED: {self.last_epoch} → {current_epoch}")
                logger.info(f"{'='*80}")
                
                # Trigger epoch start (non-blocking)
                asyncio.create_task(self._on_epoch_start(current_epoch))
                
                self.last_epoch = current_epoch
            
            # ════════════════════════════════════════════════════════════════
            # Check 2: Validation phase ended (block 0 of NEXT epoch = block 360 of PREVIOUS)
            # ════════════════════════════════════════════════════════════════
            if block_within_epoch == 0 and current_epoch > 0:
                previous_epoch = current_epoch - 1
                
                if previous_epoch not in self.validation_ended_epochs:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"⏰ VALIDATION PHASE ENDED: Epoch {previous_epoch}")
                    logger.info(f"{'='*80}")
                    
                    # Trigger validation end (non-blocking)
                    asyncio.create_task(self._on_validation_end(previous_epoch))
                    
                    self.validation_ended_epochs.add(previous_epoch)
            
            # ════════════════════════════════════════════════════════════════
            # Check 3: Epoch closed and needs reveal/consensus (check previous epochs)
            # ════════════════════════════════════════════════════════════════
            # Check up to 10 epochs back for any that need consensus
            for check_epoch in range(max(0, current_epoch - 10), current_epoch):
                if check_epoch not in self.closed_epochs:
                    # Trigger reveal check (non-blocking)
                    asyncio.create_task(self._check_for_reveals(check_epoch))
        
        except Exception as e:
            logger.error(f"❌ Error in EpochMonitor.on_block: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - log error and continue
    
    async def _on_epoch_start(self, epoch_id: int):
        """
        Handle new epoch start.
        
        Triggers:
        - EPOCH_INITIALIZATION event logging
        - Lead cache cleanup (remove old epochs)
        
        Args:
            epoch_id: The new epoch that just started
        """
        try:
            logger.info(f"🚀 Processing epoch start: {epoch_id}")
            
            # Import lifecycle functions (reuse existing code)
            from gateway.tasks.epoch_lifecycle import compute_and_log_epoch_initialization
            from gateway.utils.epoch import get_epoch_start_time_async, get_epoch_end_time_async, get_epoch_close_time_async
            from gateway.utils.leads_cache import cleanup_old_epochs
            
            # Calculate epoch boundaries using async versions
            epoch_start = await get_epoch_start_time_async(epoch_id)
            epoch_end = await get_epoch_end_time_async(epoch_id)
            epoch_close = await get_epoch_close_time_async(epoch_id)
            
            logger.info(f"   Start: {epoch_start.isoformat()}")
            logger.info(f"   End (validation): {epoch_end.isoformat()}")
            logger.info(f"   Close: {epoch_close.isoformat()}")
            
            # Log EPOCH_INITIALIZATION to transparency log
            await compute_and_log_epoch_initialization(epoch_id, epoch_start, epoch_end, epoch_close)
            
            # Clean up old epoch cache (keep only current + next)
            cleanup_old_epochs(epoch_id)
            
            logger.info(f"✅ Epoch {epoch_id} initialized")
            
        except Exception as e:
            logger.error(f"❌ Error handling epoch start for {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - log and continue
    
    async def _on_validation_end(self, epoch_id: int):
        """
        Handle validation phase end (block 360 reached).
        
        Triggers:
        - EPOCH_END event logging
        - EPOCH_INPUTS event logging (hash of all events in epoch)
        
        Args:
            epoch_id: The epoch whose validation phase just ended
        """
        try:
            logger.info(f"⏰ Processing validation end: {epoch_id}")
            
            # Import lifecycle functions
            from gateway.tasks.epoch_lifecycle import compute_and_log_epoch_inputs, log_epoch_event
            from gateway.utils.epoch import get_epoch_end_time_async, get_epoch_close_time_async
            
            epoch_end = await get_epoch_end_time_async(epoch_id)
            epoch_close = await get_epoch_close_time_async(epoch_id)
            
            logger.info(f"   Ended at: {epoch_end.isoformat()}")
            logger.info(f"   Epoch closed at: {epoch_close.isoformat()}")
            
            # Log EPOCH_END event
            await log_epoch_event("EPOCH_END", epoch_id, {
                "epoch_id": epoch_id,
                "end_time": epoch_end.isoformat(),
                "phase": "epoch_ended"
            })
            
            # Compute and log EPOCH_INPUTS (hash of all events during epoch)
            await compute_and_log_epoch_inputs(epoch_id)
            
            logger.info(f"✅ Epoch {epoch_id} validation phase complete")
            
        except Exception as e:
            logger.error(f"❌ Error handling validation end for {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - log and continue
    
    async def _check_for_reveals(self, epoch_id: int):
        """
        Check if epoch needs reveal/consensus processing.
        
        This is called for previous epochs to handle delayed reveals.
        Only processes each epoch once.
        
        Args:
            epoch_id: Epoch to check for reveals
        """
        try:
            # Skip if already processed
            if epoch_id in self.closed_epochs:
                return
            
            # Import utilities
            from gateway.utils.epoch import is_epoch_closed_async
            
            # Check if epoch is actually closed
            if not await is_epoch_closed_async(epoch_id):
                return
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔓 EPOCH {epoch_id} CLOSED - Checking for reveals...")
            logger.info(f"{'='*80}")
            
            # Check if this epoch has validation evidence
            from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
            from supabase import create_client
            
            supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            
            # Query validation evidence (run in thread to avoid blocking)
            import asyncio
            evidence_check = await asyncio.to_thread(
                lambda: supabase.table("validation_evidence_private")
                    .select("lead_id", count="exact")
                    .eq("epoch_id", epoch_id)
                    .limit(1)
                    .execute()
            )
            
            has_evidence = evidence_check.count > 0 if evidence_check.count is not None else len(evidence_check.data) > 0
            
            if not has_evidence:
                logger.info(f"   ℹ️  No validation evidence for epoch {epoch_id} - skipping")
                # Mark as closed so we don't check again
                self.closed_epochs.add(epoch_id)
                return
            
            logger.info(f"   📊 Found validation evidence - processing reveals and consensus...")
            
            # Import lifecycle functions
            from gateway.tasks.epoch_lifecycle import trigger_reveal_phase, compute_epoch_consensus
            from gateway.utils.epoch import get_epoch_close_time_async
            
            epoch_close = await get_epoch_close_time_async(epoch_id)
            time_since_close = (datetime.utcnow() - epoch_close).total_seconds()
            
            logger.info(f"   Closed at: {epoch_close.isoformat()}")
            logger.info(f"   Time since close: {time_since_close/60:.1f} minutes")
            
            # Trigger reveal phase
            await trigger_reveal_phase(epoch_id)
            
            # Wait for reveals to come in (only if epoch just closed)
            if time_since_close < 300:  # Within 5 minutes
                logger.info(f"   ⏳ Waiting 2 minutes for reveals...")
                await asyncio.sleep(120)
            else:
                logger.info(f"   ℹ️  Epoch closed {time_since_close/60:.1f} minutes ago - skipping reveal wait")
            
            # Compute consensus
            logger.info(f"   📊 Computing consensus for epoch {epoch_id}...")
            await compute_epoch_consensus(epoch_id)
            
            logger.info(f"   ✅ Epoch {epoch_id} fully processed")
            
            # Mark as closed
            self.closed_epochs.add(epoch_id)
            
            # Clean up old tracking sets to prevent memory growth
            if len(self.validation_ended_epochs) > 100:
                recent = sorted(list(self.validation_ended_epochs))[-50:]
                self.validation_ended_epochs = set(recent)
            
            if len(self.closed_epochs) > 100:
                recent = sorted(list(self.closed_epochs))[-50:]
                self.closed_epochs = set(recent)
            
        except Exception as e:
            logger.error(f"❌ Error checking reveals for epoch {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't mark as closed on error - will retry on next block
    
    def get_stats(self) -> dict:
        """
        Get monitor statistics for debugging.
        
        Returns:
            Dict with monitor state
        """
        return {
            "last_epoch": self.last_epoch,
            "validation_ended_count": len(self.validation_ended_epochs),
            "validation_ended_recent": sorted(list(self.validation_ended_epochs))[-10:] if self.validation_ended_epochs else [],
            "closed_count": len(self.closed_epochs),
            "closed_recent": sorted(list(self.closed_epochs))[-10:] if self.closed_epochs else []
        }

