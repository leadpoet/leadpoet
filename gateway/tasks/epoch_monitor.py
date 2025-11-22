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

# Use print() instead of logger to match rest of gateway
# logger = logging.getLogger(__name__)


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
        self.startup_block_count = 0  # Count blocks since startup
        
        print("🔄 EpochMonitor initialized (event-driven)")
        print("   Consensus will be delayed for first 10 blocks (startup grace period)")
    
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
            
            # Count blocks since startup (for grace period)
            self.startup_block_count += 1
            
            # ════════════════════════════════════════════════════════════════
            # Check 1: New epoch started (block 0, or first time seeing this epoch)
            # ════════════════════════════════════════════════════════════════
            if self.last_epoch is None or current_epoch > self.last_epoch:
                print(f"\n{'='*80}")
                print(f"🚀 EPOCH TRANSITION DETECTED: {self.last_epoch} → {current_epoch}")
                print(f"{'='*80}")
                
                # Trigger epoch start (non-blocking)
                asyncio.create_task(self._on_epoch_start(current_epoch))
                
                self.last_epoch = current_epoch
            
            # ════════════════════════════════════════════════════════════════
            # Check 2: Validation phase ended (block 0 of NEXT epoch = block 360 of PREVIOUS)
            # ════════════════════════════════════════════════════════════════
            if block_within_epoch == 0 and current_epoch > 0:
                previous_epoch = current_epoch - 1
                
                if previous_epoch not in self.validation_ended_epochs:
                    print(f"\n{'='*80}")
                    print(f"⏰ VALIDATION PHASE ENDED: Epoch {previous_epoch}")
                    print(f"{'='*80}")
                    
                    # Trigger validation end (non-blocking)
                    asyncio.create_task(self._on_validation_end(previous_epoch))
                    
                    self.validation_ended_epochs.add(previous_epoch)
            
            # ════════════════════════════════════════════════════════════════
            # Check 3: Epoch closed and needs reveal/consensus (check previous epochs)
            # ════════════════════════════════════════════════════════════════
            # STARTUP GRACE PERIOD: Skip consensus for first 10 blocks
            # This allows metagraph cache to warm up before triggering heavy operations
            if self.startup_block_count <= 10:
                if self.startup_block_count == 10:
                    print("✅ Startup grace period complete - consensus checks now active")
                # Skip consensus checks during startup
                return
            
            # ════════════════════════════════════════════════════════════════
            # Check 4: Deregistered miner cleanup at block 357 (before next epoch)
            # ════════════════════════════════════════════════════════════════
            # Clean up leads from miners who left the subnet
            # Runs at block 357 to clean DB BEFORE next epoch's initialization at block 360
            # This ensures validators never receive leads from deregistered miners
            if block_within_epoch == 357 and current_epoch > 0:
                if not hasattr(self, '_cleanup_epochs'):
                    self._cleanup_epochs = set()
                
                if current_epoch not in self._cleanup_epochs:
                    print(f"\n{'='*80}")
                    print(f"🧹 MINER CLEANUP TRIGGER: Block 357 of epoch {current_epoch}")
                    print(f"   Cleaning DB before epoch {current_epoch + 1} initialization...")
                    print(f"{'='*80}")
                    
                    # Trigger cleanup (non-blocking - runs in background)
                    asyncio.create_task(self._run_miner_cleanup(current_epoch))
                    
                    self._cleanup_epochs.add(current_epoch)
            
            # ════════════════════════════════════════════════════════════════
            # Check 5: Batch consensus at block 350 (captures ALL reveals)
            # ════════════════════════════════════════════════════════════════
            # Run consensus at block 350 of epoch N (for epoch N-1 reveals)
            # This ensures ALL reveals from blocks 0-349 are included
            if block_within_epoch == 350 and current_epoch > 0:
                consensus_epoch = current_epoch - 1  # Calculate consensus for previous epoch
                
                if consensus_epoch not in self.closed_epochs:
                    print(f"\n{'='*80}")
                    print(f"📊 BATCH CONSENSUS TRIGGER: Block 350 of epoch {current_epoch}")
                    print(f"   Computing consensus for epoch {consensus_epoch} reveals...")
                    print(f"{'='*80}")
                    
                    # Trigger consensus (non-blocking)
                    asyncio.create_task(self._check_for_reveals(consensus_epoch))
        
        except Exception as e:
            print(f"❌ Error in EpochMonitor.on_block: {e}")
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
            print(f"🚀 Processing epoch start: {epoch_id}")
            
            # Import lifecycle functions (reuse existing code)
            from gateway.tasks.epoch_lifecycle import compute_and_log_epoch_initialization
            from gateway.utils.epoch import get_epoch_start_time_async, get_epoch_end_time_async, get_epoch_close_time_async
            from gateway.utils.leads_cache import cleanup_old_epochs
            
            # Calculate epoch boundaries using async versions
            epoch_start = await get_epoch_start_time_async(epoch_id)
            epoch_end = await get_epoch_end_time_async(epoch_id)
            epoch_close = await get_epoch_close_time_async(epoch_id)
            
            print(f"   Start: {epoch_start.isoformat()}")
            print(f"   End (validation): {epoch_end.isoformat()}")
            print(f"   Close: {epoch_close.isoformat()}")
            
            # Log EPOCH_INITIALIZATION to transparency log
            await compute_and_log_epoch_initialization(epoch_id, epoch_start, epoch_end, epoch_close)
            
            # Clean up old epoch cache (keep only current + next)
            cleanup_old_epochs(epoch_id)
            
            print(f"✅ Epoch {epoch_id} initialized")
            
        except Exception as e:
            print(f"❌ Error handling epoch start for {epoch_id}: {e}")
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
            print(f"⏰ Processing validation end: {epoch_id}")
            
            # Import lifecycle functions
            from gateway.tasks.epoch_lifecycle import compute_and_log_epoch_inputs, log_epoch_event
            from gateway.utils.epoch import get_epoch_end_time_async, get_epoch_close_time_async
            
            epoch_end = await get_epoch_end_time_async(epoch_id)
            epoch_close = await get_epoch_close_time_async(epoch_id)
            
            print(f"   Ended at: {epoch_end.isoformat()}")
            print(f"   Epoch closed at: {epoch_close.isoformat()}")
            
            # Log EPOCH_END event
            await log_epoch_event("EPOCH_END", epoch_id, {
                "epoch_id": epoch_id,
                "end_time": epoch_end.isoformat(),
                "phase": "epoch_ended"
            })
            
            # Compute and log EPOCH_INPUTS (hash of all events during epoch)
            await compute_and_log_epoch_inputs(epoch_id)
            
            print(f"✅ Epoch {epoch_id} validation phase complete")
            
        except Exception as e:
            print(f"❌ Error handling validation end for {epoch_id}: {e}")
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
            
            print(f"\n{'='*80}")
            print(f"🔓 EPOCH {epoch_id} CLOSED - Checking for reveals...")
            print(f"{'='*80}")
            
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
                print(f"   ℹ️  No validation evidence for epoch {epoch_id} - skipping")
                # Mark as closed so we don't check again
                self.closed_epochs.add(epoch_id)
                return
            
            print(f"   📊 Found validation evidence - processing reveals and consensus...")
            
            # Import lifecycle functions
            from gateway.tasks.epoch_lifecycle import trigger_reveal_phase, compute_epoch_consensus
            from gateway.utils.epoch import get_epoch_close_time_async
            
            epoch_close = await get_epoch_close_time_async(epoch_id)
            time_since_close = (datetime.utcnow() - epoch_close).total_seconds()
            
            print(f"   Closed at: {epoch_close.isoformat()}")
            print(f"   Time since close: {time_since_close/60:.1f} minutes")
            
            # Trigger reveal phase notification
            await trigger_reveal_phase(epoch_id)
            
            # NO WAIT: Consensus triggered at block 350, all reveals should be in already
            # (Reveals accepted from block 0-349 only, enforced by reveal endpoint)
            print(f"   📊 Running batch consensus for epoch {epoch_id}...")
            print(f"   Closed {time_since_close/60:.1f} minutes ago")
            await compute_epoch_consensus(epoch_id)
            
            print(f"   ✅ Epoch {epoch_id} fully processed")
            
            # Mark as closed
            self.closed_epochs.add(epoch_id)
            
            # Clean up old tracking sets to prevent memory growth
            if len(self.validation_ended_epochs) > 100:
                recent = sorted(list(self.validation_ended_epochs))[-50:]
                self.validation_ended_epochs = set(recent)
            
            if len(self.closed_epochs) > 100:
                recent = sorted(list(self.closed_epochs))[-50:]
                self.closed_epochs = set(recent)
            
            if hasattr(self, '_cleanup_epochs') and len(self._cleanup_epochs) > 100:
                recent = sorted(list(self._cleanup_epochs))[-50:]
                self._cleanup_epochs = set(recent)
            
        except Exception as e:
            print(f"❌ Error checking reveals for epoch {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't mark as closed on error - will retry on next block
    
    async def _run_miner_cleanup(self, epoch_id: int):
        """
        Run cleanup of leads from deregistered miners.
        
        This is called at block 10 of each epoch (non-blocking background task).
        Uses cached metagraph (doesn't force refresh).
        
        Args:
            epoch_id: Current epoch number
        """
        try:
            from gateway.tasks.miner_cleanup import cleanup_deregistered_miner_leads
            
            # Run cleanup (this is async and handles all errors internally)
            await cleanup_deregistered_miner_leads(epoch_id)
        
        except Exception as e:
            print(f"❌ Error running miner cleanup for epoch {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - this is a background task
    
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

