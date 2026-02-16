"""
Epoch Monitor (Polling-based)
==============================

Polls Bittensor chain for new blocks and triggers epoch lifecycle events.

This uses the SAME polling approach as the validator (proven stable).

Architecture:
- Polls subtensor.get_current_block() every 12 seconds
- Detects epoch transitions automatically
- Triggers epoch start, validation end, and consensus phases
- Bulletproof: No WebSocket subscriptions = No WebSocket failures

Why polling instead of WebSocket:
- Validator uses polling and runs for months without issues
- Gateway used WebSocket and crashed every 2-4 hours
- Bittensor AsyncSubtensor has WebSocket reconnection bugs
- Polling is simple, reliable, and proven in production
"""

import asyncio
import logging
from datetime import datetime
import bittensor as bt

# Use print() instead of logger to match rest of gateway
# logger = logging.getLogger(__name__)


class EpochMonitor:
    """
    Monitors blocks for epoch transitions and triggers lifecycle events.
    
    Uses POLLING (like validator) instead of WebSocket subscriptions.
    
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
    
    def __init__(self, network: str = "finney"):
        """
        Initialize epoch monitor with empty state.
        
        Args:
            network: Bittensor network to connect to (default: finney)
        """
        self.network = network
        self.subtensor = None  # Will be initialized in start()
        self.last_epoch = None
        self.initialized_epochs = set()  # Epochs that SUCCESSFULLY completed initialization
        self.initializing_epochs = set()  # Epochs currently being initialized (prevents duplicate tasks)
        self.validation_ended_epochs = set()
        self.closed_epochs = set()  # Epochs that completed consensus successfully
        self.processing_epochs = set()  # Epochs currently being processed (prevents duplicate tasks)
        self.startup_block_count = 0  # Count blocks since startup
        
        print("🔄 EpochMonitor initialized (polling-based, like validator)")
        print("   Consensus will be delayed for first 10 blocks (startup grace period)")
    
    async def start(self):
        """
        Start the polling loop (like validator does).
        
        Polls subtensor.get_current_block() every 12 seconds.
        Never crashes - polling is bulletproof.
        """
        print("\n" + "="*80)
        print("🔄 STARTING EPOCH MONITOR (POLLING MODE)")
        print("="*80)
        print(f"   Network: {self.network}")
        print(f"   Poll interval: 12 seconds (approx block time)")
        print(f"   Architecture: Same as validator (proven stable)")
        print("="*80 + "\n")
        
        # Initialize subtensor (sync version - for polling)
        try:
            self.subtensor = bt.subtensor(network=self.network)
            print(f"✅ Connected to {self.network} chain")
        except Exception as e:
            print(f"❌ Failed to connect to chain: {e}")
            raise
        
        last_logged_block = None
        
        # Polling loop (like validator)
        while True:
            try:
                # Get current block (polling - bulletproof, never crashes)
                block_number = self.subtensor.get_current_block()
                
                # Log block occasionally (not every block - too spammy)
                if last_logged_block is None or block_number - last_logged_block >= 10:
                    current_epoch = block_number // 360
                    block_within_epoch = block_number % 360
                    print(f"📦 Block {block_number}: Epoch {current_epoch}, Block {block_within_epoch}/360")
                    last_logged_block = block_number
                
                # Process block
                await self._process_block(block_number)
                
                # Wait before next poll (12 seconds = approx block time)
                await asyncio.sleep(12)
                
            except Exception as e:
                print(f"❌ Error in epoch monitor polling loop: {e}")
                print("   Retrying in 30 seconds...")
                await asyncio.sleep(30)
    
    async def _process_block(self, block_number: int):
        """
        Process a block and trigger epoch lifecycle events.
        
        Args:
            block_number: Current block number from chain
        
        This method:
        1. Checks if epoch has changed (new epoch start)
        2. Checks if validation phase ended (block 0 of next epoch)
        3. Checks if epoch is closed and needs reveals/consensus
        
        All checks are non-blocking and independent.
        """
        try:
            current_epoch = block_number // 360
            block_within_epoch = block_number % 360
            
            # Count blocks since startup (for grace period)
            self.startup_block_count += 1
            
            # ════════════════════════════════════════════════════════════════
            # Check 1: New epoch started (block 0, or first time seeing this epoch)
            # ════════════════════════════════════════════════════════════════
            if self.last_epoch is None or current_epoch > self.last_epoch:
                print(f"\n{'='*80}")
                print(f"🚀 EPOCH TRANSITION DETECTED: {self.last_epoch} → {current_epoch}")
                print(f"{'='*80}")
                
                self.last_epoch = current_epoch
            
            # ════════════════════════════════════════════════════════════════
            # Check 1b: Epoch needs initialization (not yet successfully initialized)
            # This is SEPARATE from transition detection - allows retry on failure
            # ════════════════════════════════════════════════════════════════
            if current_epoch not in self.initialized_epochs:
                if current_epoch not in self.initializing_epochs:
                    # Mark as initializing BEFORE creating task (prevents duplicate tasks)
                    self.initializing_epochs.add(current_epoch)
                    print(f"🔄 Starting initialization for epoch {current_epoch}...")
                    
                    # Trigger epoch start (non-blocking)
                    asyncio.create_task(self._on_epoch_start(current_epoch))
            
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
            # Check 5: IMMEDIATE REVEAL MODE - Consensus for CURRENT epoch at block 330+
            # ════════════════════════════════════════════════════════════════
            # With immediate reveal, validators submit hash+values together during epoch.
            # Consensus can run at block 330+ (same epoch) instead of waiting for next epoch.
            # This eliminates the reveal phase entirely and reduces latency.
            print(f"   🔍 Check 5: block_within_epoch={block_within_epoch}, current_epoch={current_epoch}")
            if 330 <= block_within_epoch <= 358 and current_epoch > 0:
                # IMMEDIATE REVEAL: Compute consensus for CURRENT epoch (data already submitted)
                consensus_epoch = current_epoch
                print(f"   ✅ BLOCK {block_within_epoch} DETECTED! IMMEDIATE REVEAL - checking current epoch {consensus_epoch}")
                
                # Check if epoch is already being processed OR already completed
                # This prevents the race condition where polling loop triggers
                # consensus multiple times (once per block in 328-330 window)
                if consensus_epoch in self.processing_epochs:
                    print(f"   ⚠️  Epoch {consensus_epoch} already being processed (task running)")
                elif consensus_epoch in self.closed_epochs:
                    print(f"   ⚠️  Epoch {consensus_epoch} already completed (in closed_epochs)")
                else:
                    # Mark as processing BEFORE creating task (prevents duplicate tasks)
                    self.processing_epochs.add(consensus_epoch)
                    
                    print(f"\n{'='*80}")
                    print(f"📊 IMMEDIATE REVEAL CONSENSUS: Block {block_within_epoch} of epoch {current_epoch}")
                    print(f"   Computing consensus for CURRENT epoch {consensus_epoch} (data already submitted with hashes)")
                    print(f"{'='*80}")
                    
                    # Trigger consensus (non-blocking)
                    # IMPORTANT: skip_closed_check=True because we're running consensus for the
                    # CURRENT epoch at block 330+, which is not "closed" yet (closes at block 360)
                    asyncio.create_task(self._check_for_reveals(consensus_epoch, skip_closed_check=True))
        
        except Exception as e:
            print(f"❌ Error in EpochMonitor._process_block: {e}")
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
        
        NOTE: On success, adds epoch to initialized_epochs.
              On failure, removes from initializing_epochs so it can retry.
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
            # This can raise an exception if Supabase times out
            await compute_and_log_epoch_initialization(epoch_id, epoch_start, epoch_end, epoch_close)
            
            # Clean up old epoch cache (keep only current + next)
            cleanup_old_epochs(epoch_id)
            
            # ════════════════════════════════════════════════════════════════
            # REFRESH METAGRAPH: Force refresh using existing AsyncSubtensor
            # - Resets cache timestamp to bypass "fast path" (which skips epoch check)
            # - Uses existing get_metagraph_async() with 60s timeout, 3 retries, fallback
            # - Keeps AsyncSubtensor WebSocket alive (we're using it!)
            # - Non-blocking: runs in background task, doesn't block polling loop
            # ════════════════════════════════════════════════════════════════
            import gateway.utils.registry as registry_module
            
            # Reset cache timestamp to force the "slow path" (which checks epoch)
            with registry_module._cache_lock:
                registry_module._cache_epoch_timestamp = None
            
            try:
                # This uses the injected AsyncSubtensor - keeps WebSocket alive!
                metagraph = await registry_module.get_metagraph_async()
                print(f"🔄 Metagraph refreshed for epoch {epoch_id}: {len(metagraph.hotkeys)} neurons")
            except Exception as e:
                # Don't crash - registry.py already falls back to cached metagraph
                print(f"⚠️  Metagraph refresh failed: {e} (using cached metagraph)")
            
            # SUCCESS: Mark epoch as initialized
            self.initialized_epochs.add(epoch_id)
            self.initializing_epochs.discard(epoch_id)
            print(f"✅ Epoch {epoch_id} initialized successfully")
            
            # Clean up old tracking sets to prevent memory growth
            if len(self.initialized_epochs) > 100:
                recent = sorted(list(self.initialized_epochs))[-50:]
                self.initialized_epochs = set(recent)
            
        except Exception as e:
            print(f"❌ Error handling epoch start for {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # FAILURE: Remove from initializing so it can retry on next poll cycle
            self.initializing_epochs.discard(epoch_id)
            print(f"   🔄 Epoch {epoch_id} initialization FAILED - will retry on next poll cycle (12s)")
    
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
    
    async def _check_for_reveals(self, epoch_id: int, skip_closed_check: bool = False):
        """
        Check if epoch needs reveal/consensus processing.
        
        This is called for previous epochs to handle delayed reveals.
        Only processes each epoch once.
        
        NOTE: Deduplication is handled in _process_block() by adding
        epoch to processing_epochs BEFORE creating this task. This prevents
        the race condition where multiple tasks were created for the same epoch.
        On success, epoch moves from processing_epochs to closed_epochs.
        On failure, retries up to MAX_RETRIES times before giving up.
        
        Args:
            epoch_id: Epoch to check for reveals
            skip_closed_check: If True, skip the is_epoch_closed check (for IMMEDIATE REVEAL MODE
                               where we run consensus for current epoch at block 330+)
        """
        MAX_RETRIES = 3
        RETRY_DELAY = 30  # seconds between retries
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Import utilities
                from gateway.utils.epoch import is_epoch_closed_async
                import asyncio
                
                # Check if epoch is actually closed (skip for IMMEDIATE REVEAL MODE on current epoch)
                if not skip_closed_check and not await is_epoch_closed_async(epoch_id):
                    print(f"   ⚠️  Epoch {epoch_id} not closed yet - skipping consensus")
                    self.processing_epochs.discard(epoch_id)
                    return
                
                print(f"\n{'='*80}")
                print(f"🔓 EPOCH {epoch_id} - Computing consensus (IMMEDIATE REVEAL MODE) (attempt {attempt}/{MAX_RETRIES})")
                print(f"{'='*80}")
                
                # Check if this epoch has validation evidence with decisions populated
                # IMMEDIATE REVEAL: decision is submitted WITH hashes, so check for non-null decisions
                from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
                from supabase import create_client
                
                supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                
                # Query validation evidence with decisions (run in thread to avoid blocking)
                evidence_check = await asyncio.to_thread(
                    lambda: supabase.table("validation_evidence_private")
                        .select("lead_id", count="exact")
                        .eq("epoch_id", epoch_id)
                        .not_.is_("decision", "null")  # IMMEDIATE REVEAL: check for decision, not revealed_ts
                        .limit(1)
                        .execute()
                )
                
                has_evidence = evidence_check.count > 0 if evidence_check.count is not None else len(evidence_check.data) > 0
                
                if not has_evidence:
                    if attempt < MAX_RETRIES:
                        print(f"   ⏳ No evidence yet for epoch {epoch_id} (attempt {attempt}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"   ℹ️  No validation evidence after {MAX_RETRIES} attempts for epoch {epoch_id} - marking as closed")
                        self.processing_epochs.discard(epoch_id)
                        self.closed_epochs.add(epoch_id)
                        return
                
                print(f"   📊 Found {evidence_check.count} validation records with decisions - processing consensus...")
                
                # Import lifecycle functions
                # NOTE: IMMEDIATE REVEAL MODE (Jan 2026) - no separate reveal phase
                from gateway.tasks.epoch_lifecycle import compute_epoch_consensus
                from gateway.utils.epoch import get_epoch_close_time_async
                
                epoch_close = await get_epoch_close_time_async(epoch_id)
                time_since_close = (datetime.utcnow() - epoch_close).total_seconds()
                
                if time_since_close >= 0:
                    print(f"   Epoch closed at: {epoch_close.isoformat()} ({time_since_close/60:.1f} min ago)")
                else:
                    print(f"   Epoch closes at: {epoch_close.isoformat()} (in {-time_since_close/60:.1f} min)")
                
                # IMMEDIATE REVEAL MODE: Data submitted with hashes, compute consensus now
                print(f"   📊 Running consensus for epoch {epoch_id}...")
                await compute_epoch_consensus(epoch_id)
                
                print(f"   ✅ Epoch {epoch_id} fully processed")
                
                # SUCCESS: Move from processing to closed
                self.processing_epochs.discard(epoch_id)
                self.closed_epochs.add(epoch_id)
                
                # Clean up old tracking sets to prevent memory growth
                if len(self.validation_ended_epochs) > 100:
                    recent = sorted(list(self.validation_ended_epochs))[-50:]
                    self.validation_ended_epochs = set(recent)
                
                if len(self.closed_epochs) > 100:
                    recent = sorted(list(self.closed_epochs))[-50:]
                    self.closed_epochs = set(recent)
                
                if len(self.processing_epochs) > 100:
                    recent = sorted(list(self.processing_epochs))[-50:]
                    self.processing_epochs = set(recent)
                
                if hasattr(self, '_cleanup_epochs') and len(self._cleanup_epochs) > 100:
                    recent = sorted(list(self._cleanup_epochs))[-50:]
                    self._cleanup_epochs = set(recent)
                
                return  # Success - exit retry loop
                
            except Exception as e:
                print(f"❌ Error checking reveals for epoch {epoch_id} (attempt {attempt}/{MAX_RETRIES}): {e}")
                import traceback
                traceback.print_exc()
                
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Retrying epoch {epoch_id} in {RETRY_DELAY}s...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"   ❌ FAILED: Epoch {epoch_id} consensus failed after {MAX_RETRIES} attempts!")
                    print(f"   ⚠️  This epoch's leads will remain in pending_validation status")
                    # Remove from processing (won't be in closed_epochs, so leads stay pending)
                    self.processing_epochs.discard(epoch_id)
                    # Add to a failed set so we can track/retry later if needed
                    if not hasattr(self, 'failed_epochs'):
                        self.failed_epochs = set()
                    self.failed_epochs.add(epoch_id)
    
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
            "initialized_count": len(self.initialized_epochs),
            "initialized_recent": sorted(list(self.initialized_epochs))[-10:] if self.initialized_epochs else [],
            "initializing_current": sorted(list(self.initializing_epochs)) if self.initializing_epochs else [],
            "validation_ended_count": len(self.validation_ended_epochs),
            "validation_ended_recent": sorted(list(self.validation_ended_epochs))[-10:] if self.validation_ended_epochs else [],
            "processing_count": len(self.processing_epochs),
            "processing_current": sorted(list(self.processing_epochs)) if self.processing_epochs else [],
            "closed_count": len(self.closed_epochs),
            "closed_recent": sorted(list(self.closed_epochs))[-10:] if self.closed_epochs else []
        }

