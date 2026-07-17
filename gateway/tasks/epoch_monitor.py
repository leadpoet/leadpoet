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
import os
from datetime import datetime, timezone
import bittensor as bt

from Leadpoet.utils.subnet_epoch import (
    STATEFUL_EPOCH_MODE,
    get_epoch_mode,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.utils.subnet_epoch_archive import (
    read_exact_subnet_epoch_snapshot_from_archive,
    validate_cutover_anchor_from_archive,
)

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
        self.last_official_epoch = None
        self.last_epoch_snapshot = None
        self.netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
        self._validated_cutover_hash = None
        self._stateful_hydrated = False
        self.initialized_epochs = set()  # Epochs that SUCCESSFULLY completed initialization
        self.initializing_epochs = set()  # Epochs currently being initialized (prevents duplicate tasks)
        self.validation_ended_epochs = set()
        self.validation_ending_epochs = set()
        self.pending_validation_ends = {}
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
            self.subtensor = bt.Subtensor(network=self.network)
            print(f"✅ Connected to {self.network} chain")
        except Exception as e:
            print(f"❌ Failed to connect to chain: {e}")
            raise
        
        last_logged_block = None
        
        # Polling loop (like validator)
        while True:
            try:
                if get_epoch_mode() == STATEFUL_EPOCH_MODE:
                    cutover = load_subnet_epoch_cutover()
                    if self._validated_cutover_hash != cutover.mapping_hash:
                        from gateway.utils.epoch import (
                            validate_stateful_cutover_authority_async,
                        )

                        await validate_stateful_cutover_authority_async(cutover)
                        await asyncio.to_thread(
                            validate_cutover_anchor_from_archive,
                            cutover,
                        )
                        self._validated_cutover_hash = cutover.mapping_hash
                    snapshot = await asyncio.to_thread(
                        read_subnet_epoch_snapshot,
                        self.subtensor,
                        netuid=self.netuid,
                        finalized=True,
                    )
                    block_number = snapshot.current_block
                else:
                    snapshot = None
                    # Legacy compatibility path during staged rollout.
                    block_number = self.subtensor.get_current_block()
                
                # Log block occasionally (not every block - too spammy)
                if last_logged_block is None or block_number - last_logged_block >= 10:
                    if snapshot is not None:
                        settlement_epoch = snapshot.settlement_epoch_id(
                            load_subnet_epoch_cutover()
                        )
                        print(
                            f"📦 Block {block_number}: official subnet epoch "
                            f"{snapshot.subnet_epoch_index}, settlement epoch "
                            f"{settlement_epoch}, elapsed {snapshot.epoch_block}, "
                            f"remaining {snapshot.blocks_remaining}/{snapshot.tempo}"
                        )
                    else:
                        current_epoch = block_number // 360
                        block_within_epoch = block_number % 360
                        print(f"📦 Block {block_number}: Epoch {current_epoch}, Block {block_within_epoch}/360")
                    last_logged_block = block_number
                
                # Process block
                if snapshot is not None:
                    if not self._stateful_hydrated:
                        await self._hydrate_stateful_state(snapshot)
                        self._stateful_hydrated = True
                    await self._process_stateful_snapshot(snapshot)
                else:
                    await self._process_block(block_number)
                
                # Wait before next poll (12 seconds = approx block time)
                await asyncio.sleep(12)
                
            except Exception as e:
                print(f"❌ Error in epoch monitor polling loop: {e}")
                print("   Retrying in 30 seconds...")
                await asyncio.sleep(30)

    @staticmethod
    def _payload_datetime(value, field: str) -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"durable lifecycle event is missing {field}")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(
                f"durable lifecycle event has invalid {field}"
            ) from exc
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    async def _read_exact_stateful_snapshot(self, block_number: int):
        return await asyncio.to_thread(
            read_exact_subnet_epoch_snapshot_from_archive,
            netuid=self.netuid,
            block_number=block_number,
        )

    async def _find_stateful_transition_snapshot(
        self,
        target_official_epoch: int,
        high_snapshot,
        *,
        low_snapshot=None,
    ):
        """Locate the first exact finalized-history block for one epoch.

        ``LastEpochBlock`` is not an identity anchor because ``set_tempo`` can
        reset it without incrementing ``SubnetEpochIndex``.  Binary search on
        the monotonic official index finds the actual transition block.
        """

        cutover = load_subnet_epoch_cutover()
        if target_official_epoch < cutover.first_subnet_epoch_index:
            raise RuntimeError("cannot reconcile an epoch before the cutover")
        if high_snapshot.subnet_epoch_index < target_official_epoch:
            raise RuntimeError("transition search high block predates target epoch")

        if target_official_epoch == cutover.first_subnet_epoch_index:
            boundary = await self._read_exact_stateful_snapshot(
                cutover.cutover_block
            )
        else:
            if (
                low_snapshot is None
                or low_snapshot.current_block >= high_snapshot.current_block
                or low_snapshot.subnet_epoch_index >= target_official_epoch
            ):
                low_snapshot = await self._read_exact_stateful_snapshot(
                    cutover.cutover_block - 1
                )
            if low_snapshot.subnet_epoch_index >= target_official_epoch:
                raise RuntimeError("transition search low block is not below target")

            low_block = low_snapshot.current_block
            high_block = high_snapshot.current_block
            boundary = high_snapshot
            while high_block - low_block > 1:
                midpoint = low_block + ((high_block - low_block) // 2)
                observed = await self._read_exact_stateful_snapshot(midpoint)
                if observed.subnet_epoch_index >= target_official_epoch:
                    high_block = midpoint
                    boundary = observed
                else:
                    low_block = midpoint
                    low_snapshot = observed
            if boundary.current_block != high_block:
                boundary = await self._read_exact_stateful_snapshot(high_block)

        if boundary.subnet_epoch_index != target_official_epoch:
            raise RuntimeError(
                "official epoch transition skipped or resolved to the wrong index"
            )
        predecessor = await self._read_exact_stateful_snapshot(
            boundary.current_block - 1
        )
        if predecessor.subnet_epoch_index + 1 != target_official_epoch:
            raise RuntimeError(
                "official epoch transition has no immediate predecessor"
            )
        return boundary

    async def _close_stateful_epoch_at_boundary(
        self,
        epoch_id: int,
        boundary_snapshot,
    ) -> None:
        from gateway.tasks.epoch_lifecycle import get_durable_epoch_event
        from gateway.utils.epoch import get_current_epoch_times

        initialization = await get_durable_epoch_event(
            "EPOCH_INITIALIZATION",
            epoch_id,
        )
        if initialization is None:
            raise RuntimeError(
                f"cannot reconcile epoch {epoch_id}: durable initialization "
                "is missing"
            )
        boundaries = initialization["payload"].get("epoch_boundaries")
        if not isinstance(boundaries, dict):
            raise RuntimeError(
                f"cannot reconcile epoch {epoch_id}: initialization "
                "boundaries are missing"
            )
        epoch_start = self._payload_datetime(
            boundaries.get("start_timestamp"),
            "epoch_boundaries.start_timestamp",
        )
        epoch_end = get_current_epoch_times(boundary_snapshot)[0]
        completed = await self._on_validation_end(
            epoch_id,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            epoch_close=epoch_end,
        )
        if not completed:
            raise RuntimeError(
                f"failed to durably reconcile lifecycle end for epoch {epoch_id}"
            )

    async def _hydrate_stateful_state(self, snapshot) -> None:
        """Restore durable lifecycle state before processing a restart."""

        from gateway.tasks.epoch_lifecycle import get_durable_epoch_event

        cutover = load_subnet_epoch_cutover()
        current_epoch = snapshot.settlement_epoch_id(cutover)

        current_initialization = await get_durable_epoch_event(
            "EPOCH_INITIALIZATION",
            current_epoch,
        )
        current_boundary = None
        if current_initialization is not None:
            # _on_epoch_start will validate the full stateful authority before
            # marking this in memory. This only documents what was observed.
            print(
                f"🔄 Durable initialization found for current epoch "
                f"{current_epoch}; validating it before reuse"
            )
        else:
            # A restart cannot recreate the queue/assignment snapshot after
            # the transition has passed. Only a process observing the exact
            # official boundary may originate a missing initialization.
            current_boundary = await self._find_stateful_transition_snapshot(
                snapshot.subnet_epoch_index,
                snapshot,
            )
            if current_boundary.current_block != snapshot.current_block:
                raise RuntimeError(
                    f"cannot initialize epoch {current_epoch} after its "
                    "official boundary; durable initialization is missing"
                )

        if snapshot.subnet_epoch_index == cutover.first_subnet_epoch_index:
            return

        previous_epoch = current_epoch - 1
        previous_initialization = await get_durable_epoch_event(
            "EPOCH_INITIALIZATION",
            previous_epoch,
        )
        previous_end = await get_durable_epoch_event("EPOCH_END", previous_epoch)
        previous_inputs = await get_durable_epoch_event(
            "EPOCH_INPUTS",
            previous_epoch,
        )

        if previous_inputs is not None and previous_end is None:
            raise RuntimeError(
                f"epoch {previous_epoch} has EPOCH_INPUTS without EPOCH_END"
            )
        if previous_end is not None and previous_inputs is not None:
            self.validation_ended_epochs.add(previous_epoch)
            return
        if previous_initialization is None:
            raise RuntimeError(
                f"cannot recover epoch {previous_epoch}: no durable "
                "EPOCH_INITIALIZATION exists"
            )

        boundary = current_boundary or await self._find_stateful_transition_snapshot(
            snapshot.subnet_epoch_index,
            snapshot,
        )
        await self._close_stateful_epoch_at_boundary(previous_epoch, boundary)

    async def _reconcile_stateful_gap(self, snapshot):
        """Reconcile every safely provable transition missed by polling."""

        from gateway.tasks.epoch_lifecycle import get_durable_epoch_event

        cutover = load_subnet_epoch_cutover()
        assert self.last_official_epoch is not None
        cursor = self.last_epoch_snapshot
        for target_official in range(
            self.last_official_epoch + 1,
            snapshot.subnet_epoch_index + 1,
        ):
            boundary = await self._find_stateful_transition_snapshot(
                target_official,
                snapshot,
                low_snapshot=cursor,
            )
            ended_epoch = cutover.settlement_epoch_id(target_official) - 1
            await self._close_stateful_epoch_at_boundary(ended_epoch, boundary)

            # A historical initialization cannot be reconstructed from the
            # current queue. Require its durable snapshot before advancing.
            target_epoch = cutover.settlement_epoch_id(target_official)
            if target_official < snapshot.subnet_epoch_index:
                initialization = await get_durable_epoch_event(
                    "EPOCH_INITIALIZATION",
                    target_epoch,
                )
                if initialization is None:
                    raise RuntimeError(
                        f"cannot reconcile skipped epoch {target_epoch}: "
                        "durable initialization is missing"
                    )
                self.initialized_epochs.add(target_epoch)
            else:
                initialization = await get_durable_epoch_event(
                    "EPOCH_INITIALIZATION",
                    target_epoch,
                )
                if (
                    initialization is None
                    and boundary.current_block != snapshot.current_block
                ):
                    raise RuntimeError(
                        f"cannot initialize epoch {target_epoch} after its "
                        "official boundary; durable initialization is missing"
                    )
            cursor = boundary
        return cursor

    async def _process_stateful_snapshot(self, snapshot):
        """Drive lifecycle actions from one official scheduler observation."""

        from gateway.utils.epoch import get_current_epoch_times

        cutover = load_subnet_epoch_cutover()
        current_epoch = snapshot.settlement_epoch_id(cutover)
        official_epoch = snapshot.subnet_epoch_index
        blocks_remaining = snapshot.blocks_remaining
        initialization_snapshot = snapshot
        self.startup_block_count += 1

        if self.last_official_epoch is not None:
            if official_epoch < self.last_official_epoch:
                raise RuntimeError(
                    "official subnet epoch regressed; refusing lifecycle actions"
                )
            if official_epoch > self.last_official_epoch + 1:
                print(
                    "⚠️  Official subnet epoch polling gap detected; "
                    "reconciling durable transitions sequentially"
                )
                initialization_snapshot = await self._reconcile_stateful_gap(
                    snapshot
                )

        if self.last_official_epoch is None:
            print(
                "🚀 OFFICIAL SUBNET EPOCH OBSERVED: "
                f"{official_epoch} (settlement {current_epoch})"
            )
        elif official_epoch == self.last_official_epoch + 1:
            previous_epoch = current_epoch - 1
            boundary_snapshot = snapshot
            previous_start = None
            if self.subtensor is not None:
                from gateway.tasks.epoch_lifecycle import get_durable_epoch_event

                boundary_snapshot = await self._find_stateful_transition_snapshot(
                    official_epoch,
                    snapshot,
                    low_snapshot=self.last_epoch_snapshot,
                )
                initialization = await get_durable_epoch_event(
                    "EPOCH_INITIALIZATION",
                    previous_epoch,
                )
                if initialization is None:
                    raise RuntimeError(
                        f"cannot close epoch {previous_epoch}: durable "
                        "initialization is missing"
                    )
                boundaries = initialization["payload"].get("epoch_boundaries")
                if not isinstance(boundaries, dict):
                    raise RuntimeError(
                        f"cannot close epoch {previous_epoch}: durable "
                        "initialization boundaries are missing"
                    )
                previous_start = self._payload_datetime(
                    boundaries.get("start_timestamp"),
                    "epoch_boundaries.start_timestamp",
                )
            transition_time, _end, _close = get_current_epoch_times(
                boundary_snapshot
            )
            if previous_start is None:
                previous_start = (
                    get_current_epoch_times(self.last_epoch_snapshot)[0]
                    if self.last_epoch_snapshot is not None
                    else transition_time
                )
            initialization_snapshot = boundary_snapshot
            print("\n" + "=" * 80)
            print(
                "🚀 OFFICIAL SUBNET EPOCH TRANSITION: "
                f"{self.last_official_epoch} → {official_epoch} "
                f"(settlement {previous_epoch} → {current_epoch})"
            )
            print("=" * 80)
            self.pending_validation_ends.setdefault(
                previous_epoch,
                {"start": previous_start, "end": transition_time},
            )

        self.last_official_epoch = official_epoch
        self.last_epoch = current_epoch
        self.last_epoch_snapshot = snapshot

        for ended_epoch, boundary in list(self.pending_validation_ends.items()):
            if (
                ended_epoch not in self.validation_ended_epochs
                and ended_epoch not in self.validation_ending_epochs
            ):
                self.validation_ending_epochs.add(ended_epoch)
                asyncio.create_task(
                    self._on_validation_end(
                        ended_epoch,
                        epoch_start=boundary["start"],
                        epoch_end=boundary["end"],
                        epoch_close=boundary["end"],
                    )
                )

        if (
            current_epoch not in self.initialized_epochs
            and current_epoch not in self.initializing_epochs
        ):
            self.initializing_epochs.add(current_epoch)
            asyncio.create_task(
                self._on_epoch_start(
                    current_epoch,
                    epoch_snapshot=initialization_snapshot,
                )
            )

        if self.startup_block_count == 10:
            print("✅ Startup grace period complete")

        if 0 < blocks_remaining <= 3 and current_epoch > 0:
            if not hasattr(self, "_cleanup_epochs"):
                self._cleanup_epochs = set()
            if current_epoch not in self._cleanup_epochs:
                print(
                    "🧹 MINER CLEANUP TRIGGER: "
                    f"official epoch {official_epoch}, "
                    f"{blocks_remaining} blocks remaining"
                )
                asyncio.create_task(self._run_miner_cleanup(current_epoch))
                self._cleanup_epochs.add(current_epoch)

        if 16 <= blocks_remaining <= 30 and current_epoch > 0:
            if (
                current_epoch not in self.processing_epochs
                and current_epoch not in self.closed_epochs
            ):
                self.processing_epochs.add(current_epoch)
                _start, _end, epoch_close = get_current_epoch_times(snapshot)
                print(
                    "📊 OFFICIAL EPOCH CONSENSUS WINDOW: "
                    f"subnet epoch {official_epoch}, settlement {current_epoch}, "
                    f"{blocks_remaining} blocks remaining"
                )
                asyncio.create_task(
                    self._check_for_reveals(
                        current_epoch,
                        skip_closed_check=True,
                        epoch_close=epoch_close,
                    )
                )
    
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
            from gateway.utils.epoch import (
                assert_legacy_epoch_namespace_open_async,
            )

            await assert_legacy_epoch_namespace_open_async(
                current_epoch,
                force_refresh=True,
            )
            
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
                
                if (
                    previous_epoch not in self.validation_ended_epochs
                    and previous_epoch not in self.validation_ending_epochs
                ):
                    print(f"\n{'='*80}")
                    print(f"⏰ VALIDATION PHASE ENDED: Epoch {previous_epoch}")
                    print(f"{'='*80}")
                    
                    # Trigger validation end (non-blocking)
                    self.validation_ending_epochs.add(previous_epoch)
                    asyncio.create_task(self._on_validation_end(previous_epoch))
            
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
    
    async def _on_epoch_start(self, epoch_id: int, epoch_snapshot=None):
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
            
            if epoch_snapshot is not None:
                from gateway.utils.epoch import get_current_epoch_times

                epoch_start, epoch_end, epoch_close = get_current_epoch_times(
                    epoch_snapshot
                )
            else:
                # Calculate legacy epoch boundaries using async versions.
                epoch_start = await get_epoch_start_time_async(epoch_id)
                epoch_end = await get_epoch_end_time_async(epoch_id)
                epoch_close = await get_epoch_close_time_async(epoch_id)
            
            print(f"   Start: {epoch_start.isoformat()}")
            print(f"   End (validation): {epoch_end.isoformat()}")
            print(f"   Close: {epoch_close.isoformat()}")
            
            # Log EPOCH_INITIALIZATION to transparency log
            # This can raise an exception if Supabase times out
            await compute_and_log_epoch_initialization(
                epoch_id,
                epoch_start,
                epoch_end,
                epoch_close,
                epoch_snapshot=epoch_snapshot,
            )
            
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
    
    async def _on_validation_end(
        self,
        epoch_id: int,
        *,
        epoch_start=None,
        epoch_end=None,
        epoch_close=None,
    ):
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
            from gateway.tasks.epoch_lifecycle import (
                compute_and_log_epoch_inputs,
                load_stateful_epoch_initialization_authority,
                log_epoch_event,
            )
            from gateway.utils.epoch import get_epoch_end_time_async, get_epoch_close_time_async
            
            if epoch_end is None:
                epoch_end = await get_epoch_end_time_async(epoch_id)
            if epoch_close is None:
                epoch_close = await get_epoch_close_time_async(epoch_id)
            
            print(f"   Ended at: {epoch_end.isoformat()}")
            print(f"   Epoch closed at: {epoch_close.isoformat()}")
            
            # Log EPOCH_END event. Stateful lifecycle rows carry an explicit
            # key discriminator used by the database uniqueness authority;
            # legacy payloads remain byte-for-byte compatible.
            end_payload = {
                "epoch_id": epoch_id,
                "end_time": epoch_end.isoformat(),
                "phase": "epoch_ended",
            }
            epoch_authority = None
            if get_epoch_mode() == STATEFUL_EPOCH_MODE:
                epoch_authority = (
                    await load_stateful_epoch_initialization_authority(epoch_id)
                )
                end_payload["epoch_key_semantics"] = "settlement_ordinal"
                end_payload["epoch_authority"] = epoch_authority
            await log_epoch_event("EPOCH_END", epoch_id, end_payload)
            
            # Compute and log EPOCH_INPUTS (hash of all events during epoch)
            await compute_and_log_epoch_inputs(
                epoch_id,
                epoch_start=epoch_start,
                epoch_end=epoch_end,
                epoch_authority=epoch_authority,
            )
            
            self.validation_ended_epochs.add(epoch_id)
            self.pending_validation_ends.pop(epoch_id, None)
            print(f"✅ Epoch {epoch_id} validation phase complete")
            return True
            
        except Exception as e:
            print(f"❌ Error handling validation end for {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.validation_ending_epochs.discard(epoch_id)
    
    async def _check_for_reveals(
        self,
        epoch_id: int,
        skip_closed_check: bool = False,
        epoch_close=None,
    ):
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
        MAX_RETRIES = 5
        RETRY_DELAY = 30  # seconds between retries
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Import utilities
                from gateway.utils.epoch import (
                    get_current_epoch_context_async,
                    is_epoch_closed_async,
                )
                import asyncio

                if get_epoch_mode() == STATEFUL_EPOCH_MODE:
                    current_snapshot, workflow_epoch = (
                        await get_current_epoch_context_async(finalized=True)
                    )
                    if workflow_epoch != epoch_id or not (
                        16 <= current_snapshot.blocks_remaining <= 30
                    ):
                        print(
                            f"   ⚠️  Consensus for epoch {epoch_id} is "
                            "outside the finalized 30..16 remaining-block "
                            "window; refusing a late computation"
                        )
                        self.processing_epochs.discard(epoch_id)
                        return
                
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
                
                observed_close = epoch_close
                if observed_close is None:
                    observed_close = await get_epoch_close_time_async(epoch_id)
                time_since_close = (
                    datetime.utcnow() - observed_close
                ).total_seconds()
                
                if time_since_close >= 0:
                    print(f"   Epoch closed at: {observed_close.isoformat()} ({time_since_close/60:.1f} min ago)")
                else:
                    print(f"   Epoch closes at: {observed_close.isoformat()} (in {-time_since_close/60:.1f} min)")
                
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
