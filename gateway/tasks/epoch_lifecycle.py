"""
Epoch Lifecycle Management Task

Background task that manages epoch lifecycle events:
- EPOCH_INITIALIZATION: Combined event with epoch boundaries, queue root, and lead assignment
- EPOCH_END: Logged when validation phase ends (block 360)
- EPOCH_INPUTS: Hash of all events during epoch
- Reveal Phase: Triggered after epoch closes (block 360+)
- Consensus: Computed after reveals collected

Runs every 30 seconds to check for epoch transitions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import uuid4
import hashlib
import json

from gateway.utils.epoch import (
    get_current_epoch_id_async,
    get_epoch_start_time_async,
    get_epoch_end_time_async,
    get_epoch_close_time_async,
    is_epoch_active,
    is_epoch_closed,
    get_block_within_epoch
)
from gateway.utils.merkle import compute_merkle_root
from gateway.utils.linkedin import normalize_linkedin_url, compute_linkedin_combo_hash
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BUILD_ID, MAX_LEADS_PER_EPOCH
from supabase import create_client

# Import leads cache for prefetching
from gateway.utils.leads_cache import (
    prefetch_leads_for_next_epoch,
    cleanup_old_epochs,
    print_cache_status,
    is_prefetch_in_progress
)

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def fetch_full_leads_for_epoch(epoch_id: int) -> list:
    """
    Fetch full lead data for an epoch (for caching).
    
    This function queries pending leads, assigns them to validators,
    and fetches complete lead data with miner_hotkey.
    
    Args:
        epoch_id: The epoch to fetch leads for
    
    Returns:
        List of lead dictionaries with lead_id, lead_blob, lead_blob_hash, miner_hotkey
    """
    from gateway.utils.assignment import get_validator_set
    
    try:
        # Step 1: Query pending leads from queue (FIFO order)
        # IMPORTANT: Add .range(0, 10000) to override Supabase's default 1000 row limit
        print(f"   🔍 Querying pending leads for epoch {epoch_id}...")
        result = await asyncio.to_thread(
            lambda: supabase.table("leads_private")
                .select("lead_id")
                .eq("status", "pending_validation")
                .order("created_ts")
                .range(0, 10000)
                .execute()
        )
        
        lead_ids = [row["lead_id"] for row in result.data]
        print(f"   📊 Found {len(lead_ids)} pending leads in queue")
        
        # Step 2: Get validator set
        validator_set = await get_validator_set(epoch_id)
        print(f"   👥 Validator set: {len(validator_set)} validators")
        
        # Step 3: Determine lead assignment (simple FIFO - first N leads)
        # All validators get the same leads, so just take first MAX_LEADS_PER_EPOCH
        assigned_lead_ids = lead_ids[:MAX_LEADS_PER_EPOCH]
        print(f"   📋 Assigned {len(assigned_lead_ids)} leads for epoch {epoch_id}")
        
        # Step 4: Fetch full lead data
        print(f"   💾 Fetching full lead data from database...")
        leads_result = await asyncio.to_thread(
            lambda: supabase.table("leads_private")
                .select("lead_id, lead_blob, lead_blob_hash")
                .in_("lead_id", assigned_lead_ids)
                .execute()
        )
        
        # Step 5: Build full leads with miner_hotkey
        full_leads = []
        for lead_row in leads_result.data:
            lead_blob = lead_row.get("lead_blob", {})
            miner_hotkey = lead_blob.get("wallet_ss58", "unknown")
            
            full_leads.append({
                "lead_id": lead_row["lead_id"],
                "lead_blob": lead_blob,
                "lead_blob_hash": lead_row["lead_blob_hash"],
                "miner_hotkey": miner_hotkey
            })
        
        print(f"   ✅ Built {len(full_leads)} complete lead objects")
        return full_leads
        
    except Exception as e:
        print(f"   ❌ Error fetching leads for epoch {epoch_id}: {e}")
        raise


async def epoch_lifecycle_task():
    """
    Background task to manage epoch lifecycle events.
    
    Runs every 30 seconds and checks:
    - Is it time to start new epoch? → Log EPOCH_INITIALIZATION
    - Is it time to prefetch next epoch? → Cache leads at block 351
    - Is it time to end validation? → Log EPOCH_END + EPOCH_INPUTS
    - Is it time to close epoch? → Trigger reveal phase + consensus
    
    This task ensures all epoch events are logged deterministically and
    consensus is computed automatically.
    """
    
    print("\n" + "="*80)
    print("🚀 EPOCH LIFECYCLE TASK STARTING")
    print("="*80)
    
    last_epoch_id = None
    validation_ended_epochs = set()  # Track which epochs we've logged EPOCH_END for
    closed_epochs = set()  # Track which epochs we've processed consensus for
    prefetch_triggered_epochs = set()  # Track which epochs we've triggered prefetch for
    
    print("✅ Epoch lifecycle task initialized")
    print("   Will check for epoch transitions every 30 seconds")
    print("   Will prefetch next epoch leads at block 351")
    print("="*80 + "\n")
    
    while True:
        try:
            current_epoch = await get_current_epoch_id_async()
            now = datetime.utcnow()
            
            epoch_start = await get_epoch_start_time_async(current_epoch)
            epoch_end = await get_epoch_end_time_async(current_epoch)
            epoch_close = await get_epoch_close_time_async(current_epoch)
            
            # Debug: Show lifecycle task is running (every 30 seconds)
            time_to_end = (epoch_end - now).total_seconds()
            time_to_close = (epoch_close - now).total_seconds()
            print(f"🔄 Epoch {current_epoch} lifecycle check:")
            print(f"   Time to validation end: {time_to_end/60:.1f} min")
            print(f"   Time to epoch close: {time_to_close/60:.1f} min")
            print(f"   Last epoch: {last_epoch_id}")
            print(f"   Closed epochs: {sorted(list(closed_epochs))[-5:] if closed_epochs else []}")  # Show last 5
            
            # ========================================================================
            # Check if new epoch started
            # ========================================================================
            if last_epoch_id is None or current_epoch > last_epoch_id:
                print(f"\n{'='*80}")
                print(f"🚀 NEW EPOCH STARTED: {current_epoch}")
                print(f"{'='*80}")
                print(f"   Start: {epoch_start.isoformat()}")
                print(f"   End (validation): {epoch_end.isoformat()}")
                print(f"   Close: {epoch_close.isoformat()}")
                
                # Compute and log single atomic EPOCH_INITIALIZATION event
                # If this fails, we DON'T update last_epoch_id so next 30s cycle retries automatically
                try:
                    await compute_and_log_epoch_initialization(current_epoch, epoch_start, epoch_end, epoch_close)
                except Exception as init_error:
                    print(f"   ❌ EPOCH_INITIALIZATION failed: {init_error}")
                    print(f"   ⚠️  Will retry on next 30s cycle (NOT updating last_epoch_id)")
                    # DON'T update last_epoch_id - this ensures we retry next cycle
                    # DON'T block other operations - just continue the main loop
                    await asyncio.sleep(30)
                    continue
                
                # Clean up old epoch cache (keep only current and next)
                cleanup_old_epochs(current_epoch)
                
                last_epoch_id = current_epoch
                print(f"   ✅ Epoch {current_epoch} initialized\n")
            
            # ========================================================================
            # Check if it's time to prefetch next epoch leads (blocks 351-360)
            # ========================================================================
            try:
                block_within_epoch = get_block_within_epoch()
                
                # Trigger prefetch once when we reach block 351
                if (351 <= block_within_epoch <= 360 and 
                    current_epoch not in prefetch_triggered_epochs and 
                    not is_prefetch_in_progress()):
                    
                    next_epoch = current_epoch + 1
                    
                    print(f"\n{'='*80}")
                    print(f"🔍 PREFETCH TRIGGER: Block {block_within_epoch}/360")
                    print(f"{'='*80}")
                    print(f"   Current epoch: {current_epoch}")
                    print(f"   Prefetching for: epoch {next_epoch}")
                    print(f"   Time remaining: {360 - block_within_epoch} blocks (~{(360 - block_within_epoch) * 12}s)")
                    print(f"{'='*80}\n")
                    
                    # Mark as triggered immediately to prevent duplicate prefetch
                    prefetch_triggered_epochs.add(current_epoch)
                    
                    # Start prefetch in background (don't block lifecycle checks)
                    # This will retry with 30s timeout until successful
                    asyncio.create_task(
                        prefetch_leads_for_next_epoch(
                            next_epoch=next_epoch,
                            fetch_function=lambda: fetch_full_leads_for_epoch(next_epoch),
                            timeout=30,  # 30 second timeout per attempt
                            retry_delay=5  # 5 second delay between retries
                        )
                    )
                    
                    print(f"   ✅ Prefetch task started in background for epoch {next_epoch}\n")
                
            except Exception as e:
                print(f"   ⚠️  Could not check block number for prefetch: {e}")
                # Not critical - prefetch is optimization, workflow continues
            
            # ========================================================================
            # Check if validation phase just ended (t=67)
            # ========================================================================
            time_since_end = (now - epoch_end).total_seconds()
            if 0 <= time_since_end < 60 and current_epoch not in validation_ended_epochs:
                print(f"\n{'='*80}")
                print(f"⏰ EPOCH {current_epoch} VALIDATION PHASE ENDED")
                print(f"{'='*80}")
                print(f"   Ended at: {epoch_end.isoformat()}")
                print(f"   Epoch closed at: {epoch_close.isoformat()}")
                
                # Log EPOCH_END
                await log_epoch_event("EPOCH_END", current_epoch, {
                    "epoch_id": current_epoch,
                    "end_time": epoch_end.isoformat(),
                    "phase": "epoch_ended"
                })
                
                # Compute and log EPOCH_INPUTS hash
                await compute_and_log_epoch_inputs(current_epoch)
                
                validation_ended_epochs.add(current_epoch)
                print(f"   ✅ Epoch {current_epoch} validation phase complete\n")
            
            # ========================================================================
            # Check if ANY previous epochs need consensus (check up to 10 epochs back)
            # ========================================================================
            print(f"   🔍 Checking for closed epochs needing consensus...")
            
            # Check current epoch and up to 10 previous epochs
            epochs_to_check = range(max(1, current_epoch - 10), current_epoch + 1)
            
            for check_epoch in epochs_to_check:
                if check_epoch in closed_epochs:
                    continue  # Already processed
                
                # Calculate close time for THIS epoch
                check_epoch_close = await get_epoch_close_time_async(check_epoch)
                time_since_close = (now - check_epoch_close).total_seconds()
                
                if time_since_close >= 0:  # This epoch has closed
                    print(f"   ⚠️  EPOCH {check_epoch} IS CLOSED - Checking for evidence...")
                    # Check if this epoch has validation evidence (run in thread to avoid blocking)
                    try:
                        print(f"   🔍 Querying validation_evidence_private for epoch {check_epoch}...")
                        evidence_check = await asyncio.to_thread(
                            lambda: supabase.table("validation_evidence_private")
                                .select("lead_id", count="exact")
                                .eq("epoch_id", check_epoch)
                                .limit(1)
                                .execute()
                        )
                        has_evidence = evidence_check.count > 0 if evidence_check.count is not None else len(evidence_check.data) > 0
                        print(f"   📊 Evidence check: count={evidence_check.count}, has_evidence={has_evidence}")
                    except Exception as e:
                        print(f"   ❌ Could not check validation evidence for epoch {check_epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                        has_evidence = False
                    
                    if has_evidence:
                        print(f"\n{'='*80}")
                        print(f"🔓 EPOCH {check_epoch} CLOSED - STARTING REVEAL & CONSENSUS")
                        print(f"{'='*80}")
                        print(f"   Closed at: {check_epoch_close.isoformat()}")
                        print(f"   Time since close: {time_since_close/60:.1f} minutes")
                        
                        # Trigger reveal phase (validators should reveal their commits)
                        await trigger_reveal_phase(check_epoch)
                        
                        # Wait a bit for reveals to come in (only if epoch just closed)
                        if time_since_close < 300:  # Within 5 minutes
                            print(f"   ⏳ Waiting 2 minutes for reveals...")
                            await asyncio.sleep(120)
                        else:
                            print(f"   ℹ️  Epoch closed {time_since_close/60:.1f} minutes ago - skipping reveal wait")
                        
                        # Compute consensus for all leads in this epoch
                        print(f"   📊 About to call compute_epoch_consensus({check_epoch})...")
                        try:
                            await compute_epoch_consensus(check_epoch)
                            print(f"   ✅ Consensus computation complete for epoch {check_epoch}")
                        except Exception as e:
                            print(f"   ❌ ERROR: Consensus failed for epoch {check_epoch}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Don't mark as closed so we retry next iteration
                            continue
                        
                        closed_epochs.add(check_epoch)
                        print(f"   ✅ Epoch {check_epoch} fully processed\n")
                    elif time_since_close >= 300:  # 5 minutes after close
                        # No evidence after 5 minutes - mark as processed to avoid checking again
                        print(f"   ℹ️  Epoch {check_epoch} closed {time_since_close/60:.1f} minutes ago with no validation evidence - marking as processed")
                        closed_epochs.add(check_epoch)
                    else:
                        print(f"   ⏳ No evidence yet for epoch {check_epoch}, but only {time_since_close:.0f}s since close - will check again")
            
            # Clean up old tracking sets to prevent memory growth
            if len(validation_ended_epochs) > 100:
                # Keep only recent 50 epochs
                recent = sorted(list(validation_ended_epochs))[-50:]
                validation_ended_epochs = set(recent)
            
            if len(closed_epochs) > 100:
                recent = sorted(list(closed_epochs))[-50:]
                closed_epochs = set(recent)
            
            # Sleep 30 seconds before next check
            # Print periodic heartbeat so we know the task is alive
            print(f"   ⏰ Epoch lifecycle: Checked epoch {current_epoch} (sleeping 30s)")
            await asyncio.sleep(30)
        
        except Exception as e:
            print(f"❌ Epoch lifecycle error: {e}")
            import traceback
            traceback.print_exc()
            print(f"   🔄 Recovering from error, will retry in 30 seconds...")
            await asyncio.sleep(30)


async def log_epoch_event(event_type: str, epoch_id: int, payload: dict):
    """
    Log epoch management event to transparency log (Arweave-first).
    
    This function writes events to Arweave first (immutable source of truth),
    then mirrors to Supabase (query cache). This ensures epoch events cannot
    be tampered with by the gateway operator.
    
    Args:
        event_type: EPOCH_INITIALIZATION, EPOCH_END, EPOCH_INPUTS, etc.
        epoch_id: Epoch number
        payload: Event data
    
    Returns:
        str: Arweave transaction ID if successful, None if failed
    """
    try:
        from gateway.utils.logger import log_event
        
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)  # Handle datetime objects
        payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        
        log_entry = {
            "event_type": event_type,
            "actor_hotkey": "system",  # System-generated event
            "nonce": str(uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "payload_hash": payload_hash,
            "build_id": BUILD_ID,
            "signature": "system",  # No signature for system events
            "payload": payload
        }
        
        # Write to TEE buffer (hardware-protected)
        result = await log_event(log_entry)
        
        tee_sequence = result.get("sequence")
        print(f"   📝 Logged {event_type} for epoch {epoch_id} to TEE buffer (seq={tee_sequence})")
        return tee_sequence
    
    except Exception as e:
        print(f"   ❌ Failed to log {event_type}: {e}")
        return None


async def compute_and_log_epoch_initialization(epoch_id: int, epoch_start: datetime, epoch_end: datetime, epoch_close: datetime):
    """
    Compute and log single atomic EPOCH_INITIALIZATION event.
    
    This combines three previously separate events (EPOCH_START, QUEUE_ROOT, EPOCH_ASSIGNMENT)
    into one atomic event for efficiency and consistency. The event contains:
    - Epoch boundaries (start, end, close times)
    - Queue state (Merkle root of pending leads)
    - Lead assignment (50 leads assigned to all validators)
    
    Args:
        epoch_id: Current epoch ID
        epoch_start: Epoch start time
        epoch_end: Epoch validation end time
        epoch_close: Epoch close time
    """
    try:
        from gateway.utils.assignment import deterministic_lead_assignment, get_validator_set
        
        # ========================================================================
        # 1. Query pending leads from queue (FIFO order) - RUN IN THREAD
        # CRITICAL: Use pagination with small batches to avoid Supabase statement timeout
        # The old .range(0, 10000) query was timing out under load
        # ========================================================================
        lead_ids = []
        batch_size = 500  # Small batches to avoid timeout
        offset = 0
        max_leads = 10000  # Safety limit
        
        print(f"   📊 Fetching pending leads (batch_size={batch_size})...")
        
        while offset < max_leads:
            end = offset + batch_size - 1
            
            result = await asyncio.to_thread(
                lambda s=offset, e=end: supabase.table("leads_private")
                    .select("lead_id")
                    .eq("status", "pending_validation")
                    .order("created_ts")
                    .range(s, e)
                    .execute()
            )
            
            if not result.data:
                break
            
            batch_ids = [row["lead_id"] for row in result.data]
            lead_ids.extend(batch_ids)
            
            # If we got less than batch_size, we've reached the end
            if len(result.data) < batch_size:
                break
            
            offset += batch_size
        
        print(f"   📊 Fetched {len(lead_ids)} pending leads total")
        
        if not lead_ids:
            queue_merkle_root = "0" * 64  # Empty queue
            pending_lead_count = 0
        else:
            queue_merkle_root = compute_merkle_root(lead_ids)
            pending_lead_count = len(lead_ids)
        
        print(f"   📊 Queue State: {queue_merkle_root[:16]}... ({pending_lead_count} pending leads)")
        
        # ========================================================================
        # 2. Get validator set for this epoch (ASYNC CALL - no thread needed)
        # ========================================================================
        validator_set = await get_validator_set(epoch_id)  # Returns List[str] of hotkeys
        validator_hotkeys = validator_set  # Already a list of hotkey strings
        validator_count = len(validator_hotkeys)
        
        print(f"   👥 Validator Set: {validator_count} active validators")
        
        # ========================================================================
        # 3. Compute deterministic lead assignment (first N leads, FIFO)
        # ========================================================================
        assigned_lead_ids = deterministic_lead_assignment(
            queue_merkle_root, 
            validator_set, 
            epoch_id, 
            max_leads_per_epoch=MAX_LEADS_PER_EPOCH
        )
        
        print(f"   📋 Assignment: {len(assigned_lead_ids)} leads assigned to all validators (max={MAX_LEADS_PER_EPOCH})")
        
        # ========================================================================
        # 4. Create single atomic EPOCH_INITIALIZATION event
        # ========================================================================
        payload = {
            "epoch_id": epoch_id,
            "epoch_boundaries": {
                "start_block": epoch_id * 360,  # Approximate - actual block from blockchain
                "end_block": (epoch_id * 360) + 360,
                "start_timestamp": epoch_start.isoformat(),
                "estimated_end_timestamp": epoch_end.isoformat()
            },
            "queue_state": {
                "queue_merkle_root": queue_merkle_root,
                "pending_lead_count": pending_lead_count
            },
            "assignment": {
                "assigned_lead_ids": assigned_lead_ids,
                "assigned_to_validators": validator_hotkeys,
                "validator_count": validator_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await log_epoch_event("EPOCH_INITIALIZATION", epoch_id, payload)
        
        print(f"   ✅ EPOCH_INITIALIZATION logged: {len(assigned_lead_ids)} leads, {validator_count} validators")
    
    except Exception as e:
        print(f"   ❌ Failed to compute EPOCH_INITIALIZATION: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise so caller can retry


async def compute_and_log_epoch_inputs(epoch_id: int):
    """
    Compute hash of all events in epoch and log EPOCH_INPUTS event.
    
    This creates a deterministic hash of all transparency log events
    during the epoch, ensuring all validators see the same input data.
    
    Args:
        epoch_id: Epoch ID
    """
    try:
        epoch_start = await get_epoch_start_time_async(epoch_id)
        epoch_end = await get_epoch_end_time_async(epoch_id)
        
        # Query all events in epoch (during validation phase) - RUN IN THREAD
        result = await asyncio.to_thread(
            lambda: supabase.table("transparency_log")
                .select("id, event_type, payload_hash")
                .gte("ts", epoch_start.isoformat())
                .lte("ts", epoch_end.isoformat())
                .order("id")
                .execute()
        )
        
        events = result.data
        
        # Compute hash of all event hashes
        if events:
            event_hashes = [e["payload_hash"] for e in events]
            combined = "".join(event_hashes)
            inputs_hash = hashlib.sha256(combined.encode()).hexdigest()
        else:
            inputs_hash = "0" * 64
        
        await log_epoch_event("EPOCH_INPUTS", epoch_id, {
            "epoch_id": epoch_id,
            "inputs_hash": inputs_hash,
            "event_count": len(events),
            "start_time": epoch_start.isoformat(),
            "end_time": epoch_end.isoformat()
        })
        
        print(f"   🔢 EPOCH_INPUTS: {inputs_hash[:16]}... ({len(events)} events)")
    
    except Exception as e:
        print(f"   ❌ Failed to compute EPOCH_INPUTS: {e}")


async def trigger_reveal_phase(epoch_id: int):
    """
    Trigger reveal phase for epoch.
    
    After epoch closes, validators must reveal their committed decisions
    and rep_scores (but NOT evidence, which stays private forever).
    
    This function logs a notification event. Validators listen for epoch
    close and automatically call POST /reveal with their salt and values.
    
    Args:
        epoch_id: Epoch ID
    """
    try:
        print(f"   🔓 Validators can now reveal decisions for epoch {epoch_id}")
        
        # Query how many validators submitted commits - RUN IN THREAD
        result = await asyncio.to_thread(
            lambda: supabase.table("validation_evidence_private")
                .select("evidence_id", count="exact")
                .eq("epoch_id", epoch_id)
                .execute()
        )
        
        commit_count = result.count if result.count is not None else 0
        
        print(f"   📊 {commit_count} validation commits to reveal")
        
        # Validators will call POST /reveal independently
        # This is just a monitoring/logging step
    
    except Exception as e:
        print(f"   ❌ Failed to trigger reveal phase: {e}")


async def compute_epoch_consensus(epoch_id: int):
    """
    Compute weighted consensus for all leads in epoch.
    
    Uses V-scores (v_trust × stake) to weight validator decisions and rep_scores.
    Updates both validation_evidence_private and leads_private with final consensus outcomes.
    
    This implements all 3 priorities:
    1. Consensus aggregation - populates all new columns in leads_private
    2. Validator trust/stake population - updates validation_evidence_private
    3. Weighted consensus - uses v_trust × stake weights from Bittensor metagraph
    
    Args:
        epoch_id: Epoch ID
    """
    try:
        # Import here to avoid circular dependency
        from gateway.utils.consensus import compute_weighted_consensus
        from gateway.utils.registry import get_metagraph_async
        
        print(f"   📊 Starting consensus for epoch {epoch_id}...")
        
        # ========================================================================
        # PRIORITY 2: Populate v_trust and stake from Bittensor metagraph
        # ========================================================================
        print(f"   🔍 Step 1: Populating validator trust and stake from metagraph...")
        
        try:
            # Get metagraph to fetch v_trust and stake for all validators
            metagraph = await get_metagraph_async()
            
            # Query all evidence for this epoch that has been revealed - USE PAGINATION
            # CRITICAL: .range() does NOT override Supabase's 1000-row limit per request!
            # We must paginate through results in batches of 1000 to get all rows.
            all_evidence = []
            offset = 0
            batch_size = 1000
            
            while True:
                evidence_batch = await asyncio.to_thread(
                    lambda o=offset: supabase.table("validation_evidence_private")
                    .select("evidence_id, validator_hotkey")
                    .eq("epoch_id", epoch_id)
                    .not_.is_("decision", "null")
                        .range(o, o + batch_size - 1)
                    .execute()
            )
                
                if not evidence_batch.data:
                    break
                
                all_evidence.extend(evidence_batch.data)
                
                if len(evidence_batch.data) < batch_size:
                    break  # Last batch
                
                offset += batch_size
            
            # Use the paginated results
            class PaginatedResult:
                def __init__(self, data):
                    self.data = data
            
            evidence_result = PaginatedResult(all_evidence)
            
            print(f"   📊 Found {len(evidence_result.data)} revealed validation records")
            
            # Update v_trust and stake for each validator's evidence
            for ev in evidence_result.data:
                validator_hotkey = ev['validator_hotkey']
                evidence_id = ev['evidence_id']
                
                try:
                    # Get validator's UID in metagraph
                    if validator_hotkey in metagraph.hotkeys:
                        uid = metagraph.hotkeys.index(validator_hotkey)
                        
                        # Get stake (TAO amount) 
                        stake = float(metagraph.S[uid])
                        
                        # Get v_trust (validator trust score) - same as registry.py line 283
                        v_trust = float(metagraph.validator_trust[uid]) if hasattr(metagraph, 'validator_trust') else 0.0
                        
                        # Update evidence record with v_trust and stake - RUN IN THREAD
                        await asyncio.to_thread(
                            lambda: supabase.table("validation_evidence_private")
                                .update({
                                    "v_trust": v_trust,
                                    "stake": stake
                                })
                                .eq("evidence_id", evidence_id)
                                .execute()
                        )
                        
                        print(f"      ✅ Updated {validator_hotkey[:10]}...: v_trust={v_trust:.4f}, stake={stake:.2f} τ")
                    else:
                        print(f"      ⚠️  Validator {validator_hotkey[:10]}... not found in metagraph")
                        
                except Exception as e:
                    print(f"      ⚠️  Failed to update v_trust/stake for {validator_hotkey[:10]}...: {e}")
                    
            print(f"   ✅ Step 1 complete: Validator weights populated\n")
            
        except Exception as e:
            print(f"   ⚠️  Failed to populate validator weights: {e}")
            print(f"      Continuing with consensus calculation (may use default weights)...")
        
        # ========================================================================
        # PRIORITY 3: Weighted consensus calculation (already implemented in consensus.py)
        # ========================================================================
        # Query all leads validated in this epoch - USE PAGINATION
        # CRITICAL: .range() does NOT override Supabase's 1000-row limit per request!
        # We must paginate through results in batches of 1000 to get all rows.
        # With 900 leads × 3 validators = 2700 rows, we need 3 batches.
        
        all_lead_ids = []
        offset = 0
        batch_size = 1000
        
        while True:
            result = await asyncio.to_thread(
                lambda o=offset: supabase.table("validation_evidence_private")
                    .select("lead_id")
                    .eq("epoch_id", epoch_id)
                    .range(o, o + batch_size - 1)
                    .execute()
            )
            
            if not result.data:
                break
            
            all_lead_ids.extend([row["lead_id"] for row in result.data])
            print(f"   📥 Fetched batch {offset // batch_size + 1}: {len(result.data)} rows (total: {len(all_lead_ids)})")
            
            if len(result.data) < batch_size:
                break  # Last batch
            
            offset += batch_size
        
        # Get unique lead IDs
        unique_leads = list(set(all_lead_ids))
        
        if not unique_leads:
            print(f"   ℹ️  No leads to compute consensus for in epoch {epoch_id}")
            return
        
        print(f"   📊 Step 2: Computing consensus for {len(unique_leads)} leads...\n")
        
        approved_count = 0
        rejected_count = 0
        
        for i, lead_id in enumerate(unique_leads, 1):
            try:
                print(f"      🔍 [{i}/{len(unique_leads)}] Lead {lead_id[:8]}...")
                
                # ========================================================================
                # PRIORITY 3: Compute weighted consensus using v_trust × stake
                # ========================================================================
                outcome = await compute_weighted_consensus(lead_id, epoch_id)
                print(f"         📊 Consensus: {outcome['final_decision']} (rep: {outcome['final_rep_score']:.2f}, weight: {outcome['consensus_weight']:.2f})")
                
                # ========================================================================
                # PRIORITY 1: Aggregate validator responses and populate leads_private
                # ========================================================================
                print(f"         📦 Aggregating validator responses...")
                
                # Query all validator responses for this lead - RUN IN THREAD
                responses_result = await asyncio.to_thread(
                    lambda: supabase.table("validation_evidence_private")
                        .select("validator_hotkey, decision, rep_score, rejection_reason, revealed_ts, v_trust, stake")
                        .eq("lead_id", lead_id)
                        .eq("epoch_id", epoch_id)
                        .not_.is_("decision", "null")
                        .execute()
                )
                
                # DEBUG: Log query results
                print(f"         🔍 Query returned {len(responses_result.data)} validator responses")
                
                # ========================================================================
                # CRITICAL: If 0 responses, leave lead as pending_validation (FIFO queue)
                # ========================================================================
                if len(responses_result.data) == 0:
                    print(f"         ⚠️  WARNING: No validator responses found for lead {lead_id[:8]}...")
                    print(f"            This means either:")
                    print(f"            1. No validators revealed their decisions yet")
                    print(f"            2. Validators skipped this lead (timeout, error, etc.)")
                    print(f"            3. Query filters are too restrictive")
                    
                    # Query again WITHOUT the decision filter to debug
                    debug_result = await asyncio.to_thread(
                        lambda: supabase.table("validation_evidence_private")
                            .select("evidence_id, validator_hotkey, decision, revealed_ts")
                            .eq("lead_id", lead_id)
                            .eq("epoch_id", epoch_id)
                            .execute()
                    )
                    print(f"            Debug query (no decision filter): {len(debug_result.data)} records found")
                    if len(debug_result.data) > 0:
                        for rec in debug_result.data:
                            print(f"               - Validator {rec['validator_hotkey'][:10]}...: decision={rec['decision']}, revealed={rec['revealed_ts']}")
                    
                    # ════════════════════════════════════════════════════════════════════
                    # CRITICAL: Do NOT mark as denied - leave as pending_validation
                    # ════════════════════════════════════════════════════════════════════
                    print(f"         🔄 Keeping lead {lead_id[:8]}... as pending_validation (will retry in next epoch)")
                    print(f"            Lead stays at top of FIFO queue for next epoch")
                    
                    # Clear ALL consensus-related columns to reset for next epoch
                    try:
                        clear_result = await asyncio.to_thread(
                            lambda: supabase.table("leads_private")
                                .update({
                                    "epoch_summary": None,  # Clear epoch summary
                                    "consensus_votes": None,  # Clear consensus votes
                                    "validators_responded": None,  # Clear validators who responded
                                    "validator_responses": None,  # Clear individual validator responses
                                    "rep_score": None,  # Clear reputation score
                                })
                                .eq("lead_id", lead_id)
                                .execute()
                        )
                        print(f"         ✅ Cleared all consensus columns (epoch_summary, consensus_votes, validators_responded, validator_responses, rep_score)")
                        print(f"            Lead ready for next epoch with clean slate")
                    except Exception as e:
                        print(f"         ⚠️  Failed to clear consensus columns: {e}")
                    
                    # Skip to next lead (don't update status)
                    continue
                
                # Build validators_responded array
                validators_responded = [r['validator_hotkey'] for r in responses_result.data]
                
                # Build validator_responses array
                validator_responses = []
                for r in responses_result.data:
                    validator_responses.append({
                        "validator": r['validator_hotkey'],
                        "decision": r['decision'],
                        "rep_score": r['rep_score'],
                        "rejection_reason": r.get('rejection_reason'),
                        "submitted_at": r.get('revealed_ts'),
                        "v_trust": r.get('v_trust'),
                        "stake": r.get('stake')
                    })
                
                # Build consensus_votes object
                approve_count = sum(1 for r in responses_result.data if r['decision'] == 'approve')
                deny_count = len(responses_result.data) - approve_count
                
                consensus_votes = {
                    "total_validators": outcome['validator_count'],  # All validators who responded
                    "responded": len(responses_result.data),
                    "approve": approve_count,
                    "deny": deny_count,
                    "consensus": outcome['final_decision'],
                    "avg_rep_score": outcome['final_rep_score'],
                    "total_weight": outcome['consensus_weight'],
                    "approval_ratio": outcome['approval_ratio'],
                    "consensus_timestamp": datetime.utcnow().isoformat()
                }
                
                # Determine final status
                final_status = "approved" if outcome['final_decision'] == 'approve' else "denied"
                
                # Final rep_score (CRITICAL FIX: 0 for denied, not NULL)
                final_rep_score = outcome['final_rep_score'] if outcome['final_decision'] == 'approve' else 0
                
                # Extract ICP adjustment/multiplier from validator evidence_blob (CRITICAL FIX)
                # Validators calculate icp_adjustment during automated_checks based on ICP_DEFINITIONS
                # NEW FORMAT: values -15 to +20 (adjustment points)
                # OLD FORMAT: values 1.0, 1.5, 5.0 (multipliers) - backwards compatible
                # We need to extract it from the consensus-winning validators' evidence
                is_icp_multiplier = 0.0  # Default for new leads (0 adjustment)
                try:
                    # Query evidence_blobs from validators who approved this lead
                    # We'll take the average if validators disagree (shouldn't happen)
                    evidence_result = await asyncio.to_thread(
                        lambda: supabase.table("validation_evidence_private")
                            .select("evidence_blob")
                            .eq("lead_id", lead_id)
                            .eq("epoch_id", epoch_id)
                            .eq("decision", "approve")  # Only from approving validators
                            .not_.is_("evidence_blob", "null")
                            .execute()
                    )
                    
                    if evidence_result.data and len(evidence_result.data) > 0:
                        # Extract is_icp_multiplier from each validator's evidence_blob
                        multipliers = []
                        for record in evidence_result.data:
                            evidence_blob = record.get("evidence_blob", {})
                            if isinstance(evidence_blob, str):
                                evidence_blob = json.loads(evidence_blob)
                        
                            # Extract is_icp_multiplier directly from evidence_blob (top-level key)
                            # Values: 1.0 (default), 20.0 (+20 ICP bonus), or negative for penalties
                            multiplier = evidence_blob.get("is_icp_multiplier", 1.0)
                            multipliers.append(multiplier)
                        
                        # Use the most common value (or average if all different)
                        if multipliers:
                            from collections import Counter
                            counter = Counter(multipliers)
                            is_icp_multiplier = counter.most_common(1)[0][0]
                            # Detect format: OLD (1.0, 1.5, 5.0) vs NEW (integers -15 to +20)
                            if is_icp_multiplier in {1.0, 1.5, 5.0}:
                                print(f"         🎯 ICP Multiplier (legacy): {is_icp_multiplier}x (from {len(multipliers)} approving validators)")
                            else:
                                print(f"         🎯 ICP Adjustment: {int(is_icp_multiplier):+d} points (from {len(multipliers)} approving validators)")
                    else:
                        print(f"         📋 No approving validators with evidence_blob - using default adjustment 0")
                        
                except Exception as e:
                    print(f"         ⚠️  Could not extract ICP adjustment from evidence: {e}")
                    print(f"            Using default adjustment 0")
                
                # ========================================================================
                # Update leads_private with ALL aggregated data
                # ========================================================================
                print(f"         💾 Updating leads_private with aggregated data...")
                print(f"            - status: {final_status}")
                print(f"            - validators_responded: {len(validators_responded)} validators")
                print(f"            - validator_responses: {len(validator_responses)} responses")
                print(f"            - consensus_votes: {consensus_votes.get('approve', 0)} approve, {consensus_votes.get('deny', 0)} deny")
                print(f"            - rep_score: {final_rep_score}")
                # Log ICP value with appropriate label based on format
                if is_icp_multiplier in {1.0, 1.5, 5.0}:
                    print(f"            - icp_multiplier (legacy): {is_icp_multiplier}x")
                else:
                    print(f"            - icp_adjustment: {int(is_icp_multiplier):+d} points")
                
                try:
                    update_result = await asyncio.to_thread(
                        lambda: supabase.table("leads_private")
                            .update({
                                "status": final_status,
                                "validators_responded": validators_responded,
                                "validator_responses": validator_responses,
                                "consensus_votes": consensus_votes,
                                "rep_score": final_rep_score,
                                "is_icp_multiplier": is_icp_multiplier,
                                "rep_score_version": "v1/chksv2",  # Shortened to 9 chars (VARCHAR(10) limit)
                                "epoch_summary": outcome  # Keep existing epoch_summary for backwards compatibility
                            })
                            .eq("lead_id", lead_id)
                            .execute()
                    )
                    
                    # Verify update succeeded
                    if update_result.data and len(update_result.data) > 0:
                        print(f"         ✅ leads_private updated successfully")
                    else:
                        print(f"         ⚠️  WARNING: Update returned no data (lead_id might not exist or update failed)")
                        print(f"            Update result: {update_result}")
                
                except Exception as e:
                    print(f"         ❌ ERROR: Failed to update leads_private: {e}")
                    import traceback
                    traceback.print_exc()
                    # Don't stop consensus for other leads - just log the error
                    continue
                
                # Log CONSENSUS_RESULT publicly for miner transparency
                await log_consensus_result(lead_id, epoch_id, outcome)
                
                if outcome['final_decision'] == 'approve':
                    approved_count += 1
                else:
                    rejected_count += 1
                    
                    # CRITICAL: Increment rejection count for miner (validator-rejected leads)
                    try:
                        print(f"         📊 Lead rejected - incrementing miner's rejection count...")
                        
                        # Fetch miner hotkey from lead_blob - RUN IN THREAD
                        lead_result = await asyncio.to_thread(
                            lambda: supabase.table("leads_private")
                                .select("lead_blob")
                                .eq("lead_id", lead_id)
                                .execute()
                        )
                        
                        if lead_result.data and len(lead_result.data) > 0:
                            lead_blob = lead_result.data[0].get("lead_blob", {})
                            miner_hotkey = lead_blob.get("wallet_ss58")
                            
                            if miner_hotkey:
                                # Increment rejection count for this miner
                                # NOTE: Use mark_submission_failed() NOT increment_submission()!
                                # The submission was already counted in reserve_submission_slot() at /submit time.
                                # increment_submission() would DOUBLE-COUNT the submission.
                                from gateway.utils.rate_limiter import mark_submission_failed
                                updated_stats = mark_submission_failed(miner_hotkey)
                                
                                print(f"         ✅ Rejection count incremented for {miner_hotkey[:20]}...")
                                print(f"            Stats: submissions={updated_stats['submissions']}/10, rejections={updated_stats['rejections']}/8")
                            else:
                                print(f"         ⚠️  Could not find miner_hotkey in lead_blob")
                        else:
                            print(f"         ⚠️  Could not fetch lead_blob for rejection count")
                            
                    except Exception as e:
                        print(f"         ⚠️  Failed to increment rejection count: {e}")
                
                print(f"      ✅ Lead {lead_id[:8]}...: {final_status.upper()} (rep: {final_rep_score if final_rep_score else 0:.2f}, validators: {len(validators_responded)})")
            
            except Exception as e:
                print(f"      ❌ Failed to compute consensus for lead {lead_id[:8]}...: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n   📊 Epoch {epoch_id} consensus complete:")
        print(f"      ✅ {approved_count} leads approved")
        print(f"      ❌ {rejected_count} leads denied")
        print(f"      📊 Total leads processed: {len(unique_leads)}")
    
    except Exception as e:
        print(f"   ❌ Failed to compute epoch consensus: {e}")
        import traceback
        traceback.print_exc()


async def log_consensus_result(lead_id: str, epoch_id: int, outcome: dict):
    """
    Log CONSENSUS_RESULT event to transparency log for miner transparency.
    
    Miners can query these events to see their lead outcomes, including:
    - Final decision (approve/deny)
    - Final reputation score (weighted average)
    - Primary rejection reason (most common among validators)
    - Validator count and consensus weight
    - Email hash (for tracking specific lead)
    
    This provides full transparency to miners without revealing individual
    validator decisions or evidence.
    
    Args:
        lead_id: Lead UUID
        epoch_id: Epoch ID
        outcome: Consensus result from compute_weighted_consensus()
    """
    try:
        # Fetch lead_blob to compute email_hash and get ICP multiplier
        lead_result = await asyncio.to_thread(
            lambda: supabase.table("leads_private")
                .select("lead_blob, is_icp_multiplier")
                .eq("lead_id", lead_id)
                .execute()
        )
        
        email_hash = None
        linkedin_combo_hash = None
        is_icp_multiplier = 1.0  # Default for old leads
        if lead_result.data and len(lead_result.data) > 0:
            lead_blob = lead_result.data[0].get("lead_blob", {})
            if isinstance(lead_blob, str):
                lead_blob = json.loads(lead_blob)
            
            # Extract email and compute hash (same logic as submit.py)
            email = lead_blob.get("email", "").strip().lower()
            if email:
                email_hash = hashlib.sha256(email.encode()).hexdigest()
            
            # Compute linkedin_combo_hash for person+company duplicate detection
            linkedin_url = lead_blob.get("linkedin", "")
            company_linkedin_url = lead_blob.get("company_linkedin", "")
            linkedin_combo_hash = compute_linkedin_combo_hash(linkedin_url, company_linkedin_url)
            
            # Get ICP adjustment/multiplier (from DB column, falls back to 0 for new format if missing)
            is_icp_multiplier = lead_result.data[0].get("is_icp_multiplier", 0.0)
        
        payload = {
            "lead_id": lead_id,
            "epoch_id": epoch_id,
            "final_decision": outcome["final_decision"],
            "final_rep_score": outcome["final_rep_score"],
            "is_icp_multiplier": is_icp_multiplier,
            "primary_rejection_reason": outcome["primary_rejection_reason"],
            "validator_count": outcome["validator_count"],
            "consensus_weight": outcome["consensus_weight"]
        }
        
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)  # Handle datetime objects
        payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        
        log_entry = {
            "event_type": "CONSENSUS_RESULT",
            "actor_hotkey": "system",  # System-generated event
            "nonce": str(uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "payload_hash": payload_hash,
            "build_id": BUILD_ID,
            "signature": "system",  # No signature for system events
            "payload": payload,
            "email_hash": email_hash,  # Add email_hash for transparency_log table
            "linkedin_combo_hash": linkedin_combo_hash  # Add for person+company duplicate detection
        }
        
        # Write to TEE buffer (authoritative, hardware-protected)
        # Then mirrors to Supabase for queries
        from gateway.utils.logger import log_event
        result = await log_event(log_entry)
        
        tee_sequence = result.get("sequence")
        linkedin_hash_display = linkedin_combo_hash[:16] if linkedin_combo_hash else 'NULL'
        print(f"         📊 Logged CONSENSUS_RESULT for lead {lead_id[:8]}... (TEE seq={tee_sequence}, email={email_hash[:16] if email_hash else 'NULL'}..., linkedin_combo={linkedin_hash_display}...)")
    
    except Exception as e:
        print(f"         ❌ Failed to log CONSENSUS_RESULT for lead {lead_id[:8]}...: {e}")


if __name__ == "__main__":
    """
    Run epoch lifecycle task as standalone module.
    
    Usage: python -m gateway.tasks.epoch_lifecycle
    """
    print("🚀 Starting Epoch Lifecycle Task...")
    asyncio.run(epoch_lifecycle_task())
