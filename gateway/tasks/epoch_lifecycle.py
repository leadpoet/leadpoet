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
    get_current_epoch_id,
    get_epoch_start_time,
    get_epoch_end_time,
    get_epoch_close_time,
    is_epoch_active,
    is_epoch_closed
)
from gateway.utils.merkle import compute_merkle_root
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BUILD_ID, MAX_LEADS_PER_EPOCH
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def epoch_lifecycle_task():
    """
    Background task to manage epoch lifecycle events.
    
    Runs every 30 seconds and checks:
    - Is it time to start new epoch? → Log EPOCH_INITIALIZATION
    - Is it time to end validation? → Log EPOCH_END + EPOCH_INPUTS
    - Is it time to close epoch? → Trigger reveal phase + consensus
    
    This task ensures all epoch events are logged deterministically and
    consensus is computed automatically.
    """
    
    last_epoch_id = None
    validation_ended_epochs = set()  # Track which epochs we've logged EPOCH_END for
    closed_epochs = set()  # Track which epochs we've processed consensus for
    
    print("🚀 Epoch lifecycle task started")
    
    while True:
        try:
            current_epoch = get_current_epoch_id()
            now = datetime.utcnow()
            
            epoch_start = get_epoch_start_time(current_epoch)
            epoch_end = get_epoch_end_time(current_epoch)
            epoch_close = get_epoch_close_time(current_epoch)
            
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
                await compute_and_log_epoch_initialization(current_epoch, epoch_start, epoch_end, epoch_close)
                
                last_epoch_id = current_epoch
                print(f"   ✅ Epoch {current_epoch} initialized\n")
            
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
            # Check if epoch closed (block 360) - time to reveal + consensus
            # ========================================================================
            time_since_close = (now - epoch_close).total_seconds()
            
            # FIX: Process ANY closed epoch that hasn't been processed yet
            # (not just within 120 second window - catches missed epochs after restart)
            if time_since_close >= 0 and current_epoch not in closed_epochs:
                # Check if this epoch has validation evidence
                evidence_check = supabase.table("validation_evidence_private") \
                    .select("lead_id", count="exact") \
                    .eq("epoch_id", current_epoch) \
                    .limit(1) \
                    .execute()
                
                has_evidence = evidence_check.count > 0 if evidence_check.count is not None else len(evidence_check.data) > 0
                
                if has_evidence:
                    print(f"\n{'='*80}")
                    print(f"🔓 EPOCH {current_epoch} CLOSED - STARTING REVEAL & CONSENSUS")
                    print(f"{'='*80}")
                    print(f"   Closed at: {epoch_close.isoformat()}")
                    print(f"   Time since close: {time_since_close/60:.1f} minutes")
                    
                    # Trigger reveal phase (validators should reveal their commits)
                    await trigger_reveal_phase(current_epoch)
                    
                    # Wait a bit for reveals to come in (only if epoch just closed)
                    if time_since_close < 300:  # Within 5 minutes
                        print(f"   ⏳ Waiting 2 minutes for reveals...")
                        await asyncio.sleep(120)
                    else:
                        print(f"   ℹ️  Epoch closed {time_since_close/60:.1f} minutes ago - skipping reveal wait")
                    
                    # Compute consensus for all leads in this epoch
                    await compute_epoch_consensus(current_epoch)
                    
                    closed_epochs.add(current_epoch)
                    print(f"   ✅ Epoch {current_epoch} fully processed\n")
                elif time_since_close >= 300:  # 5 minutes after close
                    # No evidence after 5 minutes - mark as processed to avoid checking again
                    print(f"   ℹ️  Epoch {current_epoch} closed {time_since_close/60:.1f} minutes ago with no validation evidence - marking as processed")
                    closed_epochs.add(current_epoch)
            
            # Clean up old tracking sets to prevent memory growth
            if len(validation_ended_epochs) > 100:
                # Keep only recent 50 epochs
                recent = sorted(list(validation_ended_epochs))[-50:]
                validation_ended_epochs = set(recent)
            
            if len(closed_epochs) > 100:
                recent = sorted(list(closed_epochs))[-50:]
                closed_epochs = set(recent)
            
            # Sleep 30 seconds before next check
            await asyncio.sleep(30)
        
        except Exception as e:
            print(f"❌ Epoch lifecycle error: {e}")
            import traceback
            traceback.print_exc()
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
        
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
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
        # 1. Query pending leads from queue (FIFO order)
        # ========================================================================
        result = supabase.table("leads_private") \
            .select("lead_id") \
            .is_("epoch_summary", "null") \
            .order("created_ts") \
            .execute()
        
        lead_ids = [row["lead_id"] for row in result.data]
        
        if not lead_ids:
            queue_merkle_root = "0" * 64  # Empty queue
            pending_lead_count = 0
        else:
            queue_merkle_root = compute_merkle_root(lead_ids)
            pending_lead_count = len(lead_ids)
        
        print(f"   📊 Queue State: {queue_merkle_root[:16]}... ({pending_lead_count} pending leads)")
        
        # ========================================================================
        # 2. Get validator set for this epoch
        # ========================================================================
        validator_set = get_validator_set(epoch_id)  # Returns List[str] of hotkeys
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


async def compute_and_log_epoch_inputs(epoch_id: int):
    """
    Compute hash of all events in epoch and log EPOCH_INPUTS event.
    
    This creates a deterministic hash of all transparency log events
    during the epoch, ensuring all validators see the same input data.
    
    Args:
        epoch_id: Epoch ID
    """
    try:
        epoch_start = get_epoch_start_time(epoch_id)
        epoch_end = get_epoch_end_time(epoch_id)
        
        # Query all events in epoch (during validation phase)
        result = supabase.table("transparency_log") \
            .select("id, event_type, payload_hash") \
            .gte("ts", epoch_start.isoformat()) \
            .lte("ts", epoch_end.isoformat()) \
            .order("id") \
            .execute()
        
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
        
        # Query how many validators submitted commits
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id", count="exact") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        commit_count = result.count if result.count is not None else 0
        
        print(f"   📊 {commit_count} validation commits to reveal")
        
        # Validators will call POST /reveal independently
        # This is just a monitoring/logging step
    
    except Exception as e:
        print(f"   ❌ Failed to trigger reveal phase: {e}")


async def compute_epoch_consensus(epoch_id: int):
    """
    Compute weighted consensus for all leads in epoch.
    
    Uses V-scores to weight validator decisions and rep_scores.
    Updates leads_private with final consensus outcomes.
    
    Args:
        epoch_id: Epoch ID
    """
    try:
        # Import here to avoid circular dependency
        from gateway.utils.consensus import compute_weighted_consensus
        
        # Query all leads validated in this epoch
        result = supabase.table("validation_evidence_private") \
            .select("lead_id") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        # Get unique lead IDs
        unique_leads = list(set([row["lead_id"] for row in result.data]))
        
        if not unique_leads:
            print(f"   ℹ️  No leads to compute consensus for in epoch {epoch_id}")
            return
        
        print(f"   📊 Computing consensus for {len(unique_leads)} leads in epoch {epoch_id}")
        
        approved_count = 0
        rejected_count = 0
        
        for lead_id in unique_leads:
            try:
                # Compute weighted consensus for this lead
                outcome = await compute_weighted_consensus(lead_id, epoch_id)
                
                # Update leads_private with final outcome
                supabase.table("leads_private") \
                    .update({"epoch_summary": outcome}) \
                    .eq("lead_id", lead_id) \
                    .execute()
                
                # Log CONSENSUS_RESULT publicly for miner transparency
                await log_consensus_result(lead_id, epoch_id, outcome)
                
                if outcome['final_decision'] == 'approve':
                    approved_count += 1
                else:
                    rejected_count += 1
                
                print(f"      ✅ Lead {lead_id[:8]}...: {outcome['final_decision']} (rep: {outcome['final_rep_score']:.2f}, reason: {outcome['primary_rejection_reason']})")
            
            except Exception as e:
                print(f"      ❌ Failed to compute consensus for lead {lead_id[:8]}...: {e}")
        
        print(f"   📊 Consensus complete: {approved_count} approved, {rejected_count} rejected")
    
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
    
    This provides full transparency to miners without revealing individual
    validator decisions or evidence.
    
    Args:
        lead_id: Lead UUID
        epoch_id: Epoch ID
        outcome: Consensus result from compute_weighted_consensus()
    """
    try:
        payload = {
            "lead_id": lead_id,
            "epoch_id": epoch_id,
            "final_decision": outcome["final_decision"],
            "final_rep_score": outcome["final_rep_score"],
            "primary_rejection_reason": outcome["primary_rejection_reason"],
            "validator_count": outcome["validator_count"],
            "consensus_weight": outcome["consensus_weight"]
        }
        
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        
        log_entry = {
            "event_type": "CONSENSUS_RESULT",
            "actor_hotkey": "system",  # System-generated event
            "nonce": str(uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "payload_hash": payload_hash,
            "build_id": BUILD_ID,
            "signature": "system",  # No signature for system events
            "payload": payload
        }
        
        # Write to TEE buffer (authoritative, hardware-protected)
        # Then mirrors to Supabase for queries
        from gateway.utils.logger import log_event
        result = await log_event(log_entry)
        
        tee_sequence = result.get("sequence")
        print(f"         📊 Logged CONSENSUS_RESULT for lead {lead_id[:8]}... (TEE seq={tee_sequence})")
    
    except Exception as e:
        print(f"         ❌ Failed to log CONSENSUS_RESULT for lead {lead_id[:8]}...: {e}")


if __name__ == "__main__":
    """
    Run epoch lifecycle task as standalone module.
    
    Usage: python -m gateway.tasks.epoch_lifecycle
    """
    print("🚀 Starting Epoch Lifecycle Task...")
    asyncio.run(epoch_lifecycle_task())
