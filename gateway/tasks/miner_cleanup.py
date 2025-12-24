"""
Deregistered Miner Cleanup
===========================

Removes leads from miners who have deregistered from the subnet.

This runs once per epoch at block 10-15 to clean up:
- denied leads
- pending_validation leads
- validating leads

From miners who are no longer registered on the Bittensor subnet.

Why this matters:
- Prevents stale data accumulation
- Removes leads that can never be served (miner left)
- Keeps database clean and performant
"""

import asyncio
from datetime import datetime
from typing import Set, List
import json
import hashlib
import uuid

from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def cleanup_deregistered_miner_leads(epoch_id: int):
    """
    Remove leads from miners who have deregistered from the subnet.
    
    This runs at block 10-15 of each epoch to:
    1. Query all non-final leads (validating, pending_validation, denied)
    2. Check which miners are still registered on subnet
    3. Delete leads from deregistered miners
    4. Log cleanup to transparency_log and TEE
    
    Args:
        epoch_id: Current epoch number
    
    Example:
        >>> await cleanup_deregistered_miner_leads(19215)
        ✅ Cleaned up 127 leads from 3 deregistered miners
    """
    try:
        print(f"\n{'='*80}")
        print(f"🧹 DEREGISTERED MINER CLEANUP: Epoch {epoch_id}")
        print(f"{'='*80}")
        
        # ========================================================================
        # Step 1: Get metagraph (use cache - don't force refresh)
        # ========================================================================
        print(f"   📊 Fetching metagraph...")
        
        from gateway.utils.registry import get_metagraph_async
        
        try:
            metagraph = await asyncio.wait_for(
                get_metagraph_async(),
                timeout=30.0  # 30 second timeout
            )
            
            registered_hotkeys = set(metagraph.hotkeys)
            print(f"   ✅ Metagraph loaded: {len(registered_hotkeys)} registered hotkeys")
        
        except asyncio.TimeoutError:
            print(f"   ⚠️  Metagraph fetch timeout - skipping cleanup this epoch")
            return
        except Exception as e:
            print(f"   ⚠️  Metagraph fetch error: {e}")
            print(f"      Skipping cleanup this epoch")
            return
        
        # ========================================================================
        # Step 2: Query non-final leads (optimized with index)
        # ========================================================================
        print(f"   🔍 Querying non-final leads...")
        
        try:
            # Query uses composite index: idx_leads_private_status_miner
            # This is FAST even with 100K+ rows
            result = await asyncio.to_thread(
                lambda: supabase.table("leads_private")
                    .select("lead_id, miner_hotkey, status, created_ts")
                    .in_("status", ["validating", "pending_validation", "denied"])
                    .execute()
            )
            
            all_leads = result.data
            print(f"   ✅ Found {len(all_leads)} non-final leads")
        
        except Exception as e:
            print(f"   ❌ Error querying leads: {e}")
            return
        
        if not all_leads:
            print(f"   ℹ️  No non-final leads to clean up")
            print(f"{'='*80}\n")
            return
        
        # ========================================================================
        # Step 3: Identify leads from deregistered miners
        # ========================================================================
        print(f"   🔍 Checking which miners are deregistered...")
        
        leads_to_delete = []
        deregistered_miners = {}  # {miner_hotkey: count}
        
        for lead in all_leads:
            miner_hotkey = lead["miner_hotkey"]
            
            if miner_hotkey not in registered_hotkeys:
                # Miner is deregistered - mark lead for deletion
                leads_to_delete.append(lead["lead_id"])
                
                # Track deregistered miner stats
                if miner_hotkey not in deregistered_miners:
                    deregistered_miners[miner_hotkey] = {
                        "hotkey": miner_hotkey,
                        "lead_count": 0,
                        "statuses": {"validating": 0, "pending_validation": 0, "denied": 0}
                    }
                
                deregistered_miners[miner_hotkey]["lead_count"] += 1
                deregistered_miners[miner_hotkey]["statuses"][lead["status"]] += 1
        
        if not leads_to_delete:
            print(f"   ✅ All miners are still registered - no cleanup needed")
            print(f"{'='*80}\n")
            return
        
        print(f"   📊 Found {len(leads_to_delete)} leads from {len(deregistered_miners)} deregistered miners")
        
        # ========================================================================
        # Step 4: Batch delete leads (in chunks to avoid Supabase limits)
        # ========================================================================
        print(f"   🗑️  Deleting {len(leads_to_delete)} leads...")
        
        # Supabase has a limit on query size - delete in batches of 100
        BATCH_SIZE = 100
        total_deleted = 0
        total_evidence_deleted = 0
        deletion_failed = False
        deletion_error = None
        
        try:
            # CRITICAL: Delete from validation_evidence_private FIRST
            # to avoid foreign key constraint violation
            print(f"   🔍 Step 4a: Deleting validation evidence for {len(leads_to_delete)} leads...")
            
            for i in range(0, len(leads_to_delete), BATCH_SIZE):
                batch = leads_to_delete[i:i + BATCH_SIZE]
                
                print(f"      🔍 Evidence Batch {i//BATCH_SIZE + 1}: Deleting evidence for {len(batch)} leads...")
                
                # Delete validation_evidence_private rows that reference these leads
                evidence_result = await asyncio.to_thread(
                    lambda b=batch: supabase.table("validation_evidence_private")
                        .delete()
                        .in_("lead_id", b)
                        .execute()
                )
                
                evidence_deleted = len(evidence_result.data) if evidence_result.data else 0
                total_evidence_deleted += evidence_deleted
                print(f"         Deleted {evidence_deleted} validation evidence records")
            
            print(f"   ✅ Deleted {total_evidence_deleted} validation evidence records")
            print(f"   🔍 Step 4b: Deleting leads from leads_private...")
            
            # Now delete the leads themselves
            for i in range(0, len(leads_to_delete), BATCH_SIZE):
                batch = leads_to_delete[i:i + BATCH_SIZE]
                
                print(f"      🔍 Batch {i//BATCH_SIZE + 1}: Attempting to delete {len(batch)} leads...")
                print(f"         Sample lead_ids: {batch[:3]}")
                
                # Execute deletion and check result
                result = await asyncio.to_thread(
                    lambda b=batch: supabase.table("leads_private")
                        .delete()
                        .in_("lead_id", b)
                        .execute()
                )
                
                # Debug: Print full result to understand Supabase response
                print(f"         Supabase response: status={result.count if hasattr(result, 'count') else 'N/A'}, data_length={len(result.data) if result.data else 0}")
                
                # Verify deletion worked (check if response has data)
                # NOTE: Supabase delete() returns the deleted rows in result.data
                deleted_count = len(result.data) if result.data else 0
                
                if deleted_count != len(batch):
                    print(f"      ⚠️  Batch {i//BATCH_SIZE + 1}: Expected to delete {len(batch)}, actually deleted {deleted_count}")
                    print(f"         This may indicate RLS policy issues or leads already deleted")
                
                total_deleted += deleted_count
                print(f"      ✅ Batch {i//BATCH_SIZE + 1}/{(len(leads_to_delete) + BATCH_SIZE - 1)//BATCH_SIZE}: Deleted {deleted_count}/{len(batch)} leads")
            
            if total_deleted == len(leads_to_delete):
                print(f"   ✅ Successfully deleted {total_deleted} leads from deregistered miners")
            else:
                print(f"   ⚠️  Partial deletion: {total_deleted}/{len(leads_to_delete)} leads deleted")
                deletion_failed = True
                deletion_error = f"Only deleted {total_deleted}/{len(leads_to_delete)} leads"
        
        except Exception as e:
            print(f"   ❌ Error deleting leads: {e}")
            import traceback
            traceback.print_exc()
            deletion_failed = True
            deletion_error = str(e)
            # Don't fail entire cleanup - continue to logging
        
        # ========================================================================
        # Step 5: Log to transparency_log and TEE
        # ========================================================================
        print(f"   📝 Logging cleanup to transparency log...")
        
        # Prepare cleanup report (include actual deletion results)
        cleanup_report = {
            "epoch_id": epoch_id,
            "total_leads_identified": len(leads_to_delete),
            "total_leads_deleted": total_deleted,
            "total_evidence_deleted": total_evidence_deleted,
            "deletion_success": not deletion_failed,
            "deletion_error": deletion_error if deletion_failed else None,
            "deregistered_miner_count": len(deregistered_miners),
            "deregistered_miners": [
                {
                    "miner_hotkey": stats["hotkey"],
                    "leads_identified": stats["lead_count"],
                    "status_breakdown": stats["statuses"]
                }
                for stats in deregistered_miners.values()
            ],
            "cleanup_timestamp": datetime.utcnow().isoformat()
        }
        
        # Log to transparency_log (Supabase)
        # Note: epoch_id is in payload, not a separate column
        # Generate unique nonce (UUID) for this cleanup
        event_nonce = str(uuid.uuid4())
        current_timestamp = datetime.utcnow().isoformat()
        
        try:
            # Build transparency_log entry with ALL required fields
            # Based on schema: event_type, actor_hotkey, nonce (uuid), ts (timestamp),
            # payload_hash, build_id, signature, payload, tee_sequence, tee_buffered_at, tee_buffer_size
            transparency_entry = {
                "event_type": "DEREGISTERED_MINER_REMOVAL",
                "actor_hotkey": "gateway",
                "nonce": event_nonce,  # UUID for this cleanup event
                "ts": current_timestamp,  # Use 'ts' not 'timestamp'
                "payload": cleanup_report,
                "payload_hash": hashlib.sha256(
                    json.dumps(cleanup_report, sort_keys=True, default=str).encode()  # Handle datetime objects
                ).hexdigest(),
                "build_id": "gateway",  # Required field
                "signature": "gateway_internal_cleanup",  # Required field (gateway-generated event, no cryptographic signature)
                # TEE fields (set to 0 for gateway-generated events that don't go through TEE buffer)
                "tee_sequence": 0,
                "tee_buffered_at": current_timestamp,
                "tee_buffer_size": 0
            }
            
            await asyncio.to_thread(
                lambda: supabase.table("transparency_log")
                    .insert(transparency_entry)
                    .execute()
            )
            
            print(f"   ✅ Logged DEREGISTERED_MINER_REMOVAL to transparency_log")
        
        except Exception as e:
            print(f"   ⚠️  Failed to log to transparency_log: {e}")
            import traceback
            traceback.print_exc()
        
        # Log to TEE (Arweave buffer)
        try:
            from gateway.utils.logger import log_event
            
            # Create payload for TEE
            payload_json = json.dumps(cleanup_report, sort_keys=True, separators=(',', ':'), default=str)  # Handle datetime objects
            payload_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
            
            tee_event = {
                "event_type": "DEREGISTERED_MINER_REMOVAL",
                "actor_hotkey": "gateway",  # Gateway-generated event
                "nonce": event_nonce,  # Use same UUID as transparency_log
                "ts": current_timestamp,  # Use same timestamp as transparency_log
                "payload": cleanup_report,
                "payload_hash": payload_hash,
                "build_id": "gateway",
                "signature": "gateway_internal_cleanup"  # Add signature for consistency
            }
            
            # Log to TEE enclave (uses log_event - dual logging)
            result = await log_event(tee_event)
            seq_num = result.get("sequence")
            
            print(f"   ✅ Logged to TEE buffer (seq={seq_num})")
        
        except Exception as e:
            print(f"   ⚠️  Failed to log to TEE: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - transparency_log is the primary source
        
        # ========================================================================
        # Step 6: Print summary
        # ========================================================================
        print(f"\n   📊 Cleanup Summary:")
        print(f"      Leads identified for deletion: {len(leads_to_delete)}")
        print(f"      Validation evidence deleted: {total_evidence_deleted}")
        print(f"      Leads actually deleted: {total_deleted}")
        if deletion_failed:
            print(f"      ⚠️  Deletion status: FAILED - {deletion_error}")
        else:
            print(f"      ✅ Deletion status: SUCCESS")
        print(f"      Deregistered miners: {len(deregistered_miners)}")
        
        for stats in sorted(deregistered_miners.values(), key=lambda x: x["lead_count"], reverse=True)[:5]:
            hotkey_short = stats["hotkey"][:10]
            count = stats["lead_count"]
            statuses = stats["statuses"]
            print(f"      • {hotkey_short}...: {count} leads (validating: {statuses['validating']}, pending: {statuses['pending_validation']}, denied: {statuses['denied']})")
        
        if len(deregistered_miners) > 5:
            print(f"      ... and {len(deregistered_miners) - 5} more miners")
        
        print(f"{'='*80}\n")
    
    except Exception as e:
        print(f"❌ Error in cleanup_deregistered_miner_leads: {e}")
        import traceback
        traceback.print_exc()
        # Don't crash gateway - this is a background cleanup task

