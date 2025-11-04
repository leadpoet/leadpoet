"""
Merkle Checkpoint Background Task
==================================

Background task to create Merkle checkpoints every 10 minutes.

Computes Merkle root of all transparency log events since the last checkpoint
and stores it in the merkle_checkpoints table for integrity verification.

These checkpoints are used for:
- Daily on-chain anchoring (anchor.py queries latest checkpoint)
- Transparency log integrity verification
- Tamper detection (any modification would break the Merkle tree)

Author: LeadPoet Team
"""

import asyncio
from datetime import datetime
from typing import Optional
from gateway.utils.merkle import compute_merkle_root_from_hashes
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def checkpoint_task():
    """
    Background task to create Merkle checkpoints every 10 minutes.
    
    Computes Merkle root of all transparency log events since last checkpoint
    and stores it in merkle_checkpoints table.
    
    Process:
    1. Query transparency_log events since last checkpoint (by id)
    2. Extract payload_hash from each event
    3. Compute Merkle root from event hashes
    4. Store checkpoint with seq_start, seq_end, merkle_root
    5. Sleep for 10 minutes
    6. Repeat
    
    Checkpoint Structure:
        {
            "checkpoint_ts": "2024-11-04T12:00:00Z",
            "seq_start": 1234,
            "seq_end": 5678,
            "merkle_root": "abc123..."
        }
    
    Note: If no new events exist, skips checkpoint creation to avoid
    duplicate entries. The last checkpoint ID is tracked across iterations.
    """
    
    print("📊 Starting checkpoint task...")
    print(f"   Interval: 10 minutes (600 seconds)")
    print(f"   Purpose: Create Merkle checkpoints for transparency log integrity")
    print()
    
    last_checkpoint_id = 0
    
    while True:
        try:
            # Query events since last checkpoint
            result = supabase.table("transparency_log") \
                .select("id, event_type, payload_hash") \
                .gt("id", last_checkpoint_id) \
                .order("id") \
                .execute()
            
            events = result.data
            
            if not events:
                print(f"   ℹ️  No new events since last checkpoint (last_id: {last_checkpoint_id})")
                print(f"   Waiting 10 minutes...")
                await asyncio.sleep(600)  # 10 minutes
                continue
            
            # Extract event hashes for Merkle tree
            # Note: payload_hash is already a SHA256 hex hash from transparency log
            event_hashes = [e["payload_hash"] for e in events]
            
            # Compute Merkle root from pre-computed hashes
            merkle_root = compute_merkle_root_from_hashes(event_hashes)
            
            seq_start = events[0]["id"]
            seq_end = events[-1]["id"]
            event_count = len(events)
            
            print(f"📊 Creating checkpoint...")
            print(f"   Events: {event_count} ({seq_start} - {seq_end})")
            print(f"   Merkle Root: {merkle_root[:32]}...{merkle_root[-8:]}")
            
            # Insert checkpoint into merkle_checkpoints table
            checkpoint_entry = {
                "checkpoint_ts": datetime.utcnow().isoformat(),
                "seq_start": seq_start,
                "seq_end": seq_end,
                "merkle_root": merkle_root
            }
            
            supabase.table("merkle_checkpoints").insert(checkpoint_entry).execute()
            
            print(f"   ✅ Checkpoint created successfully")
            print(f"   Next checkpoint in 10 minutes...")
            print()
            
            # Update last checkpoint ID for next iteration
            last_checkpoint_id = seq_end
            
            # Sleep 10 minutes
            await asyncio.sleep(600)
        
        except Exception as e:
            print(f"❌ Checkpoint error: {e}")
            print(f"   Retrying in 10 minutes...")
            print()
            await asyncio.sleep(600)


async def create_checkpoint_now(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None
) -> Optional[dict]:
    """
    Manually create a checkpoint for a specific range of events.
    
    This function is useful for testing or creating checkpoints on-demand
    without waiting for the 10-minute interval.
    
    Args:
        start_id: Optional starting event ID (if None, starts from last checkpoint)
        end_id: Optional ending event ID (if None, uses all events up to latest)
    
    Returns:
        Dictionary with checkpoint data if successful, None otherwise
    
    Example:
        >>> import asyncio
        >>> from gateway.tasks.checkpoints import create_checkpoint_now
        >>> 
        >>> # Create checkpoint for all events
        >>> checkpoint = asyncio.run(create_checkpoint_now())
        >>> 
        >>> # Create checkpoint for specific range
        >>> checkpoint = asyncio.run(create_checkpoint_now(start_id=100, end_id=500))
    """
    try:
        print(f"📊 Creating manual checkpoint...")
        
        # Build query
        query = supabase.table("transparency_log") \
            .select("id, event_type, payload_hash") \
            .order("id")
        
        # Apply filters if provided
        if start_id is not None:
            query = query.gt("id", start_id)
            print(f"   Start ID: {start_id}")
        
        if end_id is not None:
            query = query.lte("id", end_id)
            print(f"   End ID: {end_id}")
        
        result = query.execute()
        events = result.data
        
        if not events:
            print(f"   ⚠️  No events found in range")
            return None
        
        # Extract event hashes
        event_hashes = [e["payload_hash"] for e in events]
        
        # Compute Merkle root
        merkle_root = compute_merkle_root_from_hashes(event_hashes)
        
        seq_start = events[0]["id"]
        seq_end = events[-1]["id"]
        event_count = len(events)
        
        print(f"   Events: {event_count} ({seq_start} - {seq_end})")
        print(f"   Merkle Root: {merkle_root[:32]}...{merkle_root[-8:]}")
        
        # Insert checkpoint
        checkpoint_entry = {
            "checkpoint_ts": datetime.utcnow().isoformat(),
            "seq_start": seq_start,
            "seq_end": seq_end,
            "merkle_root": merkle_root
        }
        
        result = supabase.table("merkle_checkpoints").insert(checkpoint_entry).execute()
        
        print(f"   ✅ Manual checkpoint created successfully")
        print()
        
        return checkpoint_entry
    
    except Exception as e:
        print(f"❌ Manual checkpoint failed: {e}")
        print()
        return None


async def get_latest_checkpoint() -> Optional[dict]:
    """
    Get the latest checkpoint from merkle_checkpoints table.
    
    Returns:
        Dictionary with latest checkpoint data, or None if no checkpoints exist
    
    Example:
        >>> import asyncio
        >>> from gateway.tasks.checkpoints import get_latest_checkpoint
        >>> 
        >>> checkpoint = asyncio.run(get_latest_checkpoint())
        >>> print(f"Latest checkpoint covers events {checkpoint['seq_start']} - {checkpoint['seq_end']}")
    """
    try:
        result = supabase.table("merkle_checkpoints") \
            .select("*") \
            .order("id", desc=True) \
            .limit(1) \
            .execute()
        
        if not result.data:
            return None
        
        return result.data[0]
    
    except Exception as e:
        print(f"❌ Error fetching latest checkpoint: {e}")
        return None


async def verify_checkpoint_integrity(checkpoint_id: int) -> bool:
    """
    Verify integrity of a specific checkpoint by recomputing its Merkle root.
    
    This function queries the transparency log events in the checkpoint's range,
    recomputes the Merkle root, and compares it to the stored value.
    
    Args:
        checkpoint_id: ID of the checkpoint to verify
    
    Returns:
        True if checkpoint is valid, False otherwise
    
    Example:
        >>> import asyncio
        >>> from gateway.tasks.checkpoints import verify_checkpoint_integrity
        >>> 
        >>> is_valid = asyncio.run(verify_checkpoint_integrity(checkpoint_id=123))
        >>> print(f"Checkpoint valid: {is_valid}")
    """
    try:
        # Get checkpoint
        result = supabase.table("merkle_checkpoints") \
            .select("*") \
            .eq("id", checkpoint_id) \
            .execute()
        
        if not result.data:
            print(f"❌ Checkpoint {checkpoint_id} not found")
            return False
        
        checkpoint = result.data[0]
        stored_root = checkpoint["merkle_root"]
        seq_start = checkpoint["seq_start"]
        seq_end = checkpoint["seq_end"]
        
        print(f"🔍 Verifying checkpoint {checkpoint_id}...")
        print(f"   Range: {seq_start} - {seq_end}")
        print(f"   Stored Root: {stored_root[:32]}...{stored_root[-8:]}")
        
        # Query events in range
        result = supabase.table("transparency_log") \
            .select("id, payload_hash") \
            .gte("id", seq_start) \
            .lte("id", seq_end) \
            .order("id") \
            .execute()
        
        events = result.data
        
        if not events:
            print(f"   ❌ No events found in range")
            return False
        
        # Recompute Merkle root
        event_hashes = [e["payload_hash"] for e in events]
        computed_root = compute_merkle_root_from_hashes(event_hashes)
        
        print(f"   Computed Root: {computed_root[:32]}...{computed_root[-8:]}")
        
        # Compare
        is_valid = computed_root == stored_root
        
        if is_valid:
            print(f"   ✅ Checkpoint is VALID")
        else:
            print(f"   ❌ Checkpoint is INVALID (Merkle roots don't match)")
        
        print()
        
        return is_valid
    
    except Exception as e:
        print(f"❌ Verification error: {e}")
        print()
        return False


if __name__ == "__main__":
    # For testing - create a checkpoint immediately
    asyncio.run(create_checkpoint_now())

