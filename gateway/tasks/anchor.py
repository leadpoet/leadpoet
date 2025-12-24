"""
Daily On-Chain Anchoring Task
==============================

Background task to anchor Merkle roots on-chain daily.

This task runs every 24 hours and commits the latest checkpoint root
to the transparency log as an ANCHOR_ROOT event.

In production, this would also commit to Bittensor subnet metadata or 
another on-chain storage mechanism for tamper-proof anchoring.

Author: LeadPoet Team
"""

import asyncio
from datetime import datetime
from typing import Optional
from gateway.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    BITTENSOR_NETWORK,
    BUILD_ID
)
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def daily_anchor_task():
    """
    Background task to anchor Merkle roots on-chain daily.
    
    Runs every 24 hours and commits the latest checkpoint root
    to Bittensor subnet metadata.
    
    Process:
    1. Query latest checkpoint root from merkle_checkpoints table
    2. Log ANCHOR_ROOT event to transparency log (public record)
    3. (Future) Commit to Bittensor subnet metadata (on-chain anchoring)
    4. Sleep for 24 hours
    5. Repeat
    
    Note: In production, this would submit the Merkle root to Bittensor
    chain using substrate extrinsics for tamper-proof anchoring.
    Currently logs to transparency log for public verification.
    """
    
    print("🔗 Starting daily anchor task...")
    print(f"   Network: {BITTENSOR_NETWORK}")
    print(f"   Interval: 24 hours")
    print()
    
    while True:
        try:
            # Get latest checkpoint root
            result = supabase.table("merkle_checkpoints") \
                .select("*") \
                .order("id", desc=True) \
                .limit(1) \
                .execute()
            
            if not result.data:
                print("⚠️  No checkpoint roots found")
                print(f"   Waiting 24 hours before retry...")
                await asyncio.sleep(86400)  # 24 hours
                continue
            
            latest_checkpoint = result.data[0]
            merkle_root = latest_checkpoint["merkle_root"]
            checkpoint_ts = latest_checkpoint["checkpoint_ts"]
            seq_start = latest_checkpoint.get("seq_start", 0)
            seq_end = latest_checkpoint.get("seq_end", 0)
            
            print(f"📍 Anchoring Merkle root on-chain...")
            print(f"   Root: {merkle_root[:32]}...{merkle_root[-8:]}")
            print(f"   Checkpoint Timestamp: {checkpoint_ts}")
            print(f"   Sequence Range: {seq_start} - {seq_end}")
            
            # Anchor to Bittensor (simplified - store in transparency log)
            # In production, this would use substrate extrinsic:
            # subtensor.commit_reveal(root=merkle_root, metadata={...})
            
            # For now, log ANCHOR_ROOT event to transparency log
            await log_anchor_event(
                merkle_root=merkle_root,
                checkpoint_ts=checkpoint_ts,
                seq_start=seq_start,
                seq_end=seq_end
            )
            
            print(f"   ✅ Merkle root anchored successfully")
            print(f"   Next anchor in 24 hours...")
            print()
            
            # Sleep 24 hours
            await asyncio.sleep(86400)
        
        except Exception as e:
            print(f"❌ Anchor error: {e}")
            print(f"   Retrying in 24 hours...")
            print()
            await asyncio.sleep(86400)


async def log_anchor_event(
    merkle_root: str,
    checkpoint_ts: str,
    seq_start: Optional[int] = None,
    seq_end: Optional[int] = None
):
    """
    Log ANCHOR_ROOT event to transparency log.
    
    This creates a public record of the Merkle root anchoring for
    transparency and verification purposes.
    
    Args:
        merkle_root: Merkle root hash (SHA256 hex)
        checkpoint_ts: Checkpoint timestamp (ISO format)
        seq_start: Optional sequence start ID
        seq_end: Optional sequence end ID
    
    Event Structure:
        {
            "event_type": "ANCHOR_ROOT",
            "actor_hotkey": "system",
            "payload": {
                "merkle_root": str,
                "checkpoint_ts": str,
                "seq_start": int,
                "seq_end": int,
                "network": str,
                "anchor_ts": str
            }
        }
    
    Note: This event is PUBLIC and can be queried by anyone to verify
    the Merkle root was anchored at a specific time.
    """
    import hashlib
    import json
    from uuid import uuid4
    
    payload = {
        "merkle_root": merkle_root,
        "checkpoint_ts": checkpoint_ts,
        "network": BITTENSOR_NETWORK,
        "anchor_ts": datetime.utcnow().isoformat()
    }
    
    # Add sequence range if provided
    if seq_start is not None:
        payload["seq_start"] = seq_start
    if seq_end is not None:
        payload["seq_end"] = seq_end
    
    # Compute payload hash for integrity
    payload_json = json.dumps(payload, sort_keys=True, default=str)  # Handle datetime objects
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    
    log_entry = {
        "event_type": "ANCHOR_ROOT",
        "actor_hotkey": "system",
        "nonce": str(uuid4()),
        "ts": datetime.utcnow().isoformat(),
        "payload_hash": payload_hash,
        "build_id": BUILD_ID,
        "signature": "system",
        "payload": payload
    }
    
    # Log through TEE (authoritative, hardware-protected)
    from gateway.utils.logger import log_event
    result = await log_event(log_entry)
    tee_sequence = result.get("sequence")
    
    print(f"   ✅ ANCHOR_ROOT event logged to transparency log (seq={tee_sequence})")


# Optional: Manual anchor function for testing/debugging
async def manual_anchor():
    """
    Manually trigger an anchor operation (for testing/debugging).
    
    This function can be called directly to anchor the latest checkpoint
    without waiting for the 24-hour interval.
    
    Usage:
        import asyncio
        from gateway.tasks.anchor import manual_anchor
        
        asyncio.run(manual_anchor())
    """
    print("🔗 Manual anchor triggered...")
    print()
    
    try:
        # Get latest checkpoint root
        result = supabase.table("merkle_checkpoints") \
            .select("*") \
            .order("id", desc=True) \
            .limit(1) \
            .execute()
        
        if not result.data:
            print("⚠️  No checkpoint roots found")
            return
        
        latest_checkpoint = result.data[0]
        merkle_root = latest_checkpoint["merkle_root"]
        checkpoint_ts = latest_checkpoint["checkpoint_ts"]
        seq_start = latest_checkpoint.get("seq_start", 0)
        seq_end = latest_checkpoint.get("seq_end", 0)
        
        print(f"📍 Anchoring Merkle root...")
        print(f"   Root: {merkle_root}")
        print(f"   Checkpoint Timestamp: {checkpoint_ts}")
        print(f"   Sequence Range: {seq_start} - {seq_end}")
        print()
        
        await log_anchor_event(
            merkle_root=merkle_root,
            checkpoint_ts=checkpoint_ts,
            seq_start=seq_start,
            seq_end=seq_end
        )
        
        print(f"   ✅ Manual anchor complete")
        print()
        
    except Exception as e:
        print(f"❌ Manual anchor failed: {e}")
        print()


if __name__ == "__main__":
    # For testing - run the anchor task once
    asyncio.run(manual_anchor())

