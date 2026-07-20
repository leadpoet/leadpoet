"""
Force Epoch Initialization Script
==================================

Manually triggers EPOCH_INITIALIZATION for the current official SN71 epoch.

Usage:
    cd ~/gateway
    python3 -m gateway.tasks.force_epoch_init
    
Example:
    python3 -m gateway.tasks.force_epoch_init
"""

import asyncio

async def main():
    from gateway.tasks.epoch_lifecycle import (
        compute_and_log_epoch_initialization,
        get_durable_epoch_event,
    )
    from gateway.utils.epoch import (
        get_current_epoch_context_async,
        get_current_epoch_times,
    )

    epoch_snapshot, epoch_id = await get_current_epoch_context_async(
        finalized=True
    )
    
    print(f"\n{'='*80}")
    print(f"🔧 FORCE INITIALIZING EPOCH {epoch_id}")
    print(f"{'='*80}\n")
    
    print(f"🔍 Checking if epoch {epoch_id} is already initialized...")
    existing = await get_durable_epoch_event(
        "EPOCH_INITIALIZATION",
        epoch_id,
    )
    if existing is not None:
        print(f"⚠️  Epoch {epoch_id} is already initialized!")
        print(f"   Found EPOCH_INITIALIZATION event: {existing.get('id')}")
        return
    
    print(f"✅ No existing initialization found - proceeding...\n")
    
    print(f"📅 Reading official epoch {epoch_id} boundaries...")
    epoch_start, epoch_end, epoch_close = get_current_epoch_times(
        epoch_snapshot
    )
    
    print(f"   Start: {epoch_start.isoformat()}")
    print(f"   End (validation): {epoch_end.isoformat()}")
    print(f"   Close: {epoch_close.isoformat()}")
    
    print(f"\n🚀 Triggering EPOCH_INITIALIZATION...")
    await compute_and_log_epoch_initialization(
        epoch_id,
        epoch_start,
        epoch_end,
        epoch_close,
        epoch_snapshot=epoch_snapshot,
    )
    
    print(f"\n{'='*80}")
    print(f"✅ EPOCH {epoch_id} INITIALIZED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    print(f"🎯 Next steps:")
    print(f"   1. Validators can now fetch leads for epoch {epoch_id}")
    print(f"   2. Check gateway logs for any errors")
    print(f"   3. Monitor validator logs to see if they receive leads\n")

if __name__ == "__main__":
    asyncio.run(main())
