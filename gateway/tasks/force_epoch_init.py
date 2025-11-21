"""
Force Epoch Initialization Script
==================================

Manually triggers EPOCH_INITIALIZATION for the current epoch.
Use this if the gateway was restarted mid-epoch and missed the epoch transition.

Usage:
    cd ~/gateway
    python3 -m gateway.tasks.force_epoch_init <epoch_id>
    
Example:
    python3 -m gateway.tasks.force_epoch_init 19210
"""

import asyncio
import sys
from datetime import datetime

async def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python3 -m gateway.tasks.force_epoch_init <epoch_id>")
        print("   Example: python3 -m gateway.tasks.force_epoch_init 19210")
        sys.exit(1)
    
    epoch_id = int(sys.argv[1])
    
    print(f"\n{'='*80}")
    print(f"🔧 FORCE INITIALIZING EPOCH {epoch_id}")
    print(f"{'='*80}\n")
    
    # Import epoch lifecycle functions
    from gateway.tasks.epoch_lifecycle import compute_and_log_epoch_initialization
    from gateway.utils.epoch import get_epoch_start_time_async, get_epoch_end_time_async, get_epoch_close_time_async
    from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    from supabase import create_client
    
    # Check if epoch is already initialized
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    
    print(f"🔍 Checking if epoch {epoch_id} is already initialized...")
    
    # Query transparency log for existing EPOCH_INITIALIZATION
    result = await asyncio.to_thread(
        lambda: supabase.table("transparency_log")
            .select("id")
            .eq("event_type", "EPOCH_INITIALIZATION")
            .eq("payload->>epoch_id", str(epoch_id))
            .limit(1)
            .execute()
    )
    
    if result.data:
        print(f"⚠️  Epoch {epoch_id} is already initialized!")
        print(f"   Found EPOCH_INITIALIZATION event: {result.data[0]['id']}")
        print(f"   This epoch is good - the problem might be elsewhere.\n")
        
        # Show the existing initialization
        full_result = await asyncio.to_thread(
            lambda: supabase.table("transparency_log")
                .select("*")
                .eq("event_type", "EPOCH_INITIALIZATION")
                .eq("payload->>epoch_id", str(epoch_id))
                .single()
                .execute()
        )
        
        if full_result.data:
            payload = full_result.data['payload']
            print(f"   📊 Existing initialization:")
            print(f"      Queue root: {payload.get('queue_merkle_root', 'N/A')[:16]}...")
            print(f"      Pending leads: {payload.get('pending_lead_count', 0)}")
            print(f"      Assigned leads: {len(payload.get('assigned_lead_ids', []))}")
            print(f"      Validators: {payload.get('validator_count', 0)}")
        
        sys.exit(0)
    
    print(f"✅ No existing initialization found - proceeding...\n")
    
    # Calculate epoch boundaries
    print(f"📅 Calculating epoch {epoch_id} boundaries...")
    epoch_start = await get_epoch_start_time_async(epoch_id)
    epoch_end = await get_epoch_end_time_async(epoch_id)
    epoch_close = await get_epoch_close_time_async(epoch_id)
    
    print(f"   Start: {epoch_start.isoformat()}")
    print(f"   End (validation): {epoch_end.isoformat()}")
    print(f"   Close: {epoch_close.isoformat()}")
    
    now = datetime.utcnow()
    if now < epoch_start:
        print(f"\n⚠️  WARNING: Epoch {epoch_id} hasn't started yet!")
        print(f"   Current time: {now.isoformat()}")
        print(f"   Epoch starts: {epoch_start.isoformat()}")
        response = input(f"   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            sys.exit(0)
    
    # Trigger initialization
    print(f"\n🚀 Triggering EPOCH_INITIALIZATION...")
    await compute_and_log_epoch_initialization(epoch_id, epoch_start, epoch_end, epoch_close)
    
    print(f"\n{'='*80}")
    print(f"✅ EPOCH {epoch_id} INITIALIZED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    print(f"🎯 Next steps:")
    print(f"   1. Validators can now fetch leads for epoch {epoch_id}")
    print(f"   2. Check gateway logs for any errors")
    print(f"   3. Monitor validator logs to see if they receive leads\n")

if __name__ == "__main__":
    asyncio.run(main())

