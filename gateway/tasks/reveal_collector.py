"""
Reveal Collector Background Task

Monitors epochs after they close and tracks reveal progress.

This task:
- Checks for unrevealed validations from closed epochs
- Logs validators who haven't revealed (for monitoring)
- Can be extended to send notifications or penalties

Runs every 2 minutes to check reveal progress.
"""

import asyncio
from datetime import datetime, timedelta
from gateway.utils.epoch import get_current_epoch_id_async, is_epoch_closed_async
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def reveal_collector_task():
    """
    Background task to collect reveals after epoch closes.
    
    Runs every 2 minutes and checks for unrevealed evidence
    from closed epochs.
    
    Flow:
    1. Get current epoch
    2. Check previous epoch (should be closed)
    3. Query unrevealed evidence
    4. Log validators who haven't revealed
    5. (Optional) Send notifications or apply penalties
    
    This helps monitor validator participation and detect
    non-compliant validators.
    """
    
    print("🔓 Reveal collector task started")
    
    checked_epochs = set()  # Track which epochs we've checked
    
    while True:
        try:
            current_epoch = await get_current_epoch_id_async()
            
            # Check previous epoch (should be closed by now)
            previous_epoch = current_epoch - 1
            
            # Skip if we've already checked this epoch
            if previous_epoch in checked_epochs:
                await asyncio.sleep(120)  # Sleep 2 minutes
                continue
            
            # Check if previous epoch is closed
            if not await is_epoch_closed_async(previous_epoch):
                # Previous epoch not closed yet, wait
                await asyncio.sleep(120)
                continue
            
            print(f"\n{'='*80}")
            print(f"🔍 CHECKING REVEALS FOR EPOCH {previous_epoch}")
            print(f"{'='*80}")
            
            # Query unrevealed evidence from previous epoch
            result = supabase.table("validation_evidence_private") \
                .select("evidence_id, validator_hotkey, lead_id") \
                .eq("epoch_id", previous_epoch) \
                .is_("decision", "null") \
                .execute()
            
            unrevealed = result.data
            
            if unrevealed:
                # Get unique validators who haven't revealed
                unrevealed_validators = list(set([e["validator_hotkey"] for e in unrevealed]))
                
                print(f"⚠️  {len(unrevealed)} validation(s) not revealed")
                print(f"⚠️  {len(unrevealed_validators)} validator(s) have not revealed")
                
                for validator in unrevealed_validators:
                    validator_unrevealed = [e for e in unrevealed if e["validator_hotkey"] == validator]
                    print(f"   • {validator[:20]}... ({len(validator_unrevealed)} unrevealed)")
                
                # In production, you could:
                # - Send notifications to validators
                # - Apply penalties for non-revelation
                # - Track reputation scores
                # - Exclude from future epochs
                
            else:
                print(f"✅ All validators revealed for epoch {previous_epoch}")
            
            # Mark this epoch as checked
            checked_epochs.add(previous_epoch)
            print(f"{'='*80}\n")
            
            # Clean up old tracking set to prevent memory growth
            if len(checked_epochs) > 100:
                # Keep only recent 50 epochs
                recent = sorted(list(checked_epochs))[-50:]
                checked_epochs = set(recent)
            
            # Sleep 2 minutes before next check
            await asyncio.sleep(120)
        
        except Exception as e:
            print(f"❌ Reveal collector error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(120)


async def get_reveal_statistics(epoch_id: int) -> dict:
    """
    Get comprehensive reveal statistics for an epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        {
            "epoch_id": int,
            "total_commits": int,
            "total_reveals": int,
            "reveal_percentage": float,
            "unrevealed_count": int,
            "unrevealed_validators": List[str]
        }
    """
    try:
        # Query all evidence for this epoch
        result = supabase.table("validation_evidence_private") \
            .select("evidence_id, validator_hotkey, decision, revealed_ts") \
            .eq("epoch_id", epoch_id) \
            .execute()
        
        evidences = result.data
        total_commits = len(evidences)
        
        # Count reveals
        revealed = [e for e in evidences if e["decision"] is not None]
        total_reveals = len(revealed)
        
        # Calculate percentage
        reveal_percentage = (total_reveals / total_commits * 100) if total_commits > 0 else 0
        
        # Get unrevealed validators
        unrevealed = [e for e in evidences if e["decision"] is None]
        unrevealed_validators = list(set([e["validator_hotkey"] for e in unrevealed]))
        
        return {
            "epoch_id": epoch_id,
            "total_commits": total_commits,
            "total_reveals": total_reveals,
            "reveal_percentage": round(reveal_percentage, 2),
            "unrevealed_count": len(unrevealed),
            "unrevealed_validators": unrevealed_validators
        }
    
    except Exception as e:
        print(f"❌ Error getting reveal statistics: {e}")
        return {
            "epoch_id": epoch_id,
            "total_commits": 0,
            "total_reveals": 0,
            "reveal_percentage": 0.0,
            "unrevealed_count": 0,
            "unrevealed_validators": [],
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Run reveal collector task as standalone module.
    
    Usage: python -m gateway.tasks.reveal_collector
    """
    import asyncio
    print("🚀 Starting Reveal Collector Task...")
    asyncio.run(reveal_collector_task())
