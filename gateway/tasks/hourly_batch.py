"""
Arweave Batching Task
=====================

This background task runs continuously to batch TEE events to Arweave every 3 hours.

Flow:
1. Wait 3 hours (or until buffer is full)
2. Request checkpoint from TEE via vsock
3. Compress events (gzip)
4. Upload checkpoint to Arweave
5. Wait for Arweave confirmation
6. Tell TEE to clear buffer
7. Repeat

Cost: ~$0.10/month for 3-hour batching (vs $300+/month for per-event writes)
"""

import asyncio
import gzip
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

# Import gateway utilities
from gateway.utils.tee_client import tee_client
from gateway.utils.arweave_client import upload_checkpoint, get_wallet_balance
from gateway.utils.logger import log_event
from gateway.config import BUILD_ID


# Configuration
BATCH_INTERVAL = 10800  # 3 hours in seconds
EMERGENCY_BATCH_THRESHOLD = 8000  # Trigger early batch if buffer hits this size
MAX_UPLOAD_RETRIES = 3  # Retry failed uploads


async def hourly_batch_task():
    """
    Main hourly batching task.
    
    Runs continuously, batching TEE events to Arweave every hour.
    Implements:
    - Regular hourly batching
    - Emergency batching if buffer fills
    - Retry logic with exponential backoff
    - Comprehensive logging
    """
    print("="*80)
    print("🚀 STARTING HOURLY ARWEAVE BATCH TASK")
    print("="*80)
    print(f"   Batch interval: {BATCH_INTERVAL}s ({BATCH_INTERVAL/3600:.1f} hours)")
    print(f"   Emergency threshold: {EMERGENCY_BATCH_THRESHOLD} events")
    print("="*80)
    print()
    
    # Check Arweave wallet balance
    try:
        balance = await get_wallet_balance()
        print(f"💰 Arweave wallet balance: {balance:.6f} AR")
        
        if balance < 0.01:
            print("⚠️  WARNING: Low Arweave balance!")
            print("   Please fund wallet to ensure continuous operation.")
            print("   Estimated cost: ~$0.30/month for hourly batching")
        print()
    except Exception as e:
        print(f"⚠️  Could not check wallet balance: {e}")
        print("   Continuing anyway...\n")
    
    # Calculate time until next hour boundary (top of the hour)
    now = datetime.utcnow()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    wait_seconds = int((next_hour - now).total_seconds())
    
    print(f"⏳ Waiting {wait_seconds/60:.1f} minutes until top of next hour...")
    print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Next batch: {next_hour.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Events will accumulate in TEE buffer during this time.\n")
    
    next_batch = next_hour
    
    # Add countdown progress every 5 minutes WITH EMERGENCY CHECK
    # This is critical - the old code only checked every 30 min and had NO emergency check
    remaining_time = wait_seconds
    check_interval = 300  # 5 minutes - same as main loop for consistency
    progress_interval = 1800  # 30 minutes for verbose logging
    last_progress_print = 0
    elapsed = 0
    
    initial_emergency_triggered = False
    while remaining_time > 0:
        wait_time = min(check_interval, remaining_time)
        await asyncio.sleep(wait_time)
        remaining_time -= wait_time
        elapsed += wait_time
        
        minutes_left = remaining_time / 60
        
        # Check buffer stats EVERY 5 minutes for emergency detection
        try:
            stats = await tee_client.get_buffer_stats()
            buffer_size = stats.get("size", 0)
            
            # EMERGENCY CHECK - runs every 5 minutes during INITIAL wait too!
            if buffer_size >= EMERGENCY_BATCH_THRESHOLD:
                print(f"\n🚨 EMERGENCY BATCH TRIGGERED DURING STARTUP ALIGNMENT!")
                print(f"   Buffer size: {buffer_size} events (threshold: {EMERGENCY_BATCH_THRESHOLD})")
                print(f"   Skipping alignment wait, starting batch immediately...")
                initial_emergency_triggered = True
                break  # Exit wait loop, start batch immediately
            
            # Print countdown progress every 30 minutes (less verbose)
            if elapsed - last_progress_print >= progress_interval:
                last_progress_print = elapsed
                print(f"⏰ Arweave Upload Countdown: {minutes_left:.0f} minutes remaining")
                print(f"   📊 {buffer_size} event(s) accumulated in TEE buffer")
                print(f"   Next upload: {next_batch.isoformat()}\n")
                
        except Exception as e:
            # TEE connection failed - LOG IT LOUDLY, don't silently skip
            print(f"⚠️  TEE connection failed during initial wait: {e}")
            print(f"   Cannot check for emergency batch condition!")
            if elapsed - last_progress_print >= progress_interval:
                last_progress_print = elapsed
                print(f"⏰ Arweave Upload Countdown: {minutes_left:.0f} minutes remaining\n")
    
    batch_count = 0
    
    # Main batching loop
    while True:
        try:
            batch_count += 1
            print("\n" + "="*80)
            print(f"📦 BATCH #{batch_count} - {datetime.utcnow().isoformat()}")
            print("="*80)
            
            # Step 1: Check buffer stats
            try:
                stats = await tee_client.get_buffer_stats()
                buffer_size = stats.get("size", 0)
                
                print(f"\n📊 TEE Buffer Stats:")
                print(f"   Size: {buffer_size} events")
                print(f"   Age: {stats.get('age_seconds', 0):.1f}s")
                if stats.get("sequence_range"):
                    seq_range = stats["sequence_range"]
                    print(f"   Sequence: {seq_range.get('first')} → {seq_range.get('last')}")
            except Exception as e:
                print(f"⚠️  Could not get buffer stats: {e}")
                buffer_size = 0
            
            # Step 2: Request checkpoint from TEE
            print(f"\n🔄 Requesting checkpoint from TEE...")
            checkpoint_data = await tee_client.build_checkpoint()
            
            # Handle empty buffer - still upload for continuous audit trail
            if checkpoint_data.get("status") == "empty":
                print("ℹ️  No events in TEE buffer")
                print("   Uploading empty checkpoint to maintain continuous audit trail...")
                
                # Create empty checkpoint
                checkpoint_data = {
                    "header": {
                        "checkpoint_number": batch_count,
                        "event_count": 0,
                        "merkle_root": "0" * 64,  # Empty tree
                        "time_range": {
                            "start": datetime.utcnow().isoformat(),
                            "end": datetime.utcnow().isoformat()
                        }
                    },
                    "signature": "empty_checkpoint",
                    "events": [],
                    "tree_levels": []
                }
                
                # Continue with empty upload (don't skip)
            
            # Extract checkpoint components
            header = checkpoint_data["header"]
            signature = checkpoint_data["signature"]
            events = checkpoint_data["events"]
            tree_levels = checkpoint_data["tree_levels"]
            
            print(f"✅ Checkpoint received from TEE:")
            print(f"   Checkpoint #{header['checkpoint_number']}")
            print(f"   Events: {header['event_count']}")
            print(f"   Merkle root: {header['merkle_root'][:16]}...")
            print(f"   Time range: {header['time_range']['start']} → {header['time_range']['end']}")
            
            # Step 3: Compress events
            print(f"\n📦 Compressing events...")
            events_json = json.dumps(events, default=str)  # Handle datetime objects
            events_bytes = events_json.encode('utf-8')
            compressed_events = gzip.compress(events_bytes, compresslevel=9)
            
            compression_ratio = len(compressed_events) / len(events_bytes)
            print(f"✅ Compression complete:")
            print(f"   Original: {len(events_bytes):,} bytes ({len(events_bytes)/1024:.2f} KB)")
            print(f"   Compressed: {len(compressed_events):,} bytes ({len(compressed_events)/1024:.2f} KB)")
            print(f"   Ratio: {compression_ratio:.1%} (saved {(1-compression_ratio)*100:.1f}%)")
            
            # Step 4: Upload to Arweave (with retries)
            tx_id = None
            upload_success = False
            
            for upload_attempt in range(1, MAX_UPLOAD_RETRIES + 1):
                try:
                    print(f"\n📤 Uploading to Arweave (attempt {upload_attempt}/{MAX_UPLOAD_RETRIES})...")
                    
                    tx_id = await upload_checkpoint(
                        header=header,
                        signature=signature,
                        events=compressed_events,
                        tree_levels=tree_levels
                    )
                    
                    upload_success = True
                    break
                
                except Exception as e:
                    print(f"❌ Upload attempt {upload_attempt} failed: {e}")
                    
                    if upload_attempt < MAX_UPLOAD_RETRIES:
                        retry_delay = 2 ** upload_attempt  # Exponential backoff: 2s, 4s, 8s
                        print(f"   Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"❌ All upload attempts failed!")
                        print(f"   Events remain safe in TEE buffer.")
                        print(f"   Will retry on next hourly batch.")
            
            if not upload_success:
                # Skip to next batch (events stay in buffer)
                print(f"\n⏭️  Waiting {BATCH_INTERVAL/60:.0f} minutes for next batch...")
                await asyncio.sleep(BATCH_INTERVAL)
                continue
            
            # Step 5: Verify upload succeeded
            print(f"\n🔍 Verifying upload to Arweave...")
            try:
                import requests
                # Check if content is immediately available (usually is)
                verify_response = requests.get(f"https://arweave.net/{tx_id}", timeout=10)
                if verify_response.status_code == 200:
                    print(f"✅ Upload verified: Content is available ({len(verify_response.content)} bytes)")
                elif verify_response.status_code == 202:
                    print(f"⏳ Upload accepted: Transaction pending confirmation")
                else:
                    print(f"⚠️  Upload status unclear: HTTP {verify_response.status_code}")
            except Exception as e:
                print(f"⚠️  Verification check failed (upload likely succeeded): {e}")
            
            print(f"\n✅ Checkpoint uploaded to Arweave")
            print(f"   TX ID: {tx_id}")
            if header['event_count'] == 0:
                print(f"   Note: Empty checkpoint (maintains continuous audit trail)")
            else:
                print(f"   Events: {header['event_count']}")
            print(f"   Note: Full confirmation takes 2-20 minutes to propagate")
            print(f"   Content URL: https://arweave.net/{tx_id}")
            print(f"   ViewBlock: https://viewblock.io/arweave/tx/{tx_id}")
            
            # Step 6: Log checkpoint to transparency log
            print(f"\n📝 Logging checkpoint to transparency log...")
            try:
                import uuid
                
                # Compute payload hash for transparency
                payload_data = {
                    "arweave_tx_id": tx_id,
                    "checkpoint_number": header['checkpoint_number'],
                    "event_count": header['event_count'],
                    "merkle_root": header['merkle_root'],
                    "time_range": header['time_range'],
                    "compressed_size_bytes": len(compressed_events),
                    "viewblock_url": f"https://viewblock.io/arweave/tx/{tx_id}"
                }
                
                # json and hashlib already imported at top of file
                import hashlib
                payload_json = json.dumps(payload_data, sort_keys=True, default=str)  # Handle datetime objects
                payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
                
                checkpoint_log = {
                    "event_type": "ARWEAVE_CHECKPOINT",
                    "actor_hotkey": "system",
                    "ts": datetime.utcnow().isoformat() + "Z",  # Required timestamp field
                    "nonce": str(uuid.uuid4()),  # Required field
                    "payload_hash": payload_hash,  # Required field
                    "signature": "system",  # System-generated events use "system" as signature
                    "build_id": BUILD_ID,  # Required field
                    "payload": payload_data
                }
                
                result = await log_event(checkpoint_log)
                tee_sequence = result.get("sequence")
                print(f"✅ Checkpoint logged (seq={tee_sequence})")
                
                # Also update transparency_log table with arweave_tx_id
                # This allows miners to query for TX IDs easily
                from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
                import httpx
                
                async with httpx.AsyncClient() as client:
                    # Update the checkpoint event we just logged
                    update_response = await client.patch(
                        f"{SUPABASE_URL}/rest/v1/transparency_log",
                        params={"tee_sequence": f"eq.{tee_sequence}"},
                        headers={
                            "apikey": SUPABASE_SERVICE_ROLE_KEY,
                            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                            "Content-Type": "application/json",
                            "Prefer": "return=minimal"
                        },
                        json={"arweave_tx_id": tx_id}
                    )
                    
                    if update_response.status_code in [200, 204]:
                        print(f"✅ Arweave TX ID saved to database")
                    else:
                        print(f"⚠️  Failed to save TX ID to database: {update_response.status_code}")
                        
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint: {e}")
                print(f"   (Upload succeeded, but logging failed - TX ID: {tx_id})")
            
            # Step 7: Clear TEE buffer
            print(f"\n🧹 Clearing TEE buffer...")
            clear_result = await tee_client.clear_buffer()
            print(f"✅ Buffer cleared: {clear_result.get('cleared_count', 0)} events removed")
            print(f"   Next checkpoint starts: {clear_result.get('next_checkpoint_at', 'N/A')}")
            
            # Step 8: Log success
            print(f"\n" + "="*80)
            print(f"✅ BATCH #{batch_count} COMPLETE")
            print("="*80)
            print(f"   Checkpoint: #{header['checkpoint_number']}")
            print(f"   Events batched: {header['event_count']}")
            print(f"   Arweave TX: {tx_id}")
            print(f"   View: https://viewblock.io/arweave/tx/{tx_id}")
            print(f"   Cost: ~${len(compressed_events) * 0.000002:.4f} (~$0.002 per KB)")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ BATCH #{batch_count} FAILED: {e}")
            print(f"   Events remain safe in TEE buffer.")
            import traceback
            traceback.print_exc()
        
        # Wait for BATCH_INTERVAL before next batch
        wait_seconds = BATCH_INTERVAL
        next_batch_time = datetime.utcnow() + timedelta(seconds=BATCH_INTERVAL)
        
        print(f"\n⏭️  Next batch: {next_batch_time.strftime('%Y-%m-%d %H:%M:%S')} UTC ({wait_seconds/60:.1f} minutes)")
        
        # Implement emergency batch check during wait
        # Check buffer size every 5 minutes during wait
        # Print countdown and check emergency threshold every 30 minutes
        check_interval = 300  # 5 minutes (check buffer size)
        progress_interval = 1800  # 30 minutes (print countdown + emergency check)
        checks_per_interval = wait_seconds // check_interval
        
        last_progress_print = 0
        
        emergency_triggered = False
        for check_num in range(checks_per_interval):
            await asyncio.sleep(check_interval)
            
            elapsed = (check_num + 1) * check_interval
            remaining = wait_seconds - elapsed
            minutes_left = remaining / 60
            
            # Check buffer stats EVERY 5 minutes for emergency detection
            try:
                stats = await tee_client.get_buffer_stats()
                current_size = stats.get("size", 0)
                
                # EMERGENCY CHECK - runs every 5 minutes (not just every 30 min)
                if current_size >= EMERGENCY_BATCH_THRESHOLD:
                    print(f"\n🚨 EMERGENCY BATCH TRIGGERED!")
                    print(f"   Buffer size: {current_size} events (threshold: {EMERGENCY_BATCH_THRESHOLD})")
                    print(f"   Triggering early batch to prevent overflow...")
                    emergency_triggered = True
                    break  # Exit wait loop, start next batch immediately
                
                # Print countdown progress every 30 minutes (less verbose)
                if elapsed - last_progress_print >= progress_interval or check_num == checks_per_interval - 1:
                    last_progress_print = elapsed
                    
                    if minutes_left > 0:
                        print(f"⏰ Arweave Upload Countdown: {minutes_left:.0f} minutes remaining")
                        print(f"   📊 {current_size} event(s) accumulated in TEE buffer")
                        print(f"   Next upload: {next_batch_time.isoformat()}\n")
                    
            except Exception as e:
                # TEE connection failed - print warning but continue
                if elapsed - last_progress_print >= progress_interval or check_num == checks_per_interval - 1:
                    last_progress_print = elapsed
                    if minutes_left > 0:
                        print(f"⏰ Arweave Upload Countdown: {minutes_left:.0f} minutes remaining")
                        print(f"   📊 ? event(s) accumulated in TEE buffer (TEE unavailable: {e})")
                        print(f"   Next upload: {next_batch_time.isoformat()}\n")


async def start_hourly_batch_task():
    """
    Wrapper to start hourly batch task with error recovery.
    
    If the task crashes, it will restart after a delay.
    """
    restart_delay = 60  # seconds
    
    while True:
        try:
            await hourly_batch_task()
        except Exception as e:
            print(f"\n❌ Hourly batch task crashed: {e}")
            print(f"   Restarting in {restart_delay}s...")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(restart_delay)

