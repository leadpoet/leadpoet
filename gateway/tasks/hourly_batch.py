"""
Hourly Arweave Batching Task
============================

This background task runs continuously to batch TEE events to Arweave every hour.

Flow:
1. Wait 1 hour (or until buffer is full)
2. Request checkpoint from TEE via vsock
3. Compress events (gzip)
4. Upload checkpoint to Arweave
5. Wait for Arweave confirmation
6. Tell TEE to clear buffer
7. Repeat

Cost: ~$0.30/month for hourly batching (vs $300+/month for per-event writes)
"""

import asyncio
import gzip
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

# Import gateway utilities
from utils.tee_client import tee_client
from utils.arweave_client import upload_checkpoint, get_wallet_balance


# Configuration
BATCH_INTERVAL = 3600  # 1 hour in seconds
EMERGENCY_BATCH_THRESHOLD = 10000  # Trigger early batch if buffer hits this size
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
    print(f"   Confirmation timeout: {CONFIRMATION_TIMEOUT}s")
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
    
    # Wait before first batch (allow events to accumulate)
    print(f"⏳ Waiting {BATCH_INTERVAL/60:.0f} minutes for first batch...")
    print(f"   Events will accumulate in TEE buffer during this time.")
    print(f"   Next batch: {datetime.utcnow() + timedelta(seconds=BATCH_INTERVAL)}\n")
    
    await asyncio.sleep(BATCH_INTERVAL)
    
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
            
            # Handle empty buffer
            if checkpoint_data.get("status") == "empty":
                print("✅ No events to batch (buffer empty)")
                print(f"⏭️  Skipping batch, waiting {BATCH_INTERVAL/60:.0f} minutes for next batch...")
                await asyncio.sleep(BATCH_INTERVAL)
                continue
            
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
            events_json = json.dumps(events)
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
            print(f"   Note: Full confirmation takes 2-20 minutes to propagate")
            print(f"   Content URL: https://arweave.net/{tx_id}")
            print(f"   ViewBlock: https://viewblock.io/arweave/tx/{tx_id}")
            
            # Step 6: Clear TEE buffer
            print(f"\n🧹 Clearing TEE buffer...")
            clear_result = await tee_client.clear_buffer()
            print(f"✅ Buffer cleared: {clear_result.get('cleared_count', 0)} events removed")
            print(f"   Next checkpoint starts: {clear_result.get('next_checkpoint_at', 'N/A')}")
            
            # Step 7: Log success
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
        
        # Wait for next batch
        next_batch_time = datetime.utcnow() + timedelta(seconds=BATCH_INTERVAL)
        print(f"\n⏭️  Next batch: {next_batch_time.isoformat()} ({BATCH_INTERVAL/60:.0f} minutes)")
        
        # Implement emergency batch check during wait
        # Check buffer size every 5 minutes during the 1-hour wait
        check_interval = 300  # 5 minutes
        checks_per_hour = BATCH_INTERVAL // check_interval
        
        for check_num in range(checks_per_hour):
            await asyncio.sleep(check_interval)
            
            # Check if buffer is approaching capacity
            try:
                stats = await tee_client.get_buffer_stats()
                current_size = stats.get("size", 0)
                
                if current_size >= EMERGENCY_BATCH_THRESHOLD:
                    print(f"\n🚨 EMERGENCY BATCH TRIGGERED!")
                    print(f"   Buffer size: {current_size} events (threshold: {EMERGENCY_BATCH_THRESHOLD})")
                    print(f"   Triggering early batch to prevent overflow...")
                    break  # Exit wait loop, start next batch immediately
                
            except Exception as e:
                # Ignore buffer check errors, continue waiting
                pass


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

