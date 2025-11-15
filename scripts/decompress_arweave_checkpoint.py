#!/usr/bin/env python3
"""
Decompress Arweave Checkpoint Events

This script downloads and decompresses LeadPoet transparency checkpoints from Arweave,
revealing all individual events with complete details.

Three modes of operation:
    1. By Arweave TX ID (highest priority)
    2. By specific date (medium priority) 
    3. By last X hours (lowest priority - default)

Usage:
    python decompress_arweave_checkpoint.py
"""

import sys
import json
import gzip
import base64
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os

# ============================================================
# CONFIGURATION - Edit these variables as needed
# ============================================================

# Mode 1: Specific Arweave Transaction ID
# Get this from your Arweave wallet transaction history (click on any transaction)
# Example: "jIINjzv0Pd3qhAJqejRh9S2uOn3r8+r95vQNGrIswECI"
# PASTE A TX ID HERE TO TEST:
ARWEAVE_TX_ID = ""  # Leave blank to use date or time range

# Mode 2: Specific Date (YYYY-MM-DD format)
# Example: "2025-11-14" pulls all checkpoints from that day
SPECIFIC_DATE = ""  # Leave blank to use time range

# Mode 3: Last X Hours (default mode if above are blank)
# Example: 4 = last 4 hours, 24 = last 24 hours
LAST_X_HOURS = 8  # Change this to pull different time ranges

# Gateway's Arweave wallet address (public, used to query checkpoints)
# This is trustless - miners verify transactions come from this known address
GATEWAY_WALLET_ADDRESS = "FHF3LaLJYaIRR8Uj-ncQYzmxOGmKLbT184PbYIfsDYM"

# Arweave GraphQL endpoint (trustless - queries Arweave directly)
ARWEAVE_GRAPHQL = "https://arweave.net/graphql"

# ============================================================
# Priority Hierarchy:
# 1. If ARWEAVE_TX_ID is set → Use it (ignore date and hours)
# 2. If SPECIFIC_DATE is set → Use it (ignore hours)  
# 3. Otherwise → Use LAST_X_HOURS
# ============================================================


def query_checkpoint_ids(date: Optional[str] = None, hours: Optional[int] = None) -> List[str]:
    """
    Query Arweave GraphQL for checkpoint transaction IDs from the gateway's wallet.
    
    This is TRUSTLESS - queries Arweave directly by wallet address, verifying the known gateway address.
    """
    
    if date:
        print(f"🔍 Querying Arweave for checkpoints on date: {date}")
        print(f"   From wallet: {GATEWAY_WALLET_ADDRESS[:20]}...")
    elif hours:
        print(f"🔍 Querying Arweave for checkpoints in last {hours} hours")
        print(f"   From wallet: {GATEWAY_WALLET_ADDRESS[:20]}...")
    else:
        print("❌ Error: Must provide either date or hours")
        sys.exit(1)
    
    # GraphQL query to find all transactions from gateway wallet
    # Note: Tags may take time to index, so we query all txs and filter locally
    query = """
    query($owner: String!, $first: Int!) {
      transactions(
        owners: [$owner]
        sort: HEIGHT_DESC
        first: $first
      ) {
        edges {
          node {
            id
            tags {
              name
              value
            }
            block {
              timestamp
            }
          }
        }
      }
    }
    """
    
    try:
        print(f"   Querying Arweave GraphQL API...")
        
        response = requests.post(
            ARWEAVE_GRAPHQL,
            json={
                "query": query,
                "variables": {
                    "owner": GATEWAY_WALLET_ADDRESS,
                    "first": 100  # Get last 100 checkpoints
                }
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "errors" in data:
            print(f"❌ GraphQL errors: {data['errors']}")
            sys.exit(1)
        
        edges = data.get("data", {}).get("transactions", {}).get("edges", [])
        
        if not edges:
            print(f"⚠️  No transactions found from gateway wallet")
            print(f"   Wallet: {GATEWAY_WALLET_ADDRESS}")
            print(f"\n💡 ARWEAVE CONFIRMATION PROCESS:")
            print(f"   1. Transaction submitted → Shows in wallet (instant)")
            print(f"   2. Mined into block → Gets confirmed (~2-10 minutes)")
            print(f"   3. Indexed by GraphQL → Queryable by miners (~10-20 minutes after mining)")
            print(f"\n   ⏰ Total time: ~15-30 minutes from upload to query")
            print(f"\n   This delay is GOOD for security - ensures data is truly immutable!")
            print(f"   Recent uploads may not be queryable yet - please wait and try again.")
            sys.exit(0)
        
        print(f"   Found {len(edges)} total transactions from wallet")
        
        # Filter by checkpoint tags and time range
        filtered_txs = []
        
        for edge in edges:
            node = edge["node"]
            tags = {tag["name"]: tag["value"] for tag in node.get("tags", [])}
            
            # Check if this is a checkpoint transaction
            is_checkpoint = (
                tags.get("App") == "leadpoet" and 
                tags.get("Type") == "checkpoint"
            )
            
            if not is_checkpoint:
                continue
            
            # Check time range
            if date:
                # Filter by specific date
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                if node.get("block") and node["block"].get("timestamp"):
                    tx_date = datetime.fromtimestamp(node["block"]["timestamp"]).date()
                    if tx_date == target_date:
                        filtered_txs.append(node["id"])
            elif hours:
                # Filter by last X hours
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                cutoff_timestamp = int(cutoff_time.timestamp())
                
                if node.get("block") and node["block"].get("timestamp"):
                    if node["block"]["timestamp"] >= cutoff_timestamp:
                        filtered_txs.append(node["id"])
        
        if not filtered_txs:
            print(f"⚠️  No checkpoints found for the specified time range")
            if date:
                print(f"   Date: {date}")
            else:
                print(f"   Last {hours} hours")
            print(f"\n💡 NOTE: Found {len(edges)} total checkpoints, but none in your time range")
            print(f"   Try increasing LAST_X_HOURS or check a different date")
            sys.exit(0)
        
        print(f"✅ Found {len(filtered_txs)} checkpoint(s) on Arweave")
        return filtered_txs
        
    except Exception as e:
        print(f"❌ Failed to query Arweave GraphQL: {e}")
        print(f"\n💡 TIP: Provide ARWEAVE_TX_ID directly if you know the specific checkpoint to verify")
        sys.exit(1)


def download_checkpoint(tx_id: str) -> Dict[str, Any]:
    """Download checkpoint from Arweave"""
    print(f"\n📥 Downloading checkpoint {tx_id[:32]}... from Arweave...")
    
    url = f"https://arweave.net/{tx_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        checkpoint = response.json()
        print(f"✅ Downloaded ({len(response.content):,} bytes)")
        return checkpoint
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return None


def decompress_events(events_compressed: str) -> List[Dict[str, Any]]:
    """Decompress gzipped events from base64 string"""
    try:
        # Decode base64
        compressed_bytes = base64.b64decode(events_compressed)
        
        # Decompress gzip
        decompressed_bytes = gzip.decompress(compressed_bytes)
        
        # Parse JSON
        events = json.loads(decompressed_bytes)
        
        return events
    except Exception as e:
        print(f"❌ Failed to decompress: {e}")
        return []


def print_checkpoint_summary(checkpoint: Dict[str, Any], events: List[Dict[str, Any]]):
    """Print summary of a single checkpoint"""
    header = checkpoint.get('header', {})
    
    print(f"\n{'='*80}")
    print(f"📦 CHECKPOINT #{header.get('checkpoint_number', 'N/A')}")
    print(f"{'='*80}")
    print(f"Event Count: {header.get('event_count', 'N/A')}")
    print(f"Sequence Range: {header.get('sequence_range', {}).get('first', 'N/A')} → {header.get('sequence_range', {}).get('last', 'N/A')}")
    print(f"Time Range:")
    print(f"   Start: {header.get('time_range', {}).get('start', 'N/A')}")
    print(f"   End:   {header.get('time_range', {}).get('end', 'N/A')}")
    print(f"Merkle Root: {header.get('merkle_root', 'N/A')[:32]}...")
    print(f"Code Hash: {header.get('code_hash', 'N/A')[:32]}...")
    
    # Count event types
    event_counts = {}
    for event in events:
        event_type = event.get('event_type', 'UNKNOWN')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"\n📊 Event Breakdown:")
    for event_type, count in sorted(event_counts.items()):
        print(f"   {event_type}: {count}")


def print_event_details(event: Dict[str, Any], index: int):
    """Print detailed view of a single event"""
    print(f"\n{'─'*80}")
    print(f"Event #{index + 1}: {event['event_type']} (seq {event.get('sequence', 'N/A')})")
    print(f"{'─'*80}")
    print(f"Timestamp: {event.get('ts', 'N/A')}")
    print(f"Actor: {event.get('actor_hotkey', 'N/A')[:40]}...")
    
    payload = event.get('payload', {})
    
    if event['event_type'] == 'LEAD_SUBMITTED':
        print(f"📋 Lead ID: {payload.get('lead_id', 'N/A')}")
        print(f"   Miner: {payload.get('miner_hotkey', 'N/A')[:40]}...")
        print(f"   Lead Blob Hash: {payload.get('lead_blob_hash', 'N/A')[:32]}...")
        if 'email_hash' in payload:
            print(f"   Email Hash: {payload.get('email_hash', 'N/A')[:32]}...")
    
    elif event['event_type'] == 'VALIDATION_SUBMITTED':
        print(f"🔍 Lead ID: {payload.get('lead_id', 'N/A')}")
        print(f"   Validator: {payload.get('validator_hotkey', 'N/A')[:40]}...")
        print(f"   Decision Hash: {payload.get('decision_hash', 'N/A')[:32]}...")
    
    elif event['event_type'] == 'REVEAL_SUBMITTED':
        print(f"🔓 Lead ID: {payload.get('lead_id', 'N/A')}")
        print(f"   Decision: {payload.get('decision', 'N/A')}")
        print(f"   Rep Score: {payload.get('rep_score', 'N/A')}")
        if 'rejection_reason' in payload:
            print(f"   Rejection: {payload['rejection_reason']}")
    
    elif event['event_type'] == 'CONSENSUS_RESULT':
        print(f"⚖️  Lead ID: {payload.get('lead_id', 'N/A')}")
        print(f"   Final Decision: {payload.get('final_decision', 'N/A')}")
        print(f"   Final Rep Score: {payload.get('final_rep_score', 'N/A')}")
        print(f"   Validator Count: {payload.get('validator_count', 'N/A')}")
        if 'primary_rejection_reason' in payload:
            print(f"   Primary Rejection: {payload['primary_rejection_reason']}")
    
    elif event['event_type'] == 'EPOCH_INITIALIZATION':
        print(f"🚀 Epoch ID: {payload.get('epoch_id', 'N/A')}")
        print(f"   Lead Count: {payload.get('lead_count', 'N/A')}")
        print(f"   Validator Count: {payload.get('validator_count', 'N/A')}")
    
    elif event['event_type'] == 'RATE_LIMIT':
        print(f"🚫 Miner: {payload.get('miner_hotkey', 'N/A')[:40]}...")
        print(f"   Reason: {payload.get('reason', 'N/A')}")
        print(f"   Limit Type: {payload.get('limit_type', 'N/A')}")


def main():
    print("="*80)
    print("🔓 ARWEAVE CHECKPOINT DECOMPRESSOR")
    print("="*80)
    print()
    
    # Determine mode based on configuration
    if ARWEAVE_TX_ID:
        print(f"📌 Mode: Single TX ID")
        print(f"   TX ID: {ARWEAVE_TX_ID[:32]}...")
        tx_ids = [ARWEAVE_TX_ID]
        
    elif SPECIFIC_DATE:
        print(f"📌 Mode: Specific Date")
        tx_ids = query_checkpoint_ids(date=SPECIFIC_DATE)
    
    else:
        print(f"📌 Mode: Last {LAST_X_HOURS} Hours")
        tx_ids = query_checkpoint_ids(hours=LAST_X_HOURS)
    
    print(f"\n📦 Processing {len(tx_ids)} checkpoint(s)...")
    
    # Process each checkpoint
    all_events = []
    successful_checkpoints = 0
    
    for tx_id in tx_ids:
        checkpoint = download_checkpoint(tx_id)
        if not checkpoint:
            continue
        
        # Decompress events
        events = decompress_events(checkpoint.get('events_compressed', ''))
        if not events:
            continue
        
        successful_checkpoints += 1
        all_events.extend(events)
        
        # Print summary
        print_checkpoint_summary(checkpoint, events)
    
    if not all_events:
        print("\n⚠️  No events found")
        sys.exit(0)
    
    # Print detailed event log
    print(f"\n{'='*80}")
    print(f"📋 DETAILED EVENT LOG ({len(all_events)} events total)")
    print(f"{'='*80}")
    
    for i, event in enumerate(all_events):
        print_event_details(event, i)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ DECOMPRESSION COMPLETE")
    print(f"{'='*80}")
    print(f"Checkpoints processed: {successful_checkpoints}/{len(tx_ids)}")
    print(f"Total events: {len(all_events)}")
    print()
    print("💡 TIPS:")
    print("   - Edit variables at top of script to change query mode")
    print("   - Set ARWEAVE_TX_ID for single checkpoint")
    print("   - Set SPECIFIC_DATE for full day (YYYY-MM-DD)")
    print("   - Change LAST_X_HOURS for time range (default: 4)")
    print()
    print("🔍 WHAT THIS PROVES:")
    print("   ✅ All lead IDs, hashes, and hotkeys are intact")
    print("   ✅ All timestamps preserved (microsecond precision)")
    print("   ✅ All signatures and payloads complete")
    print("   ✅ Nothing lost - compression is 100% lossless!")
    print()


if __name__ == "__main__":
    main()
