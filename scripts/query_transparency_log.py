#!/usr/bin/env python3
"""
Query Transparency Log (Supabase)

This script queries the LeadPoet transparency log from Supabase, allowing you to inspect
all events logged by the gateway in real-time.

Query modes:
    1. By email hash (highest priority) - Track specific lead's journey
    2. By event type (high priority) - See all events of a specific type
    3. By email hash + event type (combined) - E.g., CONSENSUS_RESULT for specific lead
    4. By specific date (medium priority) - All events on a date
    5. By last X hours (lowest priority - default) - Recent events

Usage:
    python query_transparency_log.py
    
Example - Track consensus for specific lead:
    EMAIL_HASH = "a3c7f8e2b4d9e1f0c8a5b6d2e9f1a4c7..."
    EVENT_TYPE = "CONSENSUS_RESULT"
"""

import sys
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION - Edit these variables as needed
# ============================================================

# Mode 1: Query by Email Hash (highest priority)
# Get this from transparency_log or by SHA256 hashing an email
# Example: "a3c7f8e2b4d9e1f0c8a5b6d2e9f1a4c7..."
EMAIL_HASH = ""  # Leave blank to use other modes

# Mode 2: Query by Event Type (high priority)
# Options: SUBMISSION_REQUEST, STORAGE_PROOF, SUBMISSION, CONSENSUS_RESULT,
#          EPOCH_INITIALIZATION, EPOCH_END, EPOCH_INPUTS, DEREGISTERED_MINER_REMOVAL, RATE_LIMIT
# NOTE: If both EMAIL_HASH and EVENT_TYPE are set, query will filter by BOTH
EVENT_TYPE = ""  # Leave blank to use other modes

# Mode 3: Specific Date (YYYY-MM-DD format)
# Example: "2025-11-20" pulls all events from that day
SPECIFIC_DATE = ""  # Leave blank to use time range

# Mode 4: Last X Hours (default mode if above are blank)
# Example: 4 = last 4 hours, 24 = last 24 hours
LAST_X_HOURS = 8  # Change this to pull different time ranges

# Maximum results to return (prevents overwhelming output)
MAX_RESULTS = 100

# ============================================================
# Priority Hierarchy:
# 1. If EMAIL_HASH is set → Use it (optionally filter by EVENT_TYPE if also set)
# 2. If EVENT_TYPE is set (without EMAIL_HASH) → Use it (ignore date and hours)
# 3. If SPECIFIC_DATE is set → Use it (ignore hours)
# 4. Otherwise → Use LAST_X_HOURS
# ============================================================

# Valid event types (for validation)
VALID_EVENT_TYPES = [
    "SUBMISSION_REQUEST",
    "STORAGE_PROOF",
    "SUBMISSION",
    "CONSENSUS_RESULT",
    "EPOCH_INITIALIZATION",
    "EPOCH_END",
    "EPOCH_INPUTS",
    "DEREGISTERED_MINER_REMOVAL",
    "RATE_LIMIT"
]


def get_supabase_client():
    """Initialize Supabase client"""
    from supabase import create_client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("❌ Error: SUPABASE_URL or SUPABASE_ANON_KEY not set in .env")
        print("   Please create a .env file with these variables:")
        print("   SUPABASE_URL=https://your-project.supabase.co")
        print("   SUPABASE_ANON_KEY=your-anon-key")
        sys.exit(1)
    
    return create_client(url, key)


def query_by_email_hash(supabase, email_hash: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query transparency log by email hash (optionally filtered by event type)"""
    if event_type:
        print(f"🔍 Querying transparency log by email hash: {email_hash[:32]}... (event_type: {event_type})")
    else:
        print(f"🔍 Querying transparency log by email hash: {email_hash[:32]}...")
    
    try:
        query = supabase.table("transparency_log")\
            .select("*")\
            .eq("email_hash", email_hash)
        
        # Optionally filter by event type
        if event_type:
            if event_type not in VALID_EVENT_TYPES:
                print(f"❌ Invalid event type: {event_type}")
                print(f"   Valid types: {', '.join(VALID_EVENT_TYPES)}")
                sys.exit(1)
            query = query.eq("event_type", event_type)
        
        result = query.order("ts", desc=True)\
            .limit(MAX_RESULTS)\
            .execute()
        
        if not result.data:
            if event_type:
                print(f"⚠️  No {event_type} events found for email hash: {email_hash[:32]}...")
            else:
                print(f"⚠️  No events found for email hash: {email_hash[:32]}...")
            return []
        
        print(f"✅ Found {len(result.data)} event(s)")
        return result.data
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


def query_by_event_type(supabase, event_type: str) -> List[Dict[str, Any]]:
    """Query transparency log by event type"""
    
    if event_type not in VALID_EVENT_TYPES:
        print(f"❌ Invalid event type: {event_type}")
        print(f"   Valid types: {', '.join(VALID_EVENT_TYPES)}")
        sys.exit(1)
    
    print(f"🔍 Querying transparency log for event type: {event_type}")
    
    try:
        result = supabase.table("transparency_log")\
            .select("*")\
            .eq("event_type", event_type)\
            .order("ts", desc=True)\
            .limit(MAX_RESULTS)\
            .execute()
        
        if not result.data:
            print(f"⚠️  No events found for type: {event_type}")
            return []
        
        print(f"✅ Found {len(result.data)} event(s)")
        return result.data
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


def query_by_date(supabase, date: str) -> List[Dict[str, Any]]:
    """Query transparency log by specific date"""
    print(f"🔍 Querying transparency log for date: {date}")
    
    try:
        # Parse date and create time range
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        result = supabase.table("transparency_log")\
            .select("*")\
            .gte("ts", start_time.isoformat())\
            .lt("ts", end_time.isoformat())\
            .order("ts", desc=True)\
            .limit(MAX_RESULTS)\
            .execute()
        
        if not result.data:
            print(f"⚠️  No events found for date: {date}")
            return []
        
        print(f"✅ Found {len(result.data)} event(s)")
        return result.data
        
    except ValueError:
        print(f"❌ Invalid date format: {date}")
        print(f"   Expected format: YYYY-MM-DD (e.g., 2025-11-20)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


def query_by_hours(supabase, hours: int) -> List[Dict[str, Any]]:
    """Query transparency log for last X hours"""
    print(f"🔍 Querying transparency log for last {hours} hours")
    
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = supabase.table("transparency_log")\
            .select("*")\
            .gte("ts", cutoff_time.isoformat())\
            .order("ts", desc=True)\
            .limit(MAX_RESULTS)\
            .execute()
        
        if not result.data:
            print(f"⚠️  No events found in last {hours} hours")
            return []
        
        print(f"✅ Found {len(result.data)} event(s)")
        return result.data
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


def print_summary(events: List[Dict[str, Any]]):
    """Print summary statistics"""
    if not events:
        return
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total Events: {len(events)}")
    
    # Count by event type
    event_counts = {}
    for event in events:
        event_type = event.get('event_type', 'UNKNOWN')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"\n📋 Event Type Breakdown:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"   {event_type}: {count}")
    
    # Count by actor
    actor_counts = {}
    for event in events:
        actor = event.get('actor_hotkey', 'UNKNOWN')[:20]
        actor_counts[actor] = actor_counts.get(actor, 0) + 1
    
    print(f"\n👥 Top Actors:")
    for actor, count in sorted(actor_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {actor}...: {count}")
    
    # Time range
    timestamps = [event.get('ts') for event in events if event.get('ts')]
    if timestamps:
        earliest = min(timestamps)
        latest = max(timestamps)
        print(f"\n⏰ Time Range:")
        print(f"   Earliest: {earliest}")
        print(f"   Latest:   {latest}")


def print_event_details(event: Dict[str, Any], index: int):
    """Print detailed view of a single event"""
    print(f"\n{'─'*80}")
    print(f"Event #{index + 1}: {event['event_type']}")
    print(f"{'─'*80}")
    print(f"Event ID: {event.get('event_id', 'N/A')}")
    print(f"Timestamp: {event.get('ts', 'N/A')}")
    print(f"Actor: {event.get('actor_hotkey', 'N/A')[:50]}...")
    print(f"Nonce: {event.get('nonce', 'N/A')}")
    
    if event.get('email_hash'):
        print(f"Email Hash: {event['email_hash'][:32]}...")
    
    if event.get('epoch_id'):
        print(f"Epoch ID: {event['epoch_id']}")
    
    payload = event.get('payload', {})
    if payload:
        print(f"\n📦 Payload:")
        
        # Pretty print based on event type
        event_type = event['event_type']
        
        if event_type == 'SUBMISSION_REQUEST':
            print(f"   Build ID: {payload.get('build_id', 'N/A')[:32]}...")
            print(f"   Lead Blob Hash: {payload.get('lead_blob_hash', 'N/A')[:32]}...")
            if 'miner_hotkey' in payload:
                print(f"   Miner: {payload['miner_hotkey'][:50]}...")
        
        elif event_type == 'STORAGE_PROOF':
            print(f"   Mirror: {payload.get('mirror', 'N/A')}")
            print(f"   Lead Blob Hash: {payload.get('lead_blob_hash', 'N/A')[:32]}...")
            print(f"   Verified Hash: {payload.get('verified_hash', 'N/A')[:32]}...")
            print(f"   Match: {payload.get('hash_match', 'N/A')}")
        
        elif event_type == 'SUBMISSION':
            print(f"   Lead ID: {payload.get('lead_id', 'N/A')}")
            print(f"   Miner: {payload.get('miner_hotkey', 'N/A')[:50]}...")
            print(f"   Lead Blob Hash: {payload.get('lead_blob_hash', 'N/A')[:32]}...")
            print(f"   Email Hash: {payload.get('email_hash', 'N/A')[:32]}...")
        
        elif event_type == 'CONSENSUS_RESULT':
            print(f"   Lead ID: {payload.get('lead_id', 'N/A')}")
            print(f"   Final Decision: {payload.get('final_decision', 'N/A')}")
            print(f"   Final Rep Score: {payload.get('final_rep_score', 'N/A')}")
            print(f"   Validator Count: {payload.get('validator_count', 'N/A')}")
            print(f"   Miner: {payload.get('miner_hotkey', 'N/A')[:50]}...")
            if 'primary_rejection_reason' in payload:
                print(f"   Rejection: {payload['primary_rejection_reason']}")
        
        elif event_type == 'EPOCH_INITIALIZATION':
            print(f"   Epoch ID: {payload.get('epoch_id', 'N/A')}")
            print(f"   Start Block: {payload.get('start_block', 'N/A')}")
            print(f"   End Block: {payload.get('end_block', 'N/A')}")
            print(f"   Queue Root: {payload.get('queue_root', 'N/A')[:32]}...")
            print(f"   Lead Count: {payload.get('lead_count', 'N/A')}")
            print(f"   Validator Count: {payload.get('validator_count', 'N/A')}")
            print(f"   Assigned Leads: {len(payload.get('assigned_lead_ids', []))}")
        
        elif event_type == 'DEREGISTERED_MINER_REMOVAL':
            print(f"   Epoch ID: {payload.get('epoch_id', 'N/A')}")
            print(f"   Deregistered Miners: {len(payload.get('deregistered_miners', []))}")
            print(f"   Total Leads Identified: {payload.get('total_leads_identified', 'N/A')}")
            print(f"   Total Leads Deleted: {payload.get('total_leads_deleted', 'N/A')}")
            print(f"   Deletion Success: {payload.get('deletion_success', 'N/A')}")
        
        elif event_type == 'RATE_LIMIT':
            print(f"   Miner: {payload.get('miner_hotkey', 'N/A')[:50]}...")
            print(f"   Reason: {payload.get('reason', 'N/A')}")
            print(f"   Limit Type: {payload.get('limit_type', 'N/A')}")
        
        else:
            # Generic payload display
            for key, value in payload.items():
                if isinstance(value, str) and len(value) > 60:
                    print(f"   {key}: {value[:60]}...")
                else:
                    print(f"   {key}: {value}")
    
    if event.get('tee_sequence'):
        print(f"\n🔒 TEE Sequence: {event['tee_sequence']}")
    
    if event.get('arweave_tx_id'):
        print(f"📦 Arweave TX: {event['arweave_tx_id']}")


def main():
    print("="*80)
    print("🔍 TRANSPARENCY LOG QUERY TOOL")
    print("="*80)
    print()
    
    # Initialize Supabase client
    print("🔌 Connecting to Supabase...")
    supabase = get_supabase_client()
    print("✅ Connected")
    print()
    
    # Determine mode based on configuration
    events = []
    
    if EMAIL_HASH:
        if EVENT_TYPE:
            print(f"📌 Mode: Email Hash + Event Type Query")
        else:
            print(f"📌 Mode: Email Hash Query")
        events = query_by_email_hash(supabase, EMAIL_HASH, event_type=EVENT_TYPE if EVENT_TYPE else None)
    
    elif EVENT_TYPE:
        print(f"📌 Mode: Event Type Query")
        events = query_by_event_type(supabase, EVENT_TYPE)
    
    elif SPECIFIC_DATE:
        print(f"📌 Mode: Specific Date")
        events = query_by_date(supabase, SPECIFIC_DATE)
    
    else:
        print(f"📌 Mode: Last {LAST_X_HOURS} Hours")
        events = query_by_hours(supabase, LAST_X_HOURS)
    
    if not events:
        print("\n⚠️  No events found")
        print()
        print("💡 TIPS:")
        print("   - Try increasing LAST_X_HOURS")
        print("   - Check SPECIFIC_DATE format (YYYY-MM-DD)")
        print("   - Verify EVENT_TYPE is spelled correctly")
        print("   - Confirm EMAIL_HASH exists in the database")
        sys.exit(0)
    
    # Print summary
    print_summary(events)
    
    # Print detailed event log
    print(f"\n{'='*80}")
    print(f"📋 DETAILED EVENT LOG ({len(events)} events)")
    print(f"{'='*80}")
    
    for i, event in enumerate(events):
        print_event_details(event, i)
    
    # Footer
    print(f"\n{'='*80}")
    print(f"✅ QUERY COMPLETE")
    print(f"{'='*80}")
    print(f"Total events displayed: {len(events)}")
    if len(events) >= MAX_RESULTS:
        print(f"⚠️  Result limit reached ({MAX_RESULTS}). Increase MAX_RESULTS to see more.")
    print()
    print("💡 TIPS:")
    print("   - Edit variables at top of script to change query mode")
    print("   - Set EMAIL_HASH to track a specific lead's journey")
    print("   - Set EVENT_TYPE to see all events of one type")
    print("   - Set BOTH EMAIL_HASH + EVENT_TYPE to filter by both (e.g., CONSENSUS_RESULT for specific email)")
    print("   - Set SPECIFIC_DATE for full day (YYYY-MM-DD)")
    print("   - Change LAST_X_HOURS for time range (default: 8)")
    print(f"   - Change MAX_RESULTS to show more/fewer events (current: {MAX_RESULTS})")
    print()
    print("🔍 WHAT THIS SHOWS:")
    print("   ✅ Real-time transparency log from Supabase")
    print("   ✅ All events with full payloads and metadata")
    print("   ✅ TEE sequences and Arweave TX IDs (when available)")
    print("   ✅ Complete audit trail for debugging and verification")
    print()
    print("📌 AVAILABLE EVENT TYPES:")
    for event_type in VALID_EVENT_TYPES:
        print(f"   - {event_type}")
    print()


if __name__ == "__main__":
    main()

