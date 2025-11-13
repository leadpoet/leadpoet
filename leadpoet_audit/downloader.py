"""
Data Downloader for LeadPoet Community Audit Tool
==================================================

This module provides functions to download events from the PUBLIC transparency_log.

**DATA SOURCE**: Public transparency_log table ONLY
**NO ACCESS TO**: Private database tables (leads_private, validation_evidence_private)

All functions use SUPABASE_ANON_KEY for public read-only access.

Author: LeadPoet Team
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from supabase import create_client, Client

# Public read-only access (hardcoded for community transparency)
# ANON key only has SELECT permission on transparency_log (via RLS policies)
SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def download_consensus_results(epoch_id: int) -> List[Dict]:
    """
    Download CONSENSUS_RESULT events for given epoch (PUBLIC data).
    
    These events are logged by the gateway after consensus calculation
    and contain: lead_id, final_decision, final_rep_score, primary_rejection_reason, etc.
    
    **TIMING**: Run this AFTER epoch closes and gateway computes consensus (epoch N+1 start)
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        List of consensus result payloads
    
    Example:
        >>> results = download_consensus_results(100)
        >>> len(results)
        50  # 50 leads validated in epoch 100
        >>> results[0]
        {
            'lead_id': '550e8400-e29b-41d4-a716-446655440001',
            'epoch_id': 100,
            'final_decision': 'approve',
            'final_rep_score': 85.3,
            'primary_rejection_reason': 'pass',
            'validator_count': 5,
            'consensus_timestamp': '2024-11-04T12:00:00Z'
        }
    """
    result = supabase.table("transparency_log") \
        .select("payload") \
        .eq("event_type", "CONSENSUS_RESULT") \
        .eq("payload->>epoch_id", str(epoch_id)) \
        .order("id") \
        .execute()
    
    consensus_results = [row["payload"] for row in result.data]
    
    print(f"📥 Downloaded {len(consensus_results)} consensus results for epoch {epoch_id}")
    
    return consensus_results


def download_submission_events(epoch_id: Optional[int] = None) -> List[Dict]:
    """
    Download SUBMISSION events to get miner_hotkey for each lead.
    
    SUBMISSION events contain the lead_id and the actor_hotkey is the miner who submitted it.
    This allows mapping lead_id -> miner_hotkey for attribution.
    
    Args:
        epoch_id: Optional epoch number to filter by (if None, returns all submissions)
    
    Returns:
        List of submission records with lead_id and miner_hotkey
    
    Example:
        >>> submissions = download_submission_events()
        >>> submissions[0]
        {
            'lead_id': '550e8400-e29b-41d4-a716-446655440001',
            'miner_hotkey': '5GNJqR...',
            'submission_timestamp': '2024-11-04T10:30:00Z'
        }
    """
    query = supabase.table("transparency_log") \
        .select("actor_hotkey, payload, ts") \
        .eq("event_type", "SUBMISSION")
    
    # Optional: filter by epoch if provided
    # Note: SUBMISSION events don't have epoch_id in payload, so we can't filter by epoch directly
    # We just return all submissions and let the analyzer match by lead_id
    
    result = query.execute()
    
    submissions = []
    for row in result.data:
        submissions.append({
            "lead_id": row["payload"]["lead_id"],
            "miner_hotkey": row["actor_hotkey"],
            "submission_timestamp": row["ts"]
        })
    
    print(f"📥 Downloaded {len(submissions)} submission events")
    
    return submissions


def download_epoch_assignment(epoch_id: int) -> Optional[Dict]:
    """
    Download EPOCH_INITIALIZATION event to get list of 50 lead_ids for the epoch.
    
    EPOCH_INITIALIZATION events are logged at epoch start and contain the epoch
    boundaries, queue state, and lead assignment in a single atomic event.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Assignment data dict (extracted from EPOCH_INITIALIZATION), or None if not found
    
    Example:
        >>> assignment = download_epoch_assignment(100)
        >>> assignment
        {
            'epoch_id': 100,
            'lead_ids': ['550e8400-...', '550e8400-...', ...],  # 50 lead_ids
            'lead_count': 50,
            'queue_root': 'abc123...',
            'validator_count': 9
        }
    """
    result = supabase.table("transparency_log") \
        .select("payload") \
        .eq("event_type", "EPOCH_INITIALIZATION") \
        .eq("payload->>epoch_id", str(epoch_id)) \
        .limit(1) \
        .execute()
    
    if not result.data:
        print(f"⚠️  No EPOCH_INITIALIZATION found for epoch {epoch_id}")
        return None
    
    epoch_init = result.data[0]["payload"]
    
    # Extract assignment data from nested structure
    assignment = {
        "epoch_id": epoch_init["epoch_id"],
        "lead_ids": epoch_init["assignment"]["assigned_lead_ids"],
        "lead_count": len(epoch_init["assignment"]["assigned_lead_ids"]),
        "queue_root": epoch_init["queue_state"]["queue_merkle_root"],
        "validator_count": epoch_init["assignment"]["validator_count"]
    }
    
    print(f"📥 Downloaded EPOCH_INITIALIZATION for epoch {epoch_id}: {assignment['lead_count']} leads assigned")
    
    return assignment


def download_epoch_events(epoch_id: int, event_types: Optional[List[str]] = None) -> List[Dict]:
    """
    Download all transparency log events for given epoch (optional: filter by event types).
    
    Utility function to download multiple event types at once.
    
    Args:
        epoch_id: Epoch number
        event_types: Optional list of event types to filter (e.g. ["CONSENSUS_RESULT", "EPOCH_ASSIGNMENT"])
    
    Returns:
        List of event dictionaries with full transparency log data
    
    Example:
        >>> events = download_epoch_events(100, ["CONSENSUS_RESULT", "EPOCH_ASSIGNMENT"])
        >>> len(events)
        51  # 50 CONSENSUS_RESULT + 1 EPOCH_ASSIGNMENT
    """
    # Note: We can't easily filter by epoch_id for all event types since only some have epoch_id in payload
    # This is a best-effort query
    
    query = supabase.table("transparency_log").select("*")
    
    if event_types:
        query = query.in_("event_type", event_types)
    
    result = query.order("id").execute()
    
    # Filter by epoch_id in payload if present
    events = []
    for row in result.data:
        payload = row.get("payload", {})
        if isinstance(payload, dict) and str(payload.get("epoch_id")) == str(epoch_id):
            events.append(row)
    
    print(f"📥 Downloaded {len(events)} events for epoch {epoch_id}")
    
    return events


def download_queue_root(epoch_id: int) -> Optional[Dict]:
    """
    Download queue state from EPOCH_INITIALIZATION event for given epoch.
    
    EPOCH_INITIALIZATION events contain the Merkle root of the pending leads queue 
    at epoch start in the queue_state nested field.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Queue state dict (extracted from EPOCH_INITIALIZATION), or None if not found
    
    Example:
        >>> queue_root = download_queue_root(100)
        >>> queue_root
        {
            'epoch_id': 100,
            'queue_root': 'abc123...',
            'pending_count': 75
        }
    """
    result = supabase.table("transparency_log") \
        .select("payload") \
        .eq("event_type", "EPOCH_INITIALIZATION") \
        .eq("payload->>epoch_id", str(epoch_id)) \
        .limit(1) \
        .execute()
    
    if not result.data:
        print(f"⚠️  No EPOCH_INITIALIZATION found for epoch {epoch_id}")
        return None
    
    epoch_init = result.data[0]["payload"]
    
    # Extract queue state from nested structure
    queue_state = {
        "epoch_id": epoch_init["epoch_id"],
        "queue_root": epoch_init["queue_state"]["queue_merkle_root"],
        "pending_count": epoch_init["queue_state"]["pending_lead_count"]
    }
    
    print(f"📥 Downloaded queue state for epoch {epoch_id}")
    
    return queue_state


# Convenience function to download all epoch data at once
def download_epoch_data(epoch_id: int) -> Dict:
    """
    Download all relevant data for given epoch in one call.
    
    This is a convenience function that calls all the individual download functions
    and returns a comprehensive data dictionary.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Dictionary containing all epoch data:
        {
            'consensus_results': List[Dict],
            'submissions': List[Dict],
            'epoch_assignment': Dict,
            'queue_root': Dict
        }
    
    Example:
        >>> data = download_epoch_data(100)
        >>> data['consensus_results']  # 50 consensus results
        >>> data['epoch_assignment']   # Assignment details
        >>> data['submissions']        # All submissions (for miner mapping)
    """
    print(f"\n📦 Downloading all data for epoch {epoch_id}...")
    print("-" * 70)
    
    data = {
        'consensus_results': download_consensus_results(epoch_id),
        'submissions': download_submission_events(),
        'epoch_assignment': download_epoch_assignment(epoch_id),
        'queue_root': download_queue_root(epoch_id)
    }
    
    print("-" * 70)
    print(f"✅ Download complete for epoch {epoch_id}\n")
    
    return data

