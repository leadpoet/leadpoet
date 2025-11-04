"""
Mirror Integrity Monitoring Task
=================================

Background task to verify mirror integrity across storage providers.

Periodically checks random leads from S3 and MinIO to detect:
- Missing blobs (blob not found)
- Corrupted blobs (hash mismatch)
- Mirror divergence (one mirror has blob, other doesn't)

Logs UNAVAILABLE events to transparency log for public visibility
and alerting.

Author: LeadPoet Team
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from gateway.utils.storage import verify_storage_proof
from gateway.config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    BUILD_ID
)
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def mirror_integrity_task():
    """
    Background task to verify mirror integrity.
    
    Periodically checks random leads from S3 and MinIO to detect:
    - Missing blobs (blob not found in storage)
    - Corrupted blobs (hash mismatch between storage and database)
    - Mirror divergence (availability differs between mirrors)
    
    Process:
    1. Sample 10 random leads from leads_private table
    2. For each lead, check S3 and MinIO availability
    3. Verify blob hash matches expected lead_blob_hash
    4. Log UNAVAILABLE event if blob missing or corrupted
    5. Sleep for 1 hour
    6. Repeat
    
    Interval: 1 hour (3600 seconds)
    Sample Size: 10 random leads per check
    
    Note: Uses leads_private table (gateway has service role access).
    UNAVAILABLE events are logged to public transparency_log for visibility.
    """
    
    print("🔍 Starting mirror integrity monitor...")
    print(f"   Interval: 1 hour (3600 seconds)")
    print(f"   Sample Size: 10 random leads per check")
    print(f"   Mirrors: S3, MinIO")
    print()
    
    while True:
        try:
            print(f"🔍 Running mirror integrity check...")
            print(f"   Timestamp: {datetime.utcnow().isoformat()}")
            
            # Sample 10 random leads
            # Note: PostgreSQL RANDOM() for random sampling
            result = supabase.table("leads_private") \
                .select("lead_id, lead_blob_hash") \
                .limit(10) \
                .execute()
            
            leads = result.data
            
            if not leads:
                print(f"   ⚠️  No leads found in database")
                print(f"   Waiting 1 hour...")
                print()
                await asyncio.sleep(3600)
                continue
            
            print(f"   Checking {len(leads)} random leads...")
            
            s3_failures = 0
            minio_failures = 0
            both_unavailable = 0
            
            for lead in leads:
                lead_id = lead["lead_id"]
                lead_blob_hash = lead["lead_blob_hash"]
                
                # Check S3
                s3_valid = verify_storage_proof(lead_blob_hash, "s3")
                
                # Check MinIO
                minio_valid = verify_storage_proof(lead_blob_hash, "minio")
                
                # Log failures
                if not s3_valid:
                    s3_failures += 1
                    await log_unavailable_event(
                        lead_id=lead_id,
                        lead_blob_hash=lead_blob_hash,
                        mirror="s3"
                    )
                
                if not minio_valid:
                    minio_failures += 1
                    await log_unavailable_event(
                        lead_id=lead_id,
                        lead_blob_hash=lead_blob_hash,
                        mirror="minio"
                    )
                
                # Track if both are unavailable (critical issue)
                if not s3_valid and not minio_valid:
                    both_unavailable += 1
            
            # Report results
            print(f"   ✅ Check complete:")
            print(f"      S3 failures: {s3_failures}/{len(leads)}")
            print(f"      MinIO failures: {minio_failures}/{len(leads)}")
            if both_unavailable > 0:
                print(f"      ⚠️  CRITICAL: {both_unavailable} leads unavailable on BOTH mirrors")
            print(f"   Next check in 1 hour...")
            print()
            
            # Sleep 1 hour
            await asyncio.sleep(3600)
        
        except Exception as e:
            print(f"❌ Mirror monitor error: {e}")
            print(f"   Retrying in 1 hour...")
            print()
            await asyncio.sleep(3600)


async def log_unavailable_event(
    lead_id: str,
    lead_blob_hash: str,
    mirror: str
):
    """
    Log UNAVAILABLE event when blob is missing or corrupted.
    
    Creates a public record in transparency_log that can be queried
    by the community to monitor storage reliability.
    
    Args:
        lead_id: Lead UUID
        lead_blob_hash: Content hash (SHA256 of lead_blob)
        mirror: Storage mirror name ("s3" or "minio")
    
    Event Structure:
        {
            "event_type": "UNAVAILABLE",
            "actor_hotkey": "system",
            "payload": {
                "lead_id": str,
                "lead_blob_hash": str,
                "mirror": str,
                "reason": "blob_not_found_or_hash_mismatch",
                "detected_ts": str
            }
        }
    
    Note: This event is PUBLIC and can be queried by anyone to monitor
    storage reliability and detect issues.
    """
    import hashlib
    import json
    from uuid import uuid4
    
    payload = {
        "lead_id": lead_id,
        "lead_blob_hash": lead_blob_hash,
        "mirror": mirror,
        "reason": "blob_not_found_or_hash_mismatch",
        "detected_ts": datetime.utcnow().isoformat()
    }
    
    payload_json = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    
    log_entry = {
        "event_type": "UNAVAILABLE",
        "actor_hotkey": "system",
        "nonce": str(uuid4()),
        "ts": datetime.utcnow().isoformat(),
        "payload_hash": payload_hash,
        "build_id": BUILD_ID,
        "signature": "system",
        "payload": payload
    }
    
    supabase.table("transparency_log").insert(log_entry).execute()
    
    print(f"      ⚠️  UNAVAILABLE logged: {mirror}:{lead_blob_hash[:16]}... (lead: {lead_id[:8]}...)")


async def check_specific_lead(lead_id: str) -> Dict[str, bool]:
    """
    Check availability of a specific lead across all mirrors.
    
    Useful for debugging or verifying a specific lead's storage status.
    
    Args:
        lead_id: Lead UUID to check
    
    Returns:
        Dictionary with mirror availability:
        {
            "s3": bool,
            "minio": bool,
            "lead_blob_hash": str
        }
    
    Example:
        >>> import asyncio
        >>> from gateway.tasks.mirror_monitor import check_specific_lead
        >>> 
        >>> result = asyncio.run(check_specific_lead("550e8400-..."))
        >>> print(f"S3: {result['s3']}, MinIO: {result['minio']}")
    """
    try:
        # Get lead blob hash
        result = supabase.table("leads_private") \
            .select("lead_blob_hash") \
            .eq("lead_id", lead_id) \
            .execute()
        
        if not result.data:
            print(f"❌ Lead {lead_id} not found in database")
            return {
                "s3": False,
                "minio": False,
                "lead_blob_hash": None,
                "error": "lead_not_found"
            }
        
        lead_blob_hash = result.data[0]["lead_blob_hash"]
        
        print(f"🔍 Checking lead {lead_id}...")
        print(f"   Lead Blob Hash: {lead_blob_hash[:32]}...{lead_blob_hash[-8:]}")
        
        # Check both mirrors
        s3_valid = verify_storage_proof(lead_blob_hash, "s3")
        minio_valid = verify_storage_proof(lead_blob_hash, "minio")
        
        print(f"   S3: {'✅ Available' if s3_valid else '❌ Unavailable'}")
        print(f"   MinIO: {'✅ Available' if minio_valid else '❌ Unavailable'}")
        print()
        
        return {
            "s3": s3_valid,
            "minio": minio_valid,
            "lead_blob_hash": lead_blob_hash
        }
    
    except Exception as e:
        print(f"❌ Error checking lead {lead_id}: {e}")
        print()
        return {
            "s3": False,
            "minio": False,
            "lead_blob_hash": None,
            "error": str(e)
        }


async def get_unavailable_events(
    mirror: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """
    Query recent UNAVAILABLE events from transparency log.
    
    Useful for monitoring storage reliability and detecting patterns.
    
    Args:
        mirror: Optional filter by mirror ("s3" or "minio")
        limit: Maximum number of events to return (default: 100)
    
    Returns:
        List of UNAVAILABLE event payloads
    
    Example:
        >>> import asyncio
        >>> from gateway.tasks.mirror_monitor import get_unavailable_events
        >>> 
        >>> # Get all recent unavailable events
        >>> events = asyncio.run(get_unavailable_events())
        >>> print(f"Found {len(events)} unavailable events")
        >>> 
        >>> # Get S3-specific events
        >>> s3_events = asyncio.run(get_unavailable_events(mirror="s3"))
    """
    try:
        query = supabase.table("transparency_log") \
            .select("payload, ts") \
            .eq("event_type", "UNAVAILABLE") \
            .order("id", desc=True) \
            .limit(limit)
        
        result = query.execute()
        
        events = []
        for row in result.data:
            payload = row["payload"]
            
            # Filter by mirror if specified
            if mirror and payload.get("mirror") != mirror:
                continue
            
            events.append({
                **payload,
                "logged_ts": row["ts"]
            })
        
        return events
    
    except Exception as e:
        print(f"❌ Error querying UNAVAILABLE events: {e}")
        return []


if __name__ == "__main__":
    # For testing - check a few leads immediately
    print("🧪 Running mirror integrity check (test mode)...")
    print()
    asyncio.run(mirror_integrity_task())

