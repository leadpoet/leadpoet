"""
TEE-First Event Logging Utility

This module provides functions to log events to the TEE buffer (primary source of truth)
and then mirror them to Supabase (query cache).

ARCHITECTURE CHANGE (Phase 2.3):
- OLD: Gateway → Arweave (instant) → Supabase mirror
- NEW: Gateway → TEE buffer → Supabase mirror
           ↓ (hourly)
        Arweave checkpoint

Key principles:
1. ALWAYS write to TEE buffer first (hardware-protected, canonical copy)
2. TEE buffer write MUST succeed or request fails (prevents censorship)
3. Mirror to Supabase asynchronously (non-authoritative cache)
4. TEE batches events to Arweave hourly (cost reduction: $300/mo → $0.30/mo)

Security guarantees:
- Events stored in TEE hardware-protected memory (operator cannot modify)
- Sequence numbers from TEE prove ordering (prevents reordering attacks)
- If request succeeds, event is guaranteed in TEE buffer
- TEE attestation proves canonical code is running

==============================================================================
TRUST HIERARCHY FOR EVENT VERIFICATION (Task 2.4)
==============================================================================

When verifying events or resolving disputes, follow this precedence:

1. TEE BUFFER (Authoritative for events < 1 hour old)
   - Location: Nitro Enclave in-memory buffer
   - Access: vsock RPC to enclave
   - Guarantees: Hardware-protected, tamper-proof, code integrity verified
   - Limitation: Only stores events from past hour (cleared after Arweave batch)

2. ARWEAVE CHECKPOINTS (Authoritative for events > 1 hour old)
   - Location: Permanent Arweave blockchain storage
   - Access: Download checkpoint via Arweave TX ID
   - Guarantees: Immutable, public, TEE-signed Merkle root
   - Usage: Canonical source after hourly batch

3. SUPABASE (Non-Authoritative - Query Cache Only)
   - Location: Supabase PostgreSQL database
   - Access: SQL queries
   - Purpose: Fast queries for epochs, leads, consensus computation
   - WARNING: NEVER use Supabase alone for event verification
   - Note: Supabase is a convenience mirror only, not source of truth

DISPUTE RESOLUTION LOGIC:
- If event timestamp < 1 hour ago: Query TEE buffer (call get_canonical_event())
- If event timestamp > 1 hour ago: Query Arweave checkpoint
- Never trust Supabase alone: Always verify against TEE or Arweave

Example: Miner claims "I submitted lead X but gateway denies it"
1. Check TEE buffer for SUBMISSION event with lead_id=X
2. If not in buffer (> 1 hour old), download relevant Arweave checkpoint
3. Verify event presence using Merkle inclusion proof
4. Supabase query is ONLY for finding which checkpoint to check

==============================================================================
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from utils.tee_client import tee_client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BUILD_ID

# Python logging
logger = logging.getLogger(__name__)

# Supabase client (optional - for mirroring only)
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logger.info("✅ Supabase client initialized for event mirroring")
    except Exception as e:
        logger.warning(f"⚠️  Supabase client initialization failed: {e}")
        logger.warning("   Event mirroring to Supabase will be skipped")
else:
    logger.warning("⚠️  Supabase credentials not configured - mirroring disabled")

# Fallback logging directory (for TEE connection failures)
FALLBACK_LOG_DIR = Path("gateway/logs/tee_fallback")
FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)


def compute_payload_hash(payload: dict) -> str:
    """
    Compute SHA256 hash of event payload for integrity verification.
    
    Args:
        payload: Event payload dictionary
    
    Returns:
        Hex-encoded SHA256 hash (64 characters)
    """
    # Canonical JSON serialization (sorted keys, no whitespace)
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)  # Handle datetime objects
    payload_bytes = payload_json.encode('utf-8')
    return hashlib.sha256(payload_bytes).hexdigest()


async def log_event(event: dict) -> Dict:
    """
    Logs event to TEE buffer (authoritative) and Supabase (cache).
    
    This is the PRIMARY logging function for all gateway events.
    
    CRITICAL SECURITY PROPERTIES:
    - TEE buffer write happens FIRST (hardware-protected canonical copy)
    - If TEE write fails, request MUST fail (prevents censorship)
    - Supabase write can fail (it's just a cache for queries)
    - Events stay in TEE memory for up to 1 hour, then batch to Arweave
    - Sequence numbers from TEE prove event ordering
    
    Flow:
        1. Write to TEE buffer via vsock → canonical copy
        2. TEE returns sequence number (monotonically increasing)
        3. Mirror to Supabase → fast queries, non-authoritative
        4. TEE will batch to Arweave hourly
    
    Args:
        event: Event dictionary with fields:
            - event_type: str (SUBMISSION_REQUEST, VALIDATION_RESULT, etc.)
            - All other event-specific fields
    
    Returns:
        dict: Response from TEE with:
            - status: "buffered"
            - sequence: int (TEE sequence number, proves ordering)
            - buffer_size: int (current buffer size)
            - overflow_warning: bool (true if buffer approaching capacity)
    
    Raises:
        RuntimeError: If TEE buffer write fails (request MUST fail)
        
    Example:
        event = {
            "event_type": "SUBMISSION_REQUEST",
            "lead_id": "uuid",
            "miner_hotkey": "ss58_address",
            "timestamp": datetime.utcnow().isoformat()
        }
        result = await log_event(event)
        # result = {"status": "buffered", "sequence": 42, "buffer_size": 100}
    """
    
    event_type = event.get("event_type", "UNKNOWN")
    
    # Ensure required fields are present (for Supabase NOT NULL constraints)
    if "payload_hash" not in event and "payload" in event:
        event["payload_hash"] = compute_payload_hash(event["payload"])
    
    if "build_id" not in event:
        event["build_id"] = BUILD_ID  # From config (e.g., "dev-local" or GitHub SHA)
    
    # ============================================================
    # Step 1: Write to TEE buffer (CRITICAL - must succeed)
    # ============================================================
    
    try:
        result = await tee_client.append_event(event)
        
        sequence = result.get("sequence")
        buffer_size = result.get("buffer_size", 0)
        overflow_warning = result.get("overflow_warning", False)
        
        logger.info(
            f"✅ Event buffered in TEE: {event_type} "
            f"(seq={sequence}, buffer={buffer_size})"
        )
        
        # Warn if buffer approaching capacity
        if overflow_warning:
            logger.warning(
                f"⚠️ TEE buffer overflow risk! Size: {buffer_size} "
                f"(threshold: 5000 events). Emergency batch may be needed."
            )
    
    except Exception as e:
        logger.error(f"❌ TEE buffer write failed: {event_type} - {e}")
        
        # Fallback: Log to file for recovery
        await _fallback_log_to_file(event, error=str(e))
        
        # CRITICAL: Request must fail if TEE write fails
        # This prevents censorship (cannot accept event and then drop it)
        raise RuntimeError(
            f"Failed to write event to TEE buffer: {e}. "
            f"Event type: {event_type}. "
            f"This is a critical failure - request cannot proceed."
        )
    
    # ============================================================
    # Step 2: Mirror to Supabase (non-critical, best-effort)
    # ============================================================
    
    if supabase:
        try:
            # Extract email_hash from payload or top-level event (for duplicate detection)
            email_hash = None
            payload = event.get("payload")
            if payload and isinstance(payload, dict):
                email_hash = payload.get("email_hash")
            
            # Fallback: check top-level event (used by CONSENSUS_RESULT, etc.)
            if not email_hash:
                email_hash = event.get("email_hash")
            
            # Extract linkedin_combo_hash from payload or top-level event
            # This is for person+company duplicate detection
            linkedin_combo_hash = None
            if payload and isinstance(payload, dict):
                linkedin_combo_hash = payload.get("linkedin_combo_hash")
            
            # Fallback: check top-level event
            if not linkedin_combo_hash:
                linkedin_combo_hash = event.get("linkedin_combo_hash")
            
            # Create Supabase entry with correct column names
            supabase_entry = {
                "event_type": event.get("event_type"),
                "actor_hotkey": event.get("actor_hotkey"),
                "nonce": event.get("nonce"),
                "ts": event.get("ts"),  # Event already uses "ts" key
                "payload_hash": event.get("payload_hash"),
                "build_id": event.get("build_id"),
                "signature": event.get("signature"),
                "payload": payload,
                # TEE metadata
                "tee_sequence": sequence,
                "tee_buffered_at": datetime.utcnow().isoformat(),
                "tee_buffer_size": buffer_size,
                # Email hash for duplicate detection (extracted from payload)
                "email_hash": email_hash,
                # LinkedIn combo hash for person+company duplicate detection
                "linkedin_combo_hash": linkedin_combo_hash
            }
            
            # Remove None values (optional fields)
            supabase_entry = {k: v for k, v in supabase_entry.items() if v is not None}
            
            # Insert into Supabase (for fast queries)
            supabase.table("transparency_log").insert(supabase_entry).execute()
            
            logger.info(f"✅ Event mirrored to Supabase: {event_type}")
        
        except Exception as e:
            # Supabase mirroring failure is NOT critical
            # TEE has canonical copy, Supabase is just a cache
            logger.warning(
                f"⚠️ Failed to mirror to Supabase: {event_type} - {e}. "
                f"Event is safe in TEE buffer (seq={sequence})."
            )
            # DO NOT raise - Supabase is non-authoritative
    else:
        logger.debug(f"⏭️  Supabase mirroring skipped (not configured): {event_type}")
    
    return result


async def _fallback_log_to_file(event: dict, error: str = ""):
    """
    Fallback logging when TEE buffer write fails.
    
    Writes event to local JSON file for manual recovery/investigation.
    Operator should monitor this directory and investigate failures.
    
    This indicates a CRITICAL failure (TEE enclave down or communication failure).
    The gateway should alert operators immediately.
    
    Args:
        event: Event that failed to write to TEE
        error: Error message describing the failure
    """
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        event_type = event.get("event_type", "UNKNOWN")
        filename = f"{timestamp}_{event_type}_TEE_FAILURE.json"
        filepath = FALLBACK_LOG_DIR / filename
        
        fallback_data = {
            "event": event,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
            "reason": "TEE buffer write failed"
        }
        
        with open(filepath, 'w') as f:
            json.dump(fallback_data, f, indent=2, sort_keys=True)
        
        logger.critical(
            f"🚨 CRITICAL: TEE BUFFER WRITE FAILED!\n"
            f"   Event type: {event_type}\n"
            f"   Error: {error}\n"
            f"   Fallback log: {filepath}\n"
            f"   📁 Fallback directory: {FALLBACK_LOG_DIR}\n"
            f"   🚨 Operator action required: Check TEE enclave health"
        )
    
    except Exception as e:
        logger.critical(
            f"❌ CRITICAL: Fallback logging also failed: {e}\n"
            f"   Event data: {json.dumps(event, indent=2, default=str)}"  # Handle datetime objects
        )


# ============================================================================
# DEPRECATED FUNCTIONS (Keep for backward compatibility during migration)
# ============================================================================

async def log_event_arweave_first(event: dict) -> Optional[str]:
    """
    [DEPRECATED] Old Arweave-first logging function.
    
    This function is deprecated in Phase 2.3. Use log_event() instead.
    
    Kept temporarily for backward compatibility during migration.
    Will be removed after all endpoints are updated.
    """
    logger.warning(
        f"⚠️ DEPRECATED: log_event_arweave_first() called. "
        f"Use log_event() instead (TEE-based)."
    )
    
    # For now, redirect to new TEE-based logging
    try:
        result = await log_event(event)
        # Old function returned arweave_tx_id, but we don't have that yet
        # Return sequence number as a placeholder
        return f"tee_seq_{result.get('sequence')}"
    except Exception as e:
        logger.error(f"Failed to log event: {e}")
        return None


# ============================================================================
# TEE BUFFER UTILITIES
# ============================================================================

async def get_canonical_event(
    event_type: str,
    event_id: str,
    timestamp: Optional[str] = None
) -> Optional[Dict]:
    """
    Retrieve canonical (authoritative) copy of an event.
    
    Implements the trust hierarchy from Task 2.4:
    1. If event < 1 hour old: Query TEE buffer (authoritative)
    2. If event > 1 hour old: Should query Arweave (Phase 3 - not yet implemented)
    3. Never trust Supabase alone (it's just a mirror for queries)
    
    Args:
        event_type: Type of event (e.g., "SUBMISSION", "VALIDATION_RESULT")
        event_id: Unique identifier for the event (e.g., lead_id, epoch_id)
        timestamp: Optional ISO8601 timestamp to determine age
    
    Returns:
        Event dict if found, None if not found
        
    Example:
        # Verify miner's submission claim
        event = await get_canonical_event("SUBMISSION", lead_id="abc-123")
        if event:
            print(f"✅ Submission verified: {event}")
        else:
            print(f"❌ No submission found for lead_id=abc-123")
    
    Note:
        - Phase 2.4: Queries TEE buffer only
        - Phase 3 (TODO): Will also query Arweave checkpoints for older events
        - Supabase is NEVER queried by this function (not authoritative)
    """
    logger.info(f"🔍 Searching for canonical event: {event_type}, id={event_id}")
    
    # ============================================================
    # Step 1: Query TEE buffer (authoritative for recent events)
    # ============================================================
    try:
        buffer = await tee_client.get_buffer()
        
        # Search buffer for matching event
        for event in buffer:
            if event.get("event_type") == event_type:
                # Match based on common ID fields
                if (event.get("lead_id") == event_id or 
                    event.get("epoch_id") == event_id or
                    event.get("nonce") == event_id):
                    logger.info(f"✅ Found event in TEE buffer (seq={event.get('sequence')})")
                    return event
        
        logger.info(f"⚠️  Event not found in TEE buffer")
        
    except Exception as e:
        logger.error(f"❌ Failed to query TEE buffer: {e}")
    
    # ============================================================
    # Step 2: Query Arweave checkpoints (Phase 3 - TODO)
    # ============================================================
    # TODO Phase 3: If event not in TEE buffer, download relevant Arweave checkpoint
    # and verify Merkle inclusion
    logger.info(f"📦 Arweave checkpoint query not yet implemented (Phase 3)")
    logger.info(f"   Event may be in Arweave if > 1 hour old")
    
    # ============================================================
    # Step 3: NEVER query Supabase alone (not authoritative)
    # ============================================================
    # Supabase is a convenience cache only. For verification, we MUST use:
    # - TEE buffer (if recent)
    # - Arweave checkpoint (if older)
    logger.warning(f"⚠️  Event not found in authoritative sources (TEE buffer or Arweave)")
    logger.warning(f"   Supabase is NOT queried by this function (non-authoritative)")
    
    return None


async def get_tee_buffer_stats() -> Dict:
    """
    Get current TEE buffer statistics.
    
    Useful for monitoring buffer health and detecting potential issues.
    
    Returns:
        dict: Buffer statistics with:
            - size: Current number of events
            - age_seconds: How long events have been accumulating
            - overflow_risk: Boolean indicating if buffer is approaching capacity
            - next_checkpoint_in_seconds: Estimated time until next hourly batch
    """
    try:
        stats = await tee_client.get_buffer_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get TEE buffer stats: {e}")
        return {
            "error": str(e),
            "size": None,
            "overflow_risk": True  # Assume risk if we can't check
        }


async def get_tee_buffer_size() -> int:
    """
    Get current TEE buffer size (number of events).
    
    Returns:
        int: Number of events in buffer, or -1 if error
    """
    try:
        size = await tee_client.get_buffer_size()
        return size
    except Exception as e:
        logger.error(f"Failed to get TEE buffer size: {e}")
        return -1
