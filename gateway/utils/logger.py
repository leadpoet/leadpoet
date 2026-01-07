"""
TEE-First Event Logging Utility (with Enclave Signing)
=======================================================

This module provides functions to log events with TEE enclave signing.
All events are signed with the gateway's enclave private key and hash-chained
for tamper-evidence.

ARCHITECTURE:
Gateway → sign_event() → Supabase (transparency_log) → Arweave (hourly checkpoint)

CANONICAL LOG_ENTRY FORMAT:
{
    "signed_event": {
        "event_type": "...",
        "timestamp": "2024-01-01T00:00:00Z",  # ONLY timestamp location
        "boot_id": "uuid",
        "monotonic_seq": 12345,
        "prev_event_hash": "abc123...",
        "payload": { ... }  # NO timestamp here!
    },
    "event_hash": "sha256(canonical_json(signed_event)).hex()",
    "enclave_pubkey": "hex",
    "enclave_signature": "hex"
}

Key principles:
1. sign_event() is the SINGLE SOURCE OF TRUTH for timestamps
2. Do NOT add timestamps in payload - they go in signed_event.timestamp
3. log_event() returns the full log_entry dict (not a status dict)
4. Callers can access event_hash from the returned log_entry

Security guarantees:
- Events signed with enclave Ed25519 key (proves gateway origin)
- Hash-chain (prev_event_hash) ensures ordering and completeness
- (boot_id, monotonic_seq) provides additional ordering within boot
- Signature can be verified by auditors using attested enclave pubkey

==============================================================================
TRUST HIERARCHY FOR EVENT VERIFICATION
==============================================================================

1. SIGNED LOG ENTRIES (Authoritative)
   - Location: transparency_log table in Supabase
   - Format: Full log_entry with signed_event, event_hash, signature
   - Verification: Recompute hash, verify Ed25519 signature, check hash-chain

2. ARWEAVE CHECKPOINTS (Authoritative for archived events)
   - Location: Permanent Arweave blockchain storage
   - Contains: Signed checkpoints with last_event_hash for chain verification
   - Usage: Canonical archive for events after hourly batch

3. AUDITOR VERIFICATION:
   - Fetch log entries from transparency_log
   - Verify each entry's signature using attested enclave pubkey
   - Verify hash-chain continuity (prev_event_hash links)
   - Compare with Arweave checkpoints for completeness

==============================================================================
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Enclave signing (CRITICAL: This is the signing authority)
# Try both import paths to support local dev and EC2 deployment
try:
    from gateway.tee.enclave_signer import sign_event, is_keypair_initialized
except ImportError:
    from tee.enclave_signer import sign_event, is_keypair_initialized

from config import BUILD_ID

# Python logging
logger = logging.getLogger(__name__)

# Supabase client accessor (lazily initialized)
def _get_supabase():
    """Get Supabase write client for logging events."""
    try:
        # Try both import paths to support local dev and EC2 deployment
        try:
            from gateway.db.client import get_write_client
        except ImportError:
            from db.client import get_write_client
        return get_write_client()
    except Exception as e:
        logger.warning(f"⚠️  Supabase client unavailable: {e}")
        return None

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


async def log_event(event_or_type, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Log an event to transparency_log.
    
    BACKWARD COMPATIBLE: Accepts BOTH old and new formats:
    
    OLD FORMAT (existing code - NO TEE SIGNING):
        await log_event({"event_type": "X", "actor_hotkey": "...", ...})
        
    NEW FORMAT (TEE signed):
        await log_event("X", {"actor_hotkey": "...", ...})
    
    The function auto-detects which format is being used:
    - If first arg is a dict with "event_type" key → OLD format
    - If first arg is a string → NEW format (TEE signed)
    
    OLD FORMAT BEHAVIOR:
    - Stores event directly to Supabase (no signing)
    - New TEE columns (event_hash, boot_id, etc.) will be NULL
    - Returns dict with "status": "buffered", "sequence": tee_sequence
    
    NEW FORMAT BEHAVIOR:
    - Signs event with enclave Ed25519 key
    - Stores full signed log_entry to Supabase
    - Returns full log_entry with event_hash, signature, etc.
    """
    
    # ============================================================
    # DETECT FORMAT: Old (dict) vs New (event_type, payload)
    # ============================================================
    
    if isinstance(event_or_type, dict):
        # OLD FORMAT: Single dict argument with event_type inside
        return await _log_event_legacy_format(event_or_type)
    elif isinstance(event_or_type, str) and payload is not None:
        # NEW FORMAT: (event_type: str, payload: dict)
        return await _log_event_signed_format(event_or_type, payload)
    else:
        raise ValueError(
            f"Invalid log_event arguments. Use either:\n"
            f"  OLD: log_event({{'event_type': '...', ...}})\n"
            f"  NEW: log_event('EVENT_TYPE', {{...}})"
        )


async def _log_event_legacy_format(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    OLD FORMAT: Log event WITHOUT TEE signing.
    
    This maintains backward compatibility with existing code.
    Events logged this way will have NULL for TEE columns
    (event_hash, boot_id, monotonic_seq, prev_event_hash, enclave_pubkey).
    
    Args:
        event: Full event dict with "event_type" key
        
    Returns:
        Dict with status info (matches old return format)
    """
    event_type = event.get("event_type", "UNKNOWN")
    
    # ============================================================
    # CRITICAL: Auto-compute payload_hash if not provided
    # (This was the original behavior - required for NOT NULL constraint)
    # ============================================================
    if "payload_hash" not in event and "payload" in event:
        event["payload_hash"] = compute_payload_hash(event["payload"])
    
    # ============================================================
    # Store to Supabase (OLD format - no signing)
    # ============================================================
    
    supabase = _get_supabase()
    if supabase:
        try:
            # Extract fields for Supabase columns (OLD schema)
            payload = event.get("payload")
            email_hash = None
            linkedin_combo_hash = None
            
            if payload and isinstance(payload, dict):
                email_hash = payload.get("email_hash")
                linkedin_combo_hash = payload.get("linkedin_combo_hash")
            
            # Fallback: check top-level event
            if not email_hash:
                email_hash = event.get("email_hash")
            if not linkedin_combo_hash:
                linkedin_combo_hash = event.get("linkedin_combo_hash")
            
            # Create Supabase entry (OLD column set)
            supabase_entry = {
                "event_type": event.get("event_type"),
                "actor_hotkey": event.get("actor_hotkey"),
                "nonce": event.get("nonce"),
                "ts": event.get("ts"),
                "payload_hash": event.get("payload_hash"),
                "build_id": event.get("build_id") or BUILD_ID,
                "signature": event.get("signature"),
                "payload": payload,
                "email_hash": email_hash,
                "linkedin_combo_hash": linkedin_combo_hash,
                # TEE columns will be NULL (not set)
            }
            
            # Remove None values
            supabase_entry = {k: v for k, v in supabase_entry.items() if v is not None}
            
            # Insert into Supabase
            supabase.table("transparency_log").insert(supabase_entry).execute()
            
            logger.info(f"✅ Event logged (legacy format): {event_type}")
            
            # Return OLD format response
            return {
                "status": "buffered",
                "sequence": 0,  # No TEE sequence in legacy mode
                "buffer_size": 0,
                "overflow_warning": False,
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to log event: {event_type} - {e}")
            await _fallback_log_to_file(event, error=str(e))
            raise RuntimeError(f"Failed to log event to Supabase: {e}")
    else:
        logger.warning(f"⚠️ Supabase not configured - event not stored: {event_type}")
        return {"status": "not_stored", "sequence": 0}


async def _log_event_signed_format(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW FORMAT: Log event WITH TEE signing.
    
    Signs the event with the enclave's Ed25519 key and stores
    the full log_entry (with signature) to Supabase.
    
    Args:
        event_type: Type of event (e.g., "WEIGHT_SUBMISSION")
        payload: Event-specific data (NO timestamp - sign_event adds it)
    
    Returns:
        Full log_entry with signed_event, event_hash, enclave_signature
    """
    
    # ============================================================
    # Step 1: Sign event with enclave key
    # ============================================================
    
    try:
        # sign_event() is the SINGLE source of timestamp truth
        log_entry = sign_event(event_type, payload)
        
        # Extract metadata for logging
        signed_event = log_entry["signed_event"]
        event_hash = log_entry["event_hash"]
        monotonic_seq = signed_event["monotonic_seq"]
        
        logger.info(
            f"✅ Event signed: {event_type} "
            f"(seq={monotonic_seq}, hash={event_hash[:16]}...)"
        )
    
    except Exception as e:
        logger.error(f"❌ Event signing failed: {event_type} - {e}")
        await _fallback_log_to_file({"event_type": event_type, "payload": payload}, error=str(e))
        raise RuntimeError(
            f"Failed to sign event: {e}. Event type: {event_type}. "
            f"This is a critical failure - request cannot proceed."
        )
    
    # ============================================================
    # Step 2: Store to Supabase (NEW format - with TEE columns)
    # ============================================================
    
    supabase = _get_supabase()
    if supabase:
        try:
            # Extract optional fields for indexing
            email_hash = payload.get("email_hash") if isinstance(payload, dict) else None
            linkedin_combo_hash = payload.get("linkedin_combo_hash") if isinstance(payload, dict) else None
            actor_hotkey = payload.get("actor_hotkey") or payload.get("validator_hotkey") or payload.get("miner_hotkey")
            
            # Create Supabase entry (NEW column set with TEE fields)
            # CRITICAL: Keep "payload" as original payload for dashboard compatibility
            # Store full signed envelope in "signed_log_entry" for auditor verification
            supabase_entry = {
                "event_type": event_type,
                "payload": payload,  # ORIGINAL payload (unchanged for dashboards)
                "signed_log_entry": log_entry,  # Full signed envelope (for auditors)
                "event_hash": event_hash,
                "enclave_pubkey": log_entry["enclave_pubkey"],
                "boot_id": signed_event["boot_id"],
                "monotonic_seq": monotonic_seq,
                "prev_event_hash": signed_event["prev_event_hash"],
                "created_at": signed_event["timestamp"],
                "actor_hotkey": actor_hotkey,
                "email_hash": email_hash,
                "linkedin_combo_hash": linkedin_combo_hash,
                "build_id": BUILD_ID,
            }
            
            # Remove None values
            supabase_entry = {k: v for k, v in supabase_entry.items() if v is not None}
            
            # Insert into Supabase
            supabase.table("transparency_log").insert(supabase_entry).execute()
            
            logger.info(f"✅ Event stored (signed): {event_type} (hash={event_hash[:16]}...)")
        
        except Exception as e:
            logger.error(f"❌ Failed to store signed event: {event_type} - {e}")
            await _fallback_log_to_file(log_entry, error=str(e))
            raise RuntimeError(
                f"Failed to store signed event to Supabase: {e}. "
                f"Event type: {event_type}."
            )
    else:
        logger.warning(f"⚠️ Supabase not configured - signed event not stored: {event_type}")
    
    # Return the full log_entry
    return log_entry


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
    
    This function is deprecated. Use log_event(event_type, payload) instead.
    """
    logger.warning(
        f"⚠️ DEPRECATED: log_event_arweave_first() called. "
        f"Use log_event(event_type, payload) instead."
    )
    
    try:
        event_type = event.pop("event_type", "UNKNOWN")
        log_entry = await log_event(event_type, event)
        return log_entry.get("event_hash")
    except Exception as e:
        logger.error(f"Failed to log event: {e}")
        return None


# ============================================================================
# TRANSPARENCY LOG QUERIES
# ============================================================================

async def get_log_entry_by_hash(event_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a signed log entry by its event_hash.
    
    This is the primary method for auditors to fetch and verify events.
    
    Args:
        event_hash: The SHA256 hash of the signed_event
    
    Returns:
        The full log_entry dict if found, None otherwise
    """
    try:
        # Try both import paths to support local dev and EC2 deployment
        try:
            from gateway.db.client import get_read_client
        except ImportError:
            from db.client import get_read_client
        read_client = get_read_client()
    except Exception as e:
        logger.error(f"Supabase not configured - cannot query log entries: {e}")
        return None
    
    try:
        result = read_client.table("transparency_log") \
            .select("payload") \
            .eq("event_hash", event_hash) \
            .limit(1) \
            .execute()
        
        if result.data:
            return result.data[0]["payload"]
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch log entry: {e}")
    return None


async def get_log_entries_for_epoch(
    netuid: int, 
    epoch_id: int, 
    event_type: Optional[str] = None
) -> list:
    """
    Retrieve all signed log entries for a specific epoch.
    
    Args:
        netuid: Subnet ID
        epoch_id: Epoch identifier
        event_type: Optional filter by event type
    
    Returns:
        List of log_entry dicts
    """
    try:
        # Try both import paths to support local dev and EC2 deployment
        try:
            from gateway.db.client import get_read_client
        except ImportError:
            from db.client import get_read_client
        read_client = get_read_client()
    except Exception as e:
        logger.error(f"Supabase not configured - cannot query log entries: {e}")
        return []
    
    try:
        query = read_client.table("transparency_log") \
            .select("payload") \
            .order("monotonic_seq", desc=False)
        
        if event_type:
            query = query.eq("event_type", event_type)
        
        # Filter by epoch_id in the payload
        # Note: This requires the payload to have netuid and epoch_id
        result = query.execute()
        
        # Filter results by netuid and epoch_id in payload
        entries = []
        for row in result.data:
            payload = row.get("payload", {})
            signed_event = payload.get("signed_event", {})
            event_payload = signed_event.get("payload", {})
            
            if (event_payload.get("netuid") == netuid and 
                event_payload.get("epoch_id") == epoch_id):
                entries.append(payload)
        
        return entries
        
    except Exception as e:
        logger.error(f"Failed to fetch log entries for epoch: {e}")
        return []


def get_signer_info() -> Dict[str, Any]:
    """
    Get information about the current enclave signer state.
    
    Returns:
        Dict with signer state information
    """
    # Try both import paths to support local dev and EC2 deployment
    try:
        from gateway.tee.enclave_signer import get_signer_state
    except ImportError:
        from tee.enclave_signer import get_signer_state
    return get_signer_state()