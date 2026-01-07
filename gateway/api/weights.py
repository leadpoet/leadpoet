"""
Weight Submission Endpoint for TEE-Verified Validators
======================================================

This endpoint accepts weight submissions from primary validators running in TEE.
The gateway acts as a VERIFIER, NOT an ORACLE - it only verifies signatures and
records authenticated data.

Verification Order (MANDATORY):
0. Basic invariants (lengths, sorted UIDs, no zeros, valid u16 range)
1. Authorization: validator_hotkey in PRIMARY_VALIDATOR_HOTKEYS
2. Single-source-of-truth: Reject duplicate (netuid, epoch_id, validator_hotkey)
3. Epoch freshness: Validate block against gateway-observed chain
4. Attestation: Nitro verification + epoch_id in user_data (fail-closed in production)
5. Hotkey binding: sr25519 signature over binding_message
6. Ed25519 signature: Over digest BYTES of weights_hash
7. Recompute hash: bundle_weights_hash must match submission.weights_hash

Security:
- FAIL-CLOSED: Production MUST verify all attestations
- Gateway does NOT interpret or decide validation results
- All events are signed and hash-chained for auditor verification
"""

import os
import base64
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Set

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Canonical imports (MUST use shared module)
from leadpoet_canonical.weights import bundle_weights_hash, compare_weights_hash
from leadpoet_canonical.chain import normalize_chain_weights
from leadpoet_canonical.binding import verify_binding_message
from leadpoet_canonical.timestamps import canonical_timestamp
from leadpoet_canonical.constants import EPOCH_LENGTH, WEIGHT_SUBMISSION_BLOCK

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/weights", tags=["weights"])

# ============================================================================
# Configuration
# ============================================================================

MAX_BLOCK_DRIFT = 30  # Max allowed drift from gateway-observed block

# Build identifier for transparency log
BUILD_ID = os.environ.get("BUILD_ID", "production-gateway-tee")

# PCR0 is the ROOT OF TRUST - code_hash in user_data is INFORMATIONAL only
# The verify_nitro_attestation_full function checks PCR0 against the allowlist
# which is fetched from GitHub automatically

# Filter empty strings - .split(",") on "" produces [""]
_primary_hotkeys_str = os.environ.get("PRIMARY_VALIDATOR_HOTKEYS", "")
PRIMARY_VALIDATOR_HOTKEYS: Set[str] = {x.strip() for x in _primary_hotkeys_str.split(",") if x.strip()}

# Allowed netuids (empty = allow all in dev mode)
_allowed_netuids_str = os.environ.get("ALLOWED_NETUIDS", "")
ALLOWED_NETUIDS: Set[int] = {int(x) for x in _allowed_netuids_str.split(",") if x.strip().isdigit()}

# Chain endpoint for binding message verification
EXPECTED_CHAIN = os.environ.get("EXPECTED_CHAIN", "wss://entrypoint-finney.opentensor.ai:443")

# Subtensor for block validation (lazily initialized)
_subtensor = None


def get_subtensor():
    """Get or create subtensor instance for chain queries."""
    global _subtensor
    if _subtensor is None:
        import bittensor as bt
        _subtensor = bt.subtensor()
    return _subtensor


# ============================================================================
# Models
# ============================================================================

class WeightSubmission(BaseModel):
    """
    Weight submission from validator TEE.
    
    MUST match Canonical Specifications exactly.
    NO floats, NO hotkeys in weights - only UIDs and u16.
    """
    netuid: int
    epoch_id: int
    block: int  # Block when weights were computed
    
    # Weights as parallel arrays (compact, matches bundle table)
    uids: List[int]  # Sorted ascending
    weights_u16: List[int]  # Corresponding u16 weights [0-65535]
    
    # Verification (from bundle_weights_hash - includes block)
    weights_hash: str  # SHA256 digest hex
    
    # Validator identity
    validator_hotkey: str  # ss58 address
    validator_enclave_pubkey: str  # Ed25519 hex
    validator_signature: str  # Ed25519 over digest BYTES (not hex string)
    
    # Attestation (with epoch_id in user_data for freshness)
    validator_attestation_b64: str
    validator_code_hash: str
    
    # Hotkey binding (proves enclave authorized by this hotkey)
    binding_message: str  # LEADPOET_VALIDATOR_BINDING|netuid=...|...
    validator_hotkey_signature: str  # sr25519 over binding_message


class WeightSubmissionResponse(BaseModel):
    """Response after weight submission."""
    success: bool
    epoch_id: int
    weights_count: int
    message: str
    weight_submission_event_hash: Optional[str] = None


# ============================================================================
# Verification Helpers
# ============================================================================

def verify_validator_attestation(
    attestation_b64: str,
    expected_pubkey: str,
    expected_epoch_id: int,
) -> tuple:
    """
    Verify the validator's AWS Nitro attestation document.
    
    FAIL-CLOSED: NO DEV MODE BYPASS. Attestation verification is ALWAYS required.
    
    PCR0 (enclave image hash) is the ROOT OF TRUST:
    - PCR0 is checked against the allowlist (fetched from GitHub)
    - code_hash in user_data is INFORMATIONAL only (do NOT trust it alone)
    - A malicious enclave could claim any code_hash, but cannot fake PCR0
    
    Returns:
        (valid: bool, extracted_data: dict)
    """
    try:
        # FAIL-CLOSED: Full Nitro verification ALWAYS required
        # NO dev mode bypass - attestation is critical security
        from leadpoet_canonical.nitro import verify_nitro_attestation_full
        
        # PCR0 is verified against the allowlist which is:
        # 1. Fetched from GitHub automatically (cached 5 minutes)
        # 2. Contains only PCR0 values for approved enclave builds
        # 3. The validator CANNOT fake PCR0 - it's inside the AWS-signed attestation
        valid, data = verify_nitro_attestation_full(
            attestation_b64=attestation_b64,
            expected_pubkey=expected_pubkey,
            expected_purpose="validator_weights",
            expected_epoch_id=expected_epoch_id,
            role="validator",  # Uses ALLOWED_VALIDATOR_PCR0_VALUES
        )
        
        if valid:
            logger.info(f"[ATTESTATION] ✅ Full Nitro verification passed")
            logger.info(f"[ATTESTATION]    PCR0: {data.get('pcr0', 'N/A')[:32]}...")
            logger.info(f"[ATTESTATION]    Steps: {data.get('verification_steps', [])[-1]}")
        else:
            logger.error(f"[ATTESTATION] ❌ Verification failed: {data.get('error', 'Unknown')}")
        
        return valid, data
        
    except ImportError as e:
        # If nitro module not available, FAIL (don't bypass)
        logger.error(f"[ATTESTATION] ❌ CRITICAL: Nitro verification module not available: {e}")
        return False, {"error": f"Nitro verification unavailable: {e}"}
    except Exception as e:
        logger.error(f"[ATTESTATION] ❌ Failed: {e}")
        return False, {"error": str(e)}


def verify_ed25519_signature(digest_bytes: bytes, signature_hex: str, pubkey_hex: str) -> bool:
    """
    Verify an Ed25519 signature over raw digest bytes.
    
    CANONICAL RULE: Ed25519 signatures are ALWAYS over SHA256 digest BYTES (32 bytes),
    never over hex strings. Transport signatures as hex for JSON compatibility.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pk.verify(bytes.fromhex(signature_hex), digest_bytes)
        return True
    except Exception as e:
        logger.error(f"[VERIFY] ❌ Ed25519 verification failed: {e}")
        return False


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/submit")
async def submit_weights(submission: WeightSubmission) -> WeightSubmissionResponse:
    """
    Submit validated weights from the primary validator TEE.
    
    GATEWAY IS A VERIFIER, NOT AN ORACLE:
    The gateway does not interpret, modify, or decide validation results.
    It only verifies enclave signatures and records authenticated data.
    """
    from gateway.utils.logger import log_event
    from gateway.db.client import get_read_client, get_write_client
    
    print(f"\n{'='*60}")
    print(f"📥 WEIGHT SUBMISSION: epoch={submission.epoch_id}, validator={submission.validator_hotkey[:16]}...")
    print(f"{'='*60}")
    
    # --- Step 0: Basic Invariant Checks ---
    print(f"   Step 0: Checking invariants...")
    
    # 0a. Length match
    if len(submission.uids) != len(submission.weights_u16):
        raise HTTPException(
            status_code=400, 
            detail=f"uids/weights_u16 length mismatch: {len(submission.uids)} vs {len(submission.weights_u16)}"
        )
    
    # 0b. UIDs sorted ascending
    if submission.uids != sorted(submission.uids):
        raise HTTPException(status_code=400, detail="uids must be sorted ascending")
    
    # 0c. No duplicate UIDs
    if len(submission.uids) != len(set(submission.uids)):
        raise HTTPException(status_code=400, detail="Duplicate UIDs detected")
    
    # 0d. All weights in valid u16 range (1, 65535] - sparse requirement (no zeros!)
    for w in submission.weights_u16:
        if not (1 <= w <= 65535):
            raise HTTPException(
                status_code=400, 
                detail=f"Weight {w} invalid. Sparse weights require all values in [1, 65535]."
            )
    
    # 0e. UIDs must be strictly increasing
    for i in range(1, len(submission.uids)):
        if submission.uids[i] <= submission.uids[i-1]:
            raise HTTPException(status_code=400, detail="UIDs must be strictly increasing")
    
    # 0f. netuid sanity check
    if ALLOWED_NETUIDS and submission.netuid not in ALLOWED_NETUIDS:
        raise HTTPException(status_code=400, detail=f"Invalid netuid: {submission.netuid}")
    
    print(f"   ✅ Invariants OK: {len(submission.uids)} weights")
    
    # --- Step 1: Authorization ---
    print(f"   Step 1: Checking authorization...")
    
    if not PRIMARY_VALIDATOR_HOTKEYS:
        logger.warning("⚠️ PRIMARY_VALIDATOR_HOTKEYS not configured - allowing all validators (dev mode)")
    elif submission.validator_hotkey not in PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(status_code=403, detail="Unauthorized validator hotkey")
    
    print(f"   ✅ Authorization OK")
    
    # --- Step 2: Single-source-of-truth (first valid wins) ---
    print(f"   Step 2: Checking for duplicates...")
    
    read_client = get_read_client()
    existing = read_client.table("published_weight_bundles") \
        .select("id") \
        .eq("netuid", submission.netuid) \
        .eq("epoch_id", submission.epoch_id) \
        .eq("validator_hotkey", submission.validator_hotkey) \
        .execute()
    
    if existing.data:
        dup_payload = {
            "epoch_id": submission.epoch_id,
            "netuid": submission.netuid,
            "validator_hotkey": submission.validator_hotkey,
        }
        dup_payload_hash = hashlib.sha256(
            json.dumps(dup_payload, sort_keys=True).encode()
        ).hexdigest()
        
        await log_event({
            "event_type": "WEIGHT_SUBMISSION_REJECTED_DUPLICATE",
            "actor_hotkey": submission.validator_hotkey,
            "nonce": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "payload_hash": dup_payload_hash,
            "build_id": BUILD_ID,
            "signature": "validator",  # Validator-initiated event
            "payload": dup_payload,
        })
        raise HTTPException(status_code=409, detail="Duplicate submission for this epoch")
    
    print(f"   ✅ No duplicate found")
    
    # --- Step 3: Epoch freshness ---
    print(f"   Step 3: Validating epoch freshness...")
    
    subtensor = get_subtensor()
    gateway_block = subtensor.get_current_block()
    gateway_epoch = gateway_block // EPOCH_LENGTH
    
    # Enforce block drift window
    if abs(gateway_block - submission.block) > MAX_BLOCK_DRIFT:
        raise HTTPException(
            status_code=400, 
            detail=f"Block drift too large: submitted {submission.block}, gateway sees {gateway_block}"
        )
    
    # Boundary-safe epoch validation
    block_in_gateway_epoch = gateway_block % EPOCH_LENGTH
    if block_in_gateway_epoch < 30:  # Early in epoch - allow previous epoch too
        valid_epochs = {gateway_epoch, gateway_epoch - 1}
    else:
        valid_epochs = {gateway_epoch}
    
    if submission.epoch_id not in valid_epochs:
        raise HTTPException(
            status_code=400, 
            detail=f"epoch_id {submission.epoch_id} not in valid range {valid_epochs}"
        )
    
    # Validate submitted epoch_id matches block-derived epoch
    expected_epoch_from_block = submission.block // EPOCH_LENGTH
    if submission.epoch_id != expected_epoch_from_block:
        raise HTTPException(
            status_code=400, 
            detail=f"epoch_id {submission.epoch_id} != block-derived {expected_epoch_from_block}"
        )
    
    # Validate submission window
    block_in_submission_epoch = submission.block % EPOCH_LENGTH
    if block_in_submission_epoch < WEIGHT_SUBMISSION_BLOCK - 15:
        raise HTTPException(
            status_code=400, 
            detail=f"Block {block_in_submission_epoch} is too early for weight submission"
        )
    
    print(f"   ✅ Epoch freshness OK: epoch={submission.epoch_id}, block={submission.block}")
    
    # --- Step 4: Attestation verification (AWS Nitro authenticity) ---
    print(f"   Step 4: Verifying Nitro attestation...")
    
    # VERIFICATION MODEL:
    # 1. Gateway verifies AWS certificate chain → proves genuine Nitro enclave
    # 2. Gateway verifies COSE signature → proves attestation untampered
    # 3. Gateway extracts PCR0 → STORED for auditor verification
    # 4. Auditors later verify: PCR0 matches code on GitHub
    #
    # This allows automatic workflow: code change → push → validator restarts → submits
    # Auditors can independently verify the validator ran the published code
    attestation_valid, attestation_data = verify_validator_attestation(
        attestation_b64=submission.validator_attestation_b64,
        expected_pubkey=submission.validator_enclave_pubkey,
        expected_epoch_id=submission.epoch_id,
    )
    if not attestation_valid:
        error_detail = attestation_data.get("error", "Unknown attestation error")
        print(f"   ❌ Attestation verification failed: {error_detail}")
        raise HTTPException(
            status_code=403, 
            detail=f"Invalid validator attestation: {error_detail}"
        )
    
    # Extract verified PCR0 (stored in bundle for auditor verification)
    verified_pcr0 = attestation_data.get("pcr0", "N/A")
    pcr0_mode = attestation_data.get("pcr0_verification_mode", "allowlist")
    print(f"   ✅ Nitro attestation OK")
    print(f"      PCR0: {verified_pcr0[:32]}...")
    print(f"      Mode: {pcr0_mode} (auditors verify against GitHub)")
    
    # --- Step 5: Hotkey binding verification ---
    print(f"   Step 5: Verifying hotkey binding...")
    
    if not verify_binding_message(
        submission.binding_message,
        submission.validator_hotkey_signature,
        submission.validator_hotkey,
        expected_netuid=submission.netuid,
        expected_chain=EXPECTED_CHAIN,
        expected_enclave_pubkey=submission.validator_enclave_pubkey,
        expected_code_hash=submission.validator_code_hash,
    ):
        raise HTTPException(status_code=403, detail="Invalid hotkey binding")
    
    print(f"   ✅ Hotkey binding OK")
    
    # --- Step 6: Ed25519 signature verification ---
    print(f"   Step 6: Verifying Ed25519 signature...")
    
    digest_bytes = bytes.fromhex(submission.weights_hash)
    if not verify_ed25519_signature(
        digest_bytes,
        submission.validator_signature,
        submission.validator_enclave_pubkey,
    ):
        raise HTTPException(status_code=401, detail="Invalid validator signature")
    
    print(f"   ✅ Ed25519 signature OK")
    
    # --- Step 7: Recompute and verify weights_hash ---
    print(f"   Step 7: Verifying weights hash...")
    
    weights_pairs = list(zip(submission.uids, submission.weights_u16))
    recomputed_hash = bundle_weights_hash(
        submission.netuid, submission.epoch_id, submission.block, weights_pairs
    )
    if recomputed_hash != submission.weights_hash:
        raise HTTPException(status_code=400, detail="weights_hash does not match payload")
    
    print(f"   ✅ Weights hash OK: {submission.weights_hash[:16]}...")
    
    # --- All checks passed: Store bundle ---
    print(f"\n   📦 All checks passed - storing bundle...")
    
    # Capture chain snapshot for equivocation detection
    chain_snapshot_block = None
    chain_snapshot_compare_hash = None
    try:
        metagraph = subtensor.metagraph(netuid=submission.netuid)
        primary_uid = None
        for uid, hk in enumerate(metagraph.hotkeys):
            if hk == submission.validator_hotkey:
                primary_uid = uid
                break
        
        if primary_uid is not None:
            chain_snapshot_block = subtensor.get_current_block()
            chain_weights = subtensor.weights(netuid=submission.netuid, uid=primary_uid)
            
            if chain_weights:
                chain_pairs = normalize_chain_weights(chain_weights)
                chain_snapshot_compare_hash = compare_weights_hash(
                    submission.netuid, submission.epoch_id, chain_pairs
                )
                print(f"   📸 Chain snapshot captured at block {chain_snapshot_block}")
    except Exception as e:
        logger.warning(f"[SNAPSHOT] ⚠️ Could not capture chain snapshot: {e}")
    
    # Log event FIRST to get event_hash
    submission_payload = {
        "epoch_id": submission.epoch_id,
        "netuid": submission.netuid,
        "block": submission.block,
        "weights_hash": submission.weights_hash,
        "validator_hotkey": submission.validator_hotkey,
        "weights_count": len(submission.uids),
        "chain_snapshot_block": chain_snapshot_block,
        "chain_snapshot_compare_hash": chain_snapshot_compare_hash,
    }
    submission_payload_hash = hashlib.sha256(
        json.dumps(submission_payload, sort_keys=True).encode()
    ).hexdigest()
    
    log_entry = await log_event({
        "event_type": "WEIGHT_SUBMISSION",
        "actor_hotkey": submission.validator_hotkey,
        "nonce": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat(),
        "payload_hash": submission_payload_hash,
        "build_id": BUILD_ID,
        "signature": submission.validator_signature,  # Use the actual validator signature
        "payload": submission_payload,
    })
    weight_submission_event_hash = log_entry.get("event_hash")
    
    # Store bundle (including PCR0 for auditor verification)
    bundle_data = {
        "netuid": submission.netuid,
        "epoch_id": submission.epoch_id,
        "block": submission.block,
        "uids": submission.uids,
        "weights_u16": submission.weights_u16,
        "weights_hash": submission.weights_hash,
        "validator_hotkey": submission.validator_hotkey,
        "validator_enclave_pubkey": submission.validator_enclave_pubkey,
        "validator_signature": submission.validator_signature,
        "validator_attestation_b64": submission.validator_attestation_b64,
        "validator_code_hash": submission.validator_code_hash,
        "validator_pcr0": verified_pcr0,  # Extracted from AWS-signed attestation for auditor verification
        "chain_snapshot_block": chain_snapshot_block,
        "chain_snapshot_compare_hash": chain_snapshot_compare_hash,
        "weight_submission_event_hash": weight_submission_event_hash,
    }
    
    write_client = get_write_client()
    try:
        write_client.table("published_weight_bundles").insert(bundle_data).execute()
    except Exception as e:
        # Handle UNIQUE constraint violation (race condition)
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            race_payload = {
                "epoch_id": submission.epoch_id,
                "netuid": submission.netuid,
                "validator_hotkey": submission.validator_hotkey,
                "reason": "UNIQUE constraint violation (concurrent submission)",
            }
            race_payload_hash = hashlib.sha256(
                json.dumps(race_payload, sort_keys=True).encode()
            ).hexdigest()
            
            await log_event({
                "event_type": "WEIGHT_SUBMISSION_REJECTED_DUPLICATE",
                "actor_hotkey": submission.validator_hotkey,
                "nonce": str(uuid.uuid4()),
                "ts": datetime.utcnow().isoformat(),
                "payload_hash": race_payload_hash,
                "build_id": BUILD_ID,
                "signature": "validator",  # Validator-initiated event
                "payload": race_payload,
            })
            raise HTTPException(status_code=409, detail="Duplicate submission (concurrent race)")
        raise
    
    print(f"   ✅ Bundle stored successfully")
    print(f"   📝 Event hash: {weight_submission_event_hash}")
    print(f"{'='*60}\n")
    
    return WeightSubmissionResponse(
        success=True,
        epoch_id=submission.epoch_id,
        weights_count=len(submission.uids),
        message="Weights accepted and logged",
        weight_submission_event_hash=weight_submission_event_hash,
    )


@router.get("/latest/{netuid}/{epoch_id}")
async def get_latest_weights(netuid: int, epoch_id: int) -> dict:
    """
    Get the latest published weights bundle for a specific netuid and epoch.
    
    USES ANON KEY FOR READS (not service role).
    Returns the complete bundle needed for auditor verification.
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    result = read_client.table("published_weight_bundles") \
        .select("*") \
        .eq("netuid", netuid) \
        .eq("epoch_id", epoch_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No weights for epoch {epoch_id}")
    
    bundle = result.data[0]
    
    return {
        "netuid": bundle["netuid"],
        "epoch_id": bundle["epoch_id"],
        "block": bundle["block"],
        "uids": bundle["uids"],
        "weights_u16": bundle["weights_u16"],
        "weights_hash": bundle["weights_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "validator_enclave_pubkey": bundle["validator_enclave_pubkey"],
        "validator_signature": bundle["validator_signature"],
        "validator_attestation_b64": bundle["validator_attestation_b64"],
        "validator_code_hash": bundle["validator_code_hash"],
        "chain_snapshot_block": bundle.get("chain_snapshot_block"),
        "chain_snapshot_compare_hash": bundle.get("chain_snapshot_compare_hash"),
        "weight_submission_event_hash": bundle.get("weight_submission_event_hash"),
    }


@router.get("/current/{netuid}")
async def get_current_weights(netuid: int) -> dict:
    """
    Get the most recently published weights bundle for a netuid.
    
    Use this to find the latest epoch without knowing the epoch_id.
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    result = read_client.table("published_weight_bundles") \
        .select("*") \
        .eq("netuid", netuid) \
        .order("epoch_id", desc=True) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No weights for netuid {netuid}")
    
    bundle = result.data[0]
    
    return {
        "netuid": bundle["netuid"],
        "epoch_id": bundle["epoch_id"],
        "block": bundle["block"],
        "uids": bundle["uids"],
        "weights_u16": bundle["weights_u16"],
        "weights_hash": bundle["weights_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "validator_enclave_pubkey": bundle["validator_enclave_pubkey"],
        "validator_signature": bundle["validator_signature"],
        "validator_attestation_b64": bundle["validator_attestation_b64"],
        "validator_code_hash": bundle["validator_code_hash"],
        "chain_snapshot_block": bundle.get("chain_snapshot_block"),
        "chain_snapshot_compare_hash": bundle.get("chain_snapshot_compare_hash"),
        "weight_submission_event_hash": bundle.get("weight_submission_event_hash"),
    }


# ============================================================================
# Transparency Log Endpoints (for auditor verification)
# ============================================================================

@router.get("/transparency/event/{event_hash}")
async def get_transparency_event(event_hash: str) -> dict:
    """
    Fetch a signed event from the transparency log by its event_hash.
    
    CRITICAL FOR AUDITORS: This endpoint allows independent verification of:
    - Event authenticity (verify enclave signature)
    - Hash-chain integrity (check prev_event_hash links)
    - Event contents (access signed payload)
    
    The returned log_entry contains:
    - signed_event: {event_type, timestamp, boot_id, monotonic_seq, prev_event_hash, payload}
    - event_hash: SHA256 of canonical signed_event
    - enclave_pubkey: Gateway pubkey that signed this event
    - enclave_signature: Ed25519 signature over event_hash
    
    Auditors MUST:
    1. Verify enclave_pubkey matches attested gateway pubkey
    2. Recompute event_hash from signed_event
    3. Verify enclave_signature over event_hash
    4. Check prev_event_hash for chain continuity
    
    Returns:
        Full log_entry object for verification
        
    Raises:
        404: Event not found
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    # Query by event_hash (unique identifier)
    result = read_client.table("transparency_log") \
        .select("payload") \
        .eq("event_hash", event_hash) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(
            status_code=404, 
            detail=f"Event not found: {event_hash[:16]}..."
        )
    
    # The payload column contains the full log_entry (signed_event + signature)
    log_entry = result.data[0]["payload"]
    
    # Validate structure before returning
    if not log_entry or not isinstance(log_entry, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid log entry format in database"
        )
    
    # For old (legacy) events, payload might not have signed_event structure
    # Return as-is and let auditor handle format differences
    return log_entry


@router.get("/transparency/events/range")
async def get_transparency_events_range(
    start_seq: int = 0,
    limit: int = 100,
    boot_id: Optional[str] = None,
) -> dict:
    """
    Fetch a range of events from the transparency log for chain verification.
    
    This endpoint supports auditors who need to verify hash-chain continuity
    across multiple events. Events are returned in monotonic_seq order.
    
    Args:
        start_seq: Starting monotonic sequence number (inclusive)
        limit: Maximum events to return (max 1000)
        boot_id: Optional filter by boot session
        
    Returns:
        {
            "events": [log_entry, ...],
            "count": int,
            "has_more": bool
        }
    """
    from gateway.db.client import get_read_client
    
    # Cap limit to prevent abuse
    limit = min(limit, 1000)
    
    read_client = get_read_client()
    
    query = read_client.table("transparency_log") \
        .select("payload, event_hash, monotonic_seq, boot_id") \
        .gte("monotonic_seq", start_seq) \
        .order("monotonic_seq", desc=False) \
        .limit(limit + 1)  # +1 to check if there's more
    
    if boot_id:
        query = query.eq("boot_id", boot_id)
    
    result = query.execute()
    
    events = result.data[:limit] if result.data else []
    has_more = len(result.data) > limit if result.data else False
    
    return {
        "events": [
            {
                "event_hash": e.get("event_hash"),
                "monotonic_seq": e.get("monotonic_seq"),
                "boot_id": e.get("boot_id"),
                "log_entry": e.get("payload"),
            }
            for e in events
        ],
        "count": len(events),
        "has_more": has_more,
    }

