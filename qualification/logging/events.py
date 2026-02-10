"""
Qualification System: Event Logging

Phase 7.1 from tasks10.md

This module provides event logging for the Lead Qualification Agent competition.
It integrates with the EXISTING transparency_log table and uses the same
signed envelope format as sourcing events.

Event Types:
- MODEL_SUBMITTED - When a miner submits a model for evaluation
- EVALUATION_COMPLETE - When model evaluation finishes
- CHAMPION_SELECTED - When champion selection runs at epoch end
- EMISSIONS_DISTRIBUTED - When 5% emissions are distributed/burned

The events are signed using the existing TEE enclave signer and stored
in the transparency_log table with the same schema as sourcing events.

CRITICAL: This uses EXISTING infrastructure. Do NOT create duplicate
logging systems or modify the existing enclave_signer behavior.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Event Type Constants
# =============================================================================

EVENT_MODEL_SUBMITTED = "MODEL_SUBMITTED"
EVENT_EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
EVENT_CHAMPION_SELECTED = "CHAMPION_SELECTED"
EVENT_EMISSIONS_DISTRIBUTED = "EMISSIONS_DISTRIBUTED"

# All qualification event types
QUALIFICATION_EVENT_TYPES = [
    EVENT_MODEL_SUBMITTED,
    EVENT_EVALUATION_COMPLETE,
    EVENT_CHAMPION_SELECTED,
    EVENT_EMISSIONS_DISTRIBUTED,
]


# =============================================================================
# Payload Models
# =============================================================================

class ModelSubmittedPayload(BaseModel):
    """Payload for MODEL_SUBMITTED event."""
    model_id: str = Field(..., description="UUID of the submitted model")
    miner_hotkey: str = Field(..., description="SS58 address of submitting miner")
    repo_url: str = Field(..., description="Git repository URL")
    commit_hash: str = Field(..., description="Git commit hash")
    code_hash: str = Field(..., description="SHA256 of model code tarball")
    payment_verified: bool = Field(..., description="Whether payment was verified")
    submission_cost_usd: float = Field(..., description="Submission cost in USD")
    created_epoch: int = Field(..., description="Bittensor epoch at submission")


class EvaluationCompletePayload(BaseModel):
    """Payload for EVALUATION_COMPLETE event."""
    evaluation_id: str = Field(..., description="UUID of the evaluation")
    model_id: str = Field(..., description="UUID of the model")
    miner_hotkey: str = Field(..., description="SS58 address of model owner")
    set_id: int = Field(..., description="Evaluation set ID")
    final_score: float = Field(..., description="Total score achieved")
    screening_1_score: Optional[float] = Field(None, description="Screening 1 score")
    screening_2_score: Optional[float] = Field(None, description="Screening 2 score")
    final_benchmark_score: Optional[float] = Field(None, description="Final benchmark score")
    total_cost_usd: float = Field(..., description="Total API cost incurred")
    icps_evaluated: int = Field(..., description="Number of ICPs evaluated")
    status: str = Field(..., description="Final status (finished, failed_screening_1, etc.)")
    db_hash: str = Field(..., description="SHA256 hash of lead_ids for audit")
    completed_epoch: int = Field(..., description="Bittensor epoch at completion")
    # NEW FIELDS for verifiability and auditability
    icp_set_hash: str = Field(..., description="SHA256 hash of ICPs used for verifiability")
    top_10_leads: List[Dict[str, Any]] = Field(default_factory=list, description="Top 10 scoring leads (PII redacted)")
    bottom_10_leads: List[Dict[str, Any]] = Field(default_factory=list, description="Bottom 10 scoring leads (PII redacted)")


class ChampionSelectedPayload(BaseModel):
    """Payload for CHAMPION_SELECTED event."""
    epoch: int = Field(..., description="Bittensor epoch")
    set_id: int = Field(..., description="Evaluation set ID")
    champion_model_id: str = Field(..., description="UUID of champion model")
    champion_hotkey: str = Field(..., description="SS58 address of champion")
    champion_score: float = Field(..., description="Champion's score")
    previous_champion_id: Optional[str] = Field(None, description="Previous champion UUID")
    previous_champion_hotkey: Optional[str] = Field(None, description="Previous champion hotkey")
    previous_champion_score: Optional[float] = Field(None, description="Previous champion score")
    margin_pct: Optional[float] = Field(None, description="Margin over previous champion (%)")
    action: str = Field(..., description="Action taken (first_champion, dethroned, no_change)")


class EmissionsDistributedPayload(BaseModel):
    """Payload for EMISSIONS_DISTRIBUTED event."""
    epoch: int = Field(..., description="Bittensor epoch")
    champion_model_id: Optional[str] = Field(None, description="UUID of champion model")
    champion_hotkey: Optional[str] = Field(None, description="SS58 address of champion")
    champion_score: Optional[float] = Field(None, description="Champion's score")
    emissions_pct: float = Field(..., description="Emissions percentage (0.05 = 5%)")
    burned: bool = Field(..., description="Whether emissions were burned")
    reason: str = Field(..., description="Reason for distribution/burn")


# =============================================================================
# In-Memory Event Log (for testing/development)
# =============================================================================

_event_log: List[Dict[str, Any]] = []


# =============================================================================
# Main Logging Function
# =============================================================================

async def log_qualification_event(
    event_type: str,
    payload: dict
) -> Dict[str, Any]:
    """
    Log a qualification event to the existing transparency_log table.
    
    Uses the same signed envelope format as sourcing events:
    - Signs event with TEE enclave
    - Computes event hash
    - Stores in transparency_log table
    
    Args:
        event_type: One of the QUALIFICATION_EVENT_TYPES
        payload: Event-specific payload dict
    
    Returns:
        Dict with signed_event, event_hash, enclave_pubkey, enclave_signature
    """
    if event_type not in QUALIFICATION_EVENT_TYPES:
        raise ValueError(f"Invalid qualification event type: {event_type}")
    
    logger.info(f"Logging qualification event: {event_type}")
    
    # Build the event envelope
    event_envelope = {
        "event_type": event_type,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    try:
        # Try to use the existing TEE enclave signer
        log_entry = await _sign_event_with_tee(event_type, event_envelope)
    except Exception as e:
        logger.warning(f"TEE signing unavailable, using mock signing: {e}")
        log_entry = _sign_event_mock(event_type, event_envelope)
    
    # Store in database
    await _store_event(event_type, log_entry)
    
    # Also store in local log for testing
    _event_log.append({
        "event_type": event_type,
        "payload": payload,
        "log_entry": log_entry,
        "logged_at": datetime.now(timezone.utc),
    })
    
    logger.info(f"Logged {event_type} event: {log_entry['event_hash'][:16]}...")
    return log_entry


async def _sign_event_with_tee(
    event_type: str,
    event_envelope: dict
) -> Dict[str, Any]:
    """
    Sign event using the existing TEE enclave signer.
    
    This imports and uses the existing sign_event function from
    gateway.tee.enclave_signer.
    """
    try:
        from gateway.tee.enclave_signer import sign_event
        
        # The existing sign_event expects (event_type, payload)
        return await sign_event(event_type, event_envelope)
        
    except ImportError:
        raise RuntimeError("TEE enclave signer not available")


def _sign_event_mock(
    event_type: str,
    event_envelope: dict
) -> Dict[str, Any]:
    """
    Mock event signing for development/testing.
    
    Produces the same structure as TEE signing but without
    actual cryptographic guarantees.
    """
    # Serialize event for hashing
    event_json = json.dumps(event_envelope, sort_keys=True, default=str)
    event_hash = hashlib.sha256(event_json.encode()).hexdigest()
    
    # Mock signature (in production, this would be from TEE)
    mock_pubkey = "mock_enclave_pubkey_" + event_hash[:16]
    mock_signature = hashlib.sha256(
        (event_hash + "_mock_signing_key").encode()
    ).hexdigest()
    
    return {
        "signed_event": event_envelope,
        "event_hash": event_hash,
        "enclave_pubkey": mock_pubkey,
        "enclave_signature": mock_signature,
    }


async def _store_event(
    event_type: str,
    log_entry: Dict[str, Any]
) -> None:
    """
    Store event in the transparency_log table.
    
    This uses the existing table schema from sourcing events.
    
    CRITICAL: Only stores events in PRODUCTION (mainnet/finney).
    On testnet, events are stored in local memory only to avoid
    polluting production transparency logs during testing.
    """
    # TESTNET GUARD: Do NOT write to production transparency_log on testnet
    bittensor_network = os.environ.get("BITTENSOR_NETWORK", "").lower()
    if bittensor_network == "test":
        logger.info(f"[TESTNET] Skipping transparency_log write for {event_type} (testnet mode)")
        return
    
    try:
        # Try to import and use Supabase client
        from qualification import supabase
        
        if supabase is None:
            logger.warning("Supabase client not initialized, event not stored in DB")
            return
        
        await supabase.table("transparency_log").insert({
            "event_type": event_type,
            "signed_event": log_entry["signed_event"],
            "event_hash": log_entry["event_hash"],
            "enclave_pubkey": log_entry["enclave_pubkey"],
            "enclave_signature": log_entry["enclave_signature"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        
        logger.info(f"[PRODUCTION] Stored event {log_entry['event_hash'][:16]} in transparency_log")
        
    except Exception as e:
        logger.warning(f"Failed to store event in database: {e}")
        # Don't fail - the in-memory log is sufficient for testing


# =============================================================================
# Specific Event Loggers
# =============================================================================

async def log_model_submitted(
    model_id: UUID,
    miner_hotkey: str,
    repo_url: str,
    commit_hash: str,
    code_hash: str,
    payment_verified: bool,
    submission_cost_usd: float,
    created_epoch: int
) -> Dict[str, Any]:
    """
    Log MODEL_SUBMITTED event.
    
    Called when a miner successfully submits a model for evaluation.
    """
    payload = ModelSubmittedPayload(
        model_id=str(model_id),
        miner_hotkey=miner_hotkey,
        repo_url=repo_url,
        commit_hash=commit_hash,
        code_hash=code_hash,
        payment_verified=payment_verified,
        submission_cost_usd=submission_cost_usd,
        created_epoch=created_epoch,
    )
    
    return await log_qualification_event(
        EVENT_MODEL_SUBMITTED,
        payload.dict()
    )


async def log_evaluation_complete(
    evaluation_id: UUID,
    model_id: UUID,
    miner_hotkey: str,
    set_id: int,
    final_score: float,
    total_cost_usd: float,
    icps_evaluated: int,
    status: str,
    db_hash: str,
    completed_epoch: int,
    icp_set_hash: str,
    top_10_leads: List[Dict[str, Any]],
    bottom_10_leads: List[Dict[str, Any]],
    screening_1_score: Optional[float] = None,
    screening_2_score: Optional[float] = None,
    final_benchmark_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Log EVALUATION_COMPLETE event with db_hash for audit.
    
    The db_hash is crucial for trustlessness - it proves which leads
    were accessible when the evaluation was run.
    
    CRITICAL ADDITIONS:
    - icp_set_hash: SHA256 hash of all ICP prompts used in this evaluation.
      Allows external auditors to verify exact ICPs that were tested.
    - top_10_leads: The 10 highest-scoring leads returned by the model (PII redacted).
      Used to audit model quality and detect gaming.
    - bottom_10_leads: The 10 lowest-scoring leads returned by the model (PII redacted).
      Used to understand failure patterns and edge cases.
    
    NOTE: This event is ONLY logged to transparency_log in PRODUCTION (mainnet/finney).
    On testnet, the event is stored in local memory only.
    """
    payload = EvaluationCompletePayload(
        evaluation_id=str(evaluation_id),
        model_id=str(model_id),
        miner_hotkey=miner_hotkey,
        set_id=set_id,
        final_score=final_score,
        screening_1_score=screening_1_score,
        screening_2_score=screening_2_score,
        final_benchmark_score=final_benchmark_score,
        total_cost_usd=total_cost_usd,
        icps_evaluated=icps_evaluated,
        status=status,
        db_hash=db_hash,
        completed_epoch=completed_epoch,
        icp_set_hash=icp_set_hash,
        top_10_leads=top_10_leads,
        bottom_10_leads=bottom_10_leads,
    )
    
    return await log_qualification_event(
        EVENT_EVALUATION_COMPLETE,
        payload.dict()
    )


async def log_champion_selected(
    epoch: int,
    set_id: int,
    champion_model_id: UUID,
    champion_hotkey: str,
    champion_score: float,
    action: str,
    previous_champion_id: Optional[UUID] = None,
    previous_champion_hotkey: Optional[str] = None,
    previous_champion_score: Optional[float] = None,
    margin_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Log CHAMPION_SELECTED event.
    
    Called at end of each epoch after champion selection runs.
    """
    payload = ChampionSelectedPayload(
        epoch=epoch,
        set_id=set_id,
        champion_model_id=str(champion_model_id),
        champion_hotkey=champion_hotkey,
        champion_score=champion_score,
        previous_champion_id=str(previous_champion_id) if previous_champion_id else None,
        previous_champion_hotkey=previous_champion_hotkey,
        previous_champion_score=previous_champion_score,
        margin_pct=margin_pct,
        action=action,
    )
    
    return await log_qualification_event(
        EVENT_CHAMPION_SELECTED,
        payload.dict()
    )


async def log_emissions_distributed(
    epoch: int,
    emissions_pct: float,
    burned: bool,
    reason: str,
    champion_model_id: Optional[UUID] = None,
    champion_hotkey: Optional[str] = None,
    champion_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Log EMISSIONS_DISTRIBUTED event.
    
    Called at end of each epoch after emissions distribution.
    """
    payload = EmissionsDistributedPayload(
        epoch=epoch,
        champion_model_id=str(champion_model_id) if champion_model_id else None,
        champion_hotkey=champion_hotkey,
        champion_score=champion_score,
        emissions_pct=emissions_pct,
        burned=burned,
        reason=reason,
    )
    
    return await log_qualification_event(
        EVENT_EMISSIONS_DISTRIBUTED,
        payload.dict()
    )


# =============================================================================
# Utility Functions
# =============================================================================

async def get_qualification_events(
    event_type: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get qualification events from the transparency log.
    
    Args:
        event_type: Filter by event type (optional)
        limit: Maximum events to return
    
    Returns:
        List of event records
    """
    try:
        from qualification import supabase
        
        if supabase is None:
            logger.warning("Supabase not configured, returning local log")
            events = _event_log
            if event_type:
                events = [e for e in events if e["event_type"] == event_type]
            return events[-limit:][::-1]
        
        query = supabase.table("transparency_log").select("*")
        
        if event_type:
            query = query.eq("event_type", event_type)
        else:
            # Filter to qualification events only
            query = query.in_("event_type", QUALIFICATION_EVENT_TYPES)
        
        response = await query.order("created_at", desc=True).limit(limit).execute()
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to get qualification events: {e}")
        # Fall back to local log
        events = _event_log
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events[-limit:][::-1]


async def get_event_by_hash(event_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific event by its hash.
    
    Args:
        event_hash: SHA256 hash of the event
    
    Returns:
        Event record or None
    """
    try:
        from qualification import supabase
        
        if supabase is None:
            # Search local log
            for event in _event_log:
                if event.get("log_entry", {}).get("event_hash") == event_hash:
                    return event
            return None
        
        response = await supabase.table("transparency_log") \
            .select("*") \
            .eq("event_hash", event_hash) \
            .execute()
        
        return response.data[0] if response.data else None
        
    except Exception as e:
        logger.error(f"Failed to get event by hash: {e}")
        return None


def verify_event_signature(
    event_hash: str,
    signature: str,
    pubkey: str
) -> bool:
    """
    Verify an event's TEE signature.
    
    In production, this verifies the Ed25519 signature from the TEE.
    For mock signatures, this just checks the hash matches.
    
    Args:
        event_hash: SHA256 hash of the event
        signature: Ed25519 signature (hex)
        pubkey: Public key (hex)
    
    Returns:
        True if signature is valid
    """
    if pubkey.startswith("mock_enclave_pubkey_"):
        # Mock signature verification
        expected_sig = hashlib.sha256(
            (event_hash + "_mock_signing_key").encode()
        ).hexdigest()
        return signature == expected_sig
    
    try:
        # Real TEE signature verification
        from nacl.signing import VerifyKey
        from nacl.encoding import HexEncoder
        
        verify_key = VerifyKey(pubkey, encoder=HexEncoder)
        verify_key.verify(bytes.fromhex(event_hash), bytes.fromhex(signature))
        return True
        
    except ImportError:
        logger.warning("nacl not installed, cannot verify TEE signatures")
        return False
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return False


def is_logging_configured() -> bool:
    """
    Check if logging is properly configured.
    
    Returns:
        True if both TEE signer and Supabase are available
    """
    tee_available = False
    supabase_available = False
    
    try:
        from gateway.tee.enclave_signer import sign_event
        tee_available = True
    except ImportError:
        pass
    
    try:
        from qualification import supabase
        supabase_available = supabase is not None
    except ImportError:
        pass
    
    return tee_available and supabase_available


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration status.
    
    Returns:
        Dict with configuration details
    """
    tee_available = False
    supabase_available = False
    
    try:
        from gateway.tee.enclave_signer import sign_event
        tee_available = True
    except ImportError:
        pass
    
    try:
        from qualification import supabase
        supabase_available = supabase is not None
    except ImportError:
        pass
    
    return {
        "tee_signer_available": tee_available,
        "supabase_available": supabase_available,
        "fully_configured": tee_available and supabase_available,
        "event_types": QUALIFICATION_EVENT_TYPES,
        "local_log_entries": len(_event_log),
    }


# =============================================================================
# Helper Functions for Verifiability
# =============================================================================

def compute_icp_set_hash(icps: List[Dict[str, Any]]) -> str:
    """
    Compute SHA256 hash of ICP set for verifiability.
    
    External auditors can use this hash to verify the exact ICPs
    that were used in the benchmark evaluation.
    
    Args:
        icps: List of ICP dictionaries (same format as sent to validator)
    
    Returns:
        SHA256 hex digest of the canonicalized ICP JSON
    """
    # Sort ICPs by icp_id for deterministic ordering
    sorted_icps = sorted(icps, key=lambda x: x.get("icp_id", ""))
    
    # Canonicalize: sorted keys, no extra whitespace
    canonical_json = json.dumps(sorted_icps, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def redact_lead_pii(lead: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact PII from a lead for logging.
    
    The logged lead should NOT contain:
    - email (personal identifier)
    - full_name (personal identifier)
    - first_name, last_name (personal identifier)
    - linkedin_url (personal identifier)
    - phone (personal identifier)
    
    The logged lead SHOULD contain:
    - business (company name)
    - role
    - industry, sub_industry
    - employee_count, geography
    - seniority
    - company_website, company_linkedin (company, not personal)
    - score fields (icp_fit, decision_maker, intent_signal, final_score)
    - failure_reason (if any)
    
    Args:
        lead: Full lead dictionary (may contain PII)
    
    Returns:
        Lead dictionary with PII redacted
    """
    if not lead:
        return {}
    
    # Fields to REDACT (PII)
    pii_fields = {"email", "full_name", "first_name", "last_name", "linkedin_url", "phone"}
    
    # Create copy with PII redacted
    redacted = {}
    for key, value in lead.items():
        if key in pii_fields:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    
    return redacted


def extract_top_bottom_leads(
    all_results: List[Dict[str, Any]],
    count: int = 10
) -> tuple:
    """
    Extract top N and bottom N leads by score.
    
    Args:
        all_results: List of dicts with "lead" and "scores" keys
        count: Number of top/bottom leads to extract (default 10)
    
    Returns:
        Tuple of (top_leads, bottom_leads), each PII-redacted
    """
    # Filter to only results with leads and scores
    valid_results = [
        r for r in all_results 
        if r.get("lead") and r.get("scores") and r["scores"].get("final_score") is not None
    ]
    
    # Sort by final_score descending
    sorted_results = sorted(
        valid_results, 
        key=lambda x: x["scores"]["final_score"], 
        reverse=True
    )
    
    # Extract top N
    top_leads = []
    for r in sorted_results[:count]:
        lead_data = redact_lead_pii(r["lead"])
        lead_data["_final_score"] = r["scores"]["final_score"]
        lead_data["_icp_fit"] = r["scores"].get("icp_fit", 0)
        lead_data["_decision_maker"] = r["scores"].get("decision_maker", 0)
        lead_data["_intent_signal"] = r["scores"].get("intent_signal_final", 0)
        lead_data["_failure_reason"] = r["scores"].get("failure_reason")
        top_leads.append(lead_data)
    
    # Extract bottom N
    bottom_leads = []
    for r in sorted_results[-count:] if len(sorted_results) > count else sorted_results:
        lead_data = redact_lead_pii(r["lead"])
        lead_data["_final_score"] = r["scores"]["final_score"]
        lead_data["_icp_fit"] = r["scores"].get("icp_fit", 0)
        lead_data["_decision_maker"] = r["scores"].get("decision_maker", 0)
        lead_data["_intent_signal"] = r["scores"].get("intent_signal_final", 0)
        lead_data["_failure_reason"] = r["scores"].get("failure_reason")
        bottom_leads.append(lead_data)
    
    return top_leads, bottom_leads


# =============================================================================
# Testing Helpers
# =============================================================================

def clear_local_log():
    """Clear the local event log for testing."""
    global _event_log
    _event_log = []
    logger.info("Local event log cleared")


def get_local_log() -> List[Dict[str, Any]]:
    """Get the local event log (for testing)."""
    return _event_log.copy()
