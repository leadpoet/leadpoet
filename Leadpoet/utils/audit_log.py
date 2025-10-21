"""
Audit Logging System for Regulatory Compliance

This module provides functions to log all regulatory events to an immutable
audit trail in Supabase. All logs are append-only and cannot be modified.

Event Types:
- submission: Miner submits leads to prospect_queue
- validation: Validator validates leads
- rejection: Validator rejects leads
- suppression: Admin suppresses/deletes leads (logged Supabase-side)
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional
import bittensor as bt


def generate_lead_hash(lead: Dict) -> str:
    """
    Generate unique SHA-256 hash for lead (for audit trail).
    
    Uses email + company + timestamp to create a unique identifier
    that can be used to track the lead through the system.
    
    Args:
        lead: Lead data dict with email, company, submission_timestamp
        
    Returns:
        str: SHA-256 hash (64 hex characters)
    """
    from Leadpoet.utils.utils_lead_extraction import get_email, get_company
    
    # Extract key fields for hash
    email = get_email(lead) or ""
    company = get_company(lead) or ""
    timestamp = lead.get("submission_timestamp") or datetime.now(timezone.utc).isoformat()
    
    # Combine fields with delimiter
    combined = "|".join([email, company, timestamp])
    
    # Generate SHA-256 hash
    lead_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    return lead_hash


def log_submission_audit(lead: Dict, wallet: str, event_type: str = "submission") -> bool:
    """
    Log audit trail for lead submission.
    
    Creates an immutable audit record in the audit_log table with:
    - Lead hash (unique identifier)
    - Wallet address (miner hotkey)
    - Attestation metadata (terms version, booleans)
    - Source provenance (URL, type, license info)
    - Timestamp and event type
    
    Args:
        lead: Lead data dict
        wallet: Wallet address (miner hotkey SS58)
        event_type: "submission", "rejection", etc.
        
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        from Leadpoet.utils.cloud_db import get_supabase_client
        from Leadpoet.utils.utils_lead_extraction import (
            get_company, get_industry, get_location
        )
        
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available - skipping audit log")
            return False
        
        # Generate unique hash for this lead
        lead_hash = generate_lead_hash(lead)
        
        # Build audit entry
        audit_entry = {
            "lead_hash": lead_hash,
            "wallet_address": wallet,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # Attestation metadata
            "terms_version_hash": lead.get("terms_version_hash"),
            "lawful_collection": lead.get("lawful_collection"),
            "no_restricted_sources": lead.get("no_restricted_sources"),
            "license_granted": lead.get("license_granted"),
            
            # Provenance metadata
            "source_url": lead.get("source_url"),
            "source_type": lead.get("source_type"),
            "license_doc_hash": lead.get("license_doc_hash"),
            
            # Anonymized lead data (no PII)
            "company": get_company(lead),
            "industry": get_industry(lead),
            "region": get_location(lead),
            
            # Additional metadata
            "metadata": {
                "submission_timestamp": lead.get("submission_timestamp"),
                "license_doc_url": lead.get("license_doc_url"),
            }
        }
        
        # Insert into audit_log (append-only table)
        result = supabase.table("audit_log").insert(audit_entry)
        
        # Check response - CustomResponse has .data and .response attributes
        if hasattr(result, 'data') and result.data:
            bt.logging.info(f"✅ Audit log created: {lead_hash[:16]}... ({event_type})")
            return True
        elif hasattr(result, 'response'):
            # Successful HTTP request but empty response (Supabase sometimes returns 201 with no body)
            if result.response.status_code in [200, 201, 204]:
                bt.logging.info(f"✅ Audit trail logged (submission)")
                return True
            else:
                bt.logging.warning(f"Failed to create audit log: HTTP {result.response.status_code} - {result.response.text[:200]}")
                return False
        else:
            bt.logging.warning(f"Failed to create audit log: Unexpected response format")
            return False
            
    except Exception as e:
        bt.logging.error(f"Error logging submission audit: {e}")
        # Don't fail the main operation if audit fails
        return False


def log_validation_audit(
    lead: Dict, 
    validator_wallet: str, 
    validation_result: Dict
) -> bool:
    """
    Log validator's validation event.
    
    Records the validator's assessment of a lead, including:
    - Whether it passed or failed
    - Validation reason
    - All attestation and provenance metadata
    
    Args:
        lead: Lead data dict
        validator_wallet: Validator wallet address (SS58)
        validation_result: Dict with 'passed', 'reason'
        
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        from Leadpoet.utils.cloud_db import get_supabase_client
        from Leadpoet.utils.utils_lead_extraction import (
            get_company, get_industry, get_location
        )
        
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available - skipping audit log")
            return False
        
        # Generate unique hash for this lead
        lead_hash = generate_lead_hash(lead)
        
        # Determine event type based on validation result
        event_type = "validation" if validation_result.get("passed") else "rejection"
        
        # Build audit entry
        audit_entry = {
            "lead_hash": lead_hash,
            "wallet_address": validator_wallet,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # Attestation metadata (from lead)
            "terms_version_hash": lead.get("terms_version_hash"),
            "lawful_collection": lead.get("lawful_collection"),
            "no_restricted_sources": lead.get("no_restricted_sources"),
            "license_granted": lead.get("license_granted"),
            
            # Provenance metadata (from lead)
            "source_url": lead.get("source_url"),
            "source_type": lead.get("source_type"),
            "license_doc_hash": lead.get("license_doc_hash"),
            
            # Validation metadata (from validation_result)
            "validation_passed": validation_result.get("passed"),
            "validation_reason": validation_result.get("reason"),
            
            # Anonymized lead data (no PII)
            "company": get_company(lead),
            "industry": get_industry(lead),
            "region": get_location(lead),
            
            # Additional metadata
            "metadata": {
                "validation_details": validation_result,
                "deep_verification": lead.get("deep_verification_results"),
                "manual_review_required": lead.get("manual_review_required"),
            }
        }
        
        # Insert into audit_log (append-only table)
        result = supabase.table("audit_log").insert(audit_entry)
        
        # Check response - CustomResponse has .data and .response attributes
        if hasattr(result, 'data') and result.data:
            bt.logging.info(f"✅ Validation audit logged: {lead_hash[:16]}... ({event_type})")
            return True
        elif hasattr(result, 'response'):
            # Successful HTTP request but empty response (Supabase sometimes returns 201 with no body)
            if result.response.status_code in [200, 201, 204]:
                bt.logging.info(f"✅ Audit trail logged (validation)")
                return True
            else:
                bt.logging.warning(f"Failed to log validation audit: HTTP {result.response.status_code} - {result.response.text[:200]}")
                return False
        else:
            bt.logging.warning(f"Failed to log validation audit: Unexpected response format")
            return False
            
    except Exception as e:
        bt.logging.error(f"Error logging validation audit: {e}")
        # Don't fail the main operation if audit fails
        return False


def get_audit_trail(lead_hash: str) -> list:
    """
    Retrieve complete audit trail for a lead.
    
    Queries the audit_log table for all events related to a specific lead,
    ordered chronologically.
    
    Args:
        lead_hash: SHA-256 hash of the lead
        
    Returns:
        list: List of audit log entries, or empty list if none found
    """
    try:
        from Leadpoet.utils.cloud_db import get_supabase_client
        
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available")
            return []
        
        # Query audit logs for this lead hash
        result = supabase.table("audit_log")\
            .select("*")\
            .eq("lead_hash", lead_hash)\
            .order("timestamp", desc=False)
        
        if result.data:
            bt.logging.info(f"Found {len(result.data)} audit entries for lead {lead_hash[:16]}...")
            return result.data
        else:
            return []
            
    except Exception as e:
        bt.logging.error(f"Error retrieving audit trail: {e}")
        return []


def get_wallet_audit_history(wallet_address: str, limit: int = 100) -> list:
    """
    Retrieve audit history for a specific wallet.
    
    Useful for reviewing a miner or validator's compliance history.
    
    Args:
        wallet_address: Wallet SS58 address
        limit: Maximum number of entries to return
        
    Returns:
        list: List of audit log entries, or empty list if none found
    """
    try:
        from Leadpoet.utils.cloud_db import get_supabase_client
        
        supabase = get_supabase_client()
        if not supabase:
            bt.logging.warning("Supabase client not available")
            return []
        
        # Query audit logs for this wallet
        result = supabase.table("audit_log")\
            .select("*")\
            .eq("wallet_address", wallet_address)\
            .order("timestamp", desc=True)\
            .limit(limit)
        
        if result.data:
            bt.logging.info(f"Found {len(result.data)} audit entries for wallet {wallet_address[:10]}...")
            return result.data
        else:
            return []
            
    except Exception as e:
        bt.logging.error(f"Error retrieving wallet audit history: {e}")
        return []

