"""
Epoch API Endpoints

Provides endpoints for epoch-related operations:
- GET /epoch/{epoch_id}/leads - Get deterministically assigned leads for epoch
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime

from gateway.utils.epoch import get_current_epoch_id, is_epoch_active, get_epoch_info
from gateway.utils.assignment import deterministic_lead_assignment, get_validator_set
from gateway.utils.signature import verify_wallet_signature
from gateway.utils.registry import is_registered_hotkey
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Create router
router = APIRouter(prefix="/epoch", tags=["Epoch"])


@router.get("/{epoch_id}/leads")
async def get_epoch_leads(
    epoch_id: int,
    validator_hotkey: str = Query(..., description="Validator's SS58 address"),
    signature: str = Query(..., description="Ed25519 signature over message")
):
    """
    Get deterministically assigned leads for given epoch with FULL lead data.
    
    All validators get the same 50 leads (first 50 from queue, FIFO).
    Returns complete lead data including lead_blob, lead_blob_hash, and miner_hotkey.
    
    Flow:
    1. Verify signature over "GET_EPOCH_LEADS:{epoch_id}:{validator_hotkey}"
    2. Verify validator is registered on subnet
    3. Verify epoch is active (blocks 0-360)
    4. Query EPOCH_INITIALIZATION from transparency_log
    5. Get validator set from metagraph
    6. Compute deterministic assignment (first 50 lead_ids)
    7. Fetch full lead data from leads_private
    8. Fetch miner_hotkey for each lead from SUBMISSION events
    9. Return full lead data
    
    Args:
        epoch_id: Epoch number
        validator_hotkey: Validator's SS58 address
        signature: Ed25519 signature (hex string)
    
    Returns:
        {
            "epoch_id": int,
            "leads": [
                {
                    "lead_id": str,
                    "lead_blob": dict,
                    "lead_blob_hash": str,
                    "miner_hotkey": str
                },
                ...
            ],
            "queue_root": str,
            "validator_count": int
        }
    
    Raises:
        403: Invalid signature or not a registered validator
        400: Epoch not active
        404: EPOCH_INITIALIZATION not found for epoch
    
    Example:
        GET /epoch/100/leads?validator_hotkey=5GNJqR...&signature=0xabc123...
    """
    
    # Step 1: Verify signature
    message = f"GET_EPOCH_LEADS:{epoch_id}:{validator_hotkey}"
    
    if not verify_wallet_signature(message, signature, validator_hotkey):
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # Step 2: Verify validator is registered
    is_registered, role = is_registered_hotkey(validator_hotkey)
    
    if not is_registered:
        raise HTTPException(
            status_code=403,
            detail="Hotkey not registered on subnet"
        )
    
    if role != "validator":
        raise HTTPException(
            status_code=403,
            detail="Only validators can fetch epoch leads"
        )
    
    # Step 3: Verify epoch is active
    if not is_epoch_active(epoch_id):
        # Only allow fetching for the current epoch
        current_epoch = get_current_epoch_id()
        if epoch_id != current_epoch:
            raise HTTPException(
                status_code=400,
                detail=f"Epoch {epoch_id} is not active. Current epoch: {current_epoch}"
            )
    
    # Step 4: Query EPOCH_INITIALIZATION from transparency_log
    try:
        result = supabase.table("transparency_log") \
            .select("payload") \
            .eq("event_type", "EPOCH_INITIALIZATION") \
            .eq("payload->>epoch_id", str(epoch_id)) \
            .order("id", desc=True) \
            .limit(1) \
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"EPOCH_INITIALIZATION not found for epoch {epoch_id}. Epoch may not have started yet."
            )
        
        epoch_init_event = result.data[0]["payload"]
        queue_root = epoch_init_event["queue_state"]["queue_merkle_root"]
        total_pending = epoch_init_event["queue_state"]["pending_lead_count"]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query EPOCH_INITIALIZATION: {str(e)}"
        )
    
    # Step 5: Get validator set
    try:
        validator_set = get_validator_set(epoch_id)
        
        if not validator_set:
            raise HTTPException(
                status_code=500,
                detail="No validators found in metagraph"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get validator set: {str(e)}"
        )
    
    # Step 6: Compute deterministic assignment (returns first 50 lead_ids)
    try:
        assigned_lead_ids = deterministic_lead_assignment(
            queue_root=queue_root,
            validator_set=validator_set,
            epoch_id=epoch_id,
            max_leads_per_epoch=50
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute assignment: {str(e)}"
        )
    
    # Step 7: Fetch full lead data from leads_private
    try:
        if not assigned_lead_ids:
            # No leads assigned for this epoch
            return {
                "epoch_id": epoch_id,
                "leads": [],
                "queue_root": queue_root,
                "validator_count": len(validator_set)
            }
        
        leads_result = supabase.table("leads_private") \
            .select("lead_id, lead_blob, lead_blob_hash") \
            .in_("lead_id", assigned_lead_ids) \
            .execute()
        
        if not leads_result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch lead data from private database"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch lead data: {str(e)}"
        )
    
    # Step 8: Fetch miner_hotkey for each lead from SUBMISSION_REQUEST events
    # Note: SUBMISSION events would be logged after storage verification, but for now
    # we query SUBMISSION_REQUEST events which contain the miner's hotkey
    full_leads = []
    for lead_row in leads_result.data:
        try:
            submission = supabase.table("transparency_log") \
                .select("actor_hotkey") \
                .eq("event_type", "SUBMISSION_REQUEST") \
                .eq("payload->>lead_id", lead_row["lead_id"]) \
                .limit(1) \
                .execute()
            
            miner_hotkey = submission.data[0]["actor_hotkey"] if submission.data else "unknown"
            
            full_leads.append({
                "lead_id": lead_row["lead_id"],
                "lead_blob": lead_row["lead_blob"],
                "lead_blob_hash": lead_row["lead_blob_hash"],
                "miner_hotkey": miner_hotkey
            })
        
        except Exception as e:
            # If we can't fetch miner_hotkey, still include the lead
            print(f"⚠️  Warning: Failed to fetch miner_hotkey for lead {lead_row['lead_id']}: {e}")
            full_leads.append({
                "lead_id": lead_row["lead_id"],
                "lead_blob": lead_row["lead_blob"],
                "lead_blob_hash": lead_row["lead_blob_hash"],
                "miner_hotkey": "unknown"
            })
    
    # Step 9: Return full lead data
    return {
        "epoch_id": epoch_id,
        "leads": full_leads,
        "queue_root": queue_root,
        "validator_count": len(validator_set)
    }


@router.get("/{epoch_id}/info")
async def get_epoch_information(epoch_id: int):
    """
    Get comprehensive information about an epoch.
    
    Public endpoint (no authentication required) for checking epoch status.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Epoch information dictionary
    
    Example:
        GET /epoch/100/info
    """
    try:
        info = get_epoch_info(epoch_id)
        return info
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get epoch info: {str(e)}"
        )


@router.get("/current")
async def get_current_epoch():
    """
    Get current epoch ID and information.
    
    Public endpoint for checking current epoch.
    
    Returns:
        {
            "current_epoch_id": int,
            "epoch_info": dict
        }
    
    Example:
        GET /epoch/current
    """
    try:
        current_epoch = get_current_epoch_id()
        info = get_epoch_info(current_epoch)
        
        return {
            "current_epoch_id": current_epoch,
            "epoch_info": info
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current epoch: {str(e)}"
        )

