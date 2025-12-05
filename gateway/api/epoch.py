"""
Epoch API Endpoints

Provides endpoints for epoch-related operations:
- GET /epoch/{epoch_id}/leads - Get deterministically assigned leads for epoch

Timing Windows:
- Blocks 0-350: Lead distribution window (validators can fetch leads)
- Blocks 351-355: Validation submission window (no new lead fetches)
- Blocks 356-359: Buffer period (epoch closing)
- Block 360+: Epoch closed (reveal phase begins)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime

from gateway.utils.epoch import get_current_epoch_id, is_epoch_active, get_epoch_info
from gateway.utils.assignment import get_validator_set  # deterministic_lead_assignment no longer needed here
from gateway.utils.signature import verify_wallet_signature
from gateway.utils.registry import is_registered_hotkey_async  # Use async version
from gateway.utils.leads_cache import get_cached_leads  # Import cache for instant lead distribution
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
    3.5. Verify within lead distribution window (blocks 0-350)
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
        400: Epoch not active, or lead distribution window closed (block > 350)
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
    
    # Use async registry check (direct call, no thread needed - uses injected AsyncSubtensor)
    import asyncio
    try:
        is_registered, role = await asyncio.wait_for(
            is_registered_hotkey_async(validator_hotkey),  # Direct async call (no thread wrapper)
            timeout=180.0  # 180 second timeout for metagraph query (testnet can be slow, allows for retries)
        )
    except asyncio.TimeoutError:
        print(f"❌ Metagraph query timed out after 180s for {validator_hotkey[:20]}...")
        raise HTTPException(
            status_code=504,
            detail="Metagraph query timeout - please retry in a moment (cache warming)"
        )
    
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
    from gateway.utils.epoch import is_epoch_active_async, get_current_epoch_id_async
    if not await is_epoch_active_async(epoch_id):
        # Only allow fetching for the current epoch
        current_epoch = await get_current_epoch_id_async()
        if epoch_id != current_epoch:
            raise HTTPException(
                status_code=400,
                detail=f"Epoch {epoch_id} is not active. Current epoch: {current_epoch}"
            )
    
    # Step 3.5: Verify within lead distribution window (blocks 0-350)
    from gateway.utils.epoch import get_block_within_epoch_async
    
    block_within_epoch = await get_block_within_epoch_async()
    if block_within_epoch > 350:
        raise HTTPException(
            status_code=400,
            detail=f"Lead distribution window closed at block 350. Current block within epoch: {block_within_epoch}. Validators must fetch leads before block 351."
        )
    
    print(f"✅ Step 3.5: Within lead distribution window (block {block_within_epoch}/350)")
    
    # ========================================================================
    # Step 3.6: Check if validator has already submitted validations for this epoch
    # ========================================================================
    # CRITICAL: Don't send leads to validators who've already submitted
    # This prevents infinite loops and wasted work
    try:
        existing_submission = supabase.table("validation_evidence_private") \
            .select("evidence_id") \
            .eq("validator_hotkey", validator_hotkey) \
            .eq("epoch_id", epoch_id) \
            .limit(1) \
            .execute()
        
        if existing_submission.data:
            print(f"⚠️  Step 3.6: Validator {validator_hotkey[:20]}... already submitted for epoch {epoch_id}")
            print(f"   Returning empty lead list (no work to do)")
            return {
                "epoch_id": epoch_id,
                "leads": [],  # Empty list - already submitted
                "queue_root": "already_submitted",
                "validator_count": 0,
                "message": f"You have already submitted validations for epoch {epoch_id}. No additional work needed.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        print(f"✅ Step 3.6: First time fetching leads for epoch {epoch_id} (no prior submission)")
    
    except Exception as e:
        # Log error but don't fail - this is just an optimization check
        print(f"⚠️  Warning: Failed to check existing submission (continuing anyway): {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # OPTIMIZATION: Check cache first (instant response, no DB query)
    # ========================================================================
    cached_leads = get_cached_leads(epoch_id)
    if cached_leads is not None:
        print(f"✅ [CACHE HIT] Returning {len(cached_leads)} cached leads for epoch {epoch_id}")
        print(f"   Response time: <100ms (no database query)")
        print(f"   Validator: {validator_hotkey[:20]}...")
        
        # Get validator set for response metadata
        try:
            validator_set = await get_validator_set(epoch_id)
            validator_count = len(validator_set) if validator_set else 0
        except:
            validator_count = 0
        
        from gateway.config import MAX_LEADS_PER_EPOCH
        return {
            "epoch_id": epoch_id,
            "leads": cached_leads,
            "queue_root": "cached",  # Queue root not needed for cached response
            "validator_count": validator_count,
            "max_leads_per_epoch": MAX_LEADS_PER_EPOCH,  # Dynamic config for validators
            "cached": True,  # Indicate this was served from cache
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Cache miss - fall back to EPOCH_INITIALIZATION snapshot (CRITICAL: must match /validate check!)
    print(f"⚠️  [CACHE MISS] Epoch {epoch_id} not cached, falling back to EPOCH_INITIALIZATION...")
    
    # Step 4: Fetch assigned leads from EPOCH_INITIALIZATION event
    # CRITICAL FIX: We MUST use the frozen snapshot from EPOCH_INITIALIZATION, NOT the current
    # queue state. The /validate endpoint checks lead_ids against EPOCH_INITIALIZATION.assignment,
    # so if we return different leads here, validation will fail with "lead not assigned to epoch".
    #
    # Previous bug: Querying current queue state could return leads from a different epoch
    # when new leads were submitted after EPOCH_INITIALIZATION.
    try:
        import asyncio
        from gateway.config import MAX_LEADS_PER_EPOCH
        
        print(f"🔍 Step 4: Fetching EPOCH_INITIALIZATION for epoch {epoch_id}...")
        
        # Query EPOCH_INITIALIZATION from transparency_log
        try:
            init_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: supabase.table("transparency_log")
                        .select("payload")
                        .eq("event_type", "EPOCH_INITIALIZATION")
                        .eq("epoch_id", epoch_id)
                        .limit(1)
                        .execute()
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print(f"❌ ERROR: EPOCH_INITIALIZATION query timed out after 30 seconds")
            raise HTTPException(
                status_code=504,
                detail="Database query timeout - gateway may be experiencing high load"
            )
        
        if not init_result.data:
            print(f"❌ ERROR: No EPOCH_INITIALIZATION found for epoch {epoch_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Epoch {epoch_id} not initialized. No leads assigned yet."
            )
        
        # Extract assigned_lead_ids from EPOCH_INITIALIZATION payload
        epoch_payload = init_result.data[0].get("payload", {})
        assigned_lead_ids = epoch_payload.get("assignment", {}).get("assigned_lead_ids", [])
        queue_root = epoch_payload.get("queue", {}).get("queue_root", "unknown")
        validator_count = epoch_payload.get("assignment", {}).get("validator_count", 0)
        
        print(f"   📊 EPOCH_INITIALIZATION: {len(assigned_lead_ids)} leads assigned")
        print(f"   📊 Queue Root: {queue_root[:16]}...")
        print(f"   📊 Validators: {validator_count}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch EPOCH_INITIALIZATION: {str(e)}"
        )
    
    # Step 5: Fetch full lead data from leads_private (with timeout)
    try:
        if not assigned_lead_ids:
            # No leads assigned for this epoch
            return {
                "epoch_id": epoch_id,
                "leads": [],
                "queue_root": queue_root,
                "validator_count": validator_count,
                "max_leads_per_epoch": MAX_LEADS_PER_EPOCH  # Dynamic config for validators
            }
        
        print(f"🔍 Step 5: Fetching {len(assigned_lead_ids)} leads from leads_private...")
        print(f"   Lead IDs: {assigned_lead_ids[:3]}... (showing first 3)")
        
        # NOTE: miner_hotkey column doesn't exist yet in Supabase
        # TODO: Add migration to create miner_hotkey column, then optimize this query
        
        # Wrap Supabase query with asyncio timeout (90 seconds)
        import asyncio
        try:
            leads_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: supabase.table("leads_private")
                        .select("lead_id, lead_blob, lead_blob_hash")
                        .in_("lead_id", assigned_lead_ids)
                        .execute()
                ),
                timeout=90.0  # 90 second timeout for database query (Supabase very slow on testnet)
            )
        except asyncio.TimeoutError:
            print(f"❌ ERROR: Supabase query timed out after 90 seconds")
            raise HTTPException(
                status_code=504,
                detail="Database query timeout - gateway may be experiencing high load"
            )
        
        print(f"✅ Fetched {len(leads_result.data) if leads_result.data else 0} leads from database")
        
        if not leads_result.data:
            print(f"❌ ERROR: No leads found in database for assigned IDs")
            print(f"   This means leads were assigned but don't exist in leads_private")
            print(f"   Assigned IDs: {assigned_lead_ids}")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch lead data from private database"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR in Step 5: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch lead data: {str(e)}"
        )
    
    # Step 6: Build full_leads with miner_hotkey extracted from lead_blob
    # NOTE: miner_hotkey column doesn't exist yet, so we extract from lead_blob (wallet_ss58)
    try:
        print(f"🔍 Step 6: Building full lead data for {len(leads_result.data)} leads...")
        full_leads = []
        for idx, lead_row in enumerate(leads_result.data):
            try:
                # Extract miner_hotkey from lead_blob (wallet_ss58 field)
                lead_blob = lead_row.get("lead_blob", {})
                miner_hotkey = lead_blob.get("wallet_ss58", "unknown")
                
                full_leads.append({
                    "lead_id": lead_row["lead_id"],
                    "lead_blob": lead_blob,
                    "lead_blob_hash": lead_row["lead_blob_hash"],
                    "miner_hotkey": miner_hotkey  # Extracted from lead_blob
                })
            except Exception as e:
                print(f"❌ ERROR building lead {idx}: {e}")
                print(f"   Lead row keys: {lead_row.keys() if hasattr(lead_row, 'keys') else 'N/A'}")
                print(f"   Lead blob keys: {lead_blob.keys() if isinstance(lead_blob, dict) else 'N/A'}")
                print(f"   Lead row: {lead_row}")
                raise
        
        print(f"✅ Step 6 complete: Built {len(full_leads)} full lead objects")
    
    except Exception as e:
        print(f"❌ ERROR in Step 6: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build lead data: {str(e)}"
        )
    
    # Step 7: Cache leads for subsequent requests (instant response for other validators)
    from gateway.utils.leads_cache import set_cached_leads
    set_cached_leads(epoch_id, full_leads)
    print(f"💾 [CACHE SET] Cached {len(full_leads)} leads for epoch {epoch_id}")
    print(f"   Subsequent validator requests will be instant (<100ms)")
    
    # Step 8: Return full lead data
    print(f"✅ Step 8: Returning {len(full_leads)} leads to validator")
    return {
        "epoch_id": epoch_id,
        "leads": full_leads,
        "queue_root": queue_root,
        "validator_count": validator_count,
        "max_leads_per_epoch": MAX_LEADS_PER_EPOCH,  # Dynamic config for validators
        "cached": False,  # This response was from DB query, not cache
        "timestamp": datetime.utcnow().isoformat()
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
        from gateway.utils.epoch import get_current_epoch_id_async
        current_epoch = await get_current_epoch_id_async()
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

