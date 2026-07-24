"""
Epoch API Endpoints

Provides endpoints for epoch-related operations:
- GET /epoch/{epoch_id}/leads - Get deterministically assigned leads for epoch

Timing Windows:
- Blocks 0-350: Lead distribution window (validators can fetch leads)
- Blocks 351-355: Validation submission window (no new lead fetches)
- Blocks 356-359: Buffer period (epoch closing)
- Block 360+: Epoch closed (reveal phase begins)

Those block ranges describe staged legacy mode.  In stateful mode, lead fetch
closes at 30 blocks remaining so no new validation batch can arrive after the
coherent consensus window begins (30 through 16 remaining).
"""

from fastapi import APIRouter, HTTPException, Query, Response
from typing import List
from datetime import datetime

from gateway.utils.assignment import get_validator_set  # deterministic_lead_assignment no longer needed here
from gateway.utils.signature import verify_wallet_signature
from gateway.utils.registry import is_registered_hotkey_async  # Use async version
from gateway.utils.leads_cache import get_cached_leads  # Import cache for instant lead distribution
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from gateway.db.client import _create_sync_client

# Supabase client (shared across threadpool workers — must stay HTTP/1-pinned;
# the default HTTP/2 HPACK encoder is not thread-safe)
supabase = _create_sync_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_LEADS_SELECT = (
    "lead_id, lead_blob, lead_blob_hash, miner_hotkey, "
    "first_name, last_name, email, role, company_name, linkedin, website, "
    "company_linkedin, industry, sub_industry, city, state, country, "
    "hq_city, hq_state, hq_country, employee_count, description"
)

_BLOB_FIELD_MAP = {
    "first": "first_name",
    "last": "last_name",
    "email": "email",
    "role": "role",
    "business": "company_name",
    "linkedin": "linkedin",
    "website": "website",
    "company_linkedin": "company_linkedin",
    "industry": "industry",
    "sub_industry": "sub_industry",
    "city": "city",
    "state": "state",
    "country": "country",
    "hq_city": "hq_city",
    "hq_state": "hq_state",
    "hq_country": "hq_country",
    "employee_count": "employee_count",
    "description": "description",
}


def _rebuild_blob(row: dict) -> dict:
    """Reconstruct a complete lead_blob from columns + overflow blob.
    Columns are the source of truth for the 18 extracted fields.
    The blob retains all other fields (attestation, flags, etc.)."""
    blob = row.get("lead_blob") or {}
    if isinstance(blob, str):
        import json
        blob = json.loads(blob)
    blob = dict(blob)
    for blob_key, col_name in _BLOB_FIELD_MAP.items():
        val = row.get(col_name)
        if val is not None:
            blob[blob_key] = val
    return blob


# Create router
router = APIRouter(prefix="/epoch", tags=["Epoch"])


@router.get("/state")
async def get_epoch_state(response: Response):
    """Expose the exact-hash SN71 scheduler state used for cutover verification."""

    from gateway.utils.epoch import get_epoch_authority_status_async

    try:
        result = await get_epoch_authority_status_async()
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Authoritative subnet epoch state unavailable: {exc}",
        ) from exc


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
    
    # Finalized state owns the workflow key. Best state is a separate liveness
    # veto and timing observation so finality lag cannot serve stale work.
    from gateway.utils.epoch import (
        get_current_epoch_admission_context_async,
        get_epoch_blocks_remaining,
        get_epoch_elapsed,
    )
    from Leadpoet.utils.subnet_epoch import SubnetEpochError

    try:
        _epoch_authority, epoch_timing, current_epoch = (
            await get_current_epoch_admission_context_async()
        )
    except SubnetEpochError as exc:
        raise HTTPException(
            status_code=503,
            detail="Authoritative subnet epoch admission state is unavailable",
        ) from exc
    if epoch_id != current_epoch:
        raise HTTPException(
            status_code=400,
            detail=f"Epoch {epoch_id} is not active. Current workflow epoch: {current_epoch}",
        )
    block_within_epoch = get_epoch_elapsed(epoch_timing)
    blocks_remaining = get_epoch_blocks_remaining(epoch_timing)
    window_closed = blocks_remaining <= 30
    if window_closed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Lead distribution window closed. "
                f"Epoch block: {block_within_epoch}; blocks remaining: {blocks_remaining}."
            ),
        )
    
    print(
        "✅ Step 3.5: Within lead distribution window "
        f"(epoch block {block_within_epoch}, {blocks_remaining} remaining)"
    )
    
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
    
    print(f"⚠️  [CACHE MISS] Epoch {epoch_id} not cached")
    
    # Step 4: Try to fetch assigned leads from EPOCH_INITIALIZATION event
    import asyncio
    from gateway.config import MAX_LEADS_PER_EPOCH
    
    assigned_lead_ids = None
    queue_root = "unknown"
    validator_count = 0
    try:
        print(f"🔍 Step 4: Checking EPOCH_INITIALIZATION for epoch {epoch_id}...")
        from gateway.tasks.epoch_lifecycle import get_durable_epoch_event

        try:
            durable_init = await asyncio.wait_for(
                get_durable_epoch_event("EPOCH_INITIALIZATION", epoch_id),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Durable epoch initialization query timed out",
            )

        if durable_init is not None:
            epoch_payload = durable_init.get("payload", {})
            assigned_lead_ids = epoch_payload.get("assignment", {}).get("assigned_lead_ids", [])
            queue_document = epoch_payload.get("queue_state") or epoch_payload.get("queue") or {}
            queue_root = queue_document.get(
                "queue_merkle_root",
                queue_document.get("queue_root", "unknown"),
            )
            validator_count = epoch_payload.get("assignment", {}).get("validator_count", 0)
            
            print(f"   ✅ EPOCH_INITIALIZATION found: {len(assigned_lead_ids)} leads assigned")
            print(f"   📊 Queue Root: {queue_root[:16] if queue_root != 'unknown' else 'unknown'}...")
            print(f"   📊 Validators: {validator_count}")
        else:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Durable EPOCH_INITIALIZATION for epoch {epoch_id} is not "
                    "available yet"
                ),
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=503,
            detail=f"Durable epoch initialization is unavailable: {str(e)}",
        ) from e

    leads_result = None
    
    # Step 5: Fetch full lead data from the immutable epoch assignment.
    if not assigned_lead_ids:
        # No leads assigned for this epoch
        return {
            "epoch_id": epoch_id,
            "leads": [],
            "queue_root": queue_root,
            "validator_count": validator_count,
            "max_leads_per_epoch": MAX_LEADS_PER_EPOCH  # Dynamic config for validators
        }
    
    # Resolve the rows committed by EPOCH_INITIALIZATION.
    if leads_result is None:
        try:
            total_leads = len(assigned_lead_ids)
            print(f"🔍 Step 5: Fetching {total_leads} leads from leads_private...")
            print(f"   Lead IDs: {assigned_lead_ids[:3]}... (showing first 3)")
            
            # CRITICAL: Use small batches (500 leads) to avoid URL length limits in .in_() queries
            # Each UUID is ~36 chars, 500 UUIDs = ~18KB which should be safe for PostgREST
            batch_size = 500
            num_batches = (total_leads + batch_size - 1) // batch_size  # Ceiling division
            
            print(f"   📦 Splitting into {num_batches} batches of ~{batch_size} leads each")
            
            all_leads_data = []
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_leads)
                batch_ids = assigned_lead_ids[start_idx:end_idx]
                
                print(f"   🔍 Fetching batch {batch_num + 1}/{num_batches} ({len(batch_ids)} leads)...")
                
                # Need to capture batch_ids in closure properly
                def make_query(ids):
                    return lambda: supabase.table("leads_private").select(_LEADS_SELECT).in_("lead_id", ids).execute()
                
                batch_result = await asyncio.wait_for(
                    asyncio.to_thread(make_query(batch_ids)),
                    timeout=90.0
                )
                
                if batch_result.data:
                    all_leads_data.extend(batch_result.data)
                    print(f"   ✅ Batch {batch_num + 1}/{num_batches}: Fetched {len(batch_result.data)} leads")
                else:
                    print(f"   ⚠️  Batch {batch_num + 1}/{num_batches}: No leads returned")
            
            # Create mock result object with aggregated data
            class MockResult:
                def __init__(self, data):
                    self.data = data
            
            leads_result = MockResult(all_leads_data)
            
            print(f"✅ Aggregated {len(leads_result.data)} total leads from {num_batches} batches")
            
            if not leads_result.data:
                print(f"❌ ERROR: No leads found in database for assigned IDs")
                print(f"   Assigned IDs: {assigned_lead_ids[:5]}...")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to fetch lead data from private database"
                )
        
        except asyncio.TimeoutError:
            print(f"❌ ERROR: Supabase query timed out after 90 seconds")
            raise HTTPException(
                status_code=504,
                detail="Database query timeout - gateway may be experiencing high load"
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
    try:
        print(f"🔍 Step 6: Building full lead data for {len(leads_result.data)} leads...")
        full_leads = []
        for idx, lead_row in enumerate(leads_result.data):
            try:
                lead_blob = _rebuild_blob(lead_row)
                miner_hotkey = lead_row.get("miner_hotkey", "unknown")
                
                full_leads.append({
                    "lead_id": lead_row["lead_id"],
                    "lead_blob": lead_blob,
                    "lead_blob_hash": lead_row["lead_blob_hash"],
                    "miner_hotkey": miner_hotkey
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
        from gateway.utils.epoch import get_epoch_info_async

        info = await get_epoch_info_async(epoch_id)
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
        from gateway.utils.epoch import (
            get_current_epoch_context_async,
            get_current_epoch_info_from_snapshot,
        )

        epoch_snapshot, current_epoch = await get_current_epoch_context_async()
        info = get_current_epoch_info_from_snapshot(epoch_snapshot)
        
        return {
            "current_epoch_id": current_epoch,
            "epoch_info": info
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current epoch: {str(e)}"
        )
