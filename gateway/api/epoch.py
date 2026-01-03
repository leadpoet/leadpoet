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
    
    # Cache miss - try EPOCH_INITIALIZATION first, fall back to direct queue query
    print(f"⚠️  [CACHE MISS] Epoch {epoch_id} not cached, trying fallback approaches...")
    
    # Step 4: Try to fetch assigned leads from EPOCH_INITIALIZATION event
    import asyncio
    from gateway.config import MAX_LEADS_PER_EPOCH
    
    assigned_lead_ids = None
    queue_root = "unknown"
    validator_count = 0
    use_direct_query = False
    
    try:
        print(f"🔍 Step 4: Checking EPOCH_INITIALIZATION for epoch {epoch_id}...")
        
        # Query EPOCH_INITIALIZATION from transparency_log
        # NOTE: epoch_id is stored INSIDE payload JSON, not as a column
        try:
            init_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: supabase.table("transparency_log")
                        .select("payload")
                        .eq("event_type", "EPOCH_INITIALIZATION")
                        .eq("payload->>epoch_id", str(epoch_id))
                        .limit(1)
                        .execute()
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print(f"⚠️  EPOCH_INITIALIZATION query timed out, falling back to direct query...")
            use_direct_query = True
            init_result = None
        
        if init_result and init_result.data:
            # Extract assigned_lead_ids from EPOCH_INITIALIZATION payload
            epoch_payload = init_result.data[0].get("payload", {})
            assigned_lead_ids = epoch_payload.get("assignment", {}).get("assigned_lead_ids", [])
            queue_root = epoch_payload.get("queue", {}).get("queue_root", "unknown")
            validator_count = epoch_payload.get("assignment", {}).get("validator_count", 0)
            
            print(f"   ✅ EPOCH_INITIALIZATION found: {len(assigned_lead_ids)} leads assigned")
            print(f"   📊 Queue Root: {queue_root[:16] if queue_root != 'unknown' else 'unknown'}...")
            print(f"   📊 Validators: {validator_count}")
        else:
            print(f"   ⚠️  No EPOCH_INITIALIZATION for epoch {epoch_id}, using direct queue query...")
            use_direct_query = True
    
    except Exception as e:
        print(f"⚠️  Error checking EPOCH_INITIALIZATION: {e}, falling back to direct query...")
        use_direct_query = True
    
    # Fallback: Query leads directly AND create EPOCH_INITIALIZATION to ensure consistency
    # CRITICAL: We must create EPOCH_INITIALIZATION so /validate uses the same snapshot
    leads_result = None  # Will be set by fallback or Step 5
    
    if use_direct_query or assigned_lead_ids is None:
        try:
            print(f"🔍 Step 4b: Querying leads directly from leads_private (fallback)...")
            
            # Query first MAX_LEADS_PER_EPOCH leads from queue that haven't been validated
            # CRITICAL: Supabase has a 1000-row limit per request, must use pagination
            batch_size = 500
            offset = 0
            rows_needed = MAX_LEADS_PER_EPOCH
            all_leads_data = []
            
            while len(all_leads_data) < rows_needed:
                start = offset
                end = min(offset + batch_size - 1, rows_needed - 1)
                
                batch_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda s=start, e=end: supabase.table("leads_private")
                            .select("lead_id, lead_blob, lead_blob_hash")
                            .eq("status", "pending_validation")
                            .order("created_ts", desc=False)
                            .range(s, e)
                            .execute()
                    ),
                    timeout=30.0
                )
                
                if not batch_result.data:
                    break
                
                all_leads_data.extend(batch_result.data)
                
                if len(batch_result.data) < (end - start + 1):
                    break
                
                offset += batch_size
            
            # Create a result-like object for backward compatibility
            class FallbackResult:
                def __init__(self, data):
                    self.data = data
            
            leads_result = FallbackResult(all_leads_data[:MAX_LEADS_PER_EPOCH])
            
            if leads_result.data:
                assigned_lead_ids = [lead["lead_id"] for lead in leads_result.data]
                print(f"   ✅ Direct query found {len(assigned_lead_ids)} pending leads")
                
                # CRITICAL: Create EPOCH_INITIALIZATION event so /validate uses same snapshot
                # This prevents the mismatch issue where new leads submitted during epoch
                # would be included in subsequent /leads calls but rejected by /validate
                import hashlib
                import json
                
                # Compute queue_root (hash of lead IDs for verification)
                queue_root = hashlib.sha256(json.dumps(assigned_lead_ids, sort_keys=True).encode()).hexdigest()
                
                # Create EPOCH_INITIALIZATION payload (MUST match epoch_lifecycle format)
                # CRITICAL: epoch_id must be INSIDE payload because validate.py queries:
                #   .eq("payload->>epoch_id", str(event.payload.epoch_id))
                init_payload = {
                    "epoch_id": epoch_id,  # CRITICAL: Must be inside payload for validate.py query
                    "queue": {
                        "queue_root": queue_root,
                        "lead_count": len(assigned_lead_ids)
                    },
                    "assignment": {
                        "assigned_lead_ids": assigned_lead_ids,
                        "validator_count": 0,  # Unknown at this point
                        "leads_per_validator": len(assigned_lead_ids)
                    },
                    "created_by": "epoch_leads_fallback",  # Mark as fallback-created
                    "created_at": datetime.utcnow().isoformat()
                }
                
                print(f"   📝 Creating EPOCH_INITIALIZATION for epoch {epoch_id} (fallback mode)...")
                print(f"   📊 Queue Root: {queue_root[:16]}...")
                
                try:
                    # Insert EPOCH_INITIALIZATION event (idempotent - will fail if exists)
                    # NOTE: epoch_id is INSIDE init_payload, not as a column (column doesn't exist)
                    # CRITICAL: Must include actor_hotkey="system" - NOT NULL constraint in DB
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: supabase.table("transparency_log")
                                .insert({
                                    "event_type": "EPOCH_INITIALIZATION",
                                    "actor_hotkey": "system",  # System-generated event (required NOT NULL)
                                    "payload": init_payload,
                                    "created_at": datetime.utcnow().isoformat()
                                })
                                .execute()
                        ),
                        timeout=30.0
                    )
                    print(f"   ✅ EPOCH_INITIALIZATION created successfully")
                except Exception as init_err:
                    # If it already exists (race condition), that's fine - use existing one
                    if "duplicate" in str(init_err).lower() or "unique" in str(init_err).lower():
                        print(f"   ℹ️  EPOCH_INITIALIZATION already exists (created by another request)")
                    elif "null value" in str(init_err).lower() or "actor_hotkey" in str(init_err).lower():
                        # This means epoch_lifecycle already created it, but we hit a race condition
                        # where our query didn't find it yet. RETRY the query instead of proceeding
                        # with potentially stale leads from the wrong epoch!
                        print(f"   ⚠️  Failed to create EPOCH_INITIALIZATION (actor_hotkey constraint)")
                        print(f"   🔄 Retrying query for existing EPOCH_INITIALIZATION (may have been created by epoch_lifecycle)...")
                        
                        try:
                            retry_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    lambda: supabase.table("transparency_log")
                                        .select("payload")
                                        .eq("event_type", "EPOCH_INITIALIZATION")
                                        .eq("payload->>epoch_id", str(epoch_id))
                                        .limit(1)
                                        .execute()
                                ),
                                timeout=30.0
                            )
                            
                            if retry_result and retry_result.data:
                                # Found it! Use the correct EPOCH_INITIALIZATION data
                                epoch_payload = retry_result.data[0].get("payload", {})
                                assigned_lead_ids = epoch_payload.get("assignment", {}).get("assigned_lead_ids", [])
                                queue_root = epoch_payload.get("queue", {}).get("queue_root", "unknown")
                                validator_count = epoch_payload.get("assignment", {}).get("validator_count", 0)
                                
                                print(f"   ✅ RETRY SUCCESS: Found EPOCH_INITIALIZATION with {len(assigned_lead_ids)} leads")
                                print(f"   🔄 Discarding fallback query results, using official EPOCH_INITIALIZATION")
                                
                                # Set leads_result to None to force re-query in Step 5 with correct lead IDs
                                leads_result = None
                            else:
                                # Still can't find it - this is a critical error
                                print(f"   ❌ RETRY FAILED: EPOCH_INITIALIZATION still not found after retry")
                                print(f"   ❌ Cannot proceed - serving wrong epoch leads would cause validation failures")
                                raise HTTPException(
                                    status_code=503,
                                    detail=f"EPOCH_INITIALIZATION not found for epoch {epoch_id} - gateway may be initializing"
                                )
                        except asyncio.TimeoutError:
                            print(f"   ❌ RETRY TIMEOUT: Cannot verify EPOCH_INITIALIZATION exists")
                            raise HTTPException(
                                status_code=504,
                                detail="Gateway timeout while verifying epoch initialization"
                            )
                    else:
                        # Unknown error - don't risk serving wrong leads
                        print(f"   ❌ Failed to create EPOCH_INITIALIZATION: {init_err}")
                        print(f"   ❌ Cannot proceed - serving wrong epoch leads would cause validation failures")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to initialize epoch {epoch_id}: {str(init_err)}"
                        )
            else:
                assigned_lead_ids = []
                print(f"   ℹ️  No pending leads in queue")
                
        except asyncio.TimeoutError:
            print(f"❌ ERROR: Direct query timed out after 90 seconds")
            raise HTTPException(
                status_code=504,
                detail="Database query timeout - gateway may be experiencing high load"
            )
        except Exception as e:
            print(f"❌ ERROR in direct query: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch leads: {str(e)}"
            )
    
    # Step 5: Fetch full lead data from leads_private (skip if already fetched in fallback)
    if not assigned_lead_ids:
        # No leads assigned for this epoch
        return {
            "epoch_id": epoch_id,
            "leads": [],
            "queue_root": queue_root,
            "validator_count": validator_count,
            "max_leads_per_epoch": MAX_LEADS_PER_EPOCH  # Dynamic config for validators
        }
    
    # Only query leads_private if we used EPOCH_INITIALIZATION (not fallback)
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
                    return lambda: supabase.table("leads_private").select("lead_id, lead_blob, lead_blob_hash").in_("lead_id", ids).execute()
                
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
    else:
        print(f"✅ Step 5: Skipped (already have {len(leads_result.data)} leads from fallback)")
    
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

