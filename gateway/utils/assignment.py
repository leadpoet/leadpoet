"""
Deterministic Lead Assignment Utilities

Assigns the first 50 leads from the queue to ALL validators.

Key Principle: SIMPLE & TRANSPARENT
All validators receive the same 50 leads (FIFO from queue):
- First 50 leads from queue (ordered by created_ts)
- Same leads assigned to ALL active validators
- No VRF, no shuffling, no complex capacity calculations

This ensures:
1. Simple, transparent assignment
2. All validators validate the same leads
3. Easy to verify and debug
4. Fair FIFO processing
"""

import hashlib
import random
from typing import List, Tuple
from uuid import UUID

from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, BITTENSOR_NETWORK, BITTENSOR_NETUID
from supabase import create_client
import bittensor as bt

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def deterministic_lead_assignment(
    queue_root: str,
    validator_set: List[str],
    epoch_id: int,
    max_leads_per_epoch: int = 50
) -> List[str]:
    """
    Assign first 50 leads from FIFO queue to ALL validators.
    
    All validators receive the SAME 50 leads for a given epoch.
    Simple FIFO (First In, First Out) assignment - no shuffling, no VRF.
    
    **FIFO Queue Processing**:
    - Leads are selected in strict FIFO order based on submission_timestamp (oldest first)
    - The first 50 pending leads from the queue are assigned for validation
    - Leads beyond the first 50 remain in the queue with their original timestamps preserved
    - They will be processed in future epochs, maintaining their timestamp priority
    - New submissions are immediately appended to the back of the queue
    - Fair, timestamp-based ordering is maintained across all submissions regardless of arrival time
    
    Algorithm:
    1. Fetch pending leads from database (ordered by created_ts ASCENDING)
    2. Take first 50 leads (oldest first)
    3. Return lead_ids - ALL validators get the same 50 leads
    
    Args:
        queue_root: Merkle root of pending leads queue (from QUEUE_ROOT event) - for verification only
        validator_set: List of validator SS58 addresses (from get_validator_set) - for logging only
        epoch_id: Current epoch number
        max_leads_per_epoch: Max leads per epoch (default: 50, same for ALL validators)
    
    Returns:
        List of lead_ids (UUID strings) - first 50 leads from queue, oldest first
    
    Example:
        >>> queue_root = "abc123..."
        >>> validators = ["5GNJqR...", "5FHneW..."]  # ALL validators get same leads
        >>> epoch_id = 100
        >>> leads = deterministic_lead_assignment(queue_root, validators, epoch_id)
        >>> leads
        ['550e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440002', ...]  # 50 leads
    
    Notes:
        - Simple FIFO: oldest 50 leads assigned to ALL validators
        - No shuffling, no VRF, no capacity calculations
        - Transparent and easy to verify
        - Any leads beyond first 50 remain in queue for next epoch with original timestamps
    """
    # Handle edge case: no validators
    if not validator_set or len(validator_set) == 0:
        print("⚠️  No validators in set - no leads assigned")
        return []
    
    # Step 1: Fetch pending leads from Private DB (ordered by created_ts ASCENDING = FIFO)
    # CRITICAL: Supabase has a 1000-row limit PER REQUEST (both .limit() and .range())
    # To fetch more than 1000 rows, we MUST use pagination.
    try:
        pending_leads = []
        batch_size = 500  # Stay well under 1000 limit for safety
        offset = 0
        rows_needed = max_leads_per_epoch + 1  # +1 to detect backlog
        
        while len(pending_leads) < rows_needed:
            # Calculate range for this batch
            start = offset
            end = min(offset + batch_size - 1, rows_needed - 1)
            
            result = supabase.table("leads_private") \
                .select("lead_id, created_ts") \
                .eq("status", "pending_validation") \
                .order("created_ts") \
                .range(start, end) \
                .execute()
            
            if not result.data:
                break  # No more data
            
            pending_leads.extend([row["lead_id"] for row in result.data])
            
            # If we got fewer rows than requested, we've reached the end
            if len(result.data) < (end - start + 1):
                break
            
            offset += batch_size
        
        print(f"   📊 Fetched {len(pending_leads)} pending leads (requested up to {rows_needed})")
    
    except Exception as e:
        print(f"❌ Failed to fetch pending leads: {e}")
        return []
    
    # Handle edge case: no pending leads
    if not pending_leads:
        print("ℹ️  No pending leads in queue")
        return []
    
    # Step 2: Take first 50 leads (oldest first, FIFO)
    # ALL validators get the same 50 leads - no shuffling, no VRF, no capacity calculations
    assigned_leads = pending_leads[:max_leads_per_epoch]
    
    validator_count = len(validator_set)
    
    print(f"📊 Lead assignment for epoch {epoch_id}:")
    print(f"   Queue root: {queue_root[:16]}...")
    print(f"   Pending leads: {len(pending_leads)}")
    print(f"   Validators: {validator_count}")
    print(f"   Assigned: {len(assigned_leads)} leads (first {max_leads_per_epoch} from queue)")
    print(f"   → ALL {validator_count} validators get the same {len(assigned_leads)} leads")
    
    if len(pending_leads) > max_leads_per_epoch:
        backlog = len(pending_leads) - max_leads_per_epoch
        print(f"   ⚠️  Backlog: {backlog} leads waiting for next epoch")
    
    return assigned_leads


async def get_validator_set(epoch_id: int) -> List[str]:
    """
    Get list of eligible validators for given epoch (ASYNC VERSION).
    
    Queries Bittensor metagraph to find all registered validators.
    
    Args:
        epoch_id: Epoch number (for logging/debugging)
    
    Returns:
        List of validator hotkeys (SS58 addresses)
    
    Example:
        >>> validators = await get_validator_set(100)
        >>> validators
        ['5GNJqR7T...', '5FHneW46...', '5EPCUjPx...']
    
    Notes:
        - Validator = active=True AND validator_permit=True
        - Uses cached metagraph (epoch-based)
        - Returns empty list if metagraph unavailable
    """
    try:
        # Use the registry utility's cached metagraph (async version)
        from gateway.utils.registry import get_metagraph_async
        
        metagraph = await get_metagraph_async()
        
        # Filter validators (active + validator_permit OR stake > 500K + permit)
        STAKE_THRESHOLD = 500000  # 500K TAO minimum
        
        validators = []
        for i, hotkey in enumerate(metagraph.hotkeys):
            # Validators must have:
            # 1. BOTH active=True AND validator_permit=True (normal path), OR
            # 2. Stake > 500K TAO AND validator_permit=True (temporary stake-based override)
            active = metagraph.active[i]
            validator_permit = metagraph.validator_permit[i]
            stake = metagraph.S[i]
            
            if (active and validator_permit) or (stake > STAKE_THRESHOLD and validator_permit):
                validators.append(hotkey)
        
        print(f"📊 Validator set for epoch {epoch_id}:")
        print(f"   Total registered: {len(metagraph.hotkeys)}")
        print(f"   Validators (active+permit OR stake>500K+permit): {len(validators)}")
        print(f"   Miners: {len(metagraph.hotkeys) - len(validators)}")
        
        return validators
    
    except Exception as e:
        print(f"❌ Failed to get validator set: {e}")
        import traceback
        traceback.print_exc()
        return []


def verify_lead_in_assignment(
    lead_id: str,
    queue_root: str,
    validator_set: List[str],
    epoch_id: int,
    max_leads_per_epoch: int = 50
) -> bool:
    """
    Verify that a specific lead is in the deterministic assignment.
    
    Useful for validators to verify they should validate a specific lead,
    or for checking if another validator's claim is valid.
    
    Args:
        lead_id: Lead UUID to verify
        queue_root: Queue root for the epoch
        validator_set: List of validators
        epoch_id: Epoch number
        max_leads_per_epoch: Max leads per epoch (default: 50)
    
    Returns:
        True if lead_id is in the assigned set for this epoch
    
    Example:
        >>> lead_id = "550e8400-e29b-41d4-a716-446655440001"
        >>> is_assigned = verify_lead_in_assignment(lead_id, queue_root, validators, 100)
        >>> is_assigned
        True
    """
    assigned_leads = deterministic_lead_assignment(
        queue_root, validator_set, epoch_id, max_leads_per_epoch
    )
    
    return lead_id in assigned_leads


def get_lead_assignment_index(
    lead_id: str,
    queue_root: str,
    validator_set: List[str],
    epoch_id: int,
    max_leads_per_epoch: int = 50
) -> int:
    """
    Get the index of a lead in the assignment (for debugging/ordering).
    
    Returns -1 if lead is not in the assignment.
    
    Args:
        lead_id: Lead UUID
        queue_root: Queue root for the epoch
        validator_set: List of validators
        epoch_id: Epoch number
        max_leads_per_epoch: Max leads per epoch (default: 50)
    
    Returns:
        Index of lead in assignment (0-indexed), or -1 if not assigned
    
    Example:
        >>> idx = get_lead_assignment_index(lead_id, queue_root, validators, 100)
        >>> idx
        42  # 43rd lead in assignment
    """
    assigned_leads = deterministic_lead_assignment(
        queue_root, validator_set, epoch_id, max_leads_per_epoch
    )
    
    try:
        return assigned_leads.index(lead_id)
    except ValueError:
        return -1


def estimate_epoch_capacity(validator_count: int, max_leads_per_epoch: int = 50) -> int:
    """
    Estimate total lead capacity for an epoch.
    
    With the new simplified design, ALL validators get the same 50 leads,
    so capacity is just the max_leads_per_epoch (not multiplied by validator count).
    
    Args:
        validator_count: Number of active validators (not used in calculation, kept for API compatibility)
        max_leads_per_epoch: Max leads per epoch (default: 50)
    
    Returns:
        Total lead capacity for the epoch (always equals max_leads_per_epoch)
    
    Example:
        >>> capacity = estimate_epoch_capacity(10)
        >>> capacity
        50  # Same 50 leads assigned to all 10 validators
    
    Notes:
        - ALL validators get the same 50 leads
        - Capacity is NOT multiplied by validator count anymore
        - This function kept for API compatibility
    """
    return max_leads_per_epoch


def get_assignment_stats(
    queue_root: str,
    validator_set: List[str],
    epoch_id: int,
    max_leads_per_epoch: int = 50
) -> dict:
    """
    Get comprehensive statistics about lead assignment for an epoch.
    
    Args:
        queue_root: Queue root for the epoch
        validator_set: List of validators
        epoch_id: Epoch number
        max_leads_per_epoch: Max leads per epoch (default: 50)
    
    Returns:
        Dictionary with assignment statistics
    
    Example:
        >>> stats = get_assignment_stats(queue_root, validators, 100)
        >>> stats
        {
            'epoch_id': 100,
            'validator_count': 10,
            'max_capacity': 50,
            'pending_leads': 150,
            'assigned_leads': 50,
            'backlog': 100,
            'utilization': 1.0  # 100%
        }
    """
    assigned_leads = deterministic_lead_assignment(
        queue_root, validator_set, epoch_id, max_leads_per_epoch
    )
    
    # Fetch total pending
    try:
        result = supabase.table("leads_private") \
            .select("lead_id", count="exact") \
            .eq("status", "pending_validation") \
            .execute()
        
        pending_count = result.count if result.count is not None else 0
    except:
        pending_count = 0
    
    validator_count = len(validator_set)
    max_capacity = estimate_epoch_capacity(validator_count, max_leads_per_epoch)
    assigned_count = len(assigned_leads)
    backlog = max(0, pending_count - assigned_count)
    utilization = assigned_count / max_capacity if max_capacity > 0 else 0
    
    return {
        'epoch_id': epoch_id,
        'validator_count': validator_count,
        'max_capacity': max_capacity,
        'pending_leads': pending_count,
        'assigned_leads': assigned_count,
        'backlog': backlog,
        'utilization': round(utilization, 3)
    }

