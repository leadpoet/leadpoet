"""
Epoch Management Utilities for LeadPoet Gateway

Manages 360-block validation epochs based on Bittensor block numbers.
This matches the epoch calculation in Leadpoet/validator/reward.py

Epoch Timeline:
- Blocks 0-360: Active validation phase (validators submit results anytime)
- Block 360+: Epoch closed (reveal phase begins, consensus computed, next epoch begins)

Epochs are synchronized with Bittensor blockchain blocks.
Each block = 12 seconds, so 360 blocks = 72 minutes total.
"""

from datetime import datetime, timedelta
import math
import os

# Configuration constants (must match validator/reward.py)
EPOCH_DURATION_BLOCKS = 360  # 72 minutes = 360 blocks × 12 sec
BITTENSOR_BLOCK_TIME_SECONDS = 12

# Network for epoch tracking (from environment variable)
_epoch_network = os.getenv("BITTENSOR_NETWORK", "finney")

# Block caching for resilient estimation (must match validator/reward.py)
_last_known_block = None
_last_known_block_time = None
import threading
_block_cache_lock = threading.Lock()


def _get_current_block() -> int:
    """
    Get current block number from Bittensor subtensor.
    Falls back to cached block + time-based estimation if subtensor unavailable.
    
    Uses retry logic to handle transient failures and rate limiting.
    This matches the logic in Leadpoet/validator/reward.py
    """
    global _last_known_block, _last_known_block_time
    
    # Retry logic for subtensor queries (handles HTTP 429 rate limits)
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            import bittensor as bt
            import time
            
            # Create subtensor connection
            subtensor = bt.subtensor(network=_epoch_network)
            current_block = subtensor.get_current_block()
            
            # Cache the successful result
            with _block_cache_lock:
                _last_known_block = current_block
                _last_known_block_time = time.time()
            
            return current_block
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Retry after delay
                print(f"⚠️  Subtensor query attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                # All retries exhausted - use cached estimation
                print(f"⚠️  Cannot get current block from subtensor after {max_retries} attempts: {e}")
                
                with _block_cache_lock:
                    if _last_known_block is not None and _last_known_block_time is not None:
                        # Calculate blocks elapsed since last known good block
                        time_elapsed = time.time() - _last_known_block_time
                        blocks_elapsed = int(time_elapsed / BITTENSOR_BLOCK_TIME_SECONDS)
                        estimated_block = _last_known_block + blocks_elapsed
                        
                        print(f"   Using cached block estimation:")
                        print(f"   Last known block: {_last_known_block} (cached {int(time_elapsed)}s ago)")
                        print(f"   Estimated current: {estimated_block} (+{blocks_elapsed} blocks)")
                        return estimated_block
                    else:
                        # No cache available - this should only happen on first run
                        raise Exception(
                            "Cannot query subtensor and no cached block available. "
                            "Please ensure subtensor is accessible."
                        )


def get_current_epoch_id() -> int:
    """
    Calculate current epoch ID based on Bittensor block number.
    
    This matches the calculation in Leadpoet/validator/reward.py:
        epoch_id = block_number // EPOCH_DURATION_BLOCKS
    
    Returns:
        Epoch ID (integer, 0-indexed)
    
    Example:
        >>> get_current_epoch_id()
        5678  # Current epoch based on block number
    """
    current_block = _get_current_block()
    epoch_id = current_block // EPOCH_DURATION_BLOCKS
    return epoch_id


def get_block_within_epoch() -> int:
    """
    Get the current block number within the current epoch (0-359).
    
    This is used to determine the validation phase timing:
    - Blocks 0-350: Lead distribution window (gateway sends leads)
    - Blocks 351-355: Validation submission window (validators submit results)
    - Blocks 356-359: Buffer period (no new submissions)
    - Block 360+: Epoch closed (next epoch begins)
    
    Returns:
        Block number within current epoch (0-359)
    
    Example:
        >>> get_block_within_epoch()
        145  # Current block is 145 blocks into the epoch
    """
    current_block = _get_current_block()
    block_within_epoch = current_block % EPOCH_DURATION_BLOCKS
    return block_within_epoch


def get_epoch_start_time(epoch_id: int) -> datetime:
    """
    Get UTC start time for given epoch ID based on block number.
    
    Uses current block and current time as reference, then calculates offset.
    
    Args:
        epoch_id: Epoch number (0-indexed)
    
    Returns:
        datetime (UTC) when epoch starts
    
    Example:
        >>> get_epoch_start_time(18895)
        datetime(2025, 11, 3, ...)  # Calculated from current block offset
    """
    try:
        # Get current reference point
        current_block = _get_current_block()
        now = datetime.utcnow()
        
        # Calculate the start block of the target epoch
        target_start_block = epoch_id * EPOCH_DURATION_BLOCKS
        
        # Calculate blocks from target epoch start to current block
        # Negative offset = epoch started in the past
        # Positive offset = epoch starts in the future
        block_offset = target_start_block - current_block
        
        # Convert block offset to time offset
        time_offset_seconds = block_offset * BITTENSOR_BLOCK_TIME_SECONDS
        
        # Apply offset to current time
        epoch_start = now + timedelta(seconds=time_offset_seconds)
        
        return epoch_start
        
    except Exception as e:
        # Fallback: return current time
        print(f"⚠️  Error calculating epoch start time: {e}")
        return datetime.utcnow()


def get_epoch_end_time(epoch_id: int) -> datetime:
    """
    Get UTC end time for given epoch ID (when validation phase stops at block 360).
    
    Validation phase lasts the full 360 blocks - validators can submit anytime during the epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when validation phase ends (block 360)
    
    Example:
        >>> start = get_epoch_start_time(100)
        >>> end = get_epoch_end_time(100)
        >>> (end - start).total_seconds()
        4320.0  # 360 blocks × 12 sec = 72 minutes
    """
    start = get_epoch_start_time(epoch_id)
    # Active validation = full 360 blocks (72 minutes)
    end = start + timedelta(seconds=EPOCH_DURATION_BLOCKS * BITTENSOR_BLOCK_TIME_SECONDS)
    
    return end


def get_epoch_close_time(epoch_id: int) -> datetime:
    """
    Get UTC close time for given epoch ID (at block 360).
    
    Epoch closes at block 360, triggering reveal phase and consensus computation.
    This is the same as get_epoch_end_time() - kept for API compatibility.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        datetime (UTC) when epoch closes (block 360)
    
    Example:
        >>> start = get_epoch_start_time(100)
        >>> close = get_epoch_close_time(100)
        >>> (close - start).total_seconds()
        4320.0  # 360 blocks × 12 sec = 72 minutes
    """
    # No grace period - epoch closes at block 360, same as end time
    return get_epoch_end_time(epoch_id)


def is_epoch_active(epoch_id: int) -> bool:
    """
    Check if epoch is currently in validation phase (blocks 0-360).
    
    During active phase:
    - Validators can fetch assigned leads
    - Validators can submit validation results (commit phase) anytime
    - New leads can be added to queue
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if validators can submit validation results
    
    Example:
        >>> is_epoch_active(get_current_epoch_id())
        True  # If current time is within validation window (blocks 0-360)
    """
    now = datetime.utcnow()
    start = get_epoch_start_time(epoch_id)
    end = get_epoch_end_time(epoch_id)
    
    return start <= now <= end


def is_epoch_in_grace_period(epoch_id: int) -> bool:
    """
    Check if epoch is in grace period (DEPRECATED - no grace period).
    
    Grace period has been removed from the design. This function always returns False
    and is kept only for API compatibility with existing code.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Always returns False (no grace period in current design)
    
    Example:
        >>> is_epoch_in_grace_period(get_current_epoch_id())
        False  # No grace period exists
    """
    # No grace period - validators can submit anytime during blocks 0-360
    return False


def is_epoch_closed(epoch_id: int) -> bool:
    """
    Check if epoch is closed (past block 360).
    
    After epoch closes:
    - Consensus is computed
    - Reveals are required
    - No more submissions accepted
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        True if epoch is fully closed
    
    Example:
        >>> is_epoch_closed(get_current_epoch_id() - 1)
        True  # Previous epoch is closed
    """
    now = datetime.utcnow()
    close = get_epoch_close_time(epoch_id)
    
    return now > close


def time_until_epoch_end(epoch_id: int) -> int:
    """
    Get seconds remaining until epoch validation phase ends.
    
    Useful for displaying countdown timers or scheduling tasks.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Seconds remaining (0 if already ended)
    
    Example:
        >>> time_until_epoch_end(get_current_epoch_id())
        3420  # 57 minutes remaining
    """
    now = datetime.utcnow()
    end = get_epoch_end_time(epoch_id)
    
    if now > end:
        return 0
    
    return int((end - now).total_seconds())


def time_until_epoch_close(epoch_id: int) -> int:
    """
    Get seconds remaining until epoch fully closes.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Seconds remaining (0 if already closed)
    
    Example:
        >>> time_until_epoch_close(get_current_epoch_id())
        3720  # 62 minutes remaining
    """
    now = datetime.utcnow()
    close = get_epoch_close_time(epoch_id)
    
    if now > close:
        return 0
    
    return int((close - now).total_seconds())


def get_epoch_phase(epoch_id: int) -> str:
    """
    Get current phase of the epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        "active" | "closed"
    
    Example:
        >>> get_epoch_phase(get_current_epoch_id())
        'active'
    """
    if is_epoch_active(epoch_id):
        return "active"
    else:
        return "closed"


def get_epoch_info(epoch_id: int) -> dict:
    """
    Get comprehensive information about an epoch.
    
    Args:
        epoch_id: Epoch number
    
    Returns:
        Dictionary with all epoch timing information
    
    Example:
        >>> info = get_epoch_info(100)
        >>> info
        {
            'epoch_id': 100,
            'start_time': '1970-01-06T00:00:00',
            'end_time': '1970-01-06T01:07:00',
            'close_time': '1970-01-06T01:12:00',
            'phase': 'closed',
            'is_active': False,
            'is_grace_period': False,
            'is_closed': True,
            'time_until_end': 0,
            'time_until_close': 0
        }
    """
    return {
        'epoch_id': epoch_id,
        'start_time': get_epoch_start_time(epoch_id).isoformat(),
        'end_time': get_epoch_end_time(epoch_id).isoformat(),
        'close_time': get_epoch_close_time(epoch_id).isoformat(),
        'phase': get_epoch_phase(epoch_id),
        'is_active': is_epoch_active(epoch_id),
        'is_grace_period': is_epoch_in_grace_period(epoch_id),
        'is_closed': is_epoch_closed(epoch_id),
        'time_until_end': time_until_epoch_end(epoch_id),
        'time_until_close': time_until_epoch_close(epoch_id)
    }

