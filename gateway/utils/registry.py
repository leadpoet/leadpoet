"""
Registry Check Utility (Spam Prevention)
========================================

Verifies that actors (miners/validators) are registered on-chain
before allowing access to the gateway.

This prevents spam from unregistered hotkeys.
"""

import bittensor as bt
from typing import Optional, Tuple
import time
import threading

# Import configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gateway.config import BITTENSOR_NETWORK, BITTENSOR_NETUID

# Cache for metagraph (epoch-based invalidation)
_metagraph_cache = None
_cache_epoch = None  # Track which epoch the cache is from
_cache_lock = threading.Lock()  # Prevent simultaneous fetches


def get_metagraph() -> bt.metagraph:
    """
    Get Bittensor metagraph (cached per epoch).
    
    The metagraph is cached for the duration of the current epoch and refreshed
    when a new epoch begins. This ensures:
    1. Multiple validators requesting leads in the same epoch use the same metagraph
    2. New registrations are picked up at epoch boundaries
    3. Thread-safe access prevents simultaneous fetches
    
    Returns:
        Metagraph object for the configured subnet
    
    Raises:
        Exception: If unable to connect to subtensor or fetch metagraph
    
    Example:
        >>> metagraph = get_metagraph()
        >>> print(f"Subnet has {len(metagraph.hotkeys)} neurons")
    
    Notes:
        - Cache is invalidated when epoch changes (every 360 blocks = 72 minutes)
        - Thread lock prevents multiple simultaneous fetches during epoch transition
        - Includes active status and validator_permit for each neuron
    """
    global _metagraph_cache, _cache_epoch
    
    # Import here to avoid circular dependency
    from gateway.utils.epoch import get_current_epoch_id
    
    current_epoch = get_current_epoch_id()
    
    # Thread-safe cache check and update
    with _cache_lock:
        # Return cached metagraph if it's from the current epoch
        if _metagraph_cache is not None and _cache_epoch == current_epoch:
            print(f"✅ Using cached metagraph for epoch {current_epoch} ({len(_metagraph_cache.hotkeys)} neurons)")
            return _metagraph_cache
        
        # Cache expired (epoch changed) or not set - fetch new metagraph
        # Use retry logic to handle rate limiting and transient failures
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if _cache_epoch is not None and _cache_epoch != current_epoch:
                    print(f"🔄 Epoch changed: {_cache_epoch} → {current_epoch}")
                    print(f"🔄 Refreshing metagraph for new epoch... (attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"🔄 Fetching metagraph for epoch {current_epoch}... (attempt {attempt + 1}/{max_retries})")
                
                print(f"   Network: {BITTENSOR_NETWORK}, NetUID: {BITTENSOR_NETUID}")
                
                # Create subtensor connection
                subtensor = bt.subtensor(network=BITTENSOR_NETWORK)
                
                # Fetch metagraph for configured subnet
                metagraph = subtensor.metagraph(netuid=BITTENSOR_NETUID)
                
                # Update cache
                _metagraph_cache = metagraph
                _cache_epoch = current_epoch
                
                print(f"✅ Metagraph cached for epoch {current_epoch}: {len(metagraph.hotkeys)} neurons registered")
                
                return metagraph
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠️  Metagraph fetch attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"   Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                # Last attempt failed - fall through to error handling below
        
        # All retries exhausted
        print(f"❌ Error fetching metagraph after {max_retries} attempts: {last_error}")
        
        # If we have a cache from previous epoch, use it with warning
        if _metagraph_cache is not None:
            print(f"⚠️  Using metagraph from previous epoch {_cache_epoch} as fallback")
            print(f"⚠️  This may not include validators who registered in epoch {current_epoch}")
            return _metagraph_cache
        
        # No cache available - raise error
        raise Exception(f"Failed to fetch metagraph and no cache available: {last_error}")


def is_registered_hotkey(hotkey: str) -> Tuple[bool, Optional[str]]:
    """
    Check if hotkey is registered on the subnet.
    
    Args:
        hotkey: SS58 address of the actor
    
    Returns:
        (is_registered, role) where:
        - is_registered: True if hotkey exists in metagraph
        - role: "validator" if stake > 0, "miner" if stake == 0, None if not registered
    
    Example:
        >>> is_registered, role = is_registered_hotkey("5GNJqR7T...")
        >>> if is_registered:
        >>>     print(f"Hotkey is a {role}")
        >>> else:
        >>>     print("Hotkey not registered")
    
    Notes:
        - Validators are identified by: active=True AND validator_permit=True
        - Miners: Either active=False OR validator_permit=False (can have stake > 0)
        - Uses cached metagraph (refreshed every 60 seconds)
    """
    try:
        # Get metagraph (cached)
        metagraph = get_metagraph()
        
        # Check if hotkey exists in metagraph
        if hotkey not in metagraph.hotkeys:
            print(f"🔍 Registry check: {hotkey[:20]}... NOT FOUND in metagraph")
            return False, None
        
        # Get UID for this hotkey
        uid = metagraph.hotkeys.index(hotkey)
        
        # Get neuron attributes
        stake = metagraph.S[uid]
        # Cast numpy bools to Python bools for consistent display
        active = bool(metagraph.active[uid])
        validator_permit = bool(metagraph.validator_permit[uid])
        
        print(f"🔍 Registry check for {hotkey[:20]}...")
        print(f"   UID: {uid}")
        print(f"   Stake: {stake:.6f} τ")
        print(f"   Active: {active}")
        print(f"   Validator Permit: {validator_permit}")
        
        # Validators must have BOTH active status AND validator permit
        # A miner can have stake > 0 but still be a miner if they lack permit
        if active and validator_permit:
            role = "validator"
            print(f"   ✅ Role: VALIDATOR (active=True, permit=True)")
        else:
            role = "miner"
            print(f"   ✅ Role: MINER (active={active}, permit={validator_permit})")
        
        return True, role
    
    except Exception as e:
        print(f"❌ Registry check error for {hotkey[:20]}...: {e}")
        return False, None


def get_validator_count() -> int:
    """
    Get the number of registered validators on the subnet.
    
    Returns:
        Number of validators (neurons with active=True AND validator_permit=True)
    
    Example:
        >>> count = get_validator_count()
        >>> print(f"Subnet has {count} validators")
    """
    try:
        metagraph = get_metagraph()
        
        # Count neurons with active status AND validator permit
        validator_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if metagraph.active[i] and metagraph.validator_permit[i]
        )
        
        return validator_count
    
    except Exception as e:
        print(f"❌ Error getting validator count: {e}")
        return 0


def get_miner_count() -> int:
    """
    Get the number of registered miners on the subnet.
    
    Returns:
        Number of miners (neurons without active status OR without validator permit)
    
    Example:
        >>> count = get_miner_count()
        >>> print(f"Subnet has {count} miners")
    """
    try:
        metagraph = get_metagraph()
        
        # Count neurons that are NOT validators
        # (either not active OR no validator permit)
        miner_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if not (metagraph.active[i] and metagraph.validator_permit[i])
        )
        
        return miner_count
    
    except Exception as e:
        print(f"❌ Error getting miner count: {e}")
        return 0


def clear_metagraph_cache():
    """
    Clear the metagraph cache.
    
    Forces a fresh fetch on the next get_metagraph() call.
    Useful for testing or when you know the metagraph has changed.
    
    Example:
        >>> clear_metagraph_cache()
        >>> metagraph = get_metagraph()  # Will fetch fresh data
    """
    global _metagraph_cache, _cache_epoch
    with _cache_lock:
        _metagraph_cache = None
        _cache_epoch = None
        print("🗑️  Metagraph cache cleared")


def print_registry_stats():
    """
    Print statistics about the subnet registry.
    
    Shows total neurons, validators, and miners.
    Useful for debugging and monitoring.
    """
    try:
        metagraph = get_metagraph()
        
        total_neurons = len(metagraph.hotkeys)
        validator_count = get_validator_count()
        miner_count = get_miner_count()
        
        print("=" * 60)
        print("Subnet Registry Statistics")
        print("=" * 60)
        print(f"Network: {BITTENSOR_NETWORK}")
        print(f"Netuid: {BITTENSOR_NETUID}")
        print(f"Total Neurons: {total_neurons}")
        print(f"Validators: {validator_count} ({validator_count/total_neurons*100:.1f}%)")
        print(f"Miners: {miner_count} ({miner_count/total_neurons*100:.1f}%)")
        print(f"Cache Epoch: {_cache_epoch if _cache_epoch is not None else 'Not cached'}")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error printing registry stats: {e}")


def get_validator_weights(validator_hotkey: str) -> tuple[float, float]:
    """
    Get validator's stake and v_trust from metagraph.
    
    This is called during COMMIT phase to snapshot validator weights.
    Critical: Must capture at COMMIT time, not REVEAL time, to prevent gaming
    (validator could unstake after seeing other decisions but before revealing).
    
    Args:
        validator_hotkey: Validator's SS58 address
    
    Returns:
        (stake, v_trust): Tuple of (TAO stake, validator trust score)
        Returns (0.0, 0.0) if validator not found or not active
    
    Example:
        >>> stake, v_trust = get_validator_weights("5FNVgRnrx...")
        >>> print(f"Stake: {stake:.6f} τ, V-Trust: {v_trust:.6f}")
    """
    try:
        metagraph = get_metagraph()
        
        # Find validator's UID
        if validator_hotkey not in metagraph.hotkeys:
            print(f"⚠️  Validator {validator_hotkey[:20]}... not found in metagraph")
            return (0.0, 0.0)
        
        uid = metagraph.hotkeys.index(validator_hotkey)
        
        # Get stake (TAO amount)
        stake = float(metagraph.S[uid])
        
        # Get v_trust (validator trust/reputation)
        # This is Bittensor's internal validator trust score (validator_trust in metagraph)
        # Defaults to 0.0 if validator has no trust yet
        v_trust = float(metagraph.validator_trust[uid]) if hasattr(metagraph, 'validator_trust') else 0.0
        
        print(f"📊 Validator weights for {validator_hotkey[:20]}...")
        print(f"   Stake: {stake:.6f} τ")
        print(f"   V-Trust: {v_trust:.6f}")
        
        return (stake, v_trust)
    
    except Exception as e:
        print(f"❌ Error fetching validator weights: {e}")
        return (0.0, 0.0)

