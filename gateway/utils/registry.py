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

# Import configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gateway.config import BITTENSOR_NETWORK, BITTENSOR_NETUID

# Cache for metagraph (1 minute TTL)
_metagraph_cache = None
_cache_timestamp = None
_CACHE_TTL = 60  # 1 minute in seconds


def get_metagraph() -> bt.metagraph:
    """
    Get Bittensor metagraph (cached for 1 minute).
    
    Caching reduces load on subtensor node and improves gateway performance.
    
    Returns:
        Metagraph object for the configured subnet
    
    Raises:
        Exception: If unable to connect to subtensor or fetch metagraph
    
    Example:
        >>> metagraph = get_metagraph()
        >>> print(f"Subnet has {len(metagraph.hotkeys)} neurons")
    """
    global _metagraph_cache, _cache_timestamp
    
    now = time.time()
    
    # Return cached metagraph if still valid
    if _metagraph_cache is not None and _cache_timestamp is not None:
        age = now - _cache_timestamp
        if age < _CACHE_TTL:
            return _metagraph_cache
    
    # Cache expired or not set - fetch new metagraph
    try:
        print(f"🔄 Fetching metagraph for {BITTENSOR_NETWORK} subnet {BITTENSOR_NETUID}...")
        
        # Create subtensor connection
        subtensor = bt.subtensor(network=BITTENSOR_NETWORK)
        
        # Fetch metagraph for configured subnet
        metagraph = subtensor.metagraph(netuid=BITTENSOR_NETUID)
        
        # Update cache
        _metagraph_cache = metagraph
        _cache_timestamp = now
        
        print(f"✅ Metagraph fetched: {len(metagraph.hotkeys)} neurons registered")
        
        return metagraph
    
    except Exception as e:
        print(f"❌ Error fetching metagraph: {e}")
        
        # If we have a stale cache, return it rather than failing
        if _metagraph_cache is not None:
            print(f"⚠️  Using stale metagraph cache (age: {now - _cache_timestamp:.0f}s)")
            return _metagraph_cache
        
        # No cache available - raise error
        raise Exception(f"Failed to fetch metagraph and no cache available: {e}")


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
    global _metagraph_cache, _cache_timestamp
    _metagraph_cache = None
    _cache_timestamp = None
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
        print(f"Cache Age: {time.time() - _cache_timestamp:.1f}s")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Error printing registry stats: {e}")

