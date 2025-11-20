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
_cache_epoch_timestamp = None  # Track when we last calculated the epoch
_cache_lock = threading.Lock()  # Prevent simultaneous fetches

# Epoch duration in seconds (360 blocks × 12 seconds/block = 4320 seconds = 72 minutes)
EPOCH_DURATION_SECONDS = 360 * 12

# Async subtensor instance (injected at gateway startup)
_async_subtensor = None


def inject_async_subtensor(async_subtensor):
    """
    Inject async subtensor instance at gateway startup.
    
    Called from main.py lifespan to provide shared AsyncSubtensor instance.
    This eliminates memory leaks and HTTP 429 errors from repeated instance creation.
    
    Args:
        async_subtensor: AsyncSubtensor instance from main.py lifespan
    
    Example:
        # In main.py lifespan:
        async with bt.AsyncSubtensor(network="finney") as async_sub:
            registry_utils.inject_async_subtensor(async_sub)
    """
    global _async_subtensor
    _async_subtensor = async_subtensor
    print(f"✅ AsyncSubtensor injected into registry utils (network: {_async_subtensor.network})")


async def get_metagraph_async() -> bt.metagraph:
    """
    Get Bittensor metagraph using injected async subtensor (ASYNC VERSION).
    
    Use this from async contexts (FastAPI endpoints, background tasks).
    For sync contexts, use get_metagraph() wrapper.
    
    The metagraph is cached for the duration of the current epoch and refreshed
    when a new epoch begins. This ensures:
    1. Multiple validators requesting leads in the same epoch use the same metagraph
    2. New registrations are picked up at epoch boundaries
    3. Thread-safe access prevents simultaneous fetches
    4. Single async subtensor (NO new instances created)
    
    Returns:
        Metagraph object for the configured subnet
    
    Raises:
        Exception: If async_subtensor not injected or unable to fetch metagraph
    """
    global _metagraph_cache, _cache_epoch, _cache_epoch_timestamp
    import time
    import asyncio
    
    if _async_subtensor is None:
        raise Exception(
            "AsyncSubtensor not injected - call inject_async_subtensor() first. "
            "This should be done in main.py lifespan."
        )
    
    # Thread-safe cache check and update
    with _cache_lock:
        # Fast path: Check if cache is still valid based on time
        # Only query subtensor if enough time has passed for a potential epoch change
        if _metagraph_cache is not None and _cache_epoch_timestamp is not None:
            time_since_cache = time.time() - _cache_epoch_timestamp
            
            # If less than 72 minutes (one epoch) have passed, cache is definitely still valid
            if time_since_cache < EPOCH_DURATION_SECONDS:
                # Cache is guaranteed to still be for the current epoch
                print(f"✅ Using cached metagraph for epoch {_cache_epoch} ({len(_metagraph_cache.hotkeys)} neurons) - {int(time_since_cache)}s old")
                return _metagraph_cache
        
        # Slow path: Need to query subtensor to determine current epoch
        # This only happens:
        # 1. On first request (no cache)
        # 2. After 72+ minutes have passed (potential epoch change)
        from gateway.utils.epoch import get_current_epoch_id_async
        current_epoch = await get_current_epoch_id_async()
        
        # Check if we already have a cached metagraph for this epoch
        if _metagraph_cache is not None and _cache_epoch == current_epoch:
            # Update timestamp to reset the 72-minute timer
            _cache_epoch_timestamp = time.time()
            print(f"✅ Using cached metagraph for epoch {current_epoch} ({len(_metagraph_cache.hotkeys)} neurons) - refreshed timestamp")
            return _metagraph_cache
        
        # Cache expired (epoch changed) or not set - fetch new metagraph
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
                
                # ════════════════════════════════════════════════════════════
                # CRITICAL: Use injected async subtensor (NO new instance!)
                # Run in thread pool to prevent blocking event loop
                # Add 60-second timeout to prevent hanging gateway
                # ════════════════════════════════════════════════════════════
                metagraph = await asyncio.wait_for(
                    asyncio.to_thread(
                        _async_subtensor.metagraph,
                        netuid=BITTENSOR_NETUID
                    ),
                    timeout=60.0  # 60s timeout - prevents hanging gateway
                )
                
                # Update cache
                _metagraph_cache = metagraph
                _cache_epoch = current_epoch
                _cache_epoch_timestamp = time.time()  # Record when we cached this epoch
                
                print(f"✅ Metagraph cached for epoch {current_epoch}: {len(metagraph.hotkeys)} neurons registered")
                
                return metagraph
            
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Metagraph fetch timed out after 60s")
                if attempt < max_retries - 1:
                    print(f"⚠️  Metagraph fetch attempt {attempt + 1}/{max_retries} TIMEOUT (60s)")
                    print(f"   Bittensor chain might be overloaded - retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                # Last attempt timed out - fall through to error handling below
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠️  Metagraph fetch attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                # Last attempt failed - fall through to error handling below
        
        # All retries exhausted
        print(f"❌ Error fetching metagraph after {max_retries} attempts: {last_error}")
        
        # If we have a cache from previous epoch, use it with warning
        if _metagraph_cache is not None:
            print(f"⚠️  Using metagraph from previous epoch {_cache_epoch} as fallback")
            print(f"⚠️  This may not include validators who registered in epoch {current_epoch}")
            
            # CRITICAL: Update timestamp to prevent retry spam on every request
            _cache_epoch_timestamp = time.time()
            print(f"⚠️  Fallback cache will be used for next 72 minutes (prevents retry spam)")
            
            return _metagraph_cache
        
        # No cache available - raise error
        raise Exception(f"Failed to fetch metagraph and no cache available: {last_error}")


def get_metagraph() -> bt.metagraph:
    """
    Get Bittensor metagraph (SYNC WRAPPER - prefer async version).
    
    DEPRECATED: This creates a temporary event loop. Use get_metagraph_async() from async contexts.
    
    Returns:
        Metagraph object for the configured subnet
    
    Raises:
        RuntimeError: If called from async context (use get_metagraph_async instead)
        Exception: If unable to fetch metagraph
    
    Example:
        >>> metagraph = get_metagraph()
        >>> print(f"Subnet has {len(metagraph.hotkeys)} neurons")
    """
    import asyncio
    
    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        # We're in async context - this is an error
        raise RuntimeError(
            "Called get_metagraph() from async context. "
            "Use 'await get_metagraph_async()' instead. "
            "This is more efficient and doesn't create a new event loop."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # We're in sync context - create temp loop and run async version
            return asyncio.run(get_metagraph_async())
        else:
            # Error was from our check above - re-raise it
            raise


async def is_registered_hotkey_async(hotkey: str) -> Tuple[bool, Optional[str]]:
    """
    Check if hotkey is registered on the subnet (ASYNC VERSION).
    
    Use this from async contexts (FastAPI endpoints). For sync, use is_registered_hotkey() wrapper.
    
    Args:
        hotkey: SS58 address of the actor
    
    Returns:
        (is_registered, role) where:
        - is_registered: True if hotkey exists in metagraph
        - role: "validator" if active=True AND permit=True, "miner" otherwise
    """
    try:
        # Get metagraph using async version (cached, no new instance)
        metagraph = await get_metagraph_async()
        
        # Check if hotkey exists in metagraph
        if hotkey not in metagraph.hotkeys:
            print(f"🔍 Registry check: {hotkey[:20]}... NOT FOUND in metagraph")
            return False, None
        
        # Get UID for this hotkey
        uid = metagraph.hotkeys.index(hotkey)
        
        # Get neuron attributes
        stake = metagraph.S[uid]
        active = bool(metagraph.active[uid])
        validator_permit = bool(metagraph.validator_permit[uid])
        
        print(f"🔍 Registry check for {hotkey[:20]}...")
        print(f"   UID: {uid}")
        print(f"   Stake: {stake:.6f} τ")
        print(f"   Active: {active}")
        print(f"   Validator Permit: {validator_permit}")
        
        # This handles validators who haven't set active flag but have significant stake
        STAKE_THRESHOLD = 500000  # 500K TAO minimum for stake-based validator classification
        
        # Validators must have:
        # 1. BOTH active=True AND validator_permit=True (normal path), OR
        # 2. Stake > 500K TAO AND validator_permit=True (temporary stake-based override)
        if (active and validator_permit) or (stake > STAKE_THRESHOLD and validator_permit):
            role = "validator"
            if active and validator_permit:
                print(f"   ✅ Role: VALIDATOR (active=True, permit=True)")
            else:
                print(f"   ✅ Role: VALIDATOR (stake={stake:.0f} τ > {STAKE_THRESHOLD}, permit=True)")
                print(f"      ⚠️  TEMPORARY: Stake-based classification (active={active})")
        else:
            role = "miner"
            print(f"   ✅ Role: MINER (active={active}, permit={validator_permit}, stake={stake:.0f})")
        
        return True, role
    
    except Exception as e:
        print(f"❌ Registry check error for {hotkey[:20]}...: {e}")
        return False, None


def is_registered_hotkey(hotkey: str) -> Tuple[bool, Optional[str]]:
    """
    Check if hotkey is registered on the subnet (SYNC WRAPPER).
    
    DEPRECATED: Use is_registered_hotkey_async() from async contexts.
    
    Args:
        hotkey: SS58 address of the actor
    
    Returns:
        (is_registered, role)
    
    Example:
        >>> is_registered, role = is_registered_hotkey("5GNJqR7T...")
        >>> if is_registered:
        >>>     print(f"Hotkey is a {role}")
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
        
        # TEMPORARY: Allow high-stake validators even if active=False
        # This handles validators who haven't set active flag but have significant stake
        STAKE_THRESHOLD = 500000  # 500K TAO minimum for stake-based validator classification
        
        # Validators must have:
        # 1. BOTH active=True AND validator_permit=True (normal path), OR
        # 2. Stake > 500K TAO AND validator_permit=True (temporary stake-based override)
        if (active and validator_permit) or (stake > STAKE_THRESHOLD and validator_permit):
            role = "validator"
            if active and validator_permit:
                print(f"   ✅ Role: VALIDATOR (active=True, permit=True)")
            else:
                print(f"   ✅ Role: VALIDATOR (stake={stake:.0f} τ > {STAKE_THRESHOLD}, permit=True)")
                print(f"      ⚠️  TEMPORARY: Stake-based classification (active={active})")
        else:
            role = "miner"
            print(f"   ✅ Role: MINER (active={active}, permit={validator_permit}, stake={stake:.0f})")
        
        return True, role
    
    except Exception as e:
        print(f"❌ Registry check error for {hotkey[:20]}...: {e}")
        return False, None


async def get_validator_count_async() -> int:
    """
    Get the number of registered validators (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_validator_count() wrapper.
    
    Returns:
        Number of validators (neurons with active=True AND validator_permit=True,
        OR stake > 500K TAO AND validator_permit=True)
    """
    try:
        metagraph = await get_metagraph_async()
        
        STAKE_THRESHOLD = 500000  # 500K TAO minimum
        
        # Count neurons that are validators (normal OR stake-based)
        validator_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if (metagraph.active[i] and metagraph.validator_permit[i]) or 
               (metagraph.S[i] > STAKE_THRESHOLD and metagraph.validator_permit[i])
        )
        
        return validator_count
    
    except Exception as e:
        print(f"❌ Error getting validator count: {e}")
        return 0


def get_validator_count() -> int:
    """
    Get the number of registered validators (SYNC WRAPPER).
    
    DEPRECATED: Use get_validator_count_async() from async contexts.
    
    Returns:
        Number of validators (neurons with active=True AND validator_permit=True,
        OR stake > 500K TAO AND validator_permit=True)
    
    Example:
        >>> count = get_validator_count()
        >>> print(f"Subnet has {count} validators")
    """
    try:
        metagraph = get_metagraph()
        
        STAKE_THRESHOLD = 500000  # 500K TAO minimum
        
        # Count neurons that are validators (normal OR stake-based)
        validator_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if (metagraph.active[i] and metagraph.validator_permit[i]) or 
               (metagraph.S[i] > STAKE_THRESHOLD and metagraph.validator_permit[i])
        )
        
        return validator_count
    
    except Exception as e:
        print(f"❌ Error getting validator count: {e}")
        return 0


async def get_miner_count_async() -> int:
    """
    Get the number of registered miners (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_miner_count() wrapper.
    
    Returns:
        Number of miners (neurons that are NOT validators)
    """
    try:
        metagraph = await get_metagraph_async()
        
        STAKE_THRESHOLD = 500000  # 500K TAO minimum
        
        # Count neurons that are NOT validators (inverse of validator logic)
        miner_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if not ((metagraph.active[i] and metagraph.validator_permit[i]) or 
                   (metagraph.S[i] > STAKE_THRESHOLD and metagraph.validator_permit[i]))
        )
        
        return miner_count
    
    except Exception as e:
        print(f"❌ Error getting miner count: {e}")
        return 0


def get_miner_count() -> int:
    """
    Get the number of registered miners (SYNC WRAPPER).
    
    DEPRECATED: Use get_miner_count_async() from async contexts.
    
    Returns:
        Number of miners (neurons that are NOT validators)
    
    Example:
        >>> count = get_miner_count()
        >>> print(f"Subnet has {count} miners")
    """
    try:
        metagraph = get_metagraph()
        
        STAKE_THRESHOLD = 500000  # 500K TAO minimum
        
        # Count neurons that are NOT validators (inverse of validator logic)
        miner_count = sum(
            1 for i in range(len(metagraph.hotkeys))
            if not ((metagraph.active[i] and metagraph.validator_permit[i]) or 
                   (metagraph.S[i] > STAKE_THRESHOLD and metagraph.validator_permit[i]))
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
    global _metagraph_cache, _cache_epoch, _cache_epoch_timestamp
    with _cache_lock:
        _metagraph_cache = None
        _cache_epoch = None
        _cache_epoch_timestamp = None
        print("🗑️  Metagraph cache cleared")


def warm_metagraph_cache(target_epoch: int) -> bool:
    """
    Proactively fetch and cache the metagraph for a specific epoch.
    
    This is called by the background metagraph warmer task at epoch boundaries
    to ensure the cache is ready before any requests arrive.
    
    Args:
        target_epoch: The epoch to fetch metagraph for
        
    Returns:
        True if fetch succeeded, False otherwise
        
    Example:
        >>> success = warm_metagraph_cache(19163)
        >>> if success:
        ...     print("Cache warmed successfully")
    """
    global _metagraph_cache, _cache_epoch, _cache_epoch_timestamp
    import time
    
    with _cache_lock:
        # Check if cache is already warmed for this epoch
        if _cache_epoch == target_epoch:
            print(f"🔥 Cache already warmed for epoch {target_epoch}")
            return True
        
        # Fetch new metagraph with retry logic
        max_retries = 8  # Aggressive retries for background warming (user wants 8 attempts)
        retry_delay = 10  # Initial 10s delay between attempts
        timeout_per_attempt = 60  # 60 second timeout per attempt (user requirement)
        
        for attempt in range(max_retries):
            try:
                print(f"🔥 Warming attempt {attempt + 1}/{max_retries} for epoch {target_epoch}...")
                print(f"   Network: {BITTENSOR_NETWORK}, NetUID: {BITTENSOR_NETUID}")
                print(f"   Timeout: {timeout_per_attempt}s per attempt")
                
                # Fetch metagraph with 60-second timeout enforcement
                # We use concurrent.futures to enforce a hard timeout on the blocking call
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                
                def _fetch_metagraph():
                    """Helper function to fetch metagraph (runs in timeout-enforced thread)"""
                    subtensor = bt.subtensor(network=BITTENSOR_NETWORK)
                    return subtensor.metagraph(netuid=BITTENSOR_NETUID)
                
                # Execute with 60-second timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_fetch_metagraph)
                    metagraph = future.result(timeout=timeout_per_attempt)  # Hard 60s timeout
                
                # Update cache
                _metagraph_cache = metagraph
                _cache_epoch = target_epoch
                _cache_epoch_timestamp = time.time()
                
                print(f"🔥 ✅ Cache warmed for epoch {target_epoch}: {len(metagraph.hotkeys)} neurons")
                return True
                
            except FuturesTimeoutError:
                # Timeout after 60 seconds
                if attempt < max_retries - 1:
                    print(f"🔥 ⚠️  Warming attempt {attempt + 1}/{max_retries} timed out after {timeout_per_attempt}s")
                    print(f"   Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 30)  # Exponential backoff, cap at 30s
                    continue
                else:
                    print(f"🔥 ❌ All {max_retries} warming attempts timed out for epoch {target_epoch}")
                    print(f"🔥 ⚠️  Workflow will continue using epoch {target_epoch - 1} cache as fallback")
                    return False
            except Exception as e:
                # Other errors (network, rate limiting, etc.)
                if attempt < max_retries - 1:
                    print(f"🔥 ⚠️  Warming attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"   Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 30)  # Exponential backoff, cap at 30s
                    continue
                else:
                    print(f"🔥 ❌ All {max_retries} warming attempts failed for epoch {target_epoch}: {e}")
                    print(f"🔥 ⚠️  Workflow will continue using epoch {target_epoch - 1} cache as fallback")
                    return False
        
        return False


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


async def get_validator_weights_async(validator_hotkey: str) -> tuple[float, float]:
    """
    Get validator's stake and v_trust from metagraph (ASYNC VERSION).
    
    Use this from async contexts. For sync, use get_validator_weights() wrapper.
    
    This is called during COMMIT phase to snapshot validator weights.
    Critical: Must capture at COMMIT time, not REVEAL time, to prevent gaming.
    
    Args:
        validator_hotkey: Validator's SS58 address
    
    Returns:
        (stake, v_trust): Tuple of (TAO stake, validator trust score)
        Returns (0.0, 0.0) if validator not found
    """
    try:
        metagraph = await get_metagraph_async()
        
        # Find validator's UID
        if validator_hotkey not in metagraph.hotkeys:
            print(f"⚠️  Validator {validator_hotkey[:20]}... not found in metagraph")
            return (0.0, 0.0)
        
        uid = metagraph.hotkeys.index(validator_hotkey)
        
        # Get stake (TAO amount)
        stake = float(metagraph.S[uid])
        
        # Get v_trust (validator trust/reputation)
        v_trust = float(metagraph.validator_trust[uid]) if hasattr(metagraph, 'validator_trust') else 0.0
        
        print(f"📊 Validator weights for {validator_hotkey[:20]}...")
        print(f"   Stake: {stake:.6f} τ")
        print(f"   V-Trust: {v_trust:.6f}")
        
        return (stake, v_trust)
    
    except Exception as e:
        print(f"❌ Error fetching validator weights: {e}")
        return (0.0, 0.0)


def get_validator_weights(validator_hotkey: str) -> tuple[float, float]:
    """
    Get validator's stake and v_trust from metagraph (SYNC WRAPPER).
    
    DEPRECATED: Use get_validator_weights_async() from async contexts.
    
    Args:
        validator_hotkey: Validator's SS58 address
    
    Returns:
        (stake, v_trust): Tuple of (TAO stake, validator trust score)
    
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

