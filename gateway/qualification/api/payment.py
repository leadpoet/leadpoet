"""
Qualification System: Payment Verification

Phase 2.2 from tasks10.md

This module handles on-chain TAO payment verification for model submissions.
Miners must pay a fee to submit models for evaluation.

DYNAMIC NETWORK SUPPORT:
- testnet: Uses TESTNET payment wallet
- mainnet: Uses MAINNET (UID 0) payment wallet
Network is determined by BITTENSOR_NETWORK environment variable.

CRITICAL: This is isolated payment verification for the qualification system only.
Do NOT modify any existing payment or chain interaction code.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# =============================================================================
# Import Chain Utilities (Dynamic Network Support)
# =============================================================================
# This module provides:
# - get_payment_wallet() - returns correct wallet for current network
# - fetch_block() - fetches blocks from chain
# - get_block_timestamp() - extracts block timestamp
# - extrinsic_failed() - checks if extrinsic failed
# - coldkey_owns_hotkey() - verifies coldkey ownership
# - get_transfer_details() - parses transfer extrinsics

from gateway.qualification.utils.chain import (
    get_payment_wallet,
    fetch_block as chain_fetch_block,
    get_block_timestamp as chain_get_block_timestamp,
    extrinsic_failed as chain_extrinsic_failed,
    coldkey_owns_hotkey as chain_coldkey_owns_hotkey,
    get_transfer_details,
    get_chain_info,
    BITTENSOR_NETWORK,
)

# =============================================================================
# Configuration
# =============================================================================

# Get the payment wallet for the current network (dynamic)
LEADPOET_WALLET = get_payment_wallet()

# Valid transfer call functions on the Bittensor chain
VALID_TRANSFER_CALLS = [
    "transfer",
    "transfer_keep_alive",
    "transfer_allow_death",
    "transfer_all",
]

# Maximum age for a payment to be valid (24 hours in seconds)
PAYMENT_MAX_AGE_SECONDS = 86400

# Minimum buffer for amount verification (allow 1% under to account for price fluctuation)
AMOUNT_BUFFER_PERCENT = 0.01

# Log configuration on module load
logger.info(f"Payment verification initialized:")
logger.info(f"  Network: {BITTENSOR_NETWORK}")
logger.info(f"  Payment Wallet: {LEADPOET_WALLET[:20]}...")


# =============================================================================
# Main Payment Verification Function
# =============================================================================

async def verify_payment(
    block_hash: str,
    extrinsic_index: int,
    miner_hotkey: str,
    required_usd: float
) -> Tuple[bool, Optional[str]]:
    """
    Verify on-chain TAO payment for model submission.
    
    Performs comprehensive verification:
    1. Check payment not already used (duplicate prevention)
    2. Block exists and contains the extrinsic
    3. Extrinsic is a valid transfer type
    4. Transfer destination is LEADPOET_WALLET (network-specific)
    5. Sender coldkey owns the miner's hotkey
    6. Amount >= required USD equivalent (based on current TAO price)
    7. Extrinsic succeeded (not failed)
    8. Payment is within the last 24 hours
    
    Args:
        block_hash: The block hash containing the payment transaction
        extrinsic_index: The index of the extrinsic within the block
        miner_hotkey: The miner's Bittensor hotkey (ss58 address)
        required_usd: The required payment amount in USD
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - (True, None) if payment is valid
        - (False, "error reason") if payment is invalid
    """
    
    logger.info(
        f"Verifying payment: block={block_hash[:16]}..., "
        f"extrinsic={extrinsic_index}, hotkey={miner_hotkey[:16]}..., "
        f"network={BITTENSOR_NETWORK}"
    )
    
    # -------------------------------------------------------------------------
    # Check 1: Payment not already used (duplicate prevention)
    # -------------------------------------------------------------------------
    is_duplicate = await payment_already_used(block_hash, extrinsic_index)
    if is_duplicate:
        logger.warning(f"Duplicate payment: block={block_hash[:16]}..., extrinsic={extrinsic_index}")
        return False, "Payment has already been used for another submission"
    
    # -------------------------------------------------------------------------
    # Check 2: Fetch block and verify it exists (with retries for block propagation)
    # -------------------------------------------------------------------------
    # CRITICAL: Blocks can take 30-60+ seconds to propagate across the Bittensor network,
    # especially on testnet. Since the miner has ALREADY PAID, we must be very patient
    # to avoid losing their payment.
    #
    # Retry strategy: 2s, 4s, 6s, 8s, 10s, 10s, 10s, 10s (total 60s max wait)
    # This gives the block plenty of time to propagate while providing feedback.
    block = None
    retry_delays = [2, 4, 6, 8, 10, 10, 10, 10]  # 60 seconds total
    total_waited = 0
    
    # Import reset function for connection recovery
    from gateway.qualification.utils.chain import reset_substrate_interface
    
    for attempt, delay in enumerate(retry_delays, 1):
        block = await fetch_block(block_hash)
        if block is not None:
            logger.info(f"✅ Block found after {attempt} attempt(s) ({total_waited}s): {block_hash[:16]}...")
            break
        
        if attempt < len(retry_delays):
            total_waited += delay
            logger.info(f"⏳ Block not yet propagated, waiting {delay}s before retry {attempt+1}/{len(retry_delays)} (total waited: {total_waited}s)...")
            
            # After 4 failed attempts (20s), reset the WebSocket connection
            # in case it's stale and causing false "not found" responses
            if attempt == 4:
                logger.info("🔄 Resetting chain connection to ensure fresh state...")
                try:
                    await reset_substrate_interface()
                except Exception as e:
                    logger.warning(f"Connection reset warning (non-fatal): {e}")
            
            await asyncio.sleep(delay)
    
    if block is None:
        total_waited = sum(retry_delays[:-1])  # Total time we waited
        logger.error(f"❌ Block not found after {len(retry_delays)} retries ({total_waited}s): {block_hash[:16]}...")
        return False, (
            f"Block not found on chain after {total_waited}s of retries. "
            f"This can happen during network congestion. Please wait 2-3 minutes and resubmit with the same block hash. "
            f"Your payment is safe - the block hash {block_hash[:16]}... will remain valid."
        )
    
    # -------------------------------------------------------------------------
    # Check 3: Get extrinsic and verify index is valid
    # -------------------------------------------------------------------------
    extrinsics = block.get("extrinsics", [])
    if extrinsic_index < 0 or extrinsic_index >= len(extrinsics):
        logger.warning(
            f"Invalid extrinsic index: {extrinsic_index} "
            f"(block has {len(extrinsics)} extrinsics)"
        )
        return False, f"Invalid extrinsic index: {extrinsic_index}"
    
    extrinsic = extrinsics[extrinsic_index]
    
    # -------------------------------------------------------------------------
    # Check 4: Verify it's a valid transfer type
    # -------------------------------------------------------------------------
    call_function = get_extrinsic_call_function(extrinsic)
    if call_function not in VALID_TRANSFER_CALLS:
        logger.warning(f"Invalid call function: {call_function}")
        return False, f"Extrinsic is not a transfer (got: {call_function})"
    
    # -------------------------------------------------------------------------
    # Check 5: Verify destination is LEADPOET_WALLET
    # -------------------------------------------------------------------------
    destination = get_extrinsic_destination(extrinsic)
    if destination != LEADPOET_WALLET:
        logger.warning(f"Invalid destination: {destination[:16] if destination else 'None'}... (expected: {LEADPOET_WALLET[:16]}...)")
        return False, "Transfer destination is not the LeadPoet wallet"
    
    # -------------------------------------------------------------------------
    # Check 6: Verify sender (coldkey) owns the miner's hotkey
    # -------------------------------------------------------------------------
    sender_coldkey = get_extrinsic_sender(extrinsic)
    if not sender_coldkey:
        logger.warning("Could not extract sender from extrinsic")
        return False, "Could not determine payment sender"
    
    owns_hotkey = await coldkey_owns_hotkey(sender_coldkey, miner_hotkey)
    if not owns_hotkey:
        logger.warning(
            f"Coldkey {sender_coldkey[:16]}... does not own hotkey {miner_hotkey[:16]}..."
        )
        return False, "Payment sender does not own the miner hotkey"
    
    # -------------------------------------------------------------------------
    # Check 7: Verify amount meets required USD
    # -------------------------------------------------------------------------
    amount_rao = get_extrinsic_amount(extrinsic)
    if amount_rao is None or amount_rao <= 0:
        logger.warning(f"Invalid transfer amount: {amount_rao}")
        return False, "Invalid transfer amount"
    
    tao_price_usd = await get_tao_price_usd()
    amount_tao = amount_rao / 1e9  # Convert rao to TAO (1 TAO = 1e9 rao)
    amount_usd = amount_tao * tao_price_usd
    
    # Allow small buffer for price fluctuation
    required_with_buffer = required_usd * (1 - AMOUNT_BUFFER_PERCENT)
    
    if amount_usd < required_with_buffer:
        logger.warning(
            f"Insufficient payment: ${amount_usd:.4f} < ${required_with_buffer:.4f} "
            f"({amount_tao:.6f} TAO @ ${tao_price_usd:.2f}/TAO)"
        )
        return False, (
            f"Insufficient payment: ${amount_usd:.2f} "
            f"(required: ${required_usd:.2f}, TAO price: ${tao_price_usd:.2f})"
        )
    
    # -------------------------------------------------------------------------
    # Check 8: Verify extrinsic succeeded (not failed)
    # -------------------------------------------------------------------------
    is_failed = await extrinsic_failed(block_hash, extrinsic_index)
    if is_failed:
        logger.warning(f"Extrinsic failed: block={block_hash[:16]}..., index={extrinsic_index}")
        return False, "Transfer extrinsic failed on-chain"
    
    # -------------------------------------------------------------------------
    # Check 9: Verify payment is within 24 hours
    # -------------------------------------------------------------------------
    block_time = await get_block_timestamp(block_hash)
    if block_time is None:
        logger.warning(f"Could not get block timestamp: {block_hash[:16]}...")
        return False, "Could not determine block timestamp"
    
    now = datetime.now(timezone.utc)
    age_seconds = (now - block_time).total_seconds()
    
    if age_seconds > PAYMENT_MAX_AGE_SECONDS:
        hours_old = age_seconds / 3600
        logger.warning(f"Payment too old: {hours_old:.1f} hours (max: 24 hours)")
        return False, f"Payment is too old ({hours_old:.1f} hours, max: 24 hours)"
    
    # Allow up to 5 minutes of clock skew (block slightly in future)
    # This is common due to clock differences between local machine and chain validators
    CLOCK_SKEW_TOLERANCE_SECONDS = 300  # 5 minutes
    
    if age_seconds < -CLOCK_SKEW_TOLERANCE_SECONDS:
        # Block is too far in the future - definitely invalid
        logger.warning(f"Block timestamp is too far in the future: {block_time} ({-age_seconds:.0f}s ahead)")
        return False, f"Block timestamp is in the future by {-age_seconds:.0f} seconds (max allowed: {CLOCK_SKEW_TOLERANCE_SECONDS}s)"
    
    if age_seconds < 0:
        # Minor clock skew - log but allow
        logger.info(f"Minor clock skew detected: block is {-age_seconds:.1f}s in future (within tolerance)")
    
    # -------------------------------------------------------------------------
    # All checks passed
    # -------------------------------------------------------------------------
    logger.info(
        f"✅ Payment verified: {amount_tao:.6f} TAO (${amount_usd:.2f}) "
        f"from {sender_coldkey[:16]}... on {BITTENSOR_NETWORK}"
    )
    
    return True, None


# =============================================================================
# Extrinsic Parsing Helpers
# =============================================================================

def get_extrinsic_call_function(extrinsic: Dict[str, Any]) -> Optional[str]:
    """
    Extract the call function name from an extrinsic.
    
    Different chain APIs may structure this differently.
    """
    # Try common structures
    if isinstance(extrinsic, dict):
        # Structure 1: extrinsic.call.call_function
        if "call" in extrinsic and isinstance(extrinsic["call"], dict):
            call = extrinsic["call"]
            if "call_function" in call:
                return call["call_function"]
            if "function" in call:
                return call["function"]
        
        # Structure 2: extrinsic.method.method
        if "method" in extrinsic:
            method = extrinsic["method"]
            if isinstance(method, dict) and "method" in method:
                return method["method"]
            if isinstance(method, str):
                return method
        
        # Structure 3: Direct call_function
        if "call_function" in extrinsic:
            return extrinsic["call_function"]
    
    logger.warning(f"Could not parse call_function from extrinsic: {type(extrinsic)}")
    return None


def get_extrinsic_destination(extrinsic: Dict[str, Any]) -> Optional[str]:
    """
    Extract the transfer destination address from an extrinsic.
    """
    if isinstance(extrinsic, dict):
        # Structure 1: extrinsic.call.call_args
        if "call" in extrinsic and isinstance(extrinsic["call"], dict):
            call = extrinsic["call"]
            args = call.get("call_args", call.get("args", []))
            if isinstance(args, list):
                for arg in args:
                    if isinstance(arg, dict) and arg.get("name") in ["dest", "destination"]:
                        value = arg.get("value")
                        if isinstance(value, dict):
                            return value.get("Id") or value.get("id")
                        return value
            if isinstance(args, dict):
                dest = args.get("dest") or args.get("destination")
                if isinstance(dest, dict):
                    return dest.get("Id") or dest.get("id")
                return dest
        
        # Structure 2: Direct dest field
        if "dest" in extrinsic:
            dest = extrinsic["dest"]
            if isinstance(dest, dict):
                return dest.get("Id") or dest.get("id")
            return dest
    
    logger.warning(f"Could not parse destination from extrinsic")
    return None


def get_extrinsic_sender(extrinsic: Dict[str, Any]) -> Optional[str]:
    """
    Extract the sender address (coldkey) from an extrinsic.
    """
    if isinstance(extrinsic, dict):
        # Structure 1: extrinsic.address
        if "address" in extrinsic:
            addr = extrinsic["address"]
            if isinstance(addr, dict):
                return addr.get("Id") or addr.get("id")
            return addr
        
        # Structure 2: extrinsic.signature.address
        if "signature" in extrinsic and isinstance(extrinsic["signature"], dict):
            sig = extrinsic["signature"]
            if "address" in sig:
                addr = sig["address"]
                if isinstance(addr, dict):
                    return addr.get("Id") or addr.get("id")
                return addr
        
        # Structure 3: extrinsic.account_id
        if "account_id" in extrinsic:
            return extrinsic["account_id"]
    
    logger.warning(f"Could not parse sender from extrinsic")
    return None


def get_extrinsic_amount(extrinsic: Dict[str, Any]) -> Optional[int]:
    """
    Extract the transfer amount (in rao) from an extrinsic.
    """
    if isinstance(extrinsic, dict):
        # Structure 1: extrinsic.call.call_args
        if "call" in extrinsic and isinstance(extrinsic["call"], dict):
            call = extrinsic["call"]
            args = call.get("call_args", call.get("args", []))
            if isinstance(args, list):
                for arg in args:
                    if isinstance(arg, dict) and arg.get("name") in ["value", "amount"]:
                        return int(arg.get("value", 0))
            if isinstance(args, dict):
                value = args.get("value") or args.get("amount")
                if value is not None:
                    return int(value)
        
        # Structure 2: Direct value field
        if "value" in extrinsic:
            return int(extrinsic["value"])
        
        if "amount" in extrinsic:
            return int(extrinsic["amount"])
    
    logger.warning(f"Could not parse amount from extrinsic")
    return None


# =============================================================================
# Chain Interaction Functions (Using qualification.utils.chain)
# =============================================================================

async def fetch_block(block_hash: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a block from the Bittensor chain.
    
    Uses the dynamic network endpoint from BITTENSOR_NETWORK.
    
    Args:
        block_hash: The block hash to fetch
    
    Returns:
        Block data as dict, or None if not found
    """
    return await chain_fetch_block(block_hash)


async def get_block_timestamp(block_hash: str) -> Optional[datetime]:
    """
    Get the timestamp of a block.
    
    Args:
        block_hash: The block hash
    
    Returns:
        Block timestamp as datetime, or None if not found
    """
    return await chain_get_block_timestamp(block_hash)


async def extrinsic_failed(block_hash: str, extrinsic_index: int) -> bool:
    """
    Check if an extrinsic failed (was not successful).
    
    Args:
        block_hash: The block hash
        extrinsic_index: The extrinsic index
    
    Returns:
        True if extrinsic failed, False if it succeeded
    """
    return await chain_extrinsic_failed(block_hash, extrinsic_index)


async def coldkey_owns_hotkey(coldkey: str, hotkey: str) -> bool:
    """
    Verify that a coldkey owns a hotkey on the Bittensor network.
    
    Args:
        coldkey: The coldkey (ss58 address) that allegedly owns the hotkey
        hotkey: The hotkey (ss58 address) to verify ownership of
    
    Returns:
        True if coldkey owns hotkey, False otherwise
    """
    return await chain_coldkey_owns_hotkey(coldkey, hotkey)


# =============================================================================
# Database Interaction Functions
# =============================================================================

async def payment_already_used(block_hash: str, extrinsic_index: int) -> bool:
    """
    Check if a payment has already been used for a model submission.
    
    Queries the qualification_payments table in Supabase.
    
    Args:
        block_hash: The block hash containing the payment
        extrinsic_index: The extrinsic index within the block
    
    Returns:
        True if payment already used, False otherwise
    """
    try:
        from supabase import create_client
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured - skipping duplicate check")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Query qualification_payments table
        response = supabase.table("qualification_payments") \
            .select("model_id") \
            .eq("block_hash", block_hash) \
            .eq("extrinsic_index", extrinsic_index) \
            .execute()
        
        if response.data and len(response.data) > 0:
            logger.warning(f"Payment already used: block={block_hash[:16]}..., extrinsic={extrinsic_index}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking payment_already_used: {e}")
        # On error, return False to allow submission (will fail later if truly duplicate)
        return False


# =============================================================================
# TAO Price Functions
# =============================================================================

async def get_tao_price_usd() -> float:
    """
    Get the current TAO price in USD from CoinGecko.
    
    Returns:
        Current TAO price in USD
    
    Caches the price for 5 minutes to avoid rate limiting.
    """
    import httpx
    from datetime import datetime, timezone
    
    # Cache to avoid hitting API too often
    if not hasattr(get_tao_price_usd, '_cache'):
        get_tao_price_usd._cache = {'price': None, 'timestamp': None}
    
    cache = get_tao_price_usd._cache
    now = datetime.now(timezone.utc)
    
    # Return cached price if less than 5 minutes old
    if cache['price'] and cache['timestamp']:
        age_seconds = (now - cache['timestamp']).total_seconds()
        if age_seconds < 300:  # 5 minutes
            return cache['price']
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bittensor", "vs_currencies": "usd"}
            )
            response.raise_for_status()
            data = response.json()
            
            price = data.get("bittensor", {}).get("usd")
            if price:
                cache['price'] = float(price)
                cache['timestamp'] = now
                logger.info(f"TAO price fetched: ${price:.2f}")
                return float(price)
            else:
                logger.warning("CoinGecko returned no price data")
                
    except Exception as e:
        logger.warning(f"Failed to fetch TAO price from CoinGecko: {e}")
    
    # Fallback to cached price if available
    if cache['price']:
        logger.info(f"Using cached TAO price: ${cache['price']:.2f}")
        return cache['price']
    
    # Last resort fallback
    logger.warning("Using fallback TAO price of $500")
    return 500.0


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_required_tao(required_usd: float, tao_price_usd: float) -> float:
    """
    Calculate required TAO amount for a given USD amount.
    
    Args:
        required_usd: Required amount in USD
        tao_price_usd: Current TAO price in USD
    
    Returns:
        Required TAO amount
    """
    if tao_price_usd <= 0:
        raise ValueError("TAO price must be positive")
    return required_usd / tao_price_usd


def rao_to_tao(rao: int) -> float:
    """Convert rao to TAO (1 TAO = 1e9 rao)."""
    return rao / 1e9


def tao_to_rao(tao: float) -> int:
    """Convert TAO to rao (1 TAO = 1e9 rao)."""
    return int(tao * 1e9)


async def get_payment_info(block_hash: str, extrinsic_index: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed payment information for logging/debugging.
    
    Used to extract actual payment amounts from the chain for database recording.
    
    Args:
        block_hash: The block hash
        extrinsic_index: The extrinsic index
    
    Returns:
        Dict with payment details:
        - block_hash, extrinsic_index
        - call_function
        - sender_coldkey
        - destination
        - amount_rao, amount_tao, amount_usd
        - tao_price_at_payment
        Or None if not found
    """
    block = await fetch_block(block_hash)
    if not block:
        return None
    
    extrinsics = block.get("extrinsics", [])
    if extrinsic_index < 0 or extrinsic_index >= len(extrinsics):
        return None
    
    extrinsic = extrinsics[extrinsic_index]
    
    amount_rao = get_extrinsic_amount(extrinsic)
    tao_price = await get_tao_price_usd()
    amount_tao = rao_to_tao(amount_rao) if amount_rao else 0.0
    amount_usd = amount_tao * tao_price if amount_tao else 0.0
    
    return {
        "block_hash": block_hash,
        "extrinsic_index": extrinsic_index,
        "call_function": get_extrinsic_call_function(extrinsic),
        "sender_coldkey": get_extrinsic_sender(extrinsic),
        "destination": get_extrinsic_destination(extrinsic),
        "amount_rao": amount_rao,
        "amount_tao": amount_tao,
        "amount_usd": amount_usd,
        "tao_price_at_payment": tao_price
    }
