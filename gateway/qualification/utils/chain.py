"""
Qualification System: Bittensor Chain Interaction
=================================================

Provides chain interaction functions with DYNAMIC network support.
Uses BITTENSOR_NETWORK environment variable to determine:
- testnet: wss://test.finney.opentensor.ai:443
- finney (mainnet): wss://entrypoint-finney.opentensor.ai:443

This module handles:
- Block fetching and parsing
- Extrinsic verification
- Metagraph queries (hotkey registration, coldkey ownership)
- Payment wallet configuration per network
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# =============================================================================
# Network Configuration (Dynamic based on BITTENSOR_NETWORK)
# =============================================================================

# Default network is mainnet (finney), override with BITTENSOR_NETWORK env var
BITTENSOR_NETWORK = os.getenv("BITTENSOR_NETWORK", "finney")
BITTENSOR_NETUID = int(os.getenv("BITTENSOR_NETUID", "71"))

# Network endpoints
NETWORK_ENDPOINTS = {
    "test": "wss://test.finney.opentensor.ai:443",
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
    "local": "ws://127.0.0.1:9944",
}

# =============================================================================
# Payment Wallet Configuration (Network-Specific)
# =============================================================================
# CRITICAL: Different wallets for testnet vs mainnet
# Testnet uses a test wallet, mainnet uses the UID 0 production wallet

# UID 0 (LeadPoet) Payment Wallet for MAINNET - COLDKEY address
# This is the coldkey that receives qualification submission payments on mainnet.
MAINNET_PAYMENT_WALLET = "5ExoWGyajvzucCqS5GxZSpuzzXEzG1oNFcDqdW3sXeTujoD7"

# Testnet Payment Wallet - Validator coldkey for testnet (netuid 401)
# This matches the miner's LEADPOET_COLDKEY_TESTNET in neurons/miner.py
TESTNET_PAYMENT_WALLET = os.getenv(
    "TESTNET_PAYMENT_WALLET",
    "5Gh5kw7rV1x7FDDd5E3Uc7YYMoeQtm4gn93c7VYeL5oUyoAD"  # Validator coldkey for testnet
)

def get_payment_wallet() -> str:
    """
    Get the payment wallet address for the current network.
    
    Returns:
        SS58 address of the payment wallet
    """
    if BITTENSOR_NETWORK == "test":
        wallet = TESTNET_PAYMENT_WALLET
        logger.info(f"Using TESTNET payment wallet: {wallet[:16]}...")
    else:
        wallet = MAINNET_PAYMENT_WALLET
        logger.info(f"Using MAINNET payment wallet: {wallet[:16]}...")
    return wallet


def get_network_endpoint() -> str:
    """
    Get the WebSocket endpoint for the current network.
    
    Returns:
        WebSocket URL for the Bittensor network
    """
    endpoint = NETWORK_ENDPOINTS.get(BITTENSOR_NETWORK, NETWORK_ENDPOINTS["finney"])
    return endpoint


# =============================================================================
# Substrate Interface (Lazy Initialization)
# =============================================================================

_substrate_interface = None
_substrate_interface_endpoint = None  # Track which endpoint we're connected to
_substrate_lock = asyncio.Lock()


def _get_current_network() -> str:
    """Get current network from environment (re-reads each time)."""
    return os.getenv("BITTENSOR_NETWORK", "finney")


async def get_substrate_interface():
    """
    Get or create the SubstrateInterface for chain queries.
    
    Uses lazy initialization and caching for efficiency.
    Thread-safe with asyncio lock.
    IMPORTANT: Re-checks network endpoint each time and reconnects if changed.
    
    Returns:
        SubstrateInterface instance
    """
    global _substrate_interface, _substrate_interface_endpoint
    
    # Get the current expected endpoint (re-read from env)
    current_network = _get_current_network()
    expected_endpoint = NETWORK_ENDPOINTS.get(current_network, NETWORK_ENDPOINTS["finney"])
    
    async with _substrate_lock:
        # Check if we need to reconnect (endpoint changed or not connected)
        if _substrate_interface is None or _substrate_interface_endpoint != expected_endpoint:
            # Close existing connection if any
            if _substrate_interface is not None:
                try:
                    logger.info(f"Network changed: {_substrate_interface_endpoint} -> {expected_endpoint}")
                    _substrate_interface.close()
                except:
                    pass
                _substrate_interface = None
            
            try:
                from substrateinterface import SubstrateInterface
                
                logger.info(f"Connecting to Bittensor chain at {expected_endpoint} ({current_network})...")
                
                # Run in thread pool to avoid blocking
                _substrate_interface = await asyncio.to_thread(
                    SubstrateInterface,
                    url=expected_endpoint,
                    ss58_format=42,  # Bittensor SS58 format
                    type_registry_preset="substrate-node-template"
                )
                
                _substrate_interface_endpoint = expected_endpoint
                logger.info(f"✅ Connected to Bittensor chain ({current_network})")
                
            except Exception as e:
                logger.error(f"Failed to connect to Bittensor chain: {e}")
                raise
        
        return _substrate_interface


async def reset_substrate_interface():
    """Reset the substrate interface (for reconnection after errors)."""
    global _substrate_interface
    async with _substrate_lock:
        if _substrate_interface:
            try:
                _substrate_interface.close()
            except:
                pass
        _substrate_interface = None


# =============================================================================
# Block Fetching
# =============================================================================

async def fetch_block(block_hash: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a block from the Bittensor chain.
    
    Args:
        block_hash: The block hash to fetch (0x-prefixed hex string)
    
    Returns:
        Block data as dict with 'extrinsics' and 'header', or None if not found
    """
    try:
        substrate = await get_substrate_interface()
        
        # Fetch block in thread pool
        block = await asyncio.to_thread(
            substrate.get_block,
            block_hash=block_hash
        )
        
        if not block:
            logger.warning(f"Block not found: {block_hash[:16]}...")
            return None
        
        # Parse extrinsics into a more usable format
        parsed_extrinsics = []
        
        # substrate-interface returns extrinsics at top level, not under 'block'
        # Check both structures for compatibility
        if "block" in block and block.get("block"):
            # Nested structure: block['block']['extrinsics']
            raw_extrinsics = block.get("block", {}).get("extrinsics", [])
            header = block.get("block", {}).get("header", {})
        else:
            # Flat structure: block['extrinsics'] directly
            raw_extrinsics = block.get("extrinsics", [])
            header = block.get("header", {})
        
        logger.debug(f"Block {block_hash[:16]}... has {len(raw_extrinsics)} extrinsics")
        
        for idx, ext in enumerate(raw_extrinsics):
            parsed = _parse_extrinsic(ext, idx)
            parsed_extrinsics.append(parsed)
        
        return {
            "header": header,
            "extrinsics": parsed_extrinsics,
            "raw_block": block
        }
        
    except Exception as e:
        logger.error(f"Error fetching block {block_hash[:16]}...: {e}")
        # Reset interface on connection errors
        if "connection" in str(e).lower() or "websocket" in str(e).lower():
            await reset_substrate_interface()
        return None


def _parse_extrinsic(ext, index: int) -> Dict[str, Any]:
    """
    Parse a raw extrinsic into a structured format.
    
    Args:
        ext: Raw extrinsic from substrate
        index: Extrinsic index in block
    
    Returns:
        Parsed extrinsic dict with call_module, call_function, call_args, address
    """
    try:
        # Handle both dict and object formats
        if hasattr(ext, 'value'):
            ext_data = ext.value
        else:
            ext_data = ext
        
        # Extract call info
        call = ext_data.get("call", {})
        if hasattr(call, 'value'):
            call = call.value
        
        call_module = call.get("call_module", "Unknown")
        call_function = call.get("call_function", "Unknown")
        call_args = call.get("call_args", [])
        
        # Extract sender address
        address = ext_data.get("address")
        if hasattr(address, 'value'):
            address = address.value
        if isinstance(address, dict):
            address = address.get("Id") or address.get("id")
        
        return {
            "index": index,
            "call": {
                "call_module": call_module,
                "call_function": call_function,
                "call_args": call_args
            },
            "address": address,
            "raw": ext_data
        }
        
    except Exception as e:
        logger.warning(f"Error parsing extrinsic {index}: {e}")
        return {
            "index": index,
            "call": {"call_module": "Unknown", "call_function": "Unknown", "call_args": []},
            "address": None,
            "raw": ext
        }


# =============================================================================
# Block Timestamp Extraction
# =============================================================================

async def get_block_timestamp(block_hash: str) -> Optional[datetime]:
    """
    Get the timestamp of a block.
    
    Extracts timestamp from the Timestamp.set inherent extrinsic.
    
    Args:
        block_hash: The block hash
    
    Returns:
        Block timestamp as datetime (UTC), or None if not found
    """
    try:
        block = await fetch_block(block_hash)
        if not block:
            return None
        
        # Look for Timestamp.set inherent in extrinsics
        for ext in block.get("extrinsics", []):
            call = ext.get("call", {})
            if call.get("call_module") == "Timestamp" and call.get("call_function") == "set":
                call_args = call.get("call_args", [])
                for arg in call_args:
                    if isinstance(arg, dict) and arg.get("name") == "now":
                        timestamp_ms = int(arg.get("value", 0))
                        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    elif isinstance(arg, int):
                        # Direct value format
                        return datetime.fromtimestamp(arg / 1000, tz=timezone.utc)
        
        logger.warning(f"No timestamp found in block {block_hash[:16]}...")
        return None
        
    except Exception as e:
        logger.error(f"Error getting block timestamp: {e}")
        return None


# =============================================================================
# Extrinsic Verification
# =============================================================================

async def extrinsic_failed(block_hash: str, extrinsic_index: int) -> bool:
    """
    Check if an extrinsic failed (was not successful).
    
    Queries block events to find ExtrinsicSuccess or ExtrinsicFailed.
    
    Args:
        block_hash: The block hash
        extrinsic_index: The extrinsic index
    
    Returns:
        True if extrinsic failed, False if it succeeded
    """
    try:
        substrate = await get_substrate_interface()
        
        # Get events for this block
        events = await asyncio.to_thread(
            substrate.get_events,
            block_hash=block_hash
        )
        
        if not events:
            logger.warning(f"No events found for block {block_hash[:16]}...")
            return True  # Assume failed if no events
        
        # Look for success/failure event for this extrinsic
        for event in events:
            try:
                event_data = event.value if hasattr(event, 'value') else event
                ext_idx = event_data.get("extrinsic_idx")
                
                if ext_idx == extrinsic_index:
                    event_info = event_data.get("event", {})
                    event_id = event_info.get("event_id", "")
                    
                    if event_id == "ExtrinsicSuccess":
                        logger.debug(f"Extrinsic {extrinsic_index} succeeded")
                        return False
                    elif event_id == "ExtrinsicFailed":
                        logger.warning(f"Extrinsic {extrinsic_index} failed")
                        return True
                        
            except Exception as e:
                logger.debug(f"Error parsing event: {e}")
                continue
        
        # If no explicit success/fail event found, assume success
        # (some older blocks may not have explicit events)
        logger.debug(f"No explicit success/fail event for extrinsic {extrinsic_index}, assuming success")
        return False
        
    except Exception as e:
        logger.error(f"Error checking extrinsic status: {e}")
        return True  # Assume failed on error


# =============================================================================
# Transfer Extrinsic Parsing
# =============================================================================

def get_transfer_details(extrinsic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract transfer details from an extrinsic.
    
    Args:
        extrinsic: Parsed extrinsic dict from fetch_block
    
    Returns:
        Dict with sender, destination, amount_rao, or None if not a transfer
    """
    call = extrinsic.get("call", {})
    call_module = call.get("call_module", "")
    call_function = call.get("call_function", "")
    
    # Check if this is a transfer
    valid_transfer_functions = [
        "transfer",
        "transfer_keep_alive",
        "transfer_allow_death",
        "transfer_all"
    ]
    
    if call_module != "Balances" or call_function not in valid_transfer_functions:
        return None
    
    call_args = call.get("call_args", [])
    
    # Extract destination and amount
    destination = None
    amount_rao = None
    
    for arg in call_args:
        if isinstance(arg, dict):
            arg_name = arg.get("name", "")
            arg_value = arg.get("value")
            
            if arg_name in ["dest", "destination"]:
                if isinstance(arg_value, dict):
                    destination = arg_value.get("Id") or arg_value.get("id")
                else:
                    destination = arg_value
                    
            elif arg_name in ["value", "amount"]:
                amount_rao = int(arg_value) if arg_value else None
    
    sender = extrinsic.get("address")
    
    return {
        "sender": sender,
        "destination": destination,
        "amount_rao": amount_rao,
        "call_function": call_function
    }


# =============================================================================
# Metagraph Queries
# =============================================================================

async def get_metagraph():
    """
    Get the current metagraph for the subnet.
    
    USES THE GATEWAY'S CACHED METAGRAPH to avoid timeout issues.
    Falls back to direct fetch if gateway cache unavailable.
    
    Returns:
        Metagraph object
    """
    # Try to use gateway's cached metagraph (already connected and cached)
    try:
        from gateway.utils.registry import get_metagraph_async
        metagraph = await get_metagraph_async()
        logger.debug(f"Using gateway cached metagraph: {len(metagraph.hotkeys)} neurons")
        return metagraph
    except ImportError:
        logger.debug("Gateway metagraph not available, using direct fetch")
    except Exception as e:
        logger.warning(f"Gateway metagraph error: {e}, trying direct fetch")
    
    # Fallback to direct fetch if gateway cache not available
    try:
        import bittensor as bt
        
        current_network = _get_current_network()
        netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
        
        # Use sync Subtensor in thread pool with timeout
        def _fetch_metagraph():
            subtensor = bt.Subtensor(network=current_network)
            return subtensor.metagraph(netuid=netuid)
        
        metagraph = await asyncio.wait_for(
            asyncio.to_thread(_fetch_metagraph),
            timeout=30.0  # 30 second timeout
        )
        
        logger.info(f"Metagraph fetched directly: {len(metagraph.hotkeys)} neurons on {current_network}")
        return metagraph
        
    except asyncio.TimeoutError:
        logger.error("Metagraph fetch timed out after 30 seconds")
        raise
    except Exception as e:
        logger.error(f"Error fetching metagraph: {e}")
        raise


async def is_hotkey_registered(hotkey: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a hotkey is registered on the subnet.
    
    Args:
        hotkey: The Bittensor hotkey (SS58 address)
    
    Returns:
        Tuple of (is_registered, role) where role is "miner" or "validator" or None
    """
    try:
        metagraph = await get_metagraph()
        
        # Check if hotkey exists in metagraph
        if hotkey not in metagraph.hotkeys:
            return False, None
        
        # Get UID for this hotkey
        uid = metagraph.hotkeys.index(hotkey)
        
        # Determine role based on validator_permit
        validator_permit = bool(metagraph.validator_permit[uid])
        
        if validator_permit:
            return True, "validator"
        else:
            return True, "miner"
            
    except Exception as e:
        logger.error(f"Error checking hotkey registration: {e}")
        return False, None


async def coldkey_owns_hotkey(coldkey: str, hotkey: str) -> bool:
    """
    Verify that a coldkey owns a hotkey on the Bittensor network.
    
    Args:
        coldkey: The coldkey (SS58 address) that allegedly owns the hotkey
        hotkey: The hotkey (SS58 address) to verify ownership of
    
    Returns:
        True if coldkey owns hotkey, False otherwise
    """
    try:
        metagraph = await get_metagraph()
        
        # Check if hotkey exists
        if hotkey not in metagraph.hotkeys:
            logger.warning(f"Hotkey {hotkey[:16]}... not found in metagraph")
            return False
        
        # Get UID and check coldkey
        uid = metagraph.hotkeys.index(hotkey)
        actual_coldkey = metagraph.coldkeys[uid]
        
        if actual_coldkey == coldkey:
            return True
        else:
            logger.warning(
                f"Coldkey mismatch: expected {coldkey[:16]}..., "
                f"actual {actual_coldkey[:16]}..."
            )
            return False
            
    except Exception as e:
        logger.error(f"Error verifying coldkey ownership: {e}")
        return False


async def get_coldkey_for_hotkey(hotkey: str) -> Optional[str]:
    """
    Get the coldkey that owns a hotkey.
    
    Args:
        hotkey: The hotkey (SS58 address)
    
    Returns:
        Coldkey SS58 address, or None if not found
    """
    try:
        metagraph = await get_metagraph()
        
        if hotkey not in metagraph.hotkeys:
            return None
        
        uid = metagraph.hotkeys.index(hotkey)
        return metagraph.coldkeys[uid]
        
    except Exception as e:
        logger.error(f"Error getting coldkey for hotkey: {e}")
        return None


# =============================================================================
# Epoch Calculation
# =============================================================================

TEMPO = 360  # Blocks per epoch (12 seconds per block = 72 minutes per epoch)


async def get_current_block() -> int:
    """
    Get the current block number.
    
    Returns:
        Current block number
    """
    try:
        substrate = await get_substrate_interface()
        
        # Get latest block hash
        block_hash = await asyncio.to_thread(substrate.get_chain_head)
        
        # Get block header
        header = await asyncio.to_thread(
            substrate.get_block_header,
            block_hash=block_hash
        )
        
        return int(header['header']['number'])
        
    except Exception as e:
        logger.error(f"Error getting current block: {e}")
        raise


async def get_current_bittensor_epoch() -> int:
    """
    Get the current Bittensor epoch.
    
    Epoch = block_number // TEMPO
    
    Returns:
        Current epoch number
    """
    block = await get_current_block()
    return block // TEMPO


# =============================================================================
# Signature Verification
# =============================================================================

def verify_sr25519_signature(
    hotkey: str,
    signature: str,
    message: str
) -> bool:
    """
    Verify an SR25519 signature from a Bittensor hotkey.
    
    USES THE SAME APPROACH AS GATEWAY'S verify_wallet_signature.
    
    Args:
        hotkey: The Bittensor hotkey (SS58 address)
        signature: Hex-encoded signature (with or without 0x prefix)
        message: The message STRING that was signed
    
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Use bittensor.Keypair (same as gateway)
        from bittensor import Keypair
        
        # Create keypair from SS58 address (public key only)
        keypair = Keypair(ss58_address=hotkey)
        
        # Clean up signature (remove 0x prefix if present)
        if signature.startswith("0x"):
            signature = signature[2:]
        
        # Verify signature
        signature_bytes = bytes.fromhex(signature)
        is_valid = keypair.verify(message, signature_bytes)
        
        return is_valid
        
    except ValueError as e:
        logger.error(f"Signature format error: {e}")
        return False
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False


def verify_hotkey_signature(
    hotkey: str,
    signature: str,
    message_data: Dict[str, Any]
) -> bool:
    """
    Verify that the signature was created by the hotkey.
    
    The message_data is JSON serialized (sorted keys) to create the message string.
    
    IMPORTANT: Miner signs with json.dumps(data, sort_keys=True) WITHOUT separators.
    We must match that exact format.
    
    Args:
        hotkey: The Bittensor hotkey (SS58 address)
        signature: Hex-encoded signature
        message_data: The data that was signed
    
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Serialize message data to JSON string (sorted keys for determinism)
        # MUST match miner format: json.dumps(data, sort_keys=True) WITHOUT separators
        message = json.dumps(message_data, sort_keys=True)
        
        logger.debug(f"Verifying signature for hotkey={hotkey[:16]}...")
        logger.debug(f"Message to verify: {message[:100]}...")
        
        return verify_sr25519_signature(hotkey, signature, message)
        
    except Exception as e:
        logger.error(f"Hotkey signature verification error: {e}")
        return False


# =============================================================================
# Module Info
# =============================================================================

def get_chain_info() -> Dict[str, Any]:
    """
    Get information about the current chain configuration.
    
    Returns:
        Dict with network, endpoint, netuid, payment_wallet
    """
    return {
        "network": BITTENSOR_NETWORK,
        "endpoint": get_network_endpoint(),
        "netuid": BITTENSOR_NETUID,
        "payment_wallet": get_payment_wallet(),
        "is_testnet": BITTENSOR_NETWORK == "test",
        "tempo": TEMPO
    }


# Log chain info on module load
if __name__ != "__main__":
    logger.info(f"Chain module initialized: network={BITTENSOR_NETWORK}, netuid={BITTENSOR_NETUID}")
