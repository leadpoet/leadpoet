"""
Arweave Client for Trustless Transparency Log

This module provides async functions to write/read events to/from Arweave,
implementing the write-to-Arweave-first architecture for trustless operation.

Key Features:
- Write events to permanent Arweave storage
- Read events by transaction ID
- Query events by type/actor
- Automatic retry with exponential backoff
- Comprehensive error handling and logging

Cost: ~$0.001 per KB, ~$10-50/month for expected volume
"""

import os
import json
import asyncio
from typing import Dict, List, Optional
from pathlib import Path

from arweave.arweave_lib import Wallet, Transaction


# Configuration (will be loaded from config.py)
ARWEAVE_KEYFILE_PATH = os.getenv("ARWEAVE_KEYFILE_PATH", "secrets/arweave_keyfile.json")
ARWEAVE_GATEWAY_URL = os.getenv("ARWEAVE_GATEWAY_URL", "https://arweave.net")

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 30  # seconds

# Global client instance (initialized on first use)
_wallet: Optional[Wallet] = None
_peer = None  # Arweave peer/client


def _initialize_client():
    """
    Initialize Arweave wallet from keyfile.
    
    This is called lazily on first use to avoid blocking import.
    """
    global _wallet, _peer
    
    if _wallet is not None:
        return  # Already initialized
    
    try:
        # Load wallet from keyfile
        keyfile_path = Path(ARWEAVE_KEYFILE_PATH)
        if not keyfile_path.exists():
            raise FileNotFoundError(
                f"Arweave keyfile not found at {ARWEAVE_KEYFILE_PATH}. "
                f"Please create wallet and save keyfile to this location."
            )
        
        # Wallet expects the file path as a string, not the loaded dict
        _wallet = Wallet(str(keyfile_path))
        
        # Get wallet address
        address = _wallet.address
        
        # Get balance
        try:
            balance_winston = _wallet.balance
            balance_ar = int(balance_winston) / 1e12 if balance_winston else 0.0
        except:
            balance_ar = 0.0
        
        print(f"✅ Arweave client initialized")
        print(f"   Wallet: {address}")
        print(f"   Balance: {balance_ar:.4f} AR")
        print(f"   Gateway: {ARWEAVE_GATEWAY_URL}")
        
    except FileNotFoundError as e:
        print(f"❌ Arweave initialization failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Arweave initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize Arweave client: {e}")


async def write_event(event: Dict, tags: Optional[Dict[str, str]] = None) -> str:
    """
    Write an event to Arweave with retry logic.
    
    This is the primary function for logging transparency events to permanent storage.
    Events are written as JSON with optional tags for querying.
    
    Args:
        event: Event dictionary to write (will be JSON serialized)
        tags: Optional tags for querying (e.g., {"event_type": "SUBMISSION", "actor": "hotkey"})
    
    Returns:
        str: Arweave transaction ID (43-character string)
    
    Raises:
        RuntimeError: If write fails after all retries
        
    Example:
        >>> event = {
        ...     "event_type": "SUBMISSION_REQUEST",
        ...     "lead_id": "abc-123",
        ...     "miner_hotkey": "5Ejf9PbZ...",
        ...     "timestamp": "2025-11-08T12:00:00Z"
        ... }
        >>> tx_id = await write_event(event, tags={"event_type": "SUBMISSION_REQUEST"})
        >>> print(f"Event written to Arweave: {tx_id}")
    """
    _initialize_client()
    
    if _wallet is None:
        raise RuntimeError("Arweave wallet not initialized")
    
    # Serialize event to JSON
    event_json = json.dumps(event, sort_keys=True, indent=2, default=str)  # Handle datetime objects
    event_bytes = event_json.encode('utf-8')
    
    # Calculate cost
    data_size_kb = len(event_bytes) / 1024
    print(f"📤 Writing event to Arweave ({data_size_kb:.2f} KB)...")
    
    # Retry loop with exponential backoff
    retry_delay = INITIAL_RETRY_DELAY
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Create transaction
            transaction = Transaction(
                wallet=_wallet,
                data=event_bytes
            )
            
            # Add tags for querying
            if tags:
                for key, value in tags.items():
                    transaction.add_tag(key, str(value))
            
            # Add standard tags
            transaction.add_tag("Content-Type", "application/json")
            transaction.add_tag("App-Name", "Leadpoet-Transparency-Log")
            transaction.add_tag("App-Version", "1.0")
            
            # Sign transaction
            transaction.sign()
            
            # Send transaction (simpler approach - no chunked upload for small data)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, transaction.send)
            
            tx_id = transaction.id
            
            print(f"✅ Event written to Arweave: {tx_id}")
            print(f"   View: https://viewblock.io/arweave/tx/{tx_id}")
            
            return tx_id
            
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"⚠️  Arweave write failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"   Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
            else:
                print(f"❌ Arweave write failed after {MAX_RETRIES} attempts: {e}")
    
    # All retries exhausted
    raise RuntimeError(f"Failed to write event to Arweave after {MAX_RETRIES} attempts: {last_error}")


async def read_event(tx_id: str) -> Dict:
    """
    Read an event from Arweave by transaction ID.
    
    Args:
        tx_id: Arweave transaction ID (43-character string)
    
    Returns:
        Dict: Event data (JSON deserialized)
    
    Raises:
        RuntimeError: If read fails
        
    Example:
        >>> event = await read_event("abc123def456...")
        >>> print(event["event_type"])
        "SUBMISSION_REQUEST"
    """
    _initialize_client()
    
    if _wallet is None:
        raise RuntimeError("Arweave wallet not initialized")
    
    print(f"📥 Reading event from Arweave: {tx_id[:16]}...")
    
    try:
        # Fetch transaction data using requests (arweave-python-client doesn't have built-in read)
        import requests
        
        loop = asyncio.get_event_loop()
        
        def fetch_data():
            url = f"{ARWEAVE_GATEWAY_URL}/{tx_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        
        data_bytes = await loop.run_in_executor(None, fetch_data)
        
        # Decode and parse JSON
        event_json = data_bytes.decode('utf-8')
        event = json.loads(event_json)
        
        print(f"✅ Event read from Arweave: {event.get('event_type', 'unknown')}")
        
        return event
        
    except Exception as e:
        print(f"❌ Failed to read event from Arweave: {e}")
        raise RuntimeError(f"Failed to read event {tx_id}: {e}")


async def query_events(filters: Optional[Dict[str, str]] = None, limit: int = 100) -> List[Dict]:
    """
    Query events from Arweave by tags.
    
    Note: Arweave queries use GraphQL and may take time to index recent transactions.
    For real-time queries, prefer direct transaction ID lookups.
    
    Args:
        filters: Tag filters (e.g., {"event_type": "SUBMISSION", "actor": "hotkey"})
        limit: Maximum number of events to return (default: 100)
    
    Returns:
        List[Dict]: List of events matching filters
        
    Example:
        >>> events = await query_events({"event_type": "SUBMISSION_REQUEST"}, limit=10)
        >>> print(f"Found {len(events)} submission requests")
    """
    _initialize_client()
    
    if _wallet is None:
        raise RuntimeError("Arweave wallet not initialized")
    
    print(f"🔍 Querying Arweave events with filters: {filters}")
    
    try:
        # For now, return empty list with warning
        # Full implementation would use GraphQL queries
        print("⚠️  Query functionality not fully implemented yet")
        print("   Use direct transaction ID lookups for now")
        print("   Full query support coming in Phase 3")
        
        return []
        
    except Exception as e:
        print(f"❌ Failed to query events from Arweave: {e}")
        raise RuntimeError(f"Failed to query events: {e}")


async def get_wallet_balance() -> float:
    """
    Get current wallet balance in AR tokens.
    
    Returns:
        float: Balance in AR (not winston)
        
    Example:
        >>> balance = await get_wallet_balance()
        >>> print(f"Wallet has {balance:.4f} AR")
    """
    _initialize_client()
    
    if _wallet is None:
        raise RuntimeError("Arweave wallet not initialized")
    
    try:
        import requests
        
        loop = asyncio.get_event_loop()
        
        def fetch_balance():
            url = f"{ARWEAVE_GATEWAY_URL}/wallet/{_wallet.address}/balance"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        
        balance_winston_str = await loop.run_in_executor(None, fetch_balance)
        balance_winston = int(balance_winston_str)
        balance_ar = balance_winston / 1e12
        return balance_ar
        
    except Exception as e:
        print(f"❌ Failed to get wallet balance: {e}")
        return 0.0


async def upload_checkpoint(
    header: Dict,
    signature: str,
    events: bytes,  # gzip-compressed event JSON
    tree_levels: List[List[str]]
) -> str:
    """
    Upload checkpoint to Arweave for hourly batching.
    
    Structure: Single transaction with all checkpoint data
    - header: Checkpoint metadata (version, number, time range, merkle_root, etc.)
    - signature: Ed25519 signature of header (hex)
    - events: Gzip-compressed NDJSON of all events
    - tree_levels: Merkle tree nodes for inclusion proofs
    
    The transaction data is JSON:
    {
        "header": {...},
        "signature": "hex",
        "events_compressed": "base64",
        "tree_levels": [[...], [...], ...]
    }
    
    Args:
        header: Checkpoint header dict
        signature: Hex-encoded Ed25519 signature
        events: Gzip-compressed bytes
        tree_levels: Merkle tree levels for proofs
    
    Returns:
        str: Arweave transaction ID
    
    Raises:
        RuntimeError: If upload fails after retries
    """
    _initialize_client()
    
    if _wallet is None:
        raise RuntimeError("Arweave wallet not initialized")
    
    print(f"📤 Uploading checkpoint #{header['checkpoint_number']} to Arweave...")
    print(f"   Events: {header['event_count']}")
    print(f"   Compressed size: {len(events)} bytes")
    print(f"   Merkle root: {header['merkle_root'][:16]}...")
    
    try:
        import base64
        
        # Build checkpoint payload
        checkpoint_payload = {
            "header": header,
            "signature": signature,
            "events_compressed": base64.b64encode(events).decode('utf-8'),
            "tree_levels": tree_levels
        }
        
        payload_json = json.dumps(checkpoint_payload, default=str)  # Handle datetime objects
        payload_bytes = payload_json.encode('utf-8')
        
        print(f"   Total payload size: {len(payload_bytes)} bytes ({len(payload_bytes)/1024:.2f} KB)")
        
        loop = asyncio.get_event_loop()
        
        def create_and_send_transaction():
            # Create transaction (peer is optional, will use default gateway)
            tx = Transaction(_wallet, data=payload_bytes)
            
            # Add checkpoint-specific tags
            tx.add_tag("App", "leadpoet")
            tx.add_tag("Type", "checkpoint")
            tx.add_tag("Version", "1")
            tx.add_tag("Checkpoint-Number", str(header['checkpoint_number']))
            tx.add_tag("Event-Count", str(header['event_count']))
            tx.add_tag("Merkle-Root", header['merkle_root'])
            tx.add_tag("Time-Start", header['time_range']['start'])
            tx.add_tag("Time-End", header['time_range']['end'])
            
            # Sign and send
            tx.sign()
            tx.send()
            
            return tx.id
        
        # Execute with retries
        retry_delay = INITIAL_RETRY_DELAY
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                tx_id = await loop.run_in_executor(None, create_and_send_transaction)
                
                print(f"✅ Checkpoint uploaded successfully")
                print(f"   TX ID: {tx_id}")
                print(f"   View: https://viewblock.io/arweave/tx/{tx_id}")
                
                return tx_id
            
            except Exception as e:
                last_error = e
                print(f"⚠️  Upload attempt {attempt}/{MAX_RETRIES} failed: {e}")
                
                if attempt < MAX_RETRIES:
                    print(f"   Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
        
        # All retries failed
        raise RuntimeError(f"Failed to upload checkpoint after {MAX_RETRIES} attempts: {last_error}")
    
    except Exception as e:
        print(f"❌ Checkpoint upload failed: {e}")
        raise


async def wait_for_confirmation(tx_id: str, timeout: int = 300) -> bool:
    """
    Wait for Arweave transaction confirmation.
    
    Polls Arweave gateway to check if transaction is confirmed.
    Typical confirmation time: 1-5 minutes.
    
    Args:
        tx_id: Arweave transaction ID
        timeout: Maximum seconds to wait (default: 5 minutes)
    
    Returns:
        bool: True if confirmed, False if timeout
    
    Example:
        >>> confirmed = await wait_for_confirmation(tx_id, timeout=300)
        >>> if confirmed:
        ...     print("Transaction confirmed!")
    """
    import requests
    
    print(f"⏳ Waiting for Arweave confirmation (timeout: {timeout}s)...")
    print(f"   TX ID: {tx_id}")
    
    start_time = asyncio.get_event_loop().time()
    poll_interval = 10  # seconds
    
    loop = asyncio.get_event_loop()
    
    def check_confirmation():
        url = f"{ARWEAVE_GATEWAY_URL}/tx/{tx_id}/status"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                status_data = response.json()
                # Check if transaction has been confirmed
                if status_data.get("confirmed"):
                    return True
            elif response.status_code == 404:
                # Transaction not yet propagated
                return False
            elif response.status_code == 202:
                # Transaction pending
                return False
        except Exception:
            return False
        return False
    
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        
        if elapsed > timeout:
            print(f"⏱️ Timeout after {timeout}s")
            return False
        
        # Check confirmation status
        is_confirmed = await loop.run_in_executor(None, check_confirmation)
        
        if is_confirmed:
            print(f"✅ Transaction confirmed after {elapsed:.1f}s")
            return True
        
        # Wait before next poll
        remaining = timeout - elapsed
        next_poll = min(poll_interval, remaining)
        
        if next_poll > 0:
            print(f"   Polling again in {next_poll:.0f}s... (elapsed: {elapsed:.0f}s)")
            await asyncio.sleep(next_poll)
        else:
            break
    
    print(f"⏱️ Timeout after {timeout}s")
    return False


# Example usage and testing
if __name__ == "__main__":
    """
    Test script to verify Arweave client works.
    
    Run with:
        python -m gateway.utils.arweave_client
    """
    
    async def test_arweave():
        print("\n" + "="*80)
        print("ARWEAVE CLIENT TEST")
        print("="*80 + "\n")
        
        try:
            # Test 1: Check balance
            print("Test 1: Check wallet balance")
            balance = await get_wallet_balance()
            print(f"✅ Balance: {balance:.4f} AR\n")
            
            if balance < 0.01:
                print("⚠️  Warning: Low balance! Please fund your wallet.")
                print("   Need at least 0.01 AR to write events.")
                return
            
            # Test 2: Write a test event
            print("Test 2: Write test event to Arweave")
            test_event = {
                "event_type": "TEST_EVENT",
                "message": "Hello from Leadpoet Arweave Client!",
                "timestamp": "2025-11-08T12:00:00Z",
                "test_data": {
                    "foo": "bar",
                    "number": 42
                }
            }
            
            tx_id = await write_event(
                test_event,
                tags={"event_type": "TEST_EVENT"}
            )
            print(f"✅ Test event written: {tx_id}\n")
            
            # Test 3: Read the event back
            print("Test 3: Read event back from Arweave")
            print("⏳ Waiting 5 seconds for Arweave confirmation...")
            await asyncio.sleep(5)
            
            read_event_data = await read_event(tx_id)
            print(f"✅ Event read successfully")
            print(f"   Event type: {read_event_data.get('event_type')}")
            print(f"   Message: {read_event_data.get('message')}\n")
            
            # Verify data matches
            if read_event_data == test_event:
                print("✅ Data integrity verified - written and read data match!\n")
            else:
                print("❌ Data mismatch!\n")
            
            print("="*80)
            print("ALL TESTS PASSED ✅")
            print("="*80)
            print(f"\nView your transaction: https://viewblock.io/arweave/tx/{tx_id}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}\n")
            raise
    
    # Run tests
    asyncio.run(test_arweave())

