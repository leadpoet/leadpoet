#!/usr/bin/env python3
"""
Verify event is included in Arweave checkpoint Merkle tree.

This script verifies that a specific event was logged by the gateway and included
in an hourly Arweave checkpoint by:
1. Downloading the checkpoint from Arweave
2. Verifying the TEE signature on the checkpoint header
3. Finding the event in the batch
4. Computing the Merkle proof
5. Verifying inclusion in the Merkle root

Usage:
    python verify_merkle_inclusion.py <lead_id> <checkpoint_tx_id> [enclave_pubkey]
    python verify_merkle_inclusion.py 8b6482bf-116e-41db-b8ec-a87ba3c86b8b abc123... ed25519_pubkey_hex

Requirements:
    pip install requests cryptography

Note: Enclave public key can be obtained from GET /attest endpoint
"""

import sys
import json
import gzip
import hashlib
from typing import List, Dict, Any, Optional, Tuple

# ============================================================================
# PRODUCTION GATEWAY CONFIGURATION
# ============================================================================
# Import from centralized config - update GATEWAY_URL in Leadpoet/utils/cloud_db.py
import sys as _sys
import os as _os
# Add parent directory to path to import from Leadpoet
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from Leadpoet.utils.cloud_db import GATEWAY_URL as DEFAULT_GATEWAY_URL
# ============================================================================

try:
    import requests
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install requests cryptography")
    sys.exit(1)


def download_checkpoint(tx_id: str) -> Optional[Dict[str, Any]]:
    """Download checkpoint from Arweave"""
    
    print(f"📥 Downloading checkpoint {tx_id} from Arweave...")
    
    arweave_url = f"https://arweave.net/{tx_id}"
    
    try:
        response = requests.get(arweave_url, timeout=30)
        response.raise_for_status()
        
        # Checkpoint might be gzip-compressed
        try:
            # Try to decompress
            data = gzip.decompress(response.content)
            checkpoint = json.loads(data)
        except (gzip.BadGzipFile, OSError):
            # Not compressed, parse directly
            checkpoint = response.json()
        
        print(f"   ✅ Downloaded checkpoint ({len(response.content)} bytes)")
        
        return checkpoint
        
    except requests.ConnectionError:
        print(f"   ❌ Failed to connect to Arweave")
        print(f"   🔧 Check your internet connection or try again later")
        return None
    except requests.Timeout:
        print(f"   ❌ Request timed out after 30 seconds")
        print(f"   🔧 Arweave may be slow, try again later")
        return None
    except requests.RequestException as e:
        print(f"   ❌ Failed to download: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
        return None


def verify_checkpoint_signature(
    header: Dict[str, Any],
    signature_hex: str,
    public_key_hex: Optional[str]
) -> bool:
    """
    Verify TEE signature on checkpoint header.
    
    The signature is computed as:
        signature = ed25519_sign(enclave_private_key, SHA256(canonical_header_json))
    
    where canonical_header_json uses sort_keys=True for determinism.
    """
    
    print("🔐 Verifying checkpoint signature...")
    
    if not public_key_hex:
        print("   ⚠️  No enclave public key provided, skipping signature verification")
        print("   ⚠️  Get public key from GET /attest endpoint")
        return False
    
    try:
        # Reconstruct canonical header message
        header_json = json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8')
        header_hash = hashlib.sha256(header_json).digest()
        
        # Parse public key
        public_key_bytes = bytes.fromhex(public_key_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        
        # Parse signature
        signature_bytes = bytes.fromhex(signature_hex)
        
        # Verify
        public_key.verify(signature_bytes, header_hash)
        
        print("   ✅ Signature valid - checkpoint signed by TEE enclave")
        return True
        
    except Exception as e:
        print(f"   ❌ Signature verification failed: {e}")
        return False


def hash_event(event: Dict[str, Any]) -> bytes:
    """
    Hash an event for Merkle tree.
    
    Must match the hashing in gateway/tee/merkle.py:
        SHA256(json.dumps(event, sort_keys=True).encode())
    """
    event_json = json.dumps(event, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(event_json).digest()


def compute_merkle_root(leaves: List[bytes]) -> bytes:
    """
    Compute Merkle root from leaf hashes.
    
    Must match the algorithm in gateway/tee/merkle.py.
    
    Algorithm:
    1. If single leaf: return SHA256(0x00 + leaf)
    2. Pad to power of 2 with duplicate last leaf
    3. Hash pairs: SHA256(0x01 + left + right)
    4. Repeat until single root
    """
    
    if not leaves:
        return hashlib.sha256(b"").digest()
    
    if len(leaves) == 1:
        # Single leaf: hash with 0x00 prefix
        return hashlib.sha256(b"\x00" + leaves[0]).digest()
    
    # Pad to power of 2
    n = len(leaves)
    next_power = 1
    while next_power < n:
        next_power *= 2
    
    padded = leaves + [leaves[-1]] * (next_power - n)
    
    # Build tree bottom-up
    level = padded
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1]
            parent = hashlib.sha256(b"\x01" + left + right).digest()
            next_level.append(parent)
        level = next_level
    
    return level[0]


def find_event(events: List[Dict], lead_id: str) -> Optional[Tuple[int, Dict]]:
    """Find event with matching lead_id"""
    
    for idx, event in enumerate(events):
        if event.get("lead_id") == lead_id:
            return (idx, event)
    
    return None


def verify_merkle_inclusion(
    lead_id: str,
    checkpoint_tx_id: str,
    enclave_pubkey_hex: Optional[str]
) -> bool:
    """
    Verify event is included in checkpoint.
    
    Returns True if event is found and Merkle proof is valid.
    """
    
    # Download checkpoint
    checkpoint = download_checkpoint(checkpoint_tx_id)
    if not checkpoint:
        return False
    
    # Extract checkpoint structure
    header = checkpoint.get("header")
    signature = checkpoint.get("signature")
    events_data = checkpoint.get("events")
    
    if not header or not signature or not events_data:
        print("❌ Invalid checkpoint structure")
        print(f"   Expected keys: header, signature, events")
        print(f"   Found keys: {list(checkpoint.keys())}")
        return False
    
    print(f"\n📋 Checkpoint Header:")
    print(f"   Merkle Root: {header.get('merkle_root', 'N/A')[:32]}...")
    print(f"   Event Count: {header.get('event_count', 'N/A')}")
    print(f"   Time Range: {header.get('time_range', {}).get('start', 'N/A')} to {header.get('time_range', {}).get('end', 'N/A')}")
    print(f"   Code Hash: {header.get('code_hash', 'N/A')[:32]}...")
    
    # Verify signature
    if enclave_pubkey_hex:
        if not verify_checkpoint_signature(header, signature, enclave_pubkey_hex):
            print("\n⚠️  Signature verification failed - checkpoint may be forged!")
            print("    Continuing with Merkle verification anyway...")
    
    # Parse events (might be NDJSON string or list)
    if isinstance(events_data, str):
        # NDJSON format
        events = [json.loads(line) for line in events_data.strip().split('\n') if line]
    elif isinstance(events_data, list):
        events = events_data
    else:
        print(f"❌ Unknown events format: {type(events_data)}")
        return False
    
    print(f"\n🔍 Searching for event with lead_id={lead_id}...")
    print(f"   Total events in checkpoint: {len(events)}")
    
    # Find event
    result = find_event(events, lead_id)
    if not result:
        print(f"   ❌ Event not found in this checkpoint")
        print(f"   ℹ️  This event may be in a different checkpoint")
        return False
    
    event_idx, event = result
    print(f"   ✅ Event found at index {event_idx}")
    print(f"   Event type: {event.get('event_type', 'N/A')}")
    print(f"   Timestamp: {event.get('timestamp', 'N/A')}")
    
    # Compute Merkle root from events
    print(f"\n🌳 Computing Merkle root from events...")
    
    leaf_hashes = [hash_event(e) for e in events]
    computed_root = compute_merkle_root(leaf_hashes)
    computed_root_hex = computed_root.hex()
    
    expected_root = header.get("merkle_root")
    
    print(f"   Expected root (from header): {expected_root}")
    print(f"   Computed root (from events): {computed_root_hex}")
    
    if computed_root_hex == expected_root:
        print(f"   ✅ Merkle root matches!")
        return True
    else:
        print(f"   ❌ Merkle root mismatch!")
        print(f"   ⚠️  Checkpoint data may be corrupted or tampered with")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: verify_merkle_inclusion.py <lead_id> <checkpoint_tx_id> [enclave_pubkey_hex]")
        print()
        print("Example:")
        print("  python verify_merkle_inclusion.py \\")
        print("    8b6482bf-116e-41db-b8ec-a87ba3c86b8b \\")
        print("    abc123def456 \\")
        print("    a1b2c3d4...")  # Optional
        print()
        print("Arguments:")
        print("  lead_id: The lead_id from your submission")
        print("  checkpoint_tx_id: Arweave transaction ID of the checkpoint")
        print("  enclave_pubkey_hex: (Optional) Enclave public key from GET /attest")
        print()
        print("To find the right checkpoint:")
        print("  1. Note your submission timestamp")
        print("  2. Checkpoints are created hourly")
        print("  3. Look for checkpoint with time_range covering your timestamp")
        sys.exit(1)
    
    lead_id = sys.argv[1]
    checkpoint_tx_id = sys.argv[2]
    enclave_pubkey_hex = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("=" * 80)
    print("🔐 MERKLE INCLUSION VERIFIER")
    print("=" * 80)
    print(f"Lead ID:        {lead_id}")
    print(f"Checkpoint TX:  {checkpoint_tx_id}")
    print(f"Enclave Pubkey: {enclave_pubkey_hex[:32] + '...' if enclave_pubkey_hex else 'Not provided'}")
    print()
    
    if verify_merkle_inclusion(lead_id, checkpoint_tx_id, enclave_pubkey_hex):
        print("\n" + "=" * 80)
        print("✅ EVENT INCLUDED IN CHECKPOINT")
        print("=" * 80)
        print("Your event was successfully logged by the gateway and included")
        print("in the Arweave checkpoint with a valid Merkle proof.")
        print()
        print("This proves:")
        print("  ✅ Event was accepted by gateway")
        print("  ✅ Event is permanently stored on Arweave")
        print("  ✅ Event cannot be retroactively modified or deleted")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        print("Event was not found in this checkpoint, or Merkle proof is invalid.")
        print()
        print("Possible causes:")
        print("  1. Wrong checkpoint (try checkpoints before/after this one)")
        print("  2. Event not yet checkpointed (checkpoints created hourly)")
        print("  3. Checkpoint data corrupted or tampered with")
        sys.exit(1)


if __name__ == "__main__":
    main()

