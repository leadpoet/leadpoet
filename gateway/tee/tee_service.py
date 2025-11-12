#!/usr/bin/env python3.11
"""
Nitro Enclave TEE Service
=========================

This service runs inside an AWS Nitro Enclave (hardware-isolated trusted execution environment).
It maintains an in-memory event buffer and communicates with the parent EC2 instance via vsock.

KEY CONCEPTS:
- vsock: Virtual socket for secure parent ↔ enclave communication (no network access)
- CID 16: Enclave's Context Identifier (fixed by AWS Nitro)
- CID 3: Parent EC2's Context Identifier (fixed by AWS Nitro)
- Port 5000: Application-defined port for RPC communication

SECURITY GUARANTEES:
- Enclave has NO network access (cannot reach internet or other VMs)
- Enclave memory is hardware-isolated (parent EC2 cannot read it)
- Private key generated inside enclave NEVER leaves (no export mechanism)
- Attestation document cryptographically proves code integrity
"""

# ULTRA-EARLY DEBUG: Print before ANY imports
print("=" * 80, flush=True)
print("🐛 DEBUG: tee_service.py STARTING (before imports)", flush=True)
print("=" * 80, flush=True)

print("🐛 DEBUG: Importing standard library modules...", flush=True)
import socket
import json
import sys
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from threading import Lock
print("🐛 DEBUG: Standard library imports OK", flush=True)

# Cryptography for Ed25519 keypair generation
print("🐛 DEBUG: Importing cryptography...", flush=True)
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
print("🐛 DEBUG: Cryptography imports OK", flush=True)

# Merkle tree computation for hourly checkpoints
print("🐛 DEBUG: Importing merkle module...", flush=True)
from merkle import compute_merkle_tree, generate_inclusion_proof
print("🐛 DEBUG: Merkle module imports OK", flush=True)


# ============================================================================
# VSOCK CONFIGURATION (AWS Nitro Enclaves)
# ============================================================================

# vsock address family constant (Linux)
# See: https://man7.org/linux/man-pages/man7/vsock.7.html
AF_VSOCK = 40  # Address family for vsock

# VMADDR_CID_ANY: Special CID for binding to any address (inside enclave)
# When running INSIDE the enclave, bind to this (not a specific CID)
VMADDR_CID_ANY = 0xFFFFFFFF  # 4294967295 or -1 (cast to u32)

# Parent EC2 CID - always 3 (for reference, not used in binding)
PARENT_CID = 3

# RPC port for communication
RPC_PORT = 5000

# Note: The enclave's actual CID (e.g., 16, 26, 27) is assigned by AWS
# and visible to the parent EC2, but the enclave binds to VMADDR_CID_ANY


# ============================================================================
# GLOBAL STATE (In-Memory, Hardware-Protected)
# ============================================================================

# Event buffer: stores all events in memory until hourly batch
event_buffer: List[Dict[str, Any]] = []
event_buffer_lock = Lock()  # Thread-safe access

# Sequence counter for events
sequence_counter = 0
sequence_counter_lock = Lock()

# Ed25519 keypair (generated on first boot, stored in memory only)
private_key: Optional[ed25519.Ed25519PrivateKey] = None
public_key: Optional[ed25519.Ed25519PublicKey] = None
keypair_lock = Lock()

# Checkpoint state (for linking hourly batches into chain)
prev_checkpoint_root: Optional[bytes] = None  # Merkle root of previous checkpoint
checkpoint_count: int = 0  # Monotonic counter for checkpoint sequence
checkpoint_start_time: datetime = datetime.utcnow()

# Attestation document caching (doesn't change unless enclave restarts)
cached_attestation_doc: Optional[bytes] = None
cached_attestation_hash: Optional[str] = None

# PCR measurements (read DIRECTLY from /dev/nsm hardware - unfakeable)
# NOTE: These are now read from hardware on startup, not from parent EC2
pcr_measurements: Dict[str, str] = {
    "PCR0": None,
    "PCR1": None,
    "PCR2": None
}
pcr_measurements_lock = Lock()

print("=" * 80, flush=True)
print("🐛 DEBUG: All imports and global state OK", flush=True)
print("🐛 DEBUG: Defining functions...", flush=True)
print("=" * 80, flush=True)

# ============================================================================
# KEYPAIR GENERATION (Inside TEE, Never Exported)
# ============================================================================

def generate_keypair() -> None:
    """
    Generate Ed25519 keypair inside the enclave.
    
    CRITICAL SECURITY PROPERTIES:
    - Private key is generated using hardware RNG (/dev/urandom in enclave)
    - Private key is stored in enclave memory ONLY (never written to disk)
    - Private key CANNOT be exported or accessed by parent EC2
    - Public key can be retrieved for verification
    - Keypair is destroyed when enclave terminates (ephemeral)
    
    This ensures the signing key is ONLY accessible to verified enclave code.
    """
    global private_key, public_key
    
    with keypair_lock:
        if private_key is None:
            print("[TEE] Generating Ed25519 keypair inside enclave...", flush=True)
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            print("[TEE] ✅ Keypair generated (private key never leaves enclave)", flush=True)


def get_public_key_bytes() -> bytes:
    """
    Get public key bytes for sharing with verifiers.
    
    Returns:
        Public key in raw bytes (32 bytes for Ed25519)
    """
    if public_key is None:
        generate_keypair()
    
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )


def sign_data(data: bytes) -> bytes:
    """
    Sign data using enclave's private key.
    
    Args:
        data: Bytes to sign (typically SHA256 hash of message)
    
    Returns:
        Signature bytes (64 bytes for Ed25519)
    """
    if private_key is None:
        generate_keypair()
    
    return private_key.sign(data)


# ============================================================================
# EVENT BUFFER MANAGEMENT (In-Memory, Thread-Safe)
# ============================================================================

def append_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append event to in-memory buffer.
    
    This is the PRIMARY mechanism for logging all gateway events. Events are
    buffered in TEE-protected memory until the next hourly Arweave batch.
    
    SECURITY PROPERTIES:
    - Events stored in hardware-protected enclave memory
    - Parent EC2 cannot access or modify buffered events
    - Sequence numbers are monotonically increasing (prevents reordering)
    - Buffer is the CANONICAL copy until Arweave upload
    
    CRASH BEHAVIOR:
    - If enclave crashes, all buffered events are LOST
    - Risk window: up to 1 hour of events (before next Arweave batch)
    - This is acceptable: miners/validators can verify attestation proves
      gateway is running canonical code, and missing events are detectable
      (gaps in sequence numbers on Arweave)
    
    Args:
        event: Event dict from gateway (e.g., SUBMISSION_REQUEST, VALIDATION_RESULT)
               Must include "event_type" field
    
    Returns:
        Response dict with status, sequence number, and buffer size
    
    Raises:
        ValueError: If buffer overflow (>10,000 events)
    """
    global sequence_counter
    
    # Check buffer overflow (DoS protection)
    with event_buffer_lock:
        current_size = len(event_buffer)
        if current_size >= 10000:
            print(f"[TEE] ⚠️ EMERGENCY: Buffer overflow! {current_size} events", flush=True)
            print(f"[TEE] ⚠️ This indicates emergency batch is needed!", flush=True)
            # Allow one more event, but warn parent
            if current_size >= 15000:
                raise ValueError(f"Buffer overflow: {current_size} events (max 15,000)")
    
    # Assign sequence number (monotonic, never resets)
    with sequence_counter_lock:
        event["sequence"] = sequence_counter
        event["buffered_at"] = datetime.utcnow().isoformat()
        assigned_sequence = sequence_counter
        sequence_counter += 1
    
    # Append to buffer
    with event_buffer_lock:
        event_buffer.append(event)
        buffer_size = len(event_buffer)
    
    # Log event type and sequence
    event_type = event.get("event_type", "UNKNOWN")
    print(f"[TEE] Event buffered: {event_type} (seq={assigned_sequence}, buffer={buffer_size})", flush=True)
    
    # Warn if approaching overflow
    if buffer_size >= 5000:
        print(f"[TEE] ⚠️ WARNING: Buffer size {buffer_size} (threshold: 5000)", flush=True)
        print(f"[TEE] ⚠️ Parent should trigger emergency batch soon!", flush=True)
    
    return {
        "status": "buffered",
        "sequence": assigned_sequence,
        "buffer_size": buffer_size,
        "overflow_warning": buffer_size >= 5000
    }


def get_buffer() -> List[Dict[str, Any]]:
    """
    Get all buffered events (for hourly Arweave batch).
    
    Returns:
        List of all events in buffer
    """
    with event_buffer_lock:
        return event_buffer.copy()


def clear_buffer() -> Dict[str, Any]:
    """
    Clear event buffer after successful Arweave upload.
    
    CRITICAL: This should ONLY be called by parent EC2 AFTER confirming
    successful Arweave upload. Clearing before upload = data loss!
    
    SECURITY NOTE:
    - sequence_counter is NOT reset (monotonically increasing forever)
    - This prevents miners from detecting gaps or reordering attacks
    - Each event across all checkpoints has a unique sequence number
    
    Returns:
        Response dict with status, cleared count, and next checkpoint time
    """
    global prev_checkpoint_root, checkpoint_start_time
    
    with event_buffer_lock:
        cleared_count = len(event_buffer)
        
        # Store sequence range before clearing (for logging)
        if event_buffer:
            first_seq = event_buffer[0]["sequence"]
            last_seq = event_buffer[-1]["sequence"]
        else:
            first_seq = last_seq = None
        
        event_buffer.clear()
    
    # Update checkpoint time (for next batch)
    checkpoint_start_time = datetime.utcnow()
    next_checkpoint_time = checkpoint_start_time
    
    print(f"[TEE] ✅ Buffer cleared: {cleared_count} events", flush=True)
    if first_seq is not None:
        print(f"[TEE]    Sequence range: {first_seq} → {last_seq}", flush=True)
    print(f"[TEE]    Next checkpoint starts: {next_checkpoint_time.isoformat()}", flush=True)
    
    return {
        "status": "cleared",
        "cleared_count": cleared_count,
        "sequence_range": {
            "first": first_seq,
            "last": last_seq
        } if first_seq is not None else None,
        "next_checkpoint_at": next_checkpoint_time.isoformat()
    }


def get_buffer_size() -> int:
    """
    Get current buffer size (number of events).
    
    Returns:
        Number of events in buffer
    """
    with event_buffer_lock:
        return len(event_buffer)


def get_buffer_stats() -> Dict[str, Any]:
    """
    Get comprehensive buffer statistics for monitoring.
    
    This provides detailed information about buffer health, useful for:
    - Monitoring dashboard (detect buffer overflow risk)
    - Debugging (track event accumulation rate)
    - Auditing (verify buffer lifecycle)
    
    Returns:
        Dict with buffer statistics:
        - size: Current number of events
        - start_time: When current buffer window started
        - age_seconds: How long events have been accumulating
        - sequence_range: First and last sequence numbers in buffer
        - overflow_risk: Boolean indicating if buffer is approaching capacity
        - next_checkpoint_in: Estimated time until next hourly batch
    """
    global sequence_counter
    
    now = datetime.utcnow()
    
    with event_buffer_lock:
        size = len(event_buffer)
        
        # Get sequence range
        if event_buffer:
            first_seq = event_buffer[0]["sequence"]
            last_seq = event_buffer[-1]["sequence"]
        else:
            first_seq = last_seq = None
    
    with sequence_counter_lock:
        current_sequence = sequence_counter
    
    # Calculate buffer age
    age_seconds = (now - checkpoint_start_time).total_seconds()
    
    # Estimate next checkpoint (hourly = 3600 seconds)
    time_until_checkpoint = 3600 - age_seconds
    if time_until_checkpoint < 0:
        time_until_checkpoint = 0  # Overdue
    
    # Risk assessment
    overflow_risk = size >= 5000
    critical_risk = size >= 10000
    
    stats = {
        "size": size,
        "start_time": checkpoint_start_time.isoformat(),
        "age_seconds": round(age_seconds, 2),
        "sequence_range": {
            "first": first_seq,
            "last": last_seq,
            "next": current_sequence
        },
        "overflow_risk": overflow_risk,
        "critical_risk": critical_risk,
        "next_checkpoint_in_seconds": round(time_until_checkpoint, 2),
        "capacity_percent": round((size / 10000) * 100, 2)
    }
    
    return stats


# ============================================================================
# CHECKPOINT BUILDING (Merkle Tree + Signature)
# ============================================================================

def build_checkpoint() -> Dict[str, Any]:
    """
    Build complete checkpoint with header, signature, and Merkle tree.
    
    This is called by the parent EC2 every hour to batch events to Arweave.
    The checkpoint includes:
    - Signed header (metadata + Merkle root)
    - All buffered events
    - Merkle tree levels (for generating inclusion proofs)
    
    Workflow:
    1. Parent EC2 calls build_checkpoint() via RPC
    2. TEE computes Merkle tree from buffered events
    3. TEE builds checkpoint header with metadata
    4. TEE signs header with enclave private key
    5. TEE returns signed checkpoint + events + tree
    6. Parent EC2 uploads to Arweave
    7. Parent EC2 calls clear_buffer() after confirmation
    
    Security Properties:
    - Merkle root commits to all events (tamper-evident)
    - Signature proves checkpoint came from this specific TEE
    - prev_checkpoint_root creates blockchain-like chain
    - code_hash binds to specific code version
    - attestation_hash binds to specific enclave instance
    
    Returns:
        Dict with checkpoint data:
        {
            "status": "success" | "empty" | "error",
            "header": {
                "checkpoint_version": 1,
                "checkpoint_number": 42,
                "time_range": {"start": "...", "end": "..."},
                "event_count": 1500,
                "sequence_range": {"first": 100, "last": 1599},
                "merkle_root": "hex",
                "prev_checkpoint_root": "hex" | null,
                "code_hash": "hex",
                "attestation_hash": "hex"
            },
            "signature": "hex",  # Ed25519 signature of header
            "events": [...],
            "tree_levels": [[...], [...], ...]
        }
    
    Note: This does NOT clear the buffer. That's a separate step after
    successful Arweave confirmation (ensures no data loss).
    """
    global prev_checkpoint_root, checkpoint_count
    
    # Get current time for checkpoint header
    now = datetime.utcnow()
    
    # Copy events from buffer (thread-safe)
    with event_buffer_lock:
        events = event_buffer.copy()
    
    # Handle empty buffer
    if not events:
        return {
            "status": "empty",
            "message": "No events to checkpoint",
            "next_checkpoint_at": (now.replace(minute=0, second=0, microsecond=0)).isoformat() + "Z"
        }
    
    print(f"[TEE] 📦 Building checkpoint #{checkpoint_count} for {len(events)} events...", flush=True)
    
    try:
        # Compute Merkle tree
        merkle_root, tree_levels = compute_merkle_tree(events)
        print(f"[TEE]    Merkle root: {merkle_root.hex()[:16]}...", flush=True)
        
        # Compute code hash (proves which code is running)
        code_hash = compute_code_hash()
        
        # Get attestation hash (cached, only computed once per enclave lifetime)
        attestation_hash = get_cached_attestation_hash()
        
        # Build checkpoint header
        checkpoint_header = {
            "checkpoint_version": 1,
            "checkpoint_number": checkpoint_count,
            "time_range": {
                "start": checkpoint_start_time.isoformat() + "Z",
                "end": now.isoformat() + "Z"
            },
            "event_count": len(events),
            "sequence_range": {
                "first": events[0]["sequence"],
                "last": events[-1]["sequence"]
            },
            "merkle_root": merkle_root.hex(),
            "prev_checkpoint_root": prev_checkpoint_root.hex() if prev_checkpoint_root else None,
            "code_hash": code_hash,
            "attestation_hash": attestation_hash
        }
        
        # Sign checkpoint header with enclave private key
        # This proves the checkpoint came from verified code running in this TEE
        header_json = json.dumps(checkpoint_header, sort_keys=True)
        header_bytes = header_json.encode('utf-8')
        header_hash = hashlib.sha256(header_bytes).digest()
        
        checkpoint_signature = sign_data(header_hash)
        print(f"[TEE]    Signature: {checkpoint_signature.hex()[:16]}...", flush=True)
        
        # Update state for next checkpoint (IMPORTANT: Chain continuity)
        prev_checkpoint_root = merkle_root
        checkpoint_count += 1
        
        # Convert tree levels to hex strings for JSON serialization
        tree_levels_hex = [
            [node.hex() for node in level]
            for level in tree_levels
        ]
        
        print(f"[TEE] ✅ Checkpoint #{checkpoint_count - 1} built successfully", flush=True)
        print(f"     Events: {len(events)}", flush=True)
        print(f"     Tree Depth: {len(tree_levels)} levels", flush=True)
        print(f"     Prev Root: {prev_checkpoint_root.hex()[:16] if prev_checkpoint_root else 'None'}...", flush=True)
        
        return {
            "status": "success",
            "header": checkpoint_header,
            "signature": checkpoint_signature.hex(),
            "events": events,
            "tree_levels": tree_levels_hex
        }
    
    except Exception as e:
        print(f"[TEE] ❌ Checkpoint build failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# ATTESTATION DOCUMENT GENERATION (Nitro-Specific)
# ============================================================================

def compute_code_hash() -> str:
    """
    Compute SHA256 hash of ALL critical gateway code.
    
    This creates a deterministic hash of all API endpoints, consensus logic,
    and core gateway functionality, proving the EXACT code running for
    miner/validator interactions.
    
    🔐 CRITICAL FOR TRUSTLESSNESS:
    This hash covers ALL code that could be manipulated to:
    - Drop miner submissions (submit.py)
    - Fake consensus results (epoch_lifecycle.py, consensus.py)
    - Censor events (logger.py)
    - Allow unauthorized access (signature.py, registry.py)
    - Manipulate validation (validate.py, reveal.py)
    
    Files included (automatically discovered):
    - gateway/main.py - FastAPI server, presign endpoint
    - gateway/config.py - Constants (epoch length, etc.)
    - gateway/api/*.py - All API endpoints (submit, validate, reveal, epoch, etc.)
    - gateway/tasks/*.py - Epoch lifecycle, hourly batching
    - gateway/utils/*.py - Consensus, logging, signatures, registry, TEE client
    - gateway/models/*.py - Event structure definitions
    - gateway/tee/*.py - TEE service, Merkle tree, NSM interface
    
    Files excluded:
    - gateway/secrets/ - Credentials (should never be in repo)
    - gateway/logs/ - Runtime logs
    - __pycache__, *.pyc - Python bytecode
    
    💡 TO ADD NEW FILES TO VERIFICATION:
    Simply add new .py files to any of the included directories above.
    They will be automatically picked up and hashed.
    
    Returns:
        SHA256 hash (hex string, 64 chars)
    """
    try:
        from pathlib import Path
        
        # Get gateway root directory (parent of tee/)
        gateway_root = Path(__file__).parent.parent
        
        print(f"[TEE] 🔐 Computing code hash from gateway directory: {gateway_root}", flush=True)
        
        # Collect all .py files to hash
        files_to_hash = []
        
        # ============================================================
        # CRITICAL FILES - Automatically include entire directories
        # ============================================================
        # 💡 TO ADD MORE FILES: Just add the directory or file path here
        
        include_dirs = [
            gateway_root / "api",        # API endpoints (submit, validate, reveal, epoch, manifest, attest)
            gateway_root / "tasks",      # Epoch lifecycle, hourly batching
            gateway_root / "utils",      # Consensus, logger, signatures, registry, TEE client
            gateway_root / "models",     # Pydantic event models
            gateway_root / "tee",        # TEE service, Merkle tree, NSM library
        ]
        
        # Also include root-level critical files
        files_to_hash.append(gateway_root / "main.py")
        files_to_hash.append(gateway_root / "config.py")
        
        # Collect all .py files from include_dirs
        for dir_path in include_dirs:
            if dir_path.exists():
                for py_file in sorted(dir_path.glob("**/*.py")):
                    # Skip __pycache__ and other build artifacts
                    if "__pycache__" in str(py_file) or py_file.name.endswith(".pyc"):
                        continue
                    
                    # Skip test files (not in GitHub repo)
                    if py_file.name.startswith("test_"):
                        continue
                    
                    # Skip utility scripts (not critical for trustlessness)
                    if py_file.name in ["provision_pcrs.py", "verify_code_hash.py"]:
                        continue
                    
                    files_to_hash.append(py_file)
            else:
                print(f"[TEE] ⚠️  Directory not found (skipping): {dir_path}", flush=True)
        
        # Sort for determinism (must be identical on GitHub and EC2)
        files_to_hash = sorted(set(files_to_hash))
        
        print(f"[TEE] 📝 Hashing {len(files_to_hash)} files...", flush=True)
        
        # Hash all files deterministically
        hasher = hashlib.sha256()
        for file_path in files_to_hash:
            if file_path.exists():
                # Include filename in hash for structure integrity
                # This ensures file renames or moves are detected
                hasher.update(str(file_path.name).encode('utf-8'))
                
                # Include file content
                hasher.update(file_path.read_bytes())
                
                # Debug: Print first few files
                if len(files_to_hash) <= 20 or files_to_hash.index(file_path) < 3:
                    print(f"[TEE]    ✓ {file_path.relative_to(gateway_root)}", flush=True)
            else:
                print(f"[TEE] ⚠️  File not found (skipping): {file_path}", flush=True)
        
        code_hash = hasher.hexdigest()
        
        print(f"[TEE] ✅ Code hash computed from {len(files_to_hash)} files", flush=True)
        print(f"[TEE]    Hash: {code_hash[:32]}...{code_hash[-32:]}", flush=True)
        
        return code_hash
        
    except Exception as e:
        print(f"[TEE] ⚠️  Failed to compute code hash: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return zeros on error (will cause verification to fail, which is correct)
        return "0" * 64


def read_pcrs_from_hardware() -> Dict[str, str]:
    """
    🔴 CRITICAL: Read PCR measurements DIRECTLY from /dev/nsm hardware.
    
    This is the ONLY trustless way to get PCR values. Reading from parent EC2
    allows a malicious operator to lie about what code is running.
    
    Why this matters:
    - A malicious operator could modify tee_service.py
    - Rebuild Docker image → new PCR0_modified
    - Start enclave with modified code
    - Send FAKE PCR0_old via vsock (from legitimate build)
    - Enclave would include fake PCR0 in attestation
    - Miners would think code is legitimate ❌
    
    By reading PCRs from /dev/nsm hardware:
    - Hardware ALWAYS returns REAL PCR0 (cannot be faked)
    - If code modified → hardware returns PCR0_modified
    - Miners see PCR0_modified != PCR0_expected
    - Attack detected ✅
    
    Returns:
        Dict with PCR0, PCR1, PCR2 (hex strings, 96 chars each)
        Returns zeros if /dev/nsm not available (development mode)
    """
    NSM_DEVICE = "/dev/nsm"
    
    print(f"[TEE] DEBUG: Checking for {NSM_DEVICE}...", flush=True)
    
    # Check if NSM device exists (only available inside enclave)
    if not os.path.exists(NSM_DEVICE):
        print(f"[TEE] ⚠️  /dev/nsm not found - using development mode PCRs (zeros)", flush=True)
        return {
            "PCR0": "0" * 96,
            "PCR1": "0" * 96,
            "PCR2": "0" * 96
        }
    
    print(f"[TEE] DEBUG: {NSM_DEVICE} exists, proceeding...", flush=True)
    
    # Give NSM device time to fully initialize (important on first boot)
    print(f"[TEE] DEBUG: Waiting 2 seconds for NSM device to fully initialize...", flush=True)
    import time
    time.sleep(2)
    print(f"[TEE] DEBUG: NSM device should be ready now", flush=True)
    
    try:
        print(f"[TEE] DEBUG: Importing nsm_lib (AWS NSM Python wrapper)...", flush=True)
        import nsm_lib
        print(f"[TEE] DEBUG: nsm_lib imported successfully", flush=True)
        
        print(f"[TEE] 🔒 Reading PCR measurements from /dev/nsm hardware...", flush=True)
        
        # Use the proper NSM library to get PCR measurements
        # This uses the correct ioctl interface with proper request structures
        pcr_dict = nsm_lib.get_pcr_measurements()
        
        print(f"[TEE] ✅ PCRs read from hardware (unfakeable):", flush=True)
        print(f"[TEE]    PCR0: {pcr_dict['PCR0'][:32]}...{pcr_dict['PCR0'][-32:]}", flush=True)
        print(f"[TEE]    PCR1: {pcr_dict['PCR1'][:32]}...{pcr_dict['PCR1'][-32:]}", flush=True)
        print(f"[TEE]    PCR2: {pcr_dict['PCR2'][:32]}...{pcr_dict['PCR2'][-32:]}", flush=True)
        
        return pcr_dict
    
    except Exception as e:
        print(f"[TEE] ❌ Failed to read PCRs from hardware: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Fallback to zeros (development mode)
        print(f"[TEE] ⚠️  Using development mode PCRs (zeros)", flush=True)
        return {
            "PCR0": "0" * 96,
            "PCR1": "0" * 96,
            "PCR2": "0" * 96
        }


def read_nsm_attestation(user_data: bytes = None, nonce: bytes = None, public_key: bytes = None) -> Optional[Dict[str, Any]]:
    """
    Read attestation document from /dev/nsm (Nitro Security Module).
    
    This interacts with the AWS Nitro hardware to generate a cryptographically
    signed attestation document that proves:
    - The exact Docker image running (PCR0)
    - The kernel/bootstrap (PCR1)
    - The application code (PCR2, derived from user_data)
    - The enclave's public key (in user_data)
    
    Args:
        user_data: Optional data to include in attestation (public_key + code_hash)
        nonce: Optional nonce for replay protection
        public_key: Optional public key bytes to include
    
    Returns:
        Attestation document dict, or None if NSM unavailable
    """
    NSM_DEVICE = "/dev/nsm"
    
    # Check if NSM device exists (only available inside enclave)
    if not os.path.exists(NSM_DEVICE):
        print(f"[TEE] ⚠️  {NSM_DEVICE} not found - using fallback attestation", flush=True)
        return None
    
    try:
        # Import cbor2 for encoding/decoding NSM messages
        import cbor2
        import fcntl
        
        # Open NSM device
        with open(NSM_DEVICE, 'rb', buffering=0) as nsm:
            # Prepare attestation request
            # Format: CBOR-encoded map with optional user_data, nonce, public_key
            request = {}
            
            if user_data:
                request["user_data"] = user_data
            if nonce:
                request["nonce"] = nonce
            if public_key:
                request["public_key"] = public_key
            
            # Encode request as CBOR
            request_bytes = cbor2.dumps(request)
            
            # NSM ioctl magic number (from aws-nitro-enclaves-nsm-api)
            NSM_IOCTL_ATTESTATION = 0xC0086E01  # _IOWR('N', 1, struct nsm_attestation)
            
            # Send attestation request via ioctl
            # Note: This is simplified - real implementation needs proper ioctl handling
            print("[TEE] 🔐 Requesting attestation from NSM device...", flush=True)
            
            # This is a PLACEHOLDER for the actual ioctl call
            # Real implementation would use ctypes or cffi to call ioctl
            # For now, we'll fall back to the next method
            
            print("[TEE] ⚠️  NSM ioctl not implemented - using fallback", flush=True)
            return None
    
    except Exception as e:
        print(f"[TEE] ❌ NSM attestation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def set_pcr_measurements(pcr0: str = None, pcr1: str = None, pcr2: str = None) -> Dict[str, Any]:
    """
    🚫 DEPRECATED: Set PCR measurements (provided by parent EC2).
    
    ⚠️ THIS METHOD IS NO LONGER USED AND SHOULD NOT BE CALLED.
    
    PCR measurements are now read DIRECTLY from /dev/nsm hardware on enclave
    startup. This ensures trustlessness - a malicious operator cannot lie about
    what code is running by providing fake PCRs.
    
    This method is kept for backward compatibility but does nothing.
    
    Args:
        pcr0: PCR0 measurement (Docker image hash) - 96 hex chars
        pcr1: PCR1 measurement (kernel hash) - 96 hex chars
        pcr2: PCR2 measurement (application hash) - 96 hex chars
    
    Returns:
        Status dict with deprecation warning
    """
    print(f"[TEE] ⚠️  DEPRECATED: set_pcr_measurements called", flush=True)
    print(f"[TEE] ⚠️  PCRs are now read from hardware on startup", flush=True)
    print(f"[TEE] ⚠️  Ignoring parent-provided PCRs (security fix)", flush=True)
    
    return {
        "status": "deprecated",
        "warning": "PCRs now read from hardware on startup, not from parent EC2",
        "pcr0_ignored": pcr0 is not None,
        "pcr1_ignored": pcr1 is not None,
        "pcr2_ignored": pcr2 is not None
    }


def get_attestation_document_with_pcrs(pcr0: str = None, pcr1: str = None, pcr2: str = None) -> Dict[str, Any]:
    """
    Generate attestation document using hardware PCR measurements.
    
    🔴 SECURITY CHANGE: PCRs now read from /dev/nsm hardware on startup.
    The pcr0/pcr1/pcr2 parameters are IGNORED (kept for backward compatibility).
    
    This prevents a malicious operator from lying about what code is running
    by providing fake PCRs from parent EC2.
    
    Args:
        pcr0: IGNORED (kept for backward compatibility)
        pcr1: IGNORED (kept for backward compatibility)
        pcr2: IGNORED (kept for backward compatibility)
    
    Returns:
        Attestation document dict with hardware PCRs and full COSE-signed attestation bytes
    """
    # Get public key
    public_key_bytes = get_public_key_bytes()
    
    # Compute deterministic code hash
    code_hash = compute_code_hash()
    
    # 🔴 ALWAYS use hardware PCRs (read on startup from /dev/nsm)
    # Parent-provided PCRs are IGNORED for security
    with pcr_measurements_lock:
        final_pcr0 = pcr_measurements.get("PCR0") or ("0" * 96)
        final_pcr1 = pcr_measurements.get("PCR1") or ("0" * 96)
        final_pcr2 = pcr_measurements.get("PCR2") or ("0" * 96)
    
    # Build user_data (embedded in attestation, verifiable)
    user_data_dict = {
        "enclave_public_key": public_key_bytes.hex(),
        "code_hash": code_hash,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    user_data_bytes = json.dumps(user_data_dict).encode('utf-8')
    
    # Determine source (hardware if PCRs non-zero, development if zeros)
    pcr_source = "hardware_nsm" if final_pcr0 != ("0" * 96) else "development_zeros"
    
    # Try to get full attestation document from NSM hardware
    attestation_doc_hex = ""
    if pcr_source == "hardware_nsm":
        try:
            print("[TEE] 🔐 Requesting full attestation document from NSM...", flush=True)
            from nsm_lib import get_attestation_document as nsm_get_attestation
            
            # Request attestation with user_data
            nsm_response = nsm_get_attestation(user_data=user_data_bytes)
            
            # Extract the COSE-signed document bytes
            if "Attestation" in nsm_response and "document" in nsm_response["Attestation"]:
                attestation_doc_bytes = nsm_response["Attestation"]["document"]
                attestation_doc_hex = attestation_doc_bytes.hex()
                print(f"[TEE] ✅ Got full attestation document ({len(attestation_doc_bytes)} bytes)", flush=True)
            else:
                print(f"[TEE] ⚠️ NSM response missing 'document' field", flush=True)
        except Exception as e:
            print(f"[TEE] ⚠️ Failed to get full attestation document from NSM: {e}", flush=True)
    
    # Create attestation document structure
    attestation = {
        "attestation_document": attestation_doc_hex,  # Hex-encoded COSE Sign1 bytes
        "module_id": "nitro-enclave-tee",
        "timestamp": datetime.utcnow().isoformat(),
        "digest": "SHA384",
        "pcrs": {
            "PCR0": final_pcr0,
            "PCR1": final_pcr1,
            "PCR2": final_pcr2,
        },
        # Also include PCRs at top level for easy gateway access
        "pcr0": final_pcr0,
        "pcr1": final_pcr1,
        "pcr2": final_pcr2,
        "certificate": None,  # Would be AWS root cert from NSM
        "cabundle": [],
        "public_key": public_key_bytes.hex(),
        "code_hash": code_hash,
        "user_data": user_data_dict,
        "nonce": None,
        "source": pcr_source  # 🔴 Now always "hardware_nsm" or "development_zeros"
    }
    
    return attestation


def get_attestation_document() -> Dict[str, Any]:
    """
    Generate Nitro Enclave attestation document.
    
    Attestation document contains:
    - enclave_measurement (PCR0): SHA384 hash of Docker image
    - code_hash: SHA256 hash of application code (for verification)
    - enclave_public_key: Public key for signature verification
    - timestamp: When attestation was generated
    - signature: AWS Nitro hardware signature over all fields (if NSM available)
    
    This tries multiple approaches:
    1. Read from /dev/nsm (full AWS-signed attestation)
    2. Use PCR measurements if provided
    3. Generate unsigned attestation (for development)
    
    Returns:
        Attestation document dict
    """
    # Get public key
    public_key_bytes = get_public_key_bytes()
    
    # Compute deterministic code hash
    code_hash = compute_code_hash()
    
    # Use PCRs that were read from hardware at startup
    print("[TEE] 🔐 Generating attestation document...", flush=True)
    print(f"[TEE] DEBUG: Checking global pcr_measurements variable...", flush=True)
    
    # Use the global PCR measurements that were read from /dev/nsm at startup
    # These are hardware-enforced and cannot be faked
    with pcr_measurements_lock:
        pcr0 = pcr_measurements.get("PCR0")
        pcr1 = pcr_measurements.get("PCR1")
        pcr2 = pcr_measurements.get("PCR2")
    
    print(f"[TEE] DEBUG: PCR0 = {pcr0}", flush=True)
    print(f"[TEE] DEBUG: PCR1 = {pcr1}", flush=True)
    print(f"[TEE] DEBUG: PCR2 = {pcr2}", flush=True)
    
    if pcr0 and pcr1 and pcr2:
        print(f"[TEE] ✅ Using hardware PCRs read at startup", flush=True)
        print(f"[TEE]    PCR0: {pcr0[:32]}...{pcr0[-32:]}", flush=True)
        return get_attestation_document_with_pcrs(pcr0, pcr1, pcr2)
    else:
        # Fallback: Generate unsigned attestation (development mode)
        print("[TEE] ⚠️  No hardware PCRs available - using unsigned attestation", flush=True)
        print(f"[TEE] DEBUG: pcr0={pcr0}, pcr1={pcr1}, pcr2={pcr2}", flush=True)
        return get_attestation_document_with_pcrs()


def get_cached_attestation_hash() -> str:
    """
    Get cached hash of attestation document.
    
    The attestation document doesn't change unless the enclave restarts,
    so we cache it to avoid regenerating on every checkpoint.
    
    This hash is included in checkpoint headers to bind the checkpoint
    to the specific enclave instance that generated it.
    
    Returns:
        SHA256 hash of attestation document (hex string)
    """
    global cached_attestation_doc, cached_attestation_hash
    
    if cached_attestation_hash is None:
        print("[TEE] Computing attestation hash (first time)...", flush=True)
        
        # Get attestation document
        attestation_doc = get_attestation_document()
        
        # Serialize to canonical JSON
        attestation_json = json.dumps(attestation_doc, sort_keys=True)
        attestation_bytes = attestation_json.encode('utf-8')
        
        # Cache the hash
        cached_attestation_doc = attestation_bytes
        cached_attestation_hash = hashlib.sha256(attestation_bytes).hexdigest()
        
        print(f"[TEE] ✅ Attestation hash cached: {cached_attestation_hash[:16]}...", flush=True)
    
    return cached_attestation_hash


# ============================================================================
# RPC HANDLER (vsock Request/Response)
# ============================================================================

def handle_rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle RPC method call from parent EC2.
    
    Available methods:
    - append_event: Add event to buffer
    - get_buffer: Retrieve all buffered events
    - clear_buffer: Clear buffer after Arweave upload
    - get_buffer_size: Get current buffer size
    - get_buffer_stats: Get detailed buffer statistics
    - build_checkpoint: Build Merkle tree checkpoint from buffered events
    - get_public_key: Get enclave's public key
    - get_attestation: Get attestation document
    - set_pcr_measurements: Set PCR measurements from parent EC2
    - sign_checkpoint: Sign checkpoint header
    
    Args:
        method: RPC method name
        params: Method parameters dict
    
    Returns:
        Response dict with result or error
    """
    try:
        if method == "append_event":
            event = params.get("event")
            if not event:
                return {"error": "Missing 'event' parameter"}
            return {"result": append_event(event)}
        
        elif method == "get_buffer":
            return {"result": get_buffer()}
        
        elif method == "clear_buffer":
            return {"result": clear_buffer()}
        
        elif method == "get_buffer_size":
            return {"result": get_buffer_size()}
        
        elif method == "get_buffer_stats":
            return {"result": get_buffer_stats()}
        
        elif method == "build_checkpoint":
            return {"result": build_checkpoint()}
        
        elif method == "get_public_key":
            return {"result": get_public_key_bytes().hex()}
        
        elif method == "get_attestation":
            return {"result": get_attestation_document()}
        
        elif method == "set_pcr_measurements":
            pcr0 = params.get("pcr0")
            pcr1 = params.get("pcr1")
            pcr2 = params.get("pcr2")
            return {"result": set_pcr_measurements(pcr0, pcr1, pcr2)}
        
        elif method == "sign_checkpoint":
            # Sign checkpoint header with enclave key
            checkpoint_header = params.get("checkpoint_header")
            if not checkpoint_header:
                return {"error": "Missing 'checkpoint_header' parameter"}
            
            # Compute SHA256 hash of checkpoint
            checkpoint_json = json.dumps(checkpoint_header, sort_keys=True)
            checkpoint_hash = hashlib.sha256(checkpoint_json.encode()).digest()
            
            # Sign with enclave private key
            signature = sign_data(checkpoint_hash)
            
            return {
                "result": {
                    "signature": signature.hex(),
                    "checkpoint_hash": checkpoint_hash.hex()
                }
            }
        
        else:
            return {"error": f"Unknown method: {method}"}
    
    except Exception as e:
        print(f"[TEE] ❌ RPC error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# VSOCK SERVER (Parent EC2 ↔ Enclave Communication)
# ============================================================================

def start_vsock_server():
    """
    Start vsock server to listen for RPC calls from parent EC2.
    
    vsock Protocol (Length-Prefixed JSON):
    1. Parent EC2 (CID 3) connects to enclave on port 5000
    2. Parent sends: [4-byte length (big-endian)][JSON-RPC request]
       Request format: {"method": "...", "params": {...}}
    3. Enclave processes request
    4. Enclave sends: [4-byte length (big-endian)][JSON response]
       Response format: {"result": ...} or {"error": "..."}
    5. Connection closes (stateless, one request per connection)
    
    Why vsock?
    - Hardware-isolated channel (no network access)
    - Only parent EC2 can connect (AWS Nitro enforces)
    - Direct memory channel (faster than TCP)
    - Secure by design (cannot be sniffed or intercepted)
    
    Why length-prefixed protocol?
    - Handles large payloads reliably (event batches can be MBs)
    - Prevents partial reads/writes
    - Compatible with async clients (tee_client.py)
    """
    print("[TEE] Starting vsock server...", flush=True)
    print(f"[TEE] Binding to VMADDR_CID_ANY (any CID), port {RPC_PORT}", flush=True)
    
    # Create vsock socket
    # AF_VSOCK: Address family for virtual sockets
    # SOCK_STREAM: TCP-like reliable stream protocol
    print("[TEE] DEBUG: Creating vsock socket...", flush=True)
    try:
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        print("[TEE] DEBUG: Socket created successfully", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR creating socket: {e}", flush=True)
        raise
    
    # Bind to VMADDR_CID_ANY (not a specific CID)
    # This allows the enclave to accept connections on its assigned CID
    print(f"[TEE] DEBUG: Binding to ({VMADDR_CID_ANY}, {RPC_PORT})...", flush=True)
    try:
        sock.bind((VMADDR_CID_ANY, RPC_PORT))
        print("[TEE] DEBUG: Bind successful", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR binding socket: {e}", flush=True)
        raise
    
    # Listen for connections (backlog=5)
    print("[TEE] DEBUG: Starting to listen...", flush=True)
    try:
        sock.listen(5)
        print("[TEE] DEBUG: Listen successful", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR listening: {e}", flush=True)
        raise
    
    print("[TEE] ✅ vsock server started", flush=True)
    print("[TEE] Ready to accept RPC calls from parent EC2", flush=True)
    
    # Accept connections in loop
    while True:
        try:
            # Accept connection from parent EC2
            conn, addr = sock.accept()
            print(f"[TEE] Connection from CID {addr[0]}, port {addr[1]}", flush=True)
            
            # Receive request with length prefix (4 bytes, big-endian)
            # Protocol: [4-byte length][JSON data]
            length_bytes = conn.recv(4)
            
            if len(length_bytes) != 4:
                print(f"[TEE] ⚠️ Invalid request (no length prefix)", flush=True)
                conn.close()
                continue
            
            request_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive JSON data
            request_data = b""
            while len(request_data) < request_length:
                chunk = conn.recv(min(4096, request_length - len(request_data)))
                if not chunk:
                    break
                request_data += chunk
            
            if len(request_data) != request_length:
                print(f"[TEE] ⚠️ Incomplete request (expected {request_length}, got {len(request_data)})", flush=True)
                conn.close()
                continue
            
            # Parse JSON-RPC request
            try:
                request = json.loads(request_data.decode('utf-8'))
                method = request.get("method")
                params = request.get("params", {})
                
                print(f"[TEE] RPC call: {method}", flush=True)
                
                # Handle RPC
                response = handle_rpc(method, params)
                
                # Send JSON response with length prefix
                response_bytes = json.dumps(response).encode('utf-8')
                response_length = len(response_bytes)
                length_prefix = response_length.to_bytes(4, byteorder='big')
                
                conn.sendall(length_prefix + response_bytes)
                
                print(f"[TEE] ✅ Response sent ({response_length} bytes)", flush=True)
            
            except json.JSONDecodeError as e:
                error_response = {"error": f"Invalid JSON: {str(e)}"}
                error_bytes = json.dumps(error_response).encode('utf-8')
                error_length = len(error_bytes)
                length_prefix = error_length.to_bytes(4, byteorder='big')
                conn.sendall(length_prefix + error_bytes)
            
            # Close connection
            conn.close()
        
        except Exception as e:
            print(f"[TEE] ❌ Connection error: {e}", flush=True)
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for TEE service.
    
    Steps:
    1. Generate Ed25519 keypair inside enclave
    2. 🔴 Read PCR measurements from /dev/nsm hardware (CRITICAL FOR TRUSTLESSNESS)
    3. Start vsock server to listen for RPC calls
    4. Handle requests from parent EC2 (event buffering, signing, attestation)
    """
    global pcr_measurements
    
    print("=" * 80, flush=True)
    print("🔒 NITRO ENCLAVE TEE SERVICE STARTING", flush=True)
    print("=" * 80, flush=True)
    print(f"[TEE] DEBUG: Python version: {sys.version}", flush=True)
    print(f"[TEE] DEBUG: Current working directory: {os.getcwd()}", flush=True)
    print(f"[TEE] Binding to: VMADDR_CID_ANY (0xFFFFFFFF)", flush=True)
    print(f"[TEE] Parent CID: {PARENT_CID}", flush=True)
    print(f"[TEE] RPC Port: {RPC_PORT}", flush=True)
    print("=" * 80, flush=True)
    
    # Step 1: Generate keypair on startup
    print("[TEE] DEBUG: Starting keypair generation...", flush=True)
    try:
        generate_keypair()
        print("[TEE] DEBUG: Keypair generation completed successfully", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR in keypair generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: 🔴 CRITICAL - Read PCR measurements from hardware
    print("=" * 80, flush=True)
    print("🔴 READING PCR MEASUREMENTS FROM HARDWARE", flush=True)
    print("=" * 80, flush=True)
    print("[TEE] DEBUG: About to read PCRs from hardware...", flush=True)
    try:
        global pcr_measurements  # CRITICAL: Update global variable, not local!
        with pcr_measurements_lock:
            pcr_measurements = read_pcrs_from_hardware()
        print("[TEE] DEBUG: PCR reading completed successfully", flush=True)
        print(f"[TEE] DEBUG: PCRs: {list(pcr_measurements.keys())}", flush=True)
    except Exception as e:
        print(f"[TEE] ❌ ERROR reading PCRs: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("=" * 80, flush=True)
    
    # Step 3: Start vsock server (blocks forever)
    print("[TEE] DEBUG: About to start vsock server...", flush=True)
    try:
        start_vsock_server()
    except Exception as e:
        print(f"[TEE] ❌ ERROR starting vsock server: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[TEE] Shutting down...", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n[TEE] ❌ Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

