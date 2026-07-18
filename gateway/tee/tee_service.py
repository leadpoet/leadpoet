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
import base64
from copy import deepcopy
from datetime import datetime
from pathlib import Path
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

print("🐛 DEBUG: Importing code hash module...", flush=True)
try:
    from gateway.tee.code_hash import compute_gateway_code_hash
except Exception:
    from code_hash import compute_gateway_code_hash
print("🐛 DEBUG: Code hash module imports OK", flush=True)


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
MAX_RPC_REQUEST_BYTES = 64 * 1024 * 1024

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
pending_checkpoint: Optional[Dict[str, Any]] = None
checkpoint_state_lock = Lock()

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

# The coordinator provider relay is initialized lazily after V2 runtime lock.
provider_egress_proxy = None
provider_egress_proxy_lock = Lock()
v2_runtime_identity = None
v2_runtime_identity_lock = Lock()
v2_peer_registry = None
v2_peer_registry_lock = Lock()
v2_tls_server = None
v2_tls_server_thread = None
v2_tls_server_lock = Lock()
v2_provider_broker = None
v2_provider_broker_lock = Lock()
v2_provider_cache_store = None
v2_provider_cache_store_lock = Lock()
v2_provider_outcome_store = None
v2_provider_outcome_store_lock = Lock()
v2_provider_semantics_authority = None
v2_provider_semantics_authority_lock = Lock()
v2_provider_evidence_authority = None
v2_provider_evidence_authority_lock = Lock()
v2_inter_enclave_client = None
v2_inter_enclave_client_lock = Lock()
v2_scoring_job_manager = None
v2_scoring_job_manager_lock = Lock()
v2_autoresearch_job_manager = None
v2_autoresearch_job_manager_lock = Lock()
v2_coordinator_job_manager = None
v2_coordinator_job_manager_lock = Lock()
v2_kms_recipient = None
v2_kms_recipient_lock = Lock()
v2_artifact_vault = None
v2_artifact_vault_lock = Lock()
v2_inter_enclave_artifact_ingest = None
v2_inter_enclave_artifact_ingest_lock = Lock()
v2_artifact_persistence_verifier = None
v2_artifact_persistence_verifier_lock = Lock()
v2_ingress_seal_cache = {}
v2_ingress_seal_cache_lock = Lock()
event_signer_initialization_lock = Lock()
event_signer_initialization = None

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


def _event_signer_module():
    try:
        from gateway.tee import enclave_signer
    except ImportError:
        import enclave_signer
    return enclave_signer


def _event_signing_identity() -> Dict[str, Any]:
    signer = _event_signer_module()
    if not signer.is_keypair_initialized():
        raise RuntimeError("transparency event signer is not initialized")
    attestation_document = signer.get_attestation_document()
    return {
        "purpose": "gateway_event_signing",
        "enclave_pubkey": signer.get_enclave_public_key_hex(),
        "code_hash": signer.get_cached_code_hash(),
        "attestation_document_b64": base64.b64encode(attestation_document).decode(
            "ascii"
        ),
        "signer_state": signer.get_signer_state(),
    }


def initialize_event_signer(prev_log_tip_hash: Optional[str]) -> Dict[str, Any]:
    """Initialize the coordinator-only event signer and emit one restart event."""
    global event_signer_initialization

    normalized_tip = str(prev_log_tip_hash or "").strip().lower() or None
    if normalized_tip is not None and (
        len(normalized_tip) != 64
        or any(char not in "0123456789abcdef" for char in normalized_tip)
    ):
        raise ValueError("previous transparency log tip must be a SHA-256 hex digest")

    with event_signer_initialization_lock:
        if event_signer_initialization is not None:
            initialized_tip = event_signer_initialization["previous_log_tip_hash"]
            if normalized_tip != initialized_tip:
                raise RuntimeError(
                    "transparency signer is already initialized with another log tip"
                )
            return dict(event_signer_initialization["response"])

        signer = _event_signer_module()
        if signer.is_keypair_initialized():
            raise RuntimeError("transparency signer state was initialized out of order")

        signer.initialize_enclave_keypair()
        code_hash = compute_code_hash()
        if code_hash == "0" * 64:
            raise RuntimeError("gateway code hash is unavailable")
        signer.set_cached_code_hash(code_hash)
        signer.generate_attestation_document(code_hash)

        restart_log_entry = signer.publish_restart_event(normalized_tip)
        restart_payload = restart_log_entry["signed_event"]["payload"]
        payload_hash = hashlib.sha256(
            json.dumps(
                restart_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        buffer_result = append_event(
            {
                "event_type": restart_log_entry["signed_event"]["event_type"],
                "event_hash": restart_log_entry["event_hash"],
                "payload_hash": payload_hash,
                "signed_log_entry": restart_log_entry,
            }
        )
        response = {
            "identity": _event_signing_identity(),
            "restart_log_entry": restart_log_entry,
            "payload_hash": payload_hash,
            "buffer": buffer_result,
        }
        event_signer_initialization = {
            "previous_log_tip_hash": normalized_tip,
            "response": response,
        }
        return dict(response)


def sign_transparency_event(
    *, event_type: str, payload: Dict[str, Any], payload_hash: str
) -> Dict[str, Any]:
    """Sign and buffer a transparency event without exposing signing material."""
    if event_signer_initialization is None:
        raise RuntimeError("transparency event signer is not initialized")
    normalized_type = str(event_type or "").strip()
    if not normalized_type:
        raise ValueError("event_type is required")
    if not isinstance(payload, dict):
        raise ValueError("event payload must be an object")
    expected_payload_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if str(payload_hash or "").strip().lower() != expected_payload_hash:
        raise ValueError("transparency payload hash mismatch")

    signer = _event_signer_module()
    log_entry = signer.sign_event(normalized_type, payload)
    buffer_result = append_event(
        {
            "event_type": normalized_type,
            "event_hash": log_entry["event_hash"],
            "payload_hash": expected_payload_hash,
            "signed_log_entry": log_entry,
        }
    )
    return {
        "log_entry": log_entry,
        "payload_hash": expected_payload_hash,
        "buffer": buffer_result,
    }


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
    """Reject the legacy unauthenticated all-buffer deletion operation."""

    return {
        "status": "rejected",
        "reason": "checkpoint_acknowledgement_required",
        "cleared_count": 0,
    }


def acknowledge_checkpoint(
    *,
    checkpoint_number: Any,
    merkle_root: Any,
    sequence_range: Any,
) -> Dict[str, Any]:
    """Commit one confirmed checkpoint and remove only its exact buffer prefix."""

    global checkpoint_count, checkpoint_start_time
    global pending_checkpoint, prev_checkpoint_root

    with checkpoint_state_lock:
        if not isinstance(pending_checkpoint, dict):
            raise ValueError("no pending checkpoint exists")
        header = pending_checkpoint.get("header")
        events = pending_checkpoint.get("events")
        if not isinstance(header, dict) or not isinstance(events, list):
            raise ValueError("pending checkpoint is invalid")
        expected_range = header.get("sequence_range")
        normalized_range = (
            dict(sequence_range) if isinstance(sequence_range, dict) else None
        )
        if (
            int(checkpoint_number) != int(header["checkpoint_number"])
            or str(merkle_root or "") != str(header["merkle_root"])
            or normalized_range != expected_range
        ):
            raise ValueError("checkpoint acknowledgement differs")
        with event_buffer_lock:
            event_count = len(events)
            if len(event_buffer) < event_count:
                raise ValueError("checkpoint buffer prefix is missing")
            if event_buffer[:event_count] != events:
                raise ValueError("checkpoint buffer prefix differs")
            del event_buffer[:event_count]
        prev_checkpoint_root = bytes.fromhex(str(header["merkle_root"]))
        checkpoint_count = int(header["checkpoint_number"]) + 1
        checkpoint_start_time = datetime.utcnow()
        pending_checkpoint = None

    print(
        f"[TEE] ✅ Checkpoint #{checkpoint_count - 1} acknowledged: "
        f"{event_count} events",
        flush=True,
    )
    return {
        "status": "acknowledged",
        "checkpoint_number": checkpoint_count - 1,
        "removed_count": event_count,
        "sequence_range": normalized_range,
        "remaining_count": get_buffer_size(),
        "next_checkpoint_at": checkpoint_start_time.isoformat(),
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
    7. Parent EC2 acknowledges the exact checkpoint after confirmation
    
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
    global pending_checkpoint
    
    with checkpoint_state_lock:
        if pending_checkpoint is not None:
            return deepcopy(pending_checkpoint)

        now = datetime.utcnow()
        with event_buffer_lock:
            events = deepcopy(event_buffer)

        if not events:
            return {
                "status": "empty",
                "message": "No events to checkpoint",
                "next_checkpoint_at": (
                    now.replace(minute=0, second=0, microsecond=0)
                ).isoformat()
                + "Z",
            }

        print(
            f"[TEE] 📦 Building checkpoint #{checkpoint_count} "
            f"for {len(events)} events...",
            flush=True,
        )

        try:
            merkle_root, tree_levels = compute_merkle_tree(events)
            print(
                f"[TEE]    Merkle root: {merkle_root.hex()[:16]}...",
                flush=True,
            )
            code_hash = compute_code_hash()
            attestation_hash = get_cached_attestation_hash()
            checkpoint_header = {
                "checkpoint_version": 1,
                "checkpoint_number": checkpoint_count,
                "time_range": {
                    "start": checkpoint_start_time.isoformat() + "Z",
                    "end": now.isoformat() + "Z",
                },
                "event_count": len(events),
                "sequence_range": {
                    "first": events[0]["sequence"],
                    "last": events[-1]["sequence"],
                },
                "merkle_root": merkle_root.hex(),
                "prev_checkpoint_root": (
                    prev_checkpoint_root.hex()
                    if prev_checkpoint_root
                    else None
                ),
                "code_hash": code_hash,
                "attestation_hash": attestation_hash,
            }
            header_json = json.dumps(checkpoint_header, sort_keys=True)
            header_hash = hashlib.sha256(
                header_json.encode("utf-8")
            ).digest()
            checkpoint_signature = sign_data(header_hash)
            print(
                f"[TEE]    Signature: "
                f"{checkpoint_signature.hex()[:16]}...",
                flush=True,
            )
            tree_levels_hex = [
                [node.hex() for node in level]
                for level in tree_levels
            ]
            print(
                f"[TEE] ✅ Checkpoint #{checkpoint_count} built and pending",
                flush=True,
            )
            print(f"     Events: {len(events)}", flush=True)
            print(
                f"     Tree Depth: {len(tree_levels)} levels",
                flush=True,
            )
            print(
                "     Prev Root: "
                f"{prev_checkpoint_root.hex()[:16] if prev_checkpoint_root else 'None'}...",
                flush=True,
            )
            pending_checkpoint = {
                "status": "success",
                "header": checkpoint_header,
                "signature": checkpoint_signature.hex(),
                "events": events,
                "tree_levels": tree_levels_hex,
            }
            return deepcopy(pending_checkpoint)
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
    """Compute SHA256 hash of all attested gateway runtime code."""
    try:
        from pathlib import Path
        gateway_root = Path(__file__).parent.parent
        print(f"[TEE] 🔐 Computing code hash from gateway directory: {gateway_root}", flush=True)
        return compute_gateway_code_hash(gateway_root, log_prefix="[TEE]")
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


def get_attestation_document_with_pcrs(
    pcr0: str = None,
    pcr1: str = None,
    pcr2: str = None,
    user_data_fields: Dict[str, Any] = None,
) -> Dict[str, Any]:
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
    if user_data_fields:
        allowed_fields = {"purpose", "epoch_id", "job_id", "config_hash", "input_root"}
        unknown_fields = set(user_data_fields) - allowed_fields
        if unknown_fields:
            raise ValueError("Unsupported attestation user_data fields")
        user_data_dict.update(user_data_fields)
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

def get_provider_egress_proxy():
    global provider_egress_proxy
    with provider_egress_proxy_lock:
        if provider_egress_proxy is not None:
            return provider_egress_proxy
        from gateway.tee.egress_proxy import EnclaveEgressProxy, configured_proxy_ports

        local_port, forwarder_port = configured_proxy_ports()
        provider_egress_proxy = EnclaveEgressProxy(
            recv_exact=_recv_exact,
            local_port=local_port,
            forwarder_port=forwarder_port,
        )
        provider_egress_proxy.start()
        print(
            "[TEE] V2 provider egress relay initialized local_port=%s forwarder_port=%s"
            % (local_port, forwarder_port),
            flush=True,
        )
        return provider_egress_proxy


def get_v2_runtime_identity():
    global v2_runtime_identity
    with v2_runtime_identity_lock:
        if v2_runtime_identity is not None:
            return v2_runtime_identity
        from gateway.tee.rpc_authority import active_enclave_role
        from gateway.tee.runtime_identity_v2 import RuntimeIdentityV2

        physical_role = active_enclave_role()
        gateway_root = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        v2_runtime_identity = RuntimeIdentityV2(
            gateway_root=gateway_root,
            physical_role=physical_role,
            signing_pubkey_supplier=lambda: get_public_key_bytes().hex(),
            pcr0_supplier=lambda: str(pcr_measurements.get("PCR0") or ""),
        )
        return v2_runtime_identity


def handle_v2_runtime_rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    manager = get_v2_runtime_identity()
    if method == "v2_configure_runtime":
        if not isinstance(params, dict) or set(params) != {
            "schema_version",
            "configuration",
            "configuration_hash",
        }:
            raise ValueError("V2 runtime configuration fields are invalid")
        if params.get("schema_version") != "leadpoet.enclave_runtime_config.v2":
            raise ValueError("V2 runtime configuration schema is invalid")
        return {
            "result": manager.configure(
                configuration=params.get("configuration"),
                expected_config_hash=str(params.get("configuration_hash") or ""),
            )
        }
    if method == "v2_get_boot_identity":
        return {"result": manager.boot_identity()}
    if method == "v2_get_transport_certificate":
        return {
            "result": {
                "certificate_pem_b64": base64.b64encode(
                    manager.transport_certificate_pem()
                ).decode("ascii"),
                "status": manager.public_status(),
            }
        }
    if method == "v2_register_peer":
        if not isinstance(params, dict) or set(params) != {
            "boot_identity",
            "certificate_pem_b64",
        }:
            raise ValueError("V2 peer registration fields are invalid")
        peer_boot = params.get("boot_identity")
        if not isinstance(peer_boot, dict):
            raise ValueError("V2 peer boot identity is invalid")
        try:
            certificate_pem = base64.b64decode(
                str(params.get("certificate_pem_b64") or ""),
                validate=True,
            )
        except Exception as exc:
            raise ValueError("V2 peer certificate is invalid") from exc
        expectation = manager.peer_release_expectation(
            str(peer_boot.get("physical_role") or "")
        )
        registry = get_v2_peer_registry()
        return {
            "result": registry.register(
                boot_identity=peer_boot,
                certificate_pem=certificate_pem,
                expected_pcr0=expectation["pcr0"],
                expected_commit_sha=expectation["commit_sha"],
                expected_build_manifest_hash=expectation["build_manifest_hash"],
            )
        }
    if method == "v2_peer_status":
        return {
            "result": {
                "registered_roles": list(get_v2_peer_registry().registered_roles()),
            }
        }
    if method == "v2_start_tls_service":
        return {"result": start_v2_tls_service()}
    if method == "v2_call_peer_health":
        if not isinstance(params, dict) or set(params) != {"physical_role"}:
            raise ValueError("V2 peer health fields are invalid")
        from gateway.tee.inter_enclave_tls import AttestedTLSRPCClient
        from gateway.tee.rpc_authority import active_enclave_role
        import secrets

        client = AttestedTLSRPCClient(
            local_physical_role=active_enclave_role(),
            local_boot_identity=manager.boot_identity(),
            local_tls_identity=manager.tls_identity(),
            peer_registry=get_v2_peer_registry(),
        )
        return {
            "result": client.call(
                target_physical_role=str(params.get("physical_role") or ""),
                method="channel_health",
                params={},
                channel_id=secrets.token_hex(16),
            )
        }
    if method == "v2_provider_broker_health":
        health = get_v2_provider_broker().health()
        return {
            "result": {
                **health,
                "egress_proxy": get_provider_egress_proxy().status(),
            }
        }
    if method == "v2_provider_semantics_health":
        return {"result": get_v2_provider_semantics_authority().health()}
    if method == "v2_get_kms_recipient":
        if not isinstance(params, dict) or set(params) != {"credential_slot"}:
            raise ValueError("V2 KMS recipient fields are invalid")
        return {
            "result": get_v2_kms_recipient().recipient_request(
                str(params.get("credential_slot") or "")
            )
        }
    if method == "v2_get_source_add_ingress_recipient":
        if not isinstance(params, dict) or set(params) != {
            "miner_hotkey",
            "adapter_ref",
            "credential_ref",
        }:
            raise ValueError("V2 SOURCE_ADD ingress recipient fields are invalid")
        return {
            "result": get_v2_kms_recipient().source_add_ingress_recipient_request(
                miner_hotkey=str(params.get("miner_hotkey") or ""),
                adapter_ref=str(params.get("adapter_ref") or ""),
                credential_ref=str(params.get("credential_ref") or ""),
            )
        }
    if method == "v2_get_openrouter_ingress_recipient":
        if not isinstance(params, dict) or set(params) != {
            "miner_hotkey",
            "credential_kind",
        }:
            raise ValueError("V2 OpenRouter ingress recipient fields are invalid")
        return {
            "result": get_v2_kms_recipient().openrouter_ingress_recipient_request(
                miner_hotkey=str(params.get("miner_hotkey") or ""),
                credential_kind=str(params.get("credential_kind") or ""),
            )
        }
    if method == "v2_seal_source_add_ingress_credential":
        if not isinstance(params, dict) or set(params) != {
            "request_id",
            "ciphertext_b64",
        }:
            raise ValueError("V2 SOURCE_ADD ingress ciphertext fields are invalid")
        from gateway.tee.source_add_credential_ingress_v2 import (
            seal_source_add_ingress_credential_v2,
        )

        request_id = str(params.get("request_id") or "").lower()
        ciphertext_b64 = str(params.get("ciphertext_b64") or "")
        from leadpoet_canonical.attested_v2 import sha256_bytes

        try:
            ciphertext_hash = sha256_bytes(
                base64.b64decode(ciphertext_b64, validate=True)
            )
        except Exception as exc:
            raise ValueError("V2 SOURCE_ADD ingress ciphertext is invalid") from exc
        cache_key = ("source_add", request_id)
        with v2_ingress_seal_cache_lock:
            existing = v2_ingress_seal_cache.get(cache_key)
            if existing is not None:
                if existing["ciphertext_hash"] != ciphertext_hash:
                    raise ValueError("V2 SOURCE_ADD ingress ciphertext changed")
                envelope = dict(existing["credential_envelope"])
            else:
                lease = get_v2_kms_recipient().unwrap_source_add_ingress_credential(
                    request_id=request_id,
                    ciphertext_b64=ciphertext_b64,
                )
                envelope = seal_source_add_ingress_credential_v2(
                    lease,
                    vault=get_v2_artifact_vault(),
                )
                v2_ingress_seal_cache[cache_key] = {
                    "ciphertext_hash": ciphertext_hash,
                    "credential_envelope": dict(envelope),
                }
        return {"result": {"credential_envelope": envelope}}
    if method == "v2_seal_openrouter_ingress_credential":
        if not isinstance(params, dict) or set(params) != {
            "request_id",
            "ciphertext_b64",
        }:
            raise ValueError("V2 OpenRouter ingress ciphertext fields are invalid")
        from gateway.tee.openrouter_credential_v2 import (
            seal_openrouter_ingress_credential_v2,
        )

        request_id = str(params.get("request_id") or "").lower()
        ciphertext_b64 = str(params.get("ciphertext_b64") or "")
        from leadpoet_canonical.attested_v2 import sha256_bytes

        try:
            ciphertext_hash = sha256_bytes(
                base64.b64decode(ciphertext_b64, validate=True)
            )
        except Exception as exc:
            raise ValueError("V2 OpenRouter ingress ciphertext is invalid") from exc
        cache_key = ("openrouter", request_id)
        with v2_ingress_seal_cache_lock:
            existing = v2_ingress_seal_cache.get(cache_key)
            if existing is not None:
                if existing["ciphertext_hash"] != ciphertext_hash:
                    raise ValueError("V2 OpenRouter ingress ciphertext changed")
                envelope = dict(existing["credential_envelope"])
            else:
                lease = get_v2_kms_recipient().unwrap_openrouter_ingress_credential(
                    request_id=request_id,
                    ciphertext_b64=ciphertext_b64,
                )
                envelope = seal_openrouter_ingress_credential_v2(
                    lease,
                    vault=get_v2_artifact_vault(),
                )
                v2_ingress_seal_cache[cache_key] = {
                    "ciphertext_hash": ciphertext_hash,
                    "credential_envelope": dict(envelope),
                }
        return {"result": {"credential_envelope": envelope}}
    if method == "v2_provision_encrypted_secret":
        if not isinstance(params, dict) or set(params) != {
            "credential_slot",
            "ciphertext_for_recipient_b64",
        }:
            raise ValueError("V2 encrypted credential fields are invalid")
        slot = str(params.get("credential_slot") or "")
        ciphertext = str(params.get("ciphertext_for_recipient_b64") or "")
        from gateway.tee.artifact_vault_v2 import ARTIFACT_MASTER_KEY_SLOT

        if slot == ARTIFACT_MASTER_KEY_SLOT:
            return {"result": provision_v2_artifact_master_key(ciphertext)}
        if v2_artifact_vault is None:
            raise RuntimeError(
                "V2 artifact master key must be provisioned before provider credentials"
            )
        credential = get_v2_kms_recipient().unwrap_credential(
            slot=slot,
            ciphertext_for_recipient_b64=ciphertext,
        )
        return {
            "result": get_v2_provider_broker().provision_credential(
                slot=slot,
                credential=credential,
            )
        }
    if method == "v2_get_job_kms_recipient":
        if not isinstance(params, dict) or set(params) != {
            "job_id",
            "credential_slot",
            "credential_value_hash",
            "key_ref_hash",
        }:
            raise ValueError("V2 job KMS recipient fields are invalid")
        return {
            "result": get_v2_kms_recipient().job_recipient_request(
                job_id=str(params.get("job_id") or ""),
                slot=str(params.get("credential_slot") or ""),
                credential_value_hash_expected=str(
                    params.get("credential_value_hash") or ""
                ),
                key_ref_hash=str(params.get("key_ref_hash") or ""),
            )
        }
    if method == "v2_provision_job_encrypted_secret":
        if not isinstance(params, dict) or set(params) != {
            "request_id",
            "ciphertext_for_recipient_b64",
        }:
            raise ValueError("V2 encrypted job credential fields are invalid")
        lease = get_v2_kms_recipient().unwrap_job_credential(
            request_id=str(params.get("request_id") or ""),
            ciphertext_for_recipient_b64=str(
                params.get("ciphertext_for_recipient_b64") or ""
            ),
        )
        return {
            "result": get_v2_provider_broker().provision_job_credential(
                job_id=lease["job_id"],
                slot=lease["credential_slot"],
                credential=lease["credential"],
                credential_value_hash_expected=lease[
                    "credential_value_hash"
                ],
            )
        }
    if method == "v2_provision_job_sealed_source_add_secret":
        if not isinstance(params, dict) or set(params) != {"envelope"}:
            raise ValueError("V2 sealed SOURCE_ADD job credential fields are invalid")
        from gateway.tee.source_add_credential_ingress_v2 import (
            unseal_source_add_job_credential_v2,
        )

        lease = unseal_source_add_job_credential_v2(
            params.get("envelope") or {},
            vault=get_v2_artifact_vault(),
        )
        return {
            "result": get_v2_provider_broker().provision_job_credential(
                job_id=lease["job_id"],
                slot=lease["credential_slot"],
                credential=lease["credential"],
                credential_value_hash_expected=lease["credential_value_hash"],
            )
        }
    if method == "v2_provision_job_sealed_openrouter_secret":
        if not isinstance(params, dict) or set(params) != {"envelope"}:
            raise ValueError(
                "V2 sealed OpenRouter job credential fields are invalid"
            )
        from gateway.tee.openrouter_credential_v2 import (
            unseal_openrouter_job_credential_v2,
        )

        lease = unseal_openrouter_job_credential_v2(
            params.get("envelope") or {},
            vault=get_v2_artifact_vault(),
        )
        return {
            "result": get_v2_provider_broker().provision_job_credential(
                job_id=lease["job_id"],
                slot=lease["credential_slot"],
                credential=lease["credential"],
                credential_value_hash_expected=lease["credential_value_hash"],
            )
        }
    if method == "v2_release_job_credentials":
        if not isinstance(params, dict) or set(params) != {"job_id"}:
            raise ValueError("V2 job credential release fields are invalid")
        return {
            "result": get_v2_provider_broker().release_job_credentials(
                str(params.get("job_id") or "")
            )
        }
    if method == "v2_list_encrypted_artifacts":
        if not isinstance(params, dict) or set(params) != {"job_id", "purpose"}:
            raise ValueError("V2 encrypted artifact query fields are invalid")
        return {
            "result": {
                "artifacts": list(
                    get_v2_artifact_vault().job_artifacts(
                        job_id=str(params.get("job_id") or ""),
                        purpose=str(params.get("purpose") or ""),
                    )
                )
            }
        }
    if method == "v2_export_encrypted_artifact":
        if not isinstance(params, dict) or set(params) != {"artifact_id"}:
            raise ValueError("V2 encrypted artifact export fields are invalid")
        return {
            "result": get_v2_artifact_vault().export_ciphertext(
                str(params.get("artifact_id") or "")
            )
        }
    if method == "v2_verify_encrypted_artifact_persistence":
        if not isinstance(params, dict) or set(params) != {
            "artifact_id",
            "attestation_job_id",
            "artifact_ref",
            "get_url",
            "head_url",
        }:
            raise ValueError("V2 encrypted artifact verification fields are invalid")
        return {
            "result": get_v2_artifact_persistence_verifier().verify(
                artifact_id=str(params.get("artifact_id") or ""),
                attestation_job_id=str(params.get("attestation_job_id") or ""),
                artifact_ref=str(params.get("artifact_ref") or ""),
                get_url=str(params.get("get_url") or ""),
                head_url=str(params.get("head_url") or ""),
            )
        }
    raise ValueError("Unknown V2 runtime method")


def get_v2_provider_broker():
    global v2_provider_broker
    with v2_provider_broker_lock:
        if v2_provider_broker is not None:
            return v2_provider_broker
        from gateway.tee.provider_broker_v2 import ProviderBrokerV2
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("provider broker is coordinator-only")
        configuration = get_v2_runtime_identity().runtime_configuration()[
            "configuration"
        ]
        credential_hashes = configuration.get("provider_ref_hashes")
        retry_hashes = configuration.get("provider_retry_policy_hashes")
        if not isinstance(credential_hashes, dict) or not isinstance(
            retry_hashes, dict
        ):
            raise RuntimeError("provider broker configuration is incomplete")
        from gateway.tee.provider_broker_v2 import provider_registry_hash

        if configuration.get("provider_registry_hash") != provider_registry_hash():
            raise RuntimeError("provider registry differs from measured routes")
        get_provider_egress_proxy()
        v2_provider_broker = ProviderBrokerV2(
            credential_ref_hashes=credential_hashes,
            retry_policy_hashes=retry_hashes,
            job_credential_slot_ref_hashes=configuration.get(
                "job_lease_slot_ref_hashes"
            ),
            artifact_sink=get_v2_artifact_vault().seal,
        )
        return v2_provider_broker


def get_v2_provider_evidence_authority():
    global v2_provider_evidence_authority
    with v2_provider_evidence_authority_lock:
        if v2_provider_evidence_authority is not None:
            return v2_provider_evidence_authority
        from gateway.tee.provider_evidence_v2 import ProviderEvidenceAuthorityV2
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("provider evidence authority is coordinator-only")
        broker = get_v2_provider_broker()
        v2_provider_evidence_authority = ProviderEvidenceAuthorityV2(
            broker=broker,
            boot_identity_supplier=get_v2_runtime_identity().boot_identity,
            sign_digest=sign_data,
            cache_store=get_v2_provider_cache_store(),
        )
        return v2_provider_evidence_authority


def _verify_v2_provider_cache_source_boot(identity):
    from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro

    if (
        str(identity.get("role") or "") != "gateway_coordinator"
        or str(identity.get("physical_role") or "") != "gateway_coordinator"
    ):
        raise RuntimeError("provider cache source is not a coordinator boot")
    # Historical same-day rows can come from a prior KMS-approved release. The
    # persistent artifact key authenticates the encrypted payload; Nitro
    # verification additionally proves the exact source boot claim and key.
    return verify_boot_identity_nitro(
        identity,
        expected_pcr0=str(identity.get("pcr0") or ""),
    )


def get_v2_provider_cache_store():
    global v2_provider_cache_store
    with v2_provider_cache_store_lock:
        if v2_provider_cache_store is not None:
            return v2_provider_cache_store
        from gateway.tee.provider_evidence_cache_store_v2 import (
            ProviderEvidenceCacheStoreV2,
        )
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("provider evidence cache is coordinator-only")
        v2_provider_cache_store = ProviderEvidenceCacheStoreV2(
            broker=get_v2_provider_broker(),
            vault=get_v2_artifact_vault(),
            source_boot_verifier=_verify_v2_provider_cache_source_boot,
        )
        return v2_provider_cache_store


def get_v2_provider_outcome_store():
    global v2_provider_outcome_store
    with v2_provider_outcome_store_lock:
        if v2_provider_outcome_store is not None:
            return v2_provider_outcome_store
        from gateway.tee.provider_outcome_store_v2 import ProviderOutcomeStoreV2
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("provider outcome store is coordinator-only")
        v2_provider_outcome_store = ProviderOutcomeStoreV2(
            broker=get_v2_provider_broker(),
            vault=get_v2_artifact_vault(),
        )
        return v2_provider_outcome_store


def get_v2_provider_semantics_authority():
    global v2_provider_semantics_authority
    with v2_provider_semantics_authority_lock:
        if v2_provider_semantics_authority is not None:
            return v2_provider_semantics_authority
        from gateway.tee.provider_semantics_v2 import ProviderSemanticsAuthorityV2
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("provider semantics authority is coordinator-only")
        v2_provider_semantics_authority = ProviderSemanticsAuthorityV2(
            broker=get_v2_provider_broker(),
            cache_store=get_v2_provider_cache_store(),
            artifact_sink=get_v2_artifact_vault().seal,
            boot_identity_supplier=get_v2_runtime_identity().boot_identity,
            sign_digest=sign_data,
            outcome_store=get_v2_provider_outcome_store(),
        )
        return v2_provider_semantics_authority


def get_v2_kms_recipient():
    global v2_kms_recipient
    with v2_kms_recipient_lock:
        if v2_kms_recipient is not None:
            return v2_kms_recipient
        from gateway.tee.kms_recipient_v2 import KMSRecipientV2
        from gateway.tee.rpc_authority import active_enclave_role
        from gateway.tee.runtime_identity_v2 import nsm_attestation_document

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("V2 KMS recipient is coordinator-only")
        runtime = get_v2_runtime_identity()
        configuration = runtime.runtime_configuration()["configuration"]
        expected = configuration.get("provider_ref_hashes")
        job_expected = configuration.get("job_lease_slot_ref_hashes")
        if not isinstance(expected, dict):
            raise RuntimeError("V2 provider credential references are unavailable")
        if not isinstance(job_expected, dict):
            raise RuntimeError("V2 job credential references are unavailable")
        artifact_key_ref = configuration.get("artifact_master_key_ref_hash")
        if not isinstance(artifact_key_ref, str):
            raise RuntimeError("V2 artifact master key reference is unavailable")
        v2_kms_recipient = KMSRecipientV2(
            boot_identity_supplier=runtime.boot_identity,
            expected_credential_ref_hashes=expected,
            expected_job_slot_ref_hashes=job_expected,
            expected_binary_ref_hashes={
                "artifact_master_key": artifact_key_ref,
            },
            attestation_supplier=nsm_attestation_document,
        )
        return v2_kms_recipient


def provision_v2_artifact_master_key(ciphertext_for_recipient_b64: str):
    global v2_artifact_vault
    from gateway.tee.artifact_vault_v2 import (
        ARTIFACT_MASTER_KEY_HASH_DOMAIN,
        ARTIFACT_MASTER_KEY_SLOT,
        EncryptedArtifactVaultV2,
    )

    with v2_artifact_vault_lock:
        if v2_artifact_vault is not None:
            raise RuntimeError("V2 artifact master key is already provisioned")
        runtime = get_v2_runtime_identity()
        policy = runtime.runtime_configuration()["configuration"].get(
            "encrypted_artifact_policy"
        )
        if not isinstance(policy, dict):
            raise RuntimeError("V2 encrypted artifact policy is unavailable")
        master_key = get_v2_kms_recipient().unwrap_binary_secret(
            slot=ARTIFACT_MASTER_KEY_SLOT,
            ciphertext_for_recipient_b64=ciphertext_for_recipient_b64,
            hash_domain=ARTIFACT_MASTER_KEY_HASH_DOMAIN,
        )
        v2_artifact_vault = EncryptedArtifactVaultV2(
            master_key=master_key,
            boot_identity_hash=runtime.boot_identity()["boot_identity_hash"],
            retention_days=int(policy["minimum_retention_days"]),
        )
        provider_slots = set(
            runtime.runtime_configuration()["configuration"][
                "provider_ref_hashes"
            ]
        )
        configured_slots = set(get_v2_kms_recipient().provisioned_slots())
        return {
            "status": "provisioning",
            "credential_slots": sorted(configured_slots),
            "missing_credential_slots": sorted(provider_slots - configured_slots),
        }


def get_v2_artifact_vault():
    if v2_artifact_vault is None:
        raise RuntimeError("V2 artifact vault is not provisioned")
    return v2_artifact_vault


def get_v2_inter_enclave_artifact_ingest():
    global v2_inter_enclave_artifact_ingest
    with v2_inter_enclave_artifact_ingest_lock:
        if v2_inter_enclave_artifact_ingest is None:
            from gateway.tee.inter_enclave_artifact_v2 import (
                InterEnclaveArtifactIngestV2,
            )
            from gateway.tee.rpc_authority import active_enclave_role

            if active_enclave_role() != "gateway_coordinator":
                raise RuntimeError("artifact ingestion is coordinator-only")
            v2_inter_enclave_artifact_ingest = InterEnclaveArtifactIngestV2(
                vault=get_v2_artifact_vault(),
            )
        return v2_inter_enclave_artifact_ingest


def get_v2_artifact_persistence_verifier():
    global v2_artifact_persistence_verifier
    with v2_artifact_persistence_verifier_lock:
        if v2_artifact_persistence_verifier is not None:
            return v2_artifact_persistence_verifier
        from gateway.tee.artifact_persistence_v2 import ArtifactPersistenceVerifierV2

        configuration = get_v2_runtime_identity().runtime_configuration()[
            "configuration"
        ]
        policy = configuration.get("encrypted_artifact_policy")
        if not isinstance(policy, dict):
            raise RuntimeError("V2 encrypted artifact policy is unavailable")
        v2_artifact_persistence_verifier = ArtifactPersistenceVerifierV2(
            vault=get_v2_artifact_vault(),
            policy=policy,
        )
        return v2_artifact_persistence_verifier


def get_v2_inter_enclave_client():
    global v2_inter_enclave_client
    with v2_inter_enclave_client_lock:
        if v2_inter_enclave_client is not None:
            return v2_inter_enclave_client
        from gateway.tee.inter_enclave_tls import AttestedTLSRPCClient
        from gateway.tee.rpc_authority import active_enclave_role

        manager = get_v2_runtime_identity()
        v2_inter_enclave_client = AttestedTLSRPCClient(
            local_physical_role=active_enclave_role(),
            local_boot_identity=manager.boot_identity(),
            local_tls_identity=manager.tls_identity(),
            peer_registry=get_v2_peer_registry(),
        )
        return v2_inter_enclave_client


def execute_v2_provider_request(request: Dict[str, Any]) -> Dict[str, Any]:
    import secrets

    return get_v2_inter_enclave_client().call(
        target_physical_role="gateway_coordinator",
        method="provider_execute",
        params=request,
        channel_id=secrets.token_hex(16),
    )


def execute_v2_provider_probe_request(request: Dict[str, Any]) -> Dict[str, Any]:
    import secrets

    return get_v2_inter_enclave_client().call(
        target_physical_role="gateway_coordinator",
        method="provider_probe_resolve",
        params=request,
        channel_id=secrets.token_hex(16),
    )


def seal_v2_inter_enclave_artifact(
    *,
    plaintext: bytes,
    job_id: str,
    purpose: str,
    artifact_kind: str,
) -> Dict[str, Any]:
    from gateway.tee.inter_enclave_artifact_v2 import (
        seal_artifact_over_attested_tls_v2,
    )

    return seal_artifact_over_attested_tls_v2(
        client=get_v2_inter_enclave_client(),
        plaintext=plaintext,
        job_id=job_id,
        purpose=purpose,
        artifact_kind=artifact_kind,
    )


def get_v2_scoring_job_manager():
    global v2_scoring_job_manager
    with v2_scoring_job_manager_lock:
        if v2_scoring_job_manager is not None:
            return v2_scoring_job_manager
        from gateway.tee.rpc_authority import SCORING_ROLES, active_enclave_role

        physical_role = active_enclave_role()
        if physical_role not in SCORING_ROLES:
            raise RuntimeError("V2 scoring manager is scoring-role only")
        runtime = get_v2_runtime_identity()
        runtime.apply_research_lab_behavior_environment()

        from gateway.tee.execution_job_manager_v2 import ExecutionJobManagerV2
        from gateway.tee.model_sandbox_v2 import (
            RunscModelSandboxV2,
            RunscSandboxConfigV2,
        )
        from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
        from gateway.tee.scoring_executor_v2 import (
            SCORING_OPERATIONS_V2,
            ScoringExecutorV2,
        )

        configuration = runtime.runtime_configuration()["configuration"]
        retry_hashes = configuration.get("provider_retry_policy_hashes")
        if not isinstance(retry_hashes, dict):
            raise RuntimeError("V2 scoring retry policy configuration is missing")
        worker_count = int(configuration.get("execution_worker_count") or 0)
        configured_worker_count = int(
            configuration.get("configured_worker_count") or 0
        )
        if not 1 <= worker_count <= 10:
            raise RuntimeError("V2 scoring execution worker count is invalid")
        if not 1 <= configured_worker_count <= 500:
            raise RuntimeError("V2 scoring configured worker count is invalid")
        sandbox_transport = BrokeredProviderTransportV2(
            execute_v2_provider_request
        )
        executor = ScoringExecutorV2(
            provider_execute=execute_v2_provider_request,
            retry_policy_hashes=retry_hashes,
            config_supplier=runtime.research_lab_config,
            execution_config=configuration["research_lab_execution_config"],
            model_sandbox=RunscModelSandboxV2(
                config=RunscSandboxConfigV2.from_measured_runtime(),
                transport=sandbox_transport,
            ),
            artifact_seal=seal_v2_inter_enclave_artifact,
        )
        v2_scoring_job_manager = ExecutionJobManagerV2(
            boot_identity_supplier=runtime.boot_identity,
            sign_digest=sign_data,
            operations=SCORING_OPERATIONS_V2,
            executor=executor,
            worker_count=worker_count,
            configured_worker_count=configured_worker_count,
        )
        return v2_scoring_job_manager


def get_v2_coordinator_job_manager():
    global v2_coordinator_job_manager
    with v2_coordinator_job_manager_lock:
        if v2_coordinator_job_manager is not None:
            return v2_coordinator_job_manager
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_coordinator":
            raise RuntimeError("V2 coordinator manager is coordinator-only")
        runtime = get_v2_runtime_identity()
        runtime.apply_research_lab_behavior_environment()

        from gateway.tee.coordinator_executor_v2 import (
            COORDINATOR_OPERATIONS_V2,
            CoordinatorExecutorV2,
            coordinator_failed_parent_graph_policy_v2,
        )
        from gateway.tee.execution_job_manager_v2 import ExecutionJobManagerV2
        from gateway.tee.openrouter_credential_v2 import (
            OpenRouterRegistrationAuthorityV2,
        )
        from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
        configuration = runtime.runtime_configuration()["configuration"]
        retry_hashes = configuration.get("provider_retry_policy_hashes")
        if not isinstance(retry_hashes, dict) or not isinstance(
            retry_hashes.get("supabase"), str
        ):
            raise RuntimeError("V2 Supabase retry policy is unavailable")
        from gateway.tee.coordinator_weight_source_v2 import (
            CoordinatorWeightSourceV2,
        )
        from gateway.tee.coordinator_allocation_source_v2 import (
            CoordinatorAllocationSourceV2,
        )
        from gateway.tee.coordinator_chain_source_v2 import (
            CoordinatorChainSourceV2,
        )
        from gateway.tee.coordinator_legacy_settlement_v2 import (
            CoordinatorLegacySettlementSourceV2,
        )
        from gateway.tee.coordinator_source_add_v2 import (
            CoordinatorSourceAddFunctionalProbeV2,
            CoordinatorSourceAddProvenanceV2,
        )
        from gateway.tee.coordinator_reward_source_v2 import (
            CoordinatorRewardSourceV2,
        )
        from gateway.tee.coordinator_active_model_source_v2 import (
            CoordinatorActiveModelSourceV2,
        )
        from gateway.tee.qualification_admission_v2 import (
            CoordinatorQualificationAdmissionV2,
        )
        from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
        source_reader = SupabaseSourceReaderV2(
            execute_provider=get_v2_provider_broker().execute,
            retry_policy_hash=retry_hashes["supabase"],
        )
        weight_source = CoordinatorWeightSourceV2(source_reader)
        chain_source = CoordinatorChainSourceV2(
            execute_provider=get_v2_provider_broker().execute,
            retry_policy_hashes=retry_hashes,
            epoch_authority=configuration[
                "research_lab_execution_config"
            ]["epoch_authority"],
        )
        legacy_settlement_source = CoordinatorLegacySettlementSourceV2(
            reader=source_reader,
            chain_source=chain_source,
            execute_provider=get_v2_provider_broker().execute,
            retry_policy_hash=retry_hashes["arweave"],
        )
        allocation_source = CoordinatorAllocationSourceV2(
            reader=source_reader,
            chain_source=chain_source,
            config_supplier=runtime.research_lab_config,
            network_supplier=lambda: str(
                configuration["research_lab_execution_config"]["deployment"][
                    "network"
                ]
            ),
        )
        qualification_admission = CoordinatorQualificationAdmissionV2(
            source_reader
        )
        source_add_provenance = CoordinatorSourceAddProvenanceV2(
            execute_provider=get_v2_provider_broker().execute,
            retry_policy_hash=retry_hashes["scrapingdog"],
            wayback_retry_policy_hash=retry_hashes["wayback"],
        )
        source_add_functional_probe = CoordinatorSourceAddFunctionalProbeV2(
            reader=source_reader,
            execute_provider=get_v2_provider_broker().execute,
        )
        reward_source = CoordinatorRewardSourceV2(
            reader=source_reader,
            chain_source=chain_source,
            config_supplier=runtime.research_lab_config,
        )
        active_model_source = CoordinatorActiveModelSourceV2(
            reader=source_reader,
            config_supplier=runtime.research_lab_config,
        )
        openrouter_registration = OpenRouterRegistrationAuthorityV2(
            broker=get_v2_provider_broker(),
            transport=BrokeredProviderTransportV2(
                get_v2_provider_broker().execute
            ),
            retry_policy_hashes=retry_hashes,
            vault=get_v2_artifact_vault(),
        )
        v2_coordinator_job_manager = ExecutionJobManagerV2(
            boot_identity_supplier=runtime.boot_identity,
            sign_digest=sign_data,
            operations=COORDINATOR_OPERATIONS_V2,
            executor=CoordinatorExecutorV2(
                artifact_evidence_supplier=lambda artifact_ids, _context: (
                    get_v2_artifact_vault().persistence_evidence(artifact_id)
                    for artifact_id in artifact_ids
                ),
                weight_source_resolver=lambda payload, context: weight_source.resolve(
                    payload=payload,
                    context=context,
                ),
                qualification_admission_resolver=lambda payload, context: (
                    qualification_admission.resolve(
                        payload=payload,
                        context=context,
                    )
                ),
                allocation_source_resolver=lambda payload, context: (
                    allocation_source.resolve(payload=payload, context=context)
                ),
                source_add_provenance_resolver=lambda payload, context: (
                    source_add_provenance.resolve(
                        payload=payload,
                        context=context,
                    )
                ),
                source_add_functional_probe_resolver=lambda payload, context: (
                    source_add_functional_probe.resolve(
                        payload=payload,
                        context=context,
                    )
                ),
                reward_source_resolver=lambda payload, context: (
                    reward_source.resolve(payload=payload, context=context)
                ),
                legacy_settlement_source_resolver=lambda payload, context: (
                    legacy_settlement_source.resolve(
                        payload=payload,
                        context=context,
                    )
                ),
                legacy_allocation_classification_resolver=(
                    lambda payload, context: (
                        legacy_settlement_source.resolve_classification(
                            payload=payload,
                            context=context,
                        )
                    )
                ),
                source_add_catalog_resolver=lambda payload, context: (
                    reward_source.catalog_snapshot(payload=payload, context=context)
                ),
                provider_outcome_supplier=(
                    get_v2_provider_semantics_authority().provider_outcome_snapshot_evidence
                ),
                openrouter_registration_resolver=(
                    openrouter_registration.execute
                ),
                openrouter_preflight_resolver=(
                    openrouter_registration.preflight
                ),
                active_private_model_resolver=lambda payload, context: (
                    active_model_source.resolve(payload=payload, context=context)
                ),
                config_supplier=runtime.research_lab_config,
            ),
            worker_count=1,
            configured_worker_count=0,
            failed_parent_graph_policy=coordinator_failed_parent_graph_policy_v2,
        )
        return v2_coordinator_job_manager


def get_v2_autoresearch_job_manager():
    global v2_autoresearch_job_manager
    with v2_autoresearch_job_manager_lock:
        if v2_autoresearch_job_manager is not None:
            return v2_autoresearch_job_manager
        from gateway.tee.rpc_authority import active_enclave_role

        if active_enclave_role() != "gateway_autoresearch":
            raise RuntimeError("V2 autoresearch manager is autoresearch-role only")
        runtime = get_v2_runtime_identity()
        runtime.apply_research_lab_behavior_environment()

        from gateway.tee.autoresearch_executor_v2 import (
            AUTORESEARCH_HOST_OPERATIONS_V2,
            AUTORESEARCH_OPERATIONS_V2,
            AutoresearchExecutorV2,
        )
        from gateway.tee.execution_job_manager_v2 import ExecutionJobManagerV2
        from gateway.tee.host_operation_channel_v2 import HostOperationChannelV2
        configuration = runtime.runtime_configuration()["configuration"]
        retry_hashes = configuration.get("provider_retry_policy_hashes")
        if not isinstance(retry_hashes, dict):
            raise RuntimeError("V2 autoresearch retry policy configuration is missing")
        worker_count = int(configuration.get("execution_worker_count") or 0)
        configured_worker_count = int(
            configuration.get("configured_worker_count") or 0
        )
        if (
            worker_count != configured_worker_count
            or not 1 <= worker_count <= 500
        ):
            raise RuntimeError("V2 autoresearch execution worker count is invalid")

        def channel_factory(job_id: str, purpose: str):
            return HostOperationChannelV2(
                job_id=job_id,
                purpose=purpose,
                boot_identity=runtime.boot_identity(),
                sign_digest=sign_data,
                allowed_operations=AUTORESEARCH_HOST_OPERATIONS_V2,
            )

        def verify_scoring_boot(identity: Dict[str, Any]):
            from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro

            physical_role = str(identity.get("physical_role") or "")
            if physical_role != "gateway_scoring":
                raise RuntimeError("autoresearch dev-eval boot role is invalid")
            expectation = runtime.release_role_expectation(physical_role)
            for field in (
                "commit_sha",
                "pcr0",
                "build_manifest_hash",
                "dependency_lock_hash",
            ):
                if identity.get(field) != expectation[field]:
                    raise RuntimeError(
                        "autoresearch dev-eval boot differs at %s" % field
                    )
            return verify_boot_identity_nitro(
                identity,
                expected_pcr0=expectation["pcr0"],
            )

        def verify_coordinator_boot(identity: Dict[str, Any]):
            from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro

            if str(identity.get("physical_role") or "") != "gateway_coordinator":
                raise RuntimeError("autoresearch provider coordinator role is invalid")
            expectation = runtime.release_role_expectation("gateway_coordinator")
            for field in (
                "commit_sha",
                "pcr0",
                "build_manifest_hash",
                "dependency_lock_hash",
            ):
                if identity.get(field) != expectation[field]:
                    raise RuntimeError(
                        "autoresearch provider coordinator differs at %s" % field
                    )
            return verify_boot_identity_nitro(
                identity,
                expected_pcr0=expectation["pcr0"],
            )

        v2_autoresearch_job_manager = ExecutionJobManagerV2(
            boot_identity_supplier=runtime.boot_identity,
            sign_digest=sign_data,
            operations=AUTORESEARCH_OPERATIONS_V2,
            executor=AutoresearchExecutorV2(
                provider_execute=execute_v2_provider_request,
                retry_policy_hashes=retry_hashes,
                config_supplier=runtime.research_lab_config,
                scoring_graph_verifier=verify_scoring_boot,
                probe_execute=execute_v2_provider_probe_request,
                coordinator_boot_verifier=verify_coordinator_boot,
                artifact_seal=seal_v2_inter_enclave_artifact,
            ),
            worker_count=worker_count,
            configured_worker_count=configured_worker_count,
            host_operation_channel_factory=channel_factory,
        )
        return v2_autoresearch_job_manager


def _handle_v2_job_rpc(
    method: str,
    params: Dict[str, Any],
    *,
    prefix: str,
    manager: Any,
) -> Dict[str, Any]:
    action = method[len(prefix):]
    if action == "health":
        return manager.health()
    if action == "submit_job":
        return manager.submit(params.get("manifest"))
    if action == "put_chunk":
        return manager.put_chunk(
            job_id=params.get("job_id"),
            offset=params.get("offset"),
            data_b64=params.get("data_b64"),
            chunk_sha256=params.get("chunk_sha256"),
        )
    if action == "seal_job":
        return manager.seal(params.get("job_id"))
    if action == "get_status":
        return manager.status(params.get("job_id"))
    if action == "cancel_job":
        return manager.cancel(params.get("job_id"))
    if action == "get_result":
        return manager.result_chunk(
            job_id=params.get("job_id"),
            offset=params.get("offset", 0),
            max_bytes=params.get("max_bytes", 512 * 1024),
        )
    if action == "get_receipt":
        return manager.receipt(params.get("job_id"))
    if action == "get_receipts":
        return list(manager.receipts(params.get("job_id")))
    if action == "get_transitions":
        return list(manager.transitions(params.get("job_id")))
    if action == "get_transport_attempts":
        return list(manager.transport_attempts(params.get("job_id")))
    if action == "get_artifact_hashes":
        return list(manager.artifact_hashes(params.get("job_id")))
    if action == "next_host_operation":
        return manager.next_host_operation(
            job_id=params.get("job_id"),
            wait_ms=params.get("wait_ms", 0),
        )
    if action == "complete_host_operation":
        return manager.complete_host_operation(
            job_id=params.get("job_id"),
            request_hash=params.get("request_hash"),
            terminal_status=params.get("terminal_status"),
            response=params.get("response"),
            failure_code=params.get("failure_code"),
        )
    if action == "get_host_operations":
        return list(manager.host_operations(params.get("job_id")))
    if action == "get_external_receipt_graphs":
        return list(manager.external_receipt_graphs(params.get("job_id")))
    raise ValueError("unknown V2 execution method")


def handle_v2_execution_rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if method.startswith("scoring_v2_"):
            result = _handle_v2_job_rpc(
                method,
                params,
                prefix="scoring_v2_",
                manager=get_v2_scoring_job_manager(),
            )
        elif method.startswith("coordinator_v2_"):
            result = _handle_v2_job_rpc(
                method,
                params,
                prefix="coordinator_v2_",
                manager=get_v2_coordinator_job_manager(),
            )
        elif method.startswith("autoresearch_v2_"):
            result = _handle_v2_job_rpc(
                method,
                params,
                prefix="autoresearch_v2_",
                manager=get_v2_autoresearch_job_manager(),
            )
        else:
            raise ValueError("unknown V2 execution namespace")
        return {"result": result}
    except Exception as exc:
        print(
            "[TEE] V2 execution RPC rejected method=%s type=%s"
            % (method, type(exc).__name__),
            flush=True,
        )
        return {"status": "error", "error": str(exc)}


def get_v2_peer_registry():
    global v2_peer_registry
    with v2_peer_registry_lock:
        if v2_peer_registry is not None:
            return v2_peer_registry
        from gateway.tee.inter_enclave_tls import AttestedPeerRegistry
        from gateway.tee.rpc_authority import active_enclave_role

        manager = get_v2_runtime_identity()
        manager.boot_identity()
        v2_peer_registry = AttestedPeerRegistry(
            local_physical_role=active_enclave_role(),
        )
        return v2_peer_registry


def handle_inter_enclave_rpc(
    method: str,
    params: Dict[str, Any],
    peer: Dict[str, Any],
) -> Dict[str, Any]:
    from gateway.tee.rpc_authority import active_enclave_role

    if method == "channel_health":
        return {
            "status": "healthy",
            "local_role": active_enclave_role(),
            "peer_role": peer["physical_role"],
            "local_boot_identity_hash": get_v2_runtime_identity()
            .boot_identity()["boot_identity_hash"],
        }
    if method == "provider_execute":
        if active_enclave_role() != "gateway_coordinator":
            raise ValueError("provider execution is coordinator-only")
        if peer["physical_role"] not in {
            "gateway_scoring",
            "gateway_autoresearch",
        }:
            raise ValueError("provider caller role is not authorized")
        return get_v2_provider_semantics_authority().execute(params)
    if method == "provider_probe_resolve":
        if active_enclave_role() != "gateway_coordinator":
            raise ValueError("provider evidence resolution is coordinator-only")
        if peer["physical_role"] != "gateway_autoresearch":
            raise ValueError("provider evidence caller role is not authorized")
        return get_v2_provider_evidence_authority().resolve(params)
    if method in {
        "artifact_seal_begin",
        "artifact_seal_chunk",
        "artifact_seal_finish",
        "artifact_seal_cancel",
    }:
        if active_enclave_role() != "gateway_coordinator":
            raise ValueError("artifact ingestion is coordinator-only")
        ingest = get_v2_inter_enclave_artifact_ingest()
        action = {
            "artifact_seal_begin": ingest.begin,
            "artifact_seal_chunk": ingest.put_chunk,
            "artifact_seal_finish": ingest.finish,
            "artifact_seal_cancel": ingest.cancel,
        }[method]
        return action(params, peer=peer)
    raise ValueError("inter-enclave method is not authorized")


def start_v2_tls_service() -> Dict[str, Any]:
    global v2_tls_server, v2_tls_server_thread
    with v2_tls_server_lock:
        if v2_tls_server_thread is not None and v2_tls_server_thread.is_alive():
            return {
                "status": "running",
                "registered_roles": list(get_v2_peer_registry().registered_roles()),
            }
        from gateway.tee.inter_enclave_tls import AttestedTLSRPCServer
        from gateway.tee.rpc_authority import active_enclave_role

        manager = get_v2_runtime_identity()
        registry = get_v2_peer_registry()
        expected_roles = manager.expected_peer_roles()
        if tuple(registry.registered_roles()) != expected_roles:
            raise RuntimeError("V2 TLS service requires every configured attested peer")
        v2_tls_server = AttestedTLSRPCServer(
            local_physical_role=active_enclave_role(),
            local_boot_identity=manager.boot_identity(),
            local_tls_identity=manager.tls_identity(),
            peer_registry=registry,
            handler=handle_inter_enclave_rpc,
        )
        import threading

        v2_tls_server_thread = threading.Thread(
            target=v2_tls_server.serve_forever,
            name="gateway-v2-inter-enclave-tls",
            daemon=True,
        )
        v2_tls_server_thread.start()
        return {
            "status": "running",
            "registered_roles": list(registry.registered_roles()),
        }


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
    - sign_checkpoint: Sign checkpoint header
    - scoring_v2_*: Submit and retrieve authoritative V2 scoring jobs
    
    Args:
        method: RPC method name
        params: Method parameters dict
    
    Returns:
        Response dict with result or error
    """
    try:
        from gateway.tee.rpc_authority import active_enclave_role, rpc_method_allowed

        enclave_role = active_enclave_role()
        if not rpc_method_allowed(enclave_role, method):
            return {
                "error": "RPC method is not authorized for enclave role %s"
                % enclave_role
            }
        if method == "append_event":
            event = params.get("event")
            if not event:
                return {"error": "Missing 'event' parameter"}
            return {"result": append_event(event)}

        elif method == "initialize_event_signer":
            return {
                "result": initialize_event_signer(params.get("prev_log_tip_hash"))
            }

        elif method == "sign_transparency_event":
            return {
                "result": sign_transparency_event(
                    event_type=params.get("event_type"),
                    payload=params.get("payload"),
                    payload_hash=params.get("payload_hash"),
                )
            }

        elif method == "get_event_signing_identity":
            return {"result": _event_signing_identity()}
        
        elif method == "get_buffer":
            return {"result": get_buffer()}
        
        elif method == "clear_buffer":
            return {"result": clear_buffer()}

        elif method == "acknowledge_checkpoint":
            return {
                "result": acknowledge_checkpoint(
                    checkpoint_number=params.get("checkpoint_number"),
                    merkle_root=params.get("merkle_root"),
                    sequence_range=params.get("sequence_range"),
                )
            }
        
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

        elif method == "role_health":
            from gateway.tee.build_identity import load_identity
            from gateway.tee.topology import role_spec

            gateway_root = Path(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            identity = load_identity(
                gateway_root=gateway_root,
                expected_role=enclave_role,
            )
            v2_status = {"status": "unavailable"}
            try:
                v2_status = get_v2_runtime_identity().public_status()
            except Exception as exc:
                v2_status = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            return {
                "result": {
                    "status": "healthy",
                    "role": enclave_role,
                    "service_role": role_spec(enclave_role)["service_role"],
                    "commit_sha": identity["commit_sha"],
                    "build_identity_hash": identity["identity_hash"],
                    "execution_manifest_hash": identity["execution_manifest_hash"],
                    "dependency_lock_hash": identity["dependency_lock_hash"],
                    "topology_hash": identity["topology_hash"],
                    "public_key": get_public_key_bytes().hex(),
                    "pcr0": pcr_measurements.get("PCR0"),
                    "v2_runtime": v2_status,
                }
            }

        elif method.startswith("v2_"):
            return handle_v2_runtime_rpc(method, params)
        
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

        elif method.startswith(
            ("scoring_v2_", "coordinator_v2_", "autoresearch_v2_")
        ):
            return handle_v2_execution_rpc(method, params)

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

def _recv_exact(conn: Any, size: int) -> bytes:
    """Read an exact bounded frame segment or return the partial bytes."""

    output = bytearray()
    while len(output) < size:
        chunk = conn.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


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
            length_bytes = _recv_exact(conn, 4)
            
            if len(length_bytes) != 4:
                print(f"[TEE] ⚠️ Invalid request (no length prefix)", flush=True)
                conn.close()
                continue
            
            request_length = int.from_bytes(length_bytes, byteorder='big')
            if request_length < 2 or request_length > MAX_RPC_REQUEST_BYTES:
                print(f"[TEE] ⚠️ Request length outside limit: {request_length}", flush=True)
                conn.close()
                continue
            
            # Receive JSON data
            request_data = _recv_exact(conn, request_length)
            
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
