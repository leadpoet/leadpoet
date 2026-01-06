"""
Gateway TEE Service (Parent EC2 Side)
=====================================

This module provides TEE initialization and management for the gateway process
running on the parent EC2 (NOT inside the enclave).

SINGLE-PROCESS REQUIREMENT:
The gateway MUST run as a single process. Multiple workers would create
divergent hash-chains with different _PREV_EVENT_HASH values, breaking
auditability. This class enforces this at runtime.

INITIALIZATION SEQUENCE:
1. Enforce single-process requirement
2. Initialize enclave keypair
3. Compute and cache code hash
4. Generate attestation document
5. Fetch last log tip from Supabase
6. Emit ENCLAVE_RESTART event

This class should be instantiated once at gateway startup in main.py.
"""

import os
from typing import Optional, Dict, Any

import hashlib
from pathlib import Path

from gateway.tee.enclave_signer import (
    initialize_enclave_keypair,
    generate_attestation_document,
    sign_event,
    get_enclave_public_key_hex,
    publish_restart_event,
    set_cached_code_hash,
    get_signer_state,
    is_keypair_initialized,
)


def compute_code_hash() -> str:
    """
    Compute SHA256 hash of ALL critical gateway code.
    
    This creates a deterministic hash of all API endpoints, consensus logic,
    and core gateway functionality, proving the EXACT code running.
    
    Files included (automatically discovered):
    - gateway/main.py - FastAPI server
    - gateway/config.py - Constants
    - gateway/api/*.py - All API endpoints
    - gateway/tasks/*.py - Epoch lifecycle, hourly batching
    - gateway/utils/*.py - Consensus, logging, signatures, registry
    - gateway/models/*.py - Event structure definitions
    - gateway/tee/*.py - TEE service, enclave signer
    
    Files excluded:
    - gateway/secrets/ - Credentials
    - __pycache__, *.pyc - Python bytecode
    
    Returns:
        SHA256 hash (hex string, 64 chars)
    """
    try:
        # Get gateway root directory
        gateway_root = Path(__file__).parent.parent
        
        print(f"[GatewayTEE] 📋 Computing code hash from: {gateway_root}", flush=True)
        
        # Collect all .py files to hash
        files_to_hash = []
        
        # Include directories
        include_dirs = [
            gateway_root / "api",
            gateway_root / "tasks",
            gateway_root / "utils",
            gateway_root / "models",
            gateway_root / "tee",
            gateway_root / "middleware",
        ]
        
        # Root-level critical files
        files_to_hash.append(gateway_root / "main.py")
        files_to_hash.append(gateway_root / "config.py")
        
        # Collect all .py files from include_dirs
        for dir_path in include_dirs:
            if dir_path.exists():
                for py_file in sorted(dir_path.glob("**/*.py")):
                    if "__pycache__" in str(py_file):
                        continue
                    if py_file.name.endswith(".pyc"):
                        continue
                    if py_file.name.startswith("test_"):
                        continue
                    files_to_hash.append(py_file)
        
        # Sort for determinism
        files_to_hash = sorted(set(files_to_hash))
        
        # Hash all files deterministically
        hasher = hashlib.sha256()
        for file_path in files_to_hash:
            if file_path.exists():
                hasher.update(str(file_path.name).encode('utf-8'))
                hasher.update(file_path.read_bytes())
        
        code_hash = hasher.hexdigest()
        print(f"[GatewayTEE]    Hash: {code_hash[:32]}...{code_hash[-32:]}", flush=True)
        
        return code_hash
        
    except Exception as e:
        print(f"[GatewayTEE] ⚠️  Failed to compute code hash: {e}", flush=True)
        return "0" * 64


class GatewayTEEService:
    """
    Gateway TEE Service - manages enclave signing for the gateway process.
    
    This class handles:
    - Single-process enforcement (fail if multiple workers)
    - Enclave keypair initialization
    - Code hash computation and caching
    - Attestation document generation
    - ENCLAVE_RESTART event emission on boot
    
    Usage:
        # In gateway/main.py startup:
        tee_service = GatewayTEEService()
        # The constructor does all initialization
    """
    
    def __init__(self, skip_restart_event: bool = False):
        """
        Initialize the Gateway TEE Service.
        
        This constructor performs all initialization steps in sequence.
        It MUST be called once at gateway startup, before any other logging.
        
        Args:
            skip_restart_event: If True, skip emitting ENCLAVE_RESTART event
                               (only for testing purposes)
        
        Raises:
            RuntimeError: If single-process requirement is violated
            RuntimeError: If initialization fails
        """
        print("[GatewayTEE] 🔐 Initializing Gateway TEE Service...", flush=True)
        
        # Step 1: Enforce single-process requirement
        self._enforce_single_process()
        
        # Step 2: Initialize enclave keypair
        print("[GatewayTEE] 🔐 Initializing enclave signing keypair...", flush=True)
        self.enclave_pubkey = initialize_enclave_keypair()
        
        # Step 3: Compute code hash ONCE at boot - never recompute
        print("[GatewayTEE] 📋 Computing code hash...", flush=True)
        self.code_hash = compute_code_hash()
        set_cached_code_hash(self.code_hash)  # Cache for attestation endpoint
        
        # Step 4: Generate attestation document binding pubkey to code
        print("[GatewayTEE] 📜 Generating attestation document...", flush=True)
        self.attestation_doc = generate_attestation_document(self.code_hash)
        
        # Step 5 & 6: Emit restart event (unless skipped)
        if not skip_restart_event:
            print("[GatewayTEE] 🔄 Emitting enclave restart event...", flush=True)
            prev_tip = self._fetch_last_log_tip()
            self.restart_entry = publish_restart_event(prev_tip)
            self._store_restart_entry(self.restart_entry)
        else:
            self.restart_entry = None
            print("[GatewayTEE] ⚠️  Skipping restart event (testing mode)", flush=True)
        
        print(f"[GatewayTEE] ✅ TEE Service initialized", flush=True)
        print(f"[GatewayTEE]    Pubkey: {self.enclave_pubkey[:16]}...{self.enclave_pubkey[-16:]}", flush=True)
        print(f"[GatewayTEE]    Code hash: {self.code_hash[:16]}...{self.code_hash[-16:]}", flush=True)
    
    def _enforce_single_process(self):
        """
        Enforce single-process requirement at runtime.
        
        Multiple workers (uvicorn --workers > 1, WEB_CONCURRENCY > 1) create
        divergent hash-chains with different _PREV_EVENT_HASH values, breaking
        global auditability.
        
        Raises:
            RuntimeError: If WEB_CONCURRENCY > 1 or UVICORN_WORKERS > 1
        """
        web_concurrency = int(os.environ.get("WEB_CONCURRENCY", "1"))
        uvicorn_workers = int(os.environ.get("UVICORN_WORKERS", "1"))
        
        if web_concurrency > 1 or uvicorn_workers > 1:
            raise RuntimeError(
                "FATAL: Gateway MUST run as a single process. "
                f"WEB_CONCURRENCY={web_concurrency}, UVICORN_WORKERS={uvicorn_workers}. "
                "Multiple workers create divergent hash-chains that break auditability. "
                "Set WEB_CONCURRENCY=1 and UVICORN_WORKERS=1."
            )
        
        print(f"[GatewayTEE] ✓ Single-process requirement satisfied", flush=True)
    
    def _fetch_last_log_tip(self) -> Optional[str]:
        """
        Fetch the last known log entry hash from Supabase.
        
        This is used to link the new boot session to the previous one,
        maintaining hash-chain integrity across restarts.
        
        Returns:
            Last event_hash from transparency_log, or None if not found
        """
        try:
            from gateway.db.client import get_read_client
            read_client = get_read_client()
            
            # Query the most recent log entry
            result = read_client.table("transparency_log") \
                .select("payload") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if result.data:
                log_entry = result.data[0]["payload"]
                prev_hash = log_entry.get("event_hash")
                print(f"[GatewayTEE] ✓ Found previous log tip: {prev_hash[:16]}...", flush=True)
                return prev_hash
            else:
                print("[GatewayTEE] ⚠️  No previous log entries found (fresh start)", flush=True)
                return None
                
        except Exception as e:
            print(f"[GatewayTEE] ⚠️  Could not fetch last log tip: {e}", flush=True)
            return None
    
    def _store_restart_entry(self, restart_entry: Dict[str, Any]):
        """
        Store restart entry immediately to maintain chain integrity.
        
        This MUST be stored before any other events are logged.
        
        Args:
            restart_entry: The log_entry from publish_restart_event()
        """
        try:
            from gateway.db.client import get_write_client
            write_client = get_write_client()
            
            # Extract fields for transparency_log schema
            signed_event = restart_entry["signed_event"]
            
            # Insert with all required columns
            write_client.table("transparency_log").insert({
                "event_type": signed_event["event_type"],
                "payload": restart_entry,  # Store entire log_entry as payload
                "event_hash": restart_entry["event_hash"],
                "enclave_pubkey": restart_entry["enclave_pubkey"],
                "boot_id": signed_event["boot_id"],
                "monotonic_seq": signed_event["monotonic_seq"],
                "prev_event_hash": signed_event["prev_event_hash"],
                "created_at": signed_event["timestamp"],
            }).execute()
            
            print(f"[GatewayTEE] ✅ Restart event stored (hash: {restart_entry['event_hash'][:16]}...)", flush=True)
            
        except Exception as e:
            print(f"[GatewayTEE] ❌ Failed to store restart event: {e}", flush=True)
            # This is critical - log but don't raise to allow gateway to start
            # The restart event will be missing from the chain, which is detectable
    
    def get_pubkey(self) -> str:
        """Get the enclave public key (hex string)."""
        return self.enclave_pubkey
    
    def get_code_hash(self) -> str:
        """Get the computed code hash."""
        return self.code_hash
    
    def get_attestation(self) -> bytes:
        """Get the attestation document."""
        return self.attestation_doc
    
    def get_state(self) -> Dict[str, Any]:
        """Get current TEE service state for debugging/monitoring."""
        return {
            "pubkey": self.enclave_pubkey[:16] + "..." if self.enclave_pubkey else None,
            "code_hash": self.code_hash[:16] + "..." if self.code_hash else None,
            "has_attestation": bool(self.attestation_doc),
            "attestation_size": len(self.attestation_doc) if self.attestation_doc else 0,
            "signer_state": get_signer_state(),
        }


# Global singleton instance (initialized on first access)
_tee_service_instance: Optional[GatewayTEEService] = None


def get_tee_service() -> GatewayTEEService:
    """
    Get the global GatewayTEEService instance.
    
    Returns:
        The singleton GatewayTEEService instance
        
    Raises:
        RuntimeError: If TEE service has not been initialized
    """
    global _tee_service_instance
    if _tee_service_instance is None:
        raise RuntimeError(
            "GatewayTEEService not initialized. "
            "Call init_tee_service() at gateway startup."
        )
    return _tee_service_instance


def init_tee_service(skip_restart_event: bool = False) -> GatewayTEEService:
    """
    Initialize the global GatewayTEEService instance.
    
    This should be called once at gateway startup in main.py.
    
    Args:
        skip_restart_event: If True, skip emitting ENCLAVE_RESTART event
        
    Returns:
        The initialized GatewayTEEService instance
        
    Raises:
        RuntimeError: If already initialized or initialization fails
    """
    global _tee_service_instance
    if _tee_service_instance is not None:
        raise RuntimeError("GatewayTEEService already initialized")
    
    _tee_service_instance = GatewayTEEService(skip_restart_event=skip_restart_event)
    return _tee_service_instance


def is_tee_initialized() -> bool:
    """Check if the TEE service has been initialized."""
    return _tee_service_instance is not None


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_single_process_enforcement():
    """Test single-process enforcement."""
    import os
    
    # Save original values
    orig_web = os.environ.get("WEB_CONCURRENCY")
    orig_uvi = os.environ.get("UVICORN_WORKERS")
    
    try:
        # Test with valid values (should not raise)
        os.environ["WEB_CONCURRENCY"] = "1"
        os.environ["UVICORN_WORKERS"] = "1"
        
        service = GatewayTEEService.__new__(GatewayTEEService)
        service._enforce_single_process()  # Should succeed
        print("✅ Single process check passed (valid values)")
        
        # Test with invalid WEB_CONCURRENCY
        os.environ["WEB_CONCURRENCY"] = "2"
        try:
            service._enforce_single_process()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "FATAL" in str(e)
            print("✅ Single process check failed correctly (WEB_CONCURRENCY=2)")
        
        # Reset and test UVICORN_WORKERS
        os.environ["WEB_CONCURRENCY"] = "1"
        os.environ["UVICORN_WORKERS"] = "4"
        try:
            service._enforce_single_process()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "FATAL" in str(e)
            print("✅ Single process check failed correctly (UVICORN_WORKERS=4)")
    
    finally:
        # Restore original values
        if orig_web is not None:
            os.environ["WEB_CONCURRENCY"] = orig_web
        elif "WEB_CONCURRENCY" in os.environ:
            del os.environ["WEB_CONCURRENCY"]
        
        if orig_uvi is not None:
            os.environ["UVICORN_WORKERS"] = orig_uvi
        elif "UVICORN_WORKERS" in os.environ:
            del os.environ["UVICORN_WORKERS"]


def test_tee_service_initialization():
    """Test basic TEE service initialization (without DB operations)."""
    global _tee_service_instance
    
    # Reset singleton
    _tee_service_instance = None
    
    # Reset enclave_signer state
    from gateway.tee.enclave_signer import _reset_for_testing
    _reset_for_testing()
    
    # Initialize with skip_restart_event to avoid DB calls
    service = GatewayTEEService(skip_restart_event=True)
    
    # Verify initialization
    assert service.enclave_pubkey is not None
    assert len(service.enclave_pubkey) == 64  # Ed25519 pubkey is 32 bytes = 64 hex chars
    assert service.code_hash is not None
    assert len(service.code_hash) == 64  # SHA256 hash is 32 bytes = 64 hex chars
    
    print("✅ TEE service initialization test passed")
    
    # Reset for next test
    _tee_service_instance = None
    _reset_for_testing()


if __name__ == "__main__":
    print("Running gateway/tee/gateway_tee_service.py unit tests...\n")
    
    test_single_process_enforcement()
    test_tee_service_initialization()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

