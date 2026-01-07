"""
Validator TEE vsock Client
==========================

This module runs on the HOST (parent EC2) and communicates with the 
validator enclave via vsock.

Usage:
    from validator_tee.vsock_client import ValidatorEnclaveClient
    
    client = ValidatorEnclaveClient()
    pubkey = client.get_public_key()
    signature = client.sign_weights(weights_hash)
    attestation = client.get_attestation(epoch_id)
"""

import socket
import json
import os
import subprocess
from typing import Dict, Any, Optional

# vsock constants
AF_VSOCK = 40
PARENT_CID = 3
RPC_PORT = 5001  # Must match tee_service.py


def get_enclave_cid() -> Optional[int]:
    """
    Get the CID of the running validator enclave.
    
    Priority:
    1. ENCLAVE_CID environment variable (for Docker containers)
    2. nitro-cli describe-enclaves (for host)
    
    Returns:
        Enclave CID or None if not running
    """
    # Check environment variable first (for Docker containers)
    env_cid = os.environ.get("ENCLAVE_CID")
    if env_cid:
        try:
            cid = int(env_cid)
            print(f"[vsock] Using ENCLAVE_CID from environment: {cid}")
            return cid
        except ValueError:
            print(f"[vsock] Invalid ENCLAVE_CID: {env_cid}")
    
    # Fall back to nitro-cli (for host)
    try:
        result = subprocess.run(
            ["nitro-cli", "describe-enclaves"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None
        
        import json as json_mod
        enclaves = json_mod.loads(result.stdout)
        
        for enclave in enclaves:
            # Look for validator enclave (by name or just return first running one)
            if enclave.get("State") == "RUNNING":
                return enclave.get("EnclaveCID")
        
        return None
        
    except Exception as e:
        print(f"[vsock] Error getting enclave CID: {e}")
        return None


class ValidatorEnclaveClient:
    """
    Client for communicating with the validator TEE enclave.
    """
    
    def __init__(self, enclave_cid: Optional[int] = None):
        """
        Initialize the enclave client.
        
        Args:
            enclave_cid: Enclave CID (auto-detected if not provided)
        """
        self.enclave_cid = enclave_cid
        self._cached_pubkey: Optional[str] = None
        self._cached_code_hash: Optional[str] = None
    
    def _get_cid(self) -> int:
        """Get enclave CID, auto-detecting if needed."""
        if self.enclave_cid is not None:
            return self.enclave_cid
        
        cid = get_enclave_cid()
        if cid is None:
            raise RuntimeError("No running validator enclave found")
        
        self.enclave_cid = cid
        return cid
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to enclave via vsock.
        
        Args:
            request: Request dict with 'command' and parameters
            
        Returns:
            Response dict from enclave
        """
        cid = self._get_cid()
        
        # Create vsock socket
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(30)  # 30 second timeout
        
        try:
            # Connect to enclave
            sock.connect((cid, RPC_PORT))
            
            # Send request
            request_data = json.dumps(request).encode()
            sock.sendall(request_data)
            sock.shutdown(socket.SHUT_WR)
            
            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            
            response = json.loads(response_data.decode())
            
            if response.get("status") == "error":
                raise RuntimeError(f"Enclave error: {response.get('error')}")
            
            return response
            
        finally:
            sock.close()
    
    def get_public_key(self) -> str:
        """
        Get the enclave's public key.
        
        Returns:
            Public key as hex string
        """
        if self._cached_pubkey:
            return self._cached_pubkey
        
        response = self._send_request({"command": "get_public_key"})
        
        self._cached_pubkey = response["public_key"]
        self._cached_code_hash = response.get("code_hash")
        
        return self._cached_pubkey
    
    def get_code_hash(self) -> str:
        """
        Get the enclave's code hash.
        
        Returns:
            Code hash as hex string
        """
        if self._cached_code_hash:
            return self._cached_code_hash
        
        response = self._send_request({"command": "get_public_key"})
        
        self._cached_pubkey = response["public_key"]
        self._cached_code_hash = response.get("code_hash")
        
        return self._cached_code_hash
    
    def sign_weights(self, weights_hash: str) -> str:
        """
        Sign a weights hash.
        
        Args:
            weights_hash: SHA256 hex string (64 chars)
            
        Returns:
            Ed25519 signature as hex string
        """
        response = self._send_request({
            "command": "sign_weights",
            "weights_hash": weights_hash
        })
        
        return response["signature"]
    
    def get_attestation(self, epoch_id: int) -> Dict[str, Any]:
        """
        Get attestation document for an epoch.
        
        Args:
            epoch_id: Epoch being attested
            
        Returns:
            Dict with attestation_b64, user_data, is_mock
        """
        response = self._send_request({
            "command": "get_attestation",
            "epoch_id": epoch_id
        })
        
        return {
            "attestation_b64": response["attestation_b64"],
            "user_data": response["user_data"],
            "is_mock": response.get("is_mock", False)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check enclave health.
        
        Returns:
            Health status dict
        """
        return self._send_request({"command": "health"})


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

_client: Optional[ValidatorEnclaveClient] = None


def _get_client() -> ValidatorEnclaveClient:
    """Get or create the enclave client singleton."""
    global _client
    if _client is None:
        _client = ValidatorEnclaveClient()
    return _client


def sign_weights_via_enclave(weights_hash: str) -> str:
    """Sign weights hash via enclave."""
    return _get_client().sign_weights(weights_hash)


def get_enclave_pubkey() -> str:
    """Get enclave public key."""
    return _get_client().get_public_key()


def get_enclave_code_hash() -> str:
    """Get enclave code hash."""
    return _get_client().get_code_hash()


def get_enclave_attestation(epoch_id: int) -> Dict[str, Any]:
    """Get enclave attestation for epoch."""
    return _get_client().get_attestation(epoch_id)


def is_enclave_available() -> bool:
    """Check if enclave is available."""
    try:
        cid = get_enclave_cid()
        return cid is not None
    except:
        return False

