"""
Validator TEE vsock Client
==========================

This module runs on the HOST (parent EC2) and communicates with the 
validator enclave via vsock.

Only authoritative V2 boot, hotkey, weight, recovery, and health RPCs are
exposed. The V1 blind-signing and host-snapshot APIs are absent.
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
MAX_RPC_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 16 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


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
    
    # Fall back to nitro-cli (for host). Resolve an absolute path first:
    # coordinator/worker processes can run with a PATH that lacks /usr/bin.
    try:
        import shutil as _shutil

        nitro_cli = _shutil.which("nitro-cli") or "/usr/bin/nitro-cli"
        result = subprocess.run(
            [nitro_cli, "describe-enclaves"],
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
    
    def _send_request(
        self,
        request: Dict[str, Any],
        *,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
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
        sock.settimeout(timeout_seconds)
        
        try:
            # Connect to enclave
            sock.connect((cid, RPC_PORT))
            
            # Send request
            request_data = json.dumps(request).encode()
            if len(request_data) < 2 or len(request_data) > MAX_RPC_REQUEST_BYTES:
                raise RuntimeError("Enclave request size is outside the allowed range")
            sock.sendall(len(request_data).to_bytes(4, byteorder="big") + request_data)
            
            # Receive response
            prefix = _recv_exact(sock, 4)
            if len(prefix) != 4:
                raise RuntimeError("Failed to read enclave response length")
            response_length = int.from_bytes(prefix, byteorder="big")
            if response_length < 2 or response_length > MAX_RPC_RESPONSE_BYTES:
                raise RuntimeError("Enclave response size is outside the allowed range")
            response_data = _recv_exact(sock, response_length)
            if len(response_data) != response_length:
                raise RuntimeError("Enclave response body is incomplete")
            
            response = json.loads(response_data.decode())
            
            if response.get("status") == "error":
                raise RuntimeError(f"Enclave error: {response.get('error')}")
            
            return response
            
        finally:
            sock.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check enclave health.
        
        Returns:
            Health status dict
        """
        return self._send_request({"command": "health"})

    def configure_authoritative_v2(
        self,
        configuration: Dict[str, Any],
        expected_config_hash: str,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "configure_authoritative_v2",
                "configuration": configuration,
                "expected_config_hash": expected_config_hash,
            },
            timeout_seconds=120,
        )
        return dict(response["boot_identity"])

    def get_authoritative_v2_boot_identity(self) -> Dict[str, Any]:
        response = self._send_request(
            {"command": "get_authoritative_v2_boot_identity"}
        )
        return dict(response["boot_identity"])

    def configure_hotkey_authority_v2(
        self,
        configuration: Dict[str, Any],
        expected_config_hash: str,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "configure_hotkey_authority_v2",
                "configuration": configuration,
                "expected_config_hash": expected_config_hash,
            },
            timeout_seconds=120,
        )
        return dict(response["hotkey_state"])

    def get_hotkey_recipient_v2(self) -> Dict[str, Any]:
        response = self._send_request({"command": "get_hotkey_recipient_v2"})
        return dict(response["recipient_request"])

    def provision_hotkey_v2(
        self,
        ciphertext_for_recipient_b64: str,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "provision_hotkey_v2",
                "ciphertext_for_recipient_b64": ciphertext_for_recipient_b64,
            },
            timeout_seconds=120,
        )
        return dict(response["hotkey_state"])

    def get_hotkey_state_v2(self) -> Dict[str, Any]:
        response = self._send_request({"command": "get_hotkey_state_v2"})
        return dict(response["hotkey_state"])

    def sign_application_message_v2(
        self,
        message: bytes,
        *,
        parent_receipt_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(message, bytes):
            raise TypeError("application message must be bytes")
        request = {
            "command": "sign_application_message_v2",
            "message_hex": message.hex(),
        }
        if parent_receipt_hash is not None:
            request["parent_receipt_hash"] = parent_receipt_hash
        response = self._send_request(request)
        return dict(response["signature_result"])

    def prepare_weight_commit_v2(
        self,
        commit_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "prepare_weight_commit_v2",
                "commit_request": commit_request,
            },
            timeout_seconds=120,
        )
        return dict(response["commit_result"])

    def sign_weight_extrinsic_v2(
        self,
        signature_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "sign_weight_extrinsic_v2",
                "signature_request": signature_request,
            },
            timeout_seconds=120,
        )
        return dict(response["signature_result"])

    def confirm_weight_publication_v2(
        self,
        weight_authorization_id: str,
        *,
        finalization_scan_id: str,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "confirm_weight_publication_v2",
                "weight_authorization_id": str(weight_authorization_id),
                "finalization_scan_id": str(finalization_scan_id),
            },
            timeout_seconds=600,
        )
        return dict(response["finalization_result"])

    def recover_weight_publication_v2(
        self,
        *,
        published_bundle: Dict[str, Any],
        weight_submission_event_hash: str,
        extrinsic_signature_results: list,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "recover_weight_publication_v2",
                "published_bundle": published_bundle,
                "weight_submission_event_hash": str(
                    weight_submission_event_hash
                ),
                "extrinsic_signature_results": list(
                    extrinsic_signature_results
                ),
            },
            timeout_seconds=600,
        )
        return dict(response["recovery_result"])

    def sign_serve_axon_extrinsic_v2(
        self,
        signature_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "sign_serve_axon_extrinsic_v2",
                "signature_request": signature_request,
            },
            timeout_seconds=120,
        )
        return dict(response["signature_result"])

    def compute_authoritative_weights_v2(
        self,
        weight_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "compute_authoritative_weights_v2",
                "weight_request": weight_request,
            },
            timeout_seconds=180,
        )
        return {
            "weight_snapshot": response["weight_snapshot"],
            "weight_result": response["weight_result"],
            "weights_signature": response["weights_signature"],
            "receipt_graph": response["receipt_graph"],
            "boot_identity": response["boot_identity"],
            "weight_authorization_id": response["weight_authorization_id"],
            "source_artifacts": response["source_artifacts"],
            "epoch_authority": response.get("epoch_authority"),
            "epoch_boundary": response.get("epoch_boundary"),
        }

    def capture_subnet_epoch_boundary_v2(
        self,
        *,
        cutover_manifest: Dict[str, Any],
        settlement_epoch_id: int,
    ) -> Dict[str, Any]:
        response = self._send_request(
            {
                "command": "capture_subnet_epoch_boundary_v2",
                "capture_request": {
                    "cutover_manifest": dict(cutover_manifest),
                    "settlement_epoch_id": int(settlement_epoch_id),
                },
            },
            timeout_seconds=180,
        )
        return dict(response["capture_result"])


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


def configure_authoritative_enclave_v2(
    configuration: Dict[str, Any],
    expected_config_hash: str,
) -> Dict[str, Any]:
    return _get_client().configure_authoritative_v2(
        configuration,
        expected_config_hash,
    )


def get_authoritative_boot_identity_v2() -> Dict[str, Any]:
    return _get_client().get_authoritative_v2_boot_identity()


def configure_hotkey_authority_v2(
    configuration: Dict[str, Any],
    expected_config_hash: str,
) -> Dict[str, Any]:
    return _get_client().configure_hotkey_authority_v2(
        configuration,
        expected_config_hash,
    )


def get_hotkey_recipient_v2() -> Dict[str, Any]:
    return _get_client().get_hotkey_recipient_v2()


def provision_hotkey_v2(ciphertext_for_recipient_b64: str) -> Dict[str, Any]:
    return _get_client().provision_hotkey_v2(ciphertext_for_recipient_b64)


def get_hotkey_state_v2() -> Dict[str, Any]:
    return _get_client().get_hotkey_state_v2()


def sign_application_message_v2(
    message: bytes,
    *,
    parent_receipt_hash: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_client().sign_application_message_v2(
        message,
        parent_receipt_hash=parent_receipt_hash,
    )


def prepare_weight_commit_v2(
    commit_request: Dict[str, Any],
) -> Dict[str, Any]:
    return _get_client().prepare_weight_commit_v2(commit_request)


def sign_weight_extrinsic_v2(
    signature_request: Dict[str, Any],
) -> Dict[str, Any]:
    return _get_client().sign_weight_extrinsic_v2(signature_request)


def sign_serve_axon_extrinsic_v2(
    signature_request: Dict[str, Any],
) -> Dict[str, Any]:
    return _get_client().sign_serve_axon_extrinsic_v2(signature_request)


def compute_authoritative_weights_via_enclave_v2(
    weight_request: Dict[str, Any],
) -> Dict[str, Any]:
    return _get_client().compute_authoritative_weights_v2(weight_request)


def is_enclave_available() -> bool:
    """Check if enclave is available."""
    try:
        cid = get_enclave_cid()
        return cid is not None
    except:
        return False
