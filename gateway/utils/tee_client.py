"""
Client for communicating with TEE enclave via vsock.

This module provides an async interface for the parent EC2 instance to
communicate with the Nitro Enclave running the TEE service.

vsock (Virtual Socket) is a socket protocol designed for VM-to-host communication,
providing a secure channel between the parent EC2 and the enclave.
"""

import socket
import json
import asyncio
import subprocess
import base64
import hashlib
from typing import Dict, List, Optional
from datetime import datetime


# vsock address family constant (Linux)
AF_VSOCK = 40  # socket.AF_VSOCK on Linux systems

# Parent EC2 CID (reserved)
PARENT_CID = 3

# RPC port for TEE communication
RPC_PORT = 5000
MAX_RPC_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RPC_RESPONSE_BYTES = 256 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


class TEEClient:
    """
    Async client for vsock RPC communication with TEE enclave.
    
    The enclave's CID (Context ID) is dynamically assigned by AWS and can be
    retrieved using `nitro-cli describe-enclaves`.
    """
    
    def __init__(self, cid: Optional[int] = None, port: int = RPC_PORT):
        """
        Initialize TEE client.
        
        Args:
            cid: Enclave CID (if None, will be auto-detected)
            port: vsock port number (default: 5000)
        """
        self.cid = cid
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._lock = asyncio.Lock()
    
    async def _get_enclave_cid(self) -> Optional[int]:
        """
        Auto-detect enclave CID from nitro-cli.
        
        Returns:
            Enclave CID or None if no enclave running
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "sudo", "nitro-cli", "describe-enclaves",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                print(f"⚠️ nitro-cli error: {stderr.decode()}")
                return None
            
            enclaves = json.loads(stdout.decode())
            
            if not enclaves:
                print("⚠️ No enclaves running")
                return None
            
            cid = enclaves[0].get("EnclaveCID")
            print(f"✅ Detected enclave CID: {cid}")
            return cid
        
        except Exception as e:
            print(f"❌ Failed to get enclave CID: {e}")
            return None
    
    async def _ensure_connected(self):
        """
        Ensure vsock connection is established.
        
        Note: Always creates a fresh connection since enclave closes socket after each RPC.
        
        Raises:
            RuntimeError: If enclave is not running or connection fails
        """
        async with self._lock:
            # If no CID provided, auto-detect
            if self.cid is None:
                self.cid = await self._get_enclave_cid()
                if self.cid is None:
                    raise RuntimeError("No enclave running - cannot connect")
            
            # Always close existing socket and create fresh connection
            # (Enclave closes socket after each RPC)
            if self._socket is not None:
                try:
                    self._socket.close()
                except:
                    pass
                self._socket = None
            
            # Create fresh vsock socket
            try:
                self._socket = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
                self._socket.settimeout(30.0)  # 30 second timeout (NSM hardware call can be slow)
                self._socket.connect((self.cid, self.port))
                print(f"✅ Connected to enclave via vsock (CID {self.cid}, port {self.port})")
            except Exception as e:
                self._socket = None
                raise RuntimeError(f"Failed to connect to enclave: {e}")
    
    async def _send_rpc(self, method: str, params: Optional[Dict] = None) -> Dict:
        """
        Send RPC request to enclave and wait for response.
        
        Protocol:
        - Send: {"method": "method_name", "params": {...}}
        - Receive: {"status": "success", "result": ...} or {"status": "error", "error": "..."}
        
        Args:
            method: RPC method name
            params: Optional parameters dict
        
        Returns:
            Result dict from enclave
        
        Raises:
            RuntimeError: If RPC fails or enclave returns error
        """
        await self._ensure_connected()
        
        # Build RPC request
        request = {
            "method": method,
            "params": params or {}
        }
        
        # Serialize to JSON
        request_json = json.dumps(request)
        request_bytes = request_json.encode('utf-8')
        
        # Send request (with length prefix)
        request_length = len(request_bytes)
        if request_length < 2 or request_length > MAX_RPC_REQUEST_BYTES:
            raise RuntimeError("RPC request size is outside the allowed range")
        length_prefix = request_length.to_bytes(4, byteorder='big')
        
        try:
            self._socket.sendall(length_prefix + request_bytes)
            
            # Receive response (read length prefix first)
            response_length_bytes = _recv_exact(self._socket, 4)
            if len(response_length_bytes) != 4:
                raise RuntimeError("Failed to read response length")
            
            response_length = int.from_bytes(response_length_bytes, byteorder='big')
            if response_length < 2 or response_length > MAX_RPC_RESPONSE_BYTES:
                raise RuntimeError("RPC response size is outside the allowed range")
            
            # Read response body
            response_bytes = _recv_exact(self._socket, response_length)
            if len(response_bytes) != response_length:
                raise RuntimeError("Connection closed by enclave")
            
            # Parse response
            response = json.loads(response_bytes.decode('utf-8'))
            
            # Check status
            if response.get("status") == "error":
                raise RuntimeError(f"Enclave error: {response.get('error')}")
            
            return response.get("result", {})
        
        except Exception as e:
            # Close socket on error (will reconnect on next call)
            if self._socket:
                self._socket.close()
                self._socket = None
            raise RuntimeError(f"RPC failed: {e}")
    
    async def append_event(self, event: Dict) -> Dict:
        """
        Append event to TEE buffer.
        
        Args:
            event: Event dict (without sequence number)
        
        Returns:
            {"status": "buffered", "sequence": N}
        """
        return await self._send_rpc("append_event", {"event": event})
    
    async def get_buffer(self) -> List[Dict]:
        """
        Retrieve all buffered events from TEE.
        
        Returns:
            List of event dicts
        """
        result = await self._send_rpc("get_buffer", {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            events = result.get("events", [])
            return events if isinstance(events, list) else []
        return []
    
    async def get_buffer_size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of events in buffer
        """
        result = await self._send_rpc("get_buffer_size", {})
        # Result is directly an int, not a dict
        return result if isinstance(result, int) else 0
    
    async def clear_buffer(self) -> Dict:
        """
        Clear buffer after successful Arweave upload.
        
        Returns:
            {"status": "cleared", "cleared_events": N, "next_checkpoint_at": "ISO8601"}
        """
        return await self._send_rpc("clear_buffer", {})
    
    async def get_public_key(self) -> bytes:
        """
        Get enclave's Ed25519 public key.
        
        Returns:
            32-byte public key (raw bytes)
        """
        result = await self._send_rpc("get_public_key", {})
        public_key_hex = result.get("public_key")
        return bytes.fromhex(public_key_hex)
    
    async def get_attestation(self) -> Dict:
        """
        Get attestation document from enclave.
        
        Returns:
            {
                "attestation_document": "hex",
                "public_key": "hex",
                "code_hash": "hex",
                "pcr0": "hex",
                "pcr1": "hex",
                "pcr2": "hex"
            }
        """
        return await self._send_rpc("get_attestation", {})

    async def role_health(self) -> Dict:
        """Return the measured physical role and build identity for this CID."""
        return await self._send_rpc("role_health", {})

    async def v2_configure_runtime(
        self,
        *,
        configuration: Dict,
        configuration_hash: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_configure_runtime",
            {
                "schema_version": "leadpoet.enclave_runtime_config.v2",
                "configuration": configuration,
                "configuration_hash": configuration_hash,
            },
        )

    async def v2_get_boot_identity(self) -> Dict:
        return await self._send_rpc("v2_get_boot_identity", {})

    async def v2_get_transport_certificate(self) -> bytes:
        result = await self._send_rpc("v2_get_transport_certificate", {})
        try:
            return base64.b64decode(
                str(result["certificate_pem_b64"]),
                validate=True,
            )
        except Exception as exc:
            raise RuntimeError("enclave V2 transport certificate is invalid") from exc

    async def v2_register_peer(
        self,
        *,
        boot_identity: Dict,
        certificate_pem: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "v2_register_peer",
            {
                "boot_identity": boot_identity,
                "certificate_pem_b64": base64.b64encode(certificate_pem).decode(
                    "ascii"
                ),
            },
        )

    async def v2_start_tls_service(self) -> Dict:
        return await self._send_rpc("v2_start_tls_service", {})

    async def v2_peer_status(self) -> Dict:
        return await self._send_rpc("v2_peer_status", {})

    async def v2_call_peer_health(self, physical_role: str) -> Dict:
        return await self._send_rpc(
            "v2_call_peer_health",
            {"physical_role": physical_role},
        )

    async def v2_provider_broker_health(self) -> Dict:
        return await self._send_rpc("v2_provider_broker_health", {})

    async def v2_provider_semantics_health(self) -> Dict:
        return await self._send_rpc("v2_provider_semantics_health", {})

    async def v2_get_kms_recipient(self, credential_slot: str) -> Dict:
        return await self._send_rpc(
            "v2_get_kms_recipient",
            {
                "credential_slot": credential_slot,
            },
        )

    async def v2_get_source_add_ingress_recipient(
        self,
        *,
        miner_hotkey: str,
        adapter_ref: str,
        credential_ref: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_get_source_add_ingress_recipient",
            {
                "miner_hotkey": miner_hotkey,
                "adapter_ref": adapter_ref,
                "credential_ref": credential_ref,
            },
        )

    async def v2_seal_source_add_ingress_credential(
        self,
        *,
        request_id: str,
        ciphertext_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_seal_source_add_ingress_credential",
            {
                "request_id": request_id,
                "ciphertext_b64": ciphertext_b64,
            },
        )

    async def v2_get_openrouter_ingress_recipient(
        self,
        *,
        miner_hotkey: str,
        credential_kind: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_get_openrouter_ingress_recipient",
            {
                "miner_hotkey": miner_hotkey,
                "credential_kind": credential_kind,
            },
        )

    async def v2_seal_openrouter_ingress_credential(
        self,
        *,
        request_id: str,
        ciphertext_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_seal_openrouter_ingress_credential",
            {
                "request_id": request_id,
                "ciphertext_b64": ciphertext_b64,
            },
        )

    async def v2_provision_encrypted_secret(
        self,
        *,
        credential_slot: str,
        ciphertext_for_recipient_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_encrypted_secret",
            {
                "credential_slot": credential_slot,
                "ciphertext_for_recipient_b64": ciphertext_for_recipient_b64,
            },
        )

    async def v2_get_job_kms_recipient(
        self,
        *,
        job_id: str,
        credential_slot: str,
        credential_value_hash: str,
        key_ref_hash: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_get_job_kms_recipient",
            {
                "job_id": job_id,
                "credential_slot": credential_slot,
                "credential_value_hash": credential_value_hash,
                "key_ref_hash": key_ref_hash,
            },
        )

    async def v2_provision_job_encrypted_secret(
        self,
        *,
        request_id: str,
        ciphertext_for_recipient_b64: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_job_encrypted_secret",
            {
                "request_id": request_id,
                "ciphertext_for_recipient_b64": ciphertext_for_recipient_b64,
            },
        )

    async def v2_provision_job_sealed_source_add_secret(
        self,
        *,
        envelope: Dict,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_job_sealed_source_add_secret",
            {"envelope": envelope},
        )

    async def v2_provision_job_sealed_openrouter_secret(
        self,
        *,
        envelope: Dict,
    ) -> Dict:
        return await self._send_rpc(
            "v2_provision_job_sealed_openrouter_secret",
            {"envelope": envelope},
        )

    async def v2_release_job_credentials(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "v2_release_job_credentials",
            {"job_id": job_id},
        )

    async def v2_list_encrypted_artifacts(self, *, job_id: str, purpose: str) -> Dict:
        return await self._send_rpc(
            "v2_list_encrypted_artifacts",
            {"job_id": job_id, "purpose": purpose},
        )

    async def v2_export_encrypted_artifact(self, artifact_id: str) -> Dict:
        return await self._send_rpc(
            "v2_export_encrypted_artifact",
            {"artifact_id": artifact_id},
        )

    async def v2_verify_encrypted_artifact_persistence(
        self,
        *,
        artifact_id: str,
        attestation_job_id: str,
        artifact_ref: str,
        get_url: str,
        head_url: str,
    ) -> Dict:
        return await self._send_rpc(
            "v2_verify_encrypted_artifact_persistence",
            {
                "artifact_id": artifact_id,
                "attestation_job_id": attestation_job_id,
                "artifact_ref": artifact_ref,
                "get_url": get_url,
                "head_url": head_url,
            },
        )
    
    async def get_buffer_stats(self) -> Dict:
        """
        Get buffer statistics from TEE.
        
        Returns:
            {
                "size": int,
                "sequence_range": {"first": int, "last": int},
                "age_seconds": float,
                ...
            }
        """
        return await self._send_rpc("get_buffer_stats", {})
    
    async def build_checkpoint(self) -> Dict:
        """
        Request checkpoint from TEE (for hourly batching).
        
        Returns:
            {
                "header": {...},
                "signature": "hex",
                "events": [...],
                "tree_levels": [...]
            }
        """
        return await self._send_rpc("build_checkpoint", {})

    async def scoring_configure_runtime(
        self,
        *,
        environment: Dict,
        configuration_hash: str,
    ) -> Dict:
        """Provision reviewed scoring env values once without logging them."""
        return await self._send_rpc(
            "scoring_configure_runtime",
            {
                "schema_version": "leadpoet.gateway_scoring_runtime.v1",
                "environment": environment,
                "configuration_hash": configuration_hash,
            },
        )

    async def scoring_health(self) -> Dict:
        """Return attested-scoring mode, bounds, and queue health."""
        return await self._send_rpc("scoring_health", {})

    async def scoring_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc("scoring_submit_job", {"manifest": manifest})

    async def scoring_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def scoring_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_seal_job", {"job_id": job_id})

    async def scoring_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_get_status", {"job_id": job_id})

    async def scoring_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_cancel_job", {"job_id": job_id})

    async def scoring_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def scoring_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_get_receipt", {"job_id": job_id})

    async def scoring_v2_health(self) -> Dict:
        return await self._send_rpc("scoring_v2_health", {})

    async def scoring_v2_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc("scoring_v2_submit_job", {"manifest": manifest})

    async def scoring_v2_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_v2_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def scoring_v2_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_seal_job", {"job_id": job_id})

    async def scoring_v2_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_get_status", {"job_id": job_id})

    async def scoring_v2_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_cancel_job", {"job_id": job_id})

    async def scoring_v2_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "scoring_v2_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def scoring_v2_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc("scoring_v2_get_receipt", {"job_id": job_id})

    async def scoring_v2_get_receipts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc("scoring_v2_get_receipts", {"job_id": job_id})

    async def scoring_v2_get_transitions(self, job_id: str) -> List[Dict]:
        return await self._send_rpc("scoring_v2_get_transitions", {"job_id": job_id})

    async def scoring_v2_get_transport_attempts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "scoring_v2_get_transport_attempts",
            {"job_id": job_id},
        )

    async def scoring_v2_get_artifact_hashes(self, job_id: str) -> List[str]:
        return await self._send_rpc(
            "scoring_v2_get_artifact_hashes", {"job_id": job_id}
        )

    async def scoring_v2_get_host_operations(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "scoring_v2_get_host_operations", {"job_id": job_id}
        )

    async def coordinator_v2_health(self) -> Dict:
        return await self._send_rpc("coordinator_v2_health", {})

    async def coordinator_v2_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_submit_job", {"manifest": manifest}
        )

    async def coordinator_v2_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def coordinator_v2_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_seal_job", {"job_id": job_id})

    async def coordinator_v2_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_get_status", {"job_id": job_id})

    async def coordinator_v2_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_cancel_job", {"job_id": job_id})

    async def coordinator_v2_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "coordinator_v2_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def coordinator_v2_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc("coordinator_v2_get_receipt", {"job_id": job_id})

    async def coordinator_v2_get_receipts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_receipts", {"job_id": job_id}
        )

    async def coordinator_v2_get_transitions(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_transitions", {"job_id": job_id}
        )

    async def coordinator_v2_get_transport_attempts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_transport_attempts",
            {"job_id": job_id},
        )

    async def coordinator_v2_get_artifact_hashes(self, job_id: str) -> List[str]:
        return await self._send_rpc(
            "coordinator_v2_get_artifact_hashes", {"job_id": job_id}
        )

    async def coordinator_v2_get_host_operations(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "coordinator_v2_get_host_operations", {"job_id": job_id}
        )

    async def autoresearch_v2_health(self) -> Dict:
        return await self._send_rpc("autoresearch_v2_health", {})

    async def autoresearch_v2_submit_job(self, manifest: Dict) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_submit_job", {"manifest": manifest}
        )

    async def autoresearch_v2_put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data: bytes,
    ) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_put_chunk",
            {
                "job_id": job_id,
                "offset": offset,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "chunk_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
        )

    async def autoresearch_v2_seal_job(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_seal_job", {"job_id": job_id}
        )

    async def autoresearch_v2_get_status(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_get_status", {"job_id": job_id}
        )

    async def autoresearch_v2_cancel_job(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_cancel_job", {"job_id": job_id}
        )

    async def autoresearch_v2_get_result(
        self,
        job_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 512 * 1024,
    ) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_get_result",
            {"job_id": job_id, "offset": offset, "max_bytes": max_bytes},
        )

    async def autoresearch_v2_get_receipt(self, job_id: str) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_get_receipt", {"job_id": job_id}
        )

    async def autoresearch_v2_get_receipts(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_get_receipts", {"job_id": job_id}
        )

    async def autoresearch_v2_get_transitions(self, job_id: str) -> List[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_get_transitions", {"job_id": job_id}
        )

    async def autoresearch_v2_get_transport_attempts(
        self, job_id: str
    ) -> List[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_get_transport_attempts", {"job_id": job_id}
        )

    async def autoresearch_v2_get_artifact_hashes(self, job_id: str) -> List[str]:
        return await self._send_rpc(
            "autoresearch_v2_get_artifact_hashes", {"job_id": job_id}
        )

    async def autoresearch_v2_next_host_operation(
        self,
        job_id: str,
        *,
        wait_ms: int = 0,
    ) -> Optional[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_next_host_operation",
            {"job_id": job_id, "wait_ms": wait_ms},
        )

    async def autoresearch_v2_complete_host_operation(
        self,
        *,
        job_id: str,
        request_hash: str,
        terminal_status: str,
        response: Optional[Dict],
        failure_code: Optional[str] = None,
    ) -> Dict:
        return await self._send_rpc(
            "autoresearch_v2_complete_host_operation",
            {
                "job_id": job_id,
                "request_hash": request_hash,
                "terminal_status": terminal_status,
                "response": response,
                "failure_code": failure_code,
            },
        )

    async def autoresearch_v2_get_host_operations(
        self, job_id: str
    ) -> List[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_get_host_operations", {"job_id": job_id}
        )

    async def autoresearch_v2_get_external_receipt_graphs(
        self, job_id: str
    ) -> List[Dict]:
        return await self._send_rpc(
            "autoresearch_v2_get_external_receipt_graphs", {"job_id": job_id}
        )
    
    def close(self):
        """Close vsock connection."""
        if self._socket:
            self._socket.close()
            self._socket = None


# Fixed CIDs are part of the measured V2 topology. Existing event/checkpoint
# callers retain ``tee_client`` as a coordinator alias.
coordinator_tee_client = TEEClient(cid=16)
scoring_tee_clients = (TEEClient(cid=17), TEEClient(cid=18))
autoresearch_tee_client = TEEClient(cid=19)
tee_client = coordinator_tee_client
