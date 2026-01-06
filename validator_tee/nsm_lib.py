"""
AWS Nitro Security Module (NSM) Python Library
===============================================

This module provides Python bindings to the AWS Nitro Secure Module (/dev/nsm)
for requesting attestation documents that contain PCR measurements.

Based on: https://github.com/aws/aws-nitro-enclaves-nsm-api

The NSM device uses ioctl() calls with specific request/response structures.
"""

import os
import struct
import ctypes
from typing import Dict, Optional
import cbor2


# NSM ioctl constants (from aws-nitro-enclaves-nsm-api)
NSM_IOCTL_MAGIC = 0x0A  # NSM device magic number
NSM_REQUEST = ctypes.c_ulong(0xCAFE)  # Actual request code

# Max sizes for NSM requests/responses
NSM_REQUEST_MAX_SIZE = 0x1000  # 4KB
NSM_RESPONSE_MAX_SIZE = 0x3000  # 12KB


class NSMError(Exception):
    """Exception raised for NSM operation errors."""
    pass


def get_attestation_document(
    user_data: Optional[bytes] = None,
    nonce: Optional[bytes] = None,
    public_key: Optional[bytes] = None
) -> Dict:
    """
    Request an attestation document from the Nitro Security Module.
    
    The attestation document contains:
    - PCR measurements (PCR0, PCR1, PCR2, etc.)
    - Module ID
    - Timestamp
    - Public key (if provided)
    - User data (if provided)
    - AWS Nitro signature
    
    Args:
        user_data: Optional user data to include in attestation (max 512 bytes)
        nonce: Optional nonce for replay protection (max 512 bytes)
        public_key: Optional public key to bind to attestation (DER format)
    
    Returns:
        Dict containing the attestation document (CBOR decoded)
    
    Raises:
        NSMError: If the NSM device is unavailable or returns an error
    """
    NSM_DEVICE = "/dev/nsm"
    
    # Check if NSM device exists
    if not os.path.exists(NSM_DEVICE):
        raise NSMError(f"{NSM_DEVICE} not found - not running in Nitro Enclave")
    
    # Build CBOR request
    # Format: {"Attestation": {"user_data": ..., "nonce": ..., "public_key": ...}}
    attestation_request = {"Attestation": {}}
    
    if user_data is not None:
        attestation_request["Attestation"]["user_data"] = user_data
    if nonce is not None:
        attestation_request["Attestation"]["nonce"] = nonce
    if public_key is not None:
        attestation_request["Attestation"]["public_key"] = public_key
    
    # Encode request as CBOR
    request_cbor = cbor2.dumps(attestation_request)
    
    # The NSM device requires a specific ioctl() structure:
    # We need to use a raw ioctl with the request/response buffers
    
    try:
        # Open NSM device
        fd = os.open(NSM_DEVICE, os.O_RDWR)
        
        try:
            # Allocate request buffer (4KB)
            request_buffer = bytearray(NSM_REQUEST_MAX_SIZE)
            request_buffer[:len(request_cbor)] = request_cbor
            
            # Allocate response buffer (12KB)
            response_buffer = bytearray(NSM_RESPONSE_MAX_SIZE)
            
            # Create ctypes buffers
            req_buf = (ctypes.c_ubyte * len(request_buffer)).from_buffer(request_buffer)
            resp_buf = (ctypes.c_ubyte * len(response_buffer)).from_buffer(response_buffer)
            
            # Define ioctl structure for NSM
            # The NSM device expects: struct nsm_message { input, input_len, output, output_len }
            class NSMMessage(ctypes.Structure):
                _fields_ = [
                    ("request", ctypes.POINTER(ctypes.c_ubyte)),
                    ("request_len", ctypes.c_uint32),
                    ("response", ctypes.POINTER(ctypes.c_ubyte)),
                    ("response_len", ctypes.c_uint32),
                ]
            
            msg = NSMMessage()
            msg.request = req_buf
            msg.request_len = len(request_cbor)
            msg.response = resp_buf
            msg.response_len = len(response_buffer)
            
            # Make ioctl call
            # ioctl number: _IOWR(NSM_IOCTL_MAGIC, 0, struct nsm_message)
            import fcntl
            
            # Calculate ioctl request number
            # _IOWR = (dir << 30) | (size << 16) | (magic << 8) | nr
            # where dir=3 (read+write), size=sizeof(nsm_message), magic=0x0A, nr=0
            _IOC_WRITE = 1
            _IOC_READ = 2
            _IOC_NONE = 0
            _IOC_NRBITS = 8
            _IOC_TYPEBITS = 8
            _IOC_SIZEBITS = 14
            _IOC_DIRBITS = 2
            
            _IOC_NRSHIFT = 0
            _IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
            _IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
            _IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
            
            def _IOC(dir, type, nr, size):
                return (dir << _IOC_DIRSHIFT) | (type << _IOC_TYPESHIFT) | \
                       (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)
            
            def _IOWR(type, nr, size):
                return _IOC(_IOC_READ | _IOC_WRITE, type, nr, size)
            
            # NSM ioctl number
            NSM_IOCTL_REQUEST = _IOWR(NSM_IOCTL_MAGIC, 0, ctypes.sizeof(NSMMessage))
            
            # Make the ioctl call
            result = fcntl.ioctl(fd, NSM_IOCTL_REQUEST, msg)
            
            # Extract response
            response_data = bytes(response_buffer[:msg.response_len])
            
            # Decode CBOR response
            response = cbor2.loads(response_data)
            
            return response
            
        finally:
            os.close(fd)
            
    except OSError as e:
        raise NSMError(f"NSM ioctl failed: {e}")
    except Exception as e:
        raise NSMError(f"NSM attestation failed: {e}")


def get_pcr_measurements() -> Dict[str, str]:
    """
    Get PCR measurements from the Nitro Security Module.
    
    Returns:
        Dict mapping PCR names to hex values: {"PCR0": "...", "PCR1": "...", ...}
    
    Raises:
        NSMError: If attestation fails
    """
    print("[NSM_LIB] 🔍 DEBUG: Starting get_pcr_measurements()", flush=True)
    
    # Request attestation document (minimal - no user data)
    print("[NSM_LIB] 🔍 DEBUG: Calling get_attestation_document()...", flush=True)
    attestation_response = get_attestation_document()
    
    print(f"[NSM_LIB] 🔍 DEBUG: Attestation response type: {type(attestation_response)}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG: Attestation response keys: {list(attestation_response.keys()) if isinstance(attestation_response, dict) else 'NOT A DICT'}", flush=True)
    
    # Extract attestation document from response
    # Response format: {"Attestation": {"document": <CBOR-encoded attestation doc>}}
    if "Attestation" not in attestation_response:
        print(f"[NSM_LIB] ❌ DEBUG: Missing 'Attestation' key. Full response: {attestation_response}", flush=True)
        raise NSMError("Invalid attestation response: missing 'Attestation' key")
    
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_response['Attestation'] type: {type(attestation_response['Attestation'])}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_response['Attestation'] keys: {list(attestation_response['Attestation'].keys()) if isinstance(attestation_response['Attestation'], dict) else 'NOT A DICT'}", flush=True)
    
    attestation_doc_bytes = attestation_response["Attestation"]["document"]
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_doc_bytes type: {type(attestation_doc_bytes)}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_doc_bytes length: {len(attestation_doc_bytes)} bytes", flush=True)
    
    # Decode the inner attestation document (COSE Sign1 structure)
    print(f"[NSM_LIB] 🔍 DEBUG: Decoding COSE Sign1 structure with cbor2...", flush=True)
    cose_sign1 = cbor2.loads(attestation_doc_bytes)
    
    print(f"[NSM_LIB] 🔍 DEBUG: COSE Sign1 type: {type(cose_sign1)}", flush=True)
    
    # COSE Sign1 is a 4-element array: [protected, unprotected, payload, signature]
    if not isinstance(cose_sign1, list) or len(cose_sign1) != 4:
        raise NSMError(f"Invalid COSE Sign1 structure: expected 4-element list, got {type(cose_sign1)} with {len(cose_sign1) if isinstance(cose_sign1, list) else 'N/A'} elements")
    
    print(f"[NSM_LIB] 🔍 DEBUG: COSE Sign1 structure validated (4 elements)", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG:   [0] protected: {len(cose_sign1[0])} bytes", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG:   [1] unprotected: {type(cose_sign1[1])}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG:   [2] payload: {len(cose_sign1[2])} bytes", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG:   [3] signature: {len(cose_sign1[3])} bytes", flush=True)
    
    # Extract the payload (element [2]) and decode it
    payload_bytes = cose_sign1[2]
    print(f"[NSM_LIB] 🔍 DEBUG: Decoding payload (actual attestation document)...", flush=True)
    attestation_doc = cbor2.loads(payload_bytes)
    
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_doc type: {type(attestation_doc)}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG: attestation_doc keys: {list(attestation_doc.keys()) if isinstance(attestation_doc, dict) else 'NOT A DICT'}", flush=True)
    
    # Print ALL keys in the attestation document
    if isinstance(attestation_doc, dict):
        for key in attestation_doc.keys():
            value_type = type(attestation_doc[key])
            if isinstance(attestation_doc[key], bytes):
                value_preview = f"<{len(attestation_doc[key])} bytes>"
            elif isinstance(attestation_doc[key], dict):
                value_preview = f"<dict with {len(attestation_doc[key])} keys>"
            else:
                value_preview = str(attestation_doc[key])[:50]
            print(f"[NSM_LIB] 🔍 DEBUG:   - {key}: {value_type} = {value_preview}", flush=True)
    
    # Extract PCRs
    # Attestation document format: {"module_id": ..., "timestamp": ..., "pcrs": {0: bytes, 1: bytes, ...}}
    if "pcrs" not in attestation_doc:
        print(f"[NSM_LIB] ❌ DEBUG: Missing 'pcrs' key in attestation document!", flush=True)
        print(f"[NSM_LIB] ❌ DEBUG: Available keys: {list(attestation_doc.keys())}", flush=True)
        raise NSMError("Invalid attestation document: missing 'pcrs' key")
    
    pcrs_dict = attestation_doc["pcrs"]
    print(f"[NSM_LIB] 🔍 DEBUG: pcrs_dict type: {type(pcrs_dict)}", flush=True)
    print(f"[NSM_LIB] 🔍 DEBUG: pcrs_dict keys: {list(pcrs_dict.keys())}", flush=True)
    
    # Convert PCRs from bytes to hex strings
    pcr_measurements = {}
    for pcr_index, pcr_bytes in pcrs_dict.items():
        pcr_hex = pcr_bytes.hex()
        pcr_measurements[f"PCR{pcr_index}"] = pcr_hex
        print(f"[NSM_LIB] 🔍 DEBUG: PCR{pcr_index} = {pcr_hex[:32]}...{pcr_hex[-32:]}", flush=True)
    
    print(f"[NSM_LIB] ✅ DEBUG: Successfully extracted {len(pcr_measurements)} PCRs", flush=True)
    return pcr_measurements

