"""
Validator TEE Host Module
=========================

Files that run on the HOST (parent EC2), NOT inside the enclave.
These communicate with the enclave via vsock.
"""

from validator_tee.host.vsock_client import (
    ValidatorEnclaveClient,
    is_enclave_available,
    get_enclave_cid,
)

__all__ = [
    "ValidatorEnclaveClient",
    "is_enclave_available",
    "get_enclave_cid",
]
