"""
Validator TEE Host Module
=========================

Files that run on the HOST (parent EC2), NOT inside the enclave.
These communicate with the enclave via vsock.
"""

from validator_tee.host.enclave_signer import (
    initialize_enclave_keypair,
    get_enclave_public_key_hex,
    get_enclave_pubkey,
    sign_weights,
    get_attestation_document_b64,
    get_attestation,
    get_code_hash,
    is_keypair_initialized,
    is_enclave_running,
    get_signer_state,
)

from validator_tee.host.vsock_client import (
    ValidatorEnclaveClient,
    is_enclave_available,
    get_enclave_cid,
)

__all__ = [
    "initialize_enclave_keypair",
    "get_enclave_public_key_hex",
    "get_enclave_pubkey",
    "sign_weights",
    "get_attestation_document_b64",
    "get_attestation",
    "get_code_hash",
    "is_keypair_initialized",
    "is_enclave_running",
    "get_signer_state",
    "ValidatorEnclaveClient",
    "is_enclave_available",
    "get_enclave_cid",
]

