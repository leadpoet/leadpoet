"""
Validator TEE Module
====================

Provides TEE (Trusted Execution Environment) functionality for the primary validator.

This module provides the HOST-SIDE interface for validator TEE operations.
All operations are delegated to the Nitro Enclave via vsock.

SECURITY CONSTRAINTS:
- The validator TEE does NOT expose a generic sign(bytes) API
- Signing is constrained to internally computed weights
- Attestation includes epoch_id to prevent replay attacks
- NO MOCK MODE: Enclave MUST be running, otherwise operations FAIL

Usage:
    from validator_tee import (
        initialize_enclave_keypair,
        get_enclave_pubkey,
        sign_weights,
        get_attestation,
        is_enclave_running,
    )
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
]
