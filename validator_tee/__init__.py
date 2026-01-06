"""
Validator TEE Module
====================

Provides TEE (Trusted Execution Environment) functionality for the primary validator.

This module runs INSIDE the Nitro Enclave and handles:
- Ed25519 keypair generation and management
- Weight signing (signs bundle_weights_hash digest bytes)
- Attestation document generation (with epoch_id for replay protection)

SECURITY CONSTRAINTS:
- The validator TEE does NOT expose a generic sign(bytes) API
- Signing is constrained to internally computed weights
- Attestation includes epoch_id to prevent replay attacks

Usage:
    from validator_tee.enclave_signer import (
        initialize_enclave_keypair,
        get_enclave_public_key_hex,
        sign_weights,
        generate_attestation_document,
    )
"""

from validator_tee.enclave_signer import (
    initialize_enclave_keypair,
    get_enclave_public_key_hex,
    sign_weights,
    sign_digest,
    generate_attestation_document,
    get_attestation_document_b64,
    get_code_hash,
    is_keypair_initialized,
)

__all__ = [
    "initialize_enclave_keypair",
    "get_enclave_public_key_hex",
    "sign_weights",
    "sign_digest",
    "generate_attestation_document",
    "get_attestation_document_b64",
    "get_code_hash",
    "is_keypair_initialized",
]

