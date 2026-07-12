"""
Validator TEE Module
====================

Provides TEE (Trusted Execution Environment) functionality for the primary validator.

This module provides the HOST-SIDE interface for validator TEE operations.
All operations are delegated to the Nitro Enclave via vsock.

Only authoritative V2 wallet and chain contexts are public. Legacy blind
weight signing, host-snapshot computation, and epoch-attestation helpers are
permanently removed.
"""
from validator_tee.host.enclave_hotkey_v2 import (
    AuthoritativeSetWeightsContextV2,
    AuthoritativeServeAxonContextV2,
    EnclaveBackedKeypairV2,
    EnclaveBackedWalletV2,
    build_enclave_backed_wallet_v2,
)
from validator_tee.host.weight_authority_v2 import (
    build_authoritative_weight_bundle_v2,
)

__all__ = [
    "AuthoritativeSetWeightsContextV2",
    "AuthoritativeServeAxonContextV2",
    "EnclaveBackedKeypairV2",
    "EnclaveBackedWalletV2",
    "build_enclave_backed_wallet_v2",
    "build_authoritative_weight_bundle_v2",
]
