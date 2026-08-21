"""
Validator TEE Module
====================

Provides TEE (Trusted Execution Environment) functionality for the primary validator.

This module provides the HOST-SIDE interface for validator TEE operations.
All operations are delegated to the Nitro Enclave via vsock.

Only authoritative V2 wallet and chain contexts are public. Legacy blind
weight signing, host-snapshot computation, and epoch-attestation helpers are
permanently removed.

The public wallet objects are resolved lazily.  Release and deploy-readiness
controllers import validator release validators through this package, but do
not carry the live validator wallet dependencies.
"""

from __future__ import annotations

from typing import Any


_HOTKEY_EXPORTS = frozenset(
    {
        "AuthoritativeSetWeightsContextV2",
        "AuthoritativeServeAxonContextV2",
        "EnclaveBackedKeypairV2",
        "EnclaveBackedWalletV2",
        "build_enclave_backed_wallet_v2",
    }
)
_WEIGHT_EXPORTS = frozenset({"build_authoritative_weight_bundle_v2"})

__all__ = [
    "AuthoritativeSetWeightsContextV2",
    "AuthoritativeServeAxonContextV2",
    "EnclaveBackedKeypairV2",
    "EnclaveBackedWalletV2",
    "build_enclave_backed_wallet_v2",
    "build_authoritative_weight_bundle_v2",
]


def __getattr__(name: str) -> Any:
    """Resolve live validator runtime exports only when they are requested."""

    if name in _HOTKEY_EXPORTS:
        from validator_tee.host import enclave_hotkey_v2

        value = getattr(enclave_hotkey_v2, name)
    elif name in _WEIGHT_EXPORTS:
        from validator_tee.host import weight_authority_v2

        value = getattr(weight_authority_v2, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
