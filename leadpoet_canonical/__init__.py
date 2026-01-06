"""
LeadPoet Canonical Module

This module provides the canonical implementations for all shared logic across
gateway, validator, and auditor components in the LeadPoet TEE-based validation system.

CRITICAL: ALL components MUST import from this module. Do NOT implement separate
versions of hashing, signing, or verification logic.

Module Structure:
    constants.py   - Single source of truth for EPOCH_LENGTH, WEIGHT_SUBMISSION_BLOCK, etc.
    weights.py     - bundle_weights_hash, compare_weights_hash, normalize_to_u16, u16_to_emit_floats
    chain.py       - normalize_chain_weights (chain weight normalization)
    binding.py     - verify_binding_message, create_binding_message
    events.py      - verify_log_entry (shared verification logic)
    timestamps.py  - canonical_timestamp() - RFC3339 UTC with Z, no microseconds
    nitro.py       - verify_nitro_attestation_full (AWS Nitro attestation verification)

Usage:
    # In gateway/api/weights.py:
    from leadpoet_canonical.weights import bundle_weights_hash, normalize_to_u16
    from leadpoet_canonical.binding import verify_binding_message

    # In neurons/validator.py:
    from leadpoet_canonical.weights import bundle_weights_hash, normalize_to_u16

    # In neurons/auditor_validator.py:
    from leadpoet_canonical.weights import compare_weights_hash

    # In scripts/auditor_verifier.py:
    from leadpoet_canonical.weights import bundle_weights_hash, compare_weights_hash
    from leadpoet_canonical.events import verify_log_entry

Security Model:
    This module is designed for a TEE-based validation system where:
    - Gateway TEE signs all logs with an enclave-generated key
    - Primary Validator TEE signs weight submissions
    - Auditor Validators verify signatures and copy weights
    - External auditors can independently verify using transparency logs and chain data

    By centralizing all canonical implementations here, we ensure:
    1. Hash functions produce identical results across all components
    2. Signature verification is consistent
    3. No component can deviate from the canonical formats
"""

# Version of the canonical module
__version__ = "1.0.0"

# Import key constants for convenience (other modules should import from constants directly)
from leadpoet_canonical.constants import (
    EPOCH_LENGTH,
    WEIGHT_SUBMISSION_BLOCK,
    MAX_BLOCK_DRIFT,
    VERSION_KEY,
    DEFAULT_NETUID,
    AUDITOR_WEIGHT_TOLERANCE,
    TRUST_LEVEL_FULL_NITRO,
    TRUST_LEVEL_SIGNATURE_ONLY,
)

__all__ = [
    # Version
    "__version__",
    # Core constants
    "EPOCH_LENGTH",
    "WEIGHT_SUBMISSION_BLOCK",
    "MAX_BLOCK_DRIFT",
    "VERSION_KEY",
    "DEFAULT_NETUID",
    "AUDITOR_WEIGHT_TOLERANCE",
    "TRUST_LEVEL_FULL_NITRO",
    "TRUST_LEVEL_SIGNATURE_ONLY",
]

