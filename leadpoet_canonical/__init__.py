"""Shared deterministic hashing, signing, chain, and reward primitives.

Normal Arena validators construct weights from accepted Arena reward state.
The protected signer verifies the same arithmetic and constrains the final
transaction. Generic gateway qualification and artifact services use their
own shared verification primitives from this package.
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
