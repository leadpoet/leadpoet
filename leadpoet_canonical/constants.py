"""
LeadPoet Canonical Constants

This module is the SINGLE SOURCE OF TRUTH for all constants used across
gateway, validator, and auditor components.

ALL components MUST import from this module. Do NOT redefine these values elsewhere.

Security Note: These constants define the timing and validation rules for the
TEE-based weight submission system. Changing these values affects the security
model and must be coordinated across all deployments.
"""

# =============================================================================
# EPOCH CONFIGURATION
# =============================================================================

# Number of blocks per epoch
# This matches Bittensor's tempo setting for the subnet
EPOCH_LENGTH = 360

# Block within epoch when weight submission occurs
# Primary validator should submit weights at this block
WEIGHT_SUBMISSION_BLOCK = 345

# Maximum allowed drift between gateway-observed block and submission block
# Submissions with larger drift are rejected to prevent replay attacks
MAX_BLOCK_DRIFT = 30

# How early before WEIGHT_SUBMISSION_BLOCK submissions are allowed
# Submissions before (WEIGHT_SUBMISSION_BLOCK - EARLY_SUBMISSION_WINDOW) are rejected
EARLY_SUBMISSION_WINDOW = 15


# =============================================================================
# BITTENSOR CONFIGURATION
# =============================================================================

# Version key for convert_weights_and_uids_for_emit()
# CRITICAL: This MUST match your pinned bittensor version's expected value
# Different bittensor versions may have different version_key semantics
VERSION_KEY = 0

# Default netuid for LeadPoet subnet
# Used as default in CLI tools, but should be overridden via arguments
DEFAULT_NETUID = 71


# =============================================================================
# CRYPTOGRAPHIC CONFIGURATION
# =============================================================================

# Ed25519 signature length in bytes
ED25519_SIGNATURE_LENGTH = 64

# Ed25519 public key length in bytes
ED25519_PUBKEY_LENGTH = 32

# SHA256 hash length in bytes
SHA256_HASH_LENGTH = 32


# =============================================================================
# WEIGHT TOLERANCE
# =============================================================================

# Maximum per-weight drift allowed for auditor comparisons (Issue 17)
# u16 → float → u16 round-trip may cause ±1 differences
# This tolerance prevents false "AUDITOR_MISMATCH" events
AUDITOR_WEIGHT_TOLERANCE = 1


# =============================================================================
# BINDING MESSAGE
# =============================================================================

# Prefix for validator binding messages (domain separation)
BINDING_MESSAGE_PREFIX = "LEADPOET_VALIDATOR_BINDING"

# Current binding message version
BINDING_MESSAGE_VERSION = "1"


# =============================================================================
# EVENT TYPES
# =============================================================================

# Gateway event types (logged to transparency_log)
EVENT_TYPE_WEIGHT_SUBMISSION = "WEIGHT_SUBMISSION"
EVENT_TYPE_WEIGHT_SUBMISSION_REJECTED = "WEIGHT_SUBMISSION_REJECTED_DUPLICATE"
EVENT_TYPE_ENCLAVE_RESTART = "ENCLAVE_RESTART"
EVENT_TYPE_EPOCH_AUDIT = "EPOCH_AUDIT"
EVENT_TYPE_ARWEAVE_CHECKPOINT = "ARWEAVE_CHECKPOINT"
EVENT_TYPE_GATEWAY_ATTESTATION = "GATEWAY_ATTESTATION"


# =============================================================================
# AUDIT STATUS VALUES
# =============================================================================

AUDIT_STATUS_VERIFIED = "VERIFIED"
AUDIT_STATUS_EQUIVOCATION_DETECTED = "EQUIVOCATION_DETECTED"
AUDIT_STATUS_AUDITOR_MISMATCH = "AUDITOR_MISMATCH"
AUDIT_STATUS_NO_TEE_BUNDLE = "NO_TEE_BUNDLE"
AUDIT_STATUS_NO_PRIMARY_WEIGHTS = "NO_PRIMARY_WEIGHTS"


# =============================================================================
# TRUST LEVELS
# =============================================================================

# Full Nitro attestation verification completed
TRUST_LEVEL_FULL_NITRO = "full_nitro"

# Only Ed25519 signatures verified (weaker trust model)
TRUST_LEVEL_SIGNATURE_ONLY = "signature_only"

