"""
Gateway Configuration
====================

Loads all environment variables for the FastAPI gateway.

Environment variables should be set in .env file in project root.
"""

import os

# Load environment variables from .env file (if dotenv available)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed - environment variables must be set directly
    pass

# ============================================================
# Gateway Build Info (for reproducible builds)
# ============================================================
BUILD_ID = os.getenv("BUILD_ID", "dev-local")
GITHUB_COMMIT = os.getenv("GITHUB_SHA", "unknown")

# ============================================================
# Supabase PostgreSQL (Private DB + Transparency Log)
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # For client requests (not used by gateway)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Gateway uses this

# Warn if Supabase credentials missing (but don't block import for testing)
if not SUPABASE_URL:
    import warnings
    warnings.warn("SUPABASE_URL environment variable not set - Supabase mirroring will fail")
if not SUPABASE_SERVICE_ROLE_KEY:
    import warnings
    warnings.warn("SUPABASE_SERVICE_ROLE_KEY environment variable not set - Supabase mirroring will fail")

# ============================================================
# AWS S3 (Primary Storage)
# ============================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "leadpoet-leads-primary")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-2")

if not AWS_ACCESS_KEY_ID:
    import warnings
    warnings.warn("AWS_ACCESS_KEY_ID environment variable not set - S3 storage will fail")
if not AWS_SECRET_ACCESS_KEY:
    import warnings
    warnings.warn("AWS_SECRET_ACCESS_KEY environment variable not set - S3 storage will fail")


# ============================================================
# Bittensor Network
# ============================================================
BITTENSOR_NETWORK = os.getenv("BITTENSOR_NETWORK", "finney")
BITTENSOR_NETUID = int(os.getenv("BITTENSOR_NETUID", "71"))

# ============================================================
# Security Settings
# ============================================================
NONCE_EXPIRY_SECONDS = int(os.getenv("NONCE_EXPIRY_SECONDS", "300"))  # 5 minutes
TIMESTAMP_TOLERANCE_SECONDS = int(os.getenv("TIMESTAMP_TOLERANCE_SECONDS", "600"))  # ±10 minutes (allows for international network latency + miner retries)
PRESIGNED_URL_EXPIRY_SECONDS = int(os.getenv("PRESIGNED_URL_EXPIRY_SECONDS", "60"))  # 1 minute

# ============================================================
# Epoch Settings
# ============================================================
# Maximum leads assigned per epoch (default: 50, set to 10 for testing)
MAX_LEADS_PER_EPOCH = int(os.getenv("MAX_LEADS_PER_EPOCH", "50"))

# ============================================================
# Redis for Nonce Tracking (Optional - uses PostgreSQL if not set)
# ============================================================
REDIS_URL = os.getenv("REDIS_URL", None)

# ============================================================
# Gateway Ed25519 Keypair for Signed Receipts
# ============================================================
# Public key is hardcoded here for miners/validators to verify signatures
# Private key is stored encrypted in gateway/secrets/gateway_private_key.pem
# Password is in GATEWAY_PRIVATE_KEY_PASSWORD environment variable

GATEWAY_PUBLIC_KEY = os.getenv(
    "GATEWAY_PUBLIC_KEY",
    "312a4709ff6ab1aed727f336e8d79d3a7941fe7d234a3ec0391960ed471e3bd6"
)

# Warn if placeholder key is still being used
if GATEWAY_PUBLIC_KEY == "REPLACE_WITH_YOUR_GATEWAY_PUBLIC_KEY_HEX_STRING_64_CHARS":
    print("⚠️  WARNING: GATEWAY_PUBLIC_KEY is not set!")
    print("   Run scripts/generate_gateway_keypair.py to generate keypair")
    print("   Signed receipts will not work until key is configured")

# Private key password (loaded from environment)
GATEWAY_PRIVATE_KEY_PASSWORD = os.getenv("GATEWAY_PRIVATE_KEY_PASSWORD", None)
GATEWAY_PRIVATE_KEY_PATH = os.getenv(
    "GATEWAY_PRIVATE_KEY_PATH",
    "gateway/secrets/gateway_private_key.pem"
)

# ============================================================
# Arweave (Immutable Transparency Log - Primary Source of Truth)
# ============================================================
ARWEAVE_KEYFILE_PATH = os.getenv("ARWEAVE_KEYFILE_PATH", "secrets/arweave_keyfile.json")
ARWEAVE_GATEWAY_URL = os.getenv("ARWEAVE_GATEWAY_URL", "https://arweave.net")

# Note: Arweave keyfile validation is done lazily on first use
# to avoid blocking startup if Arweave is temporarily unavailable

# ============================================================
# Configuration Validation
# ============================================================

def validate_config():
    """
    Validates that all required configuration is present.
    Called on application startup.
    """
    errors = []
    
    # Check Supabase
    if not SUPABASE_URL:
        errors.append("SUPABASE_URL is not set")
    if not SUPABASE_SERVICE_ROLE_KEY:
        errors.append("SUPABASE_SERVICE_ROLE_KEY is not set")
    
    # Check AWS S3
    if not AWS_ACCESS_KEY_ID:
        errors.append("AWS_ACCESS_KEY_ID is not set")
    if not AWS_SECRET_ACCESS_KEY:
        errors.append("AWS_SECRET_ACCESS_KEY is not set")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def print_config_summary():
    """
    Prints a summary of the configuration (for debugging).
    NEVER prints secrets!
    """
    print("=" * 60)
    print("Gateway Configuration Summary")
    print("=" * 60)
    print(f"Build ID: {BUILD_ID}")
    print(f"GitHub Commit: {GITHUB_COMMIT}")
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"AWS S3 Bucket: {AWS_S3_BUCKET} ({AWS_S3_REGION})")
    print(f"Arweave Gateway: {ARWEAVE_GATEWAY_URL}")
    print(f"Arweave Keyfile: {ARWEAVE_KEYFILE_PATH}")
    print(f"Bittensor Network: {BITTENSOR_NETWORK} (netuid={BITTENSOR_NETUID})")
    print(f"Nonce Expiry: {NONCE_EXPIRY_SECONDS}s")
    print(f"Timestamp Tolerance: ±{TIMESTAMP_TOLERANCE_SECONDS}s")
    print(f"Presigned URL Expiry: {PRESIGNED_URL_EXPIRY_SECONDS}s")
    print(f"Redis: {'Enabled' if REDIS_URL else 'Disabled (using PostgreSQL)'}")
    print("=" * 60)


# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")
    print("⚠️  Some features may not work correctly.")

