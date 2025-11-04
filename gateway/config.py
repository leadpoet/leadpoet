"""
Gateway Configuration
====================

Loads all environment variables for the FastAPI gateway.

Environment variables should be set in .env file in project root.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")

# ============================================================
# AWS S3 (Primary Storage)
# ============================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "leadpoet-leads-primary")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "us-east-2")

if not AWS_ACCESS_KEY_ID:
    raise ValueError("AWS_ACCESS_KEY_ID environment variable is required")
if not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS_SECRET_ACCESS_KEY environment variable is required")

# ============================================================
# MinIO (Self-Hosted Mirror)
# ============================================================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "leadpoet-leads")

if not MINIO_ACCESS_KEY:
    raise ValueError("MINIO_ACCESS_KEY environment variable is required")
if not MINIO_SECRET_KEY:
    raise ValueError("MINIO_SECRET_KEY environment variable is required")

# ============================================================
# Bittensor Network
# ============================================================
BITTENSOR_NETWORK = os.getenv("BITTENSOR_NETWORK", "finney")
BITTENSOR_NETUID = int(os.getenv("BITTENSOR_NETUID", "71"))

# ============================================================
# Security Settings
# ============================================================
NONCE_EXPIRY_SECONDS = int(os.getenv("NONCE_EXPIRY_SECONDS", "300"))  # 5 minutes
TIMESTAMP_TOLERANCE_SECONDS = int(os.getenv("TIMESTAMP_TOLERANCE_SECONDS", "120"))  # ±2 minutes
PRESIGNED_URL_EXPIRY_SECONDS = int(os.getenv("PRESIGNED_URL_EXPIRY_SECONDS", "60"))  # 1 minute

# ============================================================
# Redis for Nonce Tracking (Optional - uses PostgreSQL if not set)
# ============================================================
REDIS_URL = os.getenv("REDIS_URL", None)

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
    
    # Check MinIO
    if not MINIO_ACCESS_KEY:
        errors.append("MINIO_ACCESS_KEY is not set")
    if not MINIO_SECRET_KEY:
        errors.append("MINIO_SECRET_KEY is not set")
    
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
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"MinIO Bucket: {MINIO_BUCKET}")
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

