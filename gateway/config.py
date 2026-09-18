"""
Gateway Configuration
====================

Loads all environment variables for the FastAPI gateway.

Environment variables should be set in .env file in project root.
"""

import os
import re
import sys
from pathlib import Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AWS_STATIC_CREDENTIAL_KEYS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
}


def _instance_role_only() -> bool:
    return os.getenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "false").lower() == "true"


def _load_gateway_env_file(path: Path) -> None:
    """Load newline- or NUL-separated KEY=VALUE entries without overriding env."""
    if not path.is_file():
        return
    try:
        data = path.read_bytes()
    except OSError as exc:
        import warnings

        warnings.warn(f"Gateway env file unreadable at {path}: {exc}")
        return

    loaded = 0
    for raw in re.split(rb"[\n\0]+", data):
        line = raw.strip()
        if not line or line.startswith(b"#") or b"=" not in line:
            continue
        key_raw, value_raw = line.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="ignore").strip()
        if not _ENV_KEY_RE.fullmatch(key):
            continue
        if _instance_role_only() and key in _AWS_STATIC_CREDENTIAL_KEYS:
            continue
        if key in os.environ:
            continue
        value = value_raw.decode("utf-8", errors="ignore").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded += 1

    if loaded:
        print(
            f"Loaded {loaded} fallback env var(s) from {path}",
            file=sys.stderr,
            flush=True,
        )


_gateway_env_file = os.getenv("GATEWAY_ENV_FILE", "/home/ec2-user/.config/leadpoet/gateway.env")
_load_gateway_env_file(Path(_gateway_env_file).expanduser())

# The canonical gateway cache must precede a checkout-local developer .env.
# Explicit process environment still wins because both loaders are non-overriding.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed - environment variables must be set directly
    pass

if _instance_role_only():
    for _aws_env_key in _AWS_STATIC_CREDENTIAL_KEYS:
        os.environ.pop(_aws_env_key, None)

if os.getenv("AWS_PROFILE") and os.getenv("LEADPOET_AWS_PROFILE_OVERRIDES_ENV_KEYS", "true").lower() == "true":
    # Local/testnet operators often use a named AWS profile while .env still
    # contains older static keys. Boto3 gives env keys precedence over profiles,
    # so clear static AWS creds when an explicit profile is selected.
    for _aws_env_key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        os.environ.pop(_aws_env_key, None)

# ============================================================
# Gateway Build Info (for reproducible builds)
# ============================================================
# Import after dotenv/fallback env loading so env-provided build metadata wins.
from gateway.build_info import get_build_info

BUILD_INFO = get_build_info()
BUILD_ID = str(BUILD_INFO.get("build_id") or "unknown")
GITHUB_COMMIT = str(BUILD_INFO.get("git_commit") or "unknown")

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
# Bittensor Network
# ============================================================
BITTENSOR_NETWORK = os.getenv("BITTENSOR_NETWORK", "finney")
BITTENSOR_NETUID = int(os.getenv("BITTENSOR_NETUID", "71"))

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
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")
    print("⚠️  Some features may not work correctly.")
