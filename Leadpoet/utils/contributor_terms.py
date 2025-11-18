"""
Leadpoet Contributor Terms and Attestation System

This module handles miner attestation to contributor terms at startup.
All miners must accept these terms before participating in the network.

This is the FULL version with all miner/validator functions.
The gateway uses a minimal version at gateway/utils/contributor_terms.py.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import bittensor as bt


# ============================================================================
# CANONICAL TERMS SOURCE: GitHub Repository (Single Source of Truth)
# ============================================================================
# The official contributor terms are maintained in the GitHub repository.
# This ensures all participants (miners, validators, gateway) use the same version.
# Any updates to terms are made via GitHub commits, creating an immutable audit trail.

GITHUB_TERMS_URL = "https://cdn.jsdelivr.net/gh/leadpoet/leadpoet@main/docs/contributor_terms.md"

# Cache for terms content (avoid repeated network calls)
_CACHED_TERMS_TEXT = None
_CACHED_TERMS_HASH = None


def fetch_contributor_terms_from_github() -> Tuple[str, str]:
    """
    Fetch the canonical contributor terms from GitHub repository.
    
    Returns:
        Tuple of (terms_text, terms_hash)
        
    Raises:
        Exception if unable to fetch terms from GitHub
        
    Security:
        - Uses raw.githubusercontent.com (content-addressed)
        - Verifies content is non-empty
        - Creates SHA-256 hash for version tracking
    """
    global _CACHED_TERMS_TEXT, _CACHED_TERMS_HASH
    
    # Return cached version if available
    if _CACHED_TERMS_TEXT and _CACHED_TERMS_HASH:
        return _CACHED_TERMS_TEXT, _CACHED_TERMS_HASH
    
    try:
        import requests
        
        bt.logging.debug(f"📥 Fetching contributor terms from GitHub: {GITHUB_TERMS_URL}")
        
        response = requests.get(GITHUB_TERMS_URL, timeout=10)
        response.raise_for_status()
        
        terms_text = response.text
        
        if not terms_text or len(terms_text) < 100:
            raise Exception("Fetched terms appear invalid (too short)")
        
        # Generate SHA-256 hash of canonical terms
        terms_hash = hashlib.sha256(terms_text.encode('utf-8')).hexdigest()
        
        # Cache for future use
        _CACHED_TERMS_TEXT = terms_text
        _CACHED_TERMS_HASH = terms_hash
        
        bt.logging.debug(f"✅ Fetched contributor terms from GitHub")
        bt.logging.debug(f"   Length: {len(terms_text)} characters")
        bt.logging.debug(f"   Hash: {terms_hash[:16]}...")
        
        return terms_text, terms_hash
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch contributor terms from GitHub: {e}")
    except Exception as e:
        raise Exception(f"Error processing contributor terms: {e}")


def get_contributor_terms_text() -> str:
    """Get the canonical contributor terms text from GitHub."""
    terms_text, _ = fetch_contributor_terms_from_github()
    return terms_text


def get_terms_version_hash() -> str:
    """Get the SHA-256 hash of the canonical contributor terms from GitHub."""
    _, terms_hash = fetch_contributor_terms_from_github()
    return terms_hash


# For backward compatibility, initialize these on module load
# They cache after first fetch to avoid repeated network calls
try:
    CONTRIBUTOR_TERMS_TEXT = get_contributor_terms_text()
    TERMS_VERSION_HASH = get_terms_version_hash()
except Exception as e:
    # If GitHub is unavailable during import, log warning
    # Functions that need terms should call fetch_contributor_terms_from_github() directly
    import sys
    print(f"⚠️  Warning: Could not fetch terms from GitHub during module import: {e}", file=sys.stderr)
    CONTRIBUTOR_TERMS_TEXT = None
    TERMS_VERSION_HASH = None


def display_terms_prompt():
    """
    Display the full contributor terms to the terminal.
    Should be called on first run or when terms are updated.
    
    Fetches the latest terms from GitHub to ensure miners see current version.
    """
    try:
        terms_text, terms_hash = fetch_contributor_terms_from_github()
        print(terms_text)
        print(f"\n📋 Terms Version Hash: {terms_hash[:16]}...")
        print(f"📅 Fetched from GitHub: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"🔗 Source: {GITHUB_TERMS_URL}\n")
    except Exception as e:
        bt.logging.error(f"❌ Failed to fetch terms from GitHub: {e}")
        bt.logging.error(f"   Cannot display terms - miner startup aborted")
        raise


def verify_attestation(attestation_file: Path, current_hash: str) -> tuple[bool, str]:
    """
    Verify that an existing attestation file is valid and up-to-date.
    
    Args:
        attestation_file: Path to the attestation JSON file
        current_hash: Current terms version hash to verify against
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not attestation_file.exists():
        return False, "Attestation file does not exist"
    
    try:
        with open(attestation_file, 'r') as f:
            attestation = json.load(f)
        
        stored_hash = attestation.get("terms_version_hash")
        if stored_hash != current_hash:
            return False, f"Terms have been updated (stored: {stored_hash[:8]}, current: {current_hash[:8]})"
        
        if not attestation.get("accepted"):
            return False, "Terms not accepted in stored attestation"
        
        return True, "Attestation valid"
        
    except json.JSONDecodeError:
        return False, "Attestation file is corrupted"
    except Exception as e:
        return False, f"Error reading attestation: {str(e)}"


def get_public_ip() -> str:
    """
    Get the public IP address of this machine (optional).
    Returns empty string if unable to determine.
    """
    try:
        import socket
        # Try to connect to a public DNS server to determine our external IP
        # This doesn't actually send data, just determines routing
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ""


def create_attestation_record(wallet_address: str, terms_hash: str) -> dict:
    """
    Create a new attestation record for storage.
    
    Args:
        wallet_address: The miner's wallet hotkey SS58 address
        terms_hash: The terms version hash being accepted
        
    Returns:
        Dictionary containing attestation data
    """
    return {
        "wallet_ss58": wallet_address,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "terms_version_hash": terms_hash,
        "accepted": True,
        "ip_address": get_public_ip(),
    }


def save_attestation(attestation_data: dict, attestation_file: Path):
    """
    Save attestation data to local JSON file.
    
    Args:
        attestation_data: Dictionary containing attestation information
        attestation_file: Path where attestation should be saved
    """
    # Ensure parent directory exists
    attestation_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(attestation_file, 'w') as f:
        json.dump(attestation_data, f, indent=2)


def sync_attestation_to_supabase(attestation_data: dict, token_manager=None) -> bool:
    """
    Sync attestation to Supabase contributor_attestations table (SOURCE OF TRUTH).
    
    This is CRITICAL for security - prevents miners from manipulating local attestation files.
    Validators will query this table to verify that miners have legitimately accepted terms.
    
    Args:
        attestation_data: Dictionary containing attestation information
        token_manager: TokenManager instance for JWT authentication (required)
        
    Returns:
        True if sync succeeded, False otherwise
        
    Security:
        - Requires valid JWT token from TokenManager
        - Uses RLS policies (miners can only insert their own attestations)
        - Creates source of truth that cannot be manipulated locally
    """
    try:
        if not token_manager:
            bt.logging.error("❌ TokenManager required for attestation sync")
            return False
        
        # Get valid JWT token
        jwt_token = token_manager.get_token()
        if not jwt_token:
            bt.logging.error("❌ Failed to get valid JWT token for attestation sync")
            bt.logging.error("   Hint: Ensure your wallet is registered and network connection is available")
            return False
        
        # DEBUG: Decode JWT to see claims
        import base64
        try:
            # JWT format: header.payload.signature
            parts = jwt_token.split('.')
            if len(parts) == 3:
                # Decode payload (add padding if needed)
                payload = parts[1]
                payload += '=' * (4 - len(payload) % 4)  # Add padding
                decoded = base64.urlsafe_b64decode(payload)
                print(f"   🔍 JWT claims: {decoded.decode('utf-8')}")
        except Exception as decode_err:
            print(f"   ⚠️ Could not decode JWT for inspection: {decode_err}")
        
        bt.logging.debug(f"🔐 Syncing attestation to Supabase for wallet {attestation_data.get('wallet_ss58', 'unknown')[:10]}...")
        
        # Import CustomSupabaseClient for proper JWT authentication
        from Leadpoet.utils.cloud_db import CustomSupabaseClient
        import os
        
        SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
        SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8")
        
        # Create CustomSupabaseClient with JWT token (uses direct HTTP requests with proper auth)
        supabase = CustomSupabaseClient(SUPABASE_URL, jwt_token, SUPABASE_ANON_KEY)
        
        # Prepare data for insertion
        record = {
            "wallet_ss58": attestation_data.get("wallet_ss58"),
            "terms_version_hash": attestation_data.get("terms_version_hash"),
            "accepted": attestation_data.get("accepted", True),
            "timestamp_utc": attestation_data.get("timestamp_utc"),
            "ip_address": attestation_data.get("ip_address"),
        }
        
        # Add updated_at if present (for re-acceptance)
        if "updated_at" in attestation_data:
            record["updated_at"] = attestation_data["updated_at"]
        
        bt.logging.debug(f"   Inserting attestation record: {record['wallet_ss58'][:10]}... @ {record['terms_version_hash'][:8]}...")
        
        # Insert or update (upsert) the attestation
        # This uses the unique constraint on (wallet_ss58, terms_version_hash)
        # Note: CustomSupabaseClient.upsert() already executes the request, no need for .execute()
        result = supabase.table("contributor_attestations")\
            .upsert(record, on_conflict="wallet_ss58,terms_version_hash")
        
        if result.data:
            bt.logging.info(f"✅ Attestation synced to Supabase (SOURCE OF TRUTH)")
            bt.logging.debug(f"   Record ID: {result.data[0].get('id', 'unknown')}")
            return True
        else:
            bt.logging.error(f"❌ Failed to sync attestation - no data returned")
            return False
            
    except Exception as e:
        bt.logging.error(f"❌ Failed to sync attestation to Supabase: {e}")
        bt.logging.error(f"   This is a security-critical operation - miner cannot proceed")
        
        # Print the actual data being sent for debugging
        print(f"   🔍 Data attempted to insert: {record}")
        
        # Try to get more error details from response
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"   🔍 Server response: {e.response.text}")
        
        import traceback
        bt.logging.debug(traceback.format_exc())
        return False

