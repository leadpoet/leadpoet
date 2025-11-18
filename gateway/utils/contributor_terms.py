"""
Leadpoet Contributor Terms - Gateway Module

🔒 SECURITY: This file is inside gateway/ and included in TEE code hash verification.
Any modifications to terms verification logic will be detected by miners/validators.

This is a MINIMAL version containing ONLY what the gateway needs to verify submissions.
The full version with miner/validator functions is in Leadpoet/utils/contributor_terms.py.
"""

import hashlib
from typing import Tuple


# ============================================================================
# CANONICAL TERMS SOURCE: GitHub Repository (Single Source of Truth)
# ============================================================================
# The official contributor terms are maintained in the GitHub repository.
# This ensures all participants (miners, validators, gateway) use the same version.
# Any updates to terms are made via GitHub commits, creating an immutable audit trail.

GITHUB_TERMS_URL = "https://cdn.jsdelivr.net/gh/leadpoet/leadpoet@main/docs/contributor_terms.md"

# Cache for terms hash (avoid repeated network calls)
_CACHED_TERMS_HASH = None


def get_terms_version_hash() -> str:
    """
    Get the SHA-256 hash of the canonical contributor terms from GitHub.
    
    This is the ONLY function the gateway needs. It fetches the current terms
    from GitHub and computes their hash to verify miner submissions.
    
    Returns:
        SHA-256 hash of the current terms (hex string)
        
    Raises:
        Exception if unable to fetch terms from GitHub
    """
    global _CACHED_TERMS_HASH
    
    # Return cached version if available
    if _CACHED_TERMS_HASH:
        return _CACHED_TERMS_HASH
    
    try:
        import requests
        
        print(f"📥 Gateway: Fetching contributor terms from GitHub: {GITHUB_TERMS_URL}")
        
        response = requests.get(GITHUB_TERMS_URL, timeout=10)
        response.raise_for_status()
        
        terms_text = response.text
        
        if not terms_text or len(terms_text) < 100:
            raise Exception("Fetched terms appear invalid (too short)")
        
        # Generate SHA-256 hash of canonical terms
        terms_hash = hashlib.sha256(terms_text.encode('utf-8')).hexdigest()
        
        # Cache for future use
        _CACHED_TERMS_HASH = terms_hash
        
        print(f"✅ Gateway: Fetched contributor terms hash from GitHub")
        print(f"   Hash: {terms_hash[:16]}...")
        
        return terms_hash
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch contributor terms from GitHub: {e}")
    except Exception as e:
        raise Exception(f"Error processing contributor terms: {e}")


# Initialize hash on module load (for gateway startup verification)
try:
    TERMS_VERSION_HASH = get_terms_version_hash()
except Exception as e:
    import sys
    print(f"⚠️  Warning: Could not fetch terms hash from GitHub during gateway startup: {e}", file=sys.stderr)
    TERMS_VERSION_HASH = None
