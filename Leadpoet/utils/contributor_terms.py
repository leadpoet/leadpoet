"""
Leadpoet Contributor Terms and Attestation System

This module handles miner attestation to contributor terms at startup.
All miners must accept these terms before participating in the network.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import bittensor as bt


# Full Contributor Terms Text
CONTRIBUTOR_TERMS_TEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    LEADPOET CONTRIBUTOR TERMS OF SERVICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By proceeding, you confirm and agree to the following:

1. LAWFUL DATA COLLECTION
   You will only submit data obtained lawfully from public, first-party, or 
   licensed resale sources. You will not use paid databases (e.g., Apollo, 
   ZoomInfo, PDL, etc.) without resale agreements or breach any website's 
   terms of service.

2. OWNERSHIP & LICENSE GRANT
   You own or have the right to share all data you submit. You grant Leadpoet 
   an irrevocable, worldwide license to store, validate, sell, and distribute 
   that data to third parties.

3. ACCURACY & INTEGRITY
   You will submit accurate, non-fraudulent, and non-duplicative information.
   You will cooperate with audits or validation requests if needed.

4. RESTRICTED SOURCES
   You agree not to use any prohibited data brokers, private APIs, or scraped 
   contact databases without proper authorization. Restricted sources include 
   (but are not limited to):
   - ZoomInfo, Apollo.io, People Data Labs
   - RocketReach, Hunter.io, Snov.io
   - Lusha, Clearbit, LeadIQ
   
   You may only use such sources if you have a valid resale agreement and 
   provide proof via license document hash.

5. RESALE RIGHTS
   You acknowledge that Leadpoet and its buyers may resell, enrich, and 
   redistribute approved leads.

6. COMPLIANCE & TAKEDOWNS
   If a takedown request or legal issue arises from your submission, Leadpoet 
   may remove your data and freeze all future earned rewards. You agree to 
   respond to compliance inquiries within 10 business days.

7. INDEMNIFICATION
   You accept responsibility for your submissions. If your data causes a legal 
   claim against Leadpoet, you agree to indemnify Leadpoet against losses or 
   damages.

8. TERMS VERSION TRACKING
   Your agreement will be tied to a specific terms_version_hash for audit 
   purposes. If terms are updated, you must re-accept them before mining 
   resumes.

9. PRIVACY
   No personal KYC data is collected; your wallet address is your identity.
   Leadpoet only logs your wallet, timestamp, and accepted version hash for 
   compliance.

10. TERMINATION
    Leadpoet may suspend contributors who violate these terms or who fail 
    audits. Suspended miners will not receive rewards for any pending leads.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full Terms: https://leadpoet.com/contributor-terms

BY TYPING 'Y' BELOW, YOU ACCEPT THESE TERMS AND CONDITIONS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Generate SHA-256 hash of canonical terms for version tracking
TERMS_VERSION_HASH = hashlib.sha256(CONTRIBUTOR_TERMS_TEXT.encode('utf-8')).hexdigest()


def display_terms_prompt():
    """
    Display the full contributor terms to the terminal.
    Should be called on first run or when terms are updated.
    """
    print(CONTRIBUTOR_TERMS_TEXT)
    print(f"\n📋 Terms Version Hash: {TERMS_VERSION_HASH[:16]}...")
    print(f"📅 Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")


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

