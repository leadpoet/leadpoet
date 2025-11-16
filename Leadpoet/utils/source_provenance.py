"""
Source Provenance Validation Module

Validates lead sources against regulatory requirements:
- Checks against restricted data broker denylist
- Validates domain age (≥7 days)
- Verifies URL reachability
- Categorizes source types for compliance tracking
"""

import aiohttp
import bittensor as bt
import hashlib
import re
from typing import Tuple
from datetime import datetime, timezone


# Restricted sources denylist - prohibited data brokers and APIs
RESTRICTED_SOURCES = [
    "zoominfo.com",
    "apollo.io",
    "people-data-labs.com",
    "peopledatalabs.com",
    "rocketreach.co",
    "hunter.io",
    "snov.io",
    "lusha.com",
    "clearbit.com",
    "leadiq.com",
]

DENYLIST_VERSION = "v1.0"


def is_restricted_source(domain: str) -> bool:
    """
    Check if domain is in restricted sources denylist.
    
    Args:
        domain: Domain name to check (e.g., "example.com")
        
    Returns:
        True if domain should be blocked, False otherwise
        
    Security:
        Prevents miners from using prohibited data brokers without proper
        resale agreements. Validators will reject leads from these sources
        unless license_doc_hash is provided.
    """
    if not domain:
        return False
    
    domain_lower = domain.lower().strip()
    
    # Remove www. prefix
    if domain_lower.startswith("www."):
        domain_lower = domain_lower[4:]
    
    # Check exact match
    if domain_lower in RESTRICTED_SOURCES:
        return True
    
    # Check if any restricted domain is contained (subdomain check)
    for restricted in RESTRICTED_SOURCES:
        if restricted in domain_lower:
            return True
    
    return False


def get_denylist_info() -> dict:
    """
    Return denylist metadata for audit logs.
    
    Returns:
        Dictionary with version, count, and list of restricted sources
    """
    return {
        "version": DENYLIST_VERSION,
        "count": len(RESTRICTED_SOURCES),
        "sources": RESTRICTED_SOURCES
    }


async def validate_source_url(url: str, source_type: str) -> Tuple[bool, str]:
    """
    Validate source URL against regulatory requirements.
    
    Performs three checks:
    1. Domain not in denylist
    2. Domain age ≥ 7 days
    3. URL is reachable (HTTP HEAD = 200)
    
    Args:
        url: Source URL to validate (REQUIRED)
        source_type: Source type to verify against url - REQUIRED for security validation
        
    Returns:
        Tuple of (is_valid, reason)
        - is_valid: True if all checks pass, False otherwise
        - reason: Explanation of validation result
        
    Security:
        This prevents miners from submitting leads from:
        - Restricted data brokers (without license)
        - Newly created/suspicious domains
        - Unreachable/fake URLs
        - Spoofing proprietary_database to bypass validation
    """
    if not url:
        return False, "No source URL provided"
    
    if not source_type:
        return False, "No source_type provided (required for validation)"
    
    # Special case: Only proprietary databases can skip URL validation
    # SECURITY: Both source_url AND source_type must be "proprietary_database"
    # This prevents miners from bypassing validation by just setting url="proprietary_database"
    if url.lower() == "proprietary_database":
        if source_type.lower() == "proprietary_database":
            return True, "Valid source type: proprietary_database"
        else:
            return False, f"Security violation: source_url is 'proprietary_database' but source_type is '{source_type}'. Both must match."
    
    # Normalize URL: Add https:// if no protocol specified
    # This handles cases like "tether.to" → "https://tether.to"
    from urllib.parse import urlparse
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    # Parse domain from URL
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        
        # Clean up domain (remove port, etc.)
        if ':' in domain:
            domain = domain.split(':')[0]
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"
    
    if not domain:
        return False, "Could not extract domain from URL"
    
    # Check 1: Denylist validation
    if is_restricted_source(domain):
        return False, f"Domain {domain} is in restricted source denylist"
    
    # Check 2: Domain age validation (≥7 days)
    # Import domain age check from validator automated_checks
    try:
        from validator_models.automated_checks import check_domain_age
        
        # Create mock lead object for domain age check
        lead_mock = {"website": url}
        age_valid, age_reason = await check_domain_age(lead_mock)
        
        if not age_valid:
            return False, age_reason
    except ImportError:
        # If check_domain_age not available, log warning but don't fail
        bt.logging.warning("Domain age check not available - skipping")
    except Exception as e:
        bt.logging.debug(f"Domain age check failed: {e}")
        # Don't fail validation if age check has issues
        pass
    
    # Check 3: URL reachability
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as response:
                if response.status in [200, 301, 302, 303, 307, 308]:
                    return True, "Source URL validated"
                else:
                    return False, f"Source URL returned status {response.status}"
    except aiohttp.ClientError as e:
        return False, f"Source URL unreachable: {str(e)}"
    except Exception as e:
        return False, f"Error validating source URL: {str(e)}"


def determine_source_type(url: str, lead: dict) -> str:
    """
    Determine source type based on URL and lead metadata.
    
    Source types (in priority order):
    1. licensed_resale - Has license_doc_hash (data broker with resale agreement)
    2. proprietary_database - Has proprietary_database_name
    3. first_party_form - URL contains "contact" or "form"
    4. public_registry - LinkedIn, Crunchbase, Companies House, .gov
    5. company_site - Default (direct company website)
    
    Args:
        url: Source URL
        lead: Lead data dictionary (may contain license_doc_hash, etc.)
        
    Returns:
        Source type string: "public_registry", "company_site", "first_party_form",
        "licensed_resale", or "proprietary_database"
        
    Usage:
        This categorization helps validators understand the legitimacy and
        compliance status of each lead source.
    """
    if not url:
        return "company_site"  # Default
    
    url_lower = url.lower()
    
    # Priority 1: Check for licensed resale indicators
    if lead.get("license_doc_hash"):
        return "licensed_resale"
    
    # Priority 2: Check for proprietary database indicators
    if lead.get("proprietary_database_name"):
        return "proprietary_database"
    
    # Priority 3: Check for first-party form indicators
    if "contact" in url_lower or "form" in url_lower or "/submit" in url_lower:
        return "first_party_form"
    
    # Priority 4: Check for public registry indicators
    registry_domains = [
        "linkedin.com",
        "crunchbase.com",
        "companieshouse.gov.uk",
        ".gov",
        "sec.gov",
        "data.gov",
        "opencorporates.com",
    ]
    
    for registry in registry_domains:
        if registry in url_lower:
            return "public_registry"
    
    # Priority 5: Default to company site
    return "company_site"


def extract_domain_from_url(url: str) -> str:
    """
    Extract clean domain from URL.
    
    Args:
        url: Full URL string
        
    Returns:
        Clean domain name (e.g., "example.com")
    """
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        
        # Remove port
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        
        return domain.lower()
    except Exception:
        return ""


def validate_licensed_resale(lead: dict) -> Tuple[bool, str]:
    """
    Validate licensed resale submission.
    
    For leads with source_type = "licensed_resale", this validates that:
    1. license_doc_hash is present
    2. license_doc_hash is valid SHA-256 format (64 hex characters)
    
    Args:
        lead: Lead data dictionary
        
    Returns:
        Tuple of (is_valid, reason)
        - is_valid: True if validation passes (or not applicable), False if fails
        - reason: Explanation of validation result
        
    Usage:
        Miners using restricted data brokers (ZoomInfo, Apollo, etc.) can still
        submit leads IF they have a valid resale agreement. They must provide
        the SHA-256 hash of their license document for verification.
    """
    if lead.get("source_type") != "licensed_resale":
        return True, "Not a licensed resale submission"
    
    # Require license_doc_hash for resale submissions
    license_hash = lead.get("license_doc_hash", "")
    if not license_hash:
        return False, "Licensed resale requires license_doc_hash"
    
    # Validate hash format (SHA-256 = 64 hex chars)
    if not re.match(r'^[a-fA-F0-9]{64}$', license_hash):
        return False, "Invalid license_doc_hash format (must be SHA-256: 64 hex characters)"
    
    return True, "Licensed resale validated"


def generate_license_doc_hash(file_path: str) -> str:
    """
    Generate SHA-256 hash of license document.
    
    Miners call this to get the hash for their resale agreement document.
    The hash is then included in lead submissions to prove they have
    authorization to resell data from restricted sources.
    
    Args:
        file_path: Path to the license document file (PDF, image, etc.)
        
    Returns:
        SHA-256 hash as 64-character hexadecimal string
        
    Example Usage:
        ```python
        from Leadpoet.utils.source_provenance import generate_license_doc_hash
        
        # One-time: Generate hash of your license agreement
        license_hash = generate_license_doc_hash("path/to/resale_agreement.pdf")
        print(f"Your license hash: {license_hash}")
        
        # Add to each lead from licensed source
        lead["source_type"] = "licensed_resale"
        lead["license_doc_hash"] = license_hash
        lead["license_doc_url"] = "https://yourdomain.com/license-proof.pdf"  # Optional
        ```
        
    Security:
        The hash creates a cryptographic proof that the miner has the document.
        Validators can verify the hash matches if the document is shared.
    """
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        bt.logging.info(f"✅ Generated license document hash: {file_hash[:16]}...")
        return file_hash
        
    except FileNotFoundError:
        bt.logging.error(f"❌ License document not found: {file_path}")
        raise
    except Exception as e:
        bt.logging.error(f"❌ Error generating license hash: {e}")
        raise

