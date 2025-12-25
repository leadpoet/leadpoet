"""
LinkedIn URL Normalization Utilities
=====================================

Functions for normalizing LinkedIn URLs to canonical form for duplicate detection.
Used to prevent miners from submitting the same person at the same company
with slightly different URL variations.

Gaming vectors handled:
- Protocol variations (http/https)
- www prefix
- Query parameters and fragments
- Trailing slashes
- Case variations
- URL encoding
- Subpaths (like /posts/, /about/, /details/)
"""

import re
import hashlib
from urllib.parse import unquote


def normalize_linkedin_url(url: str, url_type: str = "profile") -> str:
    """
    Normalize LinkedIn URL to canonical form for duplicate detection.
    
    Handles all gaming vectors:
    - Protocol variations (http/https)
    - www prefix
    - Query parameters and fragments
    - Trailing slashes
    - Case variations
    - URL encoding
    - Subpaths (like /posts/, /about/, /details/)
    
    Args:
        url: Raw LinkedIn URL (e.g., "https://www.linkedin.com/in/gavin-zaentz/")
        url_type: "profile" for personal (/in/), "company" for company pages (/company/)
    
    Returns:
        Canonical form: "linkedin.com/in/{slug}" or "linkedin.com/company/{slug}"
        Empty string if URL is invalid/not LinkedIn
    
    Examples:
        >>> normalize_linkedin_url("https://www.linkedin.com/in/gavin-zaentz/", "profile")
        "linkedin.com/in/gavin-zaentz"
        
        >>> normalize_linkedin_url("https://www.linkedin.com/company/leadpoet/posts/?feedView=all", "company")
        "linkedin.com/company/leadpoet"
    """
    if not url or not isinstance(url, str):
        return ""
    
    # URL decode first (handles %20, %C3%A9, etc.)
    try:
        url = unquote(url)
    except:
        pass
    
    # Strip whitespace and convert to lowercase
    url = url.strip().lower()
    
    # Remove protocol (http://, https://)
    url = re.sub(r'^https?://', '', url)
    
    # Remove www. prefix
    url = re.sub(r'^www\.', '', url)
    
    # Must start with linkedin.com
    if not url.startswith('linkedin.com'):
        return ""
    
    # Remove query params and fragments
    url = url.split('?')[0].split('#')[0]
    
    # Clean up multiple slashes and remove trailing slash
    url = re.sub(r'/+', '/', url)
    url = url.rstrip('/')
    
    # Extract slug based on type
    if url_type == "profile":
        # Pattern: linkedin.com/in/{slug}/...
        match = re.search(r'linkedin\.com/in/([^/]+)', url)
        if match:
            slug = match.group(1)
            return f"linkedin.com/in/{slug}"
    elif url_type == "company":
        # Pattern: linkedin.com/company/{slug}/...
        match = re.search(r'linkedin\.com/company/([^/]+)', url)
        if match:
            slug = match.group(1)
            return f"linkedin.com/company/{slug}"
    
    return ""  # Invalid or unrecognized format


def compute_linkedin_combo_hash(linkedin_url: str, company_linkedin_url: str) -> str:
    """
    Compute SHA256 hash of normalized linkedin + company_linkedin combination.
    
    Creates a unique identifier for "person X at company Y" to prevent
    duplicate submissions of the same person at the same company with
    different emails.
    
    Args:
        linkedin_url: Personal LinkedIn profile URL
        company_linkedin_url: Company LinkedIn page URL
    
    Returns:
        SHA256 hex digest, or empty string if either URL is invalid
    
    Security:
        - Uses || separator to prevent collision attacks
        - Normalized URLs cannot contain || so collisions are impossible
        - e.g., "linkedin.com/in/a||b" + "c" cannot equal "linkedin.com/in/a" + "b||c"
    
    Examples:
        >>> compute_linkedin_combo_hash(
        ...     "https://www.linkedin.com/in/gavin-zaentz/",
        ...     "https://www.linkedin.com/company/leadpoet/"
        ... )
        "abc123..."  # SHA256 of "linkedin.com/in/gavin-zaentz||linkedin.com/company/leadpoet"
    """
    normalized_profile = normalize_linkedin_url(linkedin_url, "profile")
    normalized_company = normalize_linkedin_url(company_linkedin_url, "company")
    
    # Both must be valid for a meaningful hash
    if not normalized_profile or not normalized_company:
        return ""
    
    # Combine with || separator to prevent collision attacks
    combined = f"{normalized_profile}||{normalized_company}"
    
    return hashlib.sha256(combined.encode()).hexdigest()

