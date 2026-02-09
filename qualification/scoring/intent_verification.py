"""
Qualification System: Intent Signal Verification

Phase 5.1 from tasks10.md

This module implements intent signal verification for the Lead Qualification
Agent competition. It verifies that intent signals claimed by models are
real and supported by the source content.

Verification Flow:
1. Check cache for existing verification result
2. Fetch content from URL (using appropriate method per source)
3. Extract relevant text from HTML
4. Use LLM to verify claim matches content
5. Cache result for future lookups

Supported Sources:
- LinkedIn (profiles, posts) via ScrapingDog
- Job boards via ScrapingDog
- GitHub via public API
- News sites via ScrapingDog
- Company websites via ScrapingDog
- Social media via ScrapingDog

Note: ScrapingDog handles its own proxy rotation internally.
No external proxies (like Webshare) are needed for benchmarks.

CRITICAL: This is NEW intent verification logic for qualification only.
Do NOT modify any existing verification code in validator_models/ or
lead verification scripts.
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, NamedTuple

import httpx

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not installed - HTML parsing will be limited")

from gateway.qualification.models import IntentSignal, IntentSignalSource

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# API Keys (from environment)
# SECURITY: Qualification uses SEPARATE API keys with limited funds.
# If a malicious miner somehow extracts keys, they only get the
# qualification keys (limited budget), not the main sourcing keys.
#
# TODO: After beta release, change back to "SCRAPINGDOG_API_KEY" (shared with sourcing)
SCRAPINGDOG_API_KEY = os.getenv("QUALIFICATION_SCRAPINGDOG_API_KEY", "")
# TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
OPENROUTER_API_KEY = os.getenv("QUALIFICATION_OPENROUTER_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Request timeouts
DEFAULT_TIMEOUT = 15.0
LLM_TIMEOUT = 30.0

# Verification thresholds
CONFIDENCE_THRESHOLD = 70  # Minimum confidence to consider verified
CONTENT_MAX_LENGTH = 5000  # Max chars to send to LLM

# Cache TTL
DEFAULT_CACHE_TTL_DAYS = 7


# =============================================================================
# Types
# =============================================================================

class VerificationResult(NamedTuple):
    """Result of intent signal verification."""
    verified: bool
    confidence: int  # 0-100
    reason: str


class CachedVerification(NamedTuple):
    """Cached verification result."""
    cache_key: str
    url: str
    source: str
    signal_date: str
    verification_result: bool
    verification_confidence: int
    verification_reason: str
    verified_at: datetime
    expires_at: datetime


# =============================================================================
# In-Memory Cache (for fast lookups)
# =============================================================================

# Simple in-memory cache - in production, use qualification_intent_cache table
_verification_cache: Dict[str, CachedVerification] = {}


def compute_cache_key(url: str, source: str, signal_date: str) -> str:
    """
    Compute cache key for a verification request.
    
    Uses URL + source + date to ensure unique caching per signal.
    
    Args:
        url: The source URL
        source: Source type (linkedin, job_board, etc.)
        signal_date: Date of the signal (ISO format)
    
    Returns:
        SHA256 hash as cache key
    """
    key_data = f"{url.lower().strip()}|{source.lower()}|{signal_date}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


async def get_cached_verification(cache_key: str) -> Optional[CachedVerification]:
    """
    Get cached verification result if available and not expired.
    
    Args:
        cache_key: The cache key to look up
    
    Returns:
        CachedVerification if found and valid, None otherwise
    """
    cached = _verification_cache.get(cache_key)
    
    if cached:
        # Check if expired
        if datetime.now(timezone.utc) < cached.expires_at:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return cached
        else:
            # Remove expired entry
            del _verification_cache[cache_key]
            logger.debug(f"Cache expired for key: {cache_key[:8]}...")
    
    return None


async def cache_verification(
    cache_key: str,
    url: str,
    source: str,
    signal_date: str,
    verification_result: bool,
    verification_confidence: int,
    verification_reason: str,
    ttl_days: int = DEFAULT_CACHE_TTL_DAYS
):
    """
    Cache a verification result.
    
    Args:
        cache_key: The cache key
        url: Source URL
        source: Source type
        signal_date: Signal date
        verification_result: Whether verified
        verification_confidence: Confidence score (0-100)
        verification_reason: Explanation
        ttl_days: Cache TTL in days
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=ttl_days)
    
    cached = CachedVerification(
        cache_key=cache_key,
        url=url,
        source=source,
        signal_date=signal_date,
        verification_result=verification_result,
        verification_confidence=verification_confidence,
        verification_reason=verification_reason,
        verified_at=now,
        expires_at=expires
    )
    
    _verification_cache[cache_key] = cached
    logger.debug(f"Cached verification for key: {cache_key[:8]}... (TTL: {ttl_days} days)")
    
    # TODO: In production, also write to qualification_intent_cache table
    # await supabase.table("qualification_intent_cache").insert({...}).execute()


def clear_cache():
    """Clear all cached verifications."""
    _verification_cache.clear()
    logger.info("Cleared verification cache")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    now = datetime.now(timezone.utc)
    valid = sum(1 for c in _verification_cache.values() if c.expires_at > now)
    expired = len(_verification_cache) - valid
    
    return {
        "total_entries": len(_verification_cache),
        "valid_entries": valid,
        "expired_entries": expired,
    }


# =============================================================================
# Generic Intent Detection (Pre-LLM Check)
# =============================================================================

# Patterns that indicate generic/templated intent descriptions
# These are gaming attempts that produce "always pass" fallback intents
GENERIC_INTENT_PATTERNS = [
    # Exact patterns from the cipher model's fallback
    r"is\s+actively\s+operating\s+in\s+\w+",
    r"visible\s+market\s+activity",
    r"market\s+activity\s+and\s+company\s+updates",
    r"business\s+operations\s+and\s+updates",
    # Generic patterns that apply to ANY company
    r"^.{0,50}\s+is\s+(?:actively\s+)?(?:operating|expanding|growing)",
    r"company\s+(?:updates|activities|operations)",
    r"market\s+(?:activity|presence|operations)",
]

# Keywords that indicate specific (non-generic) intent
SPECIFIC_INTENT_KEYWORDS = [
    "hiring", "recruit", "job", "position", "opening",  # Hiring intent
    "launch", "released", "announced", "introduced",    # Product launch
    "raised", "funding", "series", "investment",        # Funding
    "partnership", "partnered", "collaboration",        # Partnership
    "acquired", "acquisition", "merger",                # M&A
    "expansion", "opened", "new office", "new location", # Geographic expansion
    "migrating", "adopting", "implementing",            # Technology adoption
]


def is_generic_intent_description(description: str) -> Tuple[bool, str]:
    """
    Check if an intent description is generic/templated (gaming attempt).
    
    This runs BEFORE the LLM call to save costs on obvious fallbacks.
    
    Args:
        description: The intent signal description
        
    Returns:
        Tuple of (is_generic: bool, reason: str)
    """
    desc_lower = description.lower().strip()
    
    # Check for known generic patterns
    for pattern in GENERIC_INTENT_PATTERNS:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return True, f"Generic pattern detected: matches '{pattern[:30]}...'"
    
    # Check if description has ANY specific intent keywords
    has_specific_keyword = False
    for keyword in SPECIFIC_INTENT_KEYWORDS:
        if keyword in desc_lower:
            has_specific_keyword = True
            break
    
    # Very short descriptions with no specific keywords are likely generic
    if len(desc_lower) < 80 and not has_specific_keyword:
        return True, "Description too short and lacks specific intent keywords"
    
    # Check for templated structure: "{company} is {verb}ing" with no specifics
    templated_pattern = r"^\w+(?:\s+\w+){0,3}\s+is\s+\w+ing\s+(?:in\s+)?\w+\s*\.?$"
    if re.match(templated_pattern, desc_lower) and not has_specific_keyword:
        return True, "Templated structure with no specific details"
    
    return False, "Description appears specific"


# =============================================================================
# Main Verification Function
# =============================================================================

async def verify_intent_signal(intent_signal: IntentSignal) -> Tuple[bool, int, str]:
    """
    Verify an intent signal claim.
    
    This is the main entry point for intent verification. It:
    1. PRE-CHECK: Reject known generic/templated descriptions (saves LLM cost)
    2. Checks cache for existing result
    3. Fetches content from the source URL
    4. Extracts relevant text
    5. Uses LLM to verify the claim matches the content
    6. Caches the result
    
    Args:
        intent_signal: The intent signal to verify
    
    Returns:
        Tuple of (verified: bool, confidence: int 0-100, reason: str)
    """
    logger.info(f"Verifying intent signal: {intent_signal.source} - {intent_signal.url[:50]}...")
    
    # Get source as string for comparisons
    source_str = intent_signal.source.value if isinstance(intent_signal.source, IntentSignalSource) else str(intent_signal.source)
    
    # PRE-CHECK: Reject generic/templated descriptions before expensive LLM call
    is_generic, generic_reason = is_generic_intent_description(intent_signal.description)
    if is_generic:
        logger.warning(f"Rejected generic intent: {generic_reason}")
        return False, 5, f"Generic fallback intent rejected: {generic_reason}"
    
    # Additional pre-check: "other" source type with vague description is suspicious
    if source_str.lower() == "other" and len(intent_signal.description) < 100:
        logger.warning("Rejected: 'other' source with short description")
        return False, 10, "Low-value source type 'other' with insufficient description"
    
    # Check cache first
    cache_key = compute_cache_key(intent_signal.url, source_str, intent_signal.date)
    cached = await get_cached_verification(cache_key)
    if cached:
        logger.info(f"Using cached verification: verified={cached.verification_result}")
        return cached.verification_result, cached.verification_confidence, cached.verification_reason
    
    # Fetch URL content
    try:
        content = await fetch_url_content(intent_signal.url, source_str)
    except Exception as e:
        logger.warning(f"Failed to fetch URL {intent_signal.url}: {e}")
        return False, 0, f"Failed to fetch URL: {str(e)[:100]}"
    
    if not content:
        logger.warning(f"URL returned no content: {intent_signal.url}")
        return False, 0, "URL returned no content"
    
    # Extract relevant text from content
    text = extract_verification_content(content, source_str)
    
    if not text or len(text.strip()) < 50:
        logger.warning(f"Insufficient content extracted from URL: {intent_signal.url}")
        return False, 0, "Insufficient content to verify claim"
    
    # Verify claim with LLM
    try:
        verified, confidence, reason = await llm_verify_claim(
            claim=intent_signal.description,
            url=intent_signal.url,
            date=intent_signal.date,
            content=text[:CONTENT_MAX_LENGTH]
        )
    except Exception as e:
        logger.error(f"LLM verification failed: {e}")
        return False, 0, f"LLM verification error: {str(e)[:100]}"
    
    # Cache result
    await cache_verification(
        cache_key=cache_key,
        url=intent_signal.url,
        source=source_str,
        signal_date=intent_signal.date,
        verification_result=verified,
        verification_confidence=confidence,
        verification_reason=reason,
        ttl_days=DEFAULT_CACHE_TTL_DAYS
    )
    
    logger.info(f"Verification complete: verified={verified}, confidence={confidence}")
    return verified, confidence, reason


# =============================================================================
# Content Fetching
# =============================================================================

async def fetch_url_content(url: str, source: str) -> str:
    """
    Fetch content from URL using appropriate method for the source type.
    
    Routes to the correct fetcher based on source:
    - LinkedIn: ScrapingDog LinkedIn API
    - Job boards: ScrapingDog scraper
    - GitHub: GitHub public API
    - Other: ScrapingDog generic scraper
    
    Args:
        url: The URL to fetch
        source: Source type (linkedin, job_board, github, etc.)
    
    Returns:
        Content as string (HTML or JSON depending on source)
    """
    source_lower = source.lower()
    
    if source_lower == "linkedin":
        return await scrapingdog_linkedin(url)
    elif source_lower == "job_board":
        return await scrapingdog_jobs(url)
    elif source_lower == "github":
        return await github_api(url)
    elif source_lower == "news":
        return await scrapingdog_generic(url)
    elif source_lower == "company_website":
        return await scrapingdog_generic(url)
    elif source_lower == "social_media":
        return await scrapingdog_generic(url)
    elif source_lower == "review_site":
        return await scrapingdog_generic(url)
    else:
        # Default to generic scraping
        return await scrapingdog_generic(url)


# =============================================================================
# ScrapingDog API Implementations
# =============================================================================

async def scrapingdog_linkedin(url: str) -> str:
    """
    Fetch LinkedIn content via ScrapingDog LinkedIn API.
    
    ScrapingDog handles proxy rotation internally.
    Supports: profiles (/in/), company pages (/company/), posts
    
    Args:
        url: LinkedIn URL (profile, company page, or post)
    
    Returns:
        JSON string with LinkedIn data
    """
    if not SCRAPINGDOG_API_KEY:
        raise ValueError("SCRAPINGDOG_API_KEY not configured")
    
    # Determine URL type
    if "/in/" in url:
        url_type = "profile"
    elif "/company/" in url:
        url_type = "company"
    else:
        url_type = "post"
    
    link_id = extract_linkedin_id(url)
    
    api_url = "https://api.scrapingdog.com/linkedin"
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "type": url_type,
        "linkId": link_id,
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data)


async def scrapingdog_jobs(url: str) -> str:
    """
    Fetch job board content via ScrapingDog scraper.
    
    Args:
        url: Job posting URL
    
    Returns:
        HTML content
    """
    if not SCRAPINGDOG_API_KEY:
        raise ValueError("SCRAPINGDOG_API_KEY not configured")
    
    api_url = "https://api.scrapingdog.com/scrape"
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "url": url,
        "dynamic": "false",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.text


async def scrapingdog_generic(url: str) -> str:
    """
    Generic web scraping via ScrapingDog.
    
    ScrapingDog handles proxy rotation internally.
    
    Args:
        url: URL to scrape
    
    Returns:
        HTML content
    """
    if not SCRAPINGDOG_API_KEY:
        raise ValueError("SCRAPINGDOG_API_KEY not configured")
    
    api_url = "https://api.scrapingdog.com/scrape"
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "url": url,
        "dynamic": "false",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.text


# =============================================================================
# GitHub API Implementation
# =============================================================================

async def github_api(url: str) -> str:
    """
    Fetch GitHub content via public API.
    
    Rate-limited but free. No proxy needed.
    
    Args:
        url: GitHub URL (repo, issue, PR, etc.)
    
    Returns:
        JSON content as string
    """
    # Convert github.com URL to api.github.com
    api_url = url
    
    if "github.com" in api_url:
        api_url = api_url.replace("github.com", "api.github.com/repos")
        
        # Handle blob URLs (file contents)
        if "/blob/" in api_url:
            api_url = api_url.replace("/blob/", "/contents/")
        
        # Handle tree URLs (directory listings)
        if "/tree/" in api_url:
            api_url = api_url.replace("/tree/", "/contents/")
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.text


# =============================================================================
# URL Parsing Helpers
# =============================================================================

def extract_linkedin_id(url: str) -> str:
    """
    Extract LinkedIn profile or post ID from URL.
    
    Examples:
    - linkedin.com/in/johnsmith -> johnsmith
    - linkedin.com/posts/johnsmith_activity-123 -> johnsmith_activity-123
    - linkedin.com/feed/update/urn:li:activity:123 -> urn:li:activity:123
    
    Args:
        url: LinkedIn URL
    
    Returns:
        Extracted ID
    """
    # Profile URL: /in/username
    match = re.search(r'/in/([^/?]+)', url)
    if match:
        return match.group(1)
    
    # Post URL: /posts/username_...
    match = re.search(r'/posts/([^/?]+)', url)
    if match:
        return match.group(1)
    
    # Activity URL: /feed/update/urn:li:activity:...
    match = re.search(r'/feed/update/(urn:li:[^/?]+)', url)
    if match:
        return match.group(1)
    
    # Company URL: /company/companyname
    match = re.search(r'/company/([^/?]+)', url)
    if match:
        return match.group(1)
    
    # Fallback: last path segment
    return url.rstrip('/').split('/')[-1]


def extract_github_info(url: str) -> Dict[str, str]:
    """
    Extract owner/repo/path from GitHub URL.
    
    Args:
        url: GitHub URL
    
    Returns:
        Dict with owner, repo, and optional path
    """
    # Pattern: github.com/owner/repo/...
    match = re.search(r'github\.com/([^/]+)/([^/]+)(?:/(.*))?', url)
    if match:
        return {
            "owner": match.group(1),
            "repo": match.group(2),
            "path": match.group(3) or ""
        }
    return {"owner": "", "repo": "", "path": ""}


# =============================================================================
# Content Extraction
# =============================================================================

def extract_verification_content(html_or_json: str, source: str) -> str:
    """
    Extract relevant text content for verification.
    
    Different extraction strategies per source type:
    - LinkedIn: Parse JSON response for relevant fields
    - Job boards: Extract job description sections
    - GitHub: Parse JSON for file content or README
    - Generic: Extract main content area
    
    Args:
        html_or_json: Raw content (HTML or JSON string)
        source: Source type
    
    Returns:
        Extracted text content
    """
    source_lower = source.lower()
    
    # Handle LinkedIn JSON response
    if source_lower == "linkedin":
        return _extract_linkedin_content(html_or_json)
    
    # Handle GitHub JSON response
    if source_lower == "github":
        return _extract_github_content(html_or_json)
    
    # Handle HTML content
    return _extract_html_content(html_or_json, source_lower)


def _extract_linkedin_content(json_str: str) -> str:
    """Extract content from LinkedIn API JSON response."""
    try:
        data = json.loads(json_str)
        
        parts = []
        
        # Profile data
        if "headline" in data:
            parts.append(f"Headline: {data['headline']}")
        if "summary" in data:
            parts.append(f"Summary: {data['summary']}")
        if "experience" in data:
            for exp in data.get("experience", [])[:5]:
                parts.append(f"Experience: {exp.get('title', '')} at {exp.get('company', '')}")
        
        # Post data
        if "text" in data:
            parts.append(f"Post: {data['text']}")
        if "commentary" in data:
            parts.append(f"Commentary: {data['commentary']}")
        
        # Activity data
        if "activity" in data:
            parts.append(f"Activity: {data['activity']}")
        
        return "\n".join(parts)
    except json.JSONDecodeError:
        return json_str[:CONTENT_MAX_LENGTH]


def _extract_github_content(json_str: str) -> str:
    """Extract content from GitHub API JSON response."""
    try:
        data = json.loads(json_str)
        
        parts = []
        
        # File content (base64 encoded)
        if "content" in data and "encoding" in data:
            if data["encoding"] == "base64":
                import base64
                try:
                    content = base64.b64decode(data["content"]).decode('utf-8')
                    parts.append(content[:CONTENT_MAX_LENGTH])
                except Exception:
                    pass
        
        # Repository info
        if "description" in data:
            parts.append(f"Description: {data['description']}")
        if "readme" in data:
            parts.append(f"README: {data['readme']}")
        
        # Issue/PR
        if "title" in data:
            parts.append(f"Title: {data['title']}")
        if "body" in data:
            parts.append(f"Body: {data['body']}")
        
        return "\n".join(parts) if parts else json_str[:CONTENT_MAX_LENGTH]
    except json.JSONDecodeError:
        return json_str[:CONTENT_MAX_LENGTH]


def _extract_html_content(html: str, source: str) -> str:
    """Extract text content from HTML."""
    if not BS4_AVAILABLE:
        # Fallback: basic regex-based extraction
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:CONTENT_MAX_LENGTH]
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script/style/nav/footer elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
        element.decompose()
    
    # Source-specific extraction
    content = None
    
    if source == "linkedin":
        # LinkedIn-specific selectors
        content = soup.find(class_=['feed-shared-update-v2', 'post-content', 'experience-section', 'pv-about-section'])
    
    elif source == "job_board":
        # Job board selectors
        content = soup.find(class_=['job-description', 'description', 'posting-body', 'job-details', 'job-content'])
        if not content:
            content = soup.find(id=['job-description', 'description', 'job-details'])
    
    elif source == "news":
        # News article selectors
        content = soup.find(['article', 'main'])
        if not content:
            content = soup.find(class_=['article-body', 'story-body', 'post-content', 'entry-content'])
    
    elif source == "company_website":
        # Company website - look for about/team pages
        content = soup.find(class_=['about', 'team', 'careers', 'blog-post', 'news-item'])
        if not content:
            content = soup.find(['article', 'main'])
    
    elif source == "review_site":
        # Review sites
        content = soup.find(class_=['review', 'review-content', 'user-review', 'review-text'])
    
    # Fallback to main content areas
    if not content:
        content = soup.find(['main', 'article'])
    if not content:
        # Try finding any div with substantial content
        for div in soup.find_all('div'):
            text = div.get_text(strip=True)
            if len(text) > 100:  # Found a div with real content
                content = div
                break
    if not content:
        content = soup.body
    
    if content:
        text = content.get_text(separator=' ', strip=True)[:CONTENT_MAX_LENGTH]
        # If still too short, return raw HTML text as last resort
        if len(text) < 50 and soup.body:
            text = soup.body.get_text(separator=' ', strip=True)[:CONTENT_MAX_LENGTH]
        return text
    
    return ""


# =============================================================================
# LLM Verification
# =============================================================================

async def llm_verify_claim(
    claim: str,
    url: str,
    date: str,
    content: str
) -> Tuple[bool, int, str]:
    """
    Use LLM to verify an intent signal claim matches the source content.
    
    Args:
        claim: The intent signal description/claim
        url: Source URL
        date: Claimed date of the signal
        content: Extracted text content from the source
    
    Returns:
        Tuple of (verified: bool, confidence: int 0-100, reason: str)
    """
    prompt = f"""You are verifying an intent signal claim for a B2B lead generation system.

CLAIM: {claim}
SOURCE URL: {url}
CLAIMED DATE: {date}
CONTENT EXCERPT: {content}

Your task is to determine if the content SUPPORTS the intent signal claim with SPECIFIC evidence.

CRITICAL - Reject these GENERIC/TEMPLATED claims (they are gaming attempts):
- "[Company] is actively operating in [industry]" - This is true for ANY company with a website
- "[Company] market activity and company updates" - Too vague, no specific intent
- "[Company] is expanding/growing/operating" - Generic statements without specifics
- Claims that would be true for ANY company in that industry

Verification criteria:
1. The claim must contain SPECIFIC details (hiring for X role, launched Y product, raised Z funding)
2. Generic claims like "actively operating" or "visible market activity" should be REJECTED
3. The specific details in the claim MUST appear in the content
4. The date should be reasonably close to the claimed date (within a few weeks is OK)
5. If the claim is too vague to verify (no specific action/event), mark as NOT verified

RED FLAGS (automatic fail):
- Claim contains no specific action, product, or event
- Claim could apply to any company in the industry
- Claim uses filler phrases like "market activity", "business operations", "company updates"

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{"verified": true/false, "confidence": 0-100, "reason": "Brief 1-2 sentence explanation"}}

Examples of valid responses:
{{"verified": true, "confidence": 85, "reason": "The content mentions hiring for DevOps roles which matches the claimed intent signal."}}
{{"verified": false, "confidence": 20, "reason": "The content discusses unrelated topics and does not support the claimed signal."}}
{{"verified": false, "confidence": 10, "reason": "Claim is too generic - 'actively operating' applies to any company with a website."}}
"""
    
    try:
        response_text = await openrouter_chat(prompt, model="gpt-4o-mini")
        
        # Parse JSON response
        # Handle potential markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        result = json.loads(response_text)
        
        verified_raw = result.get("verified", False)
        confidence = int(result.get("confidence", 0))
        reason = result.get("reason", "No reason provided")
        
        # Apply confidence threshold
        verified = verified_raw and confidence >= CONFIDENCE_THRESHOLD
        
        return verified, confidence, reason
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return False, 0, f"LLM response parsing error"
    except Exception as e:
        logger.error(f"LLM verification error: {e}")
        raise


async def openrouter_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenRouter LLM API.
    
    Args:
        prompt: The prompt to send
        model: Model to use (default: gpt-4o-mini)
    
    Returns:
        LLM response text
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://leadpoet.ai",
                "X-Title": "Leadpoet Qualification"
            },
            json={
                "model": f"openai/{model}",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # Lower temperature for more consistent verification
                "max_tokens": 200,
            },
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# =============================================================================
# Batch Verification
# =============================================================================

async def verify_intent_signals_batch(
    signals: list[IntentSignal]
) -> list[Tuple[bool, int, str]]:
    """
    Verify multiple intent signals (with caching).
    
    Args:
        signals: List of intent signals to verify
    
    Returns:
        List of (verified, confidence, reason) tuples
    """
    results = []
    for signal in signals:
        try:
            result = await verify_intent_signal(signal)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to verify signal {signal.url}: {e}")
            results.append((False, 0, f"Verification error: {str(e)[:50]}"))
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def is_verification_configured() -> bool:
    """Check if verification APIs are configured."""
    return bool(SCRAPINGDOG_API_KEY and OPENROUTER_API_KEY)


def get_verification_config() -> Dict[str, Any]:
    """Get verification configuration status."""
    return {
        "scrapingdog_configured": bool(SCRAPINGDOG_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "github_configured": bool(GITHUB_TOKEN),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "content_max_length": CONTENT_MAX_LENGTH,
        "cache_ttl_days": DEFAULT_CACHE_TTL_DAYS,
        "cache_stats": get_cache_stats(),
    }
