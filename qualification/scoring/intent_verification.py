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
from datetime import datetime, date, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, NamedTuple
from urllib.parse import urlparse

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

# Cache TTL — kept short so stale content (removed job posts, paywalled articles)
# doesn't cause false positives on re-verification.
DEFAULT_CACHE_TTL_DAYS = 2



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
# Snippet Verbatim Check (Pre-LLM Anti-Gaming)
# =============================================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def compute_snippet_overlap(snippet: str, content: str) -> float:
    """
    Compute what fraction of a snippet's 4-word n-grams appear in the content.

    Returns a float 0.0-1.0 representing the overlap ratio. Legitimate models
    that extract verbatim text from web pages will score near 1.0. Models that
    fabricate, template, or strip/modify text will score much lower.
    """
    norm_snippet = _normalize_text(snippet)
    norm_content = _normalize_text(content)

    snippet_words = norm_snippet.split()
    if len(snippet_words) < 4:
        return 1.0  # too short to check meaningfully

    content_set: set = set()
    content_words = norm_content.split()
    for i in range(len(content_words) - 3):
        content_set.add(tuple(content_words[i:i + 4]))

    matches = 0
    total = len(snippet_words) - 3
    for i in range(total):
        if tuple(snippet_words[i:i + 4]) in content_set:
            matches += 1

    return matches / total if total > 0 else 1.0


def check_description_grounding(description: str, source_content: str) -> float:
    """
    Check what fraction of the description's meaningful words appear in the source content.

    Unlike snippet overlap (which checks 4-grams), this checks individual content words
    (nouns, verbs, adjectives ≥5 chars, excluding stopwords) to catch LLM-generated
    descriptions that sound specific but contain claims not present in the source.

    Returns 0.0-1.0. Legitimate descriptions based on real content score >0.4.
    LLM-fabricated descriptions with injected signal words score <0.25.
    """
    _STOP_WORDS = {
        "about", "after", "being", "between", "could", "during", "every",
        "first", "their", "these", "those", "through", "under", "using",
        "which", "while", "would", "other", "there", "where", "should",
        "company", "business", "service", "services", "solution", "solutions",
        "based", "including", "across", "within", "through",
    }

    desc_lower = _normalize_text(description)
    content_lower = _normalize_text(source_content)

    content_words = set(content_lower.split())

    desc_words = [
        w for w in desc_lower.split()
        if len(w) >= 5 and w not in _STOP_WORDS
    ]

    if len(desc_words) < 3:
        return 1.0

    found = sum(1 for w in desc_words if w in content_words)
    return found / len(desc_words)


def check_signal_word_grounding(description: str, source_content: str) -> tuple:
    """
    Check whether intent signal words in the description actually appear in the source.

    Signal words are action verbs that indicate buying intent (launched, hired, raised, etc.).
    If the description contains these words but the source doesn't, the LLM likely injected
    them to satisfy a prompt requirement.

    Returns (grounded_count, total_signal_words, ungrounded_words).
    """
    _SIGNAL_WORDS = {
        "launched", "announced", "raised", "hired", "hiring", "acquired",
        "expanded", "expanding", "partnered", "partnership", "funding",
        "funded", "invested", "investment", "merged", "acquisition",
        "recruited", "recruiting", "opening", "openings",
    }

    content_lower = _normalize_text(source_content)
    content_words = set(content_lower.split())

    desc_lower = _normalize_text(description)
    desc_words = set(desc_lower.split())

    signal_in_desc = desc_words & _SIGNAL_WORDS
    if not signal_in_desc:
        return 0, 0, []

    grounded = signal_in_desc & content_words
    ungrounded = signal_in_desc - content_words

    return len(grounded), len(signal_in_desc), sorted(ungrounded)


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
# Date Precision Verification (Mechanism-Based Gaming Detection)
# =============================================================================

# Month name/abbreviation mappings for date matching
_MONTH_NAMES = {
    1: ("january", "jan"),
    2: ("february", "feb"),
    3: ("march", "mar"),
    4: ("april", "apr"),
    5: ("may", "may"),
    6: ("june", "jun"),
    7: ("july", "jul"),
    8: ("august", "aug"),
    9: ("september", "sep"),
    10: ("october", "oct"),
    11: ("november", "nov"),
    12: ("december", "dec"),
}


def strip_copyright_founded_years(content: str) -> str:
    """
    Remove copyright notices and founding year phrases from content so they
    cannot serve as false date evidence.

    Strips patterns like:
        © 2024, Copyright 2024, (c) 2024
        Founded 2015, Established 2010, Since 2008, Est. 2012

    The year digits are replaced with "XXXX" so word boundaries remain intact
    but the year can no longer match date searches.
    """
    # Copyright: © YYYY, (c) YYYY, Copyright YYYY (with optional surrounding text)
    content = re.sub(
        r'(?:©|\(c\)|copyright)\s*(?:©|\(c\))?\s*(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2})?',
        'XXXX',
        content,
        flags=re.IGNORECASE,
    )
    # Founded/Established/Since/Est.: "Founded in 2015", "Established 2010", "Since 2008"
    content = re.sub(
        r'(?:founded|established|since|est\.?)\s+(?:in\s+)?(?:19|20)\d{2}\b',
        'XXXX',
        content,
        flags=re.IGNORECASE,
    )
    return content


def strip_dynamic_boilerplate_dates(content: str) -> str:
    """
    Remove dynamic/boilerplate date patterns that pages generate automatically.

    These are NOT intent event dates — they're page rendering artifacts that
    models exploit by using date.today() and finding pages where that date
    appears as a dynamic element.

    Strips patterns like:
        Last updated: February 23, 2026
        As of 02/23/2026
        Modified on 2026-02-23
        Updated: Jan 15, 2026
        Retrieved on February 23, 2026
        Accessed 2026-02-23
        Generated on 2/23/2026

    The date portion is replaced with "XXXX" so the date can no longer
    match in check_date_precision.
    """
    # ISO dates: 2026-02-23
    _iso = r'(?:19|20)\d{2}-\d{2}-\d{2}'
    # Slash dates: 02/23/2026 or 2/23/2026
    _slash = r'\d{1,2}/\d{1,2}/(?:19|20)\d{2}'
    # Named dates: February 23, 2026 / Feb 23 2026 / 23 February 2026
    _month = (
        r'(?:january|february|march|april|may|june|july|august|september|'
        r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    )
    _named = rf'{_month}\s+\d{{1,2}}[,]?\s*(?:19|20)\d{{2}}'
    _named_dmy = rf'\d{{1,2}}\s+{_month}[,]?\s*(?:19|20)\d{{2}}'

    _any_date = rf'(?:{_iso}|{_slash}|{_named}|{_named_dmy})'

    # Boilerplate prefixes that indicate dynamic/meta dates (not content dates)
    _prefixes = (
        r'(?:last\s+)?(?:updated|modified|refreshed|generated|retrieved|accessed|fetched)'
        r'|as\s+of'
        r'|current\s+(?:as\s+of|date)'
        r'|page\s+(?:updated|generated|modified)'
        r'|date\s*:'
    )

    content = re.sub(
        rf'(?:{_prefixes})\s*(?:on\s+)?:?\s*{_any_date}',
        'XXXX',
        content,
        flags=re.IGNORECASE,
    )
    return content


def check_date_precision(claimed_date: str, content: str) -> str:
    """
    Verify how precisely a claimed date appears in the source content.

    This is the primary mechanism-based defense against date fabrication.
    Instead of pattern-matching model code, it checks the OUTPUT: does the
    claimed date actually appear in the scraped web content with sufficient
    precision?

    Args:
        claimed_date: ISO-format date string (YYYY-MM-DD) from the model output
        content: Scraped web content (already stripped of copyright/founded years)

    Returns one of:
        "verified"   – exact date (YYYY-MM-DD or "Month Day, Year") found in content
        "approximate"– month+year found but not the exact day
        "year_only"  – only the year is present (manufactured precision)
        "no_match"   – the claimed year doesn't appear at all
    """
    try:
        dt = datetime.strptime(claimed_date.strip()[:10], "%Y-%m-%d")
    except (ValueError, AttributeError):
        return "no_match"

    year = dt.year
    month = dt.month
    day = dt.day
    year_str = str(year)
    month_names = _MONTH_NAMES.get(month, ())
    content_lower = content.lower()

    # ------------------------------------------------------------------
    # Tier 1: Exact date match in any common format
    # ------------------------------------------------------------------
    # ISO: 2025-01-15
    iso_date = f"{year:04d}-{month:02d}-{day:02d}"
    if iso_date in content:
        return "verified"

    # "January 15, 2025" / "January 15 2025" / "15 January 2025"
    for full_name, abbrev in [month_names] if month_names else []:
        day_str = str(day)
        day_padded = f"{day:02d}"
        for m in (full_name, abbrev):
            # Month Day, Year
            if re.search(rf'\b{m}\s+{day_str}\b[,]?\s*{year_str}', content_lower):
                return "verified"
            if day_padded != day_str and re.search(rf'\b{m}\s+{day_padded}\b[,]?\s*{year_str}', content_lower):
                return "verified"
            # Day Month Year
            if re.search(rf'\b{day_str}\s+{m}\b[,]?\s*{year_str}', content_lower):
                return "verified"
            if day_padded != day_str and re.search(rf'\b{day_padded}\s+{m}\b[,]?\s*{year_str}', content_lower):
                return "verified"

    # JSON-LD / schema.org: "datePosted":"2025-01-15", "datePublished":"2025-01-15"
    if re.search(rf'date\w*["\']?\s*[:=]\s*["\']?{re.escape(iso_date)}', content_lower):
        return "verified"

    # MM/DD/YYYY or DD/MM/YYYY — check both orderings
    slash_mdy = f"{month:02d}/{day:02d}/{year}"
    slash_dmy = f"{day:02d}/{month:02d}/{year}"
    if slash_mdy in content or slash_dmy in content:
        return "verified"

    # ------------------------------------------------------------------
    # Tier 2: Month + Year match (approximate)
    # ------------------------------------------------------------------
    month_year_found = False

    # "January 2025" / "Jan 2025"
    for m in month_names:
        if re.search(rf'\b{m}\s+{year_str}\b', content_lower):
            month_year_found = True
            break

    # YYYY-MM (ISO prefix)
    iso_month = f"{year:04d}-{month:02d}"
    if iso_month in content:
        month_year_found = True

    # MM/YYYY
    slash_my = f"{month:02d}/{year}"
    if slash_my in content:
        month_year_found = True

    if month_year_found:
        if day == 1:
            return "approximate"
        return "verified"

    # ------------------------------------------------------------------
    # Tier 3: Year-only match (manufactured precision)
    # ------------------------------------------------------------------
    if re.search(rf'\b{year_str}\b', content_lower):
        return "year_only"

    # ------------------------------------------------------------------
    # Tier 4: No match at all
    # ------------------------------------------------------------------
    return "no_match"


# =============================================================================
# Pre-check Utilities (cheap checks that run before LLM / ScrapingDog calls)
# =============================================================================

# Maps each declared source type to URL domains that are valid for it.
# If the source type isn't in this map, we skip the mismatch check.
_SOURCE_DOMAIN_ALLOWLIST: Dict[str, frozenset] = {
    "linkedin": frozenset({"linkedin.com"}),
    "github": frozenset({"github.com", "raw.githubusercontent.com"}),
    "wikipedia": frozenset({"en.wikipedia.org", "wikipedia.org"}),
    "job_board": frozenset({
        "linkedin.com",
        "indeed.com", "glassdoor.com",
        "lever.co", "greenhouse.io", "boards.greenhouse.io",
        "jobs.lever.co", "apply.workable.com",
        "careers.google.com", "jobs.apple.com",
        "builtin.com", "ziprecruiter.com",
        "monster.com", "wellfound.com", "angel.co",
        "dice.com", "simplyhired.com",
        "roberthalf.com", "himalayas.app",
        "remoteok.com", "weworkremotely.com", "flexjobs.com",
    }),
    "news": frozenset({
        "reuters.com", "bloomberg.com", "techcrunch.com",
        "forbes.com", "cnbc.com", "wsj.com",
        "nytimes.com", "bbc.com", "bbc.co.uk",
        "apnews.com", "businessinsider.com", "venturebeat.com",
        "theverge.com", "wired.com", "axios.com",
        "ft.com", "prnewswire.com", "businesswire.com",
        "globenewswire.com",
        "yahoo.com", "finance.yahoo.com", "news.yahoo.com",
        "crunchbase.com", "news.crunchbase.com",
        "zdnet.com", "cnet.com", "siliconangle.com",
        "marketwatch.com", "thestreet.com", "seekingalpha.com",
        "benzinga.com", "investopedia.com", "medium.com",
        "theinformation.com", "www.theinformation.com",
        "protocol.com",
        "arstechnica.com",
        "engadget.com", "www.engadget.com",
        "washingtonpost.com", "www.washingtonpost.com",
        "guardian.co.uk", "www.theguardian.com", "theguardian.com",
        "economist.com", "www.economist.com",
        "nasdaq.com", "pharmaceutical-technology.com",
        "intellectia.ai", "fiercepharma.com", "fiercebiotech.com",
        "statnews.com", "biopharmadive.com", "supplychaindive.com",
        "retaildive.com", "ciodive.com", "hrdive.com",
    }),
    "social_media": frozenset({
        "twitter.com", "x.com",
        "facebook.com", "instagram.com",
        "reddit.com", "old.reddit.com",
        "youtube.com", "tiktok.com",
        "threads.net", "linkedin.com",
    }),
    "review_site": frozenset({
        "g2.com", "capterra.com", "trustpilot.com",
        "trustradius.com", "gartner.com",
        "softwareadvice.com", "getapp.com",
        "peerspot.com", "glassdoor.com", "yelp.com",
    }),
}


def _is_known_third_party_domain(domain: str) -> Optional[str]:
    """Check if a domain belongs to a known third-party platform.
    Returns the actual source type if it's a known platform, None otherwise."""
    domain = domain.lower().strip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    for source_type, domains in _SOURCE_DOMAIN_ALLOWLIST.items():
        if domain in domains:
            return source_type
    return None


def check_source_url_mismatch(source_str: str, url: str) -> Optional[str]:
    """
    Check if the declared source type is plausible given the URL domain.
    
    Returns an error message if there's a clear mismatch, None if OK.
    
    CRITICAL: If source is "company_website" but the URL is actually a known
    third-party platform (news site, job board, Wikipedia, etc.), flag it.
    This catches models that label everything as "company_website" to bypass
    source type validation.
    """
    source_lower = source_str.lower().strip()
    
    try:
        clean_url = url.strip()
        if not clean_url.lower().startswith(('http://', 'https://')):
            clean_url = 'https://' + clean_url
        url_domain = (urlparse(clean_url).hostname or "").lower()
        if url_domain.startswith("www."):
            url_domain = url_domain[4:]
    except Exception:
        return None
    
    # If source is "company_website", verify URL isn't a known third-party platform
    if source_lower in ("company_website", "other"):
        actual_type = _is_known_third_party_domain(url_domain)
        if actual_type:
            return (
                f"Source declared as '{source_str}' but URL domain '{url_domain}' "
                f"is a known {actual_type} platform — source type should be '{actual_type}'"
            )
        return None

    allowed = _SOURCE_DOMAIN_ALLOWLIST.get(source_lower)
    if allowed is None:
        return None

    try:
        clean_url = url.strip()
        if not clean_url.lower().startswith(('http://', 'https://')):
            clean_url = 'https://' + clean_url
        domain = urlparse(clean_url).hostname or ""
    except Exception:
        return None

    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain in allowed:
        return None

    return (
        f"Source type '{source_str}' declared but URL domain '{domain}' "
        f"is not a recognized {source_str} domain"
    )


def check_future_date(signal_date: Optional[str]) -> Optional[str]:
    """
    Reject dates set in the future — obviously fabricated.
    
    Returns an error message if the date is in the future, None if OK.
    """
    if not signal_date:
        return None
    try:
        parsed = date.fromisoformat(signal_date)
    except (ValueError, TypeError):
        return None
    if parsed > date.today():
        return f"Signal date {signal_date} is in the future — fabricated"
    return None


def check_company_in_content(company_name: str, text: str) -> bool:
    """
    Check whether the company name appears in the scraped page content.
    
    Uses case-insensitive matching with strict rules to prevent false positives
    from partial name matches (e.g., "Forum" matching "Forum Research" when
    the lead is "Forum Health").
    
    Rules:
    - Full name match: always passes ("Forum Health" in text)
    - Multi-word names: ALL significant words (≥4 chars) must appear, AND at least
      one word must appear adjacent to another company word (prevents coincidental
      matches where "Forum" and "Health" appear in unrelated contexts)
    - Single-word names: require word boundary match (not substring)
    """
    if not company_name or not text:
        return False
    name_lower = company_name.lower().strip()
    text_lower = text.lower()

    # Full name exact match (best case)
    if name_lower in text_lower:
        return True

    # Strip common suffixes for matching
    _SUFFIXES = {"inc", "inc.", "llc", "llc.", "ltd", "ltd.", "corp", "corp.",
                 "co", "co.", "company", "group", "holdings", "partners", "lp", "l.p."}
    words = [w for w in name_lower.split() if w not in _SUFFIXES and len(w) >= 3]

    if not words:
        return False

    # Single significant word: require word boundary match
    if len(words) == 1:
        import re
        pattern = r'\b' + re.escape(words[0]) + r'\b'
        return bool(re.search(pattern, text_lower))

    # Multi-word name: ALL significant words must appear
    if not all(w in text_lower for w in words):
        return False

    # Additional check: the company words must appear as a near-contiguous phrase
    # (within 3 words of each other) to prevent coincidental matches like
    # "Forum Research" + "public health" matching "Forum Health".
    text_words = text_lower.split()
    for i, tw in enumerate(text_words):
        if tw == words[0] or tw.startswith(words[0]):
            nearby = text_words[i:i + len(words) + 3]
            nearby_str = " ".join(nearby)
            if all(w in nearby_str for w in words):
                return True

    return False


# =============================================================================
# Main Verification Function
# =============================================================================

async def verify_intent_signal(
    intent_signal: IntentSignal,
    icp_industry: Optional[str] = None,
    icp_criteria: Optional[str] = None,
    company_name: Optional[str] = None,
    company_website: Optional[str] = None
) -> Tuple[bool, int, str, str]:
    """
    Verify an intent signal claim AND check for ICP evidence.
    
    This is the main entry point for intent verification. It:
    1. PRE-CHECK: Reject known generic/templated descriptions (saves LLM cost)
    2. Checks cache for existing result
    3. Fetches content from the source URL using ScrapingDog
    4. Extracts relevant text
    5. Uses LLM to verify:
       a) The claim is supported by the URL content
       b) The URL provides evidence the company matches ICP criteria
    6. Caches the result
    
    Args:
        intent_signal: The intent signal to verify
        icp_industry: Target industry from ICP (e.g., "Healthcare")
        icp_criteria: Additional ICP criteria (e.g., "PE-backed, 50-500 employees")
        company_name: Name of the company for verification
    
    Returns:
        Tuple of (verified: bool, confidence: int 0-100, reason: str, date_status: str)
        date_status is one of: "verified", "no_date", "fabricated"
    """
    # Defensive URL normalization (Pydantic handles this at entry, but this
    # function may be called directly in tests or non-Pydantic code paths)
    url = intent_signal.url.strip()
    if url and not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url
        intent_signal = intent_signal.model_copy(update={"url": url})
    
    logger.info(f"Verifying intent signal: {intent_signal.source} - {intent_signal.url[:50]}...")
    
    # Get source as string for comparisons
    source_str = intent_signal.source.value if isinstance(intent_signal.source, IntentSignalSource) else str(intent_signal.source)
    
    # PRE-CHECK: Reject generic/templated descriptions before expensive LLM call
    is_generic, generic_reason = is_generic_intent_description(intent_signal.description)
    if is_generic:
        logger.warning(f"Rejected generic intent: {generic_reason}")
        return False, 5, f"Generic fallback intent rejected: {generic_reason}", "fabricated"
    
    # Additional pre-check: "other" source type with vague description is suspicious
    if source_str.lower() == "other" and len(intent_signal.description) < 100:
        logger.warning("Rejected: 'other' source with short description")
        return False, 10, "Low-value source type 'other' with insufficient description", "fabricated"
    
    # PRE-CHECK: Reject future dates (obviously fabricated)
    future_err = check_future_date(intent_signal.date)
    if future_err:
        logger.warning(f"❌ Future date rejected: {future_err}")
        return False, 0, future_err, "fabricated"

    # PRE-CHECK: Source type vs URL domain mismatch
    mismatch_err = check_source_url_mismatch(source_str, intent_signal.url)
    if mismatch_err:
        logger.warning(f"❌ Source/URL mismatch: {mismatch_err}")
        return False, 0, mismatch_err, "fabricated"

    # PRE-CHECK: If source is "company_website", the signal URL domain must match
    # the lead's actual company_website domain. A signal from prnewswire.com or
    # wvcapital.com is NOT the company's own website.
    if source_str.lower() == "company_website" and company_website:
        def _base_domain(u: str) -> str:
            try:
                from urllib.parse import urlparse as _up
                h = (_up(u if u.startswith(("http://", "https://")) else f"https://{u}").hostname or "").lower()
                h = h[4:] if h.startswith("www.") else h
                parts = h.split(".")
                return ".".join(parts[-2:]) if len(parts) >= 2 else h
            except Exception:
                return ""

        lead_domain = _base_domain(company_website)
        signal_domain = _base_domain(intent_signal.url)
        if lead_domain and signal_domain and lead_domain != signal_domain:
            logger.warning(
                f"❌ company_website domain mismatch: signal={signal_domain}, lead={lead_domain}"
            )
            return False, 0, (
                f"Source is 'company_website' but signal URL domain ({signal_domain}) "
                f"doesn't match lead's company website ({lead_domain}). "
                f"Use the correct source type (news, job_board, etc.) for third-party URLs."
            ), "fabricated"

    # Check cache first (include ICP in cache key if provided)
    icp_cache_suffix = f"|{icp_industry}|{icp_criteria}" if icp_industry else ""
    cache_key = compute_cache_key(intent_signal.url + icp_cache_suffix, source_str, intent_signal.date)
    cached = await get_cached_verification(cache_key)
    if cached:
        logger.info(f"Using cached verification: verified={cached.verification_result}")
        # Legacy cache entries don't have date_status — default to "verified"
        return cached.verification_result, cached.verification_confidence, cached.verification_reason, "verified"
    
    # Fetch URL content via ScrapingDog
    try:
        content = await fetch_url_content(intent_signal.url, source_str)
    except Exception as e:
        logger.warning(f"Failed to fetch URL {intent_signal.url}: {e}")
        return False, 0, f"Failed to fetch URL: {str(e)[:100]}", "fabricated"
    
    if not content:
        logger.warning(f"URL returned no content: {intent_signal.url}")
        return False, 0, "URL returned no content", "fabricated"
    
    # Extract relevant text from content
    text = extract_verification_content(content, source_str)
    
    if not text or len(text.strip()) < 50:
        logger.warning(f"Insufficient content extracted from URL: {intent_signal.url}")
        return False, 0, "Insufficient content to verify claim", "fabricated"

    # ── URL-to-company check (BEFORE LLM — catch misattributed articles cheaply) ──
    # A model might find a great article about Company A and attribute it to Company B.
    # Skip for generic aggregator pages (job boards, review sites) where the company
    # name may not dominate the page text, and for Wikipedia which uses formal names.
    if company_name and source_str.lower() not in ("job_board", "review_site", "wikipedia"):
        if not check_company_in_content(company_name, text[:CONTENT_MAX_LENGTH]):
            logger.warning(
                f"❌ Company name '{company_name}' not found in content from {intent_signal.url[:60]}"
            )
            return False, 0, (
                f"Company '{company_name}' not mentioned in source content — "
                f"signal may be misattributed to the wrong company"
            ), "fabricated"
        else:
            logger.info(f"✓ Company '{company_name}' found in page content")

    # ── Snippet verbatim check (BEFORE LLM — saves cost on obvious fabrication) ──
    # The snippet field must contain text actually found on the source page.
    # Models that fabricate descriptions via templates, strip negative LLM
    # assessments, or construct evidence from f-strings will fail this check.
    snippet_text = getattr(intent_signal, 'snippet', None) or ""
    if len(snippet_text.strip()) >= 30 and len(text.strip()) >= 200:
        snippet_overlap = compute_snippet_overlap(snippet_text, text)
        if snippet_overlap < 0.30:
            logger.warning(
                f"❌ Snippet verbatim check FAILED: overlap={snippet_overlap:.0%} "
                f"for {intent_signal.url[:60]}"
            )
            return False, 0, (
                f"Snippet not found in source content (overlap: {snippet_overlap:.0%}). "
                f"The snippet text does not appear on the page — likely fabricated or manipulated."
            ), "fabricated"
        else:
            logger.info(f"✓ Snippet verbatim check passed: overlap={snippet_overlap:.0%}")

    # ── Description grounding check (BEFORE LLM — catch LLM-fabricated descriptions) ──
    # The snippet may be real scraped text, but the DESCRIPTION could be LLM-generated
    # embellishment that adds claims not present in the source. Check that the description's
    # key content words actually appear in the scraped text.
    if intent_signal.description and len(text.strip()) >= 200:
        desc_grounding = check_description_grounding(intent_signal.description, text[:CONTENT_MAX_LENGTH])
        if desc_grounding < 0.25:
            logger.warning(
                f"❌ Description grounding FAILED: overlap={desc_grounding:.0%} "
                f"for {intent_signal.url[:60]}"
            )
            return False, 0, (
                f"Description not grounded in source content (overlap: {desc_grounding:.0%}). "
                f"The description contains claims not found on the page — likely LLM-fabricated."
            ), "fabricated"
        else:
            logger.info(f"✓ Description grounding check passed: overlap={desc_grounding:.0%}")

    # ── Signal word grounding check (catch LLM-injected action verbs) ──
    # Models that force LLM prompts to include words like "launched", "announced",
    # "hiring" will inject these words even when the source content doesn't contain them.
    if intent_signal.description and len(text.strip()) >= 200:
        grounded_count, total_signal, ungrounded = check_signal_word_grounding(
            intent_signal.description, text[:CONTENT_MAX_LENGTH]
        )
        if total_signal > 0 and grounded_count == 0:
            logger.warning(
                f"❌ Signal word grounding FAILED: {ungrounded} not in source content"
            )
            return False, 0, (
                f"Intent signal words ({', '.join(ungrounded)}) not found in source content. "
                f"The description likely contains LLM-injected action verbs not supported by evidence."
            ), "fabricated"
        elif total_signal > 0:
            logger.info(
                f"✓ Signal word grounding: {grounded_count}/{total_signal} grounded"
                + (f", ungrounded: {ungrounded}" if ungrounded else "")
            )

    # Verify claim with LLM - now includes ICP context
    date_for_llm = intent_signal.date or "Not provided"
    try:
        verified, confidence, reason, date_status, claim_supported = await llm_verify_claim_with_icp(
            claim=intent_signal.description,
            url=intent_signal.url,
            date=date_for_llm,
            content=text[:CONTENT_MAX_LENGTH],
            icp_industry=icp_industry,
            icp_criteria=icp_criteria,
            company_name=company_name
        )
    except Exception as e:
        logger.error(f"LLM verification failed: {e}")
        return False, 0, f"LLM verification error: {str(e)[:100]}", "fabricated"
    
    # If miner submitted no date, force no_date regardless of LLM response
    if not intent_signal.date:
        date_status = "no_date"
        logger.info("No date provided by miner — treating as no_date")

    # ── Programmatic date precision override ──
    # ALL source types: an incorrect date is always 0x (misleads clients).
    #   Correct date (verified)  → 1.0x  (with age-based time decay)
    #   No date on page          → 1.0x for date-not-required / 0.5x for date-required
    #   Incorrect/fabricated date → 0x    (always — wrong data is worse than no data)
    if intent_signal.date and date_status != "fabricated":
        stripped_content = strip_dynamic_boilerplate_dates(
            strip_copyright_founded_years(text[:CONTENT_MAX_LENGTH])
        )
        precision = check_date_precision(intent_signal.date, stripped_content)

        if date_status == "verified":
            if precision == "year_only":
                date_status = "fabricated"
                confidence = 0
                reason = (
                    f"Date precision override: only the year appears in content — "
                    f"specific date {intent_signal.date} was fabricated. {reason}"
                )
                logger.warning(
                    f"❌ Date fabricated: {intent_signal.date} → year_only "
                    f"(month/day manufactured)"
                )
            elif precision == "no_match":
                date_status = "fabricated"
                confidence = 0
                reason = (
                    f"Date precision override: claimed year not found in content at all. "
                    f"{reason}"
                )
                logger.warning(
                    f"❌ Date precision rejection: {intent_signal.date} → no_match "
                    f"(treating as fabricated)"
                )

        elif date_status == "no_date":
            if precision in ("verified", "approximate"):
                date_status = "verified"
                confidence = max(confidence, 70)
                logger.info(
                    f"✓ Date precision upgrade: {intent_signal.date} found on page "
                    f"(LLM had said no_date, precision={precision})"
                )

        if precision in ("verified", "approximate") and date_status not in ("fabricated",):
            logger.info(
                f"✓ Date precision confirmed: {intent_signal.date} → {precision}"
            )
    
    # ── Claim-date coherence check ──
    # If the LLM says the claim is NOT supported by the content, the model
    # fabricated a signal about this page. Two sub-cases:
    #   a) Date appears on page (verified/approximate): the date is incidental —
    #      the model found a page with a date and fabricated a claim about it.
    #   b) Date does NOT appear on page (no_date): the model fabricated both
    #      the claim AND the date — nothing about this signal is real.
    # In either case, the signal is fabricated.
    if not claim_supported and date_status != "fabricated":
        if date_status in ("verified", "approximate"):
            date_status = "fabricated"
            confidence = 0
            reason = (
                f"Claim-date coherence failure: date found on page but "
                f"claim not supported by content. {reason}"
            )
            logger.warning(
                f"❌ Claim-date coherence: unsupported claim + incidental date → fabricated"
            )
        elif date_status == "no_date" and intent_signal.date:
            date_status = "fabricated"
            confidence = 0
            reason = (
                f"Claim-date coherence failure: claim not supported by content "
                f"and claimed date not found on page. {reason}"
            )
            logger.warning(
                f"❌ Claim-date coherence: unsupported claim + missing date → fabricated"
            )

    # Re-apply threshold after potential override
    if date_status == "fabricated":
        confidence = 0
        verified = False
    else:
        verified = verified and confidence >= CONFIDENCE_THRESHOLD

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
    
    logger.info(f"Verification complete: verified={verified}, confidence={confidence}, date_status={date_status}")
    return verified, confidence, reason, date_status


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
    elif source_lower == "wikipedia":
        return await fetch_wikipedia(url)
    else:
        # Default to generic scraping
        return await scrapingdog_generic(url)


# =============================================================================
# Wikipedia Fetcher (free, no ScrapingDog needed)
# =============================================================================

async def fetch_wikipedia(url: str) -> str:
    """
    Fetch Wikipedia content directly via httpx.
    
    Wikipedia is a free public resource - no need to use ScrapingDog credits.
    Already in ALLOWED_NETWORK_DESTINATIONS.
    
    Args:
        url: Wikipedia article URL (e.g., https://en.wikipedia.org/wiki/Aria_Systems)
    
    Returns:
        HTML content as string
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"User-Agent": "LeadPoet-Qualification/1.0"},
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text


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
    # ScrapingDog LinkedIn API supports: profile, company, post
    # Job posting URLs (/jobs/) are NOT supported by the LinkedIn API —
    # they must be scraped via the generic scraper instead.
    if "/jobs/" in url:
        logger.info(f"LinkedIn job URL detected — routing to generic scraper: {url[:80]}")
        return await scrapingdog_generic(url)
    
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
    
    # Job posting URL: /jobs/view/JOBID
    match = re.search(r'/jobs/view/(\d+)', url)
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


async def llm_verify_claim_with_icp(
    claim: str,
    url: str,
    date: str,
    content: str,
    icp_industry: Optional[str] = None,
    icp_criteria: Optional[str] = None,
    company_name: Optional[str] = None
) -> Tuple[bool, int, str, str, bool]:
    """
    Use LLM to verify an intent signal AND check for ICP evidence.
    
    This is the core verification that checks:
    1. Is the claim supported by the URL content?
    2. Does the URL provide evidence the company matches ICP criteria?
    
    Args:
        claim: The intent signal description/claim
        url: Source URL
        date: Claimed date of the signal
        content: Extracted text content from the source (via ScrapingDog)
        icp_industry: Target industry from ICP (e.g., "Healthcare")
        icp_criteria: Additional ICP criteria (e.g., "PE-backed, 50-500 employees")
        company_name: Name of the company being verified
    
    Returns:
        Tuple of (verified: bool, confidence: int 0-100, reason: str, date_status: str, claim_supported: bool)
        date_status is one of: "verified", "no_date", "fabricated"
        claim_supported is the LLM's raw boolean before threshold/ICP adjustments
    """
    # Build ICP context section — only verify INDUSTRY fit from the URL content.
    # Structural fields (employee_count, geography, company_stage) are verified
    # separately by db_verification.py against the leads database.
    icp_context = ""
    if icp_industry:
        icp_context = f"""
ICP INDUSTRY REQUIREMENT:
- Target Industry: {icp_industry}
- Company Being Verified: {company_name or 'Unknown'}

The URL content should provide EVIDENCE that this company operates in or serves
the {icp_industry} industry. Look for:
- Products/services relevant to {icp_industry}
- Industry-specific terminology, clients, or use cases
- Company description mentioning {icp_industry} or closely related fields

If the URL does NOT provide evidence of industry fit, set icp_evidence_found=false.
A job posting for "Software Engineer" does NOT prove a company is in Healthcare.
A generic company page with no industry context is insufficient.

NOTE: Do NOT penalize for missing employee count, geography, or company stage —
those are verified separately from the database.
"""

    prompt = f"""You are verifying an intent signal for a B2B lead generation system.

CLAIMED INTENT: {claim}
SOURCE URL: {url}
CLAIMED DATE: {date}
{icp_context}
URL CONTENT (scraped via ScrapingDog):
{content}

Your task: Determine if the URL content PROVES:
1. The intent claim is real and specific (not generic/templated)
2. The company matches the ICP requirements (if specified)
3. The claimed DATE is reasonable (appears in content or is plausibly recent)

REJECT these GENERIC/TEMPLATED claims (gaming attempts):
- "[Company] is actively operating in [industry]" - Too vague
- "[Company] market activity and company updates" - No specific intent
- "[Company] is expanding/growing" - Generic filler
- Claims that would be true for ANY company

VERIFICATION REQUIREMENTS:
1. Claim must have SPECIFIC details (hiring X role, launched Y product, raised Z funding)
2. Those specific details MUST appear in the scraped content
3. If an ICP industry is specified, the content must PROVE that industry fit
4. The DATE should be found in the content OR be reasonably verifiable. If the claimed date
   looks fabricated (e.g., exactly 14 days ago with no date in content), flag it.
NOTE: Do NOT check for employee count, geography, or company stage — those are verified separately.

DATE VERIFICATION (THREE possible outcomes):
- "verified": Content has a SPECIFIC date/timestamp (with month and day) that matches the claimed date.
  The date must appear with at least month+year precision — a bare year is NOT enough.
- "no_date": Content genuinely has NO dates/timestamps at all, or only has bare years with no
  month/day context. You simply cannot verify the specific date.
- "fabricated": Content has dates that CONTRADICT the claimed date, OR the claimed date shows
  MANUFACTURED PRECISION (see below).

CRITICAL — MANUFACTURED DATE PRECISION (common gaming technique):
A model may find the string "2025" on a page and claim the date "2025-01-01". The year IS on the
page, but the month and day were INVENTED. This is fabrication. Specific rules:
- If content only mentions a YEAR (e.g., "2025", "in 2024") but the claimed date is a specific
  day like "2025-01-01" or "2024-06-15", the month and day were manufactured → "fabricated"
- COPYRIGHT DATES are NOT signal dates. "© 2024" or "Copyright 2025" in a page footer is a
  website attribute, not an intent event. If the ONLY year reference is a copyright notice,
  the date is "fabricated"
- FOUNDING DATES are NOT signal dates. "Founded in 2015" or "Established 2010" is company
  metadata, not a temporal intent signal. If the date derives from a founding year, it is "fabricated"
- First-of-month dates (YYYY-01-01, YYYY-MM-01) are suspicious — real events rarely happen on
  exactly the 1st. If there is no explicit "January 1" or "1st of January" in the content,
  this is likely manufactured precision → "fabricated"

Examples of FABRICATED dates (date_status = "fabricated"):
- Claimed "2025-01-01" but content only mentions "2025" as a year — month/day were invented
- Claimed "2024-06-01" but content has "© 2024" in footer — copyright is not an intent date
- Claimed "2015-01-01" and content says "Founded in 2015" — founding year is not intent
- Claimed "2026-02-04" but content shows article dated "2025-11-15" — dates contradict
- Claimed exactly 14 days ago and page has zero dates — suspiciously convenient timedelta
- Claimed a specific recent date but URL is clearly an old/static page with a visible older date

Examples of VERIFIED dates (date_status = "verified"):
- Content says "Posted January 15, 2026" and claimed date is "2026-01-15" — exact match
- Content says "Published Feb 2026" and claimed date is "2026-02-01" — month matches
- Job posting with datePosted: "2026-01-20" matching claimed "2026-01-20"

Examples of NO DATE (date_status = "no_date"):
- Company homepage with no timestamps anywhere — impossible to verify any date
- Product page or About page with no publication dates
- Content is real and specific but simply undated

Respond with ONLY JSON (no markdown):
{{"verified": true/false, "confidence": 0-100, "reason": "1-2 sentence explanation", "icp_evidence_found": true/false, "date_status": "verified" | "no_date" | "fabricated"}}

Examples:
{{"verified": true, "confidence": 85, "reason": "Content shows hiring for DevOps roles at a healthcare company. Job posted Jan 20, 2026 matches claimed date.", "icp_evidence_found": true, "date_status": "verified"}}
{{"verified": false, "confidence": 30, "reason": "Job posting exists but no evidence this is a healthcare company as ICP requires.", "icp_evidence_found": false, "date_status": "verified"}}
{{"verified": false, "confidence": 10, "reason": "Claim is generic 'actively operating' - no specific intent shown.", "icp_evidence_found": false, "date_status": "no_date"}}
{{"verified": false, "confidence": 20, "reason": "Content dated Nov 2025 but claimed date is Feb 2026. Date appears fabricated.", "icp_evidence_found": true, "date_status": "fabricated"}}
{{"verified": false, "confidence": 15, "reason": "Claimed 2025-01-01 but content only mentions '2025' as a year. Month and day were manufactured.", "icp_evidence_found": true, "date_status": "fabricated"}}
{{"verified": false, "confidence": 10, "reason": "Date derives from copyright footer '© 2024', not an intent event.", "icp_evidence_found": false, "date_status": "fabricated"}}
"""
    
    try:
        response_text = await openrouter_chat(prompt, model="gpt-4o-mini")
        
        # Parse JSON response
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        result = json.loads(response_text)
        
        verified_raw = result.get("verified", False)
        confidence = int(result.get("confidence", 0))
        reason = result.get("reason", "No reason provided")
        icp_evidence = result.get("icp_evidence_found", True)  # Default True if not checking ICP
        
        # Parse date_status (new 3-way field) with fallback to legacy date_verified
        date_status = result.get("date_status")
        if date_status is None:
            legacy = result.get("date_verified", True)
            date_status = "verified" if legacy else "fabricated"
        # Normalize to known values
        if date_status not in ("verified", "no_date", "fabricated"):
            date_status = "verified"
        
        # If industry was specified but no evidence found, reduce confidence
        if icp_industry and not icp_evidence:
            confidence = min(confidence, 30)
            reason = f"No industry evidence found. {reason}"
        
        if date_status == "fabricated":
            # Actively fabricated date (contradicts content or suspiciously convenient)
            # Zero confidence → lead_scorer will zero the ENTIRE lead
            confidence = 0
            reason = f"Date fabrication detected. {reason}"
            logger.warning(f"❌ Date FABRICATED - ZEROING confidence (time decay gaming)")
        elif date_status == "no_date":
            # Content genuinely has no dates — not fabrication, just unverifiable.
            # The CLAIM may still be real (verified_raw stays as LLM reported).
            # Intent will be scored but capped by _score_single_intent_signal.
            logger.info(f"⚠️ No date in content - intent capped but not zeroed")
        
        # Apply confidence threshold
        verified = verified_raw and confidence >= CONFIDENCE_THRESHOLD
        
        return verified, confidence, reason, date_status, verified_raw
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return False, 0, "LLM response parsing error", "fabricated", False
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
) -> list[Tuple[bool, int, str, str]]:
    """
    Verify multiple intent signals (with caching).
    
    Args:
        signals: List of intent signals to verify
    
    Returns:
        List of (verified, confidence, reason, date_status) tuples
    """
    results = []
    for signal in signals:
        try:
            result = await verify_intent_signal(signal)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to verify signal {signal.url}: {e}")
            results.append((False, 0, f"Verification error: {str(e)[:50]}", "fabricated"))
    
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
