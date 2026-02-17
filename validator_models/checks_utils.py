"""
Shared configuration, caches, and helper functions for the validation pipeline.
This is the leaf module — no internal dependencies.
"""

import aiohttp
import asyncio
import json
import os
import re
import unicodedata
import uuid
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from Leadpoet.utils.utils_lead_extraction import get_email, get_company


# ========================================================================
# AREA-CITY MAPPINGS FOR LOCATION VERIFICATION
# ========================================================================

AREA_CITY_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'gateway', 'utils', 'area_city_mappings.json')
_AREA_CITY_MAPPINGS_CACHE = None


def normalize_accents(text: str) -> str:
    """
    Remove accents/diacritics from text for name matching.
    e.g., "José" -> "Jose", "François" -> "Francois"
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')


def load_area_city_mappings() -> dict:
    """Load area-city mappings from JSON file (cached)."""
    global _AREA_CITY_MAPPINGS_CACHE
    if _AREA_CITY_MAPPINGS_CACHE is not None:
        return _AREA_CITY_MAPPINGS_CACHE

    try:
        with open(AREA_CITY_MAPPINGS_PATH, 'r') as f:
            data = json.load(f)
            _AREA_CITY_MAPPINGS_CACHE = data.get('mappings', {})
            return _AREA_CITY_MAPPINGS_CACHE
    except Exception as e:
        print(f"⚠️ Could not load area_city_mappings.json: {e}")
        _AREA_CITY_MAPPINGS_CACHE = {}
        return _AREA_CITY_MAPPINGS_CACHE


def normalize_area_name(area: str) -> str:
    """Normalize area name for matching.
    e.g., "Greater Seattle Area" → "seattle"
    """
    area = area.lower().strip()
    area = area.replace("greater ", "").replace(" metropolitan", "").replace(" metro", "").replace(" area", "")
    return area.strip()


def is_city_in_area(city: str, area: str) -> bool:
    """
    Check if city is within the metropolitan area.

    Args:
        city: Claimed city (e.g., "Cupertino")
        area: LinkedIn location (e.g., "San Francisco Bay Area")

    Returns:
        True if city is in the area's city list
    """
    mappings = load_area_city_mappings()
    if not mappings:
        return False

    city_lower = normalize_accents(city.lower().strip())
    area_norm = normalize_area_name(area)

    # Find matching area in mappings
    for area_key, cities in mappings.items():
        key_norm = normalize_area_name(area_key)
        # Match if normalized names are equal or one contains the other
        if key_norm == area_norm or area_norm in key_norm or key_norm in area_norm:
            # Check if city is in the list (case-insensitive, accent-normalized)
            cities_normalized = [normalize_accents(c.lower()) for c in cities]
            if city_lower in cities_normalized:
                return True

    return False


# Custom exception for API infrastructure failures (should skip lead, not submit)
class EmailVerificationUnavailableError(Exception):
    """Raised when email verification API is unavailable (no credits, bad key, network error, etc.)"""
    pass

load_dotenv()

# ════════════════════════════════════════════════════════════════════
# PROXY CONFIGURATION: Support for containerized validators with proxies
# ════════════════════════════════════════════════════════════════════

# Read proxy configuration from environment variables
HTTP_PROXY_URL = os.environ.get('HTTP_PROXY')
HTTPS_PROXY_URL = os.environ.get('HTTPS_PROXY', HTTP_PROXY_URL)

# Global proxy configuration for all HTTP requests
PROXY_CONFIG = None
if HTTP_PROXY_URL:
    PROXY_CONFIG = {
        'http': HTTP_PROXY_URL,
        'https': HTTPS_PROXY_URL or HTTP_PROXY_URL
    }
    print(f"🌐 Proxy enabled: {HTTP_PROXY_URL[:50]}... (all API requests will use this IP)")

def get_aiohttp_connector():
    """
    Create aiohttp connector with proxy support if configured.

    Returns connector that should be passed to aiohttp.ClientSession()
    """
    if HTTP_PROXY_URL:
        # aiohttp handles proxies via request parameters, not connector
        return None
    return None

# ════════════════════════════════════════════════════════════════════

# MEV removed - always use TrueList for email verification
# Even if MYEMAILVERIFIER_API_KEY is set in environment, we ignore it
MYEMAILVERIFIER_API_KEY = ""  # Hardcoded empty - TrueList is the only email verifier
TRUELIST_API_KEY = os.getenv("TRUELIST_API_KEY", "")

# TrueList Batch Email Validation Configuration
# See: https://apidocs.truelist.io/#tag/Batch-email-validation
TRUELIST_BATCH_POLL_INTERVAL = 10  # seconds between status polls
TRUELIST_BATCH_TIMEOUT = 40 * 60   # 40 minutes in seconds
TRUELIST_BATCH_MAX_RETRIES = 2     # Max retry attempts for errored emails
TRUELIST_BATCH_STRATEGY = "fast"  # "fast" returns more complete results than "accurate"

# Stage 4 & 5: ScrapingDog GSE API + OpenRouter LLM
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")

# Rep Score API keys (Companies House)
COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "")

EMAIL_CACHE_FILE = "email_verification_cache.pkl"
VALIDATION_ARTIFACTS_DIR = "validation_artifacts"

CACHE_TTLS = {
    "dns_head": 24,
    "whois": 90,
    "myemailverifier": 90,
}

API_SEMAPHORE = asyncio.Semaphore(10)

os.makedirs(VALIDATION_ARTIFACTS_DIR, exist_ok=True)

# Commit-Reveal Logic for Trustless Validation

def compute_validation_hashes(decision: str, rep_score: float, evidence: dict, salt: bytes) -> dict:
    """
    Compute commit hashes for validation result.

    Args:
        decision: "approve" or "reject"
        rep_score: Reputation score (0-30)
        evidence: Evidence blob (full automated_checks_data)
        salt: Random salt for commitment

    Returns:
        {
            "decision_hash": "sha256-hex",
            "rep_score_hash": "sha256-hex",
            "evidence_hash": "sha256-hex"
        }
    """
    # Canonicalize evidence (sort keys for determinism)
    evidence_json = json.dumps(evidence, sort_keys=True, default=str)  # Handle datetime objects

    # Compute hashes
    decision_hash = hashlib.sha256(salt + decision.encode()).hexdigest()
    rep_score_hash = hashlib.sha256(salt + str(rep_score).encode()).hexdigest()
    evidence_hash = hashlib.sha256(salt + evidence_json.encode()).hexdigest()

    return {
        "decision_hash": decision_hash,
        "rep_score_hash": rep_score_hash,
        "evidence_hash": evidence_hash
    }

class LRUCache:
    """LRU Cache implementation with TTL support"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.access_order: list = []

    def __contains__(self, key: str) -> bool:
        if key in self.cache:
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return True
        return False

    def __getitem__(self, key: str) -> Any:
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            del self.timestamps[lru_key]

        # Add new item
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
        self.access_order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def is_expired(self, key: str, ttl_hours: int) -> bool:
        if key not in self.timestamps:
            return True
        age = datetime.now() - self.timestamps[key]
        return age.total_seconds() > (ttl_hours * 3600)

    def cleanup_expired(self, ttl_hours: int):
        """Remove expired items from cache"""
        expired_keys = [key for key in list(self.cache.keys()) if self.is_expired(key, ttl_hours)]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            if key in self.access_order:
                self.access_order.remove(key)

# Global cache instance
validation_cache = LRUCache(max_size=1000)

# ========================================================================
# GLOBAL COMPANY LINKEDIN CACHE
# ========================================================================
COMPANY_LINKEDIN_CACHE: Dict[str, Dict] = {}
COMPANY_LINKEDIN_CACHE_TTL_HOURS = 24

def get_company_linkedin_from_cache(company_slug: str) -> Optional[Dict]:
    """
    Get company LinkedIn data from global cache if not expired.

    Args:
        company_slug: The company slug from LinkedIn URL (e.g., "microsoft")

    Returns:
        Cached data dict or None if not cached/expired
    """
    if company_slug not in COMPANY_LINKEDIN_CACHE:
        return None

    cached = COMPANY_LINKEDIN_CACHE[company_slug]
    cached_time = cached.get("timestamp")

    if cached_time:
        # Handle both string (new) and datetime (old/in-memory) formats
        if isinstance(cached_time, str):
            try:
                cached_time = datetime.fromisoformat(cached_time)
            except ValueError:
                # Invalid timestamp, remove from cache
                del COMPANY_LINKEDIN_CACHE[company_slug]
                return None

        # Check if cache has expired
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        if age_hours > COMPANY_LINKEDIN_CACHE_TTL_HOURS:
            # Expired, remove from cache
            del COMPANY_LINKEDIN_CACHE[company_slug]
            return None

    return cached

def set_company_linkedin_cache(company_slug: str, data: Dict):
    """
    Store company LinkedIn data in global cache.

    Args:
        company_slug: The company slug from LinkedIn URL
        data: Dict with company data to cache
    """
    # Add timestamp for TTL (use isoformat string, not datetime object, to avoid JSON serialization issues)
    data["timestamp"] = datetime.now().isoformat()
    COMPANY_LINKEDIN_CACHE[company_slug] = data

    # Limit cache size (simple LRU - remove oldest if over 500 entries)
    if len(COMPANY_LINKEDIN_CACHE) > 500:
        # Remove oldest entry
        oldest_slug = None
        oldest_time = datetime.now()
        for slug, cached_data in COMPANY_LINKEDIN_CACHE.items():
            # Parse ISO string timestamp back to datetime for comparison
            timestamp_str = cached_data.get("timestamp")
            if timestamp_str:
                try:
                    cached_time = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    cached_time = datetime.now()
            else:
                cached_time = datetime.now()
            if cached_time < oldest_time:
                oldest_time = cached_time
                oldest_slug = slug
        if oldest_slug:
            del COMPANY_LINKEDIN_CACHE[oldest_slug]

# ========================================================================
# COMPANY NAME STANDARDIZATION CACHE (JSON file)
# ========================================================================
COMPANY_NAME_CACHE_FILE = os.path.join(os.path.dirname(__file__), "company_name_cache.json")

def load_company_name_cache() -> Dict[str, str]:
    """Load the company name cache from local JSON file."""
    if os.path.exists(COMPANY_NAME_CACHE_FILE):
        try:
            with open(COMPANY_NAME_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading company name cache: {e}")
    return {}

def save_company_name_cache(cache: Dict[str, str]) -> bool:
    """Save the company name cache to local JSON file."""
    try:
        with open(COMPANY_NAME_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ Error saving company name cache: {e}")
        return False

def get_standardized_company_name(company_slug: str) -> Optional[str]:
    """
    Get standardized company name from cache.

    Args:
        company_slug: The company slug from LinkedIn URL (e.g., "23andme")

    Returns:
        Standardized company name or None if not in cache
    """
    cache = load_company_name_cache()
    # Normalize slug to lowercase
    slug_normalized = company_slug.lower().strip()
    return cache.get(slug_normalized)

def set_standardized_company_name(company_slug: str, standardized_name: str) -> bool:
    """
    Save standardized company name to cache.

    Args:
        company_slug: The company slug from LinkedIn URL (e.g., "23andme")
        standardized_name: The official company name from LinkedIn (e.g., "23andMe")

    Returns:
        True if saved successfully, False otherwise
    """
    cache = load_company_name_cache()
    # Normalize slug to lowercase
    slug_normalized = company_slug.lower().strip()
    cache[slug_normalized] = standardized_name
    success = save_company_name_cache(cache)
    if success:
        print(f"   💾 Cached company name: '{slug_normalized}' → '{standardized_name}'")
    return success


def extract_root_domain(website: str) -> str:
    """Extract the root domain from a website URL, removing www. prefix"""
    if not website:
        return ""

    # Parse the URL to get the domain
    if website.startswith(("http://", "https://")):
        domain = urlparse(website).netloc
    else:
        # Handle bare domains like "firecrawl.dev" or "www.firecrawl.dev"
        domain = website.strip("/")

    # Remove www. prefix if present
    if domain.startswith("www."):
        domain = domain[4:]  # Remove "www."

    return domain


def get_cache_key(prefix: str, identifier: str) -> str:
    """Generate consistent cache key for validation results"""
    return f"{prefix}_{identifier}"


async def store_validation_artifact(lead_data: dict, validation_result: dict, stage: str):
    """Store validation result as artifact for analysis"""
    try:
        timestamp = datetime.now().isoformat()
        artifact_data = {
            "timestamp": timestamp,
            "stage": stage,
            "lead_data": lead_data,
            "validation_result": validation_result,
        }

        filename = f"validation_{stage}_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        filepath = os.path.join(VALIDATION_ARTIFACTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(artifact_data, f, indent=2, default=str)

        print(f"✅ Validation artifact stored: {filename}")
    except Exception as e:
        print(f"⚠️ Failed to store validation artifact: {e}")

async def log_validation_metrics(lead_data: dict, validation_result: dict, stage: str):
    """Log validation metrics for monitoring and analysis"""
    try:
        # Extract key metrics
        email = get_email(lead_data)
        company = get_company(lead_data)
        passed = validation_result.get("passed", False)
        reason = validation_result.get("reason", "Unknown")

        # Log to console for now (can be extended to database/metrics service)
        status_icon = "✅" if passed else "❌"
        print(f"{status_icon} Stage {stage}: {email} @ {company} - {reason}")

        # Store metrics in cache for aggregation
        metrics_key = f"metrics_{stage}_{datetime.now().strftime('%Y%m%d')}"
        current_metrics = validation_cache.get(metrics_key, {"total": 0, "passed": 0, "failed": 0})

        current_metrics["total"] += 1
        if passed:
            current_metrics["passed"] += 1
        else:
            current_metrics["failed"] += 1

        validation_cache[metrics_key] = current_metrics

    except Exception as e:
        print(f"⚠️ Failed to update metrics: {e}")

    try:
        # Log to file for persistence
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "email": get_email(lead_data),
            "company": get_company(lead_data),
            "passed": validation_result.get("passed", False),
            "reason": validation_result.get("reason", "Unknown"),
        }

        log_file = os.path.join(VALIDATION_ARTIFACTS_DIR, "validation_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")  # Handle datetime objects

    except Exception as e:
        print(f"⚠️ Failed to log validation metrics: {e}")

async def api_call_with_retry(session, url, params=None, max_retries=3, base_delay=1):
    """Make API call with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            # Pass proxy if configured (aiohttp accepts proxy as string URL)
            async with session.get(url, params=params, timeout=10, proxy=HTTP_PROXY_URL) as response:
                return response
        except Exception as e:
            if attempt == max_retries - 1:
                # All retries exhausted, raise descriptive exception
                context_info = f"URL: {url}"
                if params:
                    context_info += f", Params: {params}"
                raise RuntimeError(
                    f"API call to {url} failed after {max_retries} attempts. {context_info}"
                ) from e
            delay = base_delay * (2**attempt)  # Exponential backoff
            await asyncio.sleep(delay)
