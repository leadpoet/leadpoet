# validator_models/automated_checks.py

import aiohttp
import asyncio
import dns.resolver
import pickle
import os
import re
import uuid
import whois
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from disposable_email_domains import blocklist as DISPOSABLE_DOMAINS

# Load environment variables from .env file
load_dotenv()

# Environment variables for new APIs
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "YOUR_HUNTER_API_KEY")
ZEROBOUNCE_API_KEY = os.getenv("ZEROBOUNCE_API_KEY", "YOUR_ZEROBOUNCE_API_KEY")

GOOGLE_API_KEY = os.getenv("GSE_API_KEY", "YOUR_GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GSE_CX", "YOUR_GOOGLE_CSE_ID")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

# LLM Models for validation checks
AVAILABLE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "moonshotai/kimi-k2:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free"
]

FALLBACK_MODELS = [
    "mistralai/mistral-7b-instruct",
]

# Constants
EMAIL_CACHE_FILE = "email_verification_cache.pkl"
VALIDATION_ARTIFACTS_DIR = "validation_artifacts"

# Cache TTLs in hours
CACHE_TTLS = {
    "dns_head": 24,
    "whois": 90,  # Used for WHOIS domain age lookups
    "zerobounce": 90,  
}

# Rate limiting semaphore - limit concurrent API calls
API_SEMAPHORE = asyncio.Semaphore(10)  # Max 10 concurrent API calls

# Create validation artifacts directory if it doesn't exist
os.makedirs(VALIDATION_ARTIFACTS_DIR, exist_ok=True)

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
        email = lead_data.get("Email 1", lead_data.get("email", ""))
        company = lead_data.get("Company", lead_data.get("company", ""))
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
            "email": lead_data.get("Email 1", lead_data.get("email", "")),
            "company": lead_data.get("Company", lead_data.get("company", "")),
            "passed": validation_result.get("passed", False),
            "reason": validation_result.get("reason", "Unknown"),
        }
        
        log_file = os.path.join(VALIDATION_ARTIFACTS_DIR, "validation_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"⚠️ Failed to log validation metrics: {e}")

async def api_call_with_retry(session, url, params=None, max_retries=3, base_delay=1):
    """Make API call with exponential backoff retry logic"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=10) as response:
                return response
        except Exception as e:
            last_exception = e
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

# Stage 0: Basic Hardcoded Checks

async def check_email_regex(lead: dict) -> Tuple[bool, str]:
    """Check email format using RFC-5322 simplified regex"""
    start_time = datetime.now()
    try:
        email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
        if not email:
            return False, "No email provided"
        
        # RFC-5322 simplified regex
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        is_valid = bool(re.match(pattern, email))
        reason = "Valid email format" if is_valid else "Invalid email format"
        
        # Cache result
        cache_key = f"email_regex:{email}"
        validation_cache[cache_key] = (is_valid, reason)
        
        # Log metrics
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        await log_validation_metrics(lead, {"passed": is_valid, "reason": reason}, "email_regex")
        
        return is_valid, reason
    except Exception as e:
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        await log_validation_metrics(lead, {"passed": False, "reason": str(e)}, "email_regex")
        raise e

async def check_domain_age(lead: dict) -> Tuple[bool, str]:
    """Check domain age using WHOIS lookup"""
    website = lead.get("Website", lead.get("website", ""))
    if not website:
        return False, "No website provided"
    
    domain = extract_root_domain(website)
    if not domain:
        return False, "Invalid website format"
    
    cache_key = f"domain_age:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois"]):
        return validation_cache[cache_key]
    
    try:
        # Implement actual WHOIS lookup
        def get_domain_age_sync(domain_name):
            try:
                w = whois.whois(domain_name)
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        creation_date = w.creation_date[0]
                    else:
                        creation_date = w.creation_date
                    
                    age_days = (datetime.now() - creation_date).days
                    min_age_days = 7  # 7 days minimum
                    
                    if age_days >= min_age_days:
                        return (True, f"Domain age: {age_days} days (minimum: {min_age_days})")
                    else:
                        return (False, f"Domain too new: {age_days} days (minimum: {min_age_days})")
                else:
                    return False, "Could not determine domain creation date"
            except Exception as e:
                return False, f"WHOIS lookup failed: {str(e)}"
        
        # Run WHOIS lookup in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_domain_age_sync, domain)
        validation_cache[cache_key] = result
        return result
        
    except Exception as e:
        result = (False, f"Domain age check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_mx_record(lead: dict) -> Tuple[bool, str]:
    """Check if domain has MX records"""
    website = lead.get("Website", lead.get("website", ""))
    if not website:
        return False, "No website provided"
    
    domain = extract_root_domain(website)
    if not domain:
        return False, "Invalid website format"
    
    cache_key = f"mx_record:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]
    
    try:
        result = await check_domain_existence(domain)
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, f"MX record check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_spf_dmarc(lead: dict) -> Tuple[bool, str]:
    """
    Check SPF and DMARC DNS records (SOFT check - always passes, appends data to lead)
    
    This is a SOFT check that:
    - Checks DNS TXT record for v=spf1
    - Checks DNS TXT record at _dmarc.{domain} for v=DMARC1
    - Checks DMARC policy for p=quarantine or p=reject
    - Appends results to lead but NEVER rejects
    
    Args:
        lead: Dict containing email/website
        
    Returns:
        (True, str): Always passes, with informational message
    """
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    if not email:
        # No email to check - append default values
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        return True, "No email provided for SPF/DMARC check (SOFT - passed)"
    
    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead["has_spf"] = False
            lead["has_dmarc"] = False
            lead["dmarc_policy_strict"] = False
            return True, "Invalid email format (SOFT - passed)"
    except (IndexError, AttributeError):
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        return True, "Invalid email format (SOFT - passed)"
    
    cache_key = f"spf_dmarc:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        cached_data = validation_cache[cache_key]
        # Apply cached values to lead
        lead["has_spf"] = cached_data.get("has_spf", False)
        lead["has_dmarc"] = cached_data.get("has_dmarc", False)
        lead["dmarc_policy_strict"] = cached_data.get("dmarc_policy_strict", False)
        return True, cached_data.get("message", "SPF/DMARC check (cached)")
    
    try:
        # Initialize results
        has_spf = False
        has_dmarc = False
        dmarc_policy_strict = False
        
        # Run DNS lookups in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def check_spf_sync(domain_name):
            """Check if domain has SPF record"""
            try:
                txt_records = dns.resolver.resolve(domain_name, "TXT")
                for record in txt_records:
                    txt_string = "".join([s.decode() if isinstance(s, bytes) else s for s in record.strings])
                    if "v=spf1" in txt_string.lower():
                        return True
                return False
            except Exception:
                return False
        
        def check_dmarc_sync(domain_name):
            """Check if domain has DMARC record and return policy strictness"""
            try:
                dmarc_domain = f"_dmarc.{domain_name}"
                txt_records = dns.resolver.resolve(dmarc_domain, "TXT")
                for record in txt_records:
                    txt_string = "".join([s.decode() if isinstance(s, bytes) else s for s in record.strings])
                    txt_lower = txt_string.lower()
                    
                    if "v=dmarc1" in txt_lower:
                        # Check if policy is strict (quarantine or reject)
                        is_strict = "p=quarantine" in txt_lower or "p=reject" in txt_lower
                        return True, is_strict
                return False, False
            except Exception:
                return False, False
        
        # Execute DNS checks
        has_spf = await loop.run_in_executor(None, check_spf_sync, domain)
        has_dmarc, dmarc_policy_strict = await loop.run_in_executor(None, check_dmarc_sync, domain)
        
        # Append results to lead (SOFT check data)
        lead["has_spf"] = has_spf
        lead["has_dmarc"] = has_dmarc
        lead["dmarc_policy_strict"] = dmarc_policy_strict
        
        # Create informational message
        spf_status = "✓" if has_spf else "✗"
        dmarc_status = "✓" if has_dmarc else "✗"
        policy_status = "✓ (strict)" if dmarc_policy_strict else ("✓ (permissive)" if has_dmarc else "✗")
        
        message = f"SPF: {spf_status}, DMARC: {dmarc_status}, Policy: {policy_status}"
        
        # Cache the results
        cache_data = {
            "has_spf": has_spf,
            "has_dmarc": has_dmarc,
            "dmarc_policy_strict": dmarc_policy_strict,
            "message": message
        }
        validation_cache[cache_key] = cache_data
        
        print(f"📧 SPF/DMARC Check (SOFT): {domain} - {message}")
        
        # ALWAYS return True (SOFT check never fails)
        return True, message
        
    except Exception as e:
        # On any error, append False values and pass
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        
        message = f"SPF/DMARC check error (SOFT - passed): {str(e)}"
        print(f"⚠️ {message}")
        
        # Cache the error result
        cache_data = {
            "has_spf": False,
            "has_dmarc": False,
            "dmarc_policy_strict": False,
            "message": message
        }
        validation_cache[cache_key] = cache_data
        
        # ALWAYS return True (SOFT check never fails)
        return True, message

async def check_head_request(lead: dict) -> Tuple[bool, str]:
    """Wrapper around existing verify_company function"""
    website = lead.get("Website", lead.get("website", ""))
    if not website:
        return False, "No website provided"
    
    domain = extract_root_domain(website)
    if not domain:
        return False, "Invalid website format"
    
    cache_key = f"head_request:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]
    
    try:
        result = await verify_company(domain)
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, f"HEAD request check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_disposable(lead: dict) -> Tuple[bool, str]:
    """Check if email domain is disposable"""
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    if not email:
        return False, "No email provided"
    
    cache_key = f"disposable:{email}"
    if cache_key in validation_cache:
        return validation_cache[cache_key]
    
    try:
        is_disposable, reason = await is_disposable_email(email)
        # For validation pipeline: return True if check PASSES (email is NOT disposable)
        # return False if check FAILS (email IS disposable)
        passed = not is_disposable
        validation_cache[cache_key] = (passed, reason)
        return passed, reason
    except Exception as e:
        result = (False, f"Disposable check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_dnsbl(lead: dict) -> Tuple[bool, str]:
    """
    Check if lead's email domain is listed in Spamhaus DBL.
    
    Args:
        lead: Dict containing email field
        
    Returns:
        (bool, str): (is_valid, reason_if_invalid)
    """
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    if not email:
        return False, "No email provided"
    
    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            return True, "Invalid email format - handled by other checks"
    except (IndexError, AttributeError):
        return True, "Invalid email format - handled by other checks"
    
    # Use root domain extraction helper
    root_domain = extract_root_domain(domain)
    if not root_domain:
        return True, "Could not extract root domain"
    
    cache_key = f"dnsbl_{root_domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]
    
    try:
        async with API_SEMAPHORE:
            # Perform Spamhaus DBL lookup
            query = f"{root_domain}.dbl.spamhaus.org"
            
            # Run DNS lookup in executor to avoid blocking
            loop = asyncio.get_event_loop()
            def dns_lookup():
                try:
                    dns.resolver.resolve(query, "A")
                    return True  # Record exists = domain is blacklisted
                except dns.resolver.NXDOMAIN:
                    return False  # No record = domain is clean
                except Exception as e:
                    # On any DNS error, default to valid (don't block on infrastructure issues)
                    print(f"⚠️ DNS lookup error for {query}: {e}")
                    return False
            
            is_blacklisted = await loop.run_in_executor(None, dns_lookup)
            
            if is_blacklisted:
                result = (False, f"Domain {root_domain} blacklisted in Spamhaus DBL")
                print(f"❌ DNSBL: Domain {root_domain} found in Spamhaus blacklist")
            else:
                result = (True, f"Domain {root_domain} not in Spamhaus DBL")
                print(f"✅ DNSBL: Domain {root_domain} clean")
            
            validation_cache[cache_key] = result
            return result
            
    except Exception as e:
        # On any unexpected error, default to valid and cache the result
        result = (True, f"DNSBL check failed (defaulting to valid): {str(e)}")
        validation_cache[cache_key] = result
        print(f"⚠️ DNSBL check error for {root_domain}: {e}")
        return result

# Stage 2: ZeroBounce Check

async def check_zerobounce_email(lead: dict) -> Tuple[bool, str]:
    """Check email validity **and AI-score** using ZeroBounce API"""
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    if not email:
        return False, "No email provided"

    cache_key = f"zerobounce:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["zerobounce"]):
        return validation_cache[cache_key]

    try:
        # Mock-mode fallback - both validation and scoring work
        if "YOUR_ZEROBOUNCE_API_KEY" in ZEROBOUNCE_API_KEY:
            # Mock both API calls
            score = -1  # Mock AI score if no API key
            lead["email_score"] = score
            print(f"📊 ZeroBounce AI score for {email}: {score} (MOCK)")
            result = (True, f"Mock pass - ZeroBounce check, score={score}")
            validation_cache[cache_key] = result
            return result

        async with API_SEMAPHORE:
            async with aiohttp.ClientSession() as session:
                # 1️⃣ standard validation FIRST
                url_validate = "https://api.zerobounce.net/v2/validate"
                params_val   = {"api_key": ZEROBOUNCE_API_KEY, "email": email}
                async with session.get(url_validate, params=params_val, timeout=10) as resp_val:
                    data_val  = await resp_val.json()
                    status    = data_val.get("status", "unknown")

                # Check for problematic email statuses and fail immediately
                problematic_statuses = ["do_not_mail", "spamtrap", "abuse"]
                if status in problematic_statuses:
                    status_messages = {
                        "do_not_mail": "Email marked as do_not_mail",
                        "spamtrap": "Email is a spamtrap/honeypot - do not mail",
                        "abuse": "Email marked as abuse"
                    }
                    result = (False, status_messages.get(status, f"Email has problematic status: {status}"))
                    validation_cache[cache_key] = result
                    return result

                # Check if validation passed
                validation_passed = status in ["valid", "catch-all"]
                
                if validation_passed:
                    # 2️⃣ A.I. email score - ONLY if validation passed
                    url_score  = "https://api-us.zerobounce.net/v2/scoring"
                    params_scr = {"api_key": ZEROBOUNCE_API_KEY, "email": email}
                    async with session.get(url_score, params=params_scr, timeout=10) as resp_scr:
                        data_scr = await resp_scr.json()
                        score    = float(data_scr.get("score", -1))   # 0-10 or -1 on error
                    
                    # Store score directly on the lead so downstream code can use it
                    lead["email_score"] = score
                    print(f"📊 ZeroBounce AI score for {email}: {score}")
                else:
                    # Validation failed - don't make scoring API call
                    score = -1  # Indicate no score was obtained

        # Decide pass/fail using the original validation status
        if status == "valid":
            result = (True, f"Email validated (valid), score={score}")
        elif status == "catch-all":
            # BRD: IF catch-all, accept only IF DNS TXT record contains v=spf1
            has_spf = lead.get("has_spf", False)
            if has_spf:
                result = (True, f"Email validated (catch-all with SPF), score={score}")
            else:
                result = (False, f"Email is catch-all without SPF record")
        elif status == "invalid":
            result = (False, f"Email marked invalid")
        else:
            result = (True, f"Email status {status}, score={score} (assumed valid)")

        validation_cache[cache_key] = result
        return result

    except Exception as e:
        result = (False, f"ZeroBounce check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

# Stage 3: Google LLM Checks

async def google_search(query: str, max_results=3) -> str:
    """
    Perform Google Custom Search and return concatenated snippets.
    
    Args:
        query: Search query string
        max_results: Number of results to fetch (default 3)
        
    Returns:
        str: Concatenated snippet text from search results
    """
    # Check for required environment variables
    if "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY or "YOUR_GOOGLE_CSE_ID" in GOOGLE_CSE_ID:
        print("⚠️ Google API credentials not configured - skipping search")
        return ""
    
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("⚠️ Google API credentials missing - skipping search")
        return ""
    
    try:
        async with API_SEMAPHORE:
            # Use aiohttp for the API call
            async with aiohttp.ClientSession() as session:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CSE_ID,
                    "q": query,
                    "num": min(max_results, 10)  # Google API max is 10
                }
                
                # Make the API call directly (don't use api_call_with_retry)
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("items", [])
                        
                        # Extract and concatenate snippets
                        snippets = []
                        for item in items[:max_results]:
                            snippet = item.get("snippet", "")
                            if snippet:
                                snippets.append(snippet)
                        
                        result = " ".join(snippets)
                        print(f"🔍 Google Search: Found {len(snippets)} snippets for query: {query[:50]}...")
                        return result
                    else:
                        print(f"⚠️ Google Search API error: {response.status}")
                        return ""
                    
    except Exception as e:
        print(f"⚠️ Google Search error: {e}")
        return ""

async def call_llm(prompt: str) -> dict:
    """
    Call LLM with prompt and return parsed JSON response.
    
    Args:
        prompt: LLM prompt string
        
    Returns:
        dict: Parsed JSON response from LLM
    """
    if not OPENROUTER_KEY or "YOUR_OPENROUTER_API_KEY" in OPENROUTER_KEY:
        print("⚠️ OpenRouter API key not configured - returning empty result")
        return {}
    
    def _extract_json(response_text: str) -> dict:
        """Extract JSON from LLM response text"""
        try:
            txt = response_text.strip()
            if txt.startswith("```"):
                txt = txt.strip("`").lstrip("json").strip()
            
            # Find JSON object
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start == -1 or end == 0:
                return {}
            
            json_str = txt[start:end]
            return json.loads(json_str)
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}")
            return {}
    
    def _try_model(model_name: str) -> dict:
        """Try calling a specific model"""
        try:
            import requests
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=15
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _extract_json(content)
        except Exception as e:
            print(f"⚠️ LLM model {model_name} failed: {e}")
            return {}
    
    # Try primary model (Gemini 2 Flash)
    try:
        result = _try_model("google/gemini-2.0-flash-exp:free")
        if result:
            return result
    except Exception as e:
        print(f"⚠️ Primary LLM model failed: {e}")
    
    # Try fallback models
    import random
    for fallback_model in random.sample(FALLBACK_MODELS, k=len(FALLBACK_MODELS)):
        try:
            result = _try_model(fallback_model)
            if result:
                print(f"🔄 LLM fallback model {fallback_model} succeeded")
                return result
        except Exception as e:
            print(f"⚠️ LLM fallback {fallback_model} failed: {e}")
    
    # All models failed
    print("⚠️ All LLM models failed - returning empty result")
    return {}

async def check_llm_contact_match(lead: dict) -> Tuple[bool, str]:
    """
    HARD: Verify contact works at company using Google search + LLM
    SOFT: Add role & location alignment data to lead
    
    Args:
        lead: Dict containing contact info (name, email, company, role, location)
        
    Returns:
        (bool, str): (is_valid, reason_if_invalid)
    """
    # Extract contact information
    first_name = lead.get("First", lead.get("First Name", lead.get("first", "")))
    last_name = lead.get("Last", lead.get("Last Name", lead.get("last", "")))
    company_name = lead.get("Business", lead.get("Company", lead.get("business", "")))
    role = lead.get("role", lead.get("Role", ""))
    location = lead.get("region", lead.get("Region", lead.get("location", "")))
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    
    if not email:
        return False, "No email provided"
    
    # Build contact name
    contact_name = f"{first_name} {last_name}".strip()
    if not contact_name or not company_name:
        return True, "Insufficient data for contact verification"
    
    cache_key = f"llm_contact:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["zerobounce"]):
        cached_result = validation_cache[cache_key]
        # Apply cached soft annotations
        if len(cached_result) > 2:
            lead["llm_role_match"] = cached_result[2].get("role_match", False)
            lead["llm_location_match"] = cached_result[2].get("location_match", False)
        return cached_result[0], cached_result[1]
    
    try:
        # Build search query
        search_query = f'"{contact_name}" "{company_name}"'
        print(f"🔍 LLM Contact Search: {search_query}")
        
        # Get search snippets
        snippets = await google_search(search_query)
        if not snippets:
            result = (True, "No search results - assuming valid")
            validation_cache[cache_key] = result
            return result
        
        # Build LLM prompt
        prompt = f"""
Based on the following search results, analyze if this contact works at the specified company and verify role/location alignment.

CONTACT INFO:
- Name: {contact_name}
- Company: {company_name}
- Role: {role}
- Location: {location}

SEARCH RESULTS:
{snippets[:1500]}

Please respond with ONLY a JSON object in this exact format:
{{
    "works_at_company": true or false,
    "role_match": true or false,
    "location_match": true or false
}}

Base your decision on:
- works_at_company: Does the person appear to work at the specified company?
- role_match: Does their role align with the provided role information?
- location_match: Does their location align with the provided location?
"""
        
        # Call LLM
        llm_result = await call_llm(prompt)
        
        # Extract results with defaults
        works_at_company = llm_result.get("works_at_company", True)  # Default to valid
        role_match = llm_result.get("role_match", False)
        location_match = llm_result.get("location_match", False)
        
        # Apply SOFT annotations to lead
        lead["llm_role_match"] = role_match
        lead["llm_location_match"] = location_match
        
        # HARD decision
        if not works_at_company:
            result = (False, "Contact does not work at specified company")
        else:
            result = (True, "Contact verification passed")
        
        # Cache result with soft annotations
        validation_cache[cache_key] = (result[0], result[1], {"role_match": role_match, "location_match": location_match})
        
        print(f"✅ LLM Contact Check: {contact_name} @ {company_name} - {result[1]}")
        return result
        
    except Exception as e:
        result = (True, f"LLM contact check failed (defaulting to valid): {str(e)}")
        validation_cache[cache_key] = result
        print(f"⚠️ LLM contact check error: {e}")
        return result

async def check_llm_company_match(lead: dict) -> Tuple[bool, str]:
    """
    HARD: Verify company details using Google search + LLM
    SOFT: Add LinkedIn match data if available
    
    Args:
        lead: Dict containing company info (company_name, industry, location, linkedin_url)
        
    Returns:
        (bool, str): (is_valid, reason_if_invalid)
    """
    company_name = lead.get("Business", lead.get("Company", lead.get("business", "")))
    industry = lead.get("Industry", lead.get("industry", ""))
    location = lead.get("region", lead.get("Region", lead.get("location", "")))
    website = lead.get("Website", lead.get("website", ""))
    linkedin_url = lead.get("linkedin", lead.get("LinkedIn", ""))
    
    if not company_name:
        return False, "No company name provided"
    
    cache_key = f"llm_company:{company_name.lower().replace(' ', '_')}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["zerobounce"]):
        cached_result = validation_cache[cache_key]
        # Apply cached soft annotations
        if len(cached_result) > 2 and linkedin_url:
            lead["llm_linkedin_match"] = cached_result[2].get("linkedin_match", False)
        return cached_result[0], cached_result[1]
    
    try:
        # Build search query
        search_query = f'"{company_name}" {industry} {location}'.strip()
        print(f"🔍 LLM Company Search: {search_query}")
        
        # Get search snippets
        snippets = await google_search(search_query)
        if not snippets:
            result = (True, "No search results - assuming valid")
            validation_cache[cache_key] = result
            return result
        
        # Build LLM prompt
        prompt = f"""
Based on the following search results, verify if the company information matches what was found online.

COMPANY INFO:
- Company Name: {company_name}
- Industry: {industry}
- Location: {location}
- Website: {website}
- LinkedIn: {linkedin_url}

SEARCH RESULTS:
{snippets[:1500]}

Please respond with ONLY a JSON object in this exact format:
{{
    "company_name_match": true or false,
    "industry_match": true or false,
    "location_match": true or false,
    "linkedin_match": true or false
}}

Base your decision on:
- company_name_match: Does the company name closely match what's found online? (Allow for minor variations)
- industry_match: Does the industry classification align with the company's actual business?
- location_match: Does the location align with where the company operates?
- linkedin_match: If LinkedIn URL provided, does it match the first search result or company info?
"""
        
        # Call LLM
        llm_result = await call_llm(prompt)
        
        # Extract results with defaults
        company_name_match = llm_result.get("company_name_match", True)  # Default to valid
        industry_match = llm_result.get("industry_match", True)
        location_match = llm_result.get("location_match", True)
        linkedin_match = llm_result.get("linkedin_match", False)
        
        # Apply SOFT annotations to lead (only if LinkedIn URL exists)
        if linkedin_url:
            lead["llm_linkedin_match"] = linkedin_match
        
        # HARD decisions: ALL must be true
        failed_checks = []
        if not company_name_match:
            failed_checks.append("company name")
        if not industry_match:
            failed_checks.append("industry")
        if not location_match:
            failed_checks.append("location")
        
        if failed_checks:
            result = (False, f"Company details verification failed: {', '.join(failed_checks)}")
        else:
            result = (True, "Company verification passed")
        
        # Cache result with soft annotations
        validation_cache[cache_key] = (result[0], result[1], {"linkedin_match": linkedin_match})
        
        print(f"✅ LLM Company Check: {company_name} - {result[1]}")
        return result
        
    except Exception as e:
        result = (True, f"LLM company check failed (defaulting to valid): {str(e)}")
        validation_cache[cache_key] = result
        print(f"⚠️ LLM company check error: {e}")
        return result

async def check_icp_evidence(lead: dict) -> Tuple[bool, str]:
    """
    ICP Evidence Check (SOFT) - Search for ICP evidence confirmation
    
    This is a SOFT check that:
    - Searches (Company Name + ICP Evidence) using Google
    - Uses LLM to analyze search results and identify sources confirming ICP evidence
    - Appends results to lead but NEVER rejects
    
    Args:
        lead: Dict containing company info and ICP evidence
        
    Returns:
        (True, str): Always passes, with informational message
    """
    company_name = lead.get("Business", lead.get("Company", lead.get("business", "")))
    icp_evidence = lead.get("icp_evidence", lead.get("ICP Evidence", ""))
    
    # Initialize result - will always be appended to lead
    lead["icp_evidence_confirmed"] = False
    lead["icp_evidence_sources"] = []
    
    if not company_name:
        return True, "No company name for ICP evidence check (SOFT - passed)"
    
    if not icp_evidence:
        return True, "No ICP evidence to verify (SOFT - passed)"
    
    cache_key = f"icp_evidence:{company_name.lower().replace(' ', '_')}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["zerobounce"]):
        cached_data = validation_cache[cache_key]
        # Apply cached values to lead
        lead["icp_evidence_confirmed"] = cached_data.get("confirmed", False)
        lead["icp_evidence_sources"] = cached_data.get("sources", [])
        if "confidence" in cached_data:
            lead["icp_evidence_confidence"] = cached_data["confidence"]
        return True, cached_data.get("message", "ICP evidence check (cached)")
    
    try:
        # Build search query
        search_query = f'"{company_name}" {icp_evidence}'
        print(f"🔍 ICP Evidence Search: {search_query[:80]}...")
        
        # Get search snippets
        snippets = await google_search(search_query, max_results=5)
        if not snippets:
            message = "No search results for ICP evidence (SOFT - passed)"
            cache_data = {
                "confirmed": False,
                "sources": [],
                "message": message
            }
            validation_cache[cache_key] = cache_data
            return True, message
        
        # Build LLM prompt
        prompt = f"""
Based on the following search results, determine if there are any sources that confirm the ICP (Ideal Customer Profile) evidence for this company.

COMPANY: {company_name}
ICP EVIDENCE: {icp_evidence}

SEARCH RESULTS:
{snippets[:2000]}

Please respond with ONLY a JSON object in this exact format:
{{
    "evidence_confirmed": true or false,
    "sources": ["source1", "source2", ...],
    "confidence": "high" or "medium" or "low"
}}

Base your decision on:
- evidence_confirmed: Are there credible sources in the search results that confirm the ICP evidence?
- sources: List of specific sources/websites that confirm the evidence (max 3)
- confidence: Your confidence level in the confirmation
"""
        
        # Call LLM
        llm_result = await call_llm(prompt)
        
        # Extract results with defaults
        evidence_confirmed = llm_result.get("evidence_confirmed", False)
        sources = llm_result.get("sources", [])
        confidence = llm_result.get("confidence", "low")
        
        # Apply SOFT annotations to lead
        lead["icp_evidence_confirmed"] = evidence_confirmed
        lead["icp_evidence_sources"] = sources[:3]  # Limit to 3 sources
        lead["icp_evidence_confidence"] = confidence
        
        # Create informational message
        if evidence_confirmed:
            message = f"ICP evidence confirmed ({confidence} confidence) - {len(sources)} sources"
        else:
            message = f"ICP evidence not confirmed ({confidence} confidence)"
        
        # Cache the results
        cache_data = {
            "confirmed": evidence_confirmed,
            "sources": sources[:3],
            "confidence": confidence,
            "message": message
        }
        validation_cache[cache_key] = cache_data
        
        print(f"📊 ICP Evidence Check (SOFT): {company_name} - {message}")
        
        # ALWAYS return True (SOFT check never fails)
        return True, message
        
    except Exception as e:
        # On any error, append False values and pass
        lead["icp_evidence_confirmed"] = False
        lead["icp_evidence_sources"] = []
        lead["icp_evidence_confidence"] = "error"
        
        message = f"ICP evidence check error (SOFT - passed): {str(e)}"
        print(f"⚠️ {message}")
        
        # Cache the error result
        cache_data = {
            "confirmed": False,
            "sources": [],
            "confidence": "error",
            "message": message
        }
        validation_cache[cache_key] = cache_data
        
        # ALWAYS return True (SOFT check never fails)
        return True, message

# Main validation pipeline

async def run_automated_checks(lead: dict) -> Tuple[bool, str]:
    """Run all automated checks in stages, returning (passed, reason)"""
    
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    company = lead.get("Company", "")
    
    # ========================================================================
    # Stage 0: Hardcoded Checks (MIXED)
    # - Deduplication (handled in validate_lead_list)
    # - Email Regex, Disposable, HEAD Request
    # ========================================================================
    print(f"🔍 Stage 0: Hardcoded checks for {email} @ {company}")
    checks_stage0 = [
        check_email_regex,      # RFC-5322 regex validation (HARD)
        check_disposable,       # Filter throwaway email providers (HARD)
        check_head_request,     # Test website accessibility (HARD)
    ]
    
    for check_func in checks_stage0:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 0 failed: {reason}")
            return False, f"Stage 0 failed: {reason}"
    
    print(f"   ✅ Stage 0 passed")
    
    # ========================================================================
    # Stage 1: DNS Layer (MIXED)
    # - Domain Age, MX Record (HARD)
    # - SPF/DMARC (SOFT - always passes, appends data)
    # ========================================================================
    print(f"🔍 Stage 1: DNS layer checks for {email} @ {company}")
    checks_stage1 = [
        check_domain_age,       # WHOIS lookup, must be ≥7 days old (HARD)
        check_mx_record,        # Verify email domain has mail server (HARD)
        check_spf_dmarc,        # DNS TXT records for SPF/DMARC (SOFT)
    ]
    
    for check_func in checks_stage1:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 1 failed: {reason}")
            return False, f"Stage 1 failed: {reason}"
    
    print(f"   ✅ Stage 1 passed")
    
    # ========================================================================
    # Stage 2: Lightweight Domain Reputation Checks (HARD)
    # - DNSBL (Domain Block List) - Spamhaus DBL lookup
    # ========================================================================
    print(f"🔍 Stage 2: Domain reputation checks for {email} @ {company}")
    passed, reason = await check_dnsbl(lead)
    if not passed:
        print(f"   ❌ Stage 2 failed: {reason}")
        return False, f"Stage 2 failed: {reason}"
    
    print(f"   ✅ Stage 2 passed")
    
    # ========================================================================
    # Stage 3: ZeroBounce Check (MIXED)
    # - Email verification (HARD): Pass IF valid, IF catch-all accept only IF SPF
    # - Email scoring (SOFT): Append score to lead
    # ========================================================================
    print(f"🔍 Stage 3: ZeroBounce email validation for {email} @ {company}")
    passed, reason = await check_zerobounce_email(lead)
    if not passed:
        print(f"   ❌ Stage 3 failed: {reason}")
        return False, f"Stage 3 failed: {reason}"
    
    print(f"   ✅ Stage 3 passed")
    
    # ========================================================================
    # Stage 4: Google LLM Checks (MIXED)
    # - Contact Fuzzy Match: Verify contact works at company (HARD) + role/location (SOFT)
    # - Company Fuzzy Match: Verify company details (HARD) + LinkedIn (SOFT)
    # - ICP Evidence: Search for ICP confirmation sources (SOFT)
    # ========================================================================
    print(f"🔍 Stage 4: Google LLM checks for {email} @ {company}")
    
    # HARD checks - must pass
    checks_stage4_hard = [
        check_llm_contact_match,   # Contact works at company (HARD), role/location (SOFT)
        check_llm_company_match,   # Company name/industry/location match (HARD), LinkedIn (SOFT)
    ]
    
    for check_func in checks_stage4_hard:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 4 failed: {reason}")
            return False, f"Stage 4 failed: {reason}"
    
    # SOFT check - always passes, just appends data
    await check_icp_evidence(lead)  # ICP Evidence confirmation (SOFT - always passes)
    
    print(f"   ✅ Stage 4 passed")
    print(f"🎉 All stages passed for {email} @ {company}")
    
    return True, "All checks passed"

# Existing functions - DO NOT TOUCH (maintained for backward compatibility)

async def load_email_cache():
    if os.path.exists(EMAIL_CACHE_FILE):
        try:
            with open(EMAIL_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

async def save_email_cache(cache):
    try:
        with open(EMAIL_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

EMAIL_CACHE = asyncio.run(load_email_cache())

async def is_disposable_email(email: str) -> Tuple[bool, str]:
    domain = email.split("@")[1].lower() if "@" in email else ""
    # Return True if email IS disposable, False if NOT disposable
    is_disposable = domain in DISPOSABLE_DOMAINS
    return is_disposable, "Disposable domain" if is_disposable else "Not disposable"

async def check_domain_existence(domain: str) -> Tuple[bool, str]:
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(domain, "MX"))
        return True, "Domain has MX records"
    except Exception as e:
        return False, f"Domain check failed: {str(e)}"

async def check_hunter_email(email: str) -> Tuple[bool, str]:
    """Check email validity using Hunter API"""
    if "YOUR_HUNTER_API_KEY" in HUNTER_API_KEY:
        return True, "Mock pass"
    
    url = f"https://api.hunter.io/v2/email-verifier?email={email}&api_key={HUNTER_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                data = await response.json()
                status = data.get("data", {}).get("status", "unknown")
                is_valid = status in ["valid", "accept_all"]
                reason = data.get("data", {}).get("result", "unknown")
                return is_valid, reason
    except Exception as e:
        return False, str(e)

async def verify_company(company_domain: str) -> Tuple[bool, str]:
    if not company_domain:
        return False, "No domain provided"
    if not company_domain.startswith(("http://", "https://")):
        company_domain = f"https://{company_domain}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(company_domain, timeout=10, allow_redirects=True) as response:
                if response.status == 200:
                    return True, "Website accessible"
                elif response.status in [301, 302, 307, 308]:
                    return True, f"Website accessible (redirect: {response.status})"
                elif response.status in [401, 403]:
                    return True, f"Website accessible (requires auth: {response.status})"
                elif response.status in [405, 503]:
                    return True, f"Website accessible (service available: {response.status})"
                else:
                    return False, f"Website not accessible (status: {response.status})"
    except Exception as e:
        return False, f"Website inaccessible: {str(e)}"

async def check_duplicates(leads: list) -> Tuple[bool, dict]:
    """Check for duplicate emails and return which leads are duplicates (not first occurrence)"""
    email_first_occurrence = {}  # Track first occurrence of each email
    duplicate_leads = {}  # Track which lead indices are duplicates
    
    for i, lead in enumerate(leads):
        email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
        
        if email in email_first_occurrence:
            # This is a duplicate - mark this lead index as duplicate
            duplicate_leads[i] = email
        else:
            # First occurrence - record the lead index
            email_first_occurrence[email] = i
    
    return len(duplicate_leads) > 0, duplicate_leads

async def validate_lead_list(leads: list) -> list:
    """Main validation function - maintains backward compatibility"""
    
    # Mock mode fallback
    if "YOUR_HUNTER_API_KEY" in HUNTER_API_KEY:
        print("Mock mode: Assuming all leads pass automated checks")
        return [{
            "lead_index": i,
            "email": lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", ""))),
            "company_domain": urlparse(lead.get("Website", lead.get("website", ""))).netloc,
            "status": "Valid",
            "reason": "Mock pass"
        } for i, lead in enumerate(leads)]
    
    # Check for duplicates
    has_duplicates, duplicate_leads = await check_duplicates(leads)
    if has_duplicates:
        duplicate_emails = set(duplicate_leads.values())
        print(f"Duplicate emails detected: {duplicate_emails}")
        print(f"Duplicate lead indices: {list(duplicate_leads.keys())}")
        
        # Process all leads, but mark duplicates as invalid
        report = []
        for i, lead in enumerate(leads):
            email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
            domain = urlparse(lead.get("Website", lead.get("website", ""))).netloc if lead.get("Website") or lead.get("website") else ""
            
            if i in duplicate_leads:
                # Mark duplicate lead as invalid
                report.append({
                    "lead_index": i,
                    "email": email,
                    "company_domain": domain,
                    "status": "Invalid",
                    "reason": "Duplicate email"
                })
            else:
                # Process non-duplicate leads through automated checks
                passed, reason = await run_automated_checks(lead)
                status = "Valid" if passed else "Invalid"
                report.append({
                    "lead_index": i,
                    "email": email,
                    "company_domain": domain,
                    "status": status,
                    "reason": reason
                })
        
        return report
    
    # Process each lead through the new validation pipeline
    report = []
    for i, lead in enumerate(leads):
        email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
        domain = urlparse(lead.get("Website", lead.get("website", ""))).netloc if lead.get("Website") or lead.get("website") else ""
        
        # Run new automated checks
        passed, reason = await run_automated_checks(lead)
        
        status = "Valid" if passed else "Invalid"
        report.append({
            "lead_index": i,
            "email": email,
            "company_domain": domain,
            "status": status,
            "reason": reason
        })
    
    return report

async def collusion_check(validators: list, responses: list) -> dict:
    """Simulate PyGOD/DBScan collusion detection."""
    validator_scores = []
    for v in validators:
        for r in responses:
            validation = await v.validate_leads(r.leads)
            validator_scores.append({"hotkey": v.wallet.hotkey.ss58_address, "O_v": validation["O_v"]})
    
    # Mock PyGOD analysis
    data = np.array([[s["O_v"]] for s in validator_scores])
    detector = DOMINANT()
    detector.fit(data)
    V_c = detector.decision_score_.max()
    
    collusion_flags = {}
    for v in validators:
        collusion_flags[v.wallet.hotkey.ss58_address] = 0 if V_c > 0.7 else 1
    return collusion_flags
