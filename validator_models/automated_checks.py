import aiohttp
import asyncio
import dns.resolver
import pickle
import os
import re
import requests
import uuid
import whois
import json
import numpy as np
import unicodedata
# from pygod.detector import DOMINANT  # DEPRECATED: Only used in unused collusion_check function
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from disposable_email_domains import blocklist as DISPOSABLE_DOMAINS
from Leadpoet.utils.utils_lead_extraction import (
    get_email,
    get_website,
    get_company,
    get_first_name,
    get_last_name,
    get_location,
    get_industry,
    get_role,
    get_linkedin,
    get_field
)

MAX_REP_SCORE = 48  # Wayback (6) + SEC (12) + WHOIS/DNSBL (10) + GDELT (10) + Companies House (10) = 48

def normalize_accents(text: str) -> str:
    """
    Remove accents/diacritics from text for name matching.
    e.g., "José" -> "Jose", "François" -> "Francois"
    """
    # Normalize to NFD form (decomposes accented chars into base + combining mark)
    # Then remove combining marks (category 'Mn')
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

# Custom exception for API infrastructure failures (should skip lead, not submit)
class EmailVerificationUnavailableError(Exception):
    """Raised when email verification API is unavailable (no credits, bad key, network error, etc.)"""
    pass

load_dotenv()
MYEMAILVERIFIER_API_KEY = os.getenv("MYEMAILVERIFIER_API_KEY", "")
TRUELIST_API_KEY = os.getenv("TRUELIST_API_KEY", "")

# NEW: Stage 4 API keys (Google Search Engine + OpenRouter LLM)
GSE_CX = os.getenv("GSE_CX", "")
GSE_API_KEY = os.getenv("GSE_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")

# DuckDuckGo Search (FREE, no API key needed)
# Uses the 'ddgs' Python library which handles DDG's API properly
# Set to "true" to enable DuckDuckGo, otherwise uses GSE
USE_DDG_SEARCH = os.getenv("USE_DDG_SEARCH", "true").lower() == "true"

# Try to import ddgs library
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ ddgs library not installed. Run: pip install ddgs")

# NEW: Rep Score API keys (Companies House)
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
    import hashlib
    
    # Canonicalize evidence (sort keys for determinism)
    evidence_json = json.dumps(evidence, sort_keys=True)
    
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
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"⚠️ Failed to log validation metrics: {e}")

async def api_call_with_retry(session, url, params=None, max_retries=3, base_delay=1):
    """Make API call with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=10) as response:
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

async def check_required_fields(lead: dict) -> Tuple[bool, dict]:
    """Check that all required fields are present and non-empty"""
    required_fields = {
        "industry": ["industry", "Industry"],
        "sub_industry": ["sub_industry", "sub-industry", "Sub-industry", "Sub_industry"],
        "role": ["role", "Role"],
        "region": ["region", "Region", "location", "Location"],
    }
    
    missing_fields = []
    
    # Check for name (either full_name OR both first + last)
    full_name = lead.get("full_name") or lead.get("Full_name") or lead.get("Full Name")
    first_name = lead.get("first") or lead.get("First") or lead.get("first_name")
    last_name = lead.get("last") or lead.get("Last") or lead.get("last_name")
    
    has_name = bool(full_name) or (bool(first_name) and bool(last_name))
    if not has_name:
        missing_fields.append("contact_name")
    
    # Check other required fields
    for field_name, possible_keys in required_fields.items():
        found = False
        for key in possible_keys:
            value = lead.get(key)
            if value and str(value).strip():  # Check for non-empty string
                found = True
                break
        
        if not found:
            missing_fields.append(field_name)
    
    # Return structured rejection if any fields are missing
    if missing_fields:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_required_fields",
            "message": f"Missing required fields: {', '.join(missing_fields)}",
            "failed_fields": missing_fields
        }
    
    return True, {}

async def check_email_regex(lead: dict) -> Tuple[bool, dict]:
    """Check email format using RFC-5322 simplified regex with Unicode support (RFC 6531)"""
    try:
        email = get_email(lead)
        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            # Cache result
            cache_key = f"email_regex:no_email"
            validation_cache[cache_key] = (False, rejection_reason)
            await log_validation_metrics(lead, {"passed": False, "reason": rejection_reason["message"]}, "email_regex")
            return False, rejection_reason

        # RFC-5322 simplified regex (original ASCII validation)
        pattern_ascii = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        is_valid_ascii = bool(re.match(pattern_ascii, email))
        
        # RFC-6531 - Internationalized Email (Unicode support for international characters)
        # Allows emails like: anna.kosińska@cdprojekt.com, müller@siemens.de
        pattern_unicode = r"^[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}$"
        is_valid_unicode = bool(re.match(pattern_unicode, email, re.UNICODE))
        
        # Accept if EITHER pattern matches (ASCII OR Unicode)
        is_valid = is_valid_ascii or is_valid_unicode

        if not is_valid:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": f"Invalid email format: {email}",
                "failed_fields": ["email"]
            }
            # Cache result
            cache_key = f"email_regex:{email}"
            validation_cache[cache_key] = (False, rejection_reason)
            await log_validation_metrics(lead, {"passed": False, "reason": rejection_reason["message"]}, "email_regex")
            return False, rejection_reason

        # Valid email - cache success result
        cache_key = f"email_regex:{email}"
        validation_cache[cache_key] = (True, {})
        await log_validation_metrics(lead, {"passed": True, "reason": "Valid email format"}, "email_regex")

        return True, {}
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_email_regex",
            "message": f"Email regex check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        await log_validation_metrics(lead, {"passed": False, "reason": str(e)}, "email_regex")
        return False, rejection_reason

async def check_name_email_match(lead: dict) -> Tuple[bool, dict]:
    """
    Check if first name or last name appears in the email address.
    This is a HARD check that prevents costly API calls for leads that will fail anyway.
    
    Returns:
        (True, {}): If first OR last name found in email
        (False, rejection_reason): If NO name found in email
    """
    try:
        email = get_email(lead)
        first_name = get_first_name(lead)
        last_name = get_last_name(lead)
        
        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason
        
        if not first_name or not last_name:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": "Missing first name or last name",
                "failed_fields": ["first_name", "last_name"]
            }
            return False, rejection_reason
        
        # Extract local part of email (before @)
        local_part = email.split("@")[0].lower() if "@" in email else email.lower()
        
        # Normalize names for comparison (lowercase, remove special chars)
        first_normalized = re.sub(r'[^a-z0-9]', '', first_name.lower())
        last_normalized = re.sub(r'[^a-z0-9]', '', last_name.lower())
        local_normalized = re.sub(r'[^a-z0-9]', '', local_part)
        
        # Check if either first OR last name appears in email
        # Pattern matching: full name, first initial + last, last + first initial, etc.
        # Also handles shortened names by checking if email local part is a prefix of the name
        # Examples: "rich@" matches "Richard" (prefix check), "greg@" matches "Gregory" (prefix check)
        # Security: Requires minimum 3 characters and checks that local part matches BEGINNING of name (not substring)
        
        # Minimum match length to prevent false positives (e.g., "an" in "daniel")
        MIN_NAME_MATCH_LENGTH = 3
        
        name_match = False
        
        # Strategy 1: Check if normalized name patterns appear in local part
        # This handles: john@example.com, johndoe@example.com, jdoe@example.com
        patterns = []
        
        # Full normalized names
        if len(first_normalized) >= MIN_NAME_MATCH_LENGTH:
            patterns.append(first_normalized)  # john
        if len(last_normalized) >= MIN_NAME_MATCH_LENGTH:
            patterns.append(last_normalized)  # doe
        
        # Full name combinations
        patterns.append(f"{first_normalized}{last_normalized}")  # johndoe
        
        # Initial + last name combinations
        if len(first_normalized) > 0:
            patterns.append(f"{first_normalized[0]}{last_normalized}")  # jdoe
            patterns.append(f"{last_normalized}{first_normalized[0]}")  # doej
        
        # Check if any pattern appears in the normalized local part
        patterns = [p for p in patterns if p and len(p) >= MIN_NAME_MATCH_LENGTH]
        name_match = any(pattern in local_normalized for pattern in patterns)
        
        # Strategy 2: Check if local part matches shortened versions of the name
        # This handles: greg@example.com where first_name is "Gregory"
        # Check if local_part is a prefix of the normalized name (shortened form)
        if not name_match and len(local_normalized) >= MIN_NAME_MATCH_LENGTH:
            # Check if local_part matches beginning of first name (shortened)
            # e.g., "greg" matches "gregory" (local_part is prefix of name)
            if len(first_normalized) >= len(local_normalized):
                if first_normalized.startswith(local_normalized):
                    name_match = True
            
            # Check if local_part matches beginning of last name (shortened)
            if not name_match and len(last_normalized) >= len(local_normalized):
                if last_normalized.startswith(local_normalized):
                    name_match = True
            
            # Check if name prefixes appear in local part (reverse direction)
            # e.g., "gregory" prefix "greg" in local_part "greg"
            if not name_match:
                # Check first name prefixes (3-6 characters)
                for length in range(MIN_NAME_MATCH_LENGTH, min(len(first_normalized) + 1, 7)):
                    name_prefix = first_normalized[:length]
                    if name_prefix == local_normalized or name_prefix in local_normalized:
                        name_match = True
                        break
                
                # Check last name prefixes if still no match
                if not name_match:
                    for length in range(MIN_NAME_MATCH_LENGTH, min(len(last_normalized) + 1, 7)):
                        name_prefix = last_normalized[:length]
                        if name_prefix == local_normalized or name_prefix in local_normalized:
                            name_match = True
                            break
        
        if not name_match:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_name_email_match",
                "message": f"Name '{first_name} {last_name}' does not match email pattern '{email}'",
                "failed_fields": ["email", "first_name", "last_name"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - Name not found in email")
            return False, rejection_reason
        
        print(f"   ✅ Stage 0: {email} @ {get_company(lead)} - Name found in email")
        return True, {}
        
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_name_email_match",
            "message": f"Name-email match check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_general_purpose_email(lead: dict) -> Tuple[bool, dict]:
    """
    Check if email is a general-purpose email address (instant fail).
    
    General-purpose emails are not personal contacts and should be rejected immediately
    to save API costs and maintain lead quality.
    
    Returns:
        (True, {}): If email is NOT general purpose (personal contact)
        (False, rejection_reason): If email IS general purpose (instant fail)
    """
    try:
        email = get_email(lead)
        
        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_general_purpose_email",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason
        
        # Define general-purpose email prefixes (must match calculate-rep-score exactly)
        general_purpose_prefixes = [
            'info@', 'hello@', 'owner@', 'ceo@', 'founder@', 'contact@', 'support@',
            'team@', 'admin@', 'office@', 'mail@', 'connect@', 'help@', 'hi@',
            'welcome@', 'inquiries@', 'general@', 'feedback@', 'ask@', 'outreach@',
            'communications@', 'crew@', 'staff@', 'community@', 'reachus@', 'talk@',
            'service@'
        ]
        
        email_lower = email.lower()
        
        # Check if email starts with any general-purpose prefix
        matched_prefix = next((prefix for prefix in general_purpose_prefixes if email_lower.startswith(prefix)), None)
        
        if matched_prefix:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_general_purpose_email",
                "message": f"Email '{email}' is a general purpose email (starts with {matched_prefix}) - not a personal contact",
                "failed_fields": ["email"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - General purpose email detected: {matched_prefix}")
            return False, rejection_reason
        
        # Not a general-purpose email - proceed
        print(f"   ✅ Stage 0: {email} @ {get_company(lead)} - Personal email (not general purpose)")
        return True, {}
        
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_general_purpose_email",
            "message": f"General purpose email check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_free_email_domain(lead: dict) -> Tuple[bool, dict]:
    """
    Check if email uses a free/personal email domain (instant fail).
    
    B2B leads should use corporate email domains, not free consumer services.
    This prevents low-quality leads from free email providers.
    
    Returns:
        (True, {}): If email is corporate domain
        (False, rejection_reason): If email is free domain (gmail, yahoo, etc.)
    """
    try:
        email = get_email(lead)
        
        if not email:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_free_email_domain",
                "message": "No email provided",
                "failed_fields": ["email"]
            }
            return False, rejection_reason
        
        # Extract domain from email
        try:
            domain = email.split("@")[1].lower() if "@" in email else ""
        except IndexError:
            return True, {}  # Invalid format handled by other checks
        
        # Common free email domains (comprehensive list)
        free_domains = {
            'gmail.com', 'googlemail.com', 'yahoo.com', 'yahoo.co.uk', 'yahoo.fr',
            'outlook.com', 'hotmail.com', 'live.com', 'msn.com',
            'aol.com', 'mail.com', 'protonmail.com', 'proton.me',
            'icloud.com', 'me.com', 'mac.com',
            'zoho.com', 'yandex.com', 'gmx.com', 'mail.ru'
        }
        
        if domain in free_domains:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_free_email_domain",
                "message": f"Email uses free consumer domain '{domain}' - B2B leads require corporate email",
                "failed_fields": ["email"]
            }
            print(f"   ❌ Stage 0: {email} @ {get_company(lead)} - Free email domain rejected: {domain}")
            return False, rejection_reason
        
        # Corporate domain - proceed
        return True, {}
        
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_free_email_domain",
            "message": f"Free email domain check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

async def check_domain_age(lead: dict) -> Tuple[bool, dict]:
    """
    Check domain age using WHOIS lookup.
    Appends WHOIS data to lead object for reputation scoring.
    """
    website = get_website(lead)
    if not website:
        # Append default WHOIS data
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"domain_age:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois"]):
        cached_result = validation_cache[cache_key]
        # Restore cached WHOIS data to lead
        cached_data = validation_cache.get(f"{cache_key}_data")
        if cached_data:
            lead["whois_checked"] = cached_data.get("checked", True)
            lead["domain_age_days"] = cached_data.get("age_days")
            lead["domain_creation_date"] = cached_data.get("creation_date")
            lead["domain_registrar"] = cached_data.get("registrar")
            lead["domain_nameservers"] = cached_data.get("nameservers")
            lead["whois_updated_date"] = cached_data.get("updated_date")
            lead["whois_updated_days_ago"] = cached_data.get("whois_updated_days_ago")
        return cached_result

    try:
        # Implement actual WHOIS lookup
        def get_domain_age_sync(domain_name):
            try:
                w = whois.whois(domain_name)
                
                # Extract registrar, nameservers, and updated_date for reputation scoring
                registrar = getattr(w, 'registrar', None)
                nameservers = getattr(w, 'name_servers', None)
                if isinstance(nameservers, list):
                    nameservers = nameservers[:3]  # Limit to first 3 nameservers
                
                # Extract updated_date for WHOIS stability check
                updated_date = getattr(w, 'updated_date', None)
                if updated_date:
                    if isinstance(updated_date, list):
                        updated_date = updated_date[0]
                    # Make timezone-naive if timezone-aware
                    if hasattr(updated_date, 'tzinfo') and updated_date.tzinfo is not None:
                        updated_date = updated_date.replace(tzinfo=None)
                
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        creation_date = w.creation_date[0]
                    else:
                        creation_date = w.creation_date
                    
                    # Make creation_date timezone-naive if it's timezone-aware
                    if creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(tzinfo=None)

                    age_days = (datetime.now() - creation_date).days
                    min_age_days = 7  # 7 days minimum

                    # Calculate whois_updated_days_ago
                    whois_updated_days_ago = None
                    if updated_date:
                        whois_updated_days_ago = (datetime.now() - updated_date).days

                    # Return WHOIS data along with result
                    whois_data = {
                        "age_days": age_days,
                        "creation_date": creation_date.isoformat(),
                        "registrar": registrar,
                        "nameservers": nameservers,
                        "updated_date": updated_date.isoformat() if updated_date else None,
                        "whois_updated_days_ago": whois_updated_days_ago,
                        "checked": True
                    }

                    if age_days >= min_age_days:
                        return (True, {}, whois_data)
                    else:
                        return (False, {
                            "stage": "Stage 1: DNS Layer",
                            "check_name": "check_domain_age",
                            "message": f"Domain too new: {age_days} days (minimum: {min_age_days})",
                            "failed_fields": ["website"]
                        }, whois_data)
                else:
                    # Calculate whois_updated_days_ago even if creation_date is missing
                    whois_updated_days_ago = None
                    if updated_date:
                        whois_updated_days_ago = (datetime.now() - updated_date).days
                    
                    whois_data = {
                        "age_days": None,
                        "creation_date": None,
                        "registrar": registrar,
                        "nameservers": nameservers,
                        "updated_date": updated_date.isoformat() if updated_date else None,
                        "whois_updated_days_ago": whois_updated_days_ago,
                        "checked": True
                    }
                    return False, {
                        "stage": "Stage 1: DNS Layer",
                        "check_name": "check_domain_age",
                        "message": "Could not determine domain creation date",
                        "failed_fields": ["website"]
                    }, whois_data
            except Exception as e:
                whois_data = {
                    "age_days": None,
                    "creation_date": None,
                    "registrar": None,
                    "nameservers": None,
                    "updated_date": None,
                    "whois_updated_days_ago": None,
                    "checked": False,
                    "error": str(e)
                }
                return False, {
                    "stage": "Stage 1: DNS Layer",
                    "check_name": "check_domain_age",
                    "message": f"WHOIS lookup failed: {str(e)}",
                    "failed_fields": ["website"]
                }, whois_data

        # Run WHOIS lookup in executor to avoid blocking
        loop = asyncio.get_event_loop()
        passed, rejection_reason, whois_data = await loop.run_in_executor(None, get_domain_age_sync, domain)
        
        # Append WHOIS data to lead
        lead["whois_checked"] = whois_data.get("checked", True)
        lead["domain_age_days"] = whois_data.get("age_days")
        lead["domain_creation_date"] = whois_data.get("creation_date")
        lead["domain_registrar"] = whois_data.get("registrar")
        lead["domain_nameservers"] = whois_data.get("nameservers")
        lead["whois_updated_date"] = whois_data.get("updated_date")
        lead["whois_updated_days_ago"] = whois_data.get("whois_updated_days_ago")
        if "error" in whois_data:
            lead["whois_error"] = whois_data["error"]
        
        # Cache both result and data
        result = (passed, rejection_reason)
        validation_cache[cache_key] = result
        validation_cache[f"{cache_key}_data"] = whois_data
        
        return result

    except Exception as e:
        # Append error state
        lead["whois_checked"] = False
        lead["domain_age_days"] = None
        lead["domain_creation_date"] = None
        lead["whois_error"] = str(e)
        
        result = (False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_domain_age",
            "message": f"Domain age check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_mx_record(lead: dict) -> Tuple[bool, dict]:
    """Check if domain has MX records"""
    website = get_website(lead)
    if not website:
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        return False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"mx_record:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]

    try:
        passed, msg = await check_domain_existence(domain)
        if passed:
            result = (True, {})
        else:
            result = (False, {
                "stage": "Stage 1: DNS Layer",
                "check_name": "check_mx_record",
                "message": msg,
                "failed_fields": ["website"]
            })
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, {
            "stage": "Stage 1: DNS Layer",
            "check_name": "check_mx_record",
            "message": f"MX record check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_spf_dmarc(lead: dict) -> Tuple[bool, dict]:
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
        (True, dict): Always passes with empty dict (SOFT check)
    """
    def fail_lead(lead):
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        return lead
        
    email = get_email(lead)
    if not email:
        # No email to check - append default values
        lead = fail_lead(lead)
        return True, {}

    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead = fail_lead(lead)
            return True, {}
    except (IndexError, AttributeError):
        lead = fail_lead(lead)
        return True, {}

    cache_key = f"spf_dmarc:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        cached_data = validation_cache[cache_key]
        # Apply cached values to lead
        lead["has_spf"] = cached_data.get("has_spf", False)
        lead["has_dmarc"] = cached_data.get("has_dmarc", False)
        lead["dmarc_policy_strict"] = cached_data.get("dmarc_policy_strict", False)
        return True, {}

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
        return True, {}

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
        return True, {}

async def check_head_request(lead: dict) -> Tuple[bool, dict]:
    """Wrapper around existing verify_company function"""
    website = get_website(lead)
    if not website:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": "No website provided",
            "failed_fields": ["website"]
        }

    domain = extract_root_domain(website)
    if not domain:
        return False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": f"Invalid website format: {website}",
            "failed_fields": ["website"]
        }

    cache_key = f"head_request:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        return validation_cache[cache_key]

    try:
        passed, msg = await verify_company(domain)
        if passed:
            result = (True, {})
        else:
            result = (False, {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_head_request",
                "message": f"Website not accessible: {msg}",
                "failed_fields": ["website"]
            })
        validation_cache[cache_key] = result
        return result
    except Exception as e:
        result = (False, {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_head_request",
            "message": f"HEAD request check failed: {str(e)}",
            "failed_fields": ["website"]
        })
        validation_cache[cache_key] = result
        return result

async def check_disposable(lead: dict) -> Tuple[bool, dict]:
    """Check if email domain is disposable"""
    email = get_email(lead)
    if not email:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_disposable",
            "message": "No email provided",
            "failed_fields": ["email"]
        }
        return False, rejection_reason

    cache_key = f"disposable:{email}"
    if cache_key in validation_cache:
        return validation_cache[cache_key]

    try:
        is_disposable, reason = await is_disposable_email(email)
        # For validation pipeline: return True if check PASSES (email is NOT disposable)
        # return False if check FAILS (email IS disposable)
        if is_disposable:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_disposable",
                "message": f"Disposable email domain detected: {email}",
                "failed_fields": ["email"]
            }
            validation_cache[cache_key] = (False, rejection_reason)
            return False, rejection_reason
        else:
            validation_cache[cache_key] = (True, {})
            return True, {}
    except Exception as e:
        rejection_reason = {
            "stage": "Stage 0: Hardcoded Checks",
            "check_name": "check_disposable",
            "message": f"Disposable check failed: {str(e)}",
            "failed_fields": ["email"]
        }
        validation_cache[cache_key] = (False, rejection_reason)
        return False, rejection_reason

async def check_dnsbl(lead: dict) -> Tuple[bool, dict]:
    """
    Check if lead's email domain is listed in Spamhaus DBL.
    Appends DNSBL data to lead object for reputation scoring.

    Args:
        lead: Dict containing email field

    Returns:
        (bool, dict): (is_valid, rejection_reason_dict)
    """
    email = get_email(lead)
    if not email:
        # Append default DNSBL data
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return False, {
            "stage": "Stage 2: Domain Reputation",
            "check_name": "check_dnsbl",
            "message": "No email provided",
            "failed_fields": ["email"]
        }

    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead["dnsbl_checked"] = False
            lead["dnsbl_blacklisted"] = False
            lead["dnsbl_list"] = None
            return True, {}  # Invalid format handled by other checks
    except (IndexError, AttributeError):
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return True, {}  # Invalid format handled by other checks

    # Use root domain extraction helper
    root_domain = extract_root_domain(domain)
    if not root_domain:
        lead["dnsbl_checked"] = False
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = None
        return True, {}  # Could not extract - handled by other checks

    cache_key = f"dnsbl_{root_domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["dns_head"]):
        cached_result = validation_cache[cache_key]
        # Restore cached DNSBL data to lead
        cached_data = validation_cache.get(f"{cache_key}_data")
        if cached_data:
            lead["dnsbl_checked"] = cached_data.get("checked", True)
            lead["dnsbl_blacklisted"] = cached_data.get("blacklisted", False)
            lead["dnsbl_list"] = cached_data.get("list", "cloudflare_dbl")
            lead["dnsbl_domain"] = cached_data.get("domain", root_domain)
        return cached_result

    try:
        async with API_SEMAPHORE:
            # Perform Cloudflare DNSBL lookup (more reliable than Spamhaus for free tier)
            # Cloudflare has no rate limits and fewer false positives
            query = f"{root_domain}.dbl.cloudflare.com"

            # Run DNS lookup in executor to avoid blocking
            loop = asyncio.get_event_loop()
            def dns_lookup():
                try:
                    print(f"   🔍 DNSBL Query: {query}")
                    answers = dns.resolver.resolve(query, "A")
                    # If we get A records, domain IS blacklisted
                    a_records = [str(rdata) for rdata in answers]
                    
                    # Check for actual blacklist codes (127.0.0.x where x < 128)
                    for record in a_records:
                        if record.startswith("127.0.0."):
                            print(f"   ⚠️  DNSBL returned A records: {a_records} → BLACKLISTED")
                            return True
                    
                    # Any other response is not a confirmed blacklist
                    print(f"   ✅ DNSBL returned A records: {a_records} → CLEAN (not a blacklist code)")
                    return False
                    
                except dns.resolver.NXDOMAIN:
                    # NXDOMAIN = not in blacklist (expected for clean domains)
                    print(f"   ✅ DNSBL returned NXDOMAIN → CLEAN")
                    return False  # No record = domain is clean
                except dns.resolver.NoAnswer:
                    # No answer = not in blacklist
                    print(f"   ✅ DNSBL returned NoAnswer → CLEAN")
                    return False
                except dns.resolver.Timeout:
                    # Timeout = treat as clean (don't block on infrastructure issues)
                    print(f"   ⚠️  DNSBL query timeout for {query} → treating as CLEAN")
                    return False
                except Exception as e:
                    # On any DNS error, default to valid (don't block on infrastructure issues)
                    print(f"   ⚠️  DNS lookup error for {query}: {type(e).__name__}: {e} → treating as CLEAN")
                    return False

            is_blacklisted = await loop.run_in_executor(None, dns_lookup)

            # Append DNSBL data to lead
            lead["dnsbl_checked"] = True
            lead["dnsbl_blacklisted"] = is_blacklisted
            lead["dnsbl_list"] = "cloudflare_dbl"
            lead["dnsbl_domain"] = root_domain

            # Cache the data separately for restoration
            dnsbl_data = {
                "checked": True,
                "blacklisted": is_blacklisted,
                "list": "cloudflare_dbl",
                "domain": root_domain
            }
            validation_cache[f"{cache_key}_data"] = dnsbl_data

            if is_blacklisted:
                result = (False, {
                    "stage": "Stage 2: Domain Reputation",
                    "check_name": "check_dnsbl",
                    "message": f"Domain {root_domain} blacklisted in Cloudflare DBL",
                    "failed_fields": ["email"]
                })
                print(f"❌ DNSBL: Domain {root_domain} found in Cloudflare blacklist")
            else:
                result = (True, {})
                print(f"✅ DNSBL: Domain {root_domain} clean")

            validation_cache[cache_key] = result
            return result

    except Exception as e:
        # On any unexpected error, append error state
        lead["dnsbl_checked"] = True
        lead["dnsbl_blacklisted"] = False
        lead["dnsbl_list"] = "spamhaus_dbl"
        lead["dnsbl_domain"] = root_domain
        lead["dnsbl_error"] = str(e)
        
        result = (True, {})  # Don't block on infrastructure issues
        validation_cache[cache_key] = result
        print(f"⚠️ DNSBL check error for {root_domain}: {e}")
        return result

# Stage 3: Email Verification (MyEmailVerifier or TrueList fallback)

async def check_truelist_email(lead: dict) -> Tuple[bool, dict]:
    """
    Check email validity using TrueList API (fallback when MEV not configured).
    
    TrueList API: https://apidocs.truelist.io/#tag/Single-email-validation
    Only accepts "email_ok" status (equivalent to MEV "Valid").
    
    Retry logic: Up to 3 attempts with 10s wait between retries.
    """
    email = get_email(lead)
    if not email:
        return False, {
            "stage": "Stage 3: TrueList",
            "check_name": "check_truelist_email",
            "message": "No email provided",
            "failed_fields": ["email"]
        }

    cache_key = f"truelist:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["myemailverifier"]):
        print(f"   💾 Using cached TrueList result for: {email}")
        return validation_cache[cache_key]

    max_retries = 3
    retry_delay = 10

    for attempt in range(1, max_retries + 1):
        try:
            async with API_SEMAPHORE:
                async with aiohttp.ClientSession() as session:
                    # TrueList single email validation endpoint
                    # API docs: https://apidocs.truelist.io/#tag/Single-email-validation
                    url = "https://api.truelist.io/api/v1/verify_inline"
                    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
                    payload = {"email": email}
                    
                    if attempt == 1:
                        print(f"   📞 Calling TrueList API for: {email}")
                    else:
                        print(f"   🔄 Retry {attempt}/{max_retries} for: {email}")
                    
                    async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                        if response.status in [401, 402, 403, 429, 500, 502, 503, 504]:
                            print(f"   🚨 TrueList API error (HTTP {response.status})")
                            raise EmailVerificationUnavailableError(f"TrueList API unavailable (HTTP {response.status})")
                        
                        data = await response.json()
                        print(f"   📥 TrueList Response: {data}")
                        
                        # TrueList returns: {"emails": [{"email_sub_state": "email_ok", ...}]}
                        emails = data.get("emails", [])
                        if not emails:
                            raise Exception("No email results in TrueList response")
                        
                        email_data = emails[0]
                        status = email_data.get("email_sub_state", "unknown")
                        email_state = email_data.get("email_state", "unknown")
                        
                        # Store metadata in lead
                        lead["email_verifier_status"] = status
                        lead["email_verifier_disposable"] = status == "disposable"
                        lead["email_verifier_catch_all"] = status == "accept_all"
                        lead["email_verifier_provider"] = "truelist"
                        
                        # Only accept "email_ok" (equivalent to MEV "Valid")
                        if status == "email_ok":
                            result = (True, {})
                        elif status == "accept_all":
                            result = (False, {
                                "stage": "Stage 3: TrueList",
                                "check_name": "check_truelist_email",
                                "message": "Email is catch-all/accept-all (instant rejection)",
                                "failed_fields": ["email"]
                            })
                        elif status == "disposable":
                            result = (False, {
                                "stage": "Stage 3: TrueList",
                                "check_name": "check_truelist_email",
                                "message": "Email is from a disposable provider",
                                "failed_fields": ["email"]
                            })
                        else:
                            # Reject all other statuses (unknown, invalid, failed_*, etc.)
                            result = (False, {
                                "stage": "Stage 3: TrueList",
                                "check_name": "check_truelist_email",
                                "message": f"Email status '{status}' (only 'email_ok' accepted)",
                                "failed_fields": ["email"]
                            })
                        
                        validation_cache[cache_key] = result
                        return result
        
        except EmailVerificationUnavailableError:
            raise
        except asyncio.TimeoutError:
            if attempt < max_retries:
                print(f"   ⏳ TrueList timed out. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                raise EmailVerificationUnavailableError("TrueList API timeout (all retries exhausted)")
        except aiohttp.ClientError as e:
            if attempt < max_retries:
                print(f"   ⏳ Network error. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                raise EmailVerificationUnavailableError(f"TrueList API network error: {str(e)}")
        except Exception as e:
            if attempt < max_retries:
                print(f"   ⏳ Unexpected error. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                raise EmailVerificationUnavailableError(f"TrueList API error: {str(e)}")


async def check_myemailverifier_email(lead: dict) -> Tuple[bool, dict]:
    """
    Check email validity using MyEmailVerifier API (with retry logic)
    
    FALLBACK: If MYEMAILVERIFIER_API_KEY is not set, uses TrueList API instead.
    
    MyEmailVerifier provides real-time email verification with:
    - Syntax validation
    - Domain/MX record checks
    - Catch-all detection
    - Disposable email detection
    - Spam trap detection
    - Role account detection
    
    Retry logic: Up to 3 attempts with 10s wait between retries.
    If all retries fail, lead is SKIPPED (not approved/denied).
    
    API Documentation: https://myemailverifier.com/real-time-email-verification
    """
    # FALLBACK: Use TrueList if MEV API key not configured
    if not MYEMAILVERIFIER_API_KEY:
        if TRUELIST_API_KEY:
            print(f"   ℹ️  MEV API key not set, using TrueList fallback")
            return await check_truelist_email(lead)
        else:
            raise EmailVerificationUnavailableError("No email verification API key configured (MEV or TrueList)")
    
    email = get_email(lead)
    if not email:
        return False, {
            "stage": "Stage 3: MyEmailVerifier",
            "check_name": "check_myemailverifier_email",
            "message": "No email provided",
            "failed_fields": ["email"]
        }

    cache_key = f"myemailverifier:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["myemailverifier"]):
        print(f"   💾 Using cached MyEmailVerifier result for: {email}")
        return validation_cache[cache_key]

    # Retry logic: Up to 3 attempts
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            async with API_SEMAPHORE:
                async with aiohttp.ClientSession() as session:
                    # MyEmailVerifier API endpoint
                    # Correct endpoint format: https://client.myemailverifier.com/verifier/validate_single/{email}/{API_KEY}
                    url = f"https://client.myemailverifier.com/verifier/validate_single/{email}/{MYEMAILVERIFIER_API_KEY}"
                    
                    if attempt == 1:
                        print(f"   📞 Calling MyEmailVerifier API for: {email}")
                    else:
                        print(f"   🔄 Retry {attempt}/{max_retries} for: {email}")
                    
                    async with session.get(url, timeout=30) as response:
                        # Check for API error responses (no credits, invalid key, etc.)
                        # These should SKIP the lead entirely (not submit to Supabase)
                        if response.status in [401, 402, 403, 429, 500, 502, 503, 504]:
                            if response.status == 402:
                                print(f"   🚨 MyEmailVerifier API: No credits remaining")
                            elif response.status == 401:
                                print(f"   🚨 MyEmailVerifier API: Invalid API key")
                            elif response.status == 429:
                                print(f"   🚨 MyEmailVerifier API: Rate limit exceeded")
                            else:
                                print(f"   🚨 MyEmailVerifier API: Server error (HTTP {response.status})")
                            # Raise exception to tell validator to SKIP this lead
                            raise EmailVerificationUnavailableError(f"MyEmailVerifier API unavailable (HTTP {response.status})")
                        
                        data = await response.json()
                        
                        # Log the actual API response to prove it's not mock mode
                        print(f"   📥 MyEmailVerifier Response: {data}")
                        
                        # Parse MyEmailVerifier response
                        # API response fields (as per official docs):
                        # - Address: the email address
                        # - Status: "Valid", "Invalid", "Unknown", "Catch All", "Grey-listed"
                        # - Disposable_Domain: "true" or "false" (string)
                        # - catch_all: "true" or "false" (string)
                        # - Role_Based: "true" or "false" (string)
                        # - Free_Domain: "true" or "false" (string)
                        # - Greylisted: "true" or "false" (string)
                        # - Diagnosis: description (e.g., "Mailbox Exists and Active")
                        
                        status = data.get("Status", "Unknown")
                        
                        # Store email verification metadata in lead (for tasks2.md Phase 1)
                        lead["email_verifier_status"] = status
                        lead["email_verifier_disposable"] = data.get("Disposable_Domain", "false") == "true"
                        lead["email_verifier_catch_all"] = data.get("catch_all", "false") == "true"
                        lead["email_verifier_role_based"] = data.get("Role_Based", "false") == "true"
                        lead["email_verifier_free"] = data.get("Free_Domain", "false") == "true"
                        lead["email_verifier_diagnosis"] = data.get("Diagnosis", "")
                        
                        # Check for disposable domains
                        is_disposable = lead["email_verifier_disposable"]
                        if is_disposable:
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": "Email is from a disposable/temporary email provider",
                                "failed_fields": ["email"]
                            })
                            validation_cache[cache_key] = result
                            return result

                        # Handle validation results based on Status field
                        # API Documentation: https://myemailverifier.com/real-time-email-verification
                        # Valid statuses: "Valid", "Invalid", "Unknown", "Catch All", "Grey-listed"
                        if status == "Valid":
                            result = (True, {})
                        elif status in ["Catch All", "Catch-All", "Catch-all"]:
                            # BRD: Reject ALL catch-all emails (no exceptions)
                            # Check all variants: "Catch All", "Catch-All", "Catch-all" (API returns different formats)
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": "Email is catch-all (instant rejection)",
                                "failed_fields": ["email"]
                            })
                        elif status == "Invalid":
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": "Email marked invalid",
                                "failed_fields": ["email"]
                            })
                        elif status == "Grey-listed":
                            # IMPORTANT: Treat grey-listed as invalid (as per tasks2.md Phase 1 requirement)
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": "Email is grey-listed (treated as invalid)",
                                "failed_fields": ["email"]
                            })
                        elif status == "Unknown":
                            # IMPORTANT: Treat unknown as invalid (as per tasks2.md Phase 1 requirement)
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": "Email status unknown (treated as invalid)",
                                "failed_fields": ["email"]
                            })
                        else:
                            # SECURITY: Reject any unrecognized status (prevents API changes from bypassing validation)
                            print(f"   ⚠️  UNKNOWN MyEmailVerifier status: {status}")
                            result = (False, {
                                "stage": "Stage 3: MyEmailVerifier",
                                "check_name": "check_myemailverifier_email",
                                "message": f"Unrecognized email status '{status}' (rejected for safety)",
                                "failed_fields": ["email"]
                            })

                        validation_cache[cache_key] = result
                        return result
        
        except EmailVerificationUnavailableError:
            # API infrastructure error (402, 429, etc.) - re-raise immediately, no retry
            raise
        
        except asyncio.TimeoutError:
            # Timeout - retry if attempts remaining
            if attempt < max_retries:
                print(f"   ⏳ API timed out. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
                continue  # Retry
            else:
                # All retries exhausted - SKIP the lead
                print(f"   ❌ API timed out after {max_retries} attempts. Lead will be SKIPPED.")
                raise EmailVerificationUnavailableError("MyEmailVerifier API timeout (all retries exhausted)")
        
        except aiohttp.ClientError as e:
            # Network error - retry if attempts remaining
            if attempt < max_retries:
                print(f"   ⏳ Network error. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
                continue  # Retry
            else:
                # All retries exhausted - SKIP the lead
                print(f"   ❌ Network error after {max_retries} attempts. Lead will be SKIPPED.")
                raise EmailVerificationUnavailableError(f"MyEmailVerifier API network error: {str(e)}")
        
        except Exception as e:
            # Unexpected error - retry if attempts remaining
            if attempt < max_retries:
                print(f"   ⏳ Unexpected error. Retrying in {retry_delay}s... ({attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
                continue  # Retry
            else:
                # All retries exhausted - SKIP the lead
                print(f"   ❌ Unexpected error after {max_retries} attempts. Lead will be SKIPPED.")
                raise EmailVerificationUnavailableError(f"MyEmailVerifier API error: {str(e)}")

# Stage 4: LinkedIn/GSE Validation

async def _ddg_search(query: str, max_results: int = 10, max_retries: int = 3) -> List[Dict[str, str]]:
    """
    Perform DuckDuckGo search using the ddgs library.
    Returns GSE-compatible results: [{title, link, snippet}]
    
    Works on headless servers (VPS) - no GUI required.
    The ddgs library handles DDG's API complexity internally.
    
    Retries up to 3 times on API/request failure before raising exception.
    Raises exception on persistent failure (triggers GSE fallback).
    
    Requires: pip install ddgs
    """
    if not DDGS_AVAILABLE:
        raise Exception("ddgs library not available - install with: pip install ddgs")
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # Run synchronous ddgs in thread to avoid blocking
            # Use Yahoo backend for Stage 4 (same as Stage 5 for consistency)
            def do_search():
                return DDGS().text(query, max_results=max_results, backend='yahoo')
            
            results_raw = await asyncio.to_thread(do_search)
            
            # Convert to GSE-compatible format
            results = []
            for r in results_raw:
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            
            return results
            
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"         ⚠️ DDG API error (attempt {attempt}/{max_retries}): {str(e)}, retrying...")
                await asyncio.sleep(2)  # Wait 2 seconds before retry
            else:
                print(f"         ❌ DDG API failed after {max_retries} attempts: {str(e)}")
                raise Exception(f"DDG API failed after {max_retries} retries: {str(last_error)}")

async def search_linkedin_ddg(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> Tuple[List[dict], bool]:
    """
    Search LinkedIn using DuckDuckGo (FREE, no API key needed).
    
    Directly scrapes DuckDuckGo - no server required.
    Returns results in same format as GSE: [{title, link, snippet}]
    
    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: LinkedIn URL provided by miner (required)
        max_results: Max search results to return
    
    Returns:
        Tuple of (List of search results with title, link, snippet, url_match_exact: bool)
    """
    if not linkedin_url:
        print(f"   ⚠️ No LinkedIn URL provided")
        return [], False
    
    # Extract profile slug from LinkedIn URL
    profile_slug = linkedin_url.split("/in/")[-1].strip("/") if "/in/" in linkedin_url else None
    
    # Track if URL matched exactly (strong identity proof)
    url_match_exact = False
    
    # 7 variations - try multiple approaches to find the exact LinkedIn profile
    query_variations = [
        f'"{full_name}" linkedin {company}',                  # 1. Quoted name + company (most specific)
        f"site:linkedin.com/in {full_name} {company}",        # 2. Site-restricted + company
        f"{full_name} linkedin {company}",                    # 3. Name + LinkedIn + company
        f"site:linkedin.com/in {full_name}",                  # 4. Site-restricted (broader)
        f"{full_name} linkedin",                              # 5. Name + LinkedIn (broader)
        f'"{full_name}" site:linkedin.com',                   # 6. Quoted name + site
        f"linkedin.com/in/{profile_slug}" if profile_slug else None,  # 7. Profile slug directly
    ]
    
    # Remove None values
    query_variations = [q for q in query_variations if q]
    
    # Track API errors to distinguish "no match" from "API failure"
    api_errors = []
    
    print(f"   🔍 Trying {len(query_variations)} search variations for LinkedIn profile (DuckDuckGo, then LLM verify)...")
    
    for variation_idx, query in enumerate(query_variations, 1):
        # 3 second delay between variations (avoid rate limiting)
        if variation_idx > 1:
            print(f"      ⏳ Waiting 3s before next variation...")
            await asyncio.sleep(3)
        
        print(f"      🔄 Variation {variation_idx}/{len(query_variations)}: {query[:80]}...")
        
        try:
            items = await _ddg_search(query)
            
            if items:
                print(f"         ✅ Found {len(items)} result(s) with variation {variation_idx}")
                
                # FILTER: Only keep LinkedIn results (profile URLs)
                linkedin_results = []
                found_profile_urls = []
                
                for item in items:
                    link = item.get("link", "")
                    if "linkedin.com/in/" in link:
                        result_slug = link.split("/in/")[-1].strip("/").split("?")[0]
                        found_profile_urls.append(result_slug)
                        linkedin_results.append(item)
                    elif "linkedin.com" in link:
                        # Include other LinkedIn pages (company, posts) for context
                        linkedin_results.append(item)
                
                if not linkedin_results:
                    print(f"         ⚠️ No LinkedIn URLs in results, trying next variation...")
                    continue
                
                print(f"         ✅ Found {len(linkedin_results)} LinkedIn result(s)")
                
                # URL matching logic - all variations are name-based, verify profile matches
                if profile_slug:
                    if found_profile_urls:
                        # Check exact match first
                        exact_match = any(
                            profile_slug.lower() == result_slug.lower()
                            for result_slug in found_profile_urls
                        )
                        # Also check partial match (profile slug contained in result)
                        partial_match = any(
                            profile_slug.lower() in result_slug.lower() or result_slug.lower() in profile_slug.lower()
                            for result_slug in found_profile_urls
                        )
                        
                        if exact_match:
                            print(f"         ✅ URL MATCH: Profile '{profile_slug}' confirmed (exact)")
                            url_match_exact = True  # Strong identity proof!
                        elif partial_match:
                            print(f"         ✅ URL MATCH: Profile '{profile_slug}' confirmed (partial)")
                        else:
                            print(f"         ⚠️  URL MISMATCH: Expected '{profile_slug}' but found: {found_profile_urls[:3]}")
                            continue
                
                # FILTER 1: Clean up concatenated titles and separate profile headlines from posts
                # DDG often concatenates multiple result titles together
                profile_headlines = []
                posts = []
                
                for item in linkedin_results:
                    title = item.get("title", "")
                    
                    # DDG concatenates titles - extract only the FIRST profile
                    # Pattern: "Name - Title | LinkedIn Name2 - Title2"
                    if " | LinkedIn " in title:
                        # Take only the first profile (before the concatenation)
                        title = title.split(" | LinkedIn ")[0] + " | LinkedIn"
                        item = dict(item)  # Copy to avoid modifying original
                        item["title"] = title
                    
                    # Skip non-profile results (posts, intro requests, etc.)
                    if " on LinkedIn:" in title or " on LinkedIn :" in title:
                        posts.append(item)
                        continue
                    if title.lower().startswith("seeking intro"):
                        posts.append(item)
                        continue
                    if "profiles | LinkedIn" in title:
                        # Generic "X profiles" page, not authoritative
                        continue
                        
                    profile_headlines.append(item)
                
                # FILTER 2: Only keep results for TARGET PERSON (filter out other people)
                # DDG often returns concatenated results with multiple profiles
                name_parts = full_name.lower().split()
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[-1] if len(name_parts) > 1 else ""
                
                # Normalize accents for matching (José -> Jose, François -> Francois)
                first_name_normalized = normalize_accents(first_name)
                last_name_normalized = normalize_accents(last_name)
                
                target_person_results = []
                other_person_results = []
                
                for item in profile_headlines:
                    title_lower = item.get("title", "").lower()
                    # Normalize the title too for accent-insensitive matching
                    title_normalized = normalize_accents(title_lower)
                    
                    # Check if target person's name is in the title (accent-insensitive)
                    # This handles cases like "Jose Varatojo" matching "José Diogo Varatojo"
                    if first_name_normalized in title_normalized and last_name_normalized in title_normalized:
                        target_person_results.append(item)
                    else:
                        other_person_results.append(item)
                
                # Prioritize target person's profile headlines
                if target_person_results:
                    print(f"      📊 DDG Profile Headlines for {full_name}:")
                    for i, item in enumerate(target_person_results[:3], 1):
                        print(f"         {i}. {item.get('title', '')[:70]}")
                    if other_person_results:
                        print(f"      📊 Other profiles filtered out: {len(other_person_results)}")
                    if posts:
                        print(f"      📊 Posts filtered out: {len(posts)}")
                    # Return only target person's profile headlines (with URL match status)
                    return target_person_results[:max_results], url_match_exact
                elif profile_headlines:
                    # No exact name match - TRY NEXT VARIATION instead of returning wrong person
                    # This is important because Yahoo sometimes returns different people first
                    print(f"      ⚠️ No name match in results (found: {profile_headlines[0].get('title', '')[:50]}...)")
                    print(f"      ⏳ Trying next variation to find correct person...")
                    await asyncio.sleep(3)  # Wait before next variation
                    continue  # Try next query variation
                elif posts:
                    # Only posts found (no profile headlines) - return posts
                    print(f"      📊 DDG Posts only (no profile headlines found):")
                    for i, item in enumerate(posts[:3], 1):
                        print(f"         {i}. {item.get('title', '')[:70]}")
                    return posts[:max_results], url_match_exact
                else:
                    # No results at all - try next variation
                    print(f"         ⚠️ No usable results, trying next variation...")
                    continue
            else:
                print(f"         ❌ No results with variation {variation_idx}")
        
        except Exception as e:
            # API/request error - try next variation, but track that API failed
            print(f"         ❌ DDG API error: {str(e)}")
            api_errors.append(str(e))
            continue
    
    # If ALL variations had API errors → raise exception (triggers GSE fallback)
    if len(api_errors) == len(query_variations):
        raise Exception(f"DDG API failed for all {len(query_variations)} variations: {api_errors[0]}")
    
    # DDG worked for at least one variation but no URL match → return empty (NO GSE fallback)
    print(f"   ❌ DDG: No matching profile found after trying {len(query_variations)} variations")
    return [], False

async def search_linkedin_gse(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> List[dict]:
    """
    Search LinkedIn using Google Custom Search Engine for person's profile.
    
    OPTIMIZED: Uses 3 search variations (reduced from 5 for 60% fewer API calls):
    1. Exact URL in quotes (most specific)
    2. Profile slug only (handles www/protocol differences)
    3. Name + site:linkedin.com (broadest fallback)
    
    This provides same coverage as 5 variations but uses fewer API calls:
    - Free tier: 100 queries/day ÷ 3 = ~33 leads/day (vs 20 with 5 variations)
    - Paid tier: 10,000 queries/day ÷ 3 = ~3,333 leads/day

    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: LinkedIn URL provided by miner (required)
        max_results: Max search results to return

    Returns:
        List of search results with title, link, snippet
    """
    if not linkedin_url:
        print(f"   ⚠️ No LinkedIn URL provided")
        return []
    
    # Extract profile slug from LinkedIn URL
    # Example: https://www.linkedin.com/in/tanja-reese-cpa-31477926 → tanja-reese-cpa-31477926
    profile_slug = linkedin_url.split("/in/")[-1].strip("/") if "/in/" in linkedin_url else None
    
    # Build search query variations (in order of specificity)
    # OPTIMIZED: Reduced from 5 to 3 variations (60% fewer API calls)
    # - Removed variation #2 (URL without protocol) - redundant with #3
    # - Removed variation #4 (name + company) - redundant with #5 (broader)
    # Result: 100 queries/day free tier = ~33 leads/day (vs 20 with 5 variations)
    query_variations = [
        # 1. Exact URL in quotes (most specific - handles exact Google index)
        f'"{linkedin_url}"',
        
        # 2. Profile slug only (handles www/protocol differences - REPLACES old #2 and #3)
        f'"linkedin.com/in/{profile_slug}"' if profile_slug else None,
        
        # 4. Name + LinkedIn + company (more context)
        f'"{full_name}" linkedin "{company}"',
        
    ]
    
    # Remove None values
    query_variations = [q for q in query_variations if q]
    
    print(f"   🔍 Trying {len(query_variations)} search variations for LinkedIn profile...")
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        
        async with aiohttp.ClientSession() as session:
            # Try each query variation until we get results
            for variation_idx, query in enumerate(query_variations, 1):
                print(f"      🔄 Variation {variation_idx}/{len(query_variations)}: {query[:80]}...")
                
                params = {
                    "key": GSE_API_KEY,
                    "cx": GSE_CX,
                    "q": query,
                    "num": max_results
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        print(f"         ⚠️ GSE API error: HTTP {response.status}")
                        continue  # Try next variation
                    
                    data = await response.json()
                    results = []
                    
                    for item in data.get("items", []):
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                    
                    if results:
                        # Found results! Now check if we need URL verification
                        print(f"         ✅ Found {len(results)} result(s) with variation {variation_idx}")
                        
                        # Extract all LinkedIn profile URLs from search results
                        found_profile_urls = []
                        found_directory_urls = []
                        
                        for result in results:
                            link = result.get("link", "")
                            if "linkedin.com/in/" in link:
                                # Direct profile link (e.g., /in/john-smith-123)
                                result_slug = link.split("/in/")[-1].strip("/").split("?")[0]
                                found_profile_urls.append(result_slug)
                            elif "linkedin.com/pub/dir/" in link or "/pub/dir/" in link:
                                # Directory page (e.g., /pub/dir/Tanja/Reese)
                                found_directory_urls.append(link)
                        
                        # DECISION LOGIC (UPDATED FOR 4 VARIATIONS):
                        # - Variations 1-2: Searched for exact URL/slug → Always accept (they found the URL)
                        # - Variations 3-4: Searched by name + site OR name + linkedin + company
                        #     → If results contain direct profiles (/in/), verify URL match
                        #     → If results only contain directory pages (/pub/dir/), SKIP URL check
                        #       (directory pages will NEVER match /in/ URLs, so check is useless)
                        #       Instead, rely on LLM to verify person + company from snippets
                        
                        if variation_idx >= 3 and profile_slug:
                            # Using fallback variations (name + company search)
                            
                            if found_profile_urls:
                                # Google returned direct profile links - verify URL match
                                url_match_found = any(
                                    profile_slug.lower() == result_slug.lower() 
                                    for result_slug in found_profile_urls
                                )
                                
                                if not url_match_found:
                                    print(f"         ⚠️  URL MISMATCH: Miner provided '{profile_slug}'")
                                    print(f"         ⚠️  But Google found: {found_profile_urls[:3]}")
                                    print(f"         ❌ Failing - different profile")
                                    continue  # Try next variation
                                else:
                                    print(f"         ✅ URL MATCH: Miner's profile '{profile_slug}' confirmed")
                            
                            elif found_directory_urls:
                                # Google returned directory pages only - SKIP URL check
                                # (Directory pages like /pub/dir/ will never match /in/ profiles)
                                # LLM will verify person + company from snippets instead
                                print(f"         ℹ️  Google returned directory pages (not direct profiles)")
                                print(f"         ℹ️  Skipping URL verification - will rely on LLM to verify person+company")
                            
                            else:
                                # No LinkedIn links at all (shouldn't happen, but handle it)
                                print(f"         ⚠️  No LinkedIn links found in results")
                                continue  # Try next variation
                        
                        # Results are valid - log and return
                        print(f"      📊 GSE results:")
                        for i, result in enumerate(results[:3], 1):  # Show top 3
                            print(f"         {i}. {result['title'][:80]}")
                            print(f"            {result['snippet'][:100]}...")
                        
                        return results
                    else:
                        print(f"         ❌ No results with variation {variation_idx}")
            
            # All variations exhausted
            print(f"   ❌ GSE: No results found after trying {len(query_variations)} variations")
            return []
    
    except asyncio.TimeoutError:
        print(f"   ⚠️ GSE API timeout")
        return []
    except Exception as e:
        print(f"   ⚠️ GSE API error: {str(e)}")
        return []

async def verify_linkedin_with_llm(full_name: str, company: str, linkedin_url: str, search_results: List[dict], url_match_exact: bool = False) -> Tuple[bool, str]:
    """
    Use OpenRouter LLM to verify if search results match the person.

    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: Provided LinkedIn URL
        search_results: Google search results
        url_match_exact: If True, URL slug matched exactly (strong identity proof)

    Returns:
        (is_verified, reasoning)
    """
    try:
        if not search_results:
            return False, "No LinkedIn search results found"
        
        # DETERMINISTIC PRE-CHECK: Check company match from titles directly
        # This avoids LLM hallucination issues
        company_lower = company.lower().strip()
        
        # Normalize apostrophes (DDG returns "mcdonald ' s" with spaces, we need "mcdonald's")
        # Also handle curly apostrophes and other variants
        company_lower = company_lower.replace("'", "'").replace("'", "'").replace("`", "'")
        company_lower = re.sub(r"\s*'\s*", "'", company_lower)  # "mcdonald ' s" → "mcdonald's"
        
        # Normalize company name by removing common legal suffixes
        # e.g., "Bank Of America Corporation" → "Bank Of America"
        # e.g., "Google LLC" → "Google"
        LEGAL_SUFFIXES = [
            " corporation", " corp.", " corp", " incorporated", " inc.", " inc",
            " llc", " l.l.c.", " ltd.", " ltd", " limited", " plc", " p.l.c.",
            " co.", " co", " company", " gmbh", " ag", " sa", " nv", " bv",
            " holdings", " holding", " group", " international", " intl"
        ]
        company_normalized = company_lower
        for suffix in LEGAL_SUFFIXES:
            if company_normalized.endswith(suffix):
                company_normalized = company_normalized[:-len(suffix)].strip()
                break  # Only remove one suffix
        
        company_words = company_normalized.split()  # ["bank", "of", "america"]
        
        # Check first result (most authoritative for target person)
        first_title = search_results[0].get("title", "").lower()
        
        # Normalize apostrophes in title too (DDG returns "mcdonald ' s")
        first_title = first_title.replace("'", "'").replace("'", "'").replace("`", "'")
        first_title = re.sub(r"\s*'\s*", "'", first_title)  # "mcdonald ' s" → "mcdonald's"
        
        # Extract ONLY the headline part (before "| linkedin")
        # DDG often concatenates descriptions after "| LinkedIn"
        # e.g., "Name - Title @ Company | LinkedIn About Company: ..."
        if "| linkedin" in first_title:
            first_title = first_title.split("| linkedin")[0].strip()
        
        # Also get snippet for additional company matching
        first_snippet = search_results[0].get("snippet", "").lower()
        
        # Method 1: Exact normalized company name in title
        company_in_title = company_normalized in first_title
        
        # Method 2: All significant words of company name in title (for multi-word companies)
        # e.g., "Bank Of America" → check if "bank", "of", "america" are all in title
        if not company_in_title and len(company_words) > 1:
            significant_words = [w for w in company_words if len(w) > 2]  # Skip "of", "the", etc.
            company_in_title = all(word in first_title for word in significant_words)
        
        # Method 3: Check snippet for company name
        # First try strict "Experience: [Company]" pattern (LinkedIn profile format)
        # Then try general company name mention (for LinkedIn posts)
        company_in_snippet = False
        if not company_in_title:
            # Check for "experience: [company]" pattern (LinkedIn's standard format)
            if f"experience: {company_normalized}" in first_snippet:
                company_in_snippet = True
            elif f"experience : {company_normalized}" in first_snippet:
                company_in_snippet = True
            # Check for multi-word companies in Experience section
            elif len(company_words) > 1:
                significant_words = [w for w in company_words if len(w) > 2]
                # Must have "experience:" prefix AND all company words
                if "experience:" in first_snippet or "experience :" in first_snippet:
                    # Find the part after "experience:"
                    exp_parts = first_snippet.split("experience:")
                    if len(exp_parts) > 1:
                        experience_section = exp_parts[1][:100]  # First 100 chars after "experience:"
                        if all(word in experience_section for word in significant_words):
                            company_in_snippet = True
            
            # Method 3b: For LinkedIn posts, also accept company name anywhere in snippet
            # This handles cases where Yahoo returns posts with company mentioned in content
            # e.g., "Although Smartcar is a fully remote company..."
            if not company_in_snippet:
                if company_normalized in first_snippet:
                    company_in_snippet = True
                    print(f"   ℹ️  Company found in snippet (general mention)")
                elif len(company_words) > 1:
                    # Multi-word company - check all significant words
                    significant_words = [w for w in company_words if len(w) > 2]
                    if all(word in first_snippet for word in significant_words):
                        company_in_snippet = True
                        print(f"   ℹ️  Company words found in snippet (general mention)")
        
        # Deterministic company match decision (title OR snippet)
        deterministic_company_match = company_in_title or company_in_snippet
        
        # Show normalized company name if it differs from original
        match_location = "title" if company_in_title else ("snippet" if company_in_snippet else "NOT FOUND")
        if company_normalized != company_lower:
            print(f"   🔍 Deterministic check: Company '{company}' (normalized: '{company_normalized}') in {match_location} = {deterministic_company_match}")
        else:
            print(f"   🔍 Deterministic check: Company '{company}' in {match_location} = {deterministic_company_match}")
        print(f"      First title: {first_title[:80]}...")
        if company_in_snippet and not company_in_title:
            print(f"      First snippet: {first_snippet[:80]}...")
        
        # Prepare search results for LLM
        results_text = json.dumps(search_results, indent=2)
        
        # Build LinkedIn URL line (only if provided)
        linkedin_url_line = f"LinkedIn URL Provided: {linkedin_url}\n" if linkedin_url else ""
        
        # Explicit 3-check prompt: name match + company match + profile valid
        prompt = f"""You are validating a LinkedIn profile for B2B lead generation. Analyze these search results.

PROVIDED INFORMATION:
- Expected Name: {full_name}
- Expected Company: {company}
- LinkedIn URL: {linkedin_url}

SEARCH RESULTS:
{results_text}

CHECK THREE CRITERIA SEPARATELY:

1. NAME MATCH: Does the person name in search results match "{full_name}"?
   - Look at the profile title (e.g., "John Smith - CEO" vs "Jane Doe - VP")
   - Names must substantially match
   - Different people = name_match FALSE (e.g., "John Black" ≠ "Pranav Ramesh")

2. COMPANY MATCH: Does the profile TITLE show "{company}" as current employer?
   - ONLY look at the TITLE (e.g., "Name - Title at Company | LinkedIn")
   - IGNORE the snippet/description - it may contain outdated or unrelated info
   - ACCEPT if "{company}" appears in the TITLE
   - REJECT if TITLE shows a DIFFERENT company (e.g., "Name - CEO at OtherCorp")
   - REJECT if "Exited", "Former", or "Left" in TITLE about "{company}"
   - If NO company in TITLE, company_match = FALSE

3. PROFILE VALID: Is profile legitimate and indexed?
   - Profile appears in search results = valid

CRITICAL: Check name AND company separately. Both must match.

Respond ONLY with JSON: {{"name_match": true/false, "company_match": true/false, "profile_valid": true/false, "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}"""
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1  # Low temperature for consistency
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            ) as response:
                if response.status != 200:
                    return False, f"LLM API error: HTTP {response.status}"
                
                data = await response.json()
                llm_response = data["choices"][0]["message"]["content"]
                
                # Strip markdown code blocks if present (LLM sometimes wraps JSON in ```json ... ```)
                llm_response = llm_response.strip()
                if llm_response.startswith("```"):
                    # Remove opening ```json or ```
                    lines = llm_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # Remove first line
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]  # Remove last line
                    llm_response = "\n".join(lines).strip()
                
                # Parse JSON response with 3 separate checks
                result = json.loads(llm_response)
                
                name_match = result.get("name_match", False)
                llm_company_match = result.get("company_match", False)
                profile_valid = result.get("profile_valid", False)
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                # OVERRIDE 1: Use deterministic company match instead of LLM's decision
                # This prevents LLM hallucination issues
                company_match = deterministic_company_match
                
                if llm_company_match != deterministic_company_match:
                    print(f"   ⚠️ LLM company_match ({llm_company_match}) OVERRIDDEN by deterministic check ({deterministic_company_match})")
                    reasoning = f"[Deterministic: company '{company}' {'found' if deterministic_company_match else 'NOT found'} in title] {reasoning}"
                
                # OVERRIDE 2: If URL matched EXACTLY + company matched deterministically,
                # override LLM's name_match and confidence decisions.
                # 
                # WHY THIS IS SAFE:
                # - URL can only match if DDG found that URL when searching for the CLAIMED NAME
                # - DDG searches "Pranav Ramesh" → only returns results for that name
                # - If miner gave a different person's URL, it wouldn't appear in those results
                # - So exact URL match = the profile IS for the claimed person
                # 
                # WHY THIS IS NEEDED:
                # - DDG often returns concatenated results with OTHER people's headlines
                # - LLM compares those wrong headlines against claimed name → false negative
                # - Example: Search "Melissa Carberry" → DDG returns correct URL but 
                #   headlines mixed with other people → LLM says "name mismatch"
                #
                # REQUIRES BOTH:
                # - url_match_exact=True (strong identity proof from URL slug)
                # - deterministic_company_match=True (they work at right company)
                if url_match_exact and deterministic_company_match:
                    if not name_match or confidence < 0.5:
                        print(f"   ✅ URL EXACT MATCH + COMPANY MATCH: Overriding LLM decision")
                        print(f"      (URL slug is authoritative identity proof when searching by name)")
                        name_match = True
                        confidence = max(confidence, 0.8)  # Boost confidence for strong deterministic proof
                        profile_valid = True  # URL exists = profile is valid
                        reasoning = f"[URL exact match + company verified - identity confirmed] {reasoning}"
                
                # DEBUG: Print LLM analysis with all 3 checks
                print(f"   🤖 LLM Analysis:")
                print(f"      Name Match: {name_match} (Does {full_name} match the profile?)")
                print(f"      Company Match: {company_match} (Deterministic from title)")
                print(f"      Profile Valid: {profile_valid} (Is profile legitimate?)")
                print(f"      Confidence: {confidence}")
                print(f"      Reasoning: {reasoning}")
                
                # Verification passes ONLY if ALL THREE criteria are met:
                # 1. Name matches (prevents using wrong person's LinkedIn)
                # 2. Company matches (prevents outdated employment)
                # 3. Profile valid (prevents fake profiles)
                # 4. Confidence >= 0.5
                if name_match and company_match and profile_valid and confidence >= 0.5:
                    return True, reasoning
                else:
                    # Build detailed failure reason
                    failures = []
                    if not name_match:
                        failures.append("name mismatch")
                    if not company_match:
                        failures.append("company mismatch")
                    if not profile_valid:
                        failures.append("invalid profile")
                    if confidence < 0.5:
                        failures.append("low confidence")
                    
                    failure_str = ", ".join(failures) if failures else "unknown"
                    detailed_reason = f"{reasoning} [Failed: {failure_str}]"
                    return False, detailed_reason
    
    except asyncio.TimeoutError:
        return False, "LLM API timeout"
    except json.JSONDecodeError as e:
        return False, f"LLM response parsing error: {str(e)}"
    except Exception as e:
        return False, f"LLM verification error: {str(e)}"

async def check_linkedin_gse(lead: dict) -> Tuple[bool, dict]:
    """
    Stage 4: LinkedIn/GSE validation (HARD check).
    
    Verifies that the person works at the company using:
    1. Google Custom Search (LinkedIn)
    2. OpenRouter LLM verification
    
    This is a HARD check - instant rejection if fails.

    Args:
        lead: Lead data with full_name, company, linkedin

    Returns:
        (passed, rejection_reason)
    """
    try:
        full_name = lead.get("full_name") or lead.get("Full_name") or lead.get("Full Name")
        company = get_company(lead)
        linkedin_url = get_linkedin(lead)
        
        if not full_name:
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": "Missing full_name",
                "failed_fields": ["full_name"]
            }
        
        if not company:
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": "Missing company",
                "failed_fields": ["company"]
            }
        
        if not linkedin_url:
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": "Missing linkedin URL",
                "failed_fields": ["linkedin"]
            }
        
        # Step 1: Search LinkedIn via DuckDuckGo (FREE, default) or Google Custom Search (fallback)
        print(f"   🔍 Stage 4: Verifying LinkedIn profile for {full_name} at {company}")
        
        # DuckDuckGo is FREE and default - no API key needed
        # GSE is fallback ONLY if DDG has API/request error (not URL mismatch)
        url_match_exact = False  # Track if URL slug matched exactly (strong identity proof)
        if USE_DDG_SEARCH:
            try:
                search_results, url_match_exact = await search_linkedin_ddg(full_name, company, linkedin_url)
                # NO GSE fallback if DDG returns empty (URL mismatch or no results)
                # This prevents gaming: if DDG can't find the profile, it's likely invalid
            except Exception as e:
                print(f"   ⚠️ DuckDuckGo API/request failed: {e}, falling back to GSE")
                search_results = await search_linkedin_gse(full_name, company, linkedin_url)
                url_match_exact = False  # GSE doesn't return URL match status
        else:
            search_results = await search_linkedin_gse(full_name, company, linkedin_url)
            url_match_exact = False  # GSE doesn't return URL match status
        
        # Store search count in lead for data collection
        lead["gse_search_count"] = len(search_results)
        
        if not search_results:
            # Store LLM confidence as "none" when no search results
            lead["llm_confidence"] = "none"
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": f"LinkedIn profile {linkedin_url} not found in Google's index (may be private or invalid)",
                "failed_fields": ["linkedin"]
            }
        
        # Step 2: Verify with LLM (pass URL match status for identity override)
        verified, reasoning = await verify_linkedin_with_llm(full_name, company, linkedin_url, search_results, url_match_exact)
        
        # Store LLM confidence (low, medium, high, or "none")
        # This is derived from the LLM's confidence score
        lead["llm_confidence"] = "medium"  # Default, can be enhanced later
        
        if not verified:
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": f"LinkedIn verification failed: {reasoning}",
                "failed_fields": ["linkedin"]
            }
        
        print(f"   ✅ Stage 4: LinkedIn verified for {full_name} at {company}")
        return True, {}
    
    except Exception as e:
        return False, {
            "stage": "Stage 4: LinkedIn/GSE Validation",
            "check_name": "check_linkedin_gse",
            "message": f"LinkedIn/GSE check failed: {str(e)}",
            "failed_fields": ["linkedin"]
        }

# Rep Score: Soft Reputation Checks (SOFT - always passes, appends score)

async def check_wayback_machine(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check domain history in Wayback Machine.
    
    Returns score (0-6) based on:
    - Number of snapshots
    - Age of domain in archive
    - Consistency of snapshots
    
    This is a SOFT check - always passes, appends score.
    
    Args:
        lead: Lead data with website
    
    Returns:
        (score, metadata)
    """
    try:
        website = get_website(lead)
        if not website:
            return 0, {"checked": False, "reason": "No website provided"}
        
        domain = extract_root_domain(website)
        if not domain:
            return 0, {"checked": False, "reason": "Invalid website format"}
        
        # Query Wayback Machine CDX API (with 3 retries for timeout)
        url = f"https://web.archive.org/cdx/search/cdx"
        params = {
            "url": domain,
            "output": "json",
            "limit": 1000,
            "fl": "timestamp"
        }
        
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=15) as response:
                        if response.status != 200:
                            return 0, {"checked": False, "reason": f"Wayback API error: {response.status}"}
                        
                        data = await response.json()
                        
                        if len(data) <= 1:  # First row is header
                            return 0, {"checked": True, "snapshots": 0, "reason": "No archive history"}
                        
                        snapshots = len(data) - 1  # Exclude header
                        
                        # Parse timestamps to calculate age
                        timestamps = [row[0] for row in data[1:]]  # Skip header
                        oldest = timestamps[0] if timestamps else None
                        newest = timestamps[-1] if timestamps else None
                        
                        # Calculate age in years
                        if oldest:
                            oldest_year = int(oldest[:4])
                            current_year = datetime.now().year
                            age_years = current_year - oldest_year
                        else:
                            age_years = 0
                        
                        # Scoring logic (UPDATED: max 6 points for Wayback):
                        if snapshots < 10:
                            score = min(1.2, snapshots * 0.12)
                        elif snapshots < 50:
                            score = 1.8 + (snapshots - 10) * 0.03
                        elif snapshots < 200:
                            score = 3.6 + (snapshots - 50) * 0.008
                        else:
                            score = 5.4 + min(0.6, (snapshots - 200) * 0.0006)
                        
                        # Age bonus
                        if age_years >= 5:
                            score = min(6, score + 0.6)
                        
                        return score, {
                            "checked": True,
                            "snapshots": snapshots,
                            "age_years": age_years,
                            "oldest_snapshot": oldest,
                            "newest_snapshot": newest,
                            "score": score
                        }
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                return 0, {"checked": False, "reason": "Wayback API timeout (3 attempts)"}
            except Exception as e:
                return 0, {"checked": False, "reason": f"Wayback check error: {str(e)}"}
        
        # Fallback if loop completes without returning
        return 0, {"checked": False, "reason": "Wayback check failed unexpectedly"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"Wayback check error: {str(e)}"}

# DEPRECATED: USPTO check removed (API unreliable, scoring adjusted)
# async def check_uspto_trademarks(lead: dict) -> Tuple[float, dict]:
#     """
#     Rep Score: Check USPTO for company trademarks.
#     
#     DEPRECATED: Removed due to USPTO API reliability issues.
#     Points redistributed to other checks (Wayback: 6→8, SEC: 12→14, WHOIS/DNSBL: 10→12)
#     """
#     return 0, {"checked": False, "reason": "USPTO check deprecated"}

async def check_uspto_trademarks(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: USPTO check (DISABLED).
    
    This check has been disabled due to API reliability issues.
    Always returns 0 points.
    
    Returns:
        (0, metadata indicating check is disabled)
    """
    return 0, {"checked": False, "reason": "USPTO check disabled"}

async def check_sec_edgar(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check SEC EDGAR for company filings.
    
    Returns score (0-12) based on:
    - Number of filings
    - Recent filing activity
    - Types of filings (10-K, 10-Q, 8-K)
    
    This is a SOFT check - always passes, appends score.
    Uses official SEC.gov API (free, no API key needed - just User-Agent)
    
    Args:
        lead: Lead data with company
    
    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}
        
        print(f"   🔍 SEC: Searching for company: '{company}'")
        
        # SEC.gov requires User-Agent header with contact info (no API key needed)
        headers = {
            "User-Agent": "LeadPoet/1.0 (hello@leadpoet.com)"
        }
        
        # Try multiple company name variations for better matching
        # SEC often uses abbreviated forms (e.g., "Microsoft Corp" not "Microsoft Corporation")
        company_variations = [
            company,  # Original name
            company.replace(" Company, Inc.", "").replace(" Corporation", " Corp").replace(", Inc.", ""),  # Abbreviated
            company.split()[0] if len(company.split()) > 1 else company,  # First word only (e.g., "Microsoft")
        ]
        
        print(f"      🔍 Trying {len(company_variations)} name variations: {company_variations}")
        
        # Use SEC.gov company search endpoint to find CIK
        # This searches the submissions index for company name matches
        search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        
        # Try each variation until we find results
        async with aiohttp.ClientSession() as session:
            for idx, company_variation in enumerate(company_variations):
                print(f"      🔄 Attempt {idx+1}/{len(company_variations)}: Searching for '{company_variation}'")
                
                # Request actual filings, not just company landing page
                # type=&dateb=&owner=include&start=0
                params = {
                    "company": company_variation,
                    "action": "getcompany",
                    "type": "",  # All filing types
                    "dateb": "",  # All dates
                    "owner": "include",  # Include company filings
                    "start": "0",  # Start from first filing
                    "count": "100"  # Get up to 100 recent filings
                }
                
                async with session.get(search_url, headers=headers, params=params, timeout=15) as response:
                    if response.status != 200:
                        print(f"      ❌ SEC API returned HTTP {response.status}")
                        continue  # Try next variation
                    
                    # Parse HTML response (SEC doesn't return JSON for this endpoint)
                    html = await response.text()
                    print(f"      📄 SEC response length: {len(html)} bytes")
                    
                    # Check if company was found (HTML contains "No matching" if not found)
                    if "No matching" in html or "No results" in html:
                        print(f"      ❌ SEC: 'No matching' found for '{company_variation}'")
                        continue  # Try next variation
                    
                    # Found a result! Count filing indicators in HTML
                    print(f"      ✅ SEC: Found match for '{company_variation}'")
                    filing_types = ["10-K", "10-Q", "8-K", "S-1", "10-K/A", "10-Q/A", "4", "3", "SC 13", "DEF 14A"]
                    total_filings = 0
                    for filing_type in filing_types:
                        # Look for the filing type in HTML context (e.g., ">10-K<" or " 10-K ")
                        count = html.count(f">{filing_type}<") + html.count(f" {filing_type} ")
                        if count > 0:
                            print(f"      📊 Found {count}x {filing_type}")
                        total_filings += count
                    
                    print(f"      📊 Total filings detected: {total_filings}")
                    
                    if total_filings == 0:
                        # The HTML might be a landing page with a link to the actual filings
                        # Try to extract CIK from the HTML and query directly
                        import re
                        cik_match = re.search(r'CIK=(\d{10})', html)
                        if cik_match:
                            cik = cik_match.group(1)
                            print(f"      🔍 Found CIK: {cik}, fetching actual filings...")
                            
                            # Query the filings page directly using CIK
                            cik_params = {
                                "action": "getcompany",
                                "CIK": cik,
                                "type": "",
                                "dateb": "",
                                "owner": "include",
                                "count": "100"
                            }
                            
                            async with session.get(search_url, headers=headers, params=cik_params, timeout=15) as cik_response:
                                if cik_response.status == 200:
                                    cik_html = await cik_response.text()
                                    print(f"      📄 CIK response length: {len(cik_html)} bytes")
                                    
                                    # Count filings again (use HTML-aware matching)
                                    total_filings = 0
                                    for filing_type in filing_types:
                                        count = cik_html.count(f">{filing_type}<") + cik_html.count(f" {filing_type} ")
                                        if count > 0:
                                            print(f"      📊 Found {count}x {filing_type}")
                                        total_filings += count
                                    
                                    # DEBUG: Check if HTML contains filing table markers
                                    has_filing_table = "filingTable" in cik_html or "Filing" in cik_html
                                    print(f"      🔍 DEBUG: Has 'filingTable' or 'Filing': {has_filing_table}")
                                    
                                    # If we have a valid CIK and filing indicators but can't parse exact counts,
                                    # give partial credit (company IS SEC-registered with filings)
                                    if total_filings == 0 and has_filing_table:
                                        print(f"      ⚠️  CIK {cik} has filings but HTML parsing failed")
                                        print(f"      ✅ SEC: Giving partial credit (3.6/12) for SEC-registered company")
                                        return 3.6, {
                                            "checked": True,
                                            "filings": "unknown (parsing failed)",
                                            "score": 3.6,
                                            "cik": cik,
                                            "company_name_used": company_variation,
                                            "reason": f"Company registered with SEC (CIK {cik}) but exact filing count unavailable"
                                        }
                                    
                                    if total_filings > 0:
                                        # Success! Calculate score
                                        print(f"      📊 Total filings detected: {total_filings}")
                                        
                                        if total_filings <= 5:
                                            score = min(3.6, total_filings * 0.72)
                                        elif total_filings <= 20:
                                            score = 7.2
                                        elif total_filings <= 50:
                                            score = 9.6
                                        else:
                                            score = 12
                                        
                                        print(f"      ✅ SEC: {score}/12 pts for CIK {cik}")
                                        return score, {
                                            "checked": True,
                                            "filings": total_filings,
                                            "score": score,
                                            "cik": cik,
                                            "company_name_used": company_variation,
                                            "reason": f"Found {total_filings} SEC filing indicators for CIK {cik}"
                                        }
                        
                        print(f"      ⚠️  Match found but no filing types detected (showing first 500 chars):")
                        print(f"         {html[:500]}")
                        continue  # Try next variation
                    
                    # Scoring logic (UPDATED: max 12 points for SEC):
                    # - 1-5 filings: 3.6 points
                    # - 6-20 filings: 7.2 points
                    # - 21-50 filings: 9.6 points
                    # - 50+ filings: 12 points
                    
                    if total_filings <= 5:
                        score = min(3.6, total_filings * 0.72)
                    elif total_filings <= 20:
                        score = 7.2
                    elif total_filings <= 50:
                        score = 9.6
                    else:
                        score = 12
                    
                    print(f"      ✅ SEC: {score}/12 pts for '{company_variation}'")
                    return score, {
                        "checked": True,
                        "filings": total_filings,
                        "score": score,
                        "company_name_used": company_variation,
                        "reason": f"Found {total_filings} SEC filing indicators for {company_variation}"
                    }
            
            # All variations failed
            print(f"      ❌ SEC: No results found for any name variation")
            return 0, {
                "checked": True,
                "filings": 0,
                "variations_tried": company_variations,
                "reason": f"No SEC filings found for {company} (tried {len(company_variations)} variations)"
            }

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "SEC API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"SEC check error: {str(e)}"}


async def check_gdelt_mentions(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check GDELT for press mentions and trusted domain coverage.
    
    Returns score (0-10) based on:
    - Press wire mentions (PRNewswire, BusinessWire, GlobeNewswire, ENPresswire)
    - Trusted domain mentions (.edu, .gov, high-authority sites)
    
    This is a SOFT check - always passes, appends score.
    Uses GDELT 2.0 DOC API (free, no API key needed)
    
    Scoring breakdown:
    - 0-5 points: Press wire mentions (verified company PR)
    - 0-5 points: Trusted domain mentions (.edu, .gov, DA>60)
    
    Args:
        lead: Lead data with company
    
    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}
        
        print(f"   🔍 GDELT: Searching for company: '{company}'")
        
        # GDELT 2.0 DOC API endpoint
        # Uses free public API - no key required
        gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Query for company mentions in last 3 months
        # Format: "company name" sourcelang:eng
        # NOTE: GDELT requires minimum 5 characters in query, so append "company" for short names
        search_term = company
        if len(company) <= 4:
            search_term = f"{company} company"
            print(f"      ℹ️  Short name detected, searching: '{search_term}'")
        query = f'"{search_term}" sourcelang:eng'
        
        async with aiohttp.ClientSession() as session:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": 250,  # Get up to 250 recent articles
                "format": "json",
                "sort": "datedesc"
            }
            
            async with session.get(gdelt_url, params=params, timeout=15) as response:
                if response.status != 200:
                    print(f"      ❌ GDELT API returned HTTP {response.status}")
                    return 0, {
                        "checked": False,
                        "reason": f"GDELT API error: HTTP {response.status}"
                    }
                
                # GDELT sometimes returns HTML instead of JSON for short/uncommon company names
                # Check Content-Type before parsing to avoid json decode errors
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    # GDELT returned HTML page - treat as no coverage (not an error)
                    print(f"      ⚠️  GDELT returned HTML instead of JSON (no articles for '{company}')")
                    return 0, {
                        "checked": True,
                        "press_mentions": 0,
                        "trusted_mentions": 0,
                        "reason": f"No GDELT coverage found for {company}"
                    }
                
                data = await response.json()
                articles = data.get("articles", [])
                print(f"      📰 GDELT found {len(articles)} articles")
                
                if not articles:
                    print(f"      ❌ No GDELT articles found for '{company}'")
                    return 0, {
                        "checked": True,
                        "press_mentions": 0,
                        "trusted_mentions": 0,
                        "reason": f"No GDELT coverage found for {company}"
                    }
                
                # Parse articles for press wires and trusted domains
                press_wire_domains = {
                    "prnewswire.com",
                    "businesswire.com",
                    "globenewswire.com",
                    "enpresswire.com",
                    "prweb.com",
                    "marketwired.com"
                }
                
                trusted_tlds = {".edu", ".gov", ".mil"}
                
                # High-authority domains (Fortune 500, major news outlets, financial news)
                high_authority_domains = {
                    # Major news outlets
                    "forbes.com", "fortune.com", "bloomberg.com", "wsj.com",
                    "nytimes.com", "reuters.com", "ft.com", "economist.com",
                    "theguardian.com", "washingtonpost.com", "bbc.com", "cnbc.com",
                    # Tech news
                    "techcrunch.com", "wired.com", "theverge.com", "cnet.com",
                    "arstechnica.com", "zdnet.com", "venturebeat.com",
                    # Financial news
                    "finance.yahoo.com", "yahoo.com", "marketwatch.com", "fool.com",
                    "seekingalpha.com", "investing.com", "benzinga.com", "zacks.com",
                    "morningstar.com", "barrons.com", "investopedia.com",
                    # International business news
                    "thehindubusinessline.com", "business-standard.com", "economictimes.indiatimes.com",
                    "scmp.com", "japantimes.co.jp", "straitstimes.com"
                }
                
                press_mentions = []
                trusted_mentions = []
                seen_domains = set()  # Track unique domains (no spam)
                all_domains_found = []  # DEBUG: Track all domains for logging
                
                for article in articles:
                    url = article.get("url", "")
                    domain = article.get("domain", "")
                    title = article.get("title", "")
                    
                    # DEBUG: Track all domains
                    if domain:
                        all_domains_found.append(domain)
                    
                    # Skip if we've seen this domain (cap at 3 mentions per domain)
                    if domain in seen_domains:
                        domain_count = sum(1 for m in trusted_mentions if m["domain"] == domain)
                        if domain_count >= 3:
                            continue
                    
                    seen_domains.add(domain)
                    
                    # Check if company name appears in title (stronger signal)
                    company_in_title = company.lower() in title.lower()
                    
                    # Check for press wire mentions
                    is_press_wire = any(wire in domain for wire in press_wire_domains)
                    if is_press_wire:
                        press_mentions.append({
                            "domain": domain,
                            "url": url[:100],
                            "title": title[:100],
                            "company_in_title": company_in_title
                        })
                    
                    # Check for trusted domain mentions
                    is_trusted_tld = any(domain.endswith(tld) for tld in trusted_tlds)
                    is_high_authority = any(auth in domain for auth in high_authority_domains)
                    
                    if is_trusted_tld or is_high_authority:
                        trusted_mentions.append({
                            "domain": domain,
                            "url": url[:100],
                            "title": title[:100],
                            "company_in_title": company_in_title,
                            "type": "tld" if is_trusted_tld else "high_authority"
                        })
                
                # DEBUG: Print domain analysis
                unique_domains = set(all_domains_found)
                print(f"      🌐 Unique domains in articles: {len(unique_domains)}")
                print(f"      📰 Press wire matches: {len(press_mentions)}")
                print(f"      🏛️  Trusted domain matches: {len(trusted_mentions)}")
                
                # Show sample of domains if we didn't find any matches
                if len(press_mentions) == 0 and len(trusted_mentions) == 0 and len(unique_domains) > 0:
                    sample_domains = list(unique_domains)[:10]
                    print(f"      🔍 Sample domains (showing first 10):")
                    for d in sample_domains:
                        print(f"         - {d}")
                
                # Calculate score
                # Press wire mentions: 0-5 points
                # - 1+ mention: 2 points
                # - 3+ mentions: 3 points
                # - 5+ mentions: 4 points
                # - 10+ mentions: 5 points
                press_score = 0
                if len(press_mentions) >= 10:
                    press_score = 5.0
                elif len(press_mentions) >= 5:
                    press_score = 4.0
                elif len(press_mentions) >= 3:
                    press_score = 3.0
                elif len(press_mentions) >= 1:
                    press_score = 2.0
                
                # Trusted domain mentions: 0-5 points
                # - 1+ mention: 2 points
                # - 3+ mentions: 3 points
                # - 5+ mentions: 4 points
                # - 10+ mentions: 5 points
                trusted_score = 0
                if len(trusted_mentions) >= 10:
                    trusted_score = 5.0
                elif len(trusted_mentions) >= 5:
                    trusted_score = 4.0
                elif len(trusted_mentions) >= 3:
                    trusted_score = 3.0
                elif len(trusted_mentions) >= 1:
                    trusted_score = 2.0
                
                total_score = press_score + trusted_score
                
                print(f"      ✅ GDELT: {total_score}/10 pts (Press: {press_score}/5, Trusted: {trusted_score}/5)")
                print(f"         Press wires: {len(press_mentions)}, Trusted domains: {len(trusted_mentions)}")
                
                return total_score, {
                    "checked": True,
                    "score": total_score,
                    "press_score": press_score,
                    "trusted_score": trusted_score,
                    "press_mentions_count": len(press_mentions),
                    "trusted_mentions_count": len(trusted_mentions),
                    "press_mentions": press_mentions[:5],  # Sample of top 5
                    "trusted_mentions": trusted_mentions[:5],  # Sample of top 5
                    "reason": f"GDELT coverage: {len(press_mentions)} press mentions, {len(trusted_mentions)} trusted domain mentions"
                }

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "GDELT API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"GDELT check error: {str(e)}"}


async def check_companies_house(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check UK Companies House registry.
    
    Returns score (0-10) based on company found in UK Companies House.
    This is a SOFT check - always passes, appends score.
    Uses UK Companies House API (free, requires API key registration).
    
    API Key: Register at https://developer.company-information.service.gov.uk/
    If API key not configured, returns 0 points and continues.
    
    Args:
        lead: Lead data with company
    
    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}
        
        if not COMPANIES_HOUSE_API_KEY or COMPANIES_HOUSE_API_KEY == "":
            print(f"   ❌ Companies House: API key not configured - skipping check (0 points)")
            return 0, {
                "checked": True,
                "score": 0,
                "reason": "Companies House API key not configured (register at https://developer.company-information.service.gov.uk/)"
            }
        
        print(f"   🔍 Companies House: Searching for '{company}'")
        
        import base64
        auth_b64 = base64.b64encode(f"{COMPANIES_HOUSE_API_KEY}:".encode()).decode()
        search_url = "https://api.company-information.service.gov.uk/search/companies"
        
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Basic {auth_b64}"}
            
            async with session.get(
                search_url,
                headers=headers,
                params={"q": company, "items_per_page": 5},
                timeout=10
            ) as response:
                if response.status != 200:
                    return 0, {"checked": False, "reason": f"Companies House API error: HTTP {response.status}"}
                
                data = await response.json()
                items = data.get("items", [])
                
                if not items:
                    print(f"      ❌ Companies House: No results found")
                    return 0, {"checked": True, "score": 0, "reason": "Company not found in UK Companies House"}
                
                company_upper = company.upper()
                for item in items[:5]:
                    ch_name = item.get("title", "").upper()
                    status = item.get("company_status", "").lower()
                    
                    if company_upper == ch_name:
                        score = 10.0 if status == "active" else 8.0
                    elif company_upper in ch_name or ch_name in company_upper:
                        score = 8.0 if status == "active" else 6.0
                    else:
                        continue
                    
                    print(f"      ✅ Companies House: Found - {item.get('title')} ({status})")
                    return score, {
                        "checked": True,
                        "score": score,
                        "matched_company": item.get("title"),
                        "company_status": status
                    }
                
                return 0, {"checked": True, "score": 0, "reason": "No close name match"}
    
    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "Companies House API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"Companies House check error: {str(e)}"}


async def check_whois_dnsbl_reputation(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: WHOIS + DNSBL reputation check using cached validator data.
    
    Returns score (0-10) based on:
    - WHOIS Stability: 0-3 points (whois_updated_days_ago)
    - Registrant Consistency: 0-3 points (corporate signals)
    - Hosting Provider: 0-3 points (nameservers)
    - DNSBL: 0-1 points (not blacklisted)
    
    This is a SOFT check - always passes, appends score.
    Uses FREE data already collected in Stage 1 (WHOIS) and Stage 2 (DNSBL).
    
    Mirrors TypeScript calculate-rep-score/checks/operational.ts checks.
    
    Args:
        lead: Lead data with WHOIS and DNSBL fields
    
    Returns:
        (score, metadata)
    """
    try:
        score = 0
        details = {
            "whois_stability": 0,
            "registrant_consistency": 0,
            "hosting_provider": 0,
            "dnsbl": 0
        }
        
        # ============================================================
        # 1. WHOIS Stability (0-3 points)
        # ============================================================
        # TypeScript: checkWhoisStabilityDays() - 4 points
        # Python: 3 points (scaled down for 10-point total)
        #
        # Checks if WHOIS record was updated recently (instability signal)
        # Recent updates indicate potential domain instability, ownership changes, 
        # or drop-catch scenarios
        # ============================================================
        
        whois_updated_days = lead.get("whois_updated_days_ago")
        if isinstance(whois_updated_days, (int, float)) and whois_updated_days >= 0:
            # Scoring:
            # >= 180 days (6 months): 3.0 points (very stable)
            # >= 90 days (3 months): 2.0 points (stable)
            # >= 30 days (1 month): 1.0 points (acceptable)
            # < 30 days: 0 points (unstable)
            if whois_updated_days >= 180:
                details["whois_stability"] = 3.0
            elif whois_updated_days >= 90:
                details["whois_stability"] = 2.0
            elif whois_updated_days >= 30:
                details["whois_stability"] = 1.0
            else:
                details["whois_stability"] = 0
            
            score += details["whois_stability"]
            details["whois_updated_days_ago"] = whois_updated_days
        else:
            # Fallback: Use domain age if WHOIS update date not available
            domain_age = lead.get("domain_age_days")
            if isinstance(domain_age, (int, float)) and domain_age > 30:
                # Old domain, assume stable (weak signal)
                details["whois_stability"] = 1.0
                score += 1.0
                details["whois_updated_days_ago"] = "unavailable (used domain_age fallback)"
        
        # ============================================================
        # 2. Registrant Consistency (0-3 points)
        # ============================================================
        # TypeScript: checkRegistrantConsistency() - 3 points
        # Python: 3 points
        #
        # Counts corporate signals:
        # - Corporate registrar name (Inc, LLC, Corp, etc.)
        # - Reputable hosting providers in nameservers
        # - Established domain (> 1 year old)
        # ============================================================
        
        corporate_signals = []
        
        # Check registrar for corporate keywords
        registrar = lead.get("domain_registrar", "")
        if registrar:
            corporate_keywords = ["inc", "corp", "llc", "ltd", "company", "corporation", 
                                 "enterprises", "group", "holdings"]
            registrar_lower = registrar.lower()
            if any(keyword in registrar_lower for keyword in corporate_keywords):
                corporate_signals.append("corporate_registrant")
        
        # Check for reputable hosting providers in nameservers
        nameservers = lead.get("domain_nameservers", [])
        if isinstance(nameservers, list) and len(nameservers) > 0:
            reputable_providers = ["aws", "google", "cloudflare", "azure", "amazon"]
            for ns in nameservers:
                ns_lower = str(ns).lower()
                if any(provider in ns_lower for provider in reputable_providers):
                    corporate_signals.append("reputable_hosting")
                    break
        
        # Check domain age (> 1 year = established)
        domain_age = lead.get("domain_age_days", 0)
        if domain_age > 365:
            corporate_signals.append("established_domain")
        
        # Score based on signals count
        # 3+ signals: 3 points
        # 2 signals: 2 points
        # 1 signal: 1 point
        # 0 signals: 0 points
        if len(corporate_signals) >= 3:
            details["registrant_consistency"] = 3.0
        elif len(corporate_signals) == 2:
            details["registrant_consistency"] = 2.0
        elif len(corporate_signals) == 1:
            details["registrant_consistency"] = 1.0
        else:
            details["registrant_consistency"] = 0
        
        score += details["registrant_consistency"]
        details["corporate_signals"] = corporate_signals
        
        # ============================================================
        # 3. Hosting Provider Reputation (0-3 points)
        # ============================================================
        # TypeScript: checkHostingProviderReputation() - 3 points
        # Python: 3 points
        #
        # Checks if domain is hosted on reputable infrastructure:
        # AWS, Google Cloud, Cloudflare, Azure, Amazon
        # ============================================================
        
        if isinstance(nameservers, list) and len(nameservers) > 0:
            reputable_providers = ["aws", "google", "cloudflare", "azure", "amazon"]
            found_provider = None
            
            for ns in nameservers:
                ns_lower = str(ns).lower()
                for provider in reputable_providers:
                    if provider in ns_lower:
                        found_provider = provider
                        break
                if found_provider:
                    break
            
            if found_provider:
                details["hosting_provider"] = 3.0
                details["hosting_provider_name"] = found_provider
                score += 3.0
        
        # ============================================================
        # 4. DNSBL Reputation (0-1 points)
        # ============================================================
        # TypeScript: checkDnsblReputation() - 1 point
        # Python: 1 point
        #
        # Checks if domain is NOT blacklisted in Spamhaus DBL
        # Uses FREE data already collected in Stage 2
        # ============================================================
        
        dnsbl_checked = lead.get("dnsbl_checked")
        dnsbl_blacklisted = lead.get("dnsbl_blacklisted")
        
        if dnsbl_checked:
            if not dnsbl_blacklisted:
                details["dnsbl"] = 1.0
                score += 1.0
                details["dnsbl_status"] = "clean"
            else:
                details["dnsbl"] = 0
                details["dnsbl_status"] = "blacklisted"
                details["dnsbl_list"] = lead.get("dnsbl_list", "unknown")
        
        # ============================================================
        # Return final score and details
        # ============================================================
        
        return score, {
            "checked": True,
            "score": score,
            "max_score": 10,
            "details": details,
            "reason": f"WHOIS/DNSBL reputation: {score:.1f}/10 (Stability: {details['whois_stability']}, Consistency: {details['registrant_consistency']}, Hosting: {details['hosting_provider']}, DNSBL: {details['dnsbl']})"
        }
        
    except Exception as e:
        return 0, {
            "checked": False,
            "reason": f"WHOIS/DNSBL check error: {str(e)}"
        }


async def check_terms_attestation(lead: dict) -> Tuple[bool, dict]:
    """
    Verify miner's attestation metadata against Supabase database (SOURCE OF TRUTH).
    
    Security Checks:
    1. Query contributor_attestations table for wallet's attestation record
    2. Reject if no valid attestation exists (prevents local file manipulation)
    3. Verify lead metadata matches Supabase attestation record
    4. Validate terms version and boolean attestations
    
    This is Stage -1 (runs BEFORE all other checks) to ensure regulatory compliance.
    """
    from Leadpoet.utils.contributor_terms import TERMS_VERSION_HASH
    from Leadpoet.utils.cloud_db import get_supabase_client
    
    # Check required attestation fields in lead
    required_fields = ["wallet_ss58", "terms_version_hash", "lawful_collection", 
                      "no_restricted_sources", "license_granted"]
    
    missing = [f for f in required_fields if f not in lead]
    if missing:
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Missing attestation fields: {', '.join(missing)}",
            "failed_fields": missing
        }
    
    wallet_ss58 = lead.get("wallet_ss58")
    lead_terms_hash = lead.get("terms_version_hash")
    
    # SECURITY CHECK 1: Query Supabase for authoritative attestation record
    try:
        supabase = get_supabase_client()
        if not supabase:
            # If Supabase not available, log warning but don't fail validation
            # This prevents breaking validators during network issues
            print(f"   ⚠️  Supabase client not available - skipping attestation verification")
            return True, {}
        
        result = supabase.table("contributor_attestations")\
            .select("*")\
            .eq("wallet_ss58", wallet_ss58)\
            .eq("terms_version_hash", TERMS_VERSION_HASH)\
            .eq("accepted", True)\
            .execute()
        
        # SECURITY CHECK 2: Reject if no valid attestation in database
        if not result.data or len(result.data) == 0:
            return False, {
                "stage": "Stage -1: Terms Attestation",
                "check_name": "check_terms_attestation",
                "message": f"No valid attestation found in database for wallet {wallet_ss58[:10]}...",
                "failed_fields": ["wallet_ss58"]
            }
        
        # Attestation exists in Supabase - miner has legitimately accepted terms
        supabase_attestation = result.data[0]
        
    except Exception as e:
        # Log error but don't fail validation - prevents breaking validators
        print(f"   ⚠️  Failed to verify attestation in database: {str(e)}")
        return True, {}
    
    # SECURITY CHECK 3: Verify lead metadata matches Supabase record
    if lead_terms_hash != supabase_attestation.get("terms_version_hash"):
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Lead attestation hash mismatch (lead: {lead_terms_hash[:8]}, db: {supabase_attestation.get('terms_version_hash', '')[:8]})",
            "failed_fields": ["terms_version_hash"]
        }
    
    # Check: Verify terms version is current
    if lead_terms_hash != TERMS_VERSION_HASH:
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": f"Outdated terms version (lead: {lead_terms_hash[:8]}, current: {TERMS_VERSION_HASH[:8]})",
            "failed_fields": ["terms_version_hash"]
        }
    
    # Check: Verify boolean attestations in lead
    if not all([lead.get("lawful_collection"), 
                lead.get("no_restricted_sources"), 
                lead.get("license_granted")]):
        return False, {
            "stage": "Stage -1: Terms Attestation",
            "check_name": "check_terms_attestation",
            "message": "Incomplete attestations",
            "failed_fields": ["lawful_collection", "no_restricted_sources", "license_granted"]
        }
    
    return True, {}


async def check_source_provenance(lead: dict) -> Tuple[bool, dict]:
    """
    Verify source provenance metadata.
    
    Validates:
    - source_url is present and valid
    - source_type is in allowed list
    - Domain not in restricted sources denylist
    - Domain age ≥ 7 days (reuses existing check)
    
    This ensures miners are providing valid source information and not using
    prohibited data brokers without proper authorization.
    """
    from Leadpoet.utils.source_provenance import (
        validate_source_url,
        is_restricted_source,
        extract_domain_from_url
    )
    
    # Check required fields
    source_url = lead.get("source_url")
    source_type = lead.get("source_type")
    
    if not source_url:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": "Missing source_url",
            "failed_fields": ["source_url"]
        }
    
    if not source_type:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": "Missing source_type",
            "failed_fields": ["source_type"]
        }
    
    # Validate source_type against allowed list
    valid_types = ["public_registry", "company_site", "first_party_form", 
                   "licensed_resale", "proprietary_database"]
    if source_type not in valid_types:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": f"Invalid source_type: {source_type}",
            "failed_fields": ["source_type"]
        }
    
    # Validate source URL (checks denylist, domain age, reachability)
    # SECURITY: Pass source_type to prevent spoofing proprietary_database
    try:
        is_valid, reason = await validate_source_url(source_url, source_type)
        if not is_valid:
            return False, {
                "stage": "Stage 0.5: Source Provenance",
                "check_name": "check_source_provenance",
                "message": f"Source URL validation failed: {reason}",
                "failed_fields": ["source_url"]
            }
    except Exception as e:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_source_provenance",
            "message": f"Error validating source URL: {str(e)}",
            "failed_fields": ["source_url"]
        }
    
    # Additional check: Extract domain and verify not restricted
    # (This is redundant with validate_source_url but provides explicit feedback)
    domain = extract_domain_from_url(source_url)
    if domain and is_restricted_source(domain):
        # Only fail if NOT a licensed resale (those are handled in next check)
        if source_type != "licensed_resale":
            return False, {
                "stage": "Stage 0.5: Source Provenance",
                "check_name": "check_source_provenance",
                "message": f"Source domain {domain} is in restricted denylist",
                "failed_fields": ["source_url"]
            }
    
    return True, {}


async def check_licensed_resale_proof(lead: dict) -> Tuple[bool, dict]:
    """
    Validate license document proof for licensed resale submissions.
    
    If source_type = "licensed_resale", validates that:
    - license_doc_hash is present
    - license_doc_hash is valid SHA-256 format
    
    This allows miners to use restricted data brokers (ZoomInfo, Apollo, etc.)
    IF they have a valid resale agreement and provide cryptographic proof.
    """
    from Leadpoet.utils.source_provenance import validate_licensed_resale
    
    source_type = lead.get("source_type")
    
    # Only validate if this is a licensed resale submission
    if source_type != "licensed_resale":
        return True, {}
    
    # Validate license proof
    is_valid, reason = validate_licensed_resale(lead)
    
    if not is_valid:
        return False, {
            "stage": "Stage 0.5: Source Provenance",
            "check_name": "check_licensed_resale_proof",
            "message": reason,
            "failed_fields": ["license_doc_hash"]
        }
    
    # Log for audit trail
    license_hash = lead.get("license_doc_hash", "")
    print(f"   📄 Licensed resale detected: hash={license_hash[:16]}...")
    
    return True, {}


# ============================================================================
# STAGE 5: UNIFIED VERIFICATION (Role, Region, Industry)
# ============================================================================
# Verifies role, region, and industry in ONE LLM call after Stage 4 passes
# Uses DDG search results + fuzzy matching + LLM verification
# 
# Flow:
# 1. DDG search for ROLE (name + company + linkedin)
# 2. DDG search for REGION (company headquarters)
# 3. DDG search for INDUSTRY (what company does)
# 4. Fuzzy pre-verification (deterministic matching)
# 5. LLM verification (only for fields that need it)
# 6. Early exit if role fails → skip region/industry
# 7. Early exit if region fails → skip industry
# ============================================================================

import time

# GeoPy geocoding cache
_geocode_cache: Dict[str, Dict] = {}
_last_geocode_time = 0

def _geocode_location(location: str) -> Optional[Dict]:
    """
    Geocode a location string using Nominatim (free OpenStreetMap).
    Returns dict with city, state, country, lat, lon or None if not found.
    Rate limited to 1 request/second per Nominatim policy.
    """
    global _last_geocode_time, _geocode_cache
    
    if not location or len(location.strip()) < 2:
        return None
    
    cache_key = location.lower().strip()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]
    
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
        elapsed = time.time() - _last_geocode_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_geocode_time = time.time()
        
        geolocator = Nominatim(user_agent="leadpoet_verifier", timeout=5)
        geo = geolocator.geocode(location, addressdetails=True)
        
        if geo and geo.raw:
            address = geo.raw.get("address", {})
            result = {
                "city": address.get("city") or address.get("town") or address.get("village") or address.get("municipality"),
                "state": address.get("state") or address.get("region") or address.get("province"),
                "country": address.get("country"),
                "country_code": address.get("country_code", "").upper(),
                "lat": geo.latitude,
                "lon": geo.longitude,
                "display": geo.address
            }
            _geocode_cache[cache_key] = result
            return result
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️ Geocoding failed for '{location}': {e}")
    
    _geocode_cache[cache_key] = None
    return None


def locations_match_geopy(claimed: str, extracted: str, max_distance_km: float = 50) -> Tuple[bool, str]:
    """
    Compare two locations using GeoPy for deterministic matching.
    Returns (match: bool, reason: str)
    """
    if not claimed or not extracted:
        return False, "Missing location data - needs LLM verification"
    
    if "UNKNOWN" in extracted.upper():
        return False, "Extracted location unknown - needs LLM verification"
    
    US_STATES_SET = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
        'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
        'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
        'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
        'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina',
        'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
        'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas',
        'utah', 'vermont', 'virginia', 'washington', 'west virginia',
        'wisconsin', 'wyoming', 'district of columbia'
    }
    US_STATE_ABBREVS = {
        'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id',
        'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms',
        'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok',
        'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv',
        'wi', 'wy', 'dc'
    }
    
    claimed_lower = claimed.lower()
    
    # Count distinct US states mentioned
    states_found = set()
    for state in US_STATES_SET:
        if state in claimed_lower:
            states_found.add(state)
    for abbrev in US_STATE_ABBREVS:
        if re.search(rf'\b{abbrev}\b', claimed_lower):
            states_found.add(abbrev)
    
    if len(states_found) > 2:
        return False, f"ANTI-GAMING: Multiple states detected in claimed region: {states_found}"
    
    geo_claimed = _geocode_location(claimed)
    geo_extracted = _geocode_location(extracted)
    
    if not geo_claimed or not geo_extracted:
        claimed_lower = claimed.lower().strip()
        extracted_lower = extracted.lower().strip()
        
        def extract_city(loc: str) -> str:
            loc = loc.lower().strip()
            parts = [p.strip() for p in loc.split(',')]
            if parts:
                first_part = parts[0]
                if re.match(r'^\d+\s+', first_part):
                    street_match = re.search(r'\d+\s+(\w+)\s+(?:pkwy|blvd|ave|st|rd|dr|way|ln|ct|hwy)', first_part, re.IGNORECASE)
                    if street_match:
                        return street_match.group(1)
                    if len(parts) > 1:
                        return parts[1].strip()
                return first_part
            return loc
        
        claimed_city = extract_city(claimed)
        extracted_city = extract_city(extracted)
        
        if claimed_city and extracted_city:
            if claimed_city == extracted_city:
                return True, f"City match: {claimed_city} (geocoding unavailable)"
            if claimed_city in extracted_city or extracted_city in claimed_city:
                return True, f"City containment match: {claimed_city} in {extracted_city}"
        
        if claimed_lower in extracted_lower or extracted_lower in claimed_lower:
            return True, "String match (geocoding unavailable)"
        
        claimed_words = set(claimed_lower.replace(',', ' ').split())
        extracted_words = set(extracted_lower.replace(',', ' ').split())
        filler = {'us', 'usa', 'uk', 'gb', 'ca', 'au', 'the', 'of', 'and', 'st', 'ave', 'blvd', 'rd', 'dr', 'suite', 'floor', 'unit', 'united', 'states', 'america'}
        claimed_words -= filler
        extracted_words -= filler
        
        common = claimed_words & extracted_words
        if common:
            return True, f"Location word match: {common}"
        
        for code in ["us", "usa", "uk", "gb", "ca", "au", "de", "fr", "in", "sg", "ch", "be", "nl"]:
            if code in claimed_lower and code in extracted_lower:
                return False, f"Same country ({code.upper()}) but no city match - needs LLM verification"
        
        return False, "Geocoding unavailable and no string match - needs LLM verification"
    
    def extract_city_name(loc: str) -> str:
        loc = loc.lower().strip()
        parts = [p.strip() for p in loc.split(',')]
        if parts:
            return parts[0]
        return loc
    
    claimed_city = extract_city_name(claimed)
    extracted_city = extract_city_name(extracted)
    
    if claimed_city and extracted_city and claimed_city == extracted_city:
        return True, f"Same city: {claimed_city}"
    
    same_country = geo_claimed.get("country_code") == geo_extracted.get("country_code")
    
    if not same_country:
        return False, f"Different countries: {geo_claimed.get('country')} vs {geo_extracted.get('country')}"
    
    if geo_claimed.get("state") and geo_extracted.get("state"):
        if geo_claimed["state"].lower() == geo_extracted["state"].lower():
            return True, f"Same state: {geo_claimed['state']}"
    
    if geo_claimed.get("lat") and geo_extracted.get("lat"):
        try:
            from geopy.distance import geodesic
            dist = geodesic(
                (geo_claimed["lat"], geo_claimed["lon"]),
                (geo_extracted["lat"], geo_extracted["lon"])
            ).kilometers
            
            if dist <= max_distance_km:
                return True, f"Nearby cities ({dist:.0f}km apart)"
            elif same_country:
                return True, f"Same country ({geo_claimed.get('country_code')}), different location (remote worker likely) - {dist:.0f}km apart"
            else:
                return False, f"Cities too far apart ({dist:.0f}km)"
        except Exception:
            pass
    
    return True, f"Same country: {geo_claimed.get('country')} (remote worker/multiple offices)"


# Stage 5 Role Matching Constants
C_SUITE_EXPANSIONS = {
    "ceo": "chief executive officer",
    "cto": "chief technology officer",
    "cfo": "chief financial officer",
    "coo": "chief operating officer",
    "cmo": "chief marketing officer",
    "cio": "chief information officer",
    "cpo": "chief product officer",
    "cso": "chief strategy officer",
    "cro": "chief revenue officer",
    "chro": "chief human resources officer",
    "cdo": "chief data officer",
    "cno": "chief nursing officer",
    "cao": "chief administrative officer",
}

ROLE_ABBREVIATIONS = {
    "vp": "vice president",
    "svp": "senior vice president",
    "evp": "executive vice president",
    "avp": "assistant vice president",
    "sr": "senior",
    "sr.": "senior",
    "jr": "junior",
    "jr.": "junior",
    "dir": "director",
    "dir.": "director",
    "mgr": "manager",
    "mgr.": "manager",
    "eng": "engineer",
    "eng.": "engineer",
    "exec": "executive",
    "md": "managing director",
    "gp": "general partner",
    "pm": "product manager",
}

ROLE_EQUIVALENCIES = {
    "founder": ["founder", "co-founder", "co founder", "cofounder", "founding member"],
    "owner": ["owner", "business owner", "franchise owner", "store owner", "agent owner", "owner operator"],
    "president": ["president", "pres", "pres."],
    "partner": ["partner", "managing partner", "general partner", "senior partner", "equity partner"],
    "board": ["board member", "board director", "director", "board of directors"],
    "chair": ["chairman", "chairwoman", "chair", "chairperson", "executive chair", "executive chairman"],
}


def extract_role_from_ddg_title(title: str, snippet: str = "") -> Optional[str]:
    """Extract job role from DDG LinkedIn search result title/snippet."""
    if not title:
        return None
    
    original_title = title
    
    # Check for job posting format FIRST: "Company hiring Role [in Location] | LinkedIn"
    # This handles titles like "Chick-fil-A, Inc. hiring Sr. Lead Technical Product Owner | LinkedIn"
    job_posting_match = re.search(r'hiring\s+(.+?)(?:\s+in\s+[\w\s,]+)?(?:\s*\||\s*$)', title, re.IGNORECASE)
    if job_posting_match:
        role = job_posting_match.group(1).strip()
        # Clean up trailing location info
        role = re.sub(r'\s+in\s+[\w\s,]+$', '', role, flags=re.IGNORECASE).strip()
        if len(role) > 2 and len(role) < 100:
            return role
    
    role_keywords = [
        "ceo", "cto", "cfo", "coo", "cmo", "cio", "cpo",
        "founder", "co-founder", "cofounder", "co founder",
        "president", "vice president", "vp",
        "director", "manager", "lead", "head",
        "engineer", "developer", "analyst",
        "owner", "partner", "principal",
        "executive", "officer", "chief",
        "consultant", "advisor", "specialist",
        "product owner", "staff", "senior", "sr.",
        "operations", "business operations", "specialist"
    ]
    
    first_segment = title.split('|')[0].strip()
    first_segment = re.sub(r'\s+-\s*LinkedIn.*$', '', first_segment, flags=re.IGNORECASE).strip()
    first_segment = re.sub(r'\s*\.\.\.\s*$', '', first_segment).strip()
    
    match = re.search(r'^([^-]+)-\s*(.+?)(?:\s+at\s+|\s*$)', first_segment, re.IGNORECASE)
    if match:
        role = match.group(2).strip()
        role = re.sub(r'\s+at\s*$', '', role, flags=re.IGNORECASE).strip()
        if len(role) > 2 and not role.startswith("http") and role.lower() not in ["linkedin", "..."]:
            return role
    
    role_at_patterns = re.findall(r'(\b(?:' + '|'.join(role_keywords) + r')[^|]*?)\s+at\s+\w', original_title, re.IGNORECASE)
    if role_at_patterns:
        role = role_at_patterns[0].strip()
        if len(role) > 2:
            return role
    
    for kw in role_keywords:
        match = re.search(rf'\b({kw}[^|,]*?)\s+at\s+', original_title, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if len(role) > 2:
                return role
    
    if snippet:
        snippet_clean = snippet.strip()
        snippet_patterns = [
            r'(?:is\s+(?:the\s+)?|serves?\s+as\s+(?:the\s+)?|works?\s+as\s+(?:the\s+)?)([^.]+?)\s+(?:at|of|for)\s+',
            r'(?:current|position|title|role)[:\s]+([^|.\n]+)',
            r'(?:works? as|serving as|currently)[:\s]+([^|.\n]+)',
        ]
        for pattern in snippet_patterns:
            match = re.search(pattern, snippet_clean, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if len(role) > 2 and len(role) < 100:
                    return role
        
        for kw in role_keywords:
            match = re.search(rf'\b({kw}(?:\s+(?:and|&)\s+\w+)?(?:\s+of|\s+at)?)', snippet_clean, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if len(role) > 2:
                    return role
    
    return None


def fuzzy_match_role(claimed_role: str, extracted_role: str) -> Tuple[bool, float, str]:
    """
    Fuzzy match two roles with STRICT rules to prevent false positives.
    Returns (is_match: bool, confidence: float, reason: str)
    """
    if not claimed_role or not extracted_role:
        return False, 0.0, "Missing role data"
    
    claimed_lower = claimed_role.lower().strip()
    extracted_lower = extracted_role.lower().strip()
    
    if claimed_lower == extracted_lower:
        return True, 1.0, "Exact match"
    
    def normalize(r: str) -> str:
        r = r.lower().strip()
        r = r.replace("&", " and ")
        r = r.replace(",", " ")
        r = r.replace("-", " ")
        r = r.replace("/", " ")
        r = re.sub(r'\s+', ' ', r).strip()
        return r
    
    norm_claimed = normalize(claimed_role)
    norm_extracted = normalize(extracted_role)
    
    if norm_claimed == norm_extracted:
        return True, 1.0, "Normalized exact match"
    
    if norm_extracted in norm_claimed:
        return True, 0.95, f"Extracted role contained in claimed: '{extracted_role}' in '{claimed_role}'"
    if norm_claimed in norm_extracted:
        return True, 0.95, f"Claimed role contained in extracted: '{claimed_role}' in '{extracted_role}'"
    
    def expand_abbreviations(r: str) -> str:
        r = normalize(r)
        for abbrev, full in C_SUITE_EXPANSIONS.items():
            r = re.sub(rf'\b{abbrev}\b', full, r)
        for abbrev, full in ROLE_ABBREVIATIONS.items():
            r = re.sub(rf'\b{re.escape(abbrev)}\b', full, r)
        return r
    
    exp_claimed = expand_abbreviations(claimed_role)
    exp_extracted = expand_abbreviations(extracted_role)
    
    if exp_claimed == exp_extracted:
        return True, 1.0, "Abbreviation expansion match"
    
    if exp_extracted in exp_claimed:
        return True, 0.95, f"Expanded extracted in claimed: CEO/CTO match"
    if exp_claimed in exp_extracted:
        return True, 0.95, f"Expanded claimed in extracted: CEO/CTO match"
    
    def get_c_suite_type(role: str) -> Optional[str]:
        role_lower = role.lower()
        for abbrev, full in C_SUITE_EXPANSIONS.items():
            if re.search(rf'\b{abbrev}\b', role_lower) or full in role_lower:
                return abbrev
        return None
    
    claimed_csuite = get_c_suite_type(claimed_role)
    extracted_csuite = get_c_suite_type(extracted_role)
    
    if claimed_csuite and extracted_csuite:
        if claimed_csuite != extracted_csuite:
            return False, 0.0, f"C-Suite MISMATCH: {claimed_csuite.upper()} ≠ {extracted_csuite.upper()}"
    
    def is_business_owner(r: str) -> bool:
        r_lower = r.lower()
        return "owner" in r_lower and "product owner" not in r_lower and "product" not in r_lower.split("owner")[0]
    
    def is_product_owner(r: str) -> bool:
        return "product owner" in r.lower()
    
    if is_business_owner(claimed_role) and is_product_owner(extracted_role):
        return False, 0.0, "MISMATCH: Owner (business) ≠ Product Owner (tech role)"
    if is_product_owner(claimed_role) and is_business_owner(extracted_role):
        return False, 0.0, "MISMATCH: Product Owner (tech role) ≠ Owner (business)"
    
    departments = [
        "sales", "marketing", "engineering", "finance", "operations",
        "product", "hr", "human resources", "legal", "it", "technology",
        "customer", "business", "development", "research", "data"
    ]
    
    def get_department(r: str) -> Optional[str]:
        r_lower = r.lower()
        for dept in departments:
            if dept in r_lower:
                return dept
        return None
    
    claimed_dept = get_department(claimed_role)
    extracted_dept = get_department(extracted_role)
    
    if claimed_dept and extracted_dept and claimed_dept != extracted_dept:
        return False, 0.0, f"DEPARTMENT MISMATCH: {claimed_dept} ≠ {extracted_dept}"
    
    def has_founder(r: str) -> bool:
        r_lower = r.lower()
        return any(f in r_lower for f in ["founder", "co-founder", "cofounder", "co founder"])
    
    if has_founder(claimed_role) and has_founder(extracted_role):
        if claimed_csuite and extracted_csuite and claimed_csuite == extracted_csuite:
            return True, 1.0, "Founder + matching C-suite"
        elif not claimed_csuite and not extracted_csuite:
            return True, 0.95, "Both are founders"
        return True, 0.85, "Founder match (one has additional C-suite role)"
    
    if is_business_owner(claimed_role) and is_business_owner(extracted_role):
        return True, 0.95, "Both are business owners"
    
    if is_product_owner(claimed_role) and is_product_owner(extracted_role):
        return True, 0.95, "Both are Product Owners"
    
    def strip_common_modifiers(r: str) -> str:
        r = normalize(r)
        modifiers = [
            "assurance", "technical", "business", "global", "regional",
            "corporate", "digital", "strategic", "commercial", "associate",
            "assistant", "staff", "lead", "principal"
        ]
        words = r.split()
        core_words = [w for w in words if w not in modifiers]
        return " ".join(core_words)
    
    stripped_claimed = strip_common_modifiers(claimed_role)
    stripped_extracted = strip_common_modifiers(extracted_role)
    
    if stripped_claimed and stripped_extracted:
        if stripped_claimed == stripped_extracted:
            return True, 0.9, f"Core role match after stripping modifiers: '{stripped_claimed}'"
        if stripped_claimed in stripped_extracted or stripped_extracted in stripped_claimed:
            return True, 0.85, f"Core role containment: '{stripped_claimed}' ~ '{stripped_extracted}'"
    
    if exp_claimed in exp_extracted:
        return True, 0.9, f"Claimed role contained in extracted: '{claimed_role}' in '{extracted_role}'"
    if exp_extracted in exp_claimed:
        return True, 0.9, f"Extracted role contained in claimed: '{extracted_role}' in '{claimed_role}'"
    
    def get_meaningful_words(r: str) -> set:
        r = normalize(r)
        r = expand_abbreviations(r)
        words = set(r.split())
        filler = {"at", "of", "the", "and", "for", "in", "a", "an", "to", "&", "or"}
        return words - filler
    
    claimed_words = get_meaningful_words(claimed_role)
    extracted_words = get_meaningful_words(extracted_role)
    
    if claimed_words and extracted_words:
        intersection = claimed_words & extracted_words
        union = claimed_words | extracted_words
        jaccard = len(intersection) / len(union) if union else 0
        
        if jaccard >= 0.6:
            return True, jaccard, f"Word overlap: {jaccard:.0%} - common words: {intersection}"
    
    def expand_with_equivalencies(words: set) -> set:
        expanded = set(words)
        for word in list(words):
            for equiv_key, equiv_list in ROLE_EQUIVALENCIES.items():
                if word in equiv_list or word == equiv_key:
                    expanded.update(equiv_list)
                    expanded.add(equiv_key)
        return expanded
    
    exp_claimed_words = expand_with_equivalencies(claimed_words)
    exp_extracted_words = expand_with_equivalencies(extracted_words)
    
    equiv_intersection = exp_claimed_words & exp_extracted_words
    if len(equiv_intersection) >= 2:
        return True, 0.8, f"Equivalency match: {equiv_intersection}"
    
    jaccard = len(claimed_words & extracted_words) / len(claimed_words | extracted_words) if (claimed_words | extracted_words) else 0
    return False, jaccard, f"No match (word similarity: {jaccard:.0%})"


# Location patterns for extraction
LOCATION_PATTERNS = [
    r'headquarter(?:ed|s)?\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    r'based\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    r'located\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\s*[-–—]',
    r'\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,})',
]

def extract_location_from_text(text: str) -> Optional[str]:
    """Extract location from text using regex patterns."""
    if not text:
        return None
    
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            location = re.sub(r'\s*\|.*$', '', location)
            location = re.sub(r'\s*-.*$', '', location)
            return location
    
    return None


def fuzzy_pre_verification_stage5(
    claimed_role: str,
    claimed_region: str,
    claimed_industry: str,
    ddg_role_results: List[Dict],
    ddg_region_results: List[Dict],
    ddg_industry_results: List[Dict],
    full_name: str = "",
    company: str = "",
    role_only: bool = False
) -> Dict:
    """
    Pre-verify ROLE and REGION using fuzzy matching BEFORE sending to LLM.
    INDUSTRY is ALWAYS sent to LLM.
    
    Args:
        role_only: If True, only check role and suppress region/industry messages.
                   Used for early exit check before region/industry DDG searches.
    """
    result = {
        "role_verified": False,
        "role_extracted": None,
        "role_confidence": 0.0,
        "role_reason": "Not checked",
        "role_definitive_fail": False,
        
        "region_verified": False,
        "region_extracted": None,
        "region_confidence": 0.0,
        "region_reason": "Not checked",
        "region_hard_fail": False,
        
        "industry_verified": False,
        "industry_extracted": None,
        "industry_confidence": 0.0,
        "industry_reason": "Industry always verified by LLM (too subjective for fuzzy match)",
        
        "needs_llm": ["industry"],
    }
    
    # ROLE FUZZY MATCHING
    if ddg_role_results and claimed_role:
        name_lower = full_name.lower() if full_name else ""
        first_name = name_lower.split()[0] if name_lower else ""
        last_name = name_lower.split()[-1] if name_lower else ""
        
        best_extracted_role = None
        best_match = False
        best_confidence = 0.0
        best_reason = "No role found in DDG results"
        
        # Look at up to 15 results to include fallback results (5 primary + 5 fallback1 + 5 fallback2)
        for r in ddg_role_results[:15]:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            
            title_lower = title.lower()
            
            # Check if title contains the person's name OR is a job posting/company role listing
            # Job postings like "Company hiring Role" don't contain the person's name but prove the role exists
            is_job_posting = "hiring" in title_lower
            is_company_role = company and company.lower() in title_lower
            
            if first_name and last_name and not is_job_posting:
                first_pattern = rf'\b{re.escape(first_name)}\b'
                last_pattern = rf'\b{re.escape(last_name)}\b'
                has_name = re.search(first_pattern, title_lower) and re.search(last_pattern, title_lower)
                # Allow if: has person's name, OR is a job posting, OR is a company role listing with role keywords
                if not has_name and not is_company_role:
                    continue
            
            extracted = extract_role_from_ddg_title(title, snippet)
            
            if extracted:
                extracted_lower = extracted.lower()
                
                # Pre-compute: Check for strong role keywords (used in multiple filters)
                role_keywords_quick = ["ceo", "cto", "cfo", "coo", "founder", "president", "director", 
                                       "manager", "head", "lead", "vp", "chief", "officer"]
                has_strong_role_keyword = any(kw in extracted_lower for kw in role_keywords_quick)
                
                # Filter 1: Known invalid site names and domains
                invalid_extractions = ["wikipedia", "linkedin", "facebook", "twitter", "crunchbase", 
                                       "glassdoor", "indeed", "zoominfo", "bloomberg", "forbes", "reuters",
                                       "craft.co", "theorg.com", "the org"]
                if extracted_lower in invalid_extractions:
                    continue
                # Filter website domains (anything ending in .com, .co, .io, etc.)
                if re.match(r'^[\w\-]+\.(com|co|io|org|net)$', extracted_lower):
                    continue
                
                # Filter 1b: Too short/generic extractions
                too_short_generic = ["lead", "head", "manager", "director", "partner", "officer", 
                                    "engineer", "analyst", "the org", "the company", "org", "inc", "llc"]
                if extracted_lower in too_short_generic:
                    continue
                if len(extracted) < 5:
                    continue
                
                # Filter 1c: Truncated/garbage extractions
                if "..." in extracted or extracted_lower.endswith("- linkedin") or extracted_lower.endswith("| linkedin"):
                    continue
                
                # Filter 2: Garbage patterns that contain role keywords but aren't roles
                # Be specific to avoid false positives (e.g., "email example" in titles shouldn't block)
                garbage_patterns = [
                    "work history", "executive bio", "company profile", "contact info",
                    "phone number", "email address", "company overview", "about us",
                    "company headquarters", "company website", "biography of"
                ]
                # Only filter if these patterns appear AND no strong role keyword exists
                has_garbage = any(pattern in extracted_lower for pattern in garbage_patterns)
                if has_garbage and not has_strong_role_keyword:
                    continue
                
                # Filter 3: Location patterns (US states, countries, cities)
                # If extraction contains US state names or common location words, skip
                location_indicators = [
                    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
                    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
                    "maine", "maryland", "massachusetts", "michigan", "minnesota",
                    "mississippi", "missouri", "montana", "nebraska", "nevada",
                    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
                    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
                    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
                    "utah", "vermont", "virginia", "washington", "west virginia",
                    "wisconsin", "wyoming", "united states", "united kingdom", "canada",
                    "australia", "germany", "france", "spain", "italy", "netherlands"
                ]
                is_location = any(loc in extracted_lower for loc in location_indicators)
                # Only skip if it ONLY looks like a location (no role keywords)
                if is_location and not has_strong_role_keyword:
                    continue
                
                # Filter 4: Too long to be a job title (likely garbage)
                # But allow longer strings if they contain clear role indicators
                # (Yahoo sometimes concatenates multiple results which still contain valid roles)
                # Note: Yahoo can produce VERY long concatenated titles (300+ chars) but still contain valid roles
                if len(extracted) > 500:
                    continue
                if len(extracted) > 100 and not has_strong_role_keyword:
                    continue
                
                # Filter 5: Company name check (stricter)
                if company:
                    company_lower = company.lower()
                    
                    role_keywords = ["ceo", "cto", "cfo", "coo", "cio", "chief", "president", "director", 
                                     "manager", "founder", "owner", "partner", "head", "lead", "vp", 
                                     "vice", "executive", "officer", "analyst", "engineer", "developer"]
                    
                    has_role_keyword = any(kw in extracted_lower for kw in role_keywords)
                    
                    if not has_role_keyword:
                        # Exact match or company in extraction or extraction in company
                        if extracted_lower == company_lower:
                            continue
                        if company_lower in extracted_lower:
                            continue
                        if extracted_lower in company_lower:  # e.g., "Ori" for company "Ori Living"
                            continue
                
                # Filter 6: Full name check
                if full_name and extracted_lower == full_name.lower():
                    continue
                
                is_match, confidence, reason = fuzzy_match_role(claimed_role, extracted)
                
                if confidence > best_confidence or (not best_extracted_role and extracted):
                    best_extracted_role = extracted
                    best_match = is_match
                    best_confidence = confidence
                    best_reason = reason
        
        if best_extracted_role:
            result["role_extracted"] = best_extracted_role
            result["role_confidence"] = best_confidence
            result["role_reason"] = best_reason
            
            if best_match and best_confidence >= 0.8:
                result["role_verified"] = True
                print(f"   ✅ FUZZY ROLE MATCH: '{claimed_role}' ≈ '{best_extracted_role}'")
                print(f"      Confidence: {best_confidence:.0%} | Reason: {best_reason}")
            else:
                if best_match:
                    result["needs_llm"].append("role")
                    print(f"   ⚠️ FUZZY ROLE: Low confidence match ({best_confidence:.0%}), sending to LLM")
                else:
                    role_keywords = ["ceo", "cto", "cfo", "coo", "cio", "chief", "president", "director", 
                                     "manager", "founder", "owner", "partner", "head", "lead", "vp", 
                                     "vice", "executive", "officer", "analyst", "engineer", "developer",
                                     "consultant", "specialist", "coordinator", "supervisor", "administrator"]
                    
                    extracted_lower = best_extracted_role.lower()
                    looks_like_role = any(kw in extracted_lower for kw in role_keywords)
                    
                    if looks_like_role:
                        result["role_definitive_fail"] = True
                        print(f"   ❌ FUZZY ROLE DEFINITIVE MISMATCH: '{claimed_role}' ≠ '{best_extracted_role}'")
                        print(f"      Reason: {best_reason}")
                        print(f"      This is a HARD FAIL - will skip region/industry checks")
                    else:
                        result["needs_llm"].append("role")
                        print(f"   ⚠️ FUZZY ROLE: Extracted '{best_extracted_role}' doesn't look like a role, sending to LLM")
                        print(f"      Reason: {best_reason}")
        else:
            result["needs_llm"].append("role")
            result["role_reason"] = "Could not extract role from DDG results"
            print(f"   ⚠️ FUZZY ROLE: Could not extract role from DDG, sending to LLM")
    else:
        result["needs_llm"].append("role")
        print(f"   ⚠️ FUZZY ROLE: No DDG results or no claimed role")
    
    # REGION ANTI-GAMING CHECK (runs even in role_only mode for early exit)
    if claimed_region:
        US_STATES_SET = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina',
            'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
            'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas',
            'utah', 'vermont', 'virginia', 'washington', 'west virginia',
            'wisconsin', 'wyoming', 'district of columbia'
        }
        
        claimed_lower = claimed_region.lower()
        states_found = set()
        for state in US_STATES_SET:
            if state in claimed_lower:
                states_found.add(state)
        
        if len(states_found) >= 2:
            result["region_verified"] = False
            result["region_hard_fail"] = True
            result["region_confidence"] = 0.0
            result["region_reason"] = f"HARD FAIL: Multiple US states in claimed region: {states_found}"
            result["region_extracted"] = "REJECTED - multiple states detected"
            print(f"   ❌ ANTI-GAMING HARD FAIL: Multiple states detected in region: {states_found}")
            print(f"      Claimed region contains {len(states_found)} different US states - HARD FAIL")
            print(f"      This lead will FAIL regardless of LLM verification")
            ddg_region_results = None
    
    # If role_only mode, skip region DDG-based matching and industry checks
    # (Anti-gaming check above still runs for early exit detection)
    if role_only:
        return result
    
    # REGION FUZZY MATCHING
    if ddg_region_results and claimed_region:
        company_lower = company.lower() if company else ""
        extracted_region = None
        
        for r in ddg_region_results[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            combined = title + " " + snippet
            
            if company_lower and company_lower not in combined.lower():
                continue
            
            loc = extract_location_from_text(combined)
            if loc:
                extracted_region = loc
                break
        
        if extracted_region:
            geo_match, geo_reason = locations_match_geopy(claimed_region, extracted_region)
            
            result["region_extracted"] = extracted_region
            result["region_confidence"] = 0.95 if geo_match else 0.3
            result["region_reason"] = geo_reason
            
            if geo_match:
                result["region_verified"] = True
                print(f"   ✅ FUZZY REGION MATCH: '{claimed_region}' ≈ '{extracted_region}'")
                print(f"      Reason: {geo_reason}")
            else:
                if not result.get("region_hard_fail"):
                    result["needs_llm"].append("region")
                    print(f"   ⚠️ FUZZY REGION: GeoPy says no match, sending to LLM for verification")
                    print(f"      Claimed: {claimed_region} | Extracted: {extracted_region}")
        else:
            if not result.get("region_hard_fail"):
                result["needs_llm"].append("region")
                result["region_reason"] = "Could not extract region from DDG results"
                print(f"   ⚠️ FUZZY REGION: Could not extract location, sending to LLM")
    else:
        if not result.get("region_hard_fail"):
            result["needs_llm"].append("region")
            print(f"   ⚠️ FUZZY REGION: No DDG results or no claimed region")
    
    print(f"   🤖 INDUSTRY: Always verified by LLM (too subjective for fuzzy match)")
    
    return result


def _ddg_search_stage5_sync(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """Stage 5 DDG search helper with backend fallback."""
    
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    
    if search_type == "role":
        linkedin_url = kwargs.get("linkedin_url", "")
        
        linkedin_url_query = None
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0]
            linkedin_url_query = f'linkedin.com/in/{profile_slug}'
        
        role_simplified = re.split(r'[,&/]', role)[0].strip() if role else ""
        
        queries = [f'"{full_name}" {company} linkedin']
        fallback_queries = [
            f'"{full_name}" {role_simplified} {company}',  # Name + Role + Company
            f'{company} "{role_simplified}" linkedin',      # Company + Role (job postings, other employees)
        ]
        if linkedin_url_query:
            fallback_queries.append(linkedin_url_query)
    elif search_type == "region":
        region_hint = kwargs.get("region_hint", "")
        if region_hint:
            queries = [f'{company} {region_hint} headquarters location', f'{company} headquarters {region_hint}']
        else:
            queries = [f'{company} headquarters location']
        fallback_queries = []
    else:  # industry
        region_hint = kwargs.get("region_hint", "")
        if region_hint:
            queries = [f'{company} {region_hint} company industry', f'{company} company industry {region_hint}']
        else:
            queries = [f'{company} company industry']
        fallback_queries = []
    
    def ddg_search_with_fallback(ddgs_instance, query, max_results, backends=['yahoo', 'yandex', 'google']):
        last_error = None
        for backend in backends:
            try:
                return list(ddgs_instance.text(query, max_results=max_results, backend=backend))
            except Exception as e:
                last_error = e
                continue
        raise last_error if last_error else Exception("All backends failed")
    
    all_results = []
    try:
        with DDGS() as ddgs:
            if search_type == "role":
                print(f"   🔍 PRIMARY: Searching '{full_name}' + '{company}' + linkedin...")
            
            for i, query in enumerate(queries):
                if i > 0:
                    time.sleep(2)
                    
                try:
                    results = ddg_search_with_fallback(ddgs, query, max_results)
                    for r in results:
                        all_results.append({
                            "title": r.get("title", ""),
                            "link": r.get("href", ""),
                            "snippet": r.get("body", ""),
                            "query_type": search_type,
                            "query": query
                        })
                    if len(all_results) >= max_results:
                        break
                except Exception as e:
                    print(f"   ⚠️ DDG {search_type} query failed: {query[:40]}... ({e})")
                    continue
            
            if search_type == "role" and fallback_queries:
                role_found = False
                for r in all_results[:5]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", r.get("body", ""))
                    extracted = extract_role_from_ddg_title(title, snippet)
                    if not extracted:
                        continue
                    
                    extracted_lower = extracted.lower()
                    
                    # Filter: Invalid site names and domains (exact match or pattern)
                    invalid_extractions = ["linkedin", "wikipedia", "facebook", "twitter", "crunchbase", 
                                          "glassdoor", "indeed", "zoominfo", "craft.co", "theorg.com", 
                                          "the org", "bloomberg", "forbes", "reuters"]
                    if extracted_lower in invalid_extractions:
                        continue
                    # Filter website domains (anything ending in .com, .co, .io, etc.)
                    if re.match(r'^[\w\-]+\.(com|co|io|org|net)$', extracted_lower):
                        continue
                    
                    # Filter: Check for role keywords
                    role_keywords = ["ceo", "cto", "cfo", "coo", "founder", "president", "director", 
                                     "manager", "head", "lead", "vp", "chief", "officer", "owner", "partner"]
                    has_role_keyword = any(kw in extracted_lower for kw in role_keywords)
                    
                    # Filter: Too short/generic extractions (just "Lead", "Head" without context)
                    too_short_generic = ["lead", "head", "manager", "director", "partner", "officer", 
                                        "engineer", "analyst", "the org", "the company", "org", "inc", "llc"]
                    if extracted_lower in too_short_generic:
                        continue
                    
                    # Filter: Very short extractions (less than 5 chars) 
                    if len(extracted) < 5:
                        continue
                    
                    # Filter: Truncated/garbage extractions (contain "..." or end with "- LinkedIn")
                    if "..." in extracted or extracted_lower.endswith("- linkedin") or extracted_lower.endswith("| linkedin"):
                        continue
                    
                    # Filter: Location patterns (US states, countries) without role keyword
                    location_indicators = [
                        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
                        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
                        "maine", "maryland", "massachusetts", "michigan", "minnesota",
                        "mississippi", "missouri", "montana", "nebraska", "nevada",
                        "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
                        "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
                        "rhode island", "south carolina", "south dakota", "tennessee", "texas",
                        "utah", "vermont", "virginia", "washington", "west virginia",
                        "wisconsin", "wyoming", "united states", "united kingdom"
                    ]
                    is_location = any(loc in extracted_lower for loc in location_indicators)
                    if is_location and not has_role_keyword:
                        continue
                    
                    # Filter: Company name (exact or contained) without role keyword
                    if not has_role_keyword:
                        if company:
                            company_lower = company.lower()
                            # Exact match or company contained in extraction
                            if extracted_lower == company_lower or company_lower in extracted_lower:
                                continue
                            # Extraction contained in company (e.g., "Ori" for "Ori Living")
                            if extracted_lower in company_lower:
                                continue
                        if full_name and extracted_lower == full_name.lower():
                            continue
                    
                    role_found = True
                    print(f"   ✅ Found role in results: {extracted[:50]}")
                    break
                
                if not role_found:
                    print(f"   ⚠️ No role in primary results, trying fallbacks...")
                    time.sleep(3)
                    
                    # Run ALL fallbacks to collect all possible role sources
                    for j, query in enumerate(fallback_queries):
                        query_num = j + 2
                        print(f"   🔍 FALLBACK{query_num-1}: {query[:50]}...")
                        try:
                            results = ddg_search_with_fallback(ddgs, query, max_results)
                            for r in results:
                                all_results.append({
                                    "title": r.get("title", ""),
                                    "link": r.get("href", ""),
                                    "snippet": r.get("body", ""),
                                    "query_type": search_type,
                                    "query": query + f" (fallback{query_num-1})"
                                })
                            
                            for r in results:
                                title = r.get("title", "")
                                snippet = r.get("body", "")
                                extracted = extract_role_from_ddg_title(title, snippet)
                                if not extracted:
                                    continue
                                    
                                extracted_lower = extracted.lower()
                                
                                # Same filtering as primary
                                role_keywords = ["ceo", "cto", "cfo", "coo", "founder", "president", "director", 
                                                 "manager", "head", "lead", "vp", "chief", "officer", "owner", "partner"]
                                has_role_keyword = any(kw in extracted_lower for kw in role_keywords)
                                
                                # Filter: Invalid site names and domains
                                invalid_extractions = ["linkedin", "wikipedia", "facebook", "twitter", "crunchbase", 
                                                      "glassdoor", "indeed", "zoominfo", "craft.co", "theorg.com", 
                                                      "the org", "bloomberg", "forbes", "reuters"]
                                if extracted_lower in invalid_extractions:
                                    continue
                                # Filter website domains (anything ending in .com, .co, .io, etc.)
                                if re.match(r'^[\w\-]+\.(com|co|io|org|net)$', extracted_lower):
                                    continue
                                
                                # Filter: Too short/generic 
                                too_short_generic = ["lead", "head", "manager", "director", "partner", "officer", 
                                                    "engineer", "analyst", "the org", "the company", "org", "inc", "llc"]
                                if extracted_lower in too_short_generic:
                                    continue
                                if len(extracted) < 5:
                                    continue
                                
                                # Filter: Truncated/garbage
                                if "..." in extracted or extracted_lower.endswith("- linkedin") or extracted_lower.endswith("| linkedin"):
                                    continue
                                
                                # Filter: Location patterns
                                location_indicators = [
                                    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                                    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
                                    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
                                    "maine", "maryland", "massachusetts", "michigan", "minnesota",
                                    "mississippi", "missouri", "montana", "nebraska", "nevada",
                                    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
                                    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
                                    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
                                    "utah", "vermont", "virginia", "washington", "west virginia",
                                    "wisconsin", "wyoming", "united states", "united kingdom"
                                ]
                                is_location = any(loc in extracted_lower for loc in location_indicators)
                                if is_location and not has_role_keyword:
                                    continue
                                
                                if not has_role_keyword:
                                    if company:
                                        company_lower = company.lower()
                                        if extracted_lower == company_lower or company_lower in extracted_lower or extracted_lower in company_lower:
                                            continue
                                    if full_name and extracted_lower == full_name.lower():
                                        continue
                                
                                print(f"   ✅ Fallback{query_num-1} found role: {extracted[:50]}")
                                break  # Found a role in this fallback, move to next fallback
                            else:
                                pass  # No role found in this fallback's results
                            
                            time.sleep(2)  # Wait between fallbacks
                            
                        except Exception as e:
                            print(f"   ⚠️ Fallback query failed: {e}")
                            time.sleep(2)
                            continue
                            
    except Exception as e:
        print(f"⚠️ DDG {search_type} search failed: {e}")
    
    # Return all results including fallbacks (don't limit to max_results)
    # max_results only controls how many results per query, but we want all queries' results
    return all_results


async def _ddg_search_stage5(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """Async wrapper for Stage 5 DDG search."""
    try:
        return await asyncio.to_thread(
            _ddg_search_stage5_sync,
            search_type,
            full_name,
            company,
            role,
            max_results,
            **kwargs
        )
    except Exception as e:
        print(f"⚠️ DDG {search_type} search thread failed: {e}")
        return []


async def check_stage5_unified(lead: dict) -> Tuple[bool, dict]:
    """
    Stage 5: Unified verification of role, region, and industry.
    
    Uses DDG searches + fuzzy matching + LLM verification.
    Called AFTER Stage 4 LinkedIn verification passes.
    
    Returns:
        (passed: bool, rejection_reason: dict or None)
    """
    full_name = get_field(lead, "full_name") or ""
    company = get_company(lead) or ""
    claimed_role = get_role(lead) or ""
    claimed_region = get_location(lead) or ""
    claimed_industry = get_industry(lead) or ""
    linkedin_url = get_linkedin(lead) or ""
    website = get_website(lead) or ""
    
    if not company:
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": "No company name provided",
            "failed_fields": ["company"]
        }
    
    # Wait before DDG searches
    print(f"   ⏳ Waiting 3s before Stage 5 DDG searches...")
    await asyncio.sleep(3)
    
    # STEP 1: DDG SEARCH FOR ROLE
    print(f"   🔍 DDG: Searching for {full_name}'s role at {company}...")
    role_results = await _ddg_search_stage5("role", full_name, company, claimed_role, linkedin_url=linkedin_url)
    if role_results:
        print(f"   ✅ Found {len(role_results)} role search results")
    else:
        print(f"   ⚠️ No role results found")
    
    # EARLY EXIT CHECK: Do quick role + region anti-gaming check BEFORE region/industry DDG searches
    # This saves 6+ seconds and 2 DDG API calls when role is definitively wrong OR region is gaming
    print(f"   🔍 QUICK CHECK: Verifying role and region anti-gaming before continuing...")
    quick_result = fuzzy_pre_verification_stage5(
        claimed_role=claimed_role,
        claimed_region=claimed_region,  # Pass real region for anti-gaming check
        claimed_industry="",  # Skip industry check
        ddg_role_results=role_results,
        ddg_region_results=[],  # Empty - just checking anti-gaming on claimed_region string
        ddg_industry_results=[],  # Empty - not checking yet
        full_name=full_name,
        company=company,
        role_only=True  # Skip DDG-based region/industry matching, but anti-gaming still runs
    )
    
    # EARLY EXIT: Role definitively failed - skip region/industry DDG searches entirely
    if quick_result.get("role_definitive_fail"):
        print(f"   ❌ EARLY EXIT: Role check failed - SKIPPING region and industry DDG searches")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Role FAILED: Found '{quick_result.get('role_extracted')}' but miner claimed '{claimed_role}'",
            "failed_fields": ["role"],
            "early_exit": "role_failed_before_region_industry",
            "extracted_role": quick_result.get("role_extracted"),
            "claimed_role": claimed_role,
            "ddg_searches_skipped": ["region", "industry"]
        }
    
    # EARLY EXIT: Region anti-gaming (multiple states) - skip region/industry DDG searches
    if quick_result.get("region_hard_fail"):
        print(f"   ❌ EARLY EXIT: Region anti-gaming - SKIPPING region and industry DDG searches")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Region FAILED (anti-gaming): {quick_result.get('region_reason')}",
            "failed_fields": ["region"],
            "early_exit": "region_anti_gaming_before_ddg",
            "ddg_searches_skipped": ["region", "industry"]
        }
    
    print(f"   ⏳ Waiting 3s before region search...")
    await asyncio.sleep(3)
    
    # STEP 2: DDG SEARCH FOR REGION (only if role didn't definitively fail)
    print(f"   🔍 DDG: Searching for {company} headquarters location...")
    region_results = await _ddg_search_stage5("region", company=company, region_hint=claimed_region)
    if region_results:
        print(f"   ✅ Found {len(region_results)} region search results")
    else:
        print(f"   ⚠️ No region results found")
    
    # Note: Region anti-gaming check already done in quick_result above (before region DDG)
    # No need to check again here
    
    print(f"   ⏳ Waiting 3s before industry search...")
    await asyncio.sleep(3)
    
    # STEP 3: DDG SEARCH FOR INDUSTRY
    print(f"   🔍 DDG: Searching for {company} industry...")
    industry_results = await _ddg_search_stage5("industry", company=company, region_hint=claimed_region)
    if industry_results:
        print(f"   ✅ Found {len(industry_results)} industry search results")
    else:
        print(f"   ⚠️ No industry results found")
    
    # STEP 4: FULL FUZZY PRE-VERIFICATION (now with all results)
    print(f"   🔍 FUZZY: Full pre-verification before LLM...")
    
    fuzzy_result = fuzzy_pre_verification_stage5(
        claimed_role=claimed_role,
        claimed_region=claimed_region,
        claimed_industry=claimed_industry,
        ddg_role_results=role_results,
        ddg_region_results=region_results,
        ddg_industry_results=industry_results,
        full_name=full_name,
        company=company
    )
    
    # Note: role_definitive_fail already checked above (before region/industry DDG)
    # so we only check region anti-gaming here
    
    # EARLY EXIT: Region anti-gaming AND role already verified
    if fuzzy_result.get("region_hard_fail") and fuzzy_result.get("role_verified"):
        print(f"   ❌ EARLY EXIT: Region anti-gaming triggered - skipping industry check")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Region FAILED (anti-gaming): {fuzzy_result.get('region_reason')}",
            "failed_fields": ["region"],
            "early_exit": "region_anti_gaming",
            "role_passed": True,
            "extracted_role": fuzzy_result.get("role_extracted")
        }
    
    # Check if all fields were fuzzy-matched
    if not fuzzy_result["needs_llm"]:
        print(f"   ✅ FUZZY: All fields matched - skipping LLM!")
        lead["stage5_role_match"] = True
        lead["stage5_region_match"] = True
        lead["stage5_industry_match"] = True
        lead["stage5_extracted_role"] = fuzzy_result["role_extracted"]
        lead["stage5_extracted_region"] = fuzzy_result["region_extracted"]
        return True, None
    
    # STEP 5: LLM VERIFICATION for remaining fields
    needs_llm = fuzzy_result["needs_llm"]
    print(f"   🤖 LLM: Need to verify: {needs_llm}")
    
    # Build context
    role_context = ""
    if "role" in needs_llm and role_results:
        role_context = f"ROLE SEARCH RESULTS (searched: '{full_name}' + '{company}' + '{claimed_role}'):\n"
        for i, result in enumerate(role_results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("body", ""))
            role_context += f"{i}. {title}\n   {snippet[:200]}\n"
    
    region_context = ""
    if "region" in needs_llm and region_results:
        region_context = "\nREGION/HEADQUARTERS SEARCH RESULTS:\n"
        for i, result in enumerate(region_results[:4], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            region_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    industry_context = ""
    if "industry" in needs_llm and industry_results:
        industry_context = "\nINDUSTRY SEARCH RESULTS:\n"
        for i, result in enumerate(industry_results[:4], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            industry_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    all_search_context = role_context + region_context + industry_context
    
    # AUTO-FAIL if role needs LLM but no context
    if "role" in needs_llm and not role_context.strip():
        print(f"   ❌ AUTO-FAIL: No DDG data for role verification")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": "No search results found to verify role",
            "failed_fields": ["role"]
        }
    
    if not all_search_context.strip():
        print(f"   ❌ AUTO-FAIL: No DDG search results at all")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": "No search results available. Cannot verify without data.",
            "failed_fields": ["role", "region", "industry"]
        }
    
    # Build LLM prompt
    claims_to_verify = []
    verification_rules = []
    
    if "role" in needs_llm:
        claims_to_verify.append(f'1. ROLE: "{claimed_role}"')
        verification_rules.append("""
1. ROLE VERIFICATION (Use ONLY the ROLE SEARCH RESULTS above):
   - CRITICAL: You must ONLY use the search results provided. Do NOT use prior knowledge!
   - Look for the role in: "Name - Role at Company | LinkedIn" format
   - Allow variations: "CEO" = "Chief Executive Officer", "Co-Founder & CEO" ≈ "CEO"
   - "Owner" matches "Founder", "Co-Founder", "Principal"
   - CRITICAL: "Owner" (business) ≠ "Product Owner" (tech role)
   - COO ≠ CIO ≠ CFO (C-suite roles are DIFFERENT)
   - If search results show the claimed role → role_match = true
   - If search results show a DIFFERENT role → role_match = false, extracted_role = actual role from results
   - If search results have NO role info (just company name) → role_match = false, extracted_role = "Not found"
   - NEVER guess or use training data! Only extract what's in the search results above.
""")
    else:
        claims_to_verify.append(f'1. ROLE: "{claimed_role}" ✅ (Already verified by fuzzy match)')
    
    if "region" in needs_llm:
        claims_to_verify.append(f'2. REGION: "{claimed_region}" (company HQ location)')
        verification_rules.append("""
2. REGION VERIFICATION:
   - Look for company headquarters in search results
   - PASS if city, state, OR country matches reasonably
   - "San Jose, CA" ≈ "San Jose, California" ✓
   - Same-state = match (e.g., Brooklyn, NY ≈ New York, NY)
   - If you cannot find HQ location → region_match=true, extracted_region="UNKNOWN"
   - FAIL only if CLEAR evidence of completely different country/state
""")
    else:
        claims_to_verify.append(f'2. REGION: "{claimed_region}" ✅ (Already verified by fuzzy match)')
    
    if "industry" in needs_llm:
        claims_to_verify.append(f'3. INDUSTRY: "{claimed_industry}"')
        verification_rules.append("""
3. INDUSTRY VERIFICATION:
   - Look for what the company does in search results
   - BE VERY LENIENT: Industries often overlap and can be categorized differently
   - "Technology" covers software, SaaS, IT, tech startups ✓
   - "Fintech" → "Financial Services" ✓
   - "Venture Capital" → "Financial Services" ✓
   - "Food & Beverages" → bars, restaurants, cocktails, hospitality ✓
   - "Hospitality" → restaurants, bars, hotels, food service ✓
   - "Retail" → stores, e-commerce, consumer products ✓
   - "Healthcare" → medical devices, biotech, pharma, health services ✓
   - ONLY FAIL if industries are COMPLETELY unrelated (e.g., "Aerospace" for a restaurant)
   - PASS if claimed industry is even loosely related to what the company does
   - If unknown company → industry_match=true, extracted_industry="UNKNOWN"
""")
    else:
        claims_to_verify.append(f'3. INDUSTRY: "{claimed_industry}" ✅ (Already verified by fuzzy match)')
    
    claims_section = "\n".join(claims_to_verify)
    rules_section = "\n".join(verification_rules)
    
    response_fields = []
    if "role" in needs_llm:
        response_fields.append('"role_match": true/false,\n    "extracted_role": "role found in search results"')
    if "region" in needs_llm:
        response_fields.append('"region_match": true/false,\n    "extracted_region": "company HQ from search"')
    if "industry" in needs_llm:
        response_fields.append('"industry_match": true/false,\n    "extracted_industry": "industry from search"')
    response_fields.append('"confidence": 0.0-1.0,\n    "reasoning": "Brief explanation"')
    
    response_format = ",\n    ".join(response_fields)
    
    prompt = f"""You are verifying B2B lead data quality. Verify the following claims using the SEARCH RESULTS provided.

LEAD INFORMATION:
- Name: {full_name}
- Company: {company}
- Website: {website}
- LinkedIn: {linkedin_url}

CLAIMS TO VERIFY:
{claims_section}

{all_search_context}

VERIFICATION RULES:
{rules_section}

RESPOND WITH JSON ONLY:
{{
    {response_format}
}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0
                },
                timeout=20
            ) as response:
                if response.status != 200:
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"LLM API error: HTTP {response.status}",
                        "failed_fields": ["llm_error"]
                    }
                
                data = await response.json()
                llm_response = data["choices"][0]["message"]["content"].strip()
                
                if llm_response.startswith("```"):
                    lines = llm_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    llm_response = "\n".join(lines).strip()
                
                result = json.loads(llm_response)
                
                # Determine final results
                if fuzzy_result["role_verified"]:
                    role_match = True
                    extracted_role = fuzzy_result["role_extracted"] or claimed_role
                else:
                    role_match = result.get("role_match", False)
                    extracted_role = result.get("extracted_role", "Not found")
                
                # EARLY EXIT: Role failed after LLM
                if not role_match:
                    print(f"   ❌ EARLY EXIT: Role check failed after LLM - skipping region/industry")
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"Role FAILED: LLM found '{extracted_role}' but miner claimed '{claimed_role}'",
                        "failed_fields": ["role"],
                        "early_exit": "role_llm_failed",
                        "extracted_role": extracted_role
                    }
                
                # Region
                if fuzzy_result.get("region_hard_fail"):
                    print(f"   ❌ REGION HARD FAIL: Anti-gaming check triggered")
                    print(f"   ❌ EARLY EXIT: Region anti-gaming failed - skipping industry")
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"Region FAILED (anti-gaming): Multiple states detected",
                        "failed_fields": ["region"],
                        "early_exit": "region_anti_gaming"
                    }
                elif fuzzy_result["region_verified"]:
                    region_match = True
                    extracted_region = fuzzy_result["region_extracted"] or claimed_region
                else:
                    region_match = result.get("region_match", False)
                    extracted_region = result.get("extracted_region", "")
                
                # EARLY EXIT: Region failed after LLM
                if not region_match:
                    print(f"   ❌ EARLY EXIT: Region check failed after LLM - skipping industry")
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"Region FAILED: LLM found '{extracted_region}' but miner claimed '{claimed_region}'",
                        "failed_fields": ["region"],
                        "early_exit": "region_llm_failed"
                    }
                
                # GeoPy verification for region
                geopy_reason = ""
                if not region_match and claimed_region and extracted_region:
                    geopy_match, geopy_reason = locations_match_geopy(claimed_region, extracted_region)
                    if geopy_match:
                        print(f"   🌍 GeoPy override: {geopy_reason}")
                        region_match = True
                
                # Industry
                if fuzzy_result["industry_verified"]:
                    industry_match = True
                    extracted_industry = fuzzy_result["industry_extracted"] or claimed_industry
                else:
                    industry_match = result.get("industry_match", False)
                    extracted_industry = result.get("extracted_industry", "")
                
                all_match = role_match and region_match and industry_match
                
                # Store results on lead
                lead["stage5_role_match"] = role_match
                lead["stage5_region_match"] = region_match
                lead["stage5_industry_match"] = industry_match
                lead["stage5_extracted_role"] = extracted_role
                lead["stage5_extracted_region"] = extracted_region
                lead["stage5_extracted_industry"] = extracted_industry
                
                if all_match:
                    return True, None
                else:
                    failed_fields = []
                    if not role_match:
                        failed_fields.append("role")
                    if not region_match:
                        failed_fields.append("region")
                    if not industry_match:
                        failed_fields.append("industry")
                    
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"Stage 5 verification failed for: {', '.join(failed_fields)}",
                        "failed_fields": failed_fields,
                        "role_match": role_match,
                        "region_match": region_match,
                        "industry_match": industry_match
                    }
                
    except Exception as e:
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Stage 5 verification failed: {str(e)}",
            "failed_fields": ["exception"]
        }


# Main validation pipeline

async def run_automated_checks(lead: dict) -> Tuple[bool, dict]:
    """
    Run all automated checks in stages, returning (passed, structured_data).

    Returns:
        Tuple[bool, dict]: (passed, structured_automated_checks_data)
            - If passed: (True, structured_data with stage_1_dns, stage_2_domain, stage_3_email)
            - If failed: (False, structured_data with rejection_reason and partial check data)
            
    Structured data format (tasks2.md Phase 1):
    {
        "stage_1_dns": {
            "has_mx": bool,
            "has_spf": bool,
            "has_dmarc": bool,
            "dmarc_policy": str
        },
        "stage_2_domain": {
            "dnsbl_checked": bool,
            "dnsbl_blacklisted": bool,
            "dnsbl_list": str,
            "domain_age_days": int,
            "domain_registrar": str,
            "domain_nameservers": list,
            "whois_updated_days_ago": int
        },
        "stage_3_email": {
            "email_status": str,  # "valid", "catch-all", "invalid", "unknown"
            "email_score": int,
            "is_disposable": bool,
            "is_role_based": bool,
            "is_free": bool
        },
        "passed": bool,
        "rejection_reason": dict or None
    }
    """

    email = get_email(lead)
    company = get_company(lead)
    
    # Initialize structured data collection
    automated_checks_data = {
        "stage_0_hardcoded": {
            "name_in_email": False,
            "is_general_purpose_email": False
        },
        "stage_1_dns": {
            "has_mx": False,
            "has_spf": False,
            "has_dmarc": False,
            "dmarc_policy": None
        },
        "stage_2_domain": {
            "dnsbl_checked": False,
            "dnsbl_blacklisted": False,
            "dnsbl_list": None,
            "domain_age_days": None,
            "domain_registrar": None,
            "domain_nameservers": None,
            "whois_updated_days_ago": None
        },
        "stage_3_email": {
            "email_status": "unknown",
            "email_score": 0,
            "is_disposable": False,
            "is_role_based": False,
            "is_free": False
        },
        "stage_4_linkedin": {
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        },
        "stage_5_verification": {  # NEW: Role/Region/Industry verification
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None  # "role_failed", "region_failed", or None
        },
        "rep_score": {
            "total_score": 0,
            "max_score": MAX_REP_SCORE,
            "breakdown": {
                "wayback_machine": 0,
                "uspto_trademarks": 0,
                "sec_edgar": 0,
                "whois_dnsbl": 0,
                "gdelt": 0,
                "companies_house": 0
            }
        },
        "passed": False,
        "rejection_reason": None
    }

    # ========================================================================
    # Pre-Attestation Check: REMOVED
    # ========================================================================
    # NOTE: Attestation verification removed from validators.
    # Validators don't have Supabase credentials and shouldn't verify attestations.
    # 
    # SECURITY: Gateway verifies attestations during POST /submit:
    # - If lead is in validator queue → gateway already verified attestation
    # - Validators trust gateway's verification (gateway is TEE-protected)
    # - This prevents security bypass where validator skips check due to 401 errors
    # 
    # If you need attestation verification, implement it in gateway/api/submit.py
    print(f"🔍 Pre-Attestation Check: Skipped (gateway verifies during submission)")

    # ========================================================================
    # Source Provenance Verification: Source Validation (HARD)
    # Validates source_url, source_type, denylist, and licensed resale proof
    # ========================================================================
    print(f"🔍 Source Provenance Verification: Source validation for {email} @ {company}")
    
    checks_stage0_5 = [
        check_source_provenance,       # Validate source URL, type, denylist
        check_licensed_resale_proof,   # Validate license hash if applicable
    ]
    
    for check_func in checks_stage0_5:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Source Provenance Verification failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data
    
    print("   ✅ Source Provenance Verification passed")

    # ========================================================================
    # Stage 0: Hardcoded Checks (MIXED)
    # - Required Fields, Email Regex, Name-Email Match, General Purpose Email, Disposable, HEAD Request
    # - Deduplication (handled in validate_lead_list)
    # ========================================================================
    print(f"🔍 Stage 0: Hardcoded checks for {email} @ {company}")
    checks_stage0 = [
        check_required_fields,      # Required fields validation (HARD)
        check_email_regex,          # RFC-5322 regex validation (HARD)
        check_name_email_match,     # Name in email check (HARD) - NEW
        check_general_purpose_email,# General purpose email filter (HARD) - NEW
        check_free_email_domain,    # Reject free email domains (HARD) - NEW
        check_disposable,           # Filter throwaway email providers (HARD)
        check_head_request,         # Test website accessibility (HARD)
    ]

    for check_func in checks_stage0:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 0 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    # Collect Stage 0 data after successful checks
    automated_checks_data["stage_0_hardcoded"]["name_in_email"] = True  # Passed name-email match
    automated_checks_data["stage_0_hardcoded"]["is_general_purpose_email"] = False  # Not general purpose

    print("   ✅ Stage 0 passed")

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
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 1 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            # Collect partial Stage 2 data (WHOIS)
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

    # Collect Stage 1 DNS data after successful checks
    automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", True)  # Passed MX check
    automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
    automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
    automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"

    print("   ✅ Stage 1 passed")

    # ========================================================================
    # Stage 2: Lightweight Domain Reputation Checks (HARD)
    # - DNSBL (Domain Block List) - Spamhaus DBL lookup
    # ========================================================================
    print(f"🔍 Stage 2: Domain reputation checks for {email} @ {company}")
    passed, rejection_reason = await check_dnsbl(lead)
    
    # Collect Stage 2 domain data (DNSBL + WHOIS from Stage 1)
    automated_checks_data["stage_2_domain"]["dnsbl_checked"] = lead.get("dnsbl_checked", False)
    automated_checks_data["stage_2_domain"]["dnsbl_blacklisted"] = lead.get("dnsbl_blacklisted", False)
    automated_checks_data["stage_2_domain"]["dnsbl_list"] = lead.get("dnsbl_list")
    automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
    automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
    automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
    automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
    
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 2 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 2 passed")

    # ========================================================================
    # Stage 3: MyEmailVerifier Check (HARD)
    # - Email verification: Pass IF valid, IF catch-all accept only IF SPF
    # ========================================================================
    print(f"🔍 Stage 3: MyEmailVerifier email validation for {email} @ {company}")
    passed, rejection_reason = await check_myemailverifier_email(lead)
    
    # Collect Stage 3 email data
    # Map MyEmailVerifier status to standard format: "valid", "catch-all", "invalid", "unknown"
    raw_status = lead.get("email_verifier_status", "Unknown")
    if raw_status == "Valid":
        email_status = "valid"
    elif raw_status in ["Catch All", "Catch-All", "Catch-all"]:
        email_status = "catch-all"
    elif raw_status in ["Invalid", "Grey-listed", "Unknown"]:
        # Grey-listed and Unknown are treated as invalid (tasks2.md Phase 1 requirement)
        email_status = "invalid"
    else:
        email_status = "unknown"
    
    automated_checks_data["stage_3_email"]["email_status"] = email_status
    automated_checks_data["stage_3_email"]["email_score"] = 10 if passed else 0
    automated_checks_data["stage_3_email"]["is_disposable"] = lead.get("email_verifier_disposable", False)
    automated_checks_data["stage_3_email"]["is_role_based"] = lead.get("email_verifier_role_based", False)
    automated_checks_data["stage_3_email"]["is_free"] = lead.get("email_verifier_free", False)
    
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 3 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 3 passed")

    # ========================================================================
    # Stage 4: LinkedIn/GSE Validation (HARD)
    # ========================================================================
    print(f"🔍 Stage 4: LinkedIn/GSE validation for {email} @ {company}")
    
    passed, rejection_reason = await check_linkedin_gse(lead)
    
    # Collect Stage 4 data even on failure
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")
    
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 4 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 4 passed")
    
    # Collect Stage 4 data after successful check
    automated_checks_data["stage_4_linkedin"]["linkedin_verified"] = True
    automated_checks_data["stage_4_linkedin"]["gse_search_count"] = lead.get("gse_search_count", 0)
    automated_checks_data["stage_4_linkedin"]["llm_confidence"] = lead.get("llm_confidence", "none")

    # ========================================================================
    # Stage 5: Role/Region/Industry Verification (HARD)
    # - Uses DDG search + fuzzy matching + LLM to verify role, region, industry
    # - Early exit: if role fails → skip region/industry
    # - Early exit: if region fails → skip industry
    # - Anti-gaming: rejects if miner puts multiple states in region
    # ========================================================================
    print(f"🔍 Stage 5: Role/Region/Industry verification for {email} @ {company}")
    
    passed, rejection_reason = await check_stage5_unified(lead)
    
    # Collect Stage 5 data
    automated_checks_data["stage_5_verification"]["role_verified"] = lead.get("stage5_role_match", False)
    automated_checks_data["stage_5_verification"]["region_verified"] = lead.get("stage5_region_match", False)
    automated_checks_data["stage_5_verification"]["industry_verified"] = lead.get("stage5_industry_match", False)
    automated_checks_data["stage_5_verification"]["extracted_role"] = lead.get("stage5_extracted_role")
    automated_checks_data["stage_5_verification"]["extracted_region"] = lead.get("stage5_extracted_region")
    automated_checks_data["stage_5_verification"]["extracted_industry"] = lead.get("stage5_extracted_industry")
    
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 5 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        automated_checks_data["stage_5_verification"]["early_exit"] = rejection_reason.get("early_exit") if rejection_reason else None
        return False, automated_checks_data

    print("   ✅ Stage 5 passed")

    # ========================================================================
    # Rep Score: Soft Reputation Checks (SOFT)
    # - Wayback Machine (max 6 points), SEC (max 12 points), 
    #   WHOIS/DNSBL (max 10 points), GDELT Press/Media (max 10 points),
    #   Companies House (max 10 points)
    # - Always passes, appends scores to lead
    # - Total: 0-48 points
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company}")
    
    wayback_score, wayback_data = await check_wayback_machine(lead)
    # uspto_score, uspto_data = await check_uspto_trademarks(lead)  # DISABLED
    sec_score, sec_data = await check_sec_edgar(lead)
    whois_dnsbl_score, whois_dnsbl_data = await check_whois_dnsbl_reputation(lead)
    gdelt_score, gdelt_data = await check_gdelt_mentions(lead)
    companies_house_score, companies_house_data = await check_companies_house(lead)
    
    total_rep_score = (
        wayback_score + sec_score + whois_dnsbl_score + gdelt_score +
        companies_house_score
    )
    
    # Append to lead data
    lead["rep_score"] = total_rep_score
    lead["rep_score_details"] = {
        "wayback": wayback_data,
        "sec": sec_data,
        "whois_dnsbl": whois_dnsbl_data,
        "gdelt": gdelt_data,
        "companies_house": companies_house_data
    }
    
    # Append to automated_checks_data
    automated_checks_data["rep_score"] = {
        "total_score": total_rep_score,
        "max_score": MAX_REP_SCORE,
        "breakdown": {
            "wayback_machine": wayback_score,       # 0-6 points
            "sec_edgar": sec_score,                 # 0-12 points
            "whois_dnsbl": whois_dnsbl_score,       # 0-10 points
            "gdelt": gdelt_score,                   # 0-10 points
            "companies_house": companies_house_score      # 0-10 points
        }
    }
    
    print(f"   📊 Rep Score: {total_rep_score:.1f}/{MAX_REP_SCORE} (Wayback: {wayback_score:.1f}/6, SEC: {sec_score:.1f}/12, WHOIS/DNSBL: {whois_dnsbl_score:.1f}/10, GDELT: {gdelt_score:.1f}/10, Companies House: {companies_house_score:.1f}/10)")
    
    print(f"🎉 All stages passed for {email} @ {company}")

    # All checks passed - return structured success data
    automated_checks_data["passed"] = True
    automated_checks_data["rejection_reason"] = None
    
    # IMPORTANT: Also set rep_score on lead object for validator.py to pick up
    # validator.py looks for lead_blob.get("rep_score", 50)
    lead["rep_score"] = total_rep_score
    
    return True, automated_checks_data

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
    except aiohttp.ClientError as e:
        error_msg = str(e)
        # Handle large enterprise websites with massive headers (>8KB)
        # This is common for major companies (e.g., Siemens, Microsoft, etc.)
        if "Header value is too long" in error_msg or "Got more than" in error_msg:
            return True, "Website accessible (large enterprise headers detected)"
        return False, f"Website inaccessible: {error_msg}"
    except Exception as e:
        return False, f"Website inaccessible: {str(e)}"

async def check_duplicates(leads: list) -> Tuple[bool, dict]:
    """Check for duplicate emails and return which leads are duplicates (not first occurrence)"""
    email_first_occurrence = {}  # Track first occurrence of each email
    duplicate_leads = {}  # Track which lead indices are duplicates

    for i, lead in enumerate(leads):
        email = get_email(lead)

        if email in email_first_occurrence:
            # This is a duplicate - mark this lead index as duplicate
            duplicate_leads[i] = email
        else:
            # First occurrence - record the lead index
            email_first_occurrence[email] = i

    return len(duplicate_leads) > 0, duplicate_leads

async def validate_lead_list(leads: list) -> list:
    """Main validation function - maintains backward compatibility"""

    # Check for duplicates
    has_duplicates, duplicate_leads = await check_duplicates(leads)
    if has_duplicates:
        duplicate_emails = set(duplicate_leads.values())
        print(f"Duplicate emails detected: {duplicate_emails}")
        print(f"Duplicate lead indices: {list(duplicate_leads.keys())}")

        # Process all leads, but mark duplicates as invalid
        report = []
        for i, lead in enumerate(leads):
            email = get_email(lead)
            website = get_website(lead)
            domain = urlparse(website).netloc if website else ""

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
                passed, automated_checks_data = await run_automated_checks(lead)
                status = "Valid" if passed else "Invalid"
                # Extract rejection_reason for backwards compatibility
                reason = automated_checks_data.get("rejection_reason", {}) if not passed else {}
                report.append({
                    "lead_index": i,
                    "email": email,
                    "company_domain": domain,
                    "status": status,
                    "reason": reason,
                    "automated_checks": automated_checks_data  # NEW: Include full structured data
                })

        return report

    # Process each lead through the new validation pipeline
    report = []
    for i, lead in enumerate(leads):
        email = get_email(lead)
        website = get_website(lead)
        domain = urlparse(website).netloc if website else ""

        # Run new automated checks
        passed, automated_checks_data = await run_automated_checks(lead)

        status = "Valid" if passed else "Invalid"
        # Extract rejection_reason for backwards compatibility
        reason = automated_checks_data.get("rejection_reason", {}) if not passed else {}
        report.append({
            "lead_index": i,
            "email": email,
            "company_domain": domain,
            "status": status,
            "reason": reason,
            "automated_checks": automated_checks_data  # NEW: Include full structured data
        })

    return report

# DEPRECATED: Collusion detection function (never used in production)
# async def collusion_check(validators: list, responses: list) -> dict:
#     """Simulate PyGOD/DBScan collusion detection."""
#     validator_scores = []
#     for v in validators:
#         for r in responses:
#             validation = await v.validate_leads(r.leads)
#             validator_scores.append({"hotkey": v.wallet.hotkey.ss58_address, "O_v": validation["O_v"]})
# 
#     # Mock PyGOD analysis
#     data = np.array([[s["O_v"]] for s in validator_scores])
#     detector = DOMINANT()
#     detector.fit(data)
#     V_c = detector.decision_score_.max()
# 
#     collusion_flags = {}
#     for v in validators:
#         collusion_flags[v.wallet.hotkey.ss58_address] = 0 if V_c > 0.7 else 1
#     return collusion_flags
