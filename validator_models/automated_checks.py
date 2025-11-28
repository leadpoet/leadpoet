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
            def do_search():
                return DDGS().text(query, max_results=max_results)
            
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

async def search_linkedin_ddg(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> List[dict]:
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
        List of search results with title, link, snippet
    """
    if not linkedin_url:
        print(f"   ⚠️ No LinkedIn URL provided")
        return []
    
    # Extract profile slug from LinkedIn URL
    profile_slug = linkedin_url.split("/in/")[-1].strip("/") if "/in/" in linkedin_url else None
    
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
                
                target_person_results = []
                other_person_results = []
                
                for item in profile_headlines:
                    title_lower = item.get("title", "").lower()
                    # Check if target person's name is in the title
                    if first_name in title_lower and last_name in title_lower:
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
                    # Return only target person's profile headlines
                    return target_person_results[:max_results]
                elif profile_headlines:
                    # No exact name match - use all profile headlines
                    print(f"      📊 DDG Profile Headlines (no exact name match):")
                    for i, item in enumerate(profile_headlines[:3], 1):
                        print(f"         {i}. {item.get('title', '')[:70]}")
                    if posts:
                        print(f"      📊 Posts filtered out: {len(posts)}")
                    return profile_headlines[:max_results]
                else:
                    # No profile headlines found, fall back to posts
                    print(f"      📊 DDG Posts only (no profile headlines found):")
                    for i, item in enumerate(posts[:3], 1):
                        print(f"         {i}. {item.get('title', '')[:70]}")
                    return posts[:max_results]
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
    return []

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

async def verify_linkedin_with_llm(full_name: str, company: str, linkedin_url: str, search_results: List[dict]) -> Tuple[bool, str]:
    """
    Use OpenRouter LLM to verify if search results match the person.

    Args:
        full_name: Person's full name
        company: Company name
        linkedin_url: Provided LinkedIn URL
        search_results: Google search results

    Returns:
        (is_verified, reasoning)
    """
    try:
        if not search_results:
            return False, "No LinkedIn search results found"
        
        # DETERMINISTIC PRE-CHECK: Check company match from titles directly
        # This avoids LLM hallucination issues
        company_lower = company.lower().strip()
        
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
        
        # Method 3: Check snippet for "Experience: [Company]" pattern ONLY (LinkedIn format)
        # This is STRICT - we only match the LinkedIn standard format
        # e.g., "Experience: Bank of America" in snippet
        # We do NOT match general mentions like "Former EverCommerce" or random text
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
                
                # OVERRIDE: Use deterministic company match instead of LLM's decision
                # This prevents LLM hallucination issues
                company_match = deterministic_company_match
                
                if llm_company_match != deterministic_company_match:
                    print(f"   ⚠️ LLM company_match ({llm_company_match}) OVERRIDDEN by deterministic check ({deterministic_company_match})")
                    reasoning = f"[Deterministic: company '{company}' {'found' if deterministic_company_match else 'NOT found'} in title] {reasoning}"
                
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
        if USE_DDG_SEARCH:
            try:
                search_results = await search_linkedin_ddg(full_name, company, linkedin_url)
                # NO GSE fallback if DDG returns empty (URL mismatch or no results)
                # This prevents gaming: if DDG can't find the profile, it's likely invalid
            except Exception as e:
                print(f"   ⚠️ DuckDuckGo API/request failed: {e}, falling back to GSE")
                search_results = await search_linkedin_gse(full_name, company, linkedin_url)
        else:
            search_results = await search_linkedin_gse(full_name, company, linkedin_url)
        
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
        
        # Step 2: Verify with LLM
        verified, reasoning = await verify_linkedin_with_llm(full_name, company, linkedin_url, search_results)
        
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
        "stage_4_linkedin": {  # NEW
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        },
        "rep_score": {  # NEW
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
