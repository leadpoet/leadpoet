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

MAX_REP_SCORE = 38  # Wayback (6) + SEC (12) + WHOIS/DNSBL (10) + GDELT (10) = 38 (USPTO removed)

# Custom exception for API infrastructure failures (should skip lead, not submit)
class EmailVerificationUnavailableError(Exception):
    """Raised when email verification API is unavailable (no credits, bad key, network error, etc.)"""
    pass

load_dotenv()
MYEMAILVERIFIER_API_KEY = os.getenv("MYEMAILVERIFIER_API_KEY", "YOUR_MYEMAILVERIFIER_API_KEY")

# NEW: Stage 4 API keys (Google Search Engine + OpenRouter LLM)
GSE_CX = os.getenv("GSE_CX", "")
GSE_API_KEY = os.getenv("GSE_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")

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
        patterns = [
            first_normalized,                           # john
            last_normalized,                            # doe
            f"{first_normalized}{last_normalized}",     # johndoe
            f"{first_normalized[0]}{last_normalized}",  # jdoe
            f"{last_normalized}{first_normalized[0]}",  # doej
        ]
        
        name_match = any(pattern in local_normalized for pattern in patterns if pattern)
        
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

# Stage 3: MyEmailVerifier Check

async def check_myemailverifier_email(lead: dict) -> Tuple[bool, dict]:
    """
    Check email validity using MyEmailVerifier API (with retry logic)
    
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
                        if status == "Valid":
                            result = (True, {})
                        elif status == "Catch All":
                            # BRD: Reject ALL catch-all emails (no exceptions)
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
                            # Any other status, log and assume valid
                            result = (True, {})

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

async def search_linkedin_gse(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> List[dict]:
    """
    Search LinkedIn using Google Custom Search Engine for person's profile.
    
    Tries multiple search variations to maximize chances of finding the profile:
    1. Exact URL in quotes
    2. URL without protocol
    3. Profile slug only
    4. Name + LinkedIn + company
    5. Name + site:linkedin.com (broadest)

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
    query_variations = [
        # 1. Exact URL in quotes (most specific)
        f'"{linkedin_url}"',
        
        # 2. URL without protocol (handles http vs https differences)
        f'"{linkedin_url.replace("https://", "").replace("http://", "")}"',
        
        # 3. Profile slug only (handles www. differences)
        f'"linkedin.com/in/{profile_slug}"' if profile_slug else None,
        
        # 4. Name + LinkedIn + company (more context)
        f'"{full_name}" linkedin "{company}"',
        
        # 5. Name + site:linkedin.com (broadest - catches any LinkedIn profile for this person)
        f'"{full_name}" site:linkedin.com',
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
                        
                        # DECISION LOGIC:
                        # - Variations 1-3: Searched for exact URL → Always accept (they found the URL)
                        # - Variations 4-5: Searched by name/company
                        #     → If results contain direct profiles (/in/), verify URL match
                        #     → If results only contain directory pages (/pub/dir/), SKIP URL check
                        #       (directory pages will NEVER match /in/ URLs, so check is useless)
                        #       Instead, rely on LLM to verify person + company from snippets
                        
                        if variation_idx >= 4 and profile_slug:
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

2. COMPANY MATCH: Do search results show person CURRENTLY works at "{company}"?
   - Look for CURRENT employment/title, company name, or business description
   - ACCEPT if:
     * Company name "{company}" explicitly mentioned, OR
     * Snippet describes the company's product/service matching the business context
     * Person's title/role is consistent with working at this type of company
   - REJECT only if clearly shows DIFFERENT company name
   - Be lenient: Descriptions like "Financial Operating System" for company "Toku" are acceptable

3. PROFILE VALID: Is profile legitimate and indexed by Google?
   - Profile appears in search results = valid

CRITICAL: Check name AND company separately. Both must match.

Respond ONLY with JSON: {{"name_match": true/false, "company_match": true/false, "profile_valid": true/false, "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}"""
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-3.5-turbo",
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
                
                # Parse JSON response with 3 separate checks
                result = json.loads(llm_response)
                
                name_match = result.get("name_match", False)
                company_match = result.get("company_match", False)
                profile_valid = result.get("profile_valid", False)
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                # DEBUG: Print LLM analysis with all 3 checks
                print(f"   🤖 LLM Analysis:")
                print(f"      Name Match: {name_match} (Does {full_name} match the profile?)")
                print(f"      Company Match: {company_match} (Does person work at {company}?)")
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
        
        # Step 1: Search LinkedIn via Google Custom Search (using provided URL for targeted search)
        print(f"   🔍 Stage 4: Verifying LinkedIn profile for {full_name} at {company}")
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
        
        # Query Wayback Machine CDX API
        url = f"https://web.archive.org/cdx/search/cdx"
        params = {
            "url": domain,
            "output": "json",
            "limit": 1000,
            "fl": "timestamp"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
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
                # - 0-10 snapshots: 0-1.2 points
                # - 11-50 snapshots: 1.8-3 points
                # - 51-200 snapshots: 3.6-4.8 points
                # - 200+ snapshots: 5.4-6 points
                # - Bonus: +0.6 for age > 5 years
                
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
        return 0, {"checked": False, "reason": "Wayback API timeout"}
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
        query = f'"{company}" sourcelang:eng'
        
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
                "gdelt": 0
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
    elif raw_status == "Catch All":
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
    #   WHOIS/DNSBL (max 10 points), GDELT Press/Media (max 10 points)
    # - Always passes, appends scores to lead
    # - Total: 0-38 points (USPTO removed)
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company}")
    
    wayback_score, wayback_data = await check_wayback_machine(lead)
    # uspto_score, uspto_data = await check_uspto_trademarks(lead)  # DISABLED
    sec_score, sec_data = await check_sec_edgar(lead)
    whois_dnsbl_score, whois_dnsbl_data = await check_whois_dnsbl_reputation(lead)
    gdelt_score, gdelt_data = await check_gdelt_mentions(lead)
    
    total_rep_score = wayback_score + sec_score + whois_dnsbl_score + gdelt_score  # USPTO removed
    
    # Append to lead data
    lead["rep_score"] = total_rep_score
    lead["rep_score_details"] = {
        "wayback": wayback_data,
        "sec": sec_data,
        "whois_dnsbl": whois_dnsbl_data,
        "gdelt": gdelt_data
    }
    
    # Append to automated_checks_data
    automated_checks_data["rep_score"] = {
        "total_score": total_rep_score,
        "max_score": MAX_REP_SCORE,
        "breakdown": {
            "wayback_machine": wayback_score,       # 0-6 points
            "sec_edgar": sec_score,                 # 0-12 points
            "whois_dnsbl": whois_dnsbl_score,       # 0-10 points
            "gdelt": gdelt_score                    # 0-10 points
        }
    }
    
    print(f"   📊 Rep Score: {total_rep_score:.1f}/{MAX_REP_SCORE} (Wayback: {wayback_score:.1f}/6, SEC: {sec_score:.1f}/12, WHOIS/DNSBL: {whois_dnsbl_score:.1f}/10, GDELT: {gdelt_score:.1f}/10)")
    
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
