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
from pygod.detector import DOMINANT
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
            lead["dnsbl_list"] = cached_data.get("list", "spamhaus_dbl")
            lead["dnsbl_domain"] = cached_data.get("domain", root_domain)
        return cached_result

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

            # Append DNSBL data to lead
            lead["dnsbl_checked"] = True
            lead["dnsbl_blacklisted"] = is_blacklisted
            lead["dnsbl_list"] = "spamhaus_dbl"
            lead["dnsbl_domain"] = root_domain

            # Cache the data separately for restoration
            dnsbl_data = {
                "checked": True,
                "blacklisted": is_blacklisted,
                "list": "spamhaus_dbl",
                "domain": root_domain
            }
            validation_cache[f"{cache_key}_data"] = dnsbl_data

            if is_blacklisted:
                result = (False, {
                    "stage": "Stage 2: Domain Reputation",
                    "check_name": "check_dnsbl",
                    "message": f"Domain {root_domain} blacklisted in Spamhaus DBL",
                    "failed_fields": ["email"]
                })
                print(f"❌ DNSBL: Domain {root_domain} found in Spamhaus blacklist")
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
    Check email validity using MyEmailVerifier API
    
    MyEmailVerifier provides real-time email verification with:
    - Syntax validation
    - Domain/MX record checks
    - Catch-all detection
    - Disposable email detection
    - Spam trap detection
    - Role account detection
    
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

    try:
        async with API_SEMAPHORE:
            async with aiohttp.ClientSession() as session:
                # MyEmailVerifier API endpoint
                # Correct endpoint format: https://client.myemailverifier.com/verifier/validate_single/{email}/{API_KEY}
                url = f"https://client.myemailverifier.com/verifier/validate_single/{email}/{MYEMAILVERIFIER_API_KEY}"
                
                print(f"   📞 Calling MyEmailVerifier API for: {email}")
                
                async with session.get(url, timeout=15) as response:
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
        # Re-raise to propagate up to validator
        raise
    except asyncio.TimeoutError:
        # Timeout - SKIP the lead (API unavailable)
        print(f"   🚨 MyEmailVerifier API: Timeout (>15s)")
        raise EmailVerificationUnavailableError("MyEmailVerifier API timeout")
    except aiohttp.ClientError as e:
        # Network error - SKIP the lead (API unavailable)
        print(f"   🚨 MyEmailVerifier API: Network error")
        raise EmailVerificationUnavailableError(f"MyEmailVerifier API network error: {str(e)}")
    except Exception as e:
        # Any other error - SKIP the lead (API unavailable)
        print(f"   🚨 MyEmailVerifier API: Unexpected error")
        raise EmailVerificationUnavailableError(f"MyEmailVerifier API error: {str(e)}")

# Stage 4: LinkedIn/GSE Validation

async def search_linkedin_gse(full_name: str, company: str, max_results: int = 5) -> List[dict]:
    """
    Search LinkedIn using Google Custom Search Engine for person's profile.

    Args:
        full_name: Person's full name
        company: Company name
        max_results: Max search results to return

    Returns:
        List of search results with title, link, snippet
    """
    try:
        query = f'"{full_name}" "{company}" site:linkedin.com'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GSE_API_KEY,
            "cx": GSE_CX,
            "q": query,
            "num": max_results
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    print(f"   ⚠️ GSE API error: HTTP {response.status}")
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                
                return results
    
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
        
        # Prompt adapted from calculate-rep-score/utils/llm-client.ts
        prompt = f"""Analyze these search results to validate LinkedIn profile and employer match.

{linkedin_url_line}Company: {company}
Full Name: {full_name}

Search Results:
{results_text}

Determine:
1. Is there evidence that {full_name} currently works at {company}?
2. Do the search results suggest the LinkedIn profile is legitimate and active?
3. Are there any red flags indicating this might be a fake profile or the person no longer works there?
4. If you find a LinkedIn URL in the search results, include it in your response.

Score based on:
- Strong evidence of current employment at company = higher score
- Multiple sources confirming the connection = higher score
- Recent mentions or activity = higher score
- No contradictory information = higher score
- LinkedIn profile URL format is valid = bonus points
- If provided LinkedIn URL matches found URL = bonus points

Respond ONLY with valid JSON in this exact format: {{"linkedin_valid": true/false, "employer_match": true/false, "confidence": 0.0-1.0, "reasoning": "Brief explanation", "found_linkedin_url": "url or null"}}"""
        
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
                
                # Parse JSON response
                result = json.loads(llm_response)
                
                linkedin_valid = result.get("linkedin_valid", False)
                employer_match = result.get("employer_match", False)
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                # Verification passes if:
                # 1. LinkedIn profile is valid
                # 2. Employer match is confirmed
                # 3. Confidence is >= 0.5 (medium to high)
                if linkedin_valid and employer_match and confidence >= 0.5:
                    return True, reasoning
                else:
                    return False, reasoning
    
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
        
        # Step 1: Search LinkedIn via Google Custom Search
        print(f"   🔍 Stage 4: Searching LinkedIn for {full_name} at {company}")
        search_results = await search_linkedin_gse(full_name, company)
        
        if not search_results:
            return False, {
                "stage": "Stage 4: LinkedIn/GSE Validation",
                "check_name": "check_linkedin_gse",
                "message": f"No LinkedIn profiles found for {full_name} at {company}",
                "failed_fields": ["linkedin", "full_name", "company"]
            }
        
        # Step 2: Verify with LLM
        verified, reasoning = await verify_linkedin_with_llm(full_name, company, linkedin_url, search_results)
        
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
    
    Returns score (0-10) based on:
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
                
                # Scoring logic:
                # - 0-10 snapshots: 0-2 points
                # - 11-50 snapshots: 3-5 points
                # - 51-200 snapshots: 6-8 points
                # - 200+ snapshots: 9-10 points
                # - Bonus: +1 for age > 5 years
                
                if snapshots < 10:
                    score = min(2, snapshots * 0.2)
                elif snapshots < 50:
                    score = 3 + (snapshots - 10) * 0.05
                elif snapshots < 200:
                    score = 6 + (snapshots - 50) * 0.013
                else:
                    score = 9 + min(1, (snapshots - 200) * 0.001)
                
                # Age bonus
                if age_years >= 5:
                    score = min(10, score + 1)
                
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

async def check_uspto_trademarks(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check USPTO for company trademarks.
    
    Returns score (0-10) based on:
    - Number of registered trademarks
    - Age of trademarks
    - Active status
    
    This is a SOFT check - always passes, appends score.

    Args:
        lead: Lead data with company

    Returns:
        (score, metadata)
    """
    try:
        company = get_company(lead)
        if not company:
            return 0, {"checked": False, "reason": "No company provided"}
        
        # USPTO Trademark Search API
        # Note: This is a simplified example - USPTO API structure may vary
        headers = {"API_KEY": os.getenv("USPTO_API_KEY", "")}
        url = "https://developer.uspto.gov/ibd-api/v1/application/grants"
        params = {
            "searchText": company,
            "rows": 100
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=15) as response:
                if response.status != 200:
                    return 0, {"checked": False, "reason": f"USPTO API error: {response.status}"}
                
                data = await response.json()
                results = data.get("response", {}).get("docs", [])
                
                if not results:
                    return 0, {"checked": True, "trademarks": 0, "reason": "No trademarks found"}
                
                # Count active trademarks
                active_count = sum(1 for tm in results if tm.get("status") == "LIVE")
                total_count = len(results)
                
                # Calculate age of oldest trademark
                grant_dates = [tm.get("grantDate") for tm in results if tm.get("grantDate")]
                oldest_year = min([int(d[:4]) for d in grant_dates]) if grant_dates else datetime.now().year
                age_years = datetime.now().year - oldest_year
                
                # Scoring logic:
                # - 1-2 trademarks: 3 points
                # - 3-5 trademarks: 5 points
                # - 6-10 trademarks: 7 points
                # - 10+ trademarks: 9-10 points
                # - Bonus: +1 for age > 10 years
                
                if total_count <= 2:
                    score = min(3, total_count * 1.5)
                elif total_count <= 5:
                    score = 5
                elif total_count <= 10:
                    score = 7
                else:
                    score = 9 + min(1, (total_count - 10) * 0.1)
                
                # Age bonus
                if age_years >= 10:
                    score = min(10, score + 1)
                
                return score, {
                    "checked": True,
                    "total_trademarks": total_count,
                    "active_trademarks": active_count,
                    "age_years": age_years,
                    "score": score
                }
    
    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "USPTO API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"USPTO check error: {str(e)}"}

async def check_sec_edgar(lead: dict) -> Tuple[float, dict]:
    """
    Rep Score: Check SEC EDGAR for company filings.
    
    Returns score (0-10) based on:
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
        
        # SEC.gov requires User-Agent header with contact info (no API key needed)
        headers = {
            "User-Agent": "LeadPoet/1.0 (hello@leadpoet.com)"
        }
        
        # Use SEC.gov company search endpoint to find CIK
        # This searches the submissions index for company name matches
        search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "company": company,
            "action": "getcompany",
            "count": "10"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, params=params, timeout=15) as response:
                if response.status != 200:
                    return 0, {
                        "checked": False,
                        "reason": f"SEC API error: HTTP {response.status}"
                    }
                
                # Parse HTML response (SEC doesn't return JSON for this endpoint)
                html = await response.text()
                
                # Check if company was found (HTML contains "No matching" if not found)
                if "No matching" in html or "No results" in html:
                    return 0, {
                        "checked": True,
                        "filings": 0,
                        "reason": f"No SEC filings found for {company}"
                    }
                
                # Count filing indicators in HTML (rough estimate)
                # Look for common filing types
                filing_types = ["10-K", "10-Q", "8-K", "S-1", "10-K/A", "10-Q/A"]
                total_filings = 0
                for filing_type in filing_types:
                    total_filings += html.count(filing_type)
                
                if total_filings == 0:
                    return 0, {
                        "checked": True,
                        "filings": 0,
                        "reason": f"No filings detected for {company}"
                    }
                
                # Scoring logic (similar to old implementation):
                # - 1-5 filings: 3 points
                # - 6-20 filings: 6 points
                # - 21-50 filings: 8 points
                # - 50+ filings: 10 points
                
                if total_filings <= 5:
                    score = min(3, total_filings * 0.6)
                elif total_filings <= 20:
                    score = 6
                elif total_filings <= 50:
                    score = 8
                else:
                    score = 10
                
                return score, {
                    "checked": True,
                    "filings": total_filings,
                    "score": score,
                    "reason": f"Found {total_filings} SEC filing indicators for {company}"
                }

    except asyncio.TimeoutError:
        return 0, {"checked": False, "reason": "SEC API timeout"}
    except Exception as e:
        return 0, {"checked": False, "reason": f"SEC check error: {str(e)}"}

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
    company = lead.get("Company", "")
    
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
            "max_score": 30,
            "breakdown": {
                "wayback_machine": 0,
                "uspto_trademarks": 0,
                "sec_edgar": 0
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
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 4 failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data

    print("   ✅ Stage 4 passed")
    
    # Collect Stage 4 data after successful check
    automated_checks_data["stage_4_linkedin"]["linkedin_verified"] = True

    # ========================================================================
    # Rep Score: Soft Reputation Checks (SOFT)
    # - Wayback Machine, USPTO, SEC
    # - Always passes, appends scores to lead
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company}")
    
    wayback_score, wayback_data = await check_wayback_machine(lead)
    uspto_score, uspto_data = await check_uspto_trademarks(lead)
    sec_score, sec_data = await check_sec_edgar(lead)
    
    total_rep_score = wayback_score + uspto_score + sec_score
    
    # Append to lead data
    lead["rep_score"] = total_rep_score
    lead["rep_score_details"] = {
        "wayback": wayback_data,
        "uspto": uspto_data,
        "sec": sec_data
    }
    
    # Append to automated_checks_data
    automated_checks_data["rep_score"] = {
        "total_score": total_rep_score,
        "max_score": 30,
        "breakdown": {
            "wayback_machine": wayback_score,
            "uspto_trademarks": uspto_score,
            "sec_edgar": sec_score
        }
    }
    
    print(f"   📊 Rep Score: {total_rep_score}/30 (Wayback: {wayback_score}, USPTO: {uspto_score}, SEC: {sec_score})")
    
    print(f"🎉 All stages passed for {email} @ {company}")

    # All checks passed - return structured success data
    automated_checks_data["passed"] = True
    automated_checks_data["rejection_reason"] = None
    
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
