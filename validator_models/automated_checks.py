import aiohttp
import asyncio
import dns.resolver
import pickle
import os
import re
import uuid
import whois
import json
import numpy as np
from pygod.detector import DOMINANT
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Tuple
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

load_dotenv()
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "YOUR_HUNTER_API_KEY")
MYEMAILVERIFIER_API_KEY = os.getenv("MYEMAILVERIFIER_API_KEY", "YOUR_MYEMAILVERIFIER_API_KEY")

EMAIL_CACHE_FILE = "email_verification_cache.pkl"
VALIDATION_ARTIFACTS_DIR = "validation_artifacts"

CACHE_TTLS = {
    "dns_head": 24,
    "whois": 90,
    "myemailverifier": 90,  
}

API_SEMAPHORE = asyncio.Semaphore(10)

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

async def check_email_regex(lead: dict) -> Tuple[bool, str]:
    """Check email format using RFC-5322 simplified regex"""
    try:
        email = get_email(lead)
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
        await log_validation_metrics(lead, {"passed": is_valid, "reason": reason}, "email_regex")

        return is_valid, reason
    except Exception as e:
        await log_validation_metrics(lead, {"passed": False, "reason": str(e)}, "email_regex")
        raise e

async def check_domain_age(lead: dict) -> Tuple[bool, str]:
    """Check domain age using WHOIS lookup"""
    website = get_website(lead)
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
                    
                    # Make creation_date timezone-naive if it's timezone-aware
                    if creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(tzinfo=None)

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
    website = get_website(lead)
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
    def fail_lead(lead):
        lead["has_spf"] = False
        lead["has_dmarc"] = False
        lead["dmarc_policy_strict"] = False
        return lead
        
    email = get_email(lead)
    if not email:
        # No email to check - append default values
        lead = fail_lead(lead)
        return True, "No email provided for SPF/DMARC check (SOFT - passed)"

    # Extract domain from email
    try:
        domain = email.split("@")[1].lower() if "@" in email else ""
        if not domain:
            lead = fail_lead(lead)
            return True, "Invalid email format (SOFT - passed)"
    except (IndexError, AttributeError):
        lead = fail_lead(lead)
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
    website = get_website(lead)
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
    email = get_email(lead)
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
    email = get_email(lead)
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

# Stage 3: MyEmailVerifier Check

async def check_myemailverifier_email(lead: dict) -> Tuple[bool, str]:
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
        return False, "No email provided"

    cache_key = f"myemailverifier:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["myemailverifier"]):
        print(f"   💾 Using cached MyEmailVerifier result for: {email}")
        return validation_cache[cache_key]

    try:
        # Mock-mode fallback
        if "YOUR_MYEMAILVERIFIER_API_KEY" in MYEMAILVERIFIER_API_KEY:
            print(f"   🔧 MOCK MODE: Using mock validation (no real API call)")
            result = (True, "Mock pass - MyEmailVerifier check")
            validation_cache[cache_key] = result
            return result

        async with API_SEMAPHORE:
            async with aiohttp.ClientSession() as session:
                # MyEmailVerifier API endpoint
                # Correct endpoint format: https://client.myemailverifier.com/verifier/validate_single/{email}/{API_KEY}
                url = f"https://client.myemailverifier.com/verifier/validate_single/{email}/{MYEMAILVERIFIER_API_KEY}"
                
                print(f"   📞 Calling MyEmailVerifier API for: {email}")
                
                async with session.get(url, timeout=15) as response:
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
                    
                    # Check for disposable domains
                    is_disposable = data.get("Disposable_Domain", "false") == "true"
                    if is_disposable:
                        result = (False, "Email is from a disposable/temporary email provider")
                        validation_cache[cache_key] = result
                        return result
                    
                    # Handle validation results based on Status field
                    if status == "Valid":
                        result = (True, "Email validated (valid)")
                    elif status == "Catch All":
                        # BRD: IF catch-all, accept only IF DNS TXT record contains v=spf1
                        has_spf = lead.get("has_spf", False)
                        if has_spf:
                            result = (True, "Email validated (catch-all with SPF)")
                        else:
                            result = (False, "Email is catch-all without SPF record")
                    elif status == "Invalid":
                        result = (False, "Email marked invalid")
                    elif status == "Grey-listed":
                        # Greylisted - retry recommended after 5-10 hours, but for now pass
                        result = (True, "Email greylisted (assumed valid)")
                    elif status == "Unknown":
                        # For unknown status, default to passing to avoid false negatives
                        result = (True, "Email status unknown (assumed valid)")
                    else:
                        # Any other status, log and assume valid
                        result = (True, f"Email status {status} (assumed valid)")
                    
                    validation_cache[cache_key] = result
                    return result

    except Exception as e:
        result = (False, f"MyEmailVerifier check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_terms_attestation(lead: dict) -> Tuple[bool, str]:
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
        return False, f"Missing attestation fields: {', '.join(missing)}"
    
    wallet_ss58 = lead.get("wallet_ss58")
    lead_terms_hash = lead.get("terms_version_hash")
    
    # SECURITY CHECK 1: Query Supabase for authoritative attestation record
    try:
        supabase = get_supabase_client()
        if not supabase:
            # If Supabase not available, log warning but don't fail validation
            # This prevents breaking validators during network issues
            print(f"   ⚠️  Supabase client not available - skipping attestation verification")
            return True, "Attestation check skipped (Supabase unavailable)"
        
        result = supabase.table("contributor_attestations")\
            .select("*")\
            .eq("wallet_ss58", wallet_ss58)\
            .eq("terms_version_hash", TERMS_VERSION_HASH)\
            .eq("accepted", True)\
            .execute()
        
        # SECURITY CHECK 2: Reject if no valid attestation in database
        if not result.data or len(result.data) == 0:
            return False, f"No valid attestation found in database for wallet {wallet_ss58[:10]}..."
        
        # Attestation exists in Supabase - miner has legitimately accepted terms
        supabase_attestation = result.data[0]
        
    except Exception as e:
        # Log error but don't fail validation - prevents breaking validators
        print(f"   ⚠️  Failed to verify attestation in database: {str(e)}")
        return True, "Attestation check skipped (database error)"
    
    # SECURITY CHECK 3: Verify lead metadata matches Supabase record
    if lead_terms_hash != supabase_attestation.get("terms_version_hash"):
        return False, f"Lead attestation hash mismatch (lead: {lead_terms_hash[:8]}, db: {supabase_attestation.get('terms_version_hash', '')[:8]})"
    
    # Check: Verify terms version is current
    if lead_terms_hash != TERMS_VERSION_HASH:
        return False, f"Outdated terms version (lead: {lead_terms_hash[:8]}, current: {TERMS_VERSION_HASH[:8]})"
    
    # Check: Verify boolean attestations in lead
    if not all([lead.get("lawful_collection"), 
                lead.get("no_restricted_sources"), 
                lead.get("license_granted")]):
        return False, "Incomplete attestations"
    
    return True, "Terms attestation valid (verified against database)"


async def check_source_provenance(lead: dict) -> Tuple[bool, str]:
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
        return False, "Missing source_url"
    
    if not source_type:
        return False, "Missing source_type"
    
    # Validate source_type against allowed list
    valid_types = ["public_registry", "company_site", "first_party_form", 
                   "licensed_resale", "proprietary_database"]
    if source_type not in valid_types:
        return False, f"Invalid source_type: {source_type}"
    
    # Validate source URL (checks denylist, domain age, reachability)
    try:
        is_valid, reason = await validate_source_url(source_url)
        if not is_valid:
            return False, f"Source URL validation failed: {reason}"
    except Exception as e:
        return False, f"Error validating source URL: {str(e)}"
    
    # Additional check: Extract domain and verify not restricted
    # (This is redundant with validate_source_url but provides explicit feedback)
    domain = extract_domain_from_url(source_url)
    if domain and is_restricted_source(domain):
        # Only fail if NOT a licensed resale (those are handled in next check)
        if source_type != "licensed_resale":
            return False, f"Source domain {domain} is in restricted denylist"
    
    return True, "Source provenance verified"


async def check_licensed_resale_proof(lead: dict) -> Tuple[bool, str]:
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
        return True, "Not a licensed resale submission"
    
    # Validate license proof
    is_valid, reason = validate_licensed_resale(lead)
    
    if not is_valid:
        return False, reason
    
    # Log for audit trail
    license_hash = lead.get("license_doc_hash", "")
    print(f"   📄 Licensed resale detected: hash={license_hash[:16]}...")
    
    return True, "Licensed resale proof verified"


# Main validation pipeline

async def run_automated_checks(lead: dict) -> Tuple[bool, str]:
    """Run all automated checks in stages, returning (passed, reason)"""

    email = get_email(lead)
    company = lead.get("Company", "")

    # ========================================================================
    # Stage -1: Terms Attestation Verification (HARD)
    # Verifies miner attestation against Supabase database (source of truth)
    # ========================================================================
    print(f"🔍 Stage -1: Terms attestation check for {email} @ {company}")
    
    passed, reason = await check_terms_attestation(lead)
    if not passed:
        print(f"   ❌ Stage -1 failed: {reason}")
        return False, f"Stage -1 failed: {reason}"
    
    print("   ✅ Stage -1 passed")

    # ========================================================================
    # Stage 0.5: Source Provenance Verification (HARD)
    # Validates source_url, source_type, denylist, and licensed resale proof
    # ========================================================================
    print(f"🔍 Stage 0.5: Source provenance verification for {email} @ {company}")
    
    checks_stage0_5 = [
        check_source_provenance,       # Validate source URL, type, denylist
        check_licensed_resale_proof,   # Validate license hash if applicable
    ]
    
    for check_func in checks_stage0_5:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 0.5 failed: {reason}")
            return False, f"Stage 0.5 failed: {reason}"
    
    print("   ✅ Stage 0.5 passed")

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
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 1 failed: {reason}")
            return False, f"Stage 1 failed: {reason}"

    print("   ✅ Stage 1 passed")

    # ========================================================================
    # Stage 2: Lightweight Domain Reputation Checks (HARD)
    # - DNSBL (Domain Block List) - Spamhaus DBL lookup
    # ========================================================================
    print(f"🔍 Stage 2: Domain reputation checks for {email} @ {company}")
    passed, reason = await check_dnsbl(lead)
    if not passed:
        print(f"   ❌ Stage 2 failed: {reason}")
        return False, f"Stage 2 failed: {reason}"

    print("   ✅ Stage 2 passed")

    # ========================================================================
    # Stage 3: MyEmailVerifier Check (HARD)
    # - Email verification: Pass IF valid, IF catch-all accept only IF SPF
    # ========================================================================
    print(f"🔍 Stage 3: MyEmailVerifier email validation for {email} @ {company}")
    passed, reason = await check_myemailverifier_email(lead)
    if not passed:
        print(f"   ❌ Stage 3 failed: {reason}")
        return False, f"Stage 3 failed: {reason}"

    print("   ✅ Stage 3 passed")
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

    # Mock mode fallback
    if "YOUR_HUNTER_API_KEY" in HUNTER_API_KEY:
        print("Mock mode: Assuming all leads pass automated checks")
        return [{
            "lead_index": i,
            "email": get_email(lead),
            "company_domain": urlparse(get_website(lead)).netloc,
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
        email = get_email(lead)
        website = get_website(lead)
        domain = urlparse(website).netloc if website else ""

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
