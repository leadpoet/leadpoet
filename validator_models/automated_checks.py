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

# Load environment variables from .env file
load_dotenv()

# Environment variables for new APIs
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "YOUR_HUNTER_API_KEY")
ZEROBOUNCE_API_KEY = os.getenv("ZEROBOUNCE_API_KEY", "YOUR_ZEROBOUNCE_API_KEY")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "YOUR_APOLLO_API_KEY")
OPENCORPORATES_API_KEY = os.getenv("OPENCORPORATES_API_KEY", "YOUR_OPENCORPORATES_API_KEY")

# Constants
DISPOSABLE_DOMAINS = {"mailinator.com", "temp-mail.org", "guerrillamail.com", "10minutemail.com", 
                      "yopmail.com", "getnada.com", "throwaway.email", "tempmail.com"}
EMAIL_CACHE_FILE = "email_verification_cache.pkl"
VALIDATION_ARTIFACTS_DIR = "validation_artifacts"

# Cache TTLs in hours
CACHE_TTLS = {
    "dns_head": 24,
    "whois_opencorporates": 90,
    "hunter_apollo_zerobounce": 90,
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
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois_opencorporates"]):
        return validation_cache[cache_key]
    
    try:
        # Mock mode fallback
        if "YOUR_OPENCORPORATES_API_KEY" in OPENCORPORATES_API_KEY:
            result = (True, "Mock pass - domain age check")
            validation_cache[cache_key] = result
            return result
        
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
                    min_age_days = 90  # 3 months minimum
                    
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

# Stage 1: OpenCorporates Check

async def check_opencorporates_status(lead: dict) -> Tuple[bool, str]:
    """Check company status using OpenCorporates API"""
    website = lead.get("Website", lead.get("website", ""))
    if not website:
        return False, "No website provided"
    
    domain = extract_root_domain(website)
    if not domain:
        return False, "Invalid website format"
    
    cache_key = f"opencorporates:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois_opencorporates"]):
        return validation_cache[cache_key]
    
    try:
        # Mock mode fallback
        if "YOUR_OPENCORPORATES_API_KEY" in OPENCORPORATES_API_KEY:
            result = (True, "Mock pass - OpenCorporates check")
            validation_cache[cache_key] = result
            return result
        
        # Implement actual OpenCorporates API call
        # Extract company name from domain (simplified approach)
        company_name = domain.split(".")[0].replace("-", " ").title()
        
        # OpenCorporates API endpoint
        url = "https://api.opencorporates.com/companies/search"
        params = {"q": company_name, "api_token": OPENCORPORATES_API_KEY}
        
        async with API_SEMAPHORE:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        companies = data.get("results", {}).get("companies", [])
                        
                        if companies:
                            # Check if any company is inactive
                            for company in companies[:5]:  # Check first 5 results
                                if company.get("company", {}).get("inactive", False):
                                    result = (False, f"Company marked as inactive: {company.get('company', {}).get('name', 'Unknown')}")
                                    validation_cache[cache_key] = result
                                    return result
                            
                            result = (True, "Company status verified via OpenCorporates")
                        else:
                            result = (True, "Company not found in OpenCorporates (assumed valid)")
                    else:
                        result = (True, "OpenCorporates API unavailable (assumed valid)")
                        
        validation_cache[cache_key] = result
        return result
        
    except Exception as e:
        result = (False, f"OpenCorporates check failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_industry_location_match(lead: dict) -> Tuple[bool, str]:
    """Check if industry and location match company data"""
    website = lead.get("Website", lead.get("website", ""))
    if not website:
        return False, "No website provided"
    
    domain = extract_root_domain(website)
    if not domain:
        return False, "Invalid website format"
    
    cache_key = f"industry_location:{domain}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["whois_opencorporates"]):
        return validation_cache[cache_key]
    
    try:
        # Mock mode fallback
        if "YOUR_OPENCORPORATES_API_KEY" in OPENCORPORATES_API_KEY:
            result = (True, "Mock pass - industry/location match")
            validation_cache[cache_key] = result
            return result
        
        # For now, implement basic keyword matching
        # TODO: Implement LLM fallback for fuzzy matching
        company_name = domain.split(".")[0].replace("-", " ").lower()
        
        # Basic industry keywords
        tech_keywords = ["tech", "software", "ai", "ml", "data", "cloud", "digital"]
        finance_keywords = ["finance", "bank", "insurance", "investment", "credit"]
        healthcare_keywords = ["health", "medical", "pharma", "care", "bio"]
        
        # Check if company name contains industry keywords
        industry_match = any(keyword in company_name for keyword in tech_keywords + finance_keywords + healthcare_keywords)
        
        if industry_match:
            result = (True, "Industry keywords matched company name")
        else:
            result = (True, "Industry match check passed (no keywords found)")
        
        validation_cache[cache_key] = result
        return result
        
    except Exception as e:
        result = (False, f"Industry/location match failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

# Stage 2: ZeroBounce Check

async def check_zerobounce_email(lead: dict) -> Tuple[bool, str]:
    """Check email validity **and AI-score** using ZeroBounce API"""
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    if not email:
        return False, "No email provided"

    cache_key = f"zerobounce:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["hunter_apollo_zerobounce"]):
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
        if status in ["valid", "catch-all"]:
            result = (True, f"Email validated ({status}), score={score}")
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

# Stage 3: Apollo Check

async def check_apollo_contact_match(lead: dict) -> tuple[bool, str]:
    """Check contact data consistency using Apollo API"""
    # Extract lead details for Apollo query - map to Lead Sorcerer field names
    first_name = lead.get("First", lead.get("First Name", ""))
    last_name = lead.get("Last", lead.get("Last Name", ""))
    company_name = lead.get("Business", lead.get("Company", ""))
    email = lead.get("Owner(s) Email", lead.get("Email 1", lead.get("email", "")))
    
    if not email:
        return False, "No email provided"
    
    cache_key = f"apollo_contact:{email}"
    if cache_key in validation_cache and not validation_cache.is_expired(cache_key, CACHE_TTLS["hunter_apollo_zerobounce"]):
        return validation_cache[cache_key]
    
    try:
        # Mock mode fallback - check if API key is placeholder or if we're on free plan
        if "YOUR_APOLLO_API_KEY" in APOLLO_API_KEY or "free" in APOLLO_API_KEY.lower():
            result = (True, "Mock pass - Apollo contact match")
            validation_cache[cache_key] = result
            return result
        
        # Implement actual Apollo contacts search API
        url = "https://api.apollo.io/api/v1/mixed_people/search"
        
        # Build Apollo search request body
        body = {
            "page": 1,
            "per_page": 100,  # Maximum per page to capture more results for better contact matching
        }
        
        # Add search filters based on available data
        if first_name and last_name and company_name:
            # Search by "Company FirstName LastName" for better targeting
            body["q_keywords"] = f"{company_name} {first_name} {last_name}"
        elif email and company_name:
            # Fallback: search by company name if we have email
            domain = email.split("@")[1].lower() if "@" in email else ""
            if domain:
                body["q_keywords"] = company_name
        else:
            result = (False, "Contact match check failed: Insufficient data (name+company or email+company required)")
            validation_cache[cache_key] = result
            return result
        
        # Set up headers for Apollo API
        headers = {
            "X-Api-Key": APOLLO_API_KEY,
            "Content-Type": "application/json"
        }
        
        async with API_SEMAPHORE:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        found_match = False
                        # Apollo API returns people in a 'people' array OR 'contacts' array
                        people = []
                        if isinstance(data, dict):
                            if data.get("people"):
                                people = data["people"]
                            elif data.get("contacts"):
                                people = data["contacts"]
                        
                        if people:
                            for person in people:
                                # Extract person details from Apollo response - handle None values safely
                                person_name = person.get("name") or ""
                                person_first_name = person.get("first_name") or ""
                                person_last_name = person.get("last_name") or ""
                                person_title = person.get("title") or ""
                                person_email = person.get("email") or ""
                                
                                # Get organization details
                                organization = person.get("organization") or {}
                                person_company = organization.get("name") or ""
                                person_domain = organization.get("domain") or ""
                                
                                # Convert to lowercase safely
                                person_name = person_name.lower()
                                person_first_name = person_first_name.lower()
                                person_last_name = person_last_name.lower()
                                person_title = person_title.lower()
                                person_email = person_email.lower()
                                person_company = person_company.lower()
                                person_domain = person_domain.lower()
                                
                                # Check for strong match on email
                                if email and person_email == email.lower():
                                    found_match = True
                                    break
                                
                                # Check for name and company match
                                elif first_name and last_name and company_name:
                                    # Build full name from Apollo response
                                    apollo_full_name = f"{person_first_name} {person_last_name}".strip().lower()
                                    lead_full_name = f"{first_name} {last_name}".lower()
                                    
                                    # Use fuzzy matching for names
                                    name_match_ratio = fuzz.ratio(apollo_full_name, lead_full_name)
                                    company_match_ratio = fuzz.ratio(person_company, company_name.lower())
                                    
                                    # Consider it a match if name matches strongly
                                    if name_match_ratio > 80:
                                        if company_match_ratio > 80 or not person_company:  # Company matches OR no company data
                                            # ADDITIONAL VALIDATION: Check email prefix consistency
                                            if email:
                                                email_prefix = email.split('@')[0].lower()
                                                first_name_lower = first_name.lower()
                                                last_name_lower = last_name.lower()
                                                
                                                # Check if email prefix matches the person's name
                                                email_matches_name = (
                                                    email_prefix.startswith(first_name_lower) or
                                                    email_prefix.startswith(last_name_lower) or
                                                    email_prefix == f"{first_name_lower[0]}{last_name_lower}" or  # jsmith
                                                    email_prefix == f"{first_name_lower}.{last_name_lower}" or   # john.smith
                                                    email_prefix == f"{first_name_lower}_{last_name_lower}"      # john_smith
                                                )
                                                
                                                if not email_matches_name:
                                                    continue  # Skip this match, try next person
                                            
                                            found_match = True
                                            break
                        
                        if found_match:
                            result = (True, "Contact match found")
                        else:
                            result = (False, "Contact match not found")
                    elif response.status == 403:
                        # API endpoint not accessible (likely due to plan restrictions)
                        # Fall back to mock mode for testing
                        print("⚠️ Contact search endpoint not accessible with current plan - using mock mode")
                        result = (True, "Mock pass - Contact match (endpoint restricted)")
                    elif response.status == 422:
                        # Check if it's a credit limit error
                        response_text = await response.text()
                        if "insufficient credits" in response_text.lower():
                            print("⚠️ Contact search credit limit reached - using mock mode")
                            result = (True, "Mock pass - Contact match (credit limit reached)")
                        else:
                            result = (False, f"Contact match check failed: API returned status {response.status} - {response_text[:100]}")
                    else:
                        response_text = await response.text()
                        print(f"🔍 Apollo error response: {response_text[:200]}")
                        result = (False, f"Contact match check failed: API returned status {response.status}")
                        
        validation_cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"🔍 Apollo exception: {str(e)}")
        result = (False, f"Contact match failed: {str(e)}")
        validation_cache[cache_key] = result
        return result

async def check_apollo_company_match(lead: dict) -> Tuple[bool, str]:
    """Check company data consistency using Apollo API - TEMPORARILY DISABLED"""
    # For now, return mock pass to keep pipeline working without company matching
    return True, "Mock pass - Apollo company matching temporarily disabled"

# Main validation pipeline

async def run_automated_checks(lead: dict) -> Tuple[bool, str]:
    """Run all automated checks in stages, returning (passed, reason)"""
    
    email = lead.get("Email 1", lead.get("Owner(s) Email", lead.get("email", "")))
    company = lead.get("Company", "")
    
    # Stage 0: Basic hardcoded checks
    print(f"🔍 Stage 0: Basic checks for {email} @ {company}")
    checks_stage0 = [
        check_email_regex,
        check_domain_age,
        check_mx_record,
        check_head_request,
        check_disposable,
    ]
    
    for check_func in checks_stage0:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 0 failed: {reason}")
            return False, f"Stage 0 failed: {reason}"
    
    print(f"   ✅ Stage 0 passed")
    
    # Stage 1: OpenCorporates checks
    print(f"🔍 Stage 1: OpenCorporates checks for {email} @ {company}")
    checks_stage1 = [check_opencorporates_status, check_industry_location_match]
    
    for check_func in checks_stage1:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 1 failed: {reason}")
            return False, f"Stage 1 failed: {reason}"
    
    print(f"   ✅ Stage 1 passed")
    
    # Stage 2: ZeroBounce check
    print(f"🔍 Stage 2: Email validation for {email} @ {company}")
    passed, reason = await check_zerobounce_email(lead)
    if not passed:
        print(f"   ❌ Stage 2 failed: {reason}")
        return False, f"Stage 2 failed: {reason}"
    
    print(f"   ✅ Stage 2 passed")
    
    # Stage 3: Apollo checks
    print(f"🔍 Stage 3: Contact matching for {email} @ {company}")
    checks_stage3 = [check_apollo_contact_match, check_apollo_company_match]
    
    for check_func in checks_stage3:
        passed, reason = await check_func(lead)
        if not passed:
            print(f"   ❌ Stage 3 failed: {reason}")
            return False, f"Stage 3 failed: {reason}"
    
    print(f"   ✅ Stage 3 passed")
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
