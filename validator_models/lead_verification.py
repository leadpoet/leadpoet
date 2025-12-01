"""
Lead Verification Module

Additional verification checks for lead quality:
1. Industry Classification - Zero-shot classifier using HuggingFace (FREE)
2. Role Extraction from LinkedIn - Parse DDG search results (FREE)
3. Role Verification - Apollo.io API for job title enrichment (PAID, optional)

These checks can be added to:
- Gateway (during miner submission) for real-time filtering
- Validator (during automated_checks) for scoring

Author: LeadPoet
"""

import os
import re
import asyncio
import aiohttp
import time
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# GEOPY LOCATION MATCHING (Deterministic, no LLM hallucination)
# ============================================================================

# Cache for geocoding results to avoid rate limiting
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
    
    # Check cache first
    cache_key = location.lower().strip()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]
    
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
        # Rate limit: 1 request/second
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
        pass  # geopy not installed, fall back to LLM
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"   ⚠️ Geocoding failed for '{location}': {e}")
    except Exception as e:
        print(f"   ⚠️ Geocoding error for '{location}': {e}")
    
    _geocode_cache[cache_key] = None
    return None


def locations_match_geopy(claimed: str, extracted: str, max_distance_km: float = 50) -> Tuple[bool, str]:
    """
    Compare two locations using GeoPy for deterministic matching.
    
    Returns:
        (match: bool, reason: str)
    
    Matching rules:
    - Same country code → check state/city
    - Same state/region → PASS
    - Within max_distance_km → PASS (for nearby cities like Hollywood/Fort Lauderdale)
    """
    if not claimed or not extracted:
        return False, "Missing location data - needs LLM verification"
    
    # Handle UNKNOWN from LLM
    if "UNKNOWN" in extracted.upper():
        return False, "Extracted location unknown - needs LLM verification"
    
    geo_claimed = _geocode_location(claimed)
    geo_extracted = _geocode_location(extracted)
    
    if not geo_claimed or not geo_extracted:
        # Fallback: comprehensive string matching if geocoding fails
        claimed_lower = claimed.lower().strip()
        extracted_lower = extracted.lower().strip()
        
        # Extract city names from both strings
        # "West Hollywood, US" -> "west hollywood"
        # "West Hollywood, 750 N San Vicente Blvd" -> "west hollywood"
        def extract_city(loc: str) -> str:
            loc = loc.lower().strip()
            # Split by comma and take first part (usually city)
            parts = [p.strip() for p in loc.split(',')]
            if parts:
                return parts[0]
            return loc
        
        claimed_city = extract_city(claimed)
        extracted_city = extract_city(extracted)
        
        # Check if cities match
        if claimed_city and extracted_city:
            if claimed_city == extracted_city:
                return True, f"City match: {claimed_city} (geocoding unavailable)"
            if claimed_city in extracted_city or extracted_city in claimed_city:
                return True, f"City containment match: {claimed_city} in {extracted_city}"
        
        # Check for obvious full string matches
        if claimed_lower in extracted_lower or extracted_lower in claimed_lower:
            return True, "String match (geocoding unavailable)"
        
        # Check if any significant words match (e.g., "Denver" appears in both)
        claimed_words = set(claimed_lower.replace(',', ' ').split())
        extracted_words = set(extracted_lower.replace(',', ' ').split())
        # Remove common filler words AND country codes (we check country separately)
        filler = {'us', 'usa', 'uk', 'gb', 'ca', 'au', 'the', 'of', 'and', 'st', 'ave', 'blvd', 'rd', 'dr', 'suite', 'floor', 'unit', 'united', 'states', 'america'}
        claimed_words -= filler
        extracted_words -= filler
        
        # Check for city/region word overlap
        common = claimed_words & extracted_words
        if common:
            return True, f"Location word match: {common}"
        
        # Check country codes as fallback
        for code in ["us", "usa", "uk", "gb", "ca", "au", "de", "fr", "in", "sg", "ch", "be", "nl"]:
            if code in claimed_lower and code in extracted_lower:
                # Country matches but we can't verify city/state - send to LLM
                return False, f"Same country ({code.upper()}) but no city match - needs LLM verification"
        
        return False, "Geocoding unavailable and no string match - needs LLM verification"
    
    # ==========================================================================
    # CITY MATCH CHECK BEFORE GEOCODING DISTANCE
    # "Venice, US" should match "Venice, Florida" - same city name!
    # ==========================================================================
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
    
    # Same country?
    if geo_claimed.get("country_code") != geo_extracted.get("country_code"):
        return False, f"Different countries: {geo_claimed.get('country')} vs {geo_extracted.get('country')}"
    
    # Same state/region?
    if geo_claimed.get("state") and geo_extracted.get("state"):
        if geo_claimed["state"].lower() == geo_extracted["state"].lower():
            return True, f"Same state: {geo_claimed['state']}"
    
    # Check distance for nearby cities
    if geo_claimed.get("lat") and geo_extracted.get("lat"):
        try:
            from geopy.distance import geodesic
            dist = geodesic(
                (geo_claimed["lat"], geo_claimed["lon"]),
                (geo_extracted["lat"], geo_extracted["lon"])
            ).kilometers
            
            if dist <= max_distance_km:
                return True, f"Nearby cities ({dist:.0f}km apart)"
            else:
                return False, f"Cities too far apart ({dist:.0f}km)"
        except Exception:
            pass
    
    # Fallback: same country is good enough for large companies
    return True, f"Same country: {geo_claimed.get('country')}"

def extract_role_from_linkedin_title(title: str) -> Optional[str]:
    """
    Extract job role from LinkedIn profile title.
    
    Common patterns:
    - "Sean McClain - Founder & CEO at Absci | LinkedIn"
    - "Katie Peters - CEO at Kenex Ltd | LinkedIn"
    - "John Smith - VP of Sales | LinkedIn"
    
    Returns:
        Extracted role string or None if not found
    """
    if not title:
        return None
    
    # Remove " | LinkedIn" suffix
    title = re.sub(r'\s*\|\s*LinkedIn\s*$', '', title, flags=re.IGNORECASE)
    
    # Pattern 1: "Name - Role at Company"
    match = re.search(r'^[^-]+-\s*(.+?)\s+at\s+', title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "Name - Role | LinkedIn" (no company in title)
    match = re.search(r'^[^-]+-\s*(.+)$', title)
    if match:
        return match.group(1).strip()
    
    return None


def normalize_role(role: str) -> str:
    """Normalize role for comparison (lowercase, remove common variations)."""
    if not role:
        return ""
    
    role = role.lower().strip()
    
    # Common abbreviations
    role = role.replace("&", "and")
    role = role.replace("vp ", "vice president ")
    role = role.replace("svp ", "senior vice president ")
    role = role.replace("evp ", "executive vice president ")
    role = role.replace("ceo", "chief executive officer")
    role = role.replace("cto", "chief technology officer")
    role = role.replace("cfo", "chief financial officer")
    role = role.replace("coo", "chief operating officer")
    role = role.replace("cmo", "chief marketing officer")
    role = role.replace("cpo", "chief product officer")
    role = role.replace("dir.", "director")
    role = role.replace("mgr.", "manager")
    role = role.replace("sr.", "senior")
    role = role.replace("jr.", "junior")
    
    return role


def roles_match(role1: str, role2: str, fuzzy: bool = True) -> Tuple[bool, float]:
    """
    Check if two roles match.
    
    Args:
        role1: First role (e.g., miner's claimed role)
        role2: Second role (e.g., extracted from LinkedIn)
        fuzzy: Allow partial/fuzzy matching
    
    Returns:
        (is_match, confidence)
    """
    if not role1 or not role2:
        return False, 0.0
    
    norm1 = normalize_role(role1)
    norm2 = normalize_role(role2)
    
    # Exact match
    if norm1 == norm2:
        return True, 1.0
    
    # One contains the other
    if norm1 in norm2 or norm2 in norm1:
        return True, 0.9
    
    if not fuzzy:
        return False, 0.0
    
    # Word overlap for fuzzy matching
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    # Remove common filler words
    filler = {"at", "of", "the", "and", "for", "in", "a", "an"}
    words1 = words1 - filler
    words2 = words2 - filler
    
    if not words1 or not words2:
        return False, 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    similarity = intersection / union if union > 0 else 0
    
    # Key role words that must match
    key_roles = {"ceo", "cto", "cfo", "coo", "founder", "director", "manager", 
                 "president", "vice", "chief", "head", "lead", "owner", "partner"}
    
    key1 = words1 & key_roles
    key2 = words2 & key_roles
    
    # If key role words exist, they should match
    if key1 and key2:
        if key1 != key2:
            # Different key roles = likely mismatch
            return False, similarity * 0.5
    
    # High similarity = match
    if similarity >= 0.5:
        return True, similarity
    
    return False, similarity


async def verify_role_from_ddg(
    search_results: List[Dict],
    expected_role: str,
    expected_name: str
) -> Tuple[bool, str, dict]:
    """
    Verify role from DDG LinkedIn search results (FREE).
    
    This extracts the role from LinkedIn profile titles and compares
    to the miner's claimed role.
    
    Args:
        search_results: DDG search results with title, link, snippet
        expected_role: Role claimed by miner
        expected_name: Person's name (to filter correct profile)
    
    Returns:
        (is_verified, extracted_role, metadata)
    """
    if not search_results:
        return False, None, {"error": "No search results"}
    
    if not expected_role:
        return False, None, {"error": "No expected role provided"}
    
    # Find the best matching profile (first result is usually correct)
    name_lower = expected_name.lower() if expected_name else ""
    
    for result in search_results:
        title = result.get("title", "")
        
        # Check if this result is for the expected person
        if name_lower:
            title_lower = title.lower()
            name_parts = name_lower.split()
            if not all(part in title_lower for part in name_parts[:2]):
                continue  # Skip results for other people
        
        # Extract role from title
        extracted_role = extract_role_from_linkedin_title(title)
        
        if extracted_role:
            # Compare roles
            is_match, confidence = roles_match(expected_role, extracted_role)
            
            return is_match, extracted_role, {
                "checked": True,
                "expected_role": expected_role,
                "extracted_role": extracted_role,
                "is_match": is_match,
                "confidence": confidence,
                "source_title": title,
                "source": "linkedin_ddg"
            }
    
    return False, None, {
        "checked": True,
        "expected_role": expected_role,
        "extracted_role": None,
        "error": "Could not extract role from search results"
    }


# ============================================================================
# 1. INDUSTRY CLASSIFICATION
# ============================================================================
# Method: Zero-shot classification using HuggingFace transformers
# Model: facebook/bart-large-mnli (free, runs locally)
# Cost: Free (local) or ~$0.001/call via HuggingFace Inference API
# ============================================================================

# Top-level NAICS industries for classification
NAICS_INDUSTRIES = [
    "Agriculture, Forestry, Fishing and Hunting",
    "Mining, Quarrying, and Oil and Gas Extraction",
    "Utilities",
    "Construction",
    "Manufacturing",
    "Wholesale Trade",
    "Retail Trade",
    "Transportation and Warehousing",
    "Information Technology",
    "Finance and Insurance",
    "Real Estate and Rental and Leasing",
    "Professional, Scientific, and Technical Services",
    "Management of Companies and Enterprises",
    "Administrative and Support Services",
    "Educational Services",
    "Health Care and Social Assistance",
    "Arts, Entertainment, and Recreation",
    "Accommodation and Food Services",
    "Other Services",
    "Public Administration"
]

# Try to import transformers for local classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers library not installed. Run: pip install transformers torch")

# Global classifier instance (lazy loaded)
_classifier = None

def get_classifier():
    """Lazy load the zero-shot classifier to avoid startup delay."""
    global _classifier
    if _classifier is None and TRANSFORMERS_AVAILABLE:
        print("🔄 Loading zero-shot classifier (first time only)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU (use 0 for GPU if available)
        )
        print("✅ Zero-shot classifier loaded")
    return _classifier


async def classify_industry_local(company_description: str, provided_industry: str = None) -> Tuple[str, float, dict]:
    """
    Classify company industry using local zero-shot classification.
    
    Args:
        company_description: Text describing what the company does
        provided_industry: Industry claimed by the miner (optional, for comparison)
    
    Returns:
        (predicted_industry, confidence, metadata)
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, 0.0, {"error": "transformers library not installed"}
    
    if not company_description or len(company_description.strip()) < 10:
        return None, 0.0, {"error": "Company description too short"}
    
    try:
        classifier = get_classifier()
        if classifier is None:
            return None, 0.0, {"error": "Classifier not available"}
        
        # Run classification in thread to avoid blocking
        def classify():
            result = classifier(
                company_description[:1000],  # Limit input length
                candidate_labels=NAICS_INDUSTRIES,
                multi_label=False
            )
            return result
        
        result = await asyncio.to_thread(classify)
        
        predicted_industry = result["labels"][0]
        confidence = result["scores"][0]
        
        # Check if miner's claimed industry matches
        industry_match = None
        if provided_industry:
            # Normalize both for comparison
            provided_lower = provided_industry.lower().strip()
            predicted_lower = predicted_industry.lower().strip()
            
            # Exact or partial match
            industry_match = (
                provided_lower == predicted_lower or
                provided_lower in predicted_lower or
                predicted_lower in provided_lower
            )
        
        return predicted_industry, confidence, {
            "checked": True,
            "predicted_industry": predicted_industry,
            "confidence": confidence,
            "top_3_industries": list(zip(result["labels"][:3], result["scores"][:3])),
            "provided_industry": provided_industry,
            "industry_match": industry_match
        }
        
    except Exception as e:
        return None, 0.0, {"error": f"Classification failed: {str(e)}"}


async def verify_industry(lead: dict) -> Tuple[bool, float, dict]:
    """
    Verify lead's industry classification.
    
    This is a SOFT check - it doesn't reject leads, but provides a confidence score
    that can be used for reputation scoring.
    
    Args:
        lead: Lead data with industry, sub_industry, and optionally company_description
    
    Returns:
        (is_verified, confidence, metadata)
    """
    provided_industry = lead.get("industry") or lead.get("Industry")
    company_description = lead.get("company_description") or lead.get("description") or ""
    company_name = lead.get("company") or lead.get("Company") or ""
    
    if not provided_industry:
        return False, 0.0, {"error": "No industry provided in lead"}
    
    # If we have a company description, use it for classification
    if company_description and len(company_description) > 20:
        predicted, confidence, metadata = await classify_industry_local(
            company_description, 
            provided_industry
        )
        
        if predicted:
            is_match = metadata.get("industry_match", False)
            return is_match, confidence, metadata
    
    # If no description available, we can't verify - return neutral
    return True, 0.5, {
        "checked": False,
        "reason": "No company description available for verification",
        "provided_industry": provided_industry
    }


# ============================================================================
# 2. ROLE VERIFICATION
# ============================================================================
# Method: People enrichment API (Apollo.io, Hunter.io)
# Cost: $0.003-0.01 per lookup
# Strategy: Sample 20% of leads to keep costs low
# ============================================================================

async def verify_role_apollo(email: str, expected_role: str, expected_company: str) -> Tuple[bool, dict]:
    """
    Verify job role using Apollo.io People Enrichment API.
    
    Apollo API: https://docs.apollo.io/api/enrich
    
    Args:
        email: Contact's email address
        expected_role: Role claimed by miner (e.g., "CEO", "VP of Sales")
        expected_company: Company claimed by miner
    
    Returns:
        (is_verified, metadata)
    """
    if not APOLLO_API_KEY:
        return False, {"error": "Apollo API key not configured", "checked": False}
    
    if not email:
        return False, {"error": "No email provided", "checked": False}
    
    try:
        url = "https://api.apollo.io/v1/people/match"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache"
        }
        payload = {
            "api_key": APOLLO_API_KEY,
            "email": email
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=15) as response:
                if response.status == 401:
                    return False, {"error": "Invalid Apollo API key", "checked": False}
                if response.status == 402:
                    return False, {"error": "Apollo API credits exhausted", "checked": False}
                if response.status != 200:
                    return False, {"error": f"Apollo API error: HTTP {response.status}", "checked": False}
                
                data = await response.json()
                person = data.get("person", {})
                
                if not person:
                    return False, {
                        "checked": True,
                        "found": False,
                        "reason": "Person not found in Apollo database"
                    }
                
                # Extract Apollo's data
                apollo_title = person.get("title", "")
                apollo_company = person.get("organization", {}).get("name", "")
                
                # Normalize for comparison
                expected_role_lower = expected_role.lower().strip() if expected_role else ""
                apollo_title_lower = apollo_title.lower().strip() if apollo_title else ""
                expected_company_lower = expected_company.lower().strip() if expected_company else ""
                apollo_company_lower = apollo_company.lower().strip() if apollo_company else ""
                
                # Check role match (partial match allowed)
                role_match = (
                    expected_role_lower in apollo_title_lower or
                    apollo_title_lower in expected_role_lower or
                    # Common abbreviations
                    (expected_role_lower == "ceo" and "chief executive" in apollo_title_lower) or
                    (expected_role_lower == "cto" and "chief technology" in apollo_title_lower) or
                    (expected_role_lower == "cfo" and "chief financial" in apollo_title_lower) or
                    (expected_role_lower == "vp" and "vice president" in apollo_title_lower)
                )
                
                # Check company match
                company_match = (
                    expected_company_lower in apollo_company_lower or
                    apollo_company_lower in expected_company_lower
                )
                
                is_verified = role_match and company_match
                
                return is_verified, {
                    "checked": True,
                    "found": True,
                    "apollo_title": apollo_title,
                    "apollo_company": apollo_company,
                    "expected_role": expected_role,
                    "expected_company": expected_company,
                    "role_match": role_match,
                    "company_match": company_match,
                    "is_verified": is_verified
                }
                
    except asyncio.TimeoutError:
        return False, {"error": "Apollo API timeout", "checked": False}
    except Exception as e:
        return False, {"error": f"Apollo API error: {str(e)}", "checked": False}


async def verify_role_hunter(email: str, expected_role: str, expected_company: str) -> Tuple[bool, dict]:
    """
    Verify job role using Hunter.io Email Finder API.
    
    Hunter API: https://hunter.io/api-documentation/v2#email-finder
    
    Args:
        email: Contact's email address
        expected_role: Role claimed by miner
        expected_company: Company claimed by miner
    
    Returns:
        (is_verified, metadata)
    """
    if not HUNTER_API_KEY:
        return False, {"error": "Hunter API key not configured", "checked": False}
    
    if not email:
        return False, {"error": "No email provided", "checked": False}
    
    try:
        # Hunter's email verifier endpoint
        url = "https://api.hunter.io/v2/email-verifier"
        params = {
            "email": email,
            "api_key": HUNTER_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 401:
                    return False, {"error": "Invalid Hunter API key", "checked": False}
                if response.status == 402:
                    return False, {"error": "Hunter API credits exhausted", "checked": False}
                if response.status != 200:
                    return False, {"error": f"Hunter API error: HTTP {response.status}", "checked": False}
                
                data = await response.json()
                result = data.get("data", {})
                
                # Hunter returns email verification status
                status = result.get("status", "unknown")
                
                # Note: Hunter's email verifier doesn't return job title
                # For full enrichment, you'd need Hunter's Email Finder or Domain Search
                
                return True, {
                    "checked": True,
                    "email_status": status,
                    "reason": "Hunter only verifies email, not job title"
                }
                
    except asyncio.TimeoutError:
        return False, {"error": "Hunter API timeout", "checked": False}
    except Exception as e:
        return False, {"error": f"Hunter API error: {str(e)}", "checked": False}


async def verify_role(lead: dict, use_apollo: bool = True) -> Tuple[bool, dict]:
    """
    Verify lead's job role using enrichment API.
    
    Args:
        lead: Lead data with email, role, company
        use_apollo: If True, use Apollo.io; if False, use Hunter.io
    
    Returns:
        (is_verified, metadata)
    """
    email = lead.get("email") or lead.get("Email")
    role = lead.get("role") or lead.get("Role") or lead.get("title") or lead.get("Title")
    company = lead.get("company") or lead.get("Company")
    
    if not email:
        return False, {"error": "No email in lead"}
    
    if use_apollo and APOLLO_API_KEY:
        return await verify_role_apollo(email, role, company)
    elif HUNTER_API_KEY:
        return await verify_role_hunter(email, role, company)
    else:
        return False, {
            "error": "No enrichment API configured",
            "checked": False,
            "hint": "Set APOLLO_API_KEY or HUNTER_API_KEY in .env"
        }


# ============================================================================
# 3. REGION/LOCATION VERIFICATION
# ============================================================================
# Method: DDG search for company headquarters/location
# Cost: FREE (uses DuckDuckGo search)
# ============================================================================

# Common location patterns to extract from text
LOCATION_PATTERNS = [
    # "headquartered in City, State"
    r'headquarter(?:ed|s)?\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    # "based in City, State"  
    r'based\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    # "located in City, State"
    r'located\s+in\s+([^,\.]+(?:,\s*[^,\.]+)?)',
    # "City, State - Company"
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\s*[-–—]',
    # "Company | City, State"
    r'\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,})',
]

# US State abbreviations
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
}

# State name to abbreviation mapping
STATE_ABBREV = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
}


def normalize_location(location: str) -> dict:
    """
    Normalize a location string into components.
    
    Args:
        location: Raw location string (e.g., "San Francisco, CA", "New York, United States")
    
    Returns:
        {
            "city": str or None,
            "state": str or None (2-letter abbrev),
            "country": str or None,
            "raw": original string
        }
    """
    if not location:
        return {"city": None, "state": None, "country": None, "raw": ""}
    
    location = location.strip()
    result = {"city": None, "state": None, "country": None, "raw": location}
    
    # Split by comma
    parts = [p.strip() for p in location.split(',')]
    
    if len(parts) >= 1:
        result["city"] = parts[0]
    
    if len(parts) >= 2:
        state_or_country = parts[1].strip().upper()
        
        # Check if it's a US state abbreviation
        if state_or_country in US_STATES:
            result["state"] = state_or_country
            result["country"] = "US"
        # Check if it's a full state name
        elif parts[1].strip().lower() in STATE_ABBREV:
            result["state"] = STATE_ABBREV[parts[1].strip().lower()]
            result["country"] = "US"
        else:
            # Assume it's a country
            result["country"] = parts[1].strip()
    
    if len(parts) >= 3:
        result["country"] = parts[2].strip()
    
    return result


def extract_location_from_text(text: str) -> Optional[str]:
    """
    Extract location from text using regex patterns.
    
    Args:
        text: Text to search for location (title, snippet, etc.)
    
    Returns:
        Extracted location string or None
    """
    if not text:
        return None
    
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Clean up common suffixes
            location = re.sub(r'\s*\|.*$', '', location)
            location = re.sub(r'\s*-.*$', '', location)
            return location
    
    return None


def locations_match(loc1: str, loc2: str, fuzzy: bool = True) -> Tuple[bool, float, dict]:
    """
    Check if two locations match.
    
    Args:
        loc1: First location (miner's claimed region)
        loc2: Second location (extracted from search)
        fuzzy: Allow partial matching
    
    Returns:
        (is_match, confidence, details)
    """
    if not loc1 or not loc2:
        return False, 0.0, {"error": "Missing location"}
    
    norm1 = normalize_location(loc1)
    norm2 = normalize_location(loc2)
    
    details = {
        "claimed": norm1,
        "extracted": norm2
    }
    
    # Exact city match
    city_match = False
    if norm1["city"] and norm2["city"]:
        city1 = norm1["city"].lower().strip()
        city2 = norm2["city"].lower().strip()
        city_match = city1 == city2 or city1 in city2 or city2 in city1
    
    # State match
    state_match = False
    if norm1["state"] and norm2["state"]:
        state_match = norm1["state"] == norm2["state"]
    
    # Country match
    country_match = False
    if norm1["country"] and norm2["country"]:
        c1 = norm1["country"].lower().strip()
        c2 = norm2["country"].lower().strip()
        # Normalize common variations
        if c1 in ["us", "usa", "united states", "united states of america"]:
            c1 = "us"
        if c2 in ["us", "usa", "united states", "united states of america"]:
            c2 = "us"
        country_match = c1 == c2
    
    details["city_match"] = city_match
    details["state_match"] = state_match
    details["country_match"] = country_match
    
    # Calculate confidence - require at least city OR state match for True
    if city_match and state_match:
        return True, 1.0, details
    elif city_match and country_match:
        return True, 0.95, details
    elif state_match and country_match:
        return True, 0.8, details
    elif city_match:
        return True, 0.7, details
    elif state_match:
        # State match only (e.g., both in CA) - partial match
        return fuzzy, 0.6, details
    elif country_match:
        # Country-only match is NOT sufficient (too broad)
        # e.g., "New York" vs "Seattle" both in US - NOT a match
        return False, 0.3, details
    
    return False, 0.0, details


async def ddg_search_location(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo for location information.
    
    Uses the same ddgs library as automated_checks.py
    Uses Yahoo backend for 100% consistent results.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("⚠️ duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []
    
    try:
        with DDGS() as ddgs:
            # Use Yahoo backend for consistency (100% vs DDG's ~35%)
            # Fallback to Bing if Yahoo fails
            for backend in ['yahoo', 'bing']:
                try:
                    results = list(ddgs.text(query, max_results=max_results, backend=backend))
                    return [
                        {
                            "title": r.get("title", ""),
                            "link": r.get("href", ""),
                            "snippet": r.get("body", "")
                        }
                        for r in results
                    ]
                except Exception:
                    if backend == 'yahoo':
                        continue  # Try Bing
                    raise
    except Exception as e:
        print(f"⚠️ DDG search failed: {e}")
        return []


async def verify_region(
    company: str,
    claimed_region: str,
    use_llm: bool = True  # Default to True - most reliable
) -> Tuple[bool, str, dict]:
    """
    Verify company region/location using LLM knowledge + DDG search.
    
    Strategy:
    1. First, ask LLM directly about well-known company HQ (uses training data)
    2. If LLM says UNKNOWN, fall back to DDG search + LLM extraction
    
    Args:
        company: Company name to search
        claimed_region: Region claimed by miner (city, state, country)
        use_llm: Whether to use LLM (recommended - most reliable)
    
    Returns:
        (is_verified, extracted_location, metadata)
    """
    if not company:
        return False, None, {"error": "No company name provided", "checked": False}
    
    if not claimed_region:
        return False, None, {"error": "No claimed region provided", "checked": False}
    
    extracted_locations = []
    
    # Step 1: Ask LLM directly (uses its knowledge of well-known companies)
    if use_llm:
        extracted = await _llm_get_company_hq(company)
        if extracted and extracted.upper() != "UNKNOWN":
            extracted_locations.append({"location": extracted, "source": "llm_knowledge"})
    
    # Step 2: If LLM doesn't know, try DDG search
    if not extracted_locations:
        queries = [
            f'{company} headquarters location city',
            f'{company} corporate office address',
            f'{company} wikipedia'
        ]
        
        all_results = []
        for query in queries:
            results = await ddg_search_location(query, max_results=3)
            all_results.extend(results)
            if len(all_results) >= 5:
                break
        
        if all_results:
            # Try regex extraction
            for result in all_results:
                loc = extract_location_from_text(result.get("title", ""))
                if loc:
                    extracted_locations.append({"location": loc, "source": "ddg_title"})
                
                loc = extract_location_from_text(result.get("snippet", ""))
                if loc:
                    extracted_locations.append({"location": loc, "source": "ddg_snippet"})
            
            # Try LLM extraction from search results
            if use_llm and not extracted_locations:
                extracted = await _llm_extract_location(company, all_results)
                if extracted and extracted.upper() != "UNKNOWN":
                    extracted_locations.append({"location": extracted, "source": "llm_search"})
    
    if not extracted_locations:
        return False, None, {
            "checked": True,
            "error": "Could not determine company headquarters location",
            "claimed_region": claimed_region
        }
    
    # Compare with claimed region
    best_match = None
    best_confidence = 0.0
    
    for extracted in extracted_locations:
        is_match, confidence, details = locations_match(
            claimed_region, 
            extracted["location"]
        )
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = {
                "is_match": is_match,
                "confidence": confidence,
                "extracted_location": extracted["location"],
                "source": extracted["source"],
                "details": details
            }
    
    if best_match:
        return best_match["is_match"], best_match["extracted_location"], {
            "checked": True,
            "claimed_region": claimed_region,
            **best_match
        }
    
    return False, None, {
        "checked": True,
        "error": "No matching location found",
        "claimed_region": claimed_region,
        "extracted_locations": extracted_locations
    }


async def _llm_get_company_hq(company: str) -> Optional[str]:
    """
    Ask LLM directly about company headquarters using its training knowledge.
    
    This works well for well-known companies.
    """
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
    if not OPENROUTER_KEY:
        return None
    
    prompt = f"""What city is the headquarters of {company} located in?

RULES:
- Respond with ONLY the city and state/country, nothing else
- Format: "City, State" for US (e.g., "San Jose, CA") or "City, Country" for international (e.g., "Dublin, Ireland")
- If you're not sure or the company is not well-known, respond with exactly: UNKNOWN

ANSWER:"""
    
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
                    "max_tokens": 20,
                    "temperature": 0
                },
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    answer = answer.replace('"', '').replace("'", "").strip()
                    if answer.upper() not in ["UNKNOWN", "N/A", "NOT FOUND", ""]:
                        return answer
    except Exception as e:
        print(f"⚠️ LLM company HQ query failed: {e}")
    
    return None


async def _llm_extract_location(company: str, search_results: List[Dict]) -> Optional[str]:
    """
    Use LLM to extract company location from search results.
    
    This is the primary extraction method since regex patterns rarely work
    on unstructured DDG results.
    """
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
    if not OPENROUTER_KEY:
        print("⚠️ OPENROUTER_KEY not set - cannot extract location via LLM")
        return None
    
    # Combine search results into context
    context = "\n\n".join([
        f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}"
        for r in search_results[:5]
    ])
    
    prompt = f"""You are a location extraction assistant. Based on the search results below, determine the headquarters/main office location of the company "{company}".

SEARCH RESULTS:
{context}

INSTRUCTIONS:
1. Look for mentions of headquarters, HQ, main office, corporate office, or founded location
2. If multiple locations are mentioned, prefer the headquarters/HQ location
3. Format your response as: City, State (for US companies) or City, Country (for international)

EXAMPLES:
- "San Francisco, CA"
- "Seattle, WA" 
- "London, UK"
- "Dublin, Ireland"

If you cannot determine the location from the search results, respond with exactly: UNKNOWN

LOCATION:"""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",  # Better at extraction
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 30,
                    "temperature": 0
                },
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    # Clean up the response
                    answer = answer.replace('"', '').replace("'", "").strip()
                    if answer.upper() not in ["UNKNOWN", "N/A", "NOT FOUND", ""]:
                        return answer
                else:
                    print(f"⚠️ LLM API error: HTTP {response.status}")
    except Exception as e:
        print(f"⚠️ LLM extraction failed: {e}")
    
    return None


# ============================================================================
# FUZZY MATCHING SYSTEM (Pre-LLM verification)
# ============================================================================
# Tries to match ROLE and REGION before using LLM
# INDUSTRY is always sent to LLM (too subjective for fuzzy matching)
# This reduces LLM hallucinations for deterministic fields
# ============================================================================

# =============================================================================
# C-SUITE ABBREVIATION MAPPINGS
# =============================================================================
# STRICT: These roles are DIFFERENT and must NOT be confused with each other
# CEO ≠ CTO ≠ CFO ≠ COO ≠ CMO ≠ CIO
# =============================================================================
C_SUITE_EXPANSIONS = {
    "ceo": "chief executive officer",
    "cto": "chief technology officer",
    "cfo": "chief financial officer",
    "coo": "chief operating officer",
    "cmo": "chief marketing officer",
    "cio": "chief information officer",  # Can also be Chief Innovation Officer
    "cpo": "chief product officer",
    "cso": "chief strategy officer",  # Can also be Chief Security Officer
    "cro": "chief revenue officer",
    "chro": "chief human resources officer",
    "cdo": "chief data officer",  # Can also be Chief Digital Officer
    "cno": "chief nursing officer",
    "cao": "chief administrative officer",  # Can also be Chief Accounting Officer
}

# =============================================================================
# ROLE ABBREVIATION EXPANSIONS
# =============================================================================
# These are safe substitutions that mean the same thing
# =============================================================================
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
    "pm": "product manager",  # Could also be Project Manager - context dependent
}

# =============================================================================
# ROLE EQUIVALENCIES
# =============================================================================
# These variations are considered THE SAME ROLE
# Example: "Founder" and "Co-Founder" are essentially the same for verification
# =============================================================================
ROLE_EQUIVALENCIES = {
    # Founder variants - all mean "founding member"
    "founder": ["founder", "co-founder", "co founder", "cofounder", "founding member"],
    
    # Owner variants - business ownership (NOT Product Owner!)
    "owner": ["owner", "business owner", "franchise owner", "store owner", "agent owner", "owner operator"],
    
    # President variants
    "president": ["president", "pres", "pres."],
    
    # Partner variants
    "partner": ["partner", "managing partner", "general partner", "senior partner", "equity partner"],
    
    # Board roles
    "board": ["board member", "board director", "director", "board of directors"],
    
    # Chair roles
    "chair": ["chairman", "chairwoman", "chair", "chairperson", "executive chair", "executive chairman"],
}


def extract_role_from_ddg_title(title: str, snippet: str = "") -> Optional[str]:
    """
    Extract job role from DDG LinkedIn search result title/snippet.
    
    LinkedIn title patterns:
    - "Name - Role at Company | LinkedIn"
    - "Name - Role | LinkedIn"
    - "Name - Company | LinkedIn" (no role visible)
    - May also have role info later in concatenated titles: "... | CTO at Company | ..."
    
    Returns:
        Extracted role string or None if no role found
    """
    if not title:
        return None
    
    original_title = title  # Keep full title for pattern searching
    
    # Role keywords for matching
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
    
    # ==========================================================================
    # PRIORITY 1: Extract from FIRST LinkedIn title segment ONLY
    # "Name - Role ... | LinkedIn" or "Name - Role ... - LinkedIn"
    # This is the most reliable - it's the actual LinkedIn profile title
    # ==========================================================================
    # Get just the first title segment (before any other profiles)
    first_segment = title.split('|')[0].strip()
    first_segment = re.sub(r'\s+-\s*LinkedIn.*$', '', first_segment, flags=re.IGNORECASE).strip()
    first_segment = re.sub(r'\s*\.\.\.\s*$', '', first_segment).strip()  # Remove trailing "..."
    
    # Pattern: "Name - Role at Company" or "Name - Role"
    match = re.search(r'^([^-]+)-\s*(.+?)(?:\s+at\s+|\s*$)', first_segment, re.IGNORECASE)
    if match:
        role = match.group(2).strip()
        role = re.sub(r'\s+at\s*$', '', role, flags=re.IGNORECASE).strip()
        if len(role) > 2 and not role.startswith("http") and role.lower() not in ["linkedin", "..."]:
            return role
    
    # ==========================================================================
    # PRIORITY 2: Search for "Role at Company" patterns anywhere
    # This catches cases like: "Name - Company | ... | CTO at Company | ..."
    # ==========================================================================
    role_at_patterns = re.findall(r'(\b(?:' + '|'.join(role_keywords) + r')[^|]*?)\s+at\s+\w', original_title, re.IGNORECASE)
    if role_at_patterns:
        role = role_at_patterns[0].strip()
        if len(role) > 2:
            return role
    
    # ==========================================================================
    # PRIORITY 3: Search for keyword-based roles anywhere
    # ==========================================================================
    for kw in role_keywords:
        match = re.search(rf'\b({kw}[^|,]*?)\s+at\s+', original_title, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if len(role) > 2:
                return role
    
    # Pattern 4: Check snippet for role mentions
    if snippet:
        snippet_patterns = [
            r'(?:current|position|title|role)[:\s]+([^|.\n]+)',
            r'(?:works? as|serving as|currently)[:\s]+([^|.\n]+)',
        ]
        for pattern in snippet_patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    return None


def fuzzy_match_role(claimed_role: str, extracted_role: str) -> Tuple[bool, float, str]:
    """
    Fuzzy match two roles with STRICT rules to prevent false positives.
    
    Returns:
        (is_match: bool, confidence: float, reason: str)
    
    MATCHING RULES:
    ===============
    
    SHOULD MATCH (same role, different format):
    - "CEO" = "Chief Executive Officer" (abbreviation)
    - "CTO" = "Chief Technology Officer" (abbreviation)
    - "Founder" = "Co-Founder" (both are founders)
    - "VP of Sales" = "Vice President of Sales" (abbreviation)
    - "CEO & Co-Founder" = "Co-Founder and CEO" (same roles, word order)
    - "Product Owner" = "Sr. Product Owner" (seniority prefix)
    - "Software Engineer" = "Senior Software Engineer" (seniority prefix)
    
    SHOULD NOT MATCH (different roles):
    - "CEO" ≠ "CTO" (different C-suite)
    - "CEO" ≠ "CFO" (different C-suite)
    - "COO" ≠ "CIO" (different C-suite)
    - "Owner" ≠ "Product Owner" (business vs tech role)
    - "VP of Sales" ≠ "VP of Engineering" (different departments)
    - "Director of Marketing" ≠ "Director of Finance" (different departments)
    """
    if not claimed_role or not extracted_role:
        return False, 0.0, "Missing role data"
    
    claimed_lower = claimed_role.lower().strip()
    extracted_lower = extracted_role.lower().strip()
    
    # =========================================================================
    # STEP 1: EXACT MATCH
    # =========================================================================
    if claimed_lower == extracted_lower:
        return True, 1.0, "Exact match"
    
    # =========================================================================
    # STEP 2: NORMALIZE (remove punctuation, standardize spacing)
    # =========================================================================
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
    
    # =========================================================================
    # STEP 3: EXPAND ABBREVIATIONS
    # =========================================================================
    def expand_abbreviations(r: str) -> str:
        r = normalize(r)
        # C-suite expansions
        for abbrev, full in C_SUITE_EXPANSIONS.items():
            r = re.sub(rf'\b{abbrev}\b', full, r)
        # Other abbreviations
        for abbrev, full in ROLE_ABBREVIATIONS.items():
            r = re.sub(rf'\b{re.escape(abbrev)}\b', full, r)
        return r
    
    exp_claimed = expand_abbreviations(claimed_role)
    exp_extracted = expand_abbreviations(extracted_role)
    
    if exp_claimed == exp_extracted:
        return True, 1.0, "Abbreviation expansion match"
    
    # =========================================================================
    # STEP 4: C-SUITE STRICT MATCHING
    # C-suite roles are DIFFERENT and must NOT be confused!
    # CEO ≠ CTO ≠ CFO ≠ COO ≠ CMO ≠ CIO
    # =========================================================================
    def get_c_suite_type(role: str) -> Optional[str]:
        """Extract C-suite type from role string. Returns abbreviation."""
        role_lower = role.lower()
        for abbrev, full in C_SUITE_EXPANSIONS.items():
            # Check both abbreviation and full form
            if re.search(rf'\b{abbrev}\b', role_lower) or full in role_lower:
                return abbrev
        return None
    
    claimed_csuite = get_c_suite_type(claimed_role)
    extracted_csuite = get_c_suite_type(extracted_role)
    
    # If BOTH have C-suite roles, they MUST be the same type
    if claimed_csuite and extracted_csuite:
        if claimed_csuite != extracted_csuite:
            return False, 0.0, f"C-Suite MISMATCH: {claimed_csuite.upper()} ≠ {extracted_csuite.upper()}"
        # Same C-suite type - continue to check for compound roles
    
    # If only ONE has C-suite, they don't match (unless containment later)
    if (claimed_csuite and not extracted_csuite) or (extracted_csuite and not claimed_csuite):
        # One has C-suite, other doesn't - check if it's a compound role like "CEO & Founder"
        pass  # Will be handled by containment check later
    
    # =========================================================================
    # STEP 5: OWNER vs PRODUCT OWNER (STRICT - CRITICAL!)
    # "Owner" (business owner) ≠ "Product Owner" (tech/PM role)
    # =========================================================================
    def is_business_owner(r: str) -> bool:
        r_lower = r.lower()
        return "owner" in r_lower and "product owner" not in r_lower and "product" not in r_lower.split("owner")[0]
    
    def is_product_owner(r: str) -> bool:
        return "product owner" in r.lower()
    
    if is_business_owner(claimed_role) and is_product_owner(extracted_role):
        return False, 0.0, "MISMATCH: Owner (business) ≠ Product Owner (tech role)"
    if is_product_owner(claimed_role) and is_business_owner(extracted_role):
        return False, 0.0, "MISMATCH: Product Owner (tech role) ≠ Owner (business)"
    
    # =========================================================================
    # STEP 6: DEPARTMENT CHECK
    # "VP of Sales" ≠ "VP of Engineering" (different departments!)
    # =========================================================================
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
    
    # If both have departments and they're different, no match
    if claimed_dept and extracted_dept and claimed_dept != extracted_dept:
        return False, 0.0, f"DEPARTMENT MISMATCH: {claimed_dept} ≠ {extracted_dept}"
    
    # =========================================================================
    # STEP 7: FOUNDER MATCHING (lenient - founder = co-founder)
    # =========================================================================
    def has_founder(r: str) -> bool:
        r_lower = r.lower()
        return any(f in r_lower for f in ["founder", "co-founder", "cofounder", "co founder"])
    
    if has_founder(claimed_role) and has_founder(extracted_role):
        # Both have founder - check if C-suite also matches (for "CEO & Co-Founder" cases)
        if claimed_csuite and extracted_csuite and claimed_csuite == extracted_csuite:
            return True, 1.0, "Founder + matching C-suite"
        elif not claimed_csuite and not extracted_csuite:
            return True, 0.95, "Both are founders"
        # One has C-suite, other doesn't - still a founder match
        return True, 0.85, "Founder match (one has additional C-suite role)"
    
    # =========================================================================
    # STEP 8: OWNER MATCHING (for business owners)
    # =========================================================================
    if is_business_owner(claimed_role) and is_business_owner(extracted_role):
        return True, 0.95, "Both are business owners"
    
    # =========================================================================
    # STEP 9: PRODUCT OWNER MATCHING
    # =========================================================================
    if is_product_owner(claimed_role) and is_product_owner(extracted_role):
        return True, 0.95, "Both are Product Owners"
    
    # =========================================================================
    # STEP 10: CONTAINMENT CHECK (with expanded abbreviations)
    # "CEO" matches "CEO and Co-Founder" (CEO is contained)
    # "President" matches "President & CEO" (President is contained)
    # =========================================================================
    if exp_claimed in exp_extracted:
        return True, 0.9, f"Claimed role contained in extracted: '{claimed_role}' in '{extracted_role}'"
    if exp_extracted in exp_claimed:
        return True, 0.9, f"Extracted role contained in claimed: '{extracted_role}' in '{claimed_role}'"
    
    # =========================================================================
    # STEP 11: WORD OVERLAP (for complex roles)
    # =========================================================================
    def get_meaningful_words(r: str) -> set:
        r = normalize(r)
        # Expand abbreviations first
        r = expand_abbreviations(r)
        words = set(r.split())
        # Remove filler words
        filler = {"at", "of", "the", "and", "for", "in", "a", "an", "to", "&", "or"}
        return words - filler
    
    claimed_words = get_meaningful_words(claimed_role)
    extracted_words = get_meaningful_words(extracted_role)
    
    if claimed_words and extracted_words:
        intersection = claimed_words & extracted_words
        union = claimed_words | extracted_words
        jaccard = len(intersection) / len(union) if union else 0
        
        # Require significant overlap
        if jaccard >= 0.6:
            return True, jaccard, f"Word overlap: {jaccard:.0%} - common words: {intersection}"
    
    # =========================================================================
    # STEP 12: EQUIVALENCY CHECK
    # =========================================================================
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
    
    # =========================================================================
    # NO MATCH
    # =========================================================================
    jaccard = len(claimed_words & extracted_words) / len(claimed_words | extracted_words) if (claimed_words | extracted_words) else 0
    return False, jaccard, f"No match (word similarity: {jaccard:.0%})"


def fuzzy_pre_verification(
    claimed_role: str,
    claimed_region: str,
    claimed_industry: str,
    ddg_role_results: List[Dict],
    ddg_region_results: List[Dict],
    ddg_industry_results: List[Dict],
    full_name: str = "",
    company: str = ""
) -> Dict:
    """
    Pre-verify ROLE and REGION using fuzzy matching BEFORE sending to LLM.
    
    INDUSTRY is ALWAYS sent to LLM because:
    - Industry classifications are subjective
    - DDG results may be about different companies
    - Too many edge cases for deterministic matching
    
    Returns:
        {
            "role_verified": bool,  # True if fuzzy match succeeded
            "role_extracted": str,  # Extracted role from DDG
            "role_confidence": float,
            "role_reason": str,
            
            "region_verified": bool,
            "region_extracted": str,
            "region_confidence": float,
            "region_reason": str,
            
            "industry_verified": bool,  # Always False (LLM handles this)
            "industry_extracted": str,
            "industry_confidence": float,
            "industry_reason": str,
            
            "needs_llm": list,  # Fields that still need LLM verification
        }
    """
    result = {
        "role_verified": False,
        "role_extracted": None,
        "role_confidence": 0.0,
        "role_reason": "Not checked",
        
        "region_verified": False,
        "region_extracted": None,
        "region_confidence": 0.0,
        "region_reason": "Not checked",
        
        # Industry is ALWAYS sent to LLM
        "industry_verified": False,
        "industry_extracted": None,
        "industry_confidence": 0.0,
        "industry_reason": "Industry always verified by LLM (too subjective for fuzzy match)",
        
        "needs_llm": ["industry"],  # Industry always needs LLM
    }
    
    # =========================================================================
    # ROLE FUZZY MATCHING
    # =========================================================================
    # We try to match roles deterministically because:
    # - LinkedIn titles are structured ("Name - Role at Company")
    # - Role abbreviations are well-defined (CEO = Chief Executive Officer)
    # - C-suite roles have strict boundaries (CEO ≠ CTO ≠ COO)
    # =========================================================================
    if ddg_role_results and claimed_role:
        name_lower = full_name.lower() if full_name else ""
        first_name = name_lower.split()[0] if name_lower else ""
        last_name = name_lower.split()[-1] if name_lower else ""
        
        best_extracted_role = None
        best_match = False
        best_confidence = 0.0
        best_reason = "No role found in DDG results"
        
        for r in ddg_role_results[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            
            # FILTER: Only use results about the RIGHT person
            # Use word boundary matching to avoid "Don" matching "Donald"
            title_lower = title.lower()
            if first_name and last_name:
                # Use word boundaries: " don " or "don " at start or " don" at end
                import re
                first_pattern = rf'\b{re.escape(first_name)}\b'
                last_pattern = rf'\b{re.escape(last_name)}\b'
                if not (re.search(first_pattern, title_lower) and re.search(last_pattern, title_lower)):
                    continue  # Skip results about different people (e.g., Don ≠ Donald)
            
            # Extract role from LinkedIn title
            extracted = extract_role_from_ddg_title(title, snippet)
            
            if extracted:
                is_match, confidence, reason = fuzzy_match_role(claimed_role, extracted)
                
                # Keep track of best match (even if not matching)
                if confidence > best_confidence or (not best_extracted_role and extracted):
                    best_extracted_role = extracted
                    best_match = is_match
                    best_confidence = confidence
                    best_reason = reason
        
        if best_extracted_role:
            result["role_extracted"] = best_extracted_role
            result["role_confidence"] = best_confidence
            result["role_reason"] = best_reason
            
            # Only accept fuzzy match if confidence is high enough
            if best_match and best_confidence >= 0.8:
                result["role_verified"] = True
                print(f"   ✅ FUZZY ROLE MATCH: '{claimed_role}' ≈ '{best_extracted_role}'")
                print(f"      Confidence: {best_confidence:.0%} | Reason: {best_reason}")
            else:
                result["needs_llm"].append("role")
                if best_match:
                    print(f"   ⚠️ FUZZY ROLE: Low confidence match ({best_confidence:.0%}), sending to LLM")
                else:
                    print(f"   ❌ FUZZY ROLE MISMATCH: '{claimed_role}' ≠ '{best_extracted_role}'")
                    print(f"      Reason: {best_reason}")
        else:
            result["needs_llm"].append("role")
            result["role_reason"] = "Could not extract role from DDG results"
            print(f"   ⚠️ FUZZY ROLE: Could not extract role from DDG, sending to LLM")
    else:
        result["needs_llm"].append("role")
        print(f"   ⚠️ FUZZY ROLE: No DDG results or no claimed role")
    
    # =========================================================================
    # REGION FUZZY MATCHING (using GeoPy)
    # =========================================================================
    # GeoPy provides deterministic geographic matching:
    # - Real geographic coordinates
    # - Distance calculations
    # - State/country matching
    # =========================================================================
    if ddg_region_results and claimed_region:
        # Try to extract location from DDG results
        # FILTER: Only use results that are clearly about the company
        company_lower = company.lower() if company else ""
        extracted_region = None
        
        for r in ddg_region_results[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            combined = title + " " + snippet
            
            # Check if this result is about the right company
            if company_lower and company_lower not in combined.lower():
                continue  # Skip results about different companies
            
            # Try to extract location
            loc = extract_location_from_text(combined)
            if loc:
                extracted_region = loc
                break
        
        if extracted_region:
            # Use GeoPy for deterministic matching
            geo_match, geo_reason = locations_match_geopy(claimed_region, extracted_region)
            
            result["region_extracted"] = extracted_region
            result["region_confidence"] = 0.95 if geo_match else 0.3
            result["region_reason"] = geo_reason
            
            if geo_match:
                result["region_verified"] = True
                print(f"   ✅ FUZZY REGION MATCH: '{claimed_region}' ≈ '{extracted_region}'")
                print(f"      Reason: {geo_reason}")
            else:
                result["needs_llm"].append("region")
                print(f"   ⚠️ FUZZY REGION: GeoPy says no match, sending to LLM for verification")
                print(f"      Claimed: {claimed_region} | Extracted: {extracted_region}")
        else:
            result["needs_llm"].append("region")
            result["region_reason"] = "Could not extract region from DDG results"
            print(f"   ⚠️ FUZZY REGION: Could not extract location, sending to LLM")
    else:
        result["needs_llm"].append("region")
        print(f"   ⚠️ FUZZY REGION: No DDG results or no claimed region")
    
    # =========================================================================
    # INDUSTRY: ALWAYS SENT TO LLM
    # =========================================================================
    # We do NOT fuzzy match industry because:
    # 1. Industry classifications are subjective (is "Fintech" = "Financial Services"?)
    # 2. DDG results may be about different companies with same name
    # 3. LLM can understand context better (e.g., "Cannabis Testing Lab" → Healthcare)
    # =========================================================================
    print(f"   🤖 INDUSTRY: Always verified by LLM (too subjective for fuzzy match)")
    
    return result


# ============================================================================
# STAGE 5: UNIFIED VERIFICATION (SINGLE LLM CALL)
# ============================================================================
# Verifies role, region, and industry/sub_industry in ONE LLM call
# Uses DDG search results as context (from Stage 4)
# Cost: Same as Stage 4 (single OpenRouter call)
# ============================================================================

def _ddg_search_stage5_sync(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """
    Sync helper for Stage 5 DDG searches.
    Called via asyncio.to_thread from async code.
    
    Args:
        search_type: "role", "region", or "industry"
        full_name: Person's name (for role search)
        company: Company name
        role: Claimed role (for role search)
        max_results: Max results per query
    """
    import time
    
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    
    if search_type == "role":
        # =======================================================================
        # ROLE SEARCH ORDER (Stage 4 already validated LinkedIn URL is real):
        # 1. PRIMARY: Search LinkedIn URL directly - most reliable since Stage 4
        #             already confirmed the profile exists
        # 2. FALLBACK: General web search (name + company + role) if LinkedIn fails
        # 3. FALLBACK2: Simpler queries (name + company) if still no results
        # =======================================================================
        
        linkedin_url = kwargs.get("linkedin_url", "")
        
        # PRIMARY: LinkedIn URL search - Stage 4 already validated this URL
        # For common names at large companies, this is MUCH more reliable
        linkedin_primary_query = None
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            profile_slug = linkedin_url.split("/in/")[-1].strip("/")
            linkedin_primary_query = f'linkedin.com/in/{profile_slug}'
        
        # Set queries and fallbacks - LinkedIn is PRIMARY, general DDG is FALLBACK
        import re
        role_simplified = re.split(r'[,&/]', role)[0].strip() if role else ""
        
        # These are the FALLBACK queries if LinkedIn search fails
        queries = []  # Start empty - LinkedIn will be tried first
        fallback_queries = [
            f'"{full_name}" {company} {role_simplified}',  # General web search
            f'"{full_name}" "{company}"',  # Exact match both - most precise
            f'"{full_name}" {company}',  # Simpler fallback
        ]
        # No longer used - LinkedIn is now PRIMARY
        linkedin_fallback_query = None
    elif search_type == "region":
        # Search for company headquarters location
        # Include any known region hint to disambiguate companies with common names
        region_hint = kwargs.get("region_hint", "")
        if region_hint:
            queries = [
                f'{company} {region_hint} headquarters location',
                f'{company} headquarters {region_hint}',
            ]
        else:
            queries = [
                f'{company} headquarters location',
            ]
        fallback_queries = []
    else:  # industry
        # Search for company industry/description
        # Include any known region hint to disambiguate companies with common names
        region_hint = kwargs.get("region_hint", "")
        if region_hint:
            queries = [
                f'{company} {region_hint} company industry',
                f'{company} company industry {region_hint}',
            ]
        else:
            queries = [
                f'{company} company industry',
            ]
        fallback_queries = []
    
    def results_are_relevant(results: list, name: str, company: str) -> bool:
        """Check if DDG results are actually about the person/company.
        
        For LinkedIn URL searches: we searched an EXACT URL so trust the first result.
        For general searches: require BOTH person name AND company to appear.
        """
        if not results:
            return False
        
        name_lower = name.lower()
        company_lower = company.lower()
        first_name = name_lower.split()[0] if name_lower else ""
        last_name = name_lower.split()[-1] if name_lower else ""
        
        for r in results:
            title = r.get("title", "").lower()
            snippet = r.get("snippet", r.get("body", "")).lower()
            link = r.get("link", "").lower()
            combined = title + " " + snippet
            
            # PRIORITY 1: If it's a LinkedIn profile URL, trust it more
            # We searched the exact LinkedIn URL, so if it shows up, it's relevant
            if "linkedin.com/in/" in link:
                # Just need name match for LinkedIn results (we already searched specific URL)
                if (first_name in combined and last_name in combined) or name_lower in combined:
                    return True
            
            # PRIORITY 2: Check if result mentions the person AND company
            has_name = (first_name in combined and last_name in combined) or name_lower in combined
            has_company = company_lower in combined
            
            # Need BOTH name and company in at least one result
            if has_name and has_company:
                return True
        
        # No result had both name AND company
        return False
    
    # =========================================================================
    # BACKEND FALLBACK: Yahoo (100% consistent) → Bing (100% consistent backup)
    # This ensures reliability since we only have 1 query per check in Stage 5
    # =========================================================================
    def ddg_search_with_fallback(ddgs_instance, query, max_results, backends=['yahoo', 'bing']):
        """Try Yahoo first, then Bing if Yahoo fails."""
        last_error = None
        for backend in backends:
            try:
                return list(ddgs_instance.text(query, max_results=max_results, backend=backend))
            except Exception as e:
                last_error = e
                if backend == 'yahoo':
                    print(f"   ⚠️ Yahoo backend failed, trying Bing...")
                continue
        # All backends failed
        raise last_error if last_error else Exception("All backends failed")
    
    all_results = []
    try:
        with DDGS() as ddgs:
            # =======================================================================
            # ROLE SEARCH: LinkedIn URL is PRIMARY (most reliable for common names)
            # Stage 4 already validated the LinkedIn URL, so we can trust it
            # =======================================================================
            if search_type == "role" and linkedin_primary_query:
                print(f"   🔗 PRIMARY: Searching LinkedIn URL directly...")
                try:
                    # Extract expected profile slug to verify we get the RIGHT profile
                    expected_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower() if linkedin_url else ""
                    
                    results = ddg_search_with_fallback(ddgs, linkedin_primary_query, 10)
                    
                    # Filter results by NAME AND URL MATCH
                    # CRITICAL: With common names, multiple profiles exist - we need the RIGHT one!
                    name_lower = full_name.lower()
                    first_name = name_lower.split()[0] if name_lower else ""
                    last_name = name_lower.split()[-1] if name_lower else ""
                    
                    for r in results:
                        title = r.get("title", "").lower()
                        link = r.get("href", "").lower()
                        
                        # LinkedIn titles are: "Name - Role at Company | LinkedIn"
                        has_name = (first_name in title and last_name in title)
                        
                        # CRITICAL: Verify the URL contains the expected profile slug
                        # This prevents matching WRONG person with same name
                        has_correct_url = expected_slug and expected_slug in link
                        
                        if has_name and has_correct_url:
                            # Check if title actually contains role info
                            # LinkedIn titles: "Name - Role at Company | LinkedIn"
                            # Non-role titles: "Name - LinkedIn Schweiz", "Name | LinkedIn"
                            title_has_role = (" at " in title or 
                                             (" - " in title and "linkedin" not in title.split(" - ")[1][:15]))
                            
                            if title_has_role:
                                print(f"   ✅ LinkedIn found EXACT profile with role: {r.get('title', '')[:60]}")
                                all_results = [{
                                    "title": r.get("title", ""),
                                    "link": r.get("href", ""),
                                    "snippet": r.get("body", ""),
                                    "query_type": search_type,
                                    "query": linkedin_primary_query + " (linkedin-primary-exact)"
                                }]
                                break
                            else:
                                # Exact profile found but title doesn't show role
                                # Still use this but also trigger fallback for more context
                                print(f"   ⚠️ LinkedIn EXACT profile but no role in title: {r.get('title', '')[:50]}")
                                all_results = []  # Clear to trigger fallback
                                break
                        elif has_name and not has_correct_url:
                            # Found someone with same name but DIFFERENT profile!
                            result_slug = link.split("/in/")[-1].split("?")[0] if "/in/" in link else "unknown"
                            print(f"   ⚠️ Found DIFFERENT profile: {result_slug[:30]} (expected: {expected_slug[:30]})")
                    
                    if not all_results:
                        print(f"   ⚠️ LinkedIn search: Correct profile {expected_slug[:30]} not in top results")
                except Exception as e:
                    print(f"   ⚠️ LinkedIn primary search failed: {e}")
            
            # Run primary queries (for region/industry, or role if no LinkedIn)
            if not all_results:
                for i, query in enumerate(queries):
                    # Add delay between queries to avoid rate limiting
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
            
            # =======================================================================
            # FALLBACK: For role search, if LinkedIn failed, no relevant results,
            # OR no role could be extracted from LinkedIn results - try general DDG
            # =======================================================================
            if search_type == "role" and fallback_queries:
                # Check if we should try fallback
                should_fallback = False
                fallback_reason = ""
                
                if not results_are_relevant(all_results, full_name, company):
                    should_fallback = True
                    fallback_reason = "LinkedIn results irrelevant/empty"
                else:
                    # LinkedIn results look relevant, but can we extract a role?
                    role_found = False
                    for r in all_results[:3]:
                        title = r.get("title", "")
                        snippet = r.get("snippet", r.get("body", ""))
                        extracted = extract_role_from_ddg_title(title, snippet)
                        if extracted and extracted.lower() not in [company.lower(), "linkedin"]:
                            role_found = True
                            break
                    
                    if not role_found:
                        should_fallback = True
                        fallback_reason = "LinkedIn found but no role visible in title"
                
                if should_fallback:
                    print(f"   ⚠️ {fallback_reason}, trying general DDG search...")
                    # Keep LinkedIn results but ADD fallback results
                    time.sleep(3)  # Delay before fallback
                    
                    for query in fallback_queries:
                        try:
                            results = ddg_search_with_fallback(ddgs, query, max_results)
                            for r in results:
                                all_results.append({
                                    "title": r.get("title", ""),
                                    "link": r.get("href", ""),
                                    "snippet": r.get("body", ""),
                                    "query_type": search_type,
                                    "query": query + " (general-fallback)"
                                })
                            if results_are_relevant(all_results, full_name, company):
                                print(f"   ✅ General DDG fallback found relevant results")
                                break
                            time.sleep(2)
                        except Exception as e:
                            print(f"   ⚠️ DDG fallback query failed: {e}")
                            continue
                            
    except Exception as e:
        print(f"⚠️ DDG {search_type} search failed: {e}")
    
    return all_results[:max_results]


async def _ddg_search_stage5(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """
    Search DDG for Stage 5 verification data.
    
    Runs DDG search in a thread pool to avoid blocking async code.
    
    Args:
        search_type: "role", "region", or "industry"
        full_name: Person's name (for role search)
        company: Company name
        role: Claimed role (for role search)
        max_results: Max results per query
        region_hint: Optional region to help disambiguate common company names
    
    Returns:
        List of search results
    """
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


async def verify_stage5_unified(
    lead: dict,
    ddg_search_results: List[Dict] = None,
    verbose: bool = False
) -> dict:
    """
    Stage 5: Unified verification of role, region, and industry.
    
    Uses DDG searches for REAL-TIME data + a SINGLE LLM call to verify.
    This is called AFTER Stage 4 LinkedIn verification passes.
    
    Flow:
    1. DDG search for company LOCATION (FREE) - finds HQ/office location
    2. DDG search for company INDUSTRY (FREE) - finds what company does
    3. Combine with LinkedIn search results from Stage 4
    4. SINGLE LLM call to verify all 3 fields against DDG data
    
    This works for small/unknown companies because we're using
    real-time DDG search data, NOT relying on LLM's training knowledge.
    
    Args:
        lead: Lead data with fields:
            - full_name: Person's name
            - company/business: Company name
            - role: Claimed job role/title
            - region: Claimed location (city, state, country)
            - industry: Claimed industry
            - sub_industry: Claimed sub-industry (optional)
            - linkedin: LinkedIn URL
        ddg_search_results: Search results from Stage 4 (reused to save API calls)
    
    Returns:
        {
            "verified": bool,  # Overall pass/fail
            "role_match": bool,
            "region_match": bool, 
            "industry_match": bool,
            "extracted_role": str,
            "extracted_region": str,
            "extracted_industry": str,
            "confidence": float,
            "reasoning": str,
            "checked": bool
        }
    """
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
    if not OPENROUTER_KEY:
        return {
            "verified": False,
            "error": "OPENROUTER_KEY not set",
            "checked": False
        }
    
    # Extract lead fields
    full_name = lead.get("full_name") or lead.get("name") or ""
    company = lead.get("company") or lead.get("business") or lead.get("Company") or ""
    claimed_role = lead.get("role") or lead.get("title") or lead.get("Role") or ""
    claimed_region = lead.get("region") or lead.get("location") or lead.get("Region") or ""
    claimed_industry = lead.get("industry") or lead.get("Industry") or ""
    claimed_sub_industry = lead.get("sub_industry") or lead.get("sub-industry") or ""
    linkedin_url = lead.get("linkedin") or lead.get("LinkedIn") or ""
    website = lead.get("website") or lead.get("Website") or ""
    
    if not company:
        return {
            "verified": False,
            "error": "No company name provided",
            "checked": False
        }
    
    # =========================================================================
    # DELAY: Wait 15s after Stage 4 DDG to avoid rate limiting
    # =========================================================================
    print(f"   ⏳ Waiting 3s before Stage 5 DDG searches...")
    await asyncio.sleep(3)
    
    # =========================================================================
    # STEP 1: DDG SEARCH FOR ROLE (using name + company + role)
    # NOTE: When integrated with automated_checks.py, use Stage 4 LinkedIn
    # DDG results FIRST before this search. See comment in _ddg_search_stage5_sync.
    # =========================================================================
    print(f"   🔍 DDG: Searching for {full_name}'s role at {company}...")
    role_results = await _ddg_search_stage5("role", full_name, company, claimed_role, linkedin_url=linkedin_url)
    if role_results:
        print(f"   ✅ Found {len(role_results)} role search results")
    else:
        print(f"   ⚠️ No role results found")
    
    # Wait 15s between Stage 5 DDG searches
    await asyncio.sleep(15)
    
    # =========================================================================
    # STEP 2: DDG SEARCH FOR REGION (company headquarters)
    # Use claimed_region as hint to disambiguate companies with common names
    # =========================================================================
    print(f"   🔍 DDG: Searching for {company} headquarters location...")
    region_results = await _ddg_search_stage5("region", company=company, region_hint=claimed_region)
    if region_results:
        print(f"   ✅ Found {len(region_results)} region search results")
    else:
        print(f"   ⚠️ No region results found")
    
    # Wait 15s between Stage 5 DDG searches
    await asyncio.sleep(15)
    
    # =========================================================================
    # STEP 3: DDG SEARCH FOR INDUSTRY (what the company does)
    # Use claimed_region as hint to disambiguate companies with common names
    # =========================================================================
    print(f"   🔍 DDG: Searching for {company} industry...")
    industry_results = await _ddg_search_stage5("industry", company=company, region_hint=claimed_region)
    if industry_results:
        print(f"   ✅ Found {len(industry_results)} industry search results")
    else:
        print(f"   ⚠️ No industry results found")
    
    # =========================================================================
    # STEP 4: FUZZY PRE-VERIFICATION (Before LLM)
    # =========================================================================
    # Try to match role, region, industry using deterministic fuzzy matching
    # Only send to LLM what fuzzy matching couldn't handle
    # =========================================================================
    print(f"   🔍 FUZZY: Attempting pre-verification before LLM...")
    
    fuzzy_result = fuzzy_pre_verification(
        claimed_role=claimed_role,
        claimed_region=claimed_region,
        claimed_industry=claimed_industry,
        ddg_role_results=role_results,
        ddg_region_results=region_results,
        ddg_industry_results=industry_results,
        full_name=full_name,
        company=company
    )
    
    # Check if all fields were fuzzy-matched (no LLM needed!)
    if not fuzzy_result["needs_llm"]:
        print(f"   ✅ FUZZY: All fields matched - skipping LLM!")
        fuzzy_only_result = {
            "verified": True,
            "role_match": True,
            "extracted_role": fuzzy_result["role_extracted"] or claimed_role,
            "region_match": True,
            "extracted_region": fuzzy_result["region_extracted"] or claimed_region,
            "industry_match": True,
            "extracted_industry": fuzzy_result["industry_extracted"] or claimed_industry,
            "sub_industry_match": True,
            "confidence": min(fuzzy_result["role_confidence"], fuzzy_result["region_confidence"], fuzzy_result["industry_confidence"]),
            "reasoning": f"Fuzzy matched: Role({fuzzy_result['role_reason']}), Region({fuzzy_result['region_reason']}), Industry({fuzzy_result['industry_reason']})",
            "checked": True,
            "fuzzy_matched": True
        }
        if verbose:
            fuzzy_only_result["_debug"] = {
                "llm_prompt": None,  # No LLM used
                "llm_response_raw": None,
                "llm_response_parsed": None,
                "ddg_role_results": role_results,
                "ddg_region_results": region_results,
                "ddg_industry_results": industry_results,
                "fuzzy_result": fuzzy_result,
                "needs_llm": [],
            }
        return fuzzy_only_result
    
    # Some fields need LLM verification
    needs_llm = fuzzy_result["needs_llm"]
    print(f"   🤖 LLM: Need to verify: {needs_llm}")
    
    # =========================================================================
    # STEP 5: BUILD CONTEXT ONLY FOR FIELDS THAT NEED LLM
    # =========================================================================
    
    # Role context (only if needed)
    role_context = ""
    if "role" in needs_llm and role_results:
        role_context = f"ROLE SEARCH RESULTS (searched: '{full_name}' + '{company}' + '{claimed_role}'):\n"
        for i, result in enumerate(role_results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("body", ""))
            role_context += f"{i}. {title}\n   {snippet[:200]}\n"
    
    # Region context (only if needed)
    region_context = ""
    if "region" in needs_llm and region_results:
        region_context = "\nREGION/HEADQUARTERS SEARCH RESULTS:\n"
        for i, result in enumerate(region_results[:4], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            region_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    # Industry context (only if needed)
    industry_context = ""
    if "industry" in needs_llm and industry_results:
        industry_context = "\nINDUSTRY SEARCH RESULTS:\n"
        for i, result in enumerate(industry_results[:4], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            industry_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    # Combine context for fields that need LLM
    all_search_context = role_context + region_context + industry_context
    
    # CRITICAL: If we need to verify ROLE but have no role context, auto-fail
    # We cannot let LLM hallucinate roles from training data!
    if "role" in needs_llm and not role_context.strip():
        print(f"   ❌ AUTO-FAIL: No DDG data for role verification - cannot verify")
        auto_fail_result = {
            'verified': False,
            'role_match': False,
            'region_match': fuzzy_result.get('region_verified', False),
            'industry_match': False,
            'extracted_role': 'Not found - no DDG data',
            'extracted_region': fuzzy_result.get('region_extracted', 'Unknown'),
            'extracted_industry': 'Unknown',
            'reasoning': 'FAILED: No search results found to verify role. Cannot rely on LLM knowledge.',
            'fuzzy_matched': {
                'role': False,
                'region': fuzzy_result.get('region_verified', False),
                'industry': False
            }
        }
        if verbose:
            auto_fail_result["_debug"] = {
                "llm_prompt": None,
                "llm_response_raw": None,
                "llm_response_parsed": None,
                "ddg_role_results": role_results,
                "ddg_region_results": region_results,
                "ddg_industry_results": industry_results,
                "fuzzy_result": fuzzy_result,
                "needs_llm": needs_llm,
                "auto_fail_reason": "No DDG data for role verification"
            }
        return auto_fail_result
    
    if not all_search_context.strip():
        # No search results at all - this shouldn't happen often
        print(f"   ❌ AUTO-FAIL: No DDG search results at all")
        no_results_fail = {
            'verified': False,
            'role_match': False,
            'region_match': False,
            'industry_match': False,
            'extracted_role': 'Not found',
            'extracted_region': 'Unknown',
            'extracted_industry': 'Unknown',
            'reasoning': 'FAILED: No search results available. Cannot verify without data.',
            'fuzzy_matched': {'role': False, 'region': False, 'industry': False}
        }
        if verbose:
            no_results_fail["_debug"] = {
                "llm_prompt": None,
                "llm_response_raw": None,
                "llm_response_parsed": None,
                "ddg_role_results": role_results,
                "ddg_region_results": region_results,
                "ddg_industry_results": industry_results,
                "fuzzy_result": fuzzy_result,
                "needs_llm": needs_llm,
                "auto_fail_reason": "No DDG search results at all"
            }
        return no_results_fail
    
    # =========================================================================
    # STEP 6: DYNAMIC LLM PROMPT (Only for fields that need verification)
    # =========================================================================
    
    # Build claims section based on what needs verification
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
    
    # Build response format based on what needs verification
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
                    "temperature": 0  # Deterministic for consistent results
                },
                timeout=20
            ) as response:
                if response.status != 200:
                    return {
                        "verified": False,
                        "error": f"LLM API error: HTTP {response.status}",
                        "checked": False
                    }
                
                data = await response.json()
                llm_response_raw = data["choices"][0]["message"]["content"].strip()
                llm_response = llm_response_raw
                
                # Strip markdown code blocks
                if llm_response.startswith("```"):
                    lines = llm_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    llm_response = "\n".join(lines).strip()
                
                import json
                result = json.loads(llm_response)
                
                # Store verbose data for debugging
                verbose_data = {
                    "llm_prompt": prompt,
                    "llm_response_raw": llm_response_raw,
                    "llm_response_parsed": result,
                    "ddg_role_results": role_results,
                    "ddg_region_results": region_results,
                    "ddg_industry_results": industry_results,
                    "fuzzy_result": fuzzy_result,
                    "needs_llm": needs_llm,
                }
                
                # =========================================================================
                # MERGE FUZZY RESULTS WITH LLM RESULTS
                # =========================================================================
                # Use fuzzy results for fields that were already verified
                # Use LLM results only for fields that needed verification
                # =========================================================================
                
                # Role: Use fuzzy if verified, otherwise LLM
                if fuzzy_result["role_verified"]:
                    role_match = True
                    extracted_role = fuzzy_result["role_extracted"] or claimed_role
                else:
                    role_match = result.get("role_match", False)
                    extracted_role = result.get("extracted_role", "Not found")
                
                # Region: Use fuzzy if verified, otherwise LLM
                if fuzzy_result["region_verified"]:
                    region_match = True
                    extracted_region = fuzzy_result["region_extracted"] or claimed_region
                else:
                    region_match = result.get("region_match", False)
                    extracted_region = result.get("extracted_region", "")
                
                # Industry: Use fuzzy if verified, otherwise LLM
                if fuzzy_result["industry_verified"]:
                    industry_match = True
                    extracted_industry = fuzzy_result["industry_extracted"] or claimed_industry
                else:
                    industry_match = result.get("industry_match", False)
                    extracted_industry = result.get("extracted_industry", "")
                
                sub_industry_match = result.get("sub_industry_match", True)  # Default true if not verified
                confidence = result.get("confidence", 0.5)
                
                # =====================================================================
                # GEOPY VERIFICATION: Double-check region matching deterministically
                # LLMs can make mistakes - GeoPy uses real geographic data
                # =====================================================================
                geopy_reason = ""
                if not region_match and claimed_region and extracted_region:
                    # LLM said no match - let's verify with GeoPy
                    geopy_match, geopy_reason = locations_match_geopy(claimed_region, extracted_region)
                    if geopy_match:
                        print(f"   🌍 GeoPy override: {geopy_reason}")
                        region_match = True  # Override LLM decision
                
                # Overall verification: role, region, industry must match (HARD checks)
                # Sub-industry is a SOFT check - doesn't affect overall pass/fail
                all_match = role_match and region_match and industry_match
                
                # Build reasoning that includes fuzzy match info
                reasoning_parts = []
                if fuzzy_result["role_verified"]:
                    reasoning_parts.append(f"Role: FUZZY MATCHED ({fuzzy_result['role_reason']})")
                else:
                    reasoning_parts.append(f"Role: LLM verified")
                if fuzzy_result["region_verified"]:
                    reasoning_parts.append(f"Region: FUZZY MATCHED ({fuzzy_result['region_reason']})")
                else:
                    reasoning_parts.append(f"Region: LLM verified")
                if fuzzy_result["industry_verified"]:
                    reasoning_parts.append(f"Industry: FUZZY MATCHED ({fuzzy_result['industry_reason']})")
                else:
                    reasoning_parts.append(f"Industry: LLM verified")
                
                combined_reasoning = "; ".join(reasoning_parts)
                if result.get("reasoning"):
                    combined_reasoning += f" | LLM: {result.get('reasoning', '')}"
                
                final_result = {
                    "verified": all_match,
                    "role_match": role_match,
                    "extracted_role": extracted_role,
                    "region_match": region_match,
                    "extracted_region": extracted_region,
                    "geopy_reason": geopy_reason if geopy_reason else None,
                    "industry_match": industry_match,
                    "extracted_industry": extracted_industry,
                    "sub_industry_match": sub_industry_match,  # SOFT check - for info only
                    "sub_industry_verified": sub_industry_match,  # Flag to indicate verified/unverified
                    "extracted_sub_industry": result.get("extracted_sub_industry", ""),
                    "confidence": confidence,
                    "reasoning": combined_reasoning,
                    "checked": True,
                    "fuzzy_matched": {
                        "role": fuzzy_result["role_verified"],
                        "region": fuzzy_result["region_verified"],
                        "industry": fuzzy_result["industry_verified"]
                    }
                }
                
                # Add verbose debugging data if requested
                if verbose:
                    final_result["_debug"] = verbose_data
                
                return final_result
                
    except Exception as e:
        return {
            "verified": False,
            "error": f"Stage 5 verification failed: {str(e)}",
            "checked": False
        }


# ============================================================================
# COMBINED VERIFICATION (Legacy - kept for backwards compatibility)
# ============================================================================

async def run_lead_verification(
    lead: dict,
    verify_industry_flag: bool = True,
    verify_role_flag: bool = False,  # Disabled by default (costs money)
    verify_region_flag: bool = True,  # FREE - uses DDG
    role_sample_rate: float = 0.2    # Only verify 20% of roles
) -> dict:
    """
    Run all lead verification checks (LEGACY - use verify_stage5_unified instead).
    
    Args:
        lead: Lead data
        verify_industry_flag: Whether to run industry verification
        verify_role_flag: Whether to run role verification
        verify_region_flag: Whether to run region verification
        role_sample_rate: Probability of verifying role (0.0-1.0)
    
    Returns:
        {
            "industry_verification": {...},
            "role_verification": {...},
            "region_verification": {...},
            "verification_score": float  # 0.0-1.0
        }
    """
    import random
    
    result = {
        "industry_verification": None,
        "role_verification": None,
        "region_verification": None,
        "verification_score": 0.5  # Default neutral score
    }
    
    scores = []
    
    # Industry verification (free, always run if enabled)
    if verify_industry_flag:
        is_match, confidence, metadata = await verify_industry(lead)
        result["industry_verification"] = {
            "is_match": is_match,
            "confidence": confidence,
            **metadata
        }
        if metadata.get("checked"):
            scores.append(confidence if is_match else confidence * 0.5)
    
    # Role verification (costs money, sample only)
    if verify_role_flag and random.random() < role_sample_rate:
        is_verified, metadata = await verify_role(lead)
        result["role_verification"] = {
            "is_verified": is_verified,
            **metadata
        }
        if metadata.get("checked"):
            scores.append(1.0 if is_verified else 0.0)
    
    # Region verification (free - uses DDG)
    if verify_region_flag:
        company = lead.get("company") or lead.get("Company")
        region = lead.get("region") or lead.get("Region") or lead.get("location") or lead.get("Location")
        
        if company and region:
            is_match, extracted_location, metadata = await verify_region(company, region)
            result["region_verification"] = {
                "is_match": is_match,
                "extracted_location": extracted_location,
                **metadata
            }
            if metadata.get("checked"):
                confidence = metadata.get("confidence", 0.0)
                scores.append(confidence if is_match else confidence * 0.3)
    
    # Calculate composite score
    if scores:
        result["verification_score"] = sum(scores) / len(scores)
    
    return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    async def test():
        # Test region verification
        print("=" * 60)
        print("Testing Region Verification (DDG)")
        print("=" * 60)
        
        test_companies = [
            {"company": "PayPal", "region": "San Jose, CA"},
            {"company": "Stripe", "region": "San Francisco, CA"},
            {"company": "Microsoft", "region": "Redmond, WA"},
            {"company": "Apple", "region": "Cupertino, CA"},
            {"company": "Google", "region": "Mountain View, CA"},
            # Test incorrect region
            {"company": "Amazon", "region": "New York, NY"},  # Actually Seattle
        ]
        
        for test in test_companies:
            print(f"\n🔍 Testing: {test['company']} (claimed: {test['region']})")
            is_match, extracted, metadata = await verify_region(
                test["company"], 
                test["region"],
                use_llm=True  # Enable LLM fallback
            )
            print(f"   ✓ Match: {is_match}")
            print(f"   ✓ Extracted: {extracted}")
            print(f"   ✓ Confidence: {metadata.get('confidence', 0):.2%}")
            if metadata.get("details"):
                print(f"   ✓ City match: {metadata['details'].get('city_match')}")
                print(f"   ✓ State match: {metadata['details'].get('state_match')}")
        
        # Test industry classification
        print("\n" + "=" * 60)
        print("Testing Industry Classification")
        print("=" * 60)
        
        test_lead = {
            "company": "Stripe",
            "industry": "Finance and Insurance",
            "company_description": "Stripe is a technology company that builds economic infrastructure for the internet. Businesses use our software to accept payments and manage their businesses online."
        }
        
        is_match, confidence, metadata = await verify_industry(test_lead)
        print(f"Industry Match: {is_match}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Metadata: {metadata}")
        
        # Test role verification (only if API key is set)
        if APOLLO_API_KEY:
            print("\n" + "=" * 60)
            print("Testing Role Verification (Apollo)")
            print("=" * 60)
            
            test_lead_role = {
                "email": "test@example.com",  # Replace with real email to test
                "role": "CEO",
                "company": "Example Inc"
            }
            
            is_verified, metadata = await verify_role(test_lead_role)
            print(f"Role Verified: {is_verified}")
            print(f"Metadata: {metadata}")
        else:
            print("\n⚠️ APOLLO_API_KEY not set - skipping role verification test")
    
    asyncio.run(test())

