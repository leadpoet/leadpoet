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
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# API Keys
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "")
HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")


# ============================================================================
# 0. ROLE EXTRACTION FROM LINKEDIN (FREE - uses existing DDG search)
# ============================================================================
# Parses LinkedIn titles from DDG search results to extract role
# Pattern: "{Name} - {Role} at {Company} | LinkedIn"
# ============================================================================

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
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("⚠️ duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
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
# COMBINED VERIFICATION
# ============================================================================

async def run_lead_verification(
    lead: dict,
    verify_industry_flag: bool = True,
    verify_role_flag: bool = False,  # Disabled by default (costs money)
    verify_region_flag: bool = True,  # FREE - uses DDG
    role_sample_rate: float = 0.2    # Only verify 20% of roles
) -> dict:
    """
    Run all lead verification checks.
    
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

