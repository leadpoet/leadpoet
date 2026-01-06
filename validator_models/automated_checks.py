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
    get_field,
    get_employee_count,
    get_description
)
from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY

MAX_REP_SCORE = 48  # Wayback (6) + SEC (12) + WHOIS/DNSBL (10) + GDELT (10) + Companies House (10) = 48

# ========================================================================
# ICP (Ideal Customer Profile) Definitions
# ========================================================================
# These are the target customer profiles we want to incentivize miners to find.
# Leads matching these criteria receive a 1.5x multiplier on their rep_score during emissions.
# Format: Each ICP is defined by Sub-Industry + Role Details
# ========================================================================

# ========================================================================
# ICP (IDEAL CUSTOMER PROFILE) DEFINITIONS
# ========================================================================
# These definitions specify high-value lead profiles for ICP multiplier scoring.
# IMPORTANT: sub_industries must use EXACT names from INDUSTRY_TAXONOMY keys
# (case-sensitive, e.g., "FinTech" not "fintech", "E-Commerce" not "ecommerce")
# ========================================================================

ICP_DEFINITIONS = [
    {
        # Fuel/Energy - Operations & Technology Leaders
        "sub_industries": ["Fuel", "Oil and Gas", "Fossil Fuels", "Energy"],
        "role_details": [
            # Operations
            "coo", "chief operating officer", "director of operations", "vp of operations", 
            "vp operations", "head of operations", "operations manager", "operations director",
            # Technology
            "cto", "chief technology officer", "director of technology", "vp of technology", 
            "vp technology", "head of technology", "vp of engineering", "vp engineering",
            "engineering director", "head of engineering", "vp of it", "vp it", "it director",
            "cio", "chief information officer"
        ]
    },
    
    {
        # Agriculture/Farming - Operations & Technology Leaders
        "sub_industries": ["Agriculture", "Farming", "AgTech", "Livestock", "Aquaculture"],
        "role_details": [
            # Operations
            "coo", "chief operating officer", "director of operations", "vp of operations",
            "vp operations", "head of operations", "operations manager", "operations director",
            # Technology
            "cto", "chief technology officer", "director of technology", "vp of technology",
            "vp technology", "head of technology", "vp of engineering", "vp engineering",
            "engineering director", "head of engineering", "vp of it", "vp it", "it director",
            "cio", "chief information officer"
        ]
    },
    
    {
        # Renewable Energy - Operations, Technology & Asset Management
        "sub_industries": ["Solar", "Wind Energy", "Renewable Energy", "Clean Energy", 
                          "Biomass Energy", "Energy Storage", "Energy Efficiency"],
        "role_details": [
            # Operations
            "coo", "chief operating officer", "director of operations", "vp of operations",
            "vp operations", "head of operations", "operations manager", "operations director",
            # Technology
            "cto", "chief technology officer", "director of technology", "vp of technology",
            "vp technology", "head of technology", "vp of engineering", "vp engineering",
            "engineering director", "head of engineering", "vp of it", "vp it", "it director",
            "cio", "chief information officer",
            # Asset Management & Site Operations 
            "asset manager", "director of operation", "performance engineer", "site manager",
            "plant manager", "facility manager", "solar farm manager", "wind farm manager"
        ]
    },
    
    {
        # Winery/Horticulture - Farm & Operations Leaders
        "sub_industries": ["Winery", "Wine And Spirits", "Horticulture", "Farming", 
                          "Agriculture", "AgTech", "Hydroponics"],
        "role_details": [
            # Operations
            "coo", "chief operating officer", "director of operations", "vp of operations",
            "vp operations", "head of operations", "operations manager", "operations director",
            # Technology
            "cto", "chief technology officer", "director of technology", "vp of technology",
            "vp technology", "head of technology", "vp of engineering", "vp engineering",
            "engineering director", "head of engineering", "vp of it", "vp it", "it director",
            "cio", "chief information officer",
            # Farm Management & Precision Agriculture 
            "farm manager", "vineyard manager", "precision agriculture manager", 
            "head grower", "chief agronomist", "viticulturist", "horticulturist",
            "greenhouse manager", "crop manager", "production manager"
        ]
    },
    
    {
        # E-Commerce/Retail - Marketing & Growth Leaders
        "sub_industries": ["E-Commerce", "E-Commerce Platforms", "Retail", "Retail Technology"],
        "role_details": [
            # Marketing/Growth
            "vp ecommerce", "vp e-commerce", "vp of ecommerce", "director of ecommerce",
            "head of ecommerce", "ecommerce director", "head of growth", "director of growth",
            "vp of growth", "vp growth", "chief growth officer", "cmo", "chief marketing officer",
            "vp of marketing", "vp marketing", "director of marketing", "head of marketing",
            "vp of digital marketing", "director of digital marketing",
            # Leadership
            "founder", "co-founder", "ceo", "chief executive officer"
        ]
    },
    
    {
        # Digital Marketing/Advertising - Agency & Strategy Leaders
        "sub_industries": ["Digital Marketing", "Email Marketing", "Marketing", 
                          "Marketing Automation", "Advertising", "Advertising Platforms",
                          "Affiliate Marketing", "Content Marketing"],
        "role_details": [
            # Leadership/Strategy
            "founder", "co-founder", "ceo", "chief executive officer", 
            "director of partnerships", "vp of partnerships", "vp partnerships",
            "head of partnerships", "partnerships director",
            "head of strategy", "director of strategy", "vp of strategy", "vp strategy",
            "chief strategy officer", "cmo", "chief marketing officer",
            "managing director", "president", "managing partner"
        ]
    },
    
    {
        # AI/ML - Technical & Leadership Roles
        "sub_industries": ["Artificial Intelligence", "Machine Learning", 
                          "Natural Language Processing", "Predictive Analytics"],
        "role_details": [
            # Leadership
            "ceo", "chief executive officer", "founder", "co-founder",
            # Technology
            "cto", "chief technology officer", "vp of engineering", "vp engineering",
            "head of engineering", "engineering director", "vp of ai", "vp ai",
            "head of ai", "director of ai", "vp of machine learning", "vp machine learning",
            "head of machine learning", "director of machine learning", 
            "chief ai officer", "chief data officer",
            # Engineering
            "software engineer", "swe", "senior software engineer", "sr swe", 
            "staff software engineer", "principal software engineer", "lead software engineer",
            "software developer", "senior software developer", "sr software developer"
        ]
    },
    
    {
        # Real Estate Investment - Owners & Investment Leaders
        "sub_industries": ["Real Estate", "Real Estate Investment", "Residential", 
                          "Commercial Real Estate", "Property Development", "Property Management"],
        "role_details": [
            # Owner/Leadership
            "ceo", "chief executive officer", "owner", "co-owner", "sole operator",
            "founder", "co-founder", "managing partner", "managing director",
            "principal", "president", "partner"
        ]
    },
    
    {
        # Wealth Management/Family Office - Investment & Operations Leaders
        # Note: No "Family Office" sub-industry in taxonomy, using closest matches
        "sub_industries": ["Asset Management", "Venture Capital", "Hedge Funds", 
                          "Financial Services", "Impact Investing"],
        "role_details": [
            # Leadership
            "ceo", "chief executive officer", "president", "managing director", "managing partner",
            "principal", "partner", "founder", "co-founder",
            # Investment Leadership
            "cio", "chief investment officer", "director of investments", "vp of investments",
            "vp investments", "head of investments", "investment director", "investment manager",
            "portfolio manager", "head of portfolio management", "director of portfolio management",
            "senior portfolio manager", "lead portfolio manager",
            # Private Markets
            "head of private equity", "director of private equity", "vp private equity",
            "vp of private equity", "head of venture capital", "director of venture capital",
            "vp of venture capital", "vp venture capital", "head of vc", "director of vc",
            "head of real estate", "director of real estate", "vp real estate", "vp of real estate",
            "head of alternatives", "director of alternatives", "vp of alternatives", "vp alternatives",
            "head of direct investments", "director of direct investments",
            # Operations & Finance
            "coo", "chief operating officer", "director of operations", "vp of operations",
            "vp operations", "head of operations", "operations director",
            "cfo", "chief financial officer", "director of finance", "vp of finance",
            "vp finance", "head of finance", "finance director",
            # Wealth & Asset Management
            "family office manager", "wealth manager", "director of wealth management",
            "head of family office", "family office director", "head of wealth management",
            "asset manager", "head of asset management", "director of asset management"
        ]
    },
    
    {
        # FinTech/Banking - Risk & Compliance Leaders
        "sub_industries": ["FinTech", "Banking", "Payments", "Financial Services",
                          "Credit Cards", "Mobile Payments", "Transaction Processing"],
        "role_details": [
            # Risk & Compliance Leadership
            "cro", "chief risk officer", "vp of risk", "vp risk", "head of risk", 
            "director of risk", "risk director", "vp of risk management", "vp risk management",
            "director of risk management", "head of risk management",
            "cco", "chief compliance officer", "vp of compliance", "vp compliance", 
            "head of compliance", "director of compliance", "compliance director",
            "vp of regulatory compliance", "director of regulatory compliance",
            "head of regulatory affairs", "director of regulatory affairs",
            # Compliance Operations
            "compliance officer", "senior compliance officer", "compliance manager",
            "bsa officer", "aml officer", "kyc manager", "director of aml",
            "vp of bsa", "head of bsa", "anti-money laundering officer",
            "financial crimes manager", "director of financial crimes",
            # Risk Operations
            "risk officer", "senior risk officer", "risk manager", "enterprise risk manager",
            "operational risk manager", "credit risk manager", "director of operational risk"
        ]
    },
    
    {
        # Clinical Research/Labs - Data & Research Leaders
        "sub_industries": ["Clinical Trials", "Biotechnology", "Pharmaceutical", 
                          "Biopharma", "Life Science"],
        "role_details": [
            # Data & Research
            "data scientist", "senior data scientist", "lead data scientist", "principal data scientist",
            "data manager", "clinical data manager", "data management lead", "head of data management",
            "director of data management", "vp of data management", "vp data management",
            "biostatistician", "senior biostatistician", "lead biostatistician",
            "data analyst", "clinical data analyst", "research data analyst",
            # Leadership
            "ceo", "chief executive officer", "cto", "chief technology officer",
            "coo", "chief operating officer", "cso", "chief scientific officer",
            "vp of operations", "vp operations", "director of operations"
        ]
    },
    
    {
        # Research/Academic - Principal Investigators & Researchers
        "sub_industries": ["Higher Education", "Life Science", "Biotechnology", 
                          "Neuroscience", "Genetics"],
        "role_details": [
            # Principal Investigators & Researchers
            "principal investigator", "pi", "lead researcher", "senior researcher",
            "research director", "director of research", "head of research",
            "associate professor", "assistant professor", "professor",
            "research scientist", "senior research scientist", "staff scientist",
            "research fellow", "senior research fellow", "postdoctoral researcher",
            "lab director", "laboratory director", "research group leader",
            "department head", "department chair", "division chief"
        ]
    },
    
    {
        # Biotech/Pharma - Business Development & Scientific Leadership
        "sub_industries": ["Biotechnology", "Biopharma", "Pharmaceutical", 
                          "Genetics", "Life Science", "Bioinformatics"],
        "role_details": [
            # Business Development & Leadership
            "ceo", "chief executive officer", "founder", "co-founder",
            "cto", "chief technology officer", "cso", "chief scientific officer",
            "coo", "chief operating officer", "cmo", "chief medical officer",
            "vp of business development", "vp business development", "head of business development",
            "director of business development", "business development director",
            "bd lead", "business development lead", "business development manager",
            "vp of partnerships", "vp partnerships", "head of partnerships",
            "director of partnerships", "partnerships director",
            "vp of corporate development", "director of corporate development"
        ]
    },
    
    {
        # Broadcasting/Media (Africa Focus) - Technology & Content Leaders
        "sub_industries": ["Broadcasting", "Video", "Digital Media", "Content", 
                          "Content Delivery Network", "Telecommunications", 
                          "Digital Entertainment"],
        "role_details": [
            # Technology Leadership
            "cto", "chief technology officer", "cfo", "chief financial officer",
            "head of engineering", "vp of engineering", "vp engineering", "engineering director",
            "director of engineering",
            # Video/Streaming Specific
            "head of video", "head of streaming", "director of video", "director of streaming",
            "vp of video", "vp video", "vp of streaming", "vp streaming",
            "head of ott", "director of ott", "vp of ott", "vp ott", "ott director",
            "cdn architect", "video architect", "streaming architect", "media architect",
            "head of cdn", "director of cdn", "vp of cdn",
            # Content Operations
            "head of content operations", "director of content operations", "vp of content operations",
            "content operations manager", "head of media operations", "director of media operations",
            # Broadcast Operations
            "broadcast ops manager", "broadcast operations manager", "director of broadcast operations",
            "head of broadcast operations", "vp of broadcast operations",
            # Post-Production/Media Ops
            "post-production manager", "head of post-production", "director of post-production",
            "media ops manager", "media operations manager", "head of media ops",
            # Product (OTT)
            "head of product", "director of product", "vp of product", "vp product",
            "product director", "chief product officer", "cpo"
        ],
        # Region filter - only match leads from Africa
        "regions": ["africa", "african", "nigeria", "south africa", "kenya", "ghana", "egypt",
                    "morocco", "ethiopia", "tanzania", "uganda", "algeria", "sudan", "angola",
                    "mozambique", "cameroon", "ivory coast", "côte d'ivoire", "senegal", "zambia",
                    "zimbabwe", "rwanda", "tunisia", "libya", "democratic republic of congo", "drc",
                    "botswana", "namibia", "mauritius", "gabon", "malawi", "mali", "burkina faso",
                    "niger", "chad", "somalia", "benin", "togo", "sierra leone", "liberia",
                    "central african republic", "congo", "eritrea", "gambia", "guinea", "lesotho",
                    "madagascar", "mauritania", "swaziland", "eswatini"],
        # Custom multiplier for Africa leads (higher value than default 1.5x)
        "multiplier": 5.0
    },
    
    {
        # Hospitality/Hotels - Business Development, Owners & Operations (US)
        # Sub-industries from taxonomy: Hospitality, Hotel, Resorts (all under 'Travel and Tourism')
        "sub_industries": ["Hospitality", "Hotel", "Resorts", "Travel Accommodations", 
                          "Vacation Rental", "Tourism"],
        "role_details": [
            # Business Development
            "business development", "bd", "biz dev", "business dev", 
            "vp of business development", "vp business development", "head of business development",
            "director of business development", "business development manager", "business development director",
            "vp of bd", "head of bd", "director of bd",
            # Ownership/Leadership
            "owner", "co-owner", "business owner", "hotel owner", "property owner",
            "founder", "co-founder", "ceo", "chief executive officer",
            "president", "managing director", "general manager", "gm",
            "principal", "partner", "managing partner",
            # Operations Management
            "operations manager", "director of operations", "vp of operations", "vp operations",
            "head of operations", "operations director", "coo", "chief operating officer",
            # Hotel/Hospitality Specific
            "hotel manager", "hotel general manager", "hotel gm", "property manager",
            "resort manager", "resort general manager", "hospitality manager",
            "front office manager", "rooms division manager", "director of rooms",
            "director of hospitality", "vp of hospitality", "head of hospitality"
        ],
        # Region filter - US only
        "regions": ["united states", "usa", "us", "america", "american",
                    "california", "new york", "texas", "florida", "illinois", "pennsylvania",
                    "ohio", "georgia", "north carolina", "michigan", "new jersey", "virginia",
                    "washington", "arizona", "massachusetts", "tennessee", "indiana", "missouri",
                    "maryland", "wisconsin", "colorado", "minnesota", "south carolina", "alabama",
                    "louisiana", "kentucky", "oregon", "oklahoma", "connecticut", "utah", "iowa",
                    "nevada", "arkansas", "mississippi", "kansas", "new mexico", "nebraska",
                    "idaho", "west virginia", "hawaii", "maine", "montana", "rhode island",
                    "delaware", "south dakota", "north dakota", "alaska", "vermont", "wyoming"]
    },
    
    {
        # Small/Local Businesses - Owners (US)
        # Note: "Small Business" not in taxonomy, using "Local Business" which is closest match
        # This ICP targets business owners across various industries in the US
        "sub_industries": ["Local Business", "Local", "Retail", "Restaurants", "Food and Beverage",
                          "Professional Services", "Home Services", "Real Estate", "Construction",
                          "Automotive", "Health Care", "Fitness", "Beauty", "Consulting"],
        "role_details": [
            # Ownership
            "owner", "co-owner", "business owner", "sole proprietor", "sole operator",
            "franchise owner", "franchisee", "store owner", "shop owner",
            # Founder/Leadership
            "founder", "co-founder", "ceo", "chief executive officer",
            "president", "managing director", "principal", "partner",
            # Small Business Specific
            "proprietor", "operator", "entrepreneur"
        ],
        # Region filter - US only
        "regions": ["united states", "usa", "us", "america", "american",
                    "california", "new york", "texas", "florida", "illinois", "pennsylvania",
                    "ohio", "georgia", "north carolina", "michigan", "new jersey", "virginia",
                    "washington", "arizona", "massachusetts", "tennessee", "indiana", "missouri",
                    "maryland", "wisconsin", "colorado", "minnesota", "south carolina", "alabama",
                    "louisiana", "kentucky", "oregon", "oklahoma", "connecticut", "utah", "iowa",
                    "nevada", "arkansas", "mississippi", "kansas", "new mexico", "nebraska",
                    "idaho", "west virginia", "hawaii", "maine", "montana", "rhode island",
                    "delaware", "south dakota", "north dakota", "alaska", "vermont", "wyoming"]
    }
]

# ========================================================================
# INDUSTRY TAXONOMY VALIDATION HELPERS
# ========================================================================

def get_all_valid_industries() -> set:
    """
    Get all valid industry names from industry taxonomy.
    These are the unique industry_groups across all sub-industries.
    
    Returns:
        Set of valid industry names (case-preserved)
    """
    industries = set()
    for sub_industry, data in INDUSTRY_TAXONOMY.items():
        for group in data.get("industries", []):
            industries.add(group)
    return industries


def get_all_valid_sub_industries() -> set:
    """
    Get all valid sub-industry names from industry taxonomy.
    These are the keys of the INDUSTRY_TAXONOMY dictionary.
    
    Returns:
        Set of valid sub-industry names (case-preserved)
    """
    return set(INDUSTRY_TAXONOMY.keys())


def validate_exact_industry_match(claimed_industry: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that the claimed industry EXACTLY matches a valid industry.
    
    Args:
        claimed_industry: The industry submitted by the miner
        
    Returns:
        (is_valid, reason, matched_industry)
    """
    if not claimed_industry or not claimed_industry.strip():
        return False, "Industry is empty or missing", None
    
    claimed_clean = claimed_industry.strip()
    valid_industries = get_all_valid_industries()
    
    # Check exact match (case-insensitive)
    for valid in valid_industries:
        if valid.lower() == claimed_clean.lower():
            return True, f"Industry '{valid}' is valid (exact match)", valid
    
    # Not found - provide helpful error with valid options
    return False, f"Industry '{claimed_clean}' is NOT in industry taxonomy. Valid industries: {sorted(valid_industries)}", None


def validate_exact_sub_industry_match(claimed_sub_industry: str) -> Tuple[bool, str, Optional[str], Optional[Dict]]:
    """
    Validate that the claimed sub_industry EXACTLY matches a valid sub-industry.
    
    Args:
        claimed_sub_industry: The sub_industry submitted by the miner
        
    Returns:
        (is_valid, reason, matched_sub_industry, taxonomy_entry)
    """
    if not claimed_sub_industry or not claimed_sub_industry.strip():
        return False, "Sub-industry is empty or missing", None, None
    
    claimed_clean = claimed_sub_industry.strip()
    
    # Check exact match (case-insensitive)
    for sub_ind, data in INDUSTRY_TAXONOMY.items():
        if sub_ind.lower() == claimed_clean.lower():
            return True, f"Sub-industry '{sub_ind}' is valid (exact match)", sub_ind, data
    
    # Not found - provide helpful error
    return False, f"Sub-industry '{claimed_clean}' is NOT in industry taxonomy", None, None


def validate_industry_sub_industry_exact_pairing(matched_industry: str, matched_sub_industry: str) -> Tuple[bool, str]:
    """
    Validate that the industry is a valid industry_group for the sub_industry.
    Both must have already been validated as exact matches.
    
    Args:
        matched_industry: The validated industry name
        matched_sub_industry: The validated sub_industry name
        
    Returns:
        (is_valid, reason)
    """
    if not matched_sub_industry or matched_sub_industry not in INDUSTRY_TAXONOMY:
        return False, f"Sub-industry '{matched_sub_industry}' not found in taxonomy"
    
    valid_groups = INDUSTRY_TAXONOMY[matched_sub_industry].get("industries", [])
    
    if not valid_groups:
        # Some entries have empty industry_groups - allow any industry
        return True, f"Sub-industry '{matched_sub_industry}' has no specific industry restrictions"
    
    # Check if matched_industry is in the valid groups (case-insensitive)
    for group in valid_groups:
        if group.lower() == matched_industry.lower():
            return True, f"Industry '{matched_industry}' is valid for sub-industry '{matched_sub_industry}'"
    
    return False, f"Industry '{matched_industry}' is NOT valid for sub-industry '{matched_sub_industry}'. Valid: {valid_groups}"


# ========================================================================
# SUB-INDUSTRY VERIFICATION HELPERS (Using Industry Taxonomy) - LEGACY
# ========================================================================

def fuzzy_match_sub_industry(claimed_sub_industry: str) -> Tuple[Optional[str], Optional[Dict], float]:
    """
    LEGACY: Fuzzy match the miner's claimed sub_industry against the industry taxonomy.
    NOTE: Use validate_exact_sub_industry_match() for strict validation instead.
    
    Returns:
        (matched_key, taxonomy_entry, confidence) where:
        - matched_key: The exact key in INDUSTRY_TAXONOMY (or None if no match)
        - taxonomy_entry: Dict with 'industry_groups' and 'definition' (or None)
        - confidence: 0.0 to 1.0
    """
    if not claimed_sub_industry:
        return None, None, 0.0
    
    claimed_lower = claimed_sub_industry.strip().lower()
    
    # Try exact match first (case-insensitive)
    for key in INDUSTRY_TAXONOMY:
        if key.lower() == claimed_lower:
            return key, INDUSTRY_TAXONOMY[key], 1.0
    
    # Try contains match (if claimed is substring of taxonomy entry or vice versa)
    best_match = None
    best_confidence = 0.0
    
    for key in INDUSTRY_TAXONOMY:
        key_lower = key.lower()
        
        # Check if one contains the other
        if claimed_lower in key_lower or key_lower in claimed_lower:
            # Calculate similarity based on length ratio
            longer = max(len(claimed_lower), len(key_lower))
            shorter = min(len(claimed_lower), len(key_lower))
            confidence = shorter / longer
            
            if confidence > best_confidence:
                best_match = key
                best_confidence = confidence
        
        # Check for word overlap
        claimed_words = set(claimed_lower.replace('-', ' ').replace('/', ' ').split())
        key_words = set(key_lower.replace('-', ' ').replace('/', ' ').split())
        
        if claimed_words and key_words:
            overlap = len(claimed_words & key_words)
            total = len(claimed_words | key_words)
            word_confidence = overlap / total if total > 0 else 0
            
            if word_confidence > best_confidence:
                best_match = key
                best_confidence = word_confidence
    
    if best_match and best_confidence >= 0.5:
        return best_match, INDUSTRY_TAXONOMY[best_match], best_confidence
    
    return None, None, 0.0


def validate_industry_sub_industry_pairing(claimed_industry: str, matched_sub_industry: str) -> Tuple[bool, str]:
    """
    Validate that the miner's claimed industry is a valid industry_group for the sub_industry.
    
    Returns:
        (is_valid, reason)
    """
    if not matched_sub_industry or matched_sub_industry not in INDUSTRY_TAXONOMY:
        return False, f"Sub-industry '{matched_sub_industry}' not found in industry taxonomy"
    
    valid_groups = INDUSTRY_TAXONOMY[matched_sub_industry].get("industries", [])
    
    if not valid_groups:
        # Some entries have empty industry_groups (like "Association", "Commercial")
        return True, f"Sub-industry '{matched_sub_industry}' has no specific industry group requirements"
    
    # Normalize claimed industry for comparison
    claimed_lower = claimed_industry.strip().lower()
    
    for group in valid_groups:
        if group.lower() == claimed_lower:
            return True, f"Industry '{claimed_industry}' is valid for sub-industry '{matched_sub_industry}'"
        # Allow partial matches (e.g., "Technology" matches "Information Technology")
        if claimed_lower in group.lower() or group.lower() in claimed_lower:
            return True, f"Industry '{claimed_industry}' loosely matches valid group '{group}' for sub-industry '{matched_sub_industry}'"
    
    return False, f"Industry '{claimed_industry}' is NOT valid for sub-industry '{matched_sub_industry}'. Valid groups: {valid_groups}"


async def verify_sub_industry_with_llm(
    company: str,
    claimed_sub_industry: str,
    matched_sub_industry: str,
    definition: str,
    industry_search_results: List[Dict],
    openrouter_key: str
) -> Tuple[bool, str, float]:
    """
    Use LLM to verify that the company actually matches the claimed sub_industry,
    using the industry definition as the ground truth.
    
    Args:
        company: Company name
        claimed_sub_industry: What the miner claimed
        matched_sub_industry: The matched industry taxonomy key
        definition: sub-industry definition of the sub_industry
        industry_search_results: GSE search results about the company's industry
        openrouter_key: API key for OpenRouter
    
    Returns:
        (is_match, reasoning, confidence)
    """
    if not industry_search_results:
        return False, "No industry search results to verify against", 0.0
    
    # Build context from search results
    search_context = ""
    for i, result in enumerate(industry_search_results[:5], 1):
        title = result.get("title", "")
        snippet = result.get("snippet", result.get("body", ""))
        search_context += f"{i}. {title}\n   {snippet[:200]}\n"
    
    prompt = f"""You are verifying if a company matches a specific sub-industry classification.

COMPANY: {company}

CLAIMED SUB-INDUSTRY: {claimed_sub_industry}
MATCHED SUB-INDUSTRY CATEGORY: {matched_sub_industry}

SUB-INDUSTRY DEFINITION FOR THIS SUB-INDUSTRY:
"{definition}"

SEARCH RESULTS ABOUT THE COMPANY:
{search_context}

TASK: Based on the search results, does this company match the sub-industry definition above?

RULES:
1. The company's actual business must match the industry definition
2. Be STRICT - the company must genuinely fit the sub-industry category
3. If search results don't clearly show what the company does, return false
4. If the company operates in a DIFFERENT industry than claimed, return false

RESPOND WITH JSON ONLY:
{{
    "sub_industry_match": true/false,
    "extracted_business_type": "what the company actually does based on search results",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0
                },
                timeout=20
            ) as response:
                if response.status != 200:
                    return False, f"LLM API error: HTTP {response.status}", 0.0
                
                data = await response.json()
                llm_response = data["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response
                if llm_response.startswith("```"):
                    lines = llm_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    llm_response = "\n".join(lines).strip()
                
                result = json.loads(llm_response)
                
                is_match = result.get("sub_industry_match", False)
                reasoning = result.get("reasoning", "No reasoning provided")
                confidence = float(result.get("confidence", 0.0))
                extracted_type = result.get("extracted_business_type", "Unknown")
                
                full_reasoning = f"{reasoning} (Detected: {extracted_type})"
                
                return is_match, full_reasoning, confidence
                
    except json.JSONDecodeError as e:
        return False, f"Failed to parse LLM response: {str(e)}", 0.0
    except Exception as e:
        return False, f"LLM verification failed: {str(e)}", 0.0


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
    import hashlib
    
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
# Caches company LinkedIn data to avoid re-scraping the same company page.
# Key: company_linkedin slug (e.g., "microsoft")
# Value: Dict with company_name, industry, description, employee_count, location, timestamp
# TTL: 24 hours (companies don't change frequently)
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
# Maps company LinkedIn slug to standardized company name.
# Key: slug (e.g., "23andme")
# Value: Standardized company name from LinkedIn (e.g., "23andMe")
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
    """Check that all required fields are present and non-empty.
    
    Region validation:
    - country and city are ALWAYS required
    - state is required ONLY for United States leads
    - The validator builds the region string internally from country/state/city
    """
    required_fields = {
        "industry": ["industry", "Industry"],
        "sub_industry": ["sub_industry", "sub-industry", "Sub-industry", "Sub_industry"],
        "role": ["role", "Role"],
        "country": ["country", "Country"],
        "city": ["city", "City"],
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
    
    # Special check: state is required for US leads
    country = lead.get("country") or lead.get("Country") or ""
    country_lower = country.lower().strip() if country else ""
    us_aliases = ["united states", "usa", "us", "u.s.", "u.s.a.", "america", "united states of america"]
    
    if country_lower in us_aliases:
        state = lead.get("state") or lead.get("State") or ""
        if not state or not str(state).strip():
            missing_fields.append("state (required for US leads)")
    
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
        
        # Reject emails with "+" sign (prevents duplicate submission exploit via email aliasing)
        # Example: jwest+alias1@domain.com and jwest+alias2@domain.com are the same email
        if "+" in email.split("@")[0]:
            rejection_reason = {
                "stage": "Stage 0: Hardcoded Checks",
                "check_name": "check_email_regex",
                "message": f"Email contains '+' alias character (not allowed): {email}",
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

# Stage 3: Email Verification
# NOTE: Single-email validation functions (check_truelist_email, check_myemailverifier_email)
# have been REMOVED as of Dec 2024. All email validation now uses TrueList BATCH API
# via run_batch_automated_checks() for efficiency.
# See: submit_truelist_batch(), poll_truelist_batch(), parse_truelist_batch_csv()


# ============================================================================
# TrueList Batch Email Validation Functions
# ============================================================================
# These functions support batch email verification for improved throughput.
# See tasks9.md for the full migration plan.
# API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation
# ============================================================================

async def submit_truelist_batch(emails: List[str]) -> str:
    """
    Submit a list of emails to TrueList batch API.
    
    This function submits emails for batch verification. The batch is processed
    asynchronously by TrueList and must be polled for completion using
    poll_truelist_batch().
    
    API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation
    
    Args:
        emails: List of email addresses to validate (max 5000 per batch)
    
    Returns:
        batch_id: UUID of the created batch for polling
    
    Raises:
        EmailVerificationUnavailableError: If batch submission fails
        ValueError: If no emails provided or API key not configured
    
    Example:
        batch_id = await submit_truelist_batch(["user1@example.com", "user2@example.com"])
        # Then poll with: results = await poll_truelist_batch(batch_id)
    """
    if not emails:
        raise ValueError("No emails provided for batch validation")
    
    if not TRUELIST_API_KEY:
        raise EmailVerificationUnavailableError("TRUELIST_API_KEY not configured")
    
    # Log batch submission
    print(f"\n📧 TrueList Batch: Submitting {len(emails)} emails for validation...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # TrueList batch API endpoint
            url = "https://api.truelist.io/api/v1/batches"
            
            # IMPORTANT: TrueList batch API requires multipart/form-data, NOT JSON body
            # The 'data' parameter is a JSON string sent as a form field
            headers = {
                "Authorization": f"Bearer {TRUELIST_API_KEY}",
                # Note: Do NOT set Content-Type header - aiohttp sets it automatically for FormData
            }
            
            # CRITICAL: TrueList batch API rejects the ENTIRE batch if ANY email 
            # doesn't have an @ sign. Pre-filter to avoid this.
            # Emails without @ will be handled separately with immediate rejection.
            valid_emails = [email for email in emails if '@' in email]
            invalid_emails = [email for email in emails if '@' not in email]
            
            if invalid_emails:
                print(f"   ⚠️  Filtered {len(invalid_emails)} invalid emails (no @ sign)")
            
            if not valid_emails:
                print(f"   ❌ No valid emails to submit (all filtered)")
                return None  # Return None to indicate no batch was created
            
            # IMPORTANT: TrueList file upload is currently broken (returns 500)
            # Using JSON data format instead which works correctly
            # JSON format: {"data": [["email1"], ["email2"]], "validation_strategy": "accurate"}
            
            # Convert emails to JSON array format: [["email1"], ["email2"], ...]
            email_data = [[email] for email in valid_emails]
            
            # Generate unique batch name to avoid "Duplicate file upload" error
            unique_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
            
            json_payload = {
                "data": email_data,
                "validation_strategy": TRUELIST_BATCH_STRATEGY,  # "accurate" or "fast"
                "name": unique_name  # Unique name prevents duplicate detection
            }
            
            print(f"   📤 POST {url} (JSON format)")
            print(f"   📋 Batch name: {unique_name}")
            print(f"   📋 Strategy: {TRUELIST_BATCH_STRATEGY}")
            print(f"   📊 Email count: {len(valid_emails)}")
            
            async with session.post(
                url, 
                headers=headers, 
                json=json_payload,  # Use JSON format (file upload returns 500)
                timeout=60,  # 60s timeout for batch submission
                proxy=HTTP_PROXY_URL
            ) as response:
                
                # Handle error responses
                if response.status == 401:
                    raise EmailVerificationUnavailableError("TrueList API: Invalid or expired API key")
                elif response.status == 402:
                    raise EmailVerificationUnavailableError("TrueList API: Insufficient credits")
                elif response.status == 429:
                    raise EmailVerificationUnavailableError("TrueList API: Rate limited")
                elif response.status >= 500:
                    raise EmailVerificationUnavailableError(f"TrueList API server error: HTTP {response.status}")
                elif response.status != 200:
                    error_text = await response.text()
                    raise EmailVerificationUnavailableError(f"TrueList API error: HTTP {response.status} - {error_text[:200]}")
                
                # Parse successful response
                data = await response.json()
                
                batch_id = data.get("id")
                batch_state = data.get("batch_state", "unknown")
                email_count = data.get("email_count", 0)
                
                if not batch_id:
                    raise EmailVerificationUnavailableError("TrueList API: No batch_id in response")
                
                print(f"   ✅ Batch created successfully!")
                print(f"   🆔 Batch ID: {batch_id}")
                print(f"   📊 State: {batch_state}")
                print(f"   📧 Emails queued: {email_count}")
                
                return batch_id
    
    except aiohttp.ClientError as e:
        raise EmailVerificationUnavailableError(f"TrueList batch submission network error: {str(e)}")
    except asyncio.TimeoutError:
        raise EmailVerificationUnavailableError("TrueList batch submission timed out (60s)")
    except EmailVerificationUnavailableError:
        raise
    except Exception as e:
        raise EmailVerificationUnavailableError(f"TrueList batch submission error: {str(e)}")


async def poll_truelist_batch(batch_id: str) -> Dict[str, dict]:
    """
    Poll TrueList batch until completion or timeout.
    
    This function polls the batch status every TRUELIST_BATCH_POLL_INTERVAL seconds
    until the batch is complete or TRUELIST_BATCH_TIMEOUT is reached. When complete,
    it downloads and parses the annotated CSV to get per-email results.
    
    API Reference: https://apidocs.truelist.io/#tag/Batch-email-validation
    
    Args:
        batch_id: UUID of the batch to poll (from submit_truelist_batch)
    
    Returns:
        Dict mapping email -> result dict:
        {
            "email@domain.com": {
                "status": "email_ok",      # TrueList email_sub_state
                "passed": True,            # True if email_ok
                "needs_retry": False,      # True if unknown/timeout/error
                "rejection_reason": None   # Rejection reason if failed
            },
            ...
        }
    
    Raises:
        EmailVerificationUnavailableError: If polling times out or batch fails
    
    Example:
        batch_id = await submit_truelist_batch(emails)
        results = await poll_truelist_batch(batch_id)
        for email, result in results.items():
            if result["passed"]:
                print(f"{email} is valid")
    """
    import time
    import csv
    from io import StringIO
    
    if not batch_id:
        raise ValueError("No batch_id provided for polling")
    
    if not TRUELIST_API_KEY:
        raise EmailVerificationUnavailableError("TRUELIST_API_KEY not configured")
    
    url = f"https://api.truelist.io/api/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    
    start_time = time.time()
    poll_count = 0
    
    print(f"\n⏳ TrueList Batch: Polling for completion...")
    print(f"   🆔 Batch ID: {batch_id}")
    print(f"   ⏱️  Poll interval: {TRUELIST_BATCH_POLL_INTERVAL}s")
    print(f"   ⏰ Timeout: {TRUELIST_BATCH_TIMEOUT // 60} minutes")
    
    while True:
        elapsed = time.time() - start_time
        
        # Check timeout
        if elapsed >= TRUELIST_BATCH_TIMEOUT:
            raise EmailVerificationUnavailableError(
                f"TrueList batch polling timed out after {TRUELIST_BATCH_TIMEOUT // 60} minutes"
            )
        
        poll_count += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=headers, 
                    timeout=30,
                    proxy=HTTP_PROXY_URL
                ) as response:
                    
                    if response.status == 404:
                        raise EmailVerificationUnavailableError(f"TrueList batch not found: {batch_id}")
                    elif response.status == 401:
                        raise EmailVerificationUnavailableError("TrueList API: Invalid or expired API key")
                    elif response.status >= 500:
                        # Server error - retry polling
                        print(f"   ⚠️  Poll #{poll_count}: Server error (HTTP {response.status}), retrying...")
                        await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
                        continue
                    elif response.status != 200:
                        error_text = await response.text()
                        raise EmailVerificationUnavailableError(
                            f"TrueList API error: HTTP {response.status} - {error_text[:200]}"
                        )
                    
                    # Success (HTTP 200) - parse JSON response
                    data = await response.json()
                    
                    batch_state = data.get("batch_state", "unknown")
                    email_count = data.get("email_count", 0)
                    processed_count = data.get("processed_count", 0)
                    ok_count = data.get("ok_count", 0)
                    unknown_count = data.get("unknown_count", 0)
                    
                    # Progress update every 5 polls or when state changes
                    if poll_count % 5 == 1 or batch_state == "completed":
                        progress_pct = (processed_count / email_count * 100) if email_count > 0 else 0
                        print(f"   📊 Poll #{poll_count} ({elapsed:.0f}s): {batch_state} - {processed_count}/{email_count} ({progress_pct:.0f}%)")
                    
                    # Check if batch is complete
                    # CRITICAL: TrueList may say "completed" before all emails are processed!
                    # We must check BOTH state AND processed_count
                    if batch_state == "completed" and processed_count >= email_count:
                        print(f"   ✅ Batch fully completed!")
                        print(f"   📧 Total: {email_count}, OK: {ok_count}, Unknown: {unknown_count}")
                    elif batch_state == "completed" and processed_count < email_count:
                        # TrueList says completed but not all processed - keep polling!
                        print(f"   ⚠️ Batch says 'completed' but only {processed_count}/{email_count} processed - continuing to poll...")
                        await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
                        continue
                    
                    if batch_state == "completed":
                        
                        # CRITICAL: Wait for CSV generation to finish
                        # TrueList's "completed" state doesn't mean CSV is ready
                        # CSV generation happens asynchronously after processing
                        CSV_GENERATION_DELAY = 15  # seconds
                        print(f"   ⏳ Waiting {CSV_GENERATION_DELAY}s for CSV generation...")
                        await asyncio.sleep(CSV_GENERATION_DELAY)
                        
                        # Re-fetch batch data to get fresh CSV URLs
                        print(f"   🔄 Re-fetching batch data for fresh CSV URLs...")
                        async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as refresh_response:
                            if refresh_response.status == 200:
                                data = await refresh_response.json()
                        
                        # ============================================================
                        # FALLBACK: CSV downloads (the /emails endpoint returns 404)
                        # ============================================================
                        
                        # Get the annotated CSV URL - try multiple possible fields
                        annotated_csv_url = (
                            data.get("annotated_csv_url") or 
                            data.get("results_url") or 
                            data.get("download_url") or
                            data.get("csv_url")
                        )
                        
                        if not annotated_csv_url:
                            # CSV URL is null - TrueList may still be generating it
                            # Wait and retry polling - TrueList CSV generation is ASYNC
                            # and can take 30-60+ seconds after batch shows "completed"
                            CSV_URL_RETRY_DELAY = 10  # seconds
                            CSV_URL_MAX_RETRIES = 6   # Total: 60 seconds of waiting
                            
                            for csv_retry in range(CSV_URL_MAX_RETRIES):
                                print(f"   ⚠️  No CSV URL in response, waiting {CSV_URL_RETRY_DELAY}s and retrying ({csv_retry + 1}/{CSV_URL_MAX_RETRIES})...")
                                await asyncio.sleep(CSV_URL_RETRY_DELAY)
                                
                                # Re-poll the batch
                                async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as retry_response:
                                    if retry_response.status == 200:
                                        retry_data = await retry_response.json()
                                        annotated_csv_url = retry_data.get("annotated_csv_url")
                                        if annotated_csv_url:
                                            print(f"   ✅ CSV URL now available after retry!")
                                            data = retry_data  # Update data for later use
                                            break
                            
                            if not annotated_csv_url:
                                # ================================================================
                                # FALLBACK: Combine multiple CSV files when annotated_csv_url is null
                                # TrueList provides separate CSVs for different email categories:
                                # - highest_reach_csv_url: email_ok + accept_all emails
                                # - only_invalid_csv_url: failed emails (failed_mx, failed_no_mailbox, etc)
                                # By combining these, we can reconstruct all email results!
                                # ================================================================
                                print(f"   ⚠️  annotated_csv_url is null - trying to combine alternative CSVs...")
                                
                                combined_results = {}
                                
                                # Try highest_reach_csv_url (contains ok + accept_all)
                                highest_reach_url = data.get("highest_reach_csv_url")
                                if highest_reach_url:
                                    print(f"   📥 Downloading highest_reach CSV...")
                                    try:
                                        reach_results = await _download_and_parse_batch_csv(highest_reach_url, headers)
                                        if reach_results:
                                            print(f"   ✅ Got {len(reach_results)} emails from highest_reach CSV")
                                            combined_results.update(reach_results)
                                    except Exception as e:
                                        print(f"   ⚠️  highest_reach CSV failed: {str(e)[:50]}")
                                
                                # Try only_invalid_csv_url (contains failed emails)
                                invalid_url = data.get("only_invalid_csv_url")
                                if invalid_url:
                                    print(f"   📥 Downloading only_invalid CSV...")
                                    try:
                                        invalid_results = await _download_and_parse_batch_csv(invalid_url, headers)
                                        if invalid_results:
                                            print(f"   ✅ Got {len(invalid_results)} emails from only_invalid CSV")
                                            combined_results.update(invalid_results)
                                    except Exception as e:
                                        print(f"   ⚠️  only_invalid CSV failed: {str(e)[:50]}")
                                
                                # Try safest_bet_csv_url as additional source (email_ok only)
                                safest_url = data.get("safest_bet_csv_url")
                                if safest_url and len(combined_results) < email_count:
                                    print(f"   📥 Downloading safest_bet CSV...")
                                    try:
                                        safest_results = await _download_and_parse_batch_csv(safest_url, headers)
                                        if safest_results:
                                            # Only add emails we don't already have
                                            new_count = 0
                                            for email, result in safest_results.items():
                                                if email not in combined_results:
                                                    combined_results[email] = result
                                                    new_count += 1
                                            if new_count > 0:
                                                print(f"   ✅ Got {new_count} additional emails from safest_bet CSV")
                                    except Exception as e:
                                        print(f"   ⚠️  safest_bet CSV failed: {str(e)[:50]}")
                                
                                if combined_results:
                                    print(f"   🎉 Combined {len(combined_results)} total email results from alternative CSVs!")
                                    return combined_results
                                
                                # If alternative CSVs also failed, try constructed URLs
                                constructed_url = f"https://api.truelist.io/api/v1/batches/{batch_id}/download"
                                print(f"   ⚠️  Alternative CSVs failed, trying constructed URL: {constructed_url}")
                                
                                try:
                                    results = await _download_and_parse_batch_csv(constructed_url, headers)
                                    if results:
                                        print(f"   ✅ Constructed URL worked! Parsed {len(results)} email results")
                                        return results
                                except Exception as download_err:
                                    print(f"   ⚠️  Constructed URL failed: {str(download_err)[:100]}")
                                
                                # Final fallback: Use batch stats (won't work for individual emails)
                                print(f"   ❌ Could not download CSV results after all fallbacks. Full response:")
                                print(f"   {json.dumps(data, default=str)[:500]}")
                                return _parse_batch_status_from_response(data, batch_id)
                        
                        # Download and parse the CSV
                        print(f"   📥 Downloading results from: {annotated_csv_url[:80]}...")
                        results = await _download_and_parse_batch_csv(annotated_csv_url, headers)
                        
                        print(f"   ✅ Parsed {len(results)} email results")
                        
                        # CRITICAL FIX: If CSV has fewer results than expected, use fallback CSVs
                        if len(results) < email_count:
                            print(f"   ⚠️  CSV only had {len(results)}/{email_count} emails - using fallback CSVs...")
                            
                            combined_results = {}
                            
                            # Try highest_reach_csv_url (contains ok + accept_all)
                            highest_reach_url = data.get("highest_reach_csv_url")
                            if highest_reach_url:
                                print(f"   📥 Fallback: Downloading highest_reach CSV...")
                                try:
                                    reach_results = await _download_and_parse_batch_csv(highest_reach_url, headers)
                                    if reach_results:
                                        print(f"   ✅ Got {len(reach_results)} emails from highest_reach CSV")
                                        combined_results.update(reach_results)
                                except Exception as e:
                                    print(f"   ⚠️  highest_reach CSV failed: {str(e)[:50]}")
                            
                            # Try only_invalid_csv_url (contains failed emails)
                            invalid_url = data.get("only_invalid_csv_url")
                            if invalid_url:
                                print(f"   📥 Fallback: Downloading only_invalid CSV...")
                                try:
                                    invalid_results = await _download_and_parse_batch_csv(invalid_url, headers)
                                    if invalid_results:
                                        print(f"   ✅ Got {len(invalid_results)} emails from only_invalid CSV")
                                        combined_results.update(invalid_results)
                                except Exception as e:
                                    print(f"   ⚠️  only_invalid CSV failed: {str(e)[:50]}")
                            
                            # Try safest_bet_csv_url as additional source
                            safest_url = data.get("safest_bet_csv_url")
                            if safest_url and len(combined_results) < email_count:
                                print(f"   📥 Fallback: Downloading safest_bet CSV...")
                                try:
                                    safest_results = await _download_and_parse_batch_csv(safest_url, headers)
                                    if safest_results:
                                        print(f"   ✅ Got {len(safest_results)} emails from safest_bet CSV")
                                        for email, result in safest_results.items():
                                            if email not in combined_results:
                                                combined_results[email] = result
                                except Exception as e:
                                    print(f"   ⚠️  safest_bet CSV failed: {str(e)[:50]}")
                            
                            if combined_results:
                                print(f"   ✅ Combined fallback CSVs: {len(combined_results)} total emails")
                                return combined_results
                            else:
                                print(f"   ❌ All fallback CSVs failed or empty")
                                # Return empty results - will trigger retry logic
                        
                        return results
                    
                    elif batch_state == "failed":
                        raise EmailVerificationUnavailableError(
                            f"TrueList batch failed: {data.get('error', 'Unknown error')}"
                        )
                    
                    # Still processing - wait and poll again
                    await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
        
        except EmailVerificationUnavailableError:
            raise
        except aiohttp.ClientError as e:
            # Network error - retry polling
            print(f"   ⚠️  Poll #{poll_count}: Network error ({str(e)[:50]}), retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
        except asyncio.TimeoutError:
            # Timeout on single request - retry polling
            print(f"   ⚠️  Poll #{poll_count}: Request timeout, retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)
        except Exception as e:
            print(f"   ⚠️  Poll #{poll_count}: Unexpected error ({str(e)[:50]}), retrying...")
            await asyncio.sleep(TRUELIST_BATCH_POLL_INTERVAL)


async def _download_and_parse_batch_csv(csv_url: str, headers: dict) -> Dict[str, dict]:
    """
    Download and parse TrueList annotated CSV results.
    
    IMPORTANT: CSV downloads are done WITHOUT proxy because TrueList's
    S3 signed URLs may not work correctly through proxy servers.
    The API calls (submit, poll) still use proxy for rate limit protection.
    
    Args:
        csv_url: URL to the annotated CSV file
        headers: Auth headers for the request
    
    Returns:
        Dict mapping email -> result dict
    """
    import csv
    from io import StringIO
    
    try:
        async with aiohttp.ClientSession() as session:
            # NOTE: NO PROXY for CSV downloads - S3 signed URLs don't work through proxies
            async with session.get(
                csv_url, 
                headers=headers, 
                timeout=60
                # proxy removed - CSVs must be downloaded directly
            ) as response:
                
                if response.status != 200:
                    raise EmailVerificationUnavailableError(
                        f"Failed to download batch CSV: HTTP {response.status}"
                    )
                
                csv_content = await response.text()
                
                return parse_truelist_batch_csv(csv_content)
    
    except aiohttp.ClientError as e:
        raise EmailVerificationUnavailableError(f"Failed to download batch CSV: {str(e)}")
    except asyncio.TimeoutError:
        raise EmailVerificationUnavailableError("Batch CSV download timed out")


async def _fetch_batch_email_results(batch_id: str, headers: dict, email_count: int) -> Dict[str, dict]:
    """
    Fetch email results using TrueList's /emails endpoint with pagination.
    
    This is the CORRECT way to retrieve individual email results per the API docs:
    GET /api/v1/batches/{batch_uuid}/emails
    
    Args:
        batch_id: UUID of the completed batch
        headers: Auth headers with Bearer token
        email_count: Expected number of emails (for progress reporting)
    
    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry
    """
    # Define which statuses pass, fail, or need retry
    PASS_STATUSES = {"email_ok"}
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error"}
    
    results = {}
    page = 1
    per_page = 100  # Maximum allowed per the docs
    
    print(f"   📥 Fetching email results via /emails endpoint (paginated)...")
    
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                url = f"https://api.truelist.io/api/v1/batches/{batch_id}/emails?page={page}&per_page={per_page}"
                
                async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"   ⚠️ /emails endpoint returned HTTP {response.status}: {error_text[:100]}")
                        break
                    
                    data = await response.json()
                    email_addresses = data.get("email_addresses", [])
                    
                    if not email_addresses:
                        # No more results
                        break
                    
                    # Process each email result
                    for email_data in email_addresses:
                        # The email object structure from the API
                        email = email_data.get("email_address", email_data.get("email", "")).lower()
                        if not email:
                            continue
                        
                        email_state = email_data.get("email_state", "unknown")
                        email_sub_state = email_data.get("email_sub_state", email_state)
                        
                        # Determine pass/fail/retry
                        if email_sub_state in PASS_STATUSES:
                            results[email] = {
                                "status": email_sub_state,
                                "passed": True,
                                "needs_retry": False,
                                "rejection_reason": None
                            }
                        elif email_sub_state in RETRY_STATUSES:
                            results[email] = {
                                "status": email_sub_state,
                                "passed": False,
                                "needs_retry": True,
                                "rejection_reason": None
                            }
                        else:
                            # Failed status
                            results[email] = {
                                "status": email_sub_state,
                                "passed": False,
                                "needs_retry": False,
                                "rejection_reason": {
                                    "stage": "Stage 3",
                                    "check_name": "truelist_email_verification",
                                    "message": f"Email verification failed: {email_sub_state}",
                                    "truelist_status": email_sub_state
                                }
                            }
                    
                    print(f"   📄 Page {page}: Got {len(email_addresses)} emails (total so far: {len(results)})")
                    
                    # Check if we got all expected emails
                    if len(results) >= email_count:
                        break
                    
                    # Check if this was the last page (fewer than per_page results)
                    if len(email_addresses) < per_page:
                        break
                    
                    page += 1
                    
                    # Small delay between pages to avoid rate limiting
                    await asyncio.sleep(0.5)
        
        print(f"   ✅ Fetched {len(results)}/{email_count} email results via API")
        return results
        
    except Exception as e:
        print(f"   ⚠️ Error fetching email results: {str(e)[:100]}")
        return results


def parse_truelist_batch_csv(csv_content: str) -> Dict[str, dict]:
    """
    Parse TrueList annotated CSV into email -> result mapping.
    
    Maps TrueList statuses to our internal format:
    - email_ok → passed=True
    - accept_all, disposable, failed_* → passed=False  
    - unknown, timeout, error → needs_retry=True
    
    Args:
        csv_content: Raw CSV content from TrueList
    
    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry, rejection_reason
    """
    import csv
    from io import StringIO
    
    results = {}
    
    # Define which statuses pass, fail, or need retry
    PASS_STATUSES = {"email_ok"}
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error"}
    # All other statuses are considered failures
    
    try:
        reader = csv.DictReader(StringIO(csv_content))
        
        # Debug: Print first few lines and column names
        rows = list(reader)
        if rows:
            print(f"   📋 CSV columns: {list(rows[0].keys())}")
            print(f"   📋 First row: {dict(list(rows[0].items())[:5])}")  # First 5 fields
        else:
            print(f"   ⚠️ CSV is empty! Content preview: {csv_content[:200]}")
        
        for row in rows:
            # TrueList CSV has columns: Try multiple column name formats
            # API may use different column names: "Email Address", "email", "Email", etc.
            email = (row.get("Email Address") or row.get("email") or 
                     row.get("Email") or row.get("email_address") or "").strip().lower()
            
            if not email:
                continue
            
            # Get the detailed status - try multiple column name formats
            # TrueList uses "Email Sub-State" or "Email State"
            status = (row.get("Email Sub-State") or row.get("email_sub_state") or
                      row.get("Email State") or row.get("email_state") or 
                      row.get("sub_state") or row.get("state") or "unknown")
            status = status.lower() if status else "unknown"
            
            # Determine pass/fail/retry
            if status in PASS_STATUSES:
                results[email] = {
                    "status": status,
                    "passed": True,
                    "needs_retry": False,
                    "rejection_reason": None
                }
            elif status in RETRY_STATUSES:
                results[email] = {
                    "status": status,
                    "passed": False,
                    "needs_retry": True,
                    "rejection_reason": None  # Don't reject - will retry
                }
            else:
                # Failed status - build rejection reason
                rejection_reason = _build_email_rejection_reason(status)
                results[email] = {
                    "status": status,
                    "passed": False,
                    "needs_retry": False,
                    "rejection_reason": rejection_reason
                }
        
        return results
    
    except Exception as e:
        raise EmailVerificationUnavailableError(f"Failed to parse batch CSV: {str(e)}")


def _parse_batch_status_from_response(data: dict, batch_id: str) -> Dict[str, dict]:
    """
    Fallback: Parse batch results from API response when CSV URL is not available.
    
    This is used when annotated_csv_url is missing from the response.
    Returns aggregate counts but may not have per-email details.
    
    Args:
        data: Batch API response data
        batch_id: Batch ID for logging
    
    Returns:
        Dict with limited results (may need alternative approach)
    """
    print(f"   ⚠️  Using fallback batch parsing (no CSV URL)")
    
    # This is a fallback - in practice, TrueList should always provide the CSV URL
    # Log a warning and return empty results to trigger retry logic
    email_count = data.get("email_count", 0)
    ok_count = data.get("ok_count", 0)
    unknown_count = data.get("unknown_count", 0)
    
    print(f"   📊 Batch stats: {email_count} total, {ok_count} ok, {unknown_count} unknown")
    print(f"   ⚠️  Cannot map to individual emails without CSV - returning empty results")
    
    # Return empty dict - the orchestrator should handle this case
    return {}


def _build_email_rejection_reason(status: str) -> dict:
    """
    Build a rejection reason dict for a failed email status.
    
    Maps TrueList statuses to user-friendly rejection messages.
    
    Args:
        status: TrueList email_sub_state value
    
    Returns:
        Rejection reason dict compatible with our validation format
    """
    # Map TrueList statuses to rejection messages
    # Note: TrueList uses both "disposable" and "is_disposable" for different cases
    status_messages = {
        "accept_all": "Email is catch-all/accept-all (instant rejection)",
        "disposable": "Email is from a disposable provider",
        "is_disposable": "Email is from a disposable provider",
        "failed_no_mailbox": "Mailbox does not exist",
        "failed_syntax_check": "Invalid email syntax",
        "failed_mx_check": "Domain has no MX records (cannot receive email)",
        "role": "Email is a role-based address (e.g., info@, support@)",
        "invalid": "Email is invalid",
        "spam_trap": "Email is a known spam trap",
        "complainer": "Email owner is a known complainer",
        "ok_for_all": "Email domain accepts all emails (catch-all)",
    }
    
    message = status_messages.get(status, f"Email status '{status}' (only 'email_ok' accepted)")
    
    return {
        "stage": "Stage 3: TrueList Batch",
        "check_name": "truelist_batch_validation",
        "message": message,
        "failed_fields": ["email"],
        "truelist_status": status
    }


# ============================================================================
# Batch Helper Functions
# ============================================================================

async def submit_and_poll_truelist(emails: List[str]) -> Tuple[str, Dict[str, dict]]:
    """
    Submit batch and poll for results (combined for background task).
    
    This wrapper combines submit_truelist_batch() and poll_truelist_batch()
    for use with asyncio.create_task() in the batch orchestrator.
    
    Args:
        emails: List of email addresses to validate
    
    Returns:
        Tuple of (batch_id, results_dict) where results_dict maps email -> result
        batch_id is returned so caller can delete the batch before retrying
    """
    batch_id = await submit_truelist_batch(emails)
    results = await poll_truelist_batch(batch_id)
    return batch_id, results


async def verify_emails_inline(emails: List[str]) -> Dict[str, dict]:
    """
    Verify emails using TrueList's INLINE verification API (not batch).
    
    This is a FALLBACK for emails that TrueList's batch API silently drops.
    Some enterprise domains (spglobal.com, jacobs.com, etc.) work with inline
    verification but not batch verification.
    
    Rate limit: 10 requests/second per TrueList docs.
    Each request can verify up to 3 emails.
    
    Args:
        emails: List of email addresses to verify
    
    Returns:
        Dict mapping email -> result dict with status, passed, needs_retry
    """
    if not TRUELIST_API_KEY:
        print("   ⚠️ TRUELIST_API_KEY not configured for inline verification")
        return {email: {"needs_retry": True, "error": "No API key"} for email in emails}
    
    results = {}
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    
    # TrueList inline API accepts up to 3 emails per request (space-separated)
    BATCH_SIZE = 3
    PASS_STATUSES = {"email_ok"}  # Only email_ok passes - accept_all is rejected
    RETRY_STATUSES = {"unknown", "unknown_error", "timeout", "error", "failed_greylisted"}
    
    print(f"   🔍 Inline verification for {len(emails)} emails (TrueList batch fallback)...")
    import time as _time
    _start = _time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(emails), BATCH_SIZE):
                batch = emails[i:i+BATCH_SIZE]
                email_param = " ".join(batch)
                
                url = f"https://api.truelist.io/api/v1/verify_inline?email={email_param}"
                
                try:
                    async with session.post(url, headers=headers, timeout=35, proxy=HTTP_PROXY_URL) as response:
                        if response.status == 429:
                            print(f"   ⚠️ Rate limited, waiting 2s...")
                            await asyncio.sleep(2)
                            continue
                        
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"   ⚠️ Inline verify failed ({response.status}): {error_text[:50]}")
                            for email in batch:
                                results[email.lower()] = {"needs_retry": True, "error": f"HTTP {response.status}"}
                            continue
                        
                        data = await response.json()
                        email_results = data.get("emails", [])
                        
                        # DEBUG: Log first response to see actual structure
                        if i == 0 and email_results:
                            print(f"   📋 Inline API first response: {email_results[0]}")
                        
                        for email_data in email_results:
                            # TrueList inline uses "address" not "email_address"
                            email = email_data.get("address", email_data.get("email_address", email_data.get("email", ""))).lower()
                            if not email:
                                continue
                            
                            email_state = email_data.get("email_state", "unknown")
                            email_sub_state = email_data.get("email_sub_state", email_state)
                            
                            # DEBUG: Log non-email_ok statuses
                            if email_sub_state != "email_ok":
                                print(f"   📋 Inline status: {email} -> {email_state}/{email_sub_state}")
                            
                            if email_sub_state in PASS_STATUSES:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": True,
                                    "needs_retry": False,
                                    "rejection_reason": None
                                }
                            elif email_sub_state in RETRY_STATUSES:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": False,
                                    "needs_retry": True,
                                    "rejection_reason": None
                                }
                            else:
                                results[email] = {
                                    "status": email_sub_state,
                                    "passed": False,
                                    "needs_retry": False,
                                    "rejection_reason": {
                                        "stage": "Stage 3",
                                        "check_name": "truelist_inline_verification",
                                        "message": f"Email verification failed: {email_sub_state}",
                                        "truelist_status": email_sub_state
                                    }
                                }
                                
                except asyncio.TimeoutError:
                    print(f"   ⚠️ Inline verify timeout for: {batch}")
                    for email in batch:
                        results[email.lower()] = {"needs_retry": True, "error": "timeout"}
                except Exception as e:
                    print(f"   ⚠️ Inline verify error: {e}")
                    for email in batch:
                        results[email.lower()] = {"needs_retry": True, "error": str(e)}
                
                # Rate limit: 10 req/sec = 100ms between requests
                await asyncio.sleep(0.15)
        
        passed = sum(1 for r in results.values() if r.get("passed"))
        elapsed = _time.time() - _start
        print(f"   ✅ Inline verification: {passed}/{len(emails)} passed ({elapsed:.1f}s)")
        return results
        
    except Exception as e:
        print(f"   ⚠️ Inline verification error: {e}")
        return {email.lower(): {"needs_retry": True, "error": str(e)} for email in emails}


async def retry_truelist_batch(emails: List[str], prev_batch_id: str = None) -> Tuple[str, Dict[str, dict]]:
    """
    Submit a retry batch and poll for results.
    
    IMPORTANT: If prev_batch_id is provided, deletes it first to clear
    TrueList's duplicate detection before submitting the retry.
    
    On exception, marks all emails as needs_retry=True so the orchestrator
    can decide whether to retry again or skip.
    
    Args:
        emails: List of email addresses to retry validation
        prev_batch_id: Optional batch_id from previous retry to delete first
    
    Returns:
        Tuple of (batch_id, results_dict)
        On error, returns (None, {email: needs_retry=True})
    """
    try:
        # Delete previous retry batch if provided (clears duplicate detection)
        if prev_batch_id:
            await delete_truelist_batch(prev_batch_id)
        
        batch_id = await submit_truelist_batch(emails)
        results = await poll_truelist_batch(batch_id)
        return batch_id, results
    except Exception as e:
        print(f"   ⚠️  Retry batch failed: {str(e)[:100]}")
        # On error, mark all as needing retry (orchestrator will decide next step)
        return None, {email: {"needs_retry": True, "error": str(e)} for email in emails}


# ============================================================================
# Stage 0-2 Extraction for Batch Processing
# ============================================================================
# This function extracts Stage 0, 1, and 2 from run_automated_checks() to
# allow parallel execution with batch email verification.
# See tasks9.md for the full migration plan.
# ============================================================================

async def run_stage0_2_checks(lead: dict) -> Tuple[bool, dict]:
    """
    Run Stage 0, 1, and 2 checks only (no email verification).
    
    This function is extracted from run_automated_checks() to support
    batch email verification. It runs all checks BEFORE Stage 3 (email
    verification), which is handled separately by the batch process.
    
    The actual check functions are IDENTICAL to run_automated_checks() -
    only the orchestration is different.
    
    Stages included:
    - Pre-checks: Source provenance verification
    - Stage 0: Required fields, email regex, name-email match, 
               general purpose email, free email, disposable, HEAD request
    - Stage 1: Domain age, MX record, SPF/DMARC (parallel)
    - Stage 2: DNSBL reputation check
    
    Args:
        lead: Lead dict with all fields
    
    Returns:
        Tuple[bool, dict]: (passed, partial_automated_checks_data)
            - If passed: (True, data with stage_0, stage_1, stage_2 populated)
            - If failed: (False, data with rejection_reason)
    
    Note:
        This function does NOT run Stage 3 (email verification).
        Email verification is handled by the batch process (submit_truelist_batch
        + poll_truelist_batch).
    """
    email = get_email(lead)
    company = get_company(lead)
    
    # Initialize structured data collection (same structure as run_automated_checks)
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
        "stage_5_verification": {
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None
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
    # Gateway verifies attestations during POST /submit.
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
    # ========================================================================
    print(f"🔍 Stage 0: Hardcoded checks for {email} @ {company}")
    
    # OPTIMIZATION: Run instant checks first, then overlap HEAD request with Stage 1 DNS checks
    checks_stage0_instant = [
        check_required_fields,      # Required fields validation (HARD)
        check_email_regex,          # RFC-5322 regex validation (HARD)
        check_name_email_match,     # Name in email check (HARD)
        check_general_purpose_email,# General purpose email filter (HARD)
        check_free_email_domain,    # Reject free email domains (HARD)
        check_disposable,           # Filter throwaway email providers (HARD)
    ]

    for check_func in checks_stage0_instant:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 0 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    # Collect Stage 0 data after successful instant checks
    automated_checks_data["stage_0_hardcoded"]["name_in_email"] = True
    automated_checks_data["stage_0_hardcoded"]["is_general_purpose_email"] = False

    print("   ✅ Stage 0 instant checks passed")
    
    # OPTIMIZATION: Start HEAD request as background task (will check result after Stage 1)
    head_request_task = asyncio.create_task(check_head_request(lead))

    # ========================================================================
    # Stage 1: DNS Layer (MIXED)
    # - Domain Age, MX Record (HARD)
    # - SPF/DMARC (SOFT - always passes, appends data)
    # ========================================================================
    print(f"🔍 Stage 1: DNS layer checks for {email} @ {company}")
    
    # OPTIMIZATION: Run all Stage 1 DNS checks in parallel
    results = await asyncio.gather(
        check_domain_age(lead),
        check_mx_record(lead),
        check_spf_dmarc(lead),
        return_exceptions=True
    )
    
    # Check results
    check_names = ["check_domain_age", "check_mx_record", "check_spf_dmarc"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"   ❌ Stage 1 failed: {str(result)}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = {
                "stage": "Stage 1: DNS Layer",
                "check_name": check_names[i],
                "message": f"Check failed: {str(result)}",
                "failed_fields": ["domain"]
            }
            # Collect partial Stage 1 data even on failure
            automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", False)
            automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
            automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
            automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data
        
        passed, rejection_reason = result
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
            automated_checks_data["stage_2_domain"]["domain_age_days"] = lead.get("domain_age_days")
            automated_checks_data["stage_2_domain"]["domain_registrar"] = lead.get("domain_registrar")
            automated_checks_data["stage_2_domain"]["domain_nameservers"] = lead.get("domain_nameservers")
            automated_checks_data["stage_2_domain"]["whois_updated_days_ago"] = lead.get("whois_updated_days_ago")
            return False, automated_checks_data

    # Collect Stage 1 DNS data after successful checks
    automated_checks_data["stage_1_dns"]["has_mx"] = lead.get("has_mx", True)
    automated_checks_data["stage_1_dns"]["has_spf"] = lead.get("has_spf", False)
    automated_checks_data["stage_1_dns"]["has_dmarc"] = lead.get("has_dmarc", False)
    automated_checks_data["stage_1_dns"]["dmarc_policy"] = "strict" if lead.get("dmarc_policy_strict") else "none"

    print("   ✅ Stage 1 passed")

    # ========================================================================
    # Stage 0 (continued): HEAD Request Check
    # Check result of background HEAD request task
    # ========================================================================
    print(f"🔍 Stage 0: Website HEAD request check for {email} @ {company}")
    passed, rejection_reason = await head_request_task
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 0 (HEAD request) failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data
    
    print("   ✅ Stage 0 (HEAD request) passed")

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
    # STOP HERE - Stage 3 (email verification) is handled by batch process
    # ========================================================================
    # Mark as passed up to Stage 2
    # The batch orchestrator will handle email verification separately
    automated_checks_data["passed"] = True  # Passed Stage 0-2
    
    print(f"   ✅ Stage 0-2 complete for {email} @ {company}")
    return True, automated_checks_data


async def run_stage4_5_repscore(
    lead: dict,
    email_result: dict,
    stage0_2_data: dict
) -> Tuple[bool, dict]:
    """
    Run Stage 4, Stage 5, and Rep Score checks only.
    
    This function is extracted from run_automated_checks() to support
    batch email verification. It runs AFTER the lead has passed both:
    1. TrueList batch email verification (email_result)
    2. Stage 0-2 checks (stage0_2_data from run_stage0_2_checks)
    
    The actual check functions (check_linkedin_gse, check_stage5_unified,
    check_wayback_machine, etc.) are called EXACTLY as in run_automated_checks().
    
    Args:
        lead: Lead dict with email, company, linkedin, etc.
        email_result: Result from TrueList batch for this email
                     {"status": "email_ok", "passed": True, "rejection_reason": None}
        stage0_2_data: Partial automated_checks_data from run_stage0_2_checks()
    
    Returns:
        Tuple[bool, dict]: (passed, complete_automated_checks_data)
    """
    email = get_email(lead)
    company = get_company(lead)
    
    # ========================================================================
    # MERGE: Start with stage0_2_data and extend with Stage 3-5 + Rep Score
    # ========================================================================
    automated_checks_data = stage0_2_data.copy()
    
    # Ensure Stage 3-5 and rep_score sections exist
    if "stage_3_email" not in automated_checks_data:
        automated_checks_data["stage_3_email"] = {
            "email_status": "unknown",
            "email_score": 0,
            "is_disposable": False,
            "is_role_based": False,
            "is_free": False
        }
    if "stage_4_linkedin" not in automated_checks_data:
        automated_checks_data["stage_4_linkedin"] = {
            "linkedin_verified": False,
            "gse_search_count": 0,
            "llm_confidence": "none"
        }
    if "stage_5_verification" not in automated_checks_data:
        automated_checks_data["stage_5_verification"] = {
            "role_verified": False,
            "region_verified": False,
            "industry_verified": False,
            "extracted_role": None,
            "extracted_region": None,
            "extracted_industry": None,
            "early_exit": None
        }
    if "rep_score" not in automated_checks_data:
        automated_checks_data["rep_score"] = {
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
        }
    
    # ========================================================================
    # Stage 3: Populate from Batch Email Result (NO API CALL - already done)
    # ========================================================================
    print(f"🔍 Stage 3: Email verification (from batch) for {email} @ {company}")
    
    # Map TrueList batch status to internal format for lead["email_verifier_status"]
    # This matches the mapping in run_automated_checks() Stage 3 data collection
    batch_status = email_result.get("status", "unknown")
    
    if batch_status == "email_ok":
        lead["email_verifier_status"] = "Valid"
        email_status = "valid"
        email_passed = True
    elif batch_status == "accept_all":
        lead["email_verifier_status"] = "Catch-All"
        email_status = "catch-all"
        # Catch-all passes only if SPF exists (checked in Stage 1)
        has_spf = automated_checks_data.get("stage_1_dns", {}).get("has_spf", False)
        email_passed = has_spf
    elif batch_status in ["disposable"]:
        lead["email_verifier_status"] = "Disposable"
        email_status = "invalid"
        email_passed = False
    elif batch_status in ["failed_no_mailbox", "failed_syntax_check", "failed_mx_check"]:
        lead["email_verifier_status"] = "Invalid"
        email_status = "invalid"
        email_passed = False
    else:
        # unknown, timeout, error - should have been retried, treat as failure
        lead["email_verifier_status"] = "Unknown"
        email_status = "unknown"
        email_passed = False
    
    # Populate batch result flags on lead (for downstream compatibility)
    lead["email_verifier_disposable"] = email_result.get("is_disposable", False)
    lead["email_verifier_role_based"] = email_result.get("is_role_based", False)
    lead["email_verifier_free"] = email_result.get("is_free", False)
    
    # Collect Stage 3 email data
    automated_checks_data["stage_3_email"]["email_status"] = email_status
    automated_checks_data["stage_3_email"]["email_score"] = 10 if email_passed else 0
    automated_checks_data["stage_3_email"]["is_disposable"] = lead.get("email_verifier_disposable", False)
    automated_checks_data["stage_3_email"]["is_role_based"] = lead.get("email_verifier_role_based", False)
    automated_checks_data["stage_3_email"]["is_free"] = lead.get("email_verifier_free", False)
    
    if not email_passed:
        rejection_reason = email_result.get("rejection_reason") or {
            "stage": "Stage 3: Email Verification (Batch)",
            "check_name": "truelist_batch",
            "message": f"Email verification failed: {batch_status}",
            "failed_fields": ["email"]
        }
        print(f"   ❌ Stage 3 failed: {rejection_reason.get('message', 'Email verification failed')}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data
    
    print("   ✅ Stage 3 passed (batch verified)")
    
    # ========================================================================
    # Stage 4: LinkedIn/GSE Validation (HARD)
    # EXTRACTED VERBATIM from run_automated_checks()
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
    # EXTRACTED VERBATIM from run_automated_checks()
    # - Uses ScrapingDog search + fuzzy matching + LLM to verify role, region, industry
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
    # EXTRACTED VERBATIM from run_automated_checks()
    # - Wayback Machine (max 6 points), SEC (max 12 points), 
    #   WHOIS/DNSBL (max 10 points), GDELT Press/Media (max 10 points),
    #   Companies House (max 10 points)
    # - Always passes, appends scores to lead
    # - Total: 0-48 points
    # ========================================================================
    print(f"📊 Rep Score: Running soft checks for {email} @ {company} (parallel execution)")
    
    # OPTIMIZATION: Run all rep score checks in parallel to save time
    # Old: Sequential execution = 6-12s total
    # New: Parallel execution = 3-4s total (time of slowest API)
    results = await asyncio.gather(
        check_wayback_machine(lead),
        check_sec_edgar(lead),
        check_whois_dnsbl_reputation(lead),
        check_gdelt_mentions(lead),
        check_companies_house(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )
    
    # Unpack results (handle exceptions gracefully)
    wayback_score, wayback_data = results[0] if not isinstance(results[0], Exception) else (0, {"error": str(results[0])})
    sec_score, sec_data = results[1] if not isinstance(results[1], Exception) else (0, {"error": str(results[1])})
    whois_dnsbl_score, whois_dnsbl_data = results[2] if not isinstance(results[2], Exception) else (0, {"error": str(results[2])})
    gdelt_score, gdelt_data = results[3] if not isinstance(results[3], Exception) else (0, {"error": str(results[3])})
    companies_house_score, companies_house_data = results[4] if not isinstance(results[4], Exception) else (0, {"error": str(results[4])})
    
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
    
    # ========================================================================
    # ICP Adjustment Calculation (NEW SYSTEM - Absolute Points)
    # Replaces the old multiplier system with absolute point adjustments
    # ========================================================================
    icp_adjustment = calculate_icp_adjustment(lead)
    # Store in is_icp_multiplier field for backwards compatibility
    # Values: -15 to +20 (new format) vs 1.0/1.5/5.0 (old format)
    lead["is_icp_multiplier"] = float(icp_adjustment)
    automated_checks_data["is_icp_multiplier"] = float(icp_adjustment)

    # ========================================================================
    # Company Name Standardization (only on approval)
    # ========================================================================
    # Use the company LinkedIn slug to get/set the standardized company name.
    # This ensures all leads with the same company_linkedin URL have the same
    # standardized company name, regardless of how the miner submitted it.
    # ========================================================================
    company_slug = lead.get("company_linkedin_slug")
    company_linkedin_data = lead.get("company_linkedin_data")

    if company_slug:
        # Check cache first
        standardized_name = get_standardized_company_name(company_slug)

        if standardized_name:
            # Cache hit - use cached standardized name
            print(f"   📦 Company name from cache: '{company_slug}' → '{standardized_name}'")
        else:
            # Cache miss - get from Stage 4 scraped data and save to cache
            if company_linkedin_data and company_linkedin_data.get("company_name_from_linkedin"):
                standardized_name = company_linkedin_data["company_name_from_linkedin"]
                set_standardized_company_name(company_slug, standardized_name)
            else:
                # Fallback to miner's submitted company name if no scraped data
                standardized_name = company
                print(f"   ⚠️ No scraped company name available, using submitted: '{standardized_name}'")

        # Set on lead and automated_checks_data
        lead["company_standardized"] = standardized_name
        automated_checks_data["company_standardized"] = standardized_name
        print(f"   ✅ Company standardized: '{company}' → '{standardized_name}'")
    else:
        # No company_linkedin_slug - use submitted company name
        lead["company_standardized"] = company
        automated_checks_data["company_standardized"] = company
        print(f"   ⚠️ No company LinkedIn slug, using submitted name: '{company}'")

    print(f"🎉 All stages passed for {email} @ {company}")

    # All checks passed - return structured success data
    automated_checks_data["passed"] = True
    automated_checks_data["rejection_reason"] = None

    # IMPORTANT: Also set rep_score on lead object for validator.py to pick up
    # validator.py looks for lead_blob.get("rep_score", 50)
    lead["rep_score"] = total_rep_score

    return True, automated_checks_data


async def submit_and_poll_truelist(emails: List[str]) -> Tuple[str, Dict[str, dict]]:
    """
    Submit batch and poll for results (combined for background task).
    
    This is a helper function that combines submit_truelist_batch() and
    poll_truelist_batch() for use with asyncio.create_task().
    
    Args:
        emails: List of email addresses to validate
    
    Returns:
        Tuple of (batch_id, results_dict) where results_dict maps email -> result
        batch_id is returned so caller can delete the batch before retrying
    """
    batch_id = await submit_truelist_batch(emails)
    results = await poll_truelist_batch(batch_id)
    return batch_id, results


async def delete_truelist_batch(batch_id: str) -> bool:
    """
    Delete a TrueList batch.
    
    IMPORTANT: This must be called before retrying emails that were in the batch.
    TrueList detects duplicate email content and rejects re-submissions.
    Deleting the batch clears TrueList's duplicate detection for those emails.
    
    Args:
        batch_id: The batch ID to delete
    
    Returns:
        True if deleted successfully, False otherwise
    """
    if not batch_id:
        return False
    
    url = f"https://api.truelist.io/api/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                if response.status == 204:
                    print(f"   🗑️ Deleted batch {batch_id[:8]}... (clearing duplicate detection)")
                    return True
                else:
                    print(f"   ⚠️ Failed to delete batch {batch_id[:8]}... (status {response.status})")
                    return False
    except Exception as e:
        print(f"   ⚠️ Error deleting batch: {str(e)[:50]}")
        return False


async def delete_all_truelist_batches() -> int:
    """
    Delete ALL TrueList batches to clear duplicate detection.
    
    CRITICAL: TrueList detects duplicate emails across ALL batches ever submitted.
    Even if a batch is "completed", TrueList remembers the emails and may return
    incomplete CSV results for subsequent batches containing the same emails.
    
    This function queries all batches and deletes them one by one.
    Should be called before submitting a new batch in each epoch.
    
    Returns:
        Number of batches deleted
    """
    url = "https://api.truelist.io/api/v1/batches"
    headers = {"Authorization": f"Bearer {TRUELIST_API_KEY}"}
    deleted_count = 0
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get list of all batches
            async with session.get(url, headers=headers, timeout=30, proxy=HTTP_PROXY_URL) as response:
                if response.status != 200:
                    print(f"   ⚠️ Failed to list batches (status {response.status})")
                    return 0
                
                data = await response.json()
                batches = data.get("batches", [])
                
                if not batches:
                    print(f"   ✅ No old batches to delete")
                    return 0
                
                print(f"   🗑️ Deleting {len(batches)} old TrueList batches to clear duplicate detection...")
                
                # Delete each batch
                for batch in batches:
                    batch_id = batch.get("id")
                    if batch_id:
                        delete_url = f"{url}/{batch_id}"
                        try:
                            async with session.delete(delete_url, headers=headers, timeout=10, proxy=HTTP_PROXY_URL) as del_response:
                                if del_response.status == 204:
                                    deleted_count += 1
                        except Exception:
                            pass  # Silently skip failed deletes
                
                print(f"   ✅ Deleted {deleted_count}/{len(batches)} batches")
                return deleted_count
                
    except Exception as e:
        print(f"   ⚠️ Error deleting batches: {str(e)[:50]}")
        return deleted_count


async def retry_truelist_batch(emails: List[str], prev_batch_id: str = None) -> Tuple[str, Dict[str, dict]]:
    """
    Submit a retry batch and poll for results.
    
    IMPORTANT: If prev_batch_id is provided, deletes it first to clear
    TrueList's duplicate detection before submitting the retry.
    
    Args:
        emails: List of email addresses to retry
        prev_batch_id: Optional batch_id from previous retry to delete first
    
    Returns:
        Tuple of (batch_id, results_dict)
        On error, returns (None, {email: needs_retry=True})
    """
    try:
        # Delete previous retry batch if provided (clears duplicate detection)
        if prev_batch_id:
            await delete_truelist_batch(prev_batch_id)
        
        batch_id = await submit_truelist_batch(emails)
        results = await poll_truelist_batch(batch_id)
        return batch_id, results
    except Exception as e:
        print(f"   ⚠️ Retry batch error: {e}")
        # On error, mark all as needing retry
        return None, {email: {"needs_retry": True, "error": str(e)} for email in emails}


async def run_centralized_truelist_batch(leads: List[dict]) -> Dict[str, dict]:
    """
    COORDINATOR ONLY: Run TrueList batch on ALL leads at once.
    
    This function extracts all emails from leads, submits them to TrueList,
    handles retries (up to 3 times), and falls back to inline verification.
    
    The coordinator calls this BEFORE distributing leads to workers.
    Workers then receive the precomputed results with their leads.
    
    Flow:
    1. Extract all valid emails from leads
    2. Delete old TrueList batches (clean slate)
    3. Submit batch with all emails
    4. Poll for completion
    5. Retry any emails with errors (up to 3 times total)
    6. Fall back to inline verification for remaining errors
    7. Return complete results dict
    
    Args:
        leads: List of ALL lead dicts from gateway (e.g., 2700 leads)
    
    Returns:
        Dict mapping email (lowercase) -> result dict with:
        - passed: bool
        - status: str (email_ok, failed_*, etc.)
        - needs_retry: bool (if unresolved)
        - rejection_reason: dict (if failed)
    
    NOTE: This function is ONLY called by the coordinator.
    Workers should use precomputed_email_results parameter of run_batch_automated_checks.
    """
    print(f"\n{'='*60}")
    print(f"📧 COORDINATOR: Centralized TrueList batch for {len(leads)} leads")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # ========================================================================
    # Step 1: Extract all valid emails
    # ========================================================================
    emails = []
    email_to_lead_idx = {}  # Track which lead each email came from (for debugging)
    
    for i, lead in enumerate(leads):
        # Handle both formats: {"lead_blob": {...}} wrapper OR flat lead dict
        lead_blob = lead.get("lead_blob", lead) if isinstance(lead, dict) else lead
        email = get_email(lead_blob)
        if email and '@' in email:
            email_lower = email.lower()
            emails.append(email_lower)
            email_to_lead_idx[email_lower] = i
    
    print(f"   📧 Extracted {len(emails)} valid emails from {len(leads)} leads")
    
    if not emails:
        print(f"   ⚠️ No valid emails found - returning empty results")
        return {}
    
    # ========================================================================
    # Step 2: Clean up old TrueList batches
    # ========================================================================
    print(f"   🧹 Cleaning up old TrueList batches...")
    await delete_all_truelist_batches()
    
    # ========================================================================
    # Step 3: Submit batch and poll (with retries)
    # ========================================================================
    email_results = {}
    batch_id = None
    
    # Try batch up to 3 times total
    for batch_attempt in range(3):
        try:
            print(f"   🚀 Submitting TrueList batch (attempt {batch_attempt + 1}/3) for {len(emails)} emails...")
            
            batch_id = await submit_truelist_batch(emails)
            results = await poll_truelist_batch(batch_id)
            
            # Merge results
            email_results.update(results)
            
            # Check for emails that need retry
            needs_retry = []
            for email in emails:
                result = email_results.get(email)
                if result is None or result.get("needs_retry"):
                    needs_retry.append(email)
            
            print(f"   ✅ Batch {batch_attempt + 1} complete: {len(results)} results, {len(needs_retry)} need retry")
            
            if not needs_retry:
                # All emails resolved
                break
            
            if batch_attempt < 2:
                # More retries available
                print(f"   🔄 Retrying {len(needs_retry)} emails in 10s...")
                
                # Delete batch before retry (clears duplicate detection)
                if batch_id:
                    await delete_truelist_batch(batch_id)
                    batch_id = None
                
                await asyncio.sleep(10)
                emails = needs_retry  # Only retry failed emails
            
        except Exception as e:
            print(f"   ❌ TrueList batch attempt {batch_attempt + 1} failed: {e}")
            
            # Delete batch before retry
            if batch_id:
                try:
                    await delete_truelist_batch(batch_id)
                except:
                    pass
                batch_id = None
            
            if batch_attempt < 2:
                await asyncio.sleep(10 * (batch_attempt + 1))
    
    # ========================================================================
    # Step 4: Inline fallback for remaining errors
    # ========================================================================
    needs_inline = []
    for email in emails:
        result = email_results.get(email)
        if result is None or result.get("needs_retry"):
            needs_inline.append(email)
    
    if needs_inline:
        print(f"   🔄 Falling back to inline verification for {len(needs_inline)} emails...")
        
        try:
            inline_results = await verify_emails_inline(needs_inline)
            email_results.update(inline_results)
            print(f"   ✅ Inline verification complete: {len(inline_results)} results")
        except Exception as e:
            print(f"   ❌ Inline verification failed: {e}")
            # Mark remaining as unresolved
            for email in needs_inline:
                if email not in email_results or email_results[email].get("needs_retry"):
                    email_results[email] = {
                        "needs_retry": True,
                        "error": f"All verification methods failed: {str(e)}"
                    }
    
    # ========================================================================
    # Step 5: Summary
    # ========================================================================
    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)
    
    passed = sum(1 for r in email_results.values() if r.get("passed"))
    failed = sum(1 for r in email_results.values() if not r.get("passed") and not r.get("needs_retry"))
    unresolved = sum(1 for r in email_results.values() if r.get("needs_retry"))
    
    print(f"\n{'='*60}")
    print(f"📊 CENTRALIZED TRUELIST COMPLETE")
    print(f"{'='*60}")
    print(f"   📦 Total leads from gateway: {len(leads)}")
    print(f"   📧 Total emails processed: {len(email_results)}")
    print(f"   ✅ Passed (email_ok): {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Unresolved: {unresolved}")
    print(f"   ⏱️  TIME: {elapsed_mins}m {elapsed_secs}s ({elapsed:.1f} seconds total)")
    print(f"{'='*60}\n")
    
    return email_results


async def run_batch_automated_checks(
    leads: List[dict],
    container_id: int = 0,
    precomputed_email_results: Dict[str, dict] = None,
    leads_file_path: str = None
) -> List[Tuple[bool, dict]]:
    """
    Batch validation with SEQUENTIAL Stage 0-2 and Stage 4-5.
    Stage 0-2 runs IN PARALLEL with coordinator's centralized TrueList batch.
    
    This REPLACES calling run_automated_checks() individually for each lead.
    Orchestrates the full batch flow without modifying any actual validation checks.
    
    Flow (when leads_file_path is provided - worker/coordinator polling mode):
    1. Run Stage 0-2 SEQUENTIALLY for all leads
    2. POLL leads_file_path for truelist_results (coordinator updates file when done)
    3. Use polled results for Stage 4-5
    
    Flow (when precomputed_email_results is provided - already has results):
    1. Run Stage 0-2 SEQUENTIALLY for all leads
    2. Use precomputed email results directly
    3. Run Stage 4-5 SEQUENTIALLY
    
    Args:
        leads: List of lead dicts (e.g., 110 leads per container)
        container_id: Container ID (0-29) for logging.
        precomputed_email_results: Dict mapping email (lowercase) -> result dict.
                                   If provided, skip polling and use these directly.
        leads_file_path: Path to shared leads file for polling truelist_results.
                         If provided, poll this file after Stage 0-2 until truelist_results is available.
    
    Returns:
        List of (passed, automated_checks_data) tuples in SAME ORDER as input
        - passed: True (approved), False (rejected), or None (skipped)
    
    CRITICAL: Results are returned in the SAME ORDER as input leads.
    """
    print(f"📦 Starting batch validation for {len(leads)} leads")
    start_time = time.time()
    
    n = len(leads)
    
    # Handle empty batch
    if n == 0:
        print("   ⚠️ Empty batch - nothing to validate")
        return []
    
    # Initialize results array with None (will be filled in order)
    results = [None] * n  # Index-based for order preservation
    
    # ========================================================================
    # Step 1: Extract emails and build lookup maps
    # ========================================================================
    emails = []
    email_to_idx = {}  # email (lowercase) -> index in leads list
    
    for i, lead in enumerate(leads):
        email = get_email(lead)
        if email:
            email_lower = email.lower()  # Normalize to lowercase for matching with CSV results
            emails.append(email_lower)
            email_to_idx[email_lower] = i
        else:
            # No email - immediate rejection
            results[i] = (False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Pre-Batch",
                    "check_name": "email_extraction",
                    "message": "No email found in lead",
                                "failed_fields": ["email"]
                }
            })
    
    print(f"   📧 Extracted {len(emails)} emails from {n} leads")
    
    # Check if all leads rejected (no valid emails)
    if not emails:
        print("   ⚠️ No valid emails found - all leads rejected")
        return results
    
    # ========================================================================
    # Step 1.5: FAST PRE-FILTER - Reject emails without @ sign
    # ========================================================================
    # This is a super low-latency check that:
    # 1. Prevents TrueList batch from failing (it rejects entire batch if ANY email lacks @)
    # 2. Saves Stage 0-2 processing time for obviously invalid emails
    
    valid_emails = []
    invalid_syntax_count = 0
    
    for email in emails:
        if '@' not in email:
            # Instant rejection - no @ sign
            idx = email_to_idx[email]
            results[idx] = (False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Pre-Batch",
                    "check_name": "email_syntax_prefilter",
                    "message": "Email missing @ symbol (instant rejection)",
                                "failed_fields": ["email"]
                }
            })
            invalid_syntax_count += 1
        else:
            valid_emails.append(email)
    
    if invalid_syntax_count > 0:
        print(f"   ⚡ Pre-filter: Rejected {invalid_syntax_count} emails (missing @ sign)")
    
    print(f"   ✅ {len(valid_emails)} valid emails ready for batch processing")
    
    # Update email list and check if any remain
    emails = valid_emails
    
    if not emails:
        print("   ⚠️ No valid emails after pre-filter - all leads rejected")
        return results
    
    # ========================================================================
    # Step 2: Determine TrueList results source
    # ========================================================================
    # Centralized TrueList is handled EXTERNALLY by coordinator's background task.
    # This function just runs Stage 0-2, then polls file OR uses precomputed results.
    
    has_precomputed = precomputed_email_results is not None and len(precomputed_email_results) > 0
    needs_polling = leads_file_path is not None and not has_precomputed
    
    if has_precomputed:
        print(f"   📥 Using precomputed TrueList results ({len(precomputed_email_results)} emails)")
    elif needs_polling:
        print(f"   ⏳ Will poll {leads_file_path} for TrueList results after Stage 0-2")
    else:
        print(f"   ⚠️ No TrueList source - leads will fail email verification")
    
    # ========================================================================
    # Step 2.5: STAGGER DELAY for Stage 0-2 (prevents WHOIS rate limiting)
    # ========================================================================
    # With centralized TrueList, all containers start Stage 0-2 simultaneously.
    # This causes WHOIS servers to rate-limit us (connection resets).
    # Add container-specific delay so WHOIS requests are staggered across containers.
    STAGE0_2_STAGGER_DELAY_SECONDS = 8  # 8s between containers
    stagger_delay = container_id * STAGE0_2_STAGGER_DELAY_SECONDS
    
    if stagger_delay > 0:
        print(f"   ⏳ Container {container_id}: Waiting {stagger_delay}s before Stage 0-2 (staggered WHOIS)...")
        await asyncio.sleep(stagger_delay)
    
    # ========================================================================
    # Step 3: Run Stage 0-2 SEQUENTIALLY (while TrueList batch processes)
    # ========================================================================
    print(f"   🔍 Running Stage 0-2 checks SEQUENTIALLY for {n} leads...")
    
    stage0_2_results = []  # List of (passed, data) in order, indexed by lead position
    
    for i, lead in enumerate(leads):
        email = get_email(lead)
        
        # Skip leads without email (already rejected in Step 1)
        if not email:
            stage0_2_results.append((False, results[i][1] if results[i] else {}))
            continue
        
        print(f"   Stage 0-2: Lead {i+1}/{n} ({email})")
        
        try:
            passed, data = await run_stage0_2_checks(lead)
            stage0_2_results.append((passed, data))
        except Exception as e:
            print(f"      ❌ Stage 0-2 error: {e}")
            stage0_2_results.append((False, {
                "passed": False,
                "rejection_reason": {
                    "stage": "Stage 0-2",
                    "check_name": "run_stage0_2_checks",
                    "message": f"Stage 0-2 error: {str(e)}",
                    "error": str(e)
                }
            }))
        
        # 0.5-second delay between Stage 0-2 leads (rate limiting)
        if i < len(leads) - 1:
            await asyncio.sleep(0.5)
    
    stage0_2_passed_count = sum(1 for passed, _ in stage0_2_results if passed)
    print(f"   ✅ Stage 0-2 complete: {stage0_2_passed_count}/{n} passed")
    
    # ========================================================================
    # Step 4: Get TrueList results (precomputed OR poll file)
    # ========================================================================
    
    email_results = {}
    
    if has_precomputed:
        # Use precomputed results directly
        email_results = precomputed_email_results
        print(f"   ✅ Using precomputed email results: {len(email_results)} emails")
    elif needs_polling:
        # POLL the leads file until truelist_results is available (not None)
        print(f"   ⏳ Polling for TrueList results from coordinator...")
        
        poll_interval = 5  # seconds
        max_poll_time = 1200  # 20 minutes max wait
        poll_start = time.time()
        poll_waited = 0
        
        while True:
            try:
                import json
                with open(leads_file_path, 'r') as f:
                    file_data = json.load(f)
                    file_truelist = file_data.get("truelist_results")
                    
                    if file_truelist is not None:
                        # Results available (dict, possibly empty if coordinator failed)
                        email_results = file_truelist
                        print(f"   ✅ Received TrueList results from coordinator: {len(email_results)} emails (waited {poll_waited}s)")
                        break
                    else:
                        # Still None = in progress
                        if poll_waited % 30 == 0 and poll_waited > 0:
                            print(f"   ⏳ Still waiting for TrueList... ({poll_waited}s elapsed)")
            except Exception as e:
                print(f"   ⚠️ Error reading leads file: {e}")
            
            await asyncio.sleep(poll_interval)
            poll_waited += poll_interval
            
            if poll_waited >= max_poll_time:
                print(f"   ❌ Timeout waiting for TrueList results ({max_poll_time}s)")
                print(f"   ⚠️ Leads will fail email verification")
                email_results = {}
                break
    else:
        # No source - all leads fail email verification
        print(f"   ⚠️ No TrueList results available - leads will fail email verification")
        email_results = {}
    
    # ========================================================================
    # Step 5: Categorize leads
    # ========================================================================
    stage4_5_queue = []  # List of (index, lead, email_result, stage0_2_data)
    needs_retry = []     # List of emails that errored
    
    for i, lead in enumerate(leads):
        email = get_email(lead)
        
        # Skip leads without email (already rejected)
        if not email:
            continue
        
        email_lower = email.lower()  # Use lowercase for lookup (CSV results are lowercase)
        stage0_2_passed, stage0_2_data = stage0_2_results[i]
        email_result = email_results.get(email_lower, None)  # None if not in results
        
        if not stage0_2_passed:
            # Failed Stage 0-2 → immediate reject
            results[i] = (False, stage0_2_data)
        elif email_result is None:
            # Email NOT IN results at all
            if has_precomputed or needs_polling:
                # Using precomputed/polled results: Coordinator couldn't verify this email → skip
                # (Coordinator has already done retries, so we trust the absence)
                results[i] = (None, {
                    "skipped": True,
                    "reason": "EmailNotInPrecomputedResults",
                    "message": "Coordinator could not verify this email"
                })
            else:
                # No external results: Queue for retry
                # This happens when TrueList's CSV doesn't include all emails
                needs_retry.append(email_lower)
        elif email_result.get("needs_retry"):
            # Email explicitly errored
            if has_precomputed or needs_polling:
                # WORKER MODE: Coordinator marked as needing retry but couldn't resolve → skip
                results[i] = (None, {
                    "skipped": True,
                    "reason": "EmailVerificationIncomplete",
                    "message": "Coordinator could not complete email verification"
                })
            else:
                # COORDINATOR MODE: Queue for retry
                needs_retry.append(email_lower)
        elif email_result.get("passed"):
            # Both passed → queue for Stage 4-5
            stage4_5_queue.append((i, lead, email_result, stage0_2_data))
        else:
            # Email explicitly failed (has status) → reject
            rejection_data = stage0_2_data.copy()
            rejection_data["passed"] = False
            rejection_data["rejection_reason"] = email_result.get("rejection_reason") or {
                "stage": "Stage 3: Email Verification (Batch)",
                "check_name": "truelist_batch",
                "message": f"Email verification failed: {email_result.get('status', 'unknown')}",
                                "failed_fields": ["email"]
            }
            results[i] = (False, rejection_data)
    
    print(f"   📊 Categorization: {len(stage4_5_queue)} ready for Stage 4-5, {sum(1 for r in results if r and r[0] == False)} rejected, {len(needs_retry)} need retry")
    
    # ========================================================================
    # Step 6: Start Stage 4-5 SEQUENTIALLY + Handle retries in parallel
    # ========================================================================
    
    # Start retry batch if needed (runs in background)
    # NOTE: When polling file, retries are handled by coordinator - workers don't retry
    retry_task = None
    inline_task = None  # Inline verification task (runs in background after retries exhaust)
    retry_attempt = 0
    last_retry_batch_id = None  # Track retry batch_id for deletion before next retry
    
    if needs_retry and not has_precomputed and not needs_polling:
        # COORDINATOR MODE ONLY: Retry failed emails
        # CRITICAL: Delete the original batch BEFORE retrying
        # TrueList detects duplicate email content and rejects re-submissions.
        # Deleting the batch clears their duplicate detection for those emails.
        if original_batch_id:
            await delete_truelist_batch(original_batch_id)
        
        print(f"   🔄 Starting retry batch #1 for {len(needs_retry)} emails...")
        retry_task = asyncio.create_task(retry_truelist_batch(needs_retry, None))
    
    # Process Stage 4-5 queue SEQUENTIALLY
    queue_idx = 0
    total_stage4_5 = len(stage4_5_queue)
    
    while queue_idx < len(stage4_5_queue) or retry_task is not None or inline_task is not None:
        # Process next lead in Stage 4-5 queue (if available)
        if queue_idx < len(stage4_5_queue):
            idx, lead, email_result, stage0_2_data = stage4_5_queue[queue_idx]
            email = get_email(lead)
            print(f"   Stage 4-5: Lead {queue_idx+1}/{len(stage4_5_queue)} ({email})")
            
            try:
                passed, data = await run_stage4_5_repscore(lead, email_result, stage0_2_data)
                results[idx] = (passed, data)
            except Exception as e:
                print(f"      ❌ Stage 4-5 error: {e}")
                results[idx] = (False, {
                    "passed": False,
                    "rejection_reason": {
                        "stage": "Stage 4-5",
                        "check_name": "run_stage4_5_repscore",
                        "message": f"Stage 4-5 error: {str(e)}",
                        "error": str(e)
                    }
                })
            
            queue_idx += 1
            
            # No delay between Stage 4-5 leads - ScrapingDog/OpenRouter can handle it
            # (Stage 0-2 still has 1s delay for DNS/HEAD request rate limiting)
        
        # Check if retry batch completed (non-blocking check)
        if retry_task is not None and retry_task.done():
            try:
                last_retry_batch_id, retry_results = retry_task.result()
            except Exception as e:
                print(f"   ⚠️ Retry batch failed: {e}")
                last_retry_batch_id = None
                retry_results = {email: {"needs_retry": True, "error": str(e)} for email in needs_retry}
            
            retry_task = None
            still_needs_retry = []
            
            for email in needs_retry:
                result = retry_results.get(email, {"needs_retry": True})
                idx = email_to_idx[email]
                stage0_2_passed, stage0_2_data = stage0_2_results[idx]
                
                if result.get("needs_retry"):
                    still_needs_retry.append(email)
                elif result.get("passed"):
                    # Retry succeeded → add to Stage 4-5 queue
                    print(f"   ✅ Retry succeeded for: {email}")
                    stage4_5_queue.append((idx, leads[idx], result, stage0_2_data))
                else:
                    # Retry failed → reject
                    rejection_data = stage0_2_data.copy()
                    rejection_data["passed"] = False
                    rejection_data["rejection_reason"] = result.get("rejection_reason") or {
                        "stage": "Stage 3: Email Verification (Batch Retry)",
                        "check_name": "truelist_batch",
                        "message": f"Email verification failed after retry: {result.get('status', 'unknown')}",
                        "failed_fields": ["email"]
                    }
                    results[idx] = (False, rejection_data)
            
            needs_retry = still_needs_retry
            retry_attempt += 1
            
            print(f"   📊 After retry #{retry_attempt}: {len(still_needs_retry)} still pending, {len(stage4_5_queue) - queue_idx} added to queue")
            
            # Start next retry if needed and haven't exceeded max
            if needs_retry and retry_attempt < TRUELIST_BATCH_MAX_RETRIES:
                print(f"   🔄 Starting retry batch #{retry_attempt+1} for {len(needs_retry)} emails...")
                # Pass the previous retry batch_id so it gets deleted before new submission
                retry_task = asyncio.create_task(retry_truelist_batch(needs_retry, last_retry_batch_id))
            elif needs_retry and retry_attempt >= TRUELIST_BATCH_MAX_RETRIES:
                # ================================================================
                # INLINE FALLBACK: Retries exhausted, start inline in BACKGROUND
                # Stage 4-5 continues processing while inline runs
                # ================================================================
                print(f"   🔍 Starting inline verification for {len(needs_retry)} emails (retries exhausted)...")
                inline_task = asyncio.create_task(verify_emails_inline(needs_retry))
        
        # Check if inline verification completed (non-blocking check)
        if inline_task is not None and inline_task.done():
            try:
                inline_results = inline_task.result()
            except Exception as e:
                print(f"   ⚠️ Inline verification failed: {e}")
                inline_results = {email: {"needs_retry": True, "error": str(e)} for email in needs_retry}
            
            inline_task = None
            
            for email in needs_retry:
                idx = email_to_idx[email]
                stage0_2_passed, stage0_2_data = stage0_2_results[idx]
                result = inline_results.get(email.lower(), {"needs_retry": True})
                
                if result.get("passed"):
                    # Inline passed → add to Stage 4-5 queue
                    print(f"   ✅ Inline verified: {email}")
                    stage4_5_queue.append((idx, leads[idx], result, stage0_2_data))
                elif result.get("needs_retry"):
                    # Still can't verify → skip
                    print(f"   ⏭️ Cannot verify (batch + inline failed): {email}")
                    results[idx] = (None, {
                        "skipped": True,
                        "reason": "EmailVerificationUnavailable",
                        "message": f"Email verification unavailable after batch + inline"
                    })
                else:
                    # Inline explicitly failed → reject
                    rejection_data = stage0_2_data.copy()
                    rejection_data["passed"] = False
                    rejection_data["rejection_reason"] = result.get("rejection_reason") or {
                        "stage": "Stage 3: Email Verification (Inline)",
                        "check_name": "truelist_inline",
                        "message": f"Email failed inline verification: {result.get('status', 'unknown')}",
                        "failed_fields": ["email"]
                    }
                    results[idx] = (False, rejection_data)
            
            # Clear needs_retry since we've handled them
            needs_retry = []
            print(f"   📊 After inline: {len(stage4_5_queue) - queue_idx} leads added to Stage 4-5 queue")
        
        # If queue is empty but tasks pending, wait briefly before checking again
        if queue_idx >= len(stage4_5_queue) and (retry_task is not None or inline_task is not None):
            await asyncio.sleep(1)
    
    # ========================================================================
    # Step 7: (Moved) Inline verification now happens inside the while loop
    # immediately after retries are exhausted, so leads get added to Stage 4-5 queue
    # ========================================================================
    
    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = time.time() - start_time
    passed_count = sum(1 for r in results if r and r[0] is True)
    failed_count = sum(1 for r in results if r and r[0] is False)
    skipped_count = sum(1 for r in results if r and r[0] is None)
    
    print(f"📦 Batch validation complete in {elapsed:.1f}s")
    print(f"   ✅ Passed: {passed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏭️ Skipped: {skipped_count}")
    
    return results


# NOTE: check_myemailverifier_email() has been REMOVED as of Dec 2024.
# All email validation now uses TrueList BATCH API via run_batch_automated_checks().

# Stage 4: LinkedIn/GSE Validation

async def search_linkedin_gse(full_name: str, company: str, linkedin_url: str = None, max_results: int = 5) -> Tuple[List[dict], bool]:
    """
    Search LinkedIn using ScrapingDog Google Search API.
    
    Uses 3 search variations:
    1. Exact URL in quotes (most specific)
    2. Profile slug only (handles www/protocol differences)
    3. Name + company + "linkedin" (broadest fallback)

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
    
    if not SCRAPINGDOG_API_KEY:
        raise Exception("SCRAPINGDOG_API_KEY not set")
    
    # Extract profile slug from LinkedIn URL
    profile_slug = linkedin_url.split("/in/")[-1].strip("/") if "/in/" in linkedin_url else None
    
    # Track if URL matched exactly (strong identity proof)
    url_match_exact = False
    
    # Build search query variations (in order of specificity)
    query_variations = [
        # 1. Exact URL in quotes (most specific)
        f'"{linkedin_url}"',
        
        # 2. Profile slug only (handles www/protocol differences)
        f'"linkedin.com/in/{profile_slug}"' if profile_slug else None,
        
        # 3. Name + LinkedIn + company (more context)
        f'"{full_name}" linkedin "{company}"',
    ]
    
    # Remove None values
    query_variations = [q for q in query_variations if q]
    
    print(f"   🔍 Trying {len(query_variations)} search variations for LinkedIn profile...")
    
    def _search_linkedin_sync(query: str) -> List[dict]:
        """Synchronous ScrapingDog search helper for Stage 4"""
        try:
            url = "https://api.scrapingdog.com/google"
            params = {
                "api_key": SCRAPINGDOG_API_KEY,
                "query": query,
                "results": max_results
            }
            
            response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG)
            if response.status_code != 200:
                print(f"         ⚠️ GSE API error: HTTP {response.status_code}: {response.text}")
                return []
            
            data = response.json()
            items = []
            
            # Convert ScrapingDog format to standard format
            for item in data.get("organic_results", []):
                items.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                    
            return items
        except Exception as e:
            print(f"         ⚠️ Request error: {str(e)}")
            return []
    
    try:
        # Try each query variation until we get results
        for variation_idx, query in enumerate(query_variations, 1):
            print(f"      🔄 Variation {variation_idx}/{len(query_variations)}: {query[:80]}...")
            
            try:
                # Execute sync request in thread pool to keep function async
                items = await asyncio.to_thread(_search_linkedin_sync, query)
                
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
                            # Normalize slugs for comparison (remove hyphens/underscores)
                            profile_slug_norm = profile_slug.lower().replace("-", "").replace("_", "")
                            
                            # Check exact match first (normalized)
                            exact_match = any(
                                profile_slug_norm == result_slug.lower().replace("-", "").replace("_", "")
                                for result_slug in found_profile_urls
                            )
                            # Also check partial match (profile slug contained in result, normalized)
                            partial_match = any(
                                profile_slug_norm in result_slug.lower().replace("-", "").replace("_", "") or 
                                result_slug.lower().replace("-", "").replace("_", "") in profile_slug_norm
                                    for result_slug in found_profile_urls
                                )
                                
                            if exact_match:
                                print(f"         ✅ URL MATCH: Profile '{profile_slug}' confirmed (exact)")
                                url_match_exact = True  # Strong identity proof!
                            elif partial_match:
                                print(f"         ✅ URL MATCH: Profile '{profile_slug}' confirmed (partial)")
                                url_match_exact = "partial"  # Partial match
                            else:
                                print(f"         ⚠️  URL MISMATCH: Expected '{profile_slug}' but found: {found_profile_urls[:3]}")
                                continue
                    
                    # FILTER 1: Clean up concatenated titles and separate profile headlines from posts
                    # ScrapingDog often concatenates multiple result titles together
                    profile_headlines = []
                    posts = []
                    
                    for item in linkedin_results:
                        title = item.get("title", "")
                        
                        # ScrapingDog concatenates titles - extract only the FIRST profile
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
                        # Skip directory pages (but not profiles that just have this text concatenated)
                        # Check both: title pattern AND link pattern
                        # Check for various LinkedIn directory page patterns
                        is_directory_title = ("profiles | LinkedIn" in title or 
                                             "profiles - LinkedIn" in title or
                                             "profiles on LinkedIn" in title)
                        
                        if is_directory_title:
                            # Only skip if it's actually a directory page (/pub/dir/) or starts with "N+ profiles"
                            link = item.get("link", "")
                            is_directory_link = "/pub/dir/" in link or "/directory/" in link
                            starts_with_profiles = re.match(r'^\d+\+?\s+"?[^"]*"?\s+profiles', title.lower())
                            
                            if is_directory_link or starts_with_profiles:
                                continue  # Skip directory pages
                            # Otherwise, keep it - might be a valid profile with concatenated text
                            
                        profile_headlines.append(item)
                    
                    # FILTER 2: Only keep results for TARGET PERSON (filter out other people)
                    # ScrapingDog often returns concatenated results with multiple profiles
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
                        link = item.get("link", "")
                        
                        # Normalize the title too for accent-insensitive matching
                        title_normalized = normalize_accents(title_lower)
                        
                        # PRIORITY: If this result's URL matches our target profile slug, it's THE profile!
                        # Skip name-in-title check - URL match is authoritative proof of identity
                        if profile_slug and "linkedin.com/in/" in link:
                            result_slug = link.split("/in/")[-1].strip("/").split("?")[0]
                            if profile_slug.lower() == result_slug.lower():
                                target_person_results.append(item)
                                continue  # Skip name check - URL match is definitive
                        
                        # Check if target person's name is in the title (accent-insensitive)
                        # This handles cases like "Jose Varatojo" matching "José Diogo Varatojo"
                        if first_name_normalized in title_normalized and last_name_normalized in title_normalized:
                            target_person_results.append(item)
                        else:
                            other_person_results.append(item)
                    
                    # Prioritize target person's profile headlines
                    if target_person_results:
                        print(f"      📊 GSE Profile Headlines for {full_name}:")
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
                        # This is important because ScrapingDog sometimes returns different people first
                        print(f"      ⚠️ No name match in results (found: {profile_headlines[0].get('title', '')[:50]}...)")
                        print(f"      🔄 Trying next variation to find correct person...")
                        # No delay needed - ScrapingDog has no rate limiting
                        continue  # Try next query variation
                    elif posts:
                        # Only posts found (no profile headlines) - return posts
                        print(f"      📊 ScrapingDog Posts only (no profile headlines found):")
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
                print(f"         ⚠️ Error for variation {variation_idx}: {str(e)}")
                continue
            
            # All variations exhausted
            print(f"   ❌ GSE: No results found after trying {len(query_variations)} variations")
        return [], False
    
    except Exception as e:
        print(f"   ⚠️ GSE API error: {str(e)}")
        return [], False

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
        
        # Normalize apostrophes (ScrapingDog returns "mcdonald ' s" with spaces, we need "mcdonald's")
        # Also handle curly apostrophes and other variants
        company_lower = company_lower.replace("'", "'").replace("'", "'").replace("`", "'")
        company_lower = re.sub(r"\s*'\s*", "'", company_lower)  # "mcdonald ' s" → "mcdonald's"
        company_lower = re.sub(r"\s*-\s*", "-", company_lower)  # "chick - fil - a" → "chick-fil-a"
        
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
        
        # CRITICAL FIX: When URL matched exactly, use the result with matching URL
        # for company verification, NOT just the first result.
        # This handles common names where multiple people have the same name.
        target_result = search_results[0]  # Default to first
        
        if url_match_exact and linkedin_url:
            # Find the result with the matching URL
            profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower() if "/in/" in linkedin_url else None
            if profile_slug:
                for result in search_results:
                    result_url = result.get("link", "").lower()
                    if f"/in/{profile_slug}" in result_url:
                        target_result = result
                        print(f"   🎯 Using URL-matched result for company check: {result_url[:50]}")
                        break
        
        # Check the TARGET result (URL-matched or first)
        first_title = target_result.get("title", "").lower()
        
        # Normalize apostrophes and hyphens in title too (ScrapingDog returns "mcdonald ' s" and "chick - fil - a")
        first_title = first_title.replace("'", "'").replace("'", "'").replace("`", "'")
        first_title = re.sub(r"\s*'\s*", "'", first_title)  # "mcdonald ' s" → "mcdonald's"
        first_title = re.sub(r"\s*-\s*", "-", first_title)  # "chick - fil - a" → "chick-fil-a"
        
        # Extract ONLY the headline part (before "| linkedin")
        # ScrapingDog often concatenates descriptions after "| LinkedIn"
        # e.g., "Name - Title @ Company | LinkedIn About Company: ..."
        if "| linkedin" in first_title:
            first_title = first_title.split("| linkedin")[0].strip()
        
        # Also get snippet for additional company matching
        first_snippet = target_result.get("snippet", "").lower()
        first_snippet = re.sub(r"\s*-\s*", "-", first_snippet)  # Normalize hyphens
        
        # Check if title is truncated (contains "...")
        # If truncated, we may be missing the company name
        title_truncated = "..." in first_title
        
        # Method 1: Exact normalized company name in title
        company_in_title = company_normalized in first_title
        
        # Method 2: All significant words of company name in title (for multi-word companies)
        # e.g., "Bank Of America" → check if "bank", "of", "america" are all in title
        if not company_in_title and len(company_words) > 1:
            significant_words = [w for w in company_words if len(w) > 2]  # Skip "of", "the", etc.
            company_in_title = all(word in first_title for word in significant_words)
        
        # Method 2b: FUZZY MATCH - Handle shortened company names
        # e.g., "Sirona" in title matches "Sirona Hygiene" claimed
        # Extract company name from title (typically after "at " or "@ ")
        if not company_in_title:
            # Try to extract company from title patterns like "Name - Title at Company" or "Name @ Company"
            title_company_match = re.search(r'(?:at|@)\s+([^|\-]+?)(?:\s*[\|\-]|$)', first_title, re.IGNORECASE)
            if title_company_match:
                extracted_company = title_company_match.group(1).strip().lower()
                # Remove common trailing words that aren't part of company name
                extracted_company = re.sub(r'\s+(linkedin|profile|page).*$', '', extracted_company)
                # Remove parenthetical content and other junk after company name
                extracted_company = re.sub(r'\s*\([^)]*\).*$', '', extracted_company)  # Remove "(anything)" and everything after
                extracted_company = re.sub(r'\s*\|.*$', '', extracted_company)  # Remove "|" and everything after
                extracted_company = extracted_company.strip()
                
                # Normalize the extracted company name (remove legal suffixes)
                extracted_company_normalized = extracted_company
                for suffix in LEGAL_SUFFIXES:
                    if extracted_company_normalized.endswith(suffix):
                        extracted_company_normalized = extracted_company_normalized[:-len(suffix)].strip()
                        break
                
                # Check if either is a substring of the other (fuzzy match)
                # e.g., "sirona" ⊆ "sirona hygiene" OR "sirona hygiene" ⊇ "sirona"
                if len(extracted_company_normalized) >= 4 and len(company_normalized) >= 4:  # Both must be substantial
                    # Bidirectional containment check
                    if (extracted_company_normalized in company_normalized or 
                        company_normalized in extracted_company_normalized):
                        # Additional safety: must share at least 80% of longer word if both are single words
                        # This prevents "Meta" matching "Metamorph" 
                        if ' ' in extracted_company_normalized or ' ' in company_normalized:
                            # Multi-word: accept substring match
                            company_in_title = True
                            print(f"   ✅ FUZZY MATCH: '{extracted_company_normalized}' ≈ '{company_normalized}' (substring match)")
                        else:
                            # Single-word: check similarity
                            longer = max(extracted_company_normalized, company_normalized, key=len)
                            shorter = min(extracted_company_normalized, company_normalized, key=len)
                            if len(shorter) / len(longer) >= 0.6:  # At least 60% length ratio
                                company_in_title = True
                                print(f"   ✅ FUZZY MATCH: '{extracted_company_normalized}' ≈ '{company_normalized}' (similar length)")
        
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
        
        # Method 4: If first result is truncated and no company found, check OTHER results
        # ScrapingDog often truncates titles, so we need to look at more results
        if not company_in_title and not company_in_snippet and title_truncated and len(search_results) > 1:
            print(f"   ℹ️  First title truncated, checking other results for company...")
            for idx, result in enumerate(search_results[1:4], start=2):  # Check next 3 results
                other_title = result.get("title", "").lower()
                other_snippet = result.get("snippet", "").lower()
                
                # Normalize the other title/snippet
                other_title = other_title.replace("'", "'").replace("'", "'").replace("`", "'")
                other_title = re.sub(r"\s*'\s*", "'", other_title)
                other_title = re.sub(r"\s*-\s*", "-", other_title)
                
                # Check if company in this result
                if company_normalized in other_title or company_normalized in other_snippet:
                    company_in_title = True
                    print(f"   ✅ Company found in result #{idx}: {other_title[:60]}...")
                    break
                
                # Check multi-word company
                if len(company_words) > 1:
                    significant_words = [w for w in company_words if len(w) > 2]
                    if all(word in other_title or word in other_snippet for word in significant_words):
                        company_in_title = True
                        print(f"   ✅ Company words found in result #{idx}: {other_title[:60]}...")
                        break
        
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
        results_text = json.dumps(search_results, indent=2, default=str)  # Handle any datetime objects
        
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
   - Names must substantially match, but allow common variants:
     * Spelling variants (e.g., "Jacobson" = "Jacobsen", "Smith" = "Smyth", "Steven" = "Stephen")
     * Shortened names (e.g., "Ben" = "Benjamin", "Mike" = "Michael", "Chris" = "Christopher")
     * Middle names/initials present or absent (e.g., "John Smith" = "John A. Smith")
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
            "temperature": 0  # Zero temperature for deterministic results
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
                # - URL can only match if ScrapingDog found that URL when searching for the CLAIMED NAME
                # - ScrapingDog searches "Pranav Ramesh" → only returns results for that name
                # - If miner gave a different person's URL, it wouldn't appear in those results
                # - So exact URL match = the profile IS for the claimed person
                # 
                # WHY THIS IS NEEDED:
                # - ScrapingDog often returns concatenated results with OTHER people's headlines
                # - LLM compares those wrong headlines against claimed name → false negative
                # - Example: Search "Melissa Carberry" → ScrapingDog returns correct URL but 
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
                
                # OVERRIDE 3: If URL matched (exact OR partial) + name matches, but company not in results
                # (due to ScrapingDog returning truncated titles/posts), PASS Stage 4 and let Stage 5 verify company.
                #
                # WHY THIS IS SAFE:
                # - URL match (even partial) + name match = we verified the profile exists and belongs to right person
                # - Profile slug matching means ScrapingDog found that URL when searching for the person's name
                # - If miner gave wrong URL, it wouldn't appear in name-based search results
                # - Stage 4's job = verify IDENTITY (person exists on LinkedIn)
                # - Stage 5's job = verify EMPLOYMENT (person works at claimed company/role/region)
                # - If company not found in title, likely due to ScrapingDog truncation ("Chief Technology ...")
                # - Stage 5 will independently verify company by searching "Name + Company + Role"
                # - If miner lied about company, Stage 5 will catch it
                #
                # REQUIRES:
                # - url_match_exact=True OR we found the URL in results (partial match)
                # - name_match=True (identity confirmed)
                # - company_match=False (company not in results, but will be verified in Stage 5)
                # Extract profile slug from linkedin_url to check for partial matches
                profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0] if linkedin_url and "/in/" in linkedin_url else None
                profile_slug_norm = profile_slug.lower().replace("-", "").replace("_", "") if profile_slug else ""
                has_url_match = url_match_exact or any(
                    profile_slug_norm in result.get("link", "").lower().replace("-", "").replace("_", "")
                    for result in search_results[:5] if "linkedin.com/in/" in result.get("link", "")
                )
                
                if has_url_match and name_match and not company_match:
                    print(f"   ⚠️ OVERRIDE: URL + Name match, but company not in GSE results")
                    print(f"      → Passing Stage 4 (identity verified)")
                    print(f"      → Stage 5 will verify company/role/region")
                    company_match = True  # Override to pass Stage 4
                    profile_valid = True
                    confidence = max(confidence, 0.7)
                    reasoning = f"[URL + name verified, company check deferred to Stage 5] {reasoning}"
                
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
        
        # Step 1: Search LinkedIn via ScrapingDog GSE
        print(f"   🔍 Stage 4: Verifying LinkedIn profile for {full_name} at {company}")
        
        # ScrapingDog GSE search for LinkedIn profile (returns url_match_exact status)
        search_results, url_match_exact = await search_linkedin_gse(full_name, company, linkedin_url)
        
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
        
        # Extract role from Stage 4's confirmed profile title for use in Stage 5
        # This provides an authoritative role source directly from the LinkedIn profile
        stage4_extracted_role = None
        if search_results and len(search_results) > 0:
            # Try extracting role from ALL search results (not just first one)
            # This handles cases where the first result's title is malformed/concatenated
            # but subsequent results have cleaner titles with the role
            for result in search_results[:3]:  # Try first 3 results
                result_title = result.get("title", "")
                result_snippet = result.get("snippet", "")
                
                # Try title first (most reliable)
                role_from_title = extract_role_from_search_title(
                    result_title, "", company_name=company, full_name=full_name
                )
                
                if role_from_title:
                    # Sanity check: Role should be reasonable length (not garbage from concatenation)
                    if len(role_from_title) < 100 and " ... " not in role_from_title:
                        stage4_extracted_role = role_from_title
                        break  # Found a clean role, stop searching
            
            # If title extraction failed, try snippet as fallback
            # Snippets often contain LinkedIn's meta description with role info
            if not stage4_extracted_role:
                for result in search_results[:3]:
                    result_snippet = result.get("snippet", "")
                    if result_snippet:
                        role_from_snippet = extract_role_from_search_title(
                            "", result_snippet, company_name=company, full_name=full_name
                        )
                        if role_from_snippet and len(role_from_snippet) < 100:
                            stage4_extracted_role = role_from_snippet
                            break
            
            if stage4_extracted_role:
                lead["stage4_extracted_role"] = stage4_extracted_role
                print(f"   📝 Stage 4: Extracted role from profile: '{stage4_extracted_role}'")
        
        # ========================================================================
        # EXTRACT PERSON LOCATION FROM LINKEDIN SEARCH RESULTS (NEW)
        # ========================================================================
        # The person's profile header location often appears in Google snippets.
        # This is the PERSON's location, not the company headquarters.
        # Format in snippets: "...School of Business. New York, New York, United States."
        # 
        # IMPORTANT: Only extract location from results that match the miner's
        # provided LinkedIn URL. This prevents extracting location from a different
        # person with the same name.
        # ========================================================================
        stage4_extracted_location = None
        if search_results and len(search_results) > 0:
            # Extract profile slug from miner's provided LinkedIn URL
            profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower() if linkedin_url and "/in/" in linkedin_url else None
            
            # Try extracting location from search result snippets
            # ONLY from results that match the miner's LinkedIn profile URL
            for result in search_results[:5]:  # Check first 5 results
                result_url = result.get("link", result.get("url", "")).lower()
                result_snippet = result.get("snippet", result.get("body", ""))
                
                # ENFORCE: Only extract from results that match the profile slug
                if profile_slug and "linkedin.com/in/" in result_url:
                    # Extract slug from result URL
                    result_slug = result_url.split("/in/")[-1].strip("/").split("?")[0]
                    
                    # Normalize for comparison (handle hyphens, underscores)
                    profile_slug_norm = profile_slug.replace("-", "").replace("_", "")
                    result_slug_norm = result_slug.replace("-", "").replace("_", "")
                    
                    if profile_slug_norm != result_slug_norm:
                        # URL doesn't match miner's profile - skip this result
                        continue
                
                if result_snippet:
                    location = extract_person_location_from_linkedin_snippet(result_snippet)
                    if location:
                        stage4_extracted_location = location
                        print(f"   📍 Stage 4: Extracted person location from VERIFIED profile URL")
                        break
            
            if stage4_extracted_location:
                lead["stage4_extracted_location"] = stage4_extracted_location
                print(f"   📍 Stage 4: Extracted person location from profile: '{stage4_extracted_location}'")
        
        # ========================================================================
        # STAGE 4: COMPANY LINKEDIN VALIDATION (NEW)
        # ========================================================================
        # Validates company_linkedin URL, verifies company name matches, and caches
        # company data (industry, description, employee_count) for Stage 5.
        # FAIL HERE = Stage 5 never runs = saves all Stage 5 API costs
        # ========================================================================
        
        company_linkedin = lead.get("company_linkedin", "") or ""
        
        if company_linkedin:
            print(f"   🏢 Stage 4: Validating company LinkedIn URL...")
            
            # Step 1: Validate URL format (must be /company/, not /in/)
            url_valid, url_reason, company_slug = validate_company_linkedin_url(company_linkedin)
            
            if not url_valid:
                print(f"   ❌ Stage 4: Company LinkedIn URL INVALID: {url_reason}")
                return False, {
                    "stage": "Stage 4: Company LinkedIn Validation",
                    "check_name": "check_linkedin_gse",
                    "message": f"Company LinkedIn URL is invalid: {url_reason}",
                    "failed_fields": ["company_linkedin"],
                    "provided_url": company_linkedin,
                    "expected_format": "https://linkedin.com/company/{company-name}"
                }
            
            print(f"   ✅ Stage 4: Company LinkedIn URL format valid: /company/{company_slug}")
            
            # Step 2: Check global cache first
            cached_data = get_company_linkedin_from_cache(company_slug)
            
            if cached_data:
                print(f"   📦 Stage 4: Using CACHED company LinkedIn data for '{company_slug}'")
                
                # Verify company name still matches (cache might have different company)
                cached_company_name = cached_data.get("company_name_from_linkedin", "")
                if cached_company_name:
                    # Check if cached company matches current company claim
                    cached_lower = cached_company_name.lower().strip()
                    claimed_lower = company.lower().strip()
                    
                    if cached_lower != claimed_lower and cached_lower not in claimed_lower and claimed_lower not in cached_lower:
                        print(f"   ❌ Stage 4: Cached company name '{cached_company_name}' doesn't match claimed '{company}'")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn page shows '{cached_company_name}' but miner claimed '{company}'",
                            "failed_fields": ["company_linkedin", "company"],
                            "linkedin_company": cached_company_name,
                            "claimed_company": company
                        }
                
                # Store cached data on lead for Stage 5
                lead["company_linkedin_verified"] = True
                lead["company_linkedin_slug"] = company_slug
                lead["company_linkedin_data"] = cached_data
                lead["company_linkedin_from_cache"] = True
                
                # Log what data we have cached
                if cached_data.get("employee_count"):
                    print(f"   📊 Cached employee count: {cached_data['employee_count']}")
                if cached_data.get("industry"):
                    print(f"   🏭 Cached industry: {cached_data['industry']}")
                if cached_data.get("description"):
                    print(f"   📝 Cached description: {cached_data['description'][:80]}...")
            else:
                # Step 3: Not cached - scrape company LinkedIn page via GSE
                print(f"   🔍 Stage 4: Scraping company LinkedIn page for '{company_slug}'...")
                scraped_data = await scrape_company_linkedin_gse(company_slug, company)
                
                if scraped_data.get("success"):
                    # Step 4: Verify company name matches
                    if not scraped_data.get("company_name_match"):
                        linkedin_company = scraped_data.get("company_name_from_linkedin", "Unknown")
                        print(f"   ❌ Stage 4: Company name MISMATCH: LinkedIn shows '{linkedin_company}' but miner claimed '{company}'")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn page shows '{linkedin_company}' but miner claimed '{company}'",
                            "failed_fields": ["company_linkedin", "company"],
                            "linkedin_company": linkedin_company,
                            "claimed_company": company
                        }
                    
                    print(f"   ✅ Stage 4: Company name verified: '{company}'")
                    
                    # Step 5: Cache the data globally (only on success)
                    set_company_linkedin_cache(company_slug, scraped_data)
                    print(f"   💾 Stage 4: Cached company LinkedIn data for future leads")
                    
                    # Store on lead for Stage 5
                    lead["company_linkedin_verified"] = True
                    lead["company_linkedin_slug"] = company_slug
                    lead["company_linkedin_data"] = scraped_data
                    lead["company_linkedin_from_cache"] = False
                    
                    # Log what data we scraped
                    if scraped_data.get("employee_count"):
                        print(f"   📊 Scraped employee count: {scraped_data['employee_count']}")
                    if scraped_data.get("industry"):
                        print(f"   🏭 Scraped industry: {scraped_data['industry']}")
                    if scraped_data.get("description"):
                        print(f"   📝 Scraped description: {scraped_data['description'][:80]}...")
                else:
                    # Scraping failed - check if it's a URL mismatch (reject) or just scraping error (warn)
                    error_msg = scraped_data.get('error', 'Unknown')
                    
                    # If URL doesn't match, this is a CRITICAL error - reject immediately
                    if "does not match expected slug" in error_msg or "URL mismatch" in error_msg:
                        print(f"   ❌ Stage 4: Company LinkedIn URL mismatch: {error_msg}")
                        return False, {
                            "stage": "Stage 4: Company LinkedIn Validation",
                            "check_name": "check_linkedin_gse",
                            "message": f"Company LinkedIn URL is incorrect or ambiguous: {error_msg}",
                            "failed_fields": ["company_linkedin"],
                            "provided_url": company_linkedin,
                            "hint": "The URL provided returns a different company page. Ensure you're using the exact LinkedIn company slug."
                        }
                    
                    # Other scraping errors (network, API, etc.) - warn but don't fail
                    # Stage 5 will use fallback GSE searches
                    print(f"   ⚠️ Stage 4: Could not scrape company LinkedIn: {error_msg}")
                    print(f"   ⚠️ Stage 4: Stage 5 will use fallback GSE searches for industry/employee data")
                    lead["company_linkedin_verified"] = True  # URL was valid format
                    lead["company_linkedin_slug"] = company_slug
                    lead["company_linkedin_data"] = None  # No data - Stage 5 will fallback
                    lead["company_linkedin_from_cache"] = False
        else:
            # No company_linkedin provided - Stage 5 will use fallback GSE searches
            print(f"   ⚠️ Stage 4: No company_linkedin URL provided")
            lead["company_linkedin_verified"] = False
            lead["company_linkedin_data"] = None
        
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
                    async with session.get(url, params=params, timeout=15, proxy=HTTP_PROXY_URL) as response:
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
        
        # Remove duplicates while preserving order (e.g., if abbreviated = original)
        company_variations = list(dict.fromkeys(company_variations))
        
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
                
                async with session.get(search_url, headers=headers, params=params, timeout=7, proxy=HTTP_PROXY_URL) as response:
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
                            
                            async with session.get(search_url, headers=headers, params=cik_params, timeout=7, proxy=HTTP_PROXY_URL) as cik_response:
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
            
            async with session.get(gdelt_url, params=params, timeout=15, proxy=HTTP_PROXY_URL) as response:
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
                timeout=10,
                proxy=HTTP_PROXY_URL
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
# Uses ScrapingDog search results + fuzzy matching + LLM verification
# 
# Flow:
# 1. ScrapingDog search for ROLE (name + company + linkedin)
# 2. ScrapingDog search for REGION (company headquarters)
# 3. ScrapingDog search for INDUSTRY (what company does)
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
    
    # Count distinct US states mentioned (use word boundaries to avoid "kansas" matching "arkansas")
    states_found = set()
    for state in US_STATES_SET:
        # e.g., "Arkansas" should not match "kansas"
        pattern = r'\b' + re.escape(state) + r'\b'
        if re.search(pattern, claimed_lower):
            states_found.add(state)
    for abbrev in US_STATE_ABBREVS:
        if re.search(rf'\b{abbrev}\b', claimed_lower):
            states_found.add(abbrev)
    
    # Special case: "west virginia" should not also count "virginia"
    if 'west virginia' in states_found and 'virginia' in states_found:
        states_found.discard('virginia')
    
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
    # Additional C-suite roles (conservative additions)
    "ciso": "chief information security officer",
    "clo": "chief legal officer",
    "cco": "chief compliance officer",
    "cgo": "chief growth officer",
    "ctpo": "chief technology product officer",
    "csco": "chief supply chain officer",
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
    # Additional common abbreviations (conservative additions)
    "acct": "accountant",
    "admin": "administrator",
    "asst": "assistant",
    "coord": "coordinator",
    "rep": "representative",
    "supv": "supervisor",
    "tech": "technician",
}

ROLE_EQUIVALENCIES = {
    "founder": ["founder", "co-founder", "co founder", "cofounder", "founding", "founding member", "founding partner"],
    "owner": ["owner", "business owner", "franchise owner", "store owner", "agent owner", "owner operator"],
    "president": ["president", "pres", "pres."],
    "partner": ["partner", "managing partner", "general partner", "senior partner", "equity partner", "founding partner"],
    "board": ["board member", "board director", "director", "board of directors"],
    "chair": ["chairman", "chairwoman", "chair", "chairperson", "executive chair", "executive chairman"],
    "attorney": ["attorney", "counsel", "lawyer", "legal counsel", "associate attorney", "staff attorney"],
    "recruiting": ["recruiting", "recruitment", "recruitments", "recruiter", "talent acquisition", "staffing"],
    # Sales and Business Development are often used interchangeably
    "sales": ["sales", "business development", "bd", "biz dev", "revenue", "commercial"],
    "business development": ["business development", "sales", "bd", "biz dev", "revenue", "commercial"],
    # HR/People - all refer to the same function
    "hr": ["hr", "human resources", "people", "talent", "people operations", "people ops"],
    "human resources": ["human resources", "hr", "people", "talent", "people operations", "people ops"],
    "people": ["people", "hr", "human resources", "talent", "people operations"],
    # Operations abbreviation
    "ops": ["ops", "operations"],
    "operations": ["operations", "ops"],
    # Customer-facing roles - often overlap
    "customer success": ["customer success", "customer service", "client success", "account management", "client services"],
    "customer service": ["customer service", "customer success", "support", "client services", "client support"],
    "support": ["support", "customer service", "customer support", "technical support", "client support"],
}


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison: lowercase, remove extra spaces, normalize hyphens/numbers."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Normalize spaces around numbers (J 2 Health -> j2health)
    text = re.sub(r'\s+(\d+)\s*', r'\1', text)
    text = re.sub(r'(\d+)\s+', r'\1', text)
    # Normalize hyphens with spaces
    text = re.sub(r'\s*-\s*', '-', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_role_from_search_title(title: str, snippet: str = "", company_name: str = "", full_name: str = "") -> Optional[str]:
    """
    Extract job role from ScrapingDog LinkedIn search result title/snippet.
    
    CRITICAL: Only extract ROLES, not company names. LinkedIn titles have two formats:
    1. "Name - Role @ Company | LinkedIn" (has role)
    2. "Name - Company | LinkedIn" (NO role, just company)
    
    We must distinguish between these and return None for format 2.
    
    If full_name is provided, verify the extracted role is associated with that person
    (role should appear near the person's name in the text).
    
    PRIORITY: If title is truncated ("..."), prioritize snippet extraction for complete data.
    """
    # Need at least title OR snippet to extract from
    if not title and not snippet:
        return None
    
    original_title = title if title else ""
    
    # Normalize company name for comparison
    company_normalized = normalize_for_comparison(company_name) if company_name else ""
    
    # For name proximity check - extract name parts for matching
    name_parts = []
    if full_name:
        name_parts = [p.lower() for p in full_name.split() if len(p) > 2]
    
    role_keywords = [
        # C-suite and executive
        "ceo", "cto", "cfo", "coo", "cmo", "cio", "cpo",
        "founder", "co-founder", "cofounder", "co founder",
        "president", "vice president", "vp",
        "executive", "officer", "chief",
        
        # Management and leadership
        "director", "manager", "lead", "head",
        "owner", "partner", "principal",
        "supervisor", "coordinator",
        
        # Technical roles
        "engineer", "developer", "analyst", "architect", "designer",
        "technician", "programmer", "administrator", "sysadmin",
        
        # Professional services
        "consultant", "advisor", "specialist",
        "attorney", "lawyer", "counsel", "paralegal",  # Legal
        "accountant", "auditor", "controller", "treasurer", "bookkeeper", "comptroller",  # Finance/Accounting
        
        # Healthcare (specific, low false positive risk)
        "physician", "surgeon", "nurse", "pharmacist", "dentist", "therapist",
        
        # Academic
        "professor", "teacher", "instructor", "lecturer",
        "scientist", "researcher",
        
        # Sales and customer-facing
        "representative", "agent", "broker", "account executive",
        
        # General professional
        "product owner", "staff", "senior", "sr.", "jr.",
        "associate", "assistant",
        "creative", "marketing", "sales", "hr", "human resources",
        "operations", "business operations",
        
        # Administrative
        "receptionist", "secretary", "clerk", "registrar",
        
        # Finance/Investment
        "investor", "trader", "banker",
    ]
    
    def has_role_keyword(text: str) -> bool:
        """Check if text contains a role keyword."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in role_keywords)
    
    def is_company_name(text: str) -> bool:
        """Check if text is likely a company name, not a role."""
        if not company_name:
            return False
        text_norm = normalize_for_comparison(text)
        # Check if extraction matches company name (normalized)
        if text_norm == company_normalized:
            return True
        # Check if company is contained in extraction
        if company_normalized and company_normalized in text_norm:
            return True
        # Check if extraction is contained in company (e.g., "J2" for "J2 Health")
        if text_norm and len(text_norm) > 2 and text_norm in company_normalized:
            return True
        return False
    
    # Check for job posting format FIRST: "Company hiring Role [in Location] | LinkedIn"
    job_posting_match = re.search(r'hiring\s+(.+?)(?:\s+in\s+[\w\s,]+)?(?:\s*\||\s*$)', title, re.IGNORECASE)
    if job_posting_match:
        role = job_posting_match.group(1).strip()
        role = re.sub(r'\s+in\s+[\w\s,]+$', '', role, flags=re.IGNORECASE).strip()
        if len(role) > 2 and len(role) < 100 and not is_company_name(role) and _is_valid_role_extraction(role):
            return role
    
    first_segment = title.split('|')[0].strip()
    first_segment = re.sub(r'\s+-\s*LinkedIn.*$', '', first_segment, flags=re.IGNORECASE).strip()
    first_segment = re.sub(r'\s*\.\.\.\s*$', '', first_segment).strip()
    
    # Common role abbreviations that are valid even if ≤2 chars
    common_abbreviations = ['vp', 'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cpo', 'svp', 'evp']
    
    # PATTERN 1: "Name - Role @ Company" or "Name - Role at Company"
    # This is the MOST reliable pattern - role has separator before company
    role_at_match = re.search(r'^[^-]+-\s*(.+?)\s+(?:@|at)\s+', first_segment, re.IGNORECASE)
    if role_at_match:
        role = role_at_match.group(1).strip()
        # Allow common abbreviations even if ≤2 chars (VP, CEO, CTO, etc.)
        is_common_abbrev = role.lower() in common_abbreviations
        
        if (len(role) > 2 or is_common_abbrev) and not is_company_name(role) and _is_valid_role_extraction(role):
            return role
    
    # PATTERN 2: Look for "Role @ Company" or "Role at Company" anywhere
    role_at_patterns = re.findall(r'(\b(?:' + '|'.join(role_keywords) + r')[^|@]*?)\s+(?:@|at)\s+\w', original_title, re.IGNORECASE)
    if role_at_patterns:
        role = role_at_patterns[0].strip()
        # Allow common abbreviations even if ≤2 chars
        is_common_abbrev = role.lower() in common_abbreviations
        
        if (len(role) > 2 or is_common_abbrev) and not is_company_name(role) and _is_valid_role_extraction(role):
            return role
    
    # PATTERN 3: "Name - Something" where Something contains role keywords
    # ONLY use this if we find role keywords - otherwise it's probably the company name
    match = re.search(r'^([^-]+)-\s*(.+?)$', first_segment, re.IGNORECASE)
    if match:
        potential_role = match.group(2).strip()
        # Clean up trailing "at Company" or "@Company"
        potential_role = re.sub(r'\s+(?:@|at)\s+.*$', '', potential_role, flags=re.IGNORECASE).strip()
        
        # NEW: If company name appears at the end (e.g., "Managing Director Wajer Yachts"),
        # try to strip it off to isolate the role
        if company_name and company_normalized:
            # Check if company name is at the end of potential_role
            potential_role_norm = normalize_for_comparison(potential_role)
            if potential_role_norm.endswith(company_normalized):
                # Strip company from end: "Managing Director Wajer Yachts" → "Managing Director"
                potential_role = potential_role[:-(len(company_name))].strip()
            elif company_normalized in potential_role_norm:
                # Company somewhere in the middle/end, try to remove it
                # Split on company name and take the part before it
                parts = re.split(re.escape(company_name), potential_role, flags=re.IGNORECASE)
                if parts and parts[0].strip():
                    potential_role = parts[0].strip()
        
        # ONLY return if it has a role keyword AND is not a company name
        if len(potential_role) > 2 and has_role_keyword(potential_role) and not is_company_name(potential_role) and _is_valid_role_extraction(potential_role):
            return potential_role
    
    # PATTERN 4: Look for compound titles in title (e.g., "President & Co-Founder @ Company")
    compound_patterns = [
        r'\b((?:co-?)?founder\s+(?:and|&)\s+(?:ceo|cto|cfo|coo|president|cmo))\s*(?:@|at)',
        r'\b((?:ceo|cto|cfo|coo|president|cmo)\s+(?:and|&)\s+(?:co-?)?founder)\s*(?:@|at)',
        r'\b(president\s+(?:and|&)\s+(?:co-?)?founder)\s*(?:@|at)',
        r'\b((?:co-?)?founder\s+(?:and|&)\s+president)\s*(?:@|at)',
    ]
    for pattern in compound_patterns:
        match = re.search(pattern, original_title, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if not is_company_name(role) and _is_valid_role_extraction(role):
                return role
    
    # PATTERN 5: Look for role keywords followed by "at" anywhere
    for kw in role_keywords:
        match = re.search(rf'\b({kw}[^|,@]*?)\s+(?:@|at)\s+', original_title, re.IGNORECASE)
        if match:
            role = match.group(1).strip()
            if len(role) > 2 and not is_company_name(role) and _is_valid_role_extraction(role):
                return role
    
    # SNIPPET PATTERNS - look in snippet for role info
    if snippet:
        snippet_clean = snippet.strip()
        
        # Skip garbage snippets entirely
        garbage_indicators = [
            "session details", "read more", "click here", "learn more",
            "view profile", "see the complete", "view full", "show more",
            "scientific index", "company profile", "funding", "competitors"
        ]
        if any(g in snippet_clean.lower() for g in garbage_indicators):
            return None
        
        # NAME PROXIMITY CHECK: If name provided, verify snippet mentions this person
        # near any extracted role (to avoid extracting role of wrong person)
        # 
        # EXCEPTION: Skip this check for LinkedIn PROFILE snippets where:
        # - The person's name IS in the title (confirms it's their profile)
        # - But NOT in snippet (because profile speaks in first person)
        # This is common for LinkedIn profiles: "Responsible AI Lead... As a technical advisor..."
        is_linkedin_profile_snippet = False
        if name_parts and len(name_parts) >= 2 and original_title:
            title_lower = original_title.lower()
            # Check if name is in title (LinkedIn format: "Name - Role")
            name_in_title = any(part in title_lower for part in name_parts if len(part) > 2)
            if name_in_title:
                is_linkedin_profile_snippet = True
        
        # Only do name proximity check if NOT a LinkedIn profile snippet
        if name_parts and len(name_parts) >= 2 and not is_linkedin_profile_snippet:
            snippet_lower = snippet_clean.lower()
            # Check for last name (usually more unique) or first name
            last_name = name_parts[-1] if len(name_parts[-1]) > 2 else None
            first_name = name_parts[0] if len(name_parts[0]) > 2 else None
            
            name_mentioned = False
            if last_name and last_name in snippet_lower:
                name_mentioned = True
            elif first_name and first_name in snippet_lower:
                name_mentioned = True
            # Also check for common title patterns like "Mr.", "Ms.", "Dr."
            elif any(title in snippet_lower for title in ['mr.', 'ms.', 'dr.', 'prof.']):
                # If there's a title, be more lenient
                name_mentioned = True
            
            if not name_mentioned:
                # This snippet doesn't mention our person at all, skip it
                return None
        
        # HIGH-PRIORITY: Look for compound titles first (e.g., "Founder and CEO", "President & Co-Founder")
        compound_patterns = [
            # "Founder and CEO John Smith" or "CEO and Founder John Smith"
            r'\b((?:co-?)?founder\s+(?:and|&)\s+(?:ceo|cto|cfo|coo|president|cmo))\b',
            r'\b((?:ceo|cto|cfo|coo|president|cmo)\s+(?:and|&)\s+(?:co-?)?founder)\b',
            # "President and Co-Founder"
            r'\b(president\s+(?:and|&)\s+(?:co-?)?founder)\b',
            r'\b((?:co-?)?founder\s+(?:and|&)\s+president)\b',
            # "Chief Executive Officer and Founder"
            r'\b(chief\s+\w+\s+officer\s+(?:and|&)\s+(?:co-?)?founder)\b',
            r'\b((?:co-?)?founder\s+(?:and|&)\s+chief\s+\w+\s+officer)\b',
        ]
        for pattern in compound_patterns:
            match = re.search(pattern, snippet_clean, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if not is_company_name(role):
                    return role
        
        snippet_patterns = [
            # LinkedIn snippet pattern: "is currently a Director of Partnerships and Growth at akoyaGO"
            # This must come FIRST to avoid the shorter "is [Role] at" pattern below matching too early
            r'\b(?:is|was)\s+currently\s+(?:a|an|the)\s+([A-Za-z\s&,\-]+?)\s+at\s+',
            # "rose to the role of Chief Operating Officer" - capture full C-suite title
            r'(?:role\s+of|as\s+(?:a|the)?)\s*(chief\s+\w+\s+officer)\b',
            r'(?:was\s+(?:a|the)\s+)(chief\s+\w+\s+officer)\b',
            r'(?:served\s+as\s+)(chief\s+\w+\s+officer)\b',
            # "currently a Chief of Staff" or "currently a Chief Technology Officer"
            r'\b(?:currently\s+(?:a|the|an)\s+)(chief\s+of\s+staff[^|,.]{0,20})\b',
            r'\b(?:currently\s+(?:a|the|an)\s+)(chief\s+\w+\s+officer)\b',
            r'\b(?:currently\s+(?:a|the|an)\s+)((?:senior\s+)?vice\s+president[^|,.]{0,30})',
            # SPECIAL: "serves as the Director of [Department/Area] at Company"
            # This must come BEFORE the generic "serves as X at" pattern to capture full "Director of X" roles
            r'(?:serves?\s+as\s+(?:the\s+)?)((?:director|head|vp|vice\s+president|manager|leader|chief)\s+of\s+[A-Za-z\s&,\-]+?)\s+at\s+',
            # "John is the CEO at Company" or "John serves as Director at..." (without "of")
            # NOTE: Only match " at " here, not "of" or "for" which can appear WITHIN roles (e.g., "Director of Sales")
            r'(?:is\s+(?:the\s+)?|serves?\s+as\s+(?:the\s+)?|works?\s+as\s+(?:the\s+)?)([^.]+?)\s+at\s+',
            # "from Founder & CEO, Name" or "by CEO Name" (common in press releases)
            r'(?:from|by)\s+([A-Za-z\s&]+?(?:' + '|'.join(role_keywords[:15]) + r')[A-Za-z\s&]*?),?\s+[A-Z][a-z]+',
            # "John, CEO at Company" - common in non-LinkedIn sources  
            r',\s*([A-Za-z\s&]+?(?:' + '|'.join(role_keywords[:20]) + r')[A-Za-z\s&]*?)\s+at\s+',
            # "currently serving as Director"
            r'(?:works? as|serving as|position)[:\s]+([^|.\n]+)',
            # Academic: "Associate Professor of X at University"
            r'\b((?:assistant|associate|full|adjunct|visiting)?\s*professor\s+of\s+[^|,.]+)',
            # "Name is an Associate Professor"
            r'\b(?:is\s+an?\s+)((?:assistant|associate|full)?\s*professor[^|,.]{0,30})',
        ]
        for pattern in snippet_patterns:
            match = re.search(pattern, snippet_clean, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if len(role) > 2 and len(role) < 80:
                    # Reject if starts with common garbage words
                    if role.lower().startswith(('now ', 'the ', 'a ', 'an ', 'this ', 'that ', 'is ', 'was ', 'in ', 'of ', 'at ', 'i ')):
                        continue
                    if is_company_name(role):
                        continue
                    # Final validation
                    if _is_valid_role_extraction(role):
                        return role
        
        # PATTERN: "Company [C-Suite Abbreviation]" (e.g., "Eucalyptus CSO", "Microsoft CFO")
        # Common in LinkedIn directory listings where format is "Name. Company ROLE."
        if company_name:
            # Check for C-suite abbreviations right after company name
            c_suite_abbrevs = ['ceo', 'cfo', 'cto', 'coo', 'cmo', 'cio', 'cpo', 'cso', 'ciso', 'clo', 'cco', 'cgo', 'ctpo', 'csco']
            company_escaped = re.escape(company_name)
            for abbrev in c_suite_abbrevs:
                # Match "Company ABBREV" where ABBREV is at word boundary
                pattern = rf'{company_escaped}\s+({abbrev})\b'
                match = re.search(pattern, snippet_clean, re.IGNORECASE)
                if match:
                    role_abbrev = match.group(1).strip()
                    # Expand abbreviation to full title for validation
                    if role_abbrev.lower() in C_SUITE_EXPANSIONS:
                        return C_SUITE_EXPANSIONS[role_abbrev.lower()].title()
                    elif role_abbrev.upper() in ['CSO', 'CISO', 'CLO', 'CCO', 'CGO', 'CTPO', 'CSCO']:
                        # Return as-is for less common abbreviations
                        return role_abbrev.upper()
        
        # PRIORITY: LinkedIn directory format: "Name. Role @ Company"
        # E.g., "Allison Constable. VP of Sales, Ad Measurement @ DISQO"
        # This pattern allows commas in the role title
        linkedin_dir_pattern = r'\.\s+([A-Za-z\s,]+?)\s+@\s+'
        match = re.search(linkedin_dir_pattern, snippet_clean)
        if match:
            potential_role = match.group(1).strip()
            # Remove any trailing commas
            potential_role = potential_role.rstrip(',').strip()
            if len(potential_role) > 2 and has_role_keyword(potential_role) and not is_company_name(potential_role):
                if _is_valid_role_extraction(potential_role):
                    return potential_role
        
        found_roles = []
        for kw in role_keywords:
            # Match FULL ROLE: capture everything from word boundary to keyword, stopping at "at/@"
            # E.g., "VP of Sales @ DISQO" or "Chief Technology Advisor at Microsoft"
            match = re.search(rf'((?:[A-Za-z,]+\s+){{0,6}}{kw})\s+(?:at|@)\s+', snippet_clean, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                if len(role) > 2 and len(role) < 100 and not is_company_name(role):
                    found_roles.append(role)
        
        # Return the longest found role (compound titles are longer)
        if found_roles:
            best_role = max(found_roles, key=len)
            # Final validation
            if _is_valid_role_extraction(best_role):
                return best_role
    
    return None


def _is_valid_role_extraction(role: str) -> bool:
    """Final validation to filter garbage role extractions."""
    if not role:
        return False
    
    role_lower = role.lower().strip()
    
    # Filter: "assistant to X", "secretary to X" patterns (wrong person's role)
    if re.search(r'\b(assistant|secretary|aide|exec\s+assistant)\s+to\s+', role_lower):
        return False
    
    # Filter: Single-word garbage that snuck through
    garbage_single_words = [
        'enthusiasm', 'passion', 'expert', 'professional', 'leader',
        'elder', 'son', 'daughter', 'father', 'mother', 'brother', 'sister',
        'currently', 'former', 'previously', 'now', 'recent',
        'best', 'top', 'great', 'amazing', 'excellent',
        'view', 'see', 'click', 'read', 'more', 'about',
        'profile', 'page', 'website', 'site', 'link',
        'executive', 'officer', 'director', 'manager',  # Too generic without context
    ]
    if role_lower in garbage_single_words:
        return False
    
    # Filter: Family/personal terms
    if re.search(r'\b(elder|son|daughter|wife|husband|father|mother|brother|sister)\b', role_lower):
        return False
    
    # Filter: Too short and not a known role abbreviation
    known_short_roles = ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cpo', 'vp', 'svp', 'evp', 'md', 'gm']
    if len(role_lower) < 4 and role_lower not in known_short_roles:
        return False
    
    # Filter: Starts with articles/prepositions/conjunctions (incomplete extraction)
    if role_lower.startswith(('the ', 'a ', 'an ', 'to ', 'for ', 'at ', 'in ', 'of ', 'by ', 'as ', 'currently ', 'and ', 'or ', 'but ', 'with ', 'from ', 'on ')):
        return False
    
    # Filter: Contains obvious non-role patterns
    if re.search(r'(view|click|read|learn|see|show)\s+(more|profile|full)', role_lower):
        return False
    
    # Filter: Single generic role words that need more context (but keep founder, CEO, etc)
    too_generic_single = ['head', 'lead', 'officer', 'assistant', 'executive']
    if role_lower in too_generic_single:
        return False
    
    # Filter: Known invalid tool/site names
    invalid_tools = ['contactout', 'rocketreach', 'apollo', 'leadiq', 'lusha', 
                     'seamless', 'hunter', 'clearbit', 'datanyze', 'discoverorg',
                     'insideview', 'owler', 'zoominfo', 'crunchbase']
    if role_lower in invalid_tools or any(tool in role_lower for tool in invalid_tools):
        return False
    
    # Filter: Garbage descriptive phrases (not roles)
    garbage_phrases = [
        'practice on the', 'focuses on the', 'specializes in the',
        'expertise in the', 'experience in the', 'works in the',
        'involved in the', 'engaged in the', 'active in the',
        'known for the', 'recognized for', 'awarded for',
        'session details', 'read more about', 'learn more',
        'areas of', 'field of', 'domain of', 'realm of'
    ]
    if any(phrase in role_lower for phrase in garbage_phrases):
        return False
    
    # Filter: Web UI text patterns (browser/website interface text)
    web_ui_patterns = [
        'opens in', 'new window', 'new tab', 'click to', 'click here',
        'tap to', 'swipe to', 'scroll to', 'navigate to',
        'opens a new', 'link opens', 'external link',
        'download', 'print', 'share', 'save', 'bookmark',
        'sign in', 'log in', 'register', 'subscribe',
        'terms of', 'privacy policy', 'cookies', 'consent'
    ]
    if any(pattern in role_lower for pattern in web_ui_patterns):
        return False
    
    # Filter: "licensed to" patterns (legal/regulatory text, not roles)
    if 'licensed' in role_lower and ('to practice' in role_lower or 'by the state' in role_lower):
        return False
    
    # Filter: Department/Region patterns (e.g., "Public Sector, Canada", "Healthcare, UK")
    # These appear in LinkedIn when showing department + location, not job titles
    # Pattern: "[Department/Sector], [Country/Region]"
    country_indicators = [
        ', canada', ', uk', ', usa', ', us', ', united states', ', united kingdom',
        ', india', ', australia', ', germany', ', france', ', spain', ', italy',
        ', mexico', ', brazil', ', china', ', japan', ', singapore', ', ireland'
    ]
    if any(indicator in role_lower for indicator in country_indicators):
        # Exception: If it has clear role keywords, it might be "VP Sales, Canada" (valid)
        clear_role_keywords = ['vice president', 'director', 'manager', 'ceo', 'cto', 'cfo', 
                               'head', 'lead', 'engineer', 'analyst', 'consultant']
        has_clear_role = any(kw in role_lower for kw in clear_role_keywords)
        if not has_clear_role:
            return False
    
    # Filter: "Sector" without role keywords (e.g., "Public Sector", "Private Sector", "Healthcare Sector")
    # These are department/industry descriptors, not job titles
    if 'sector' in role_lower:
        # Must have a clear role keyword to be valid (e.g., "Public Sector Director" is valid)
        sector_role_keywords = ['director', 'manager', 'head', 'lead', 'coordinator', 
                                'specialist', 'analyst', 'consultant', 'officer']
        if not any(kw in role_lower for kw in sector_role_keywords):
            return False
    
    # Filter: Truncated/malformed role endings (e.g., "Executive Assistant Human" from "Human Resources")
    # These indicate concatenation errors or incomplete extraction
    suspicious_endings = [
        ' human$',  # "Executive Assistant Human" (should be "Human Resources")
        ' resources$',  # Standalone (should be paired with department)
        ' services$',  # Standalone (should be paired with type)
        ' operations$',  # Standalone (should be paired with type)
        ' support$',  # Too generic alone
        ' team$',  # "Marketing Team" is not a role
        ' department$',  # Department name, not role
    ]
    
    for ending in suspicious_endings:
        # Only filter if it's a trailing word that doesn't make sense as a role
        if re.search(ending, role_lower):
            # Exception: "Human Resources" together is valid
            if ending == ' human$' and role_lower.endswith(' human'):
                # Check if it's "Executive Assistant Human" type pattern (invalid)
                # vs "Director of Human Relations" (valid with "of")
                if ' of ' not in role_lower and ' for ' not in role_lower:
                    return False
            # Exception: "Customer Support" is valid
            elif ending == ' support$' and any(prefix in role_lower for prefix in ['customer', 'technical', 'client', 'it ']):
                continue
            # Exception: "Business Operations" is valid  
            elif ending == ' operations$' and any(prefix in role_lower for prefix in ['business', 'field', 'plant', 'network']):
                continue
            # Exception: "Professional Services" is valid
            elif ending == ' services$' and any(prefix in role_lower for prefix in ['professional', 'managed', 'cloud', 'customer']):
                continue
            # Exception: "Human Resources" together is valid
            elif ending == ' resources$' and 'human resources' in role_lower:
                continue
            else:
                return False
    
    return True


def validate_role_format(role: str, full_name: str = "", company: str = "") -> Tuple[bool, str]:
    """
    Validate role FORMAT for gaming patterns BEFORE fuzzy matching.
    
    This catches malformed roles that might pass content matching:
    - Person's name embedded in role
    - Company name embedded in role ("at Cloudfactory", "Morgan Stanley Wealth Management")
    - Marketing taglines/sentences
    - Geographic locations at end (should be in region field)
    - Excessively long roles (> 80 chars)
    
    Returns: (is_valid: bool, rejection_reason: str)
    """
    if not role or not role.strip():
        return False, "Empty role"
    
    role = role.strip()
    role_lower = role.lower()
    
    # ========================================================================
    # CHECK 1: Role too long (legitimate titles are < 80 chars)
    # ========================================================================
    # Examples of valid long roles:
    #   - "Vice President of Global Sales and Marketing" (47 chars)
    #   - "Senior Director of Enterprise Customer Success" (47 chars)
    # Examples of gaming (stuffed roles):
    #   - "CEO/Board Member at Cloudfactory. Unlocking the Disruptive Potential..." (130+ chars)
    # ========================================================================
    if len(role) > 80:
        return False, f"Role too long ({len(role)} chars > 80). Remove taglines and extra info."
    
    # ========================================================================
    # CHECK 2: Marketing sentences/taglines (period followed by sentence)
    # ========================================================================
    # Pattern: ". X" where X is capital letter starting a sentence
    # Examples:
    #   - "CEO. Unlocking the potential of AI for the world" → FAIL
    #   - "Sr. Director" → PASS (abbreviation)
    #   - "V.P. of Sales" → PASS (abbreviation)
    # ========================================================================
    # Look for sentence patterns (period + space + 3+ words)
    if re.search(r'\.\s+[A-Z][a-z]+\s+[a-z]+\s+[a-z]+', role):
        return False, "Role contains marketing sentence/tagline. Use just the job title."
    
    # ========================================================================
    # CHECK 3: Role ends with country/city names (geographic gaming)
    # ========================================================================
    # Pattern: "- Vietnam, Cambodia" or "- Asia Pacific"
    # These should be in the region field, not role
    # ========================================================================
    geographic_endings = [
        # Countries
        r'[-–,]\s*(Vietnam|Cambodia|India|China|Philippines|Indonesia|Thailand|Malaysia|Singapore)',
        r'[-–,]\s*(Mexico|Canada|Brazil|Argentina|Chile|Colombia)',
        r'[-–,]\s*(Germany|France|UK|Spain|Italy|Netherlands|Belgium|Switzerland|Austria)',
        r'[-–,]\s*(Japan|Korea|Taiwan|Hong Kong)',
        r'[-–,]\s*(Australia|New Zealand)',
        r'[-–,]\s*(Nigeria|Kenya|Egypt|South Africa|Morocco)',
        r'[-–,]\s*(UAE|Saudi Arabia|Qatar|Kuwait)',
        r'[-–,]\s*(United States|United Kingdom)',
        # Regions (if at end of role)
        r'[-–]\s*(APAC|EMEA|LATAM|MENA)\s*$',
        r'[-–]\s*(Asia Pacific|Asia-Pacific)\s*$',
    ]
    for pattern in geographic_endings:
        if re.search(pattern, role, re.IGNORECASE):
            return False, "Role ends with geographic location. Put location in region/country field."
    
    # ========================================================================
    # CHECK 4: "at [Company]" pattern embedded in role
    # ========================================================================
    # Pattern: "CEO at CloudFactory" or "CEO/Board Member at Cloudfactory"
    # The role should NOT include where the person works
    # ========================================================================
    if company:
        company_lower = company.lower().strip()
        # Escape regex special characters in company name
        company_escaped = re.escape(company_lower)
        # Check for "at {company}" pattern
        if re.search(rf'\bat\s+{company_escaped}', role_lower):
            return False, f"Role contains 'at {company}'. Just provide the job title."
        # Also check for company name at the end of role
        # E.g., "Managing Director, Chief Operating Officer- Field Management, Morgan Stanley Wealth Management"
        if role_lower.rstrip().endswith(company_lower):
            return False, f"Role ends with company name '{company}'. Just provide the job title."
        # Check for company name anywhere with a comma separator
        # E.g., "VP Sales, Morgan Stanley"
        if re.search(rf',\s*{company_escaped}\s*$', role_lower):
            return False, f"Role ends with ', {company}'. Just provide the job title."
    
    # ========================================================================
    # CHECK 5: Person's name embedded in role
    # ========================================================================
    # Pattern: "Jones - Associate Director -" where "Jones" is the person's last name
    # The role should NOT include the person's name
    # Exception: Very short name parts (< 3 chars) might match legitimate words
    # ========================================================================
    if full_name:
        name_parts = full_name.lower().split()
        for name_part in name_parts:
            # Skip very short name parts (might match words like "VP", "IT", etc.)
            if len(name_part) < 3:
                continue
            # Skip common name parts that could be legitimate words in roles
            # E.g., "Grant" (name) vs "Grant Manager" (role)
            common_role_words = ['grant', 'case', 'mark', 'bill', 'will', 'ray', 'joy', 'hope', 'faith', 'grace', 'dean', 'chase']
            if name_part in common_role_words:
                continue
            # Check if name appears as a standalone word in role
            if re.search(rf'\b{re.escape(name_part)}\b', role_lower):
                return False, f"Role contains person's name '{name_part}'. Just provide the job title."
    
    # ========================================================================
    # CHECK 6: Multiple distinct roles (job title stuffing)
    # ========================================================================
    # Pattern: "Managing Director, Chief Operating Officer- Field Management"
    # Multiple C-suite or Director-level titles in one field suggests gaming
    # CAREFUL: "Co-Founder & CEO" is valid (Founder + 1 role)
    # ========================================================================
    c_suite_count = 0
    director_count = 0
    # Use word boundary matching to avoid false positives like 'cto' in 'director'
    c_suite_patterns = ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cpo', 'chief executive', 'chief technology', 
                        'chief financial', 'chief operating', 'chief marketing', 'chief information', 'chief product']
    director_patterns = ['managing director', 'executive director', 'senior director', 'director of']
    
    for pattern in c_suite_patterns:
        # Use word boundary to avoid matching 'cto' in 'director' or 'coo' in 'coordinator'
        if re.search(rf'\b{re.escape(pattern)}\b', role_lower):
            c_suite_count += 1
    for pattern in director_patterns:
        if re.search(rf'\b{re.escape(pattern)}\b', role_lower):
            director_count += 1
    
    # Allow: "CEO & Co-Founder" (1 c-suite + founder)
    # Allow: "VP of Sales & Marketing" (1 role, multiple functions)
    # Fail: "CEO, CFO" (2 c-suite roles)
    # Fail: "Managing Director, Chief Operating Officer" (director + c-suite)
    has_founder = 'founder' in role_lower
    if c_suite_count > 1:
        return False, f"Role contains multiple C-suite titles. Submit one role per lead."
    if c_suite_count >= 1 and director_count >= 1 and not has_founder:
        return False, f"Role contains both C-suite and Director titles. Submit one role per lead."
    
    # ========================================================================
    # CHECK 6b: Multiple comma-separated distinct roles (non-C-suite stuffing)
    # ========================================================================
    # Pattern: "Director of Marketing, Analyst, Operations Manager"
    # This catches cases where miner stuffs multiple different job titles
    # CAREFUL: Don't catch "VP of Sales and Marketing" (one role, multiple depts)
    # CAREFUL: Don't catch "Senior Engineer, Backend" (role + specialization)
    # ========================================================================
    # Role keywords that indicate a distinct job title
    role_title_keywords = [
        'manager', 'director', 'analyst', 'engineer', 'developer', 'designer',
        'coordinator', 'specialist', 'consultant', 'advisor', 'associate',
        'executive', 'officer', 'president', 'owner', 'partner', 'principal',
        'lead', 'head', 'supervisor', 'administrator', 'representative'
    ]
    
    # Split by comma and check each segment for role keywords
    if ',' in role:
        segments = [s.strip().lower() for s in role.split(',') if s.strip()]
        segments_with_roles = []
        for seg in segments:
            # Check if this segment contains a role keyword
            for keyword in role_title_keywords:
                if re.search(rf'\b{keyword}\b', seg):
                    segments_with_roles.append(seg)
                    break  # Only count once per segment
        
        # If 3+ comma-separated segments each contain a role keyword, reject
        # (Using 3+ to avoid false positives on "Director of Marketing, Sales")
        if len(segments_with_roles) >= 3:
            return False, f"Role contains {len(segments_with_roles)} distinct job titles. Submit one role per lead."
    
    # ========================================================================
    # CHECK 7: Trailing dashes/garbage formatting
    # ========================================================================
    # Pattern: "Associate Director -" or "- VP Sales -"
    # Clean formatting shouldn't have leading/trailing dashes
    # ========================================================================
    if re.search(r'^\s*[-–]\s*|\s*[-–]\s*$', role):
        return False, "Role has trailing/leading dashes. Clean up formatting."
    
    return True, ""


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
    
    # ANTI-GAMING: Expanded department/function list to catch role finessing
    # E.g., "VP of Risk" vs "VP of Treasury" should FAIL (different functions)
    # NOTE: Multi-word departments MUST come first so "business development" matches before "business"
    departments = [
        # Multi-word departments first (check these before single words)
        "business development", "supply chain", "human resources",
        "customer success", "customer service", "account management",
        "change management", "project management",
        "information security", "cybersecurity",
        # Single-word departments
        "sales", "marketing", "engineering", "finance", "operations",
        "product", "hr", "legal", "it", "technology",
        "customer", "business", "development", "research", "data",
        # Financial functions
        "risk", "treasury", "accounting", "audit", "compliance",
        # Operations functions
        "logistics", "procurement", "readiness",
        # Strategy/Leadership functions
        "strategy", "transformation",
        # Security/Tech functions
        "security", "infrastructure",
        # Customer-facing
        "support"
    ]
    
    # Equivalent department pairs - these are interchangeable (GTM functions)
    EQUIVALENT_DEPT_PAIRS = {
        frozenset({"sales", "business development"}),
        frozenset({"sales", "business"}),  # "business development" sometimes matches "business"
        frozenset({"business development", "business"}),
        frozenset({"hr", "human resources"}),
    }
    
    def are_equivalent_depts(dept1: str, dept2: str) -> bool:
        """Check if two departments are considered equivalent."""
        if dept1 == dept2:
            return True
        return frozenset({dept1, dept2}) in EQUIVALENT_DEPT_PAIRS
    
    def get_department(r: str) -> Optional[str]:
        r_lower = r.lower()
        for dept in departments:
            if dept in r_lower:
                return dept
        return None
    
    claimed_dept = get_department(claimed_role)
    extracted_dept = get_department(extracted_role)
    
    if claimed_dept and extracted_dept and claimed_dept != extracted_dept:
        # First check: Are these equivalent departments? (e.g., sales ≈ business development)
        if are_equivalent_depts(claimed_dept, extracted_dept):
            pass  # Equivalent departments, continue to other checks
        else:
            # Second check: Multi-function roles that share a department
            # ANTI-GAMING: Only allow this if EXTRACTED has 2+ departments
            # (miner can't control what LinkedIn shows, so this prevents padding)
            extracted_depts = [d for d in departments if d in extracted_role.lower()]
            
            if len(extracted_depts) >= 2:
                # LinkedIn shows a genuine multi-function role
                # Check if there's overlap with claimed role
                common_depts = [d for d in extracted_depts if d in claimed_role.lower()]
                if not common_depts:
                    return False, 0.0, f"DEPARTMENT MISMATCH: {claimed_dept} ≠ {extracted_dept}"
                # else: common_depts exist, bypass allowed - continue to other checks
            else:
                # LinkedIn shows single-function role, miner claims different dept
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
# Patterns for location extraction
# Group 1: Use IGNORECASE (common phrases that can be any case)
# Group 2: Case-sensitive (require actual capitalization for city/state)
# Location pattern: matches City or "City, State" or "City, Country"
_LOC_CITY_STATE = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*(?:[A-Z]{2}|[A-Z][a-z]+))?'

LOCATION_PATTERNS_IGNORECASE = [
    # Primary patterns - most reliable
    rf'headquarter(?:ed|s)?\s+in\s+({_LOC_CITY_STATE})\b',   # "headquartered in X"
    rf'based\s+in\s+({_LOC_CITY_STATE})\b',                  # "based in San Jose, CA"
    rf'located\s+in\s+({_LOC_CITY_STATE})\b',                # "located in X"
    rf'offices?\s+in\s+({_LOC_CITY_STATE})\b',               # "office(s) in X"
    rf'hq\s+in\s+({_LOC_CITY_STATE})\b',                     # "hq in X"
    rf'hq:\s*({_LOC_CITY_STATE})\b',                         # "HQ: City"
    # Secondary patterns
    r'(?:^|[^\w])([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)-based\s+(?:company|firm|startup)',  # "Paris-based company"
    rf'company\s+(?:from|in)\s+({_LOC_CITY_STATE})\b',       # "company from/in X"
    # Location in context patterns
    rf'founded\s+in\s+({_LOC_CITY_STATE})\b',                # "founded in X"
    rf'started\s+in\s+({_LOC_CITY_STATE})\b',                # "started in X"
    rf'from\s+({_LOC_CITY_STATE})\s+(?:is|was|that)',        # "from San Francisco, CA is"
    rf'(?:startup|company|firm)\s+in\s+({_LOC_CITY_STATE})\b',  # "startup in X"
]

# Case-sensitive patterns for "City, ST" format
LOCATION_PATTERNS_CASESENSITIVE = [
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\s*[-–—]',  # "New York, NY -"
    r'\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b',     # "| New York, NY" (must end at word boundary)
    r',\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\s*$',    # ", City, ST" at end
    # Match standalone "City, State" or "City, Country" patterns in text
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:[A-Z]{2}|[A-Z][a-z]+))\b',  # "San Francisco, CA" or "Lyon, France"
]

# Major tech hub cities - for direct city name matching when patterns fail
MAJOR_CITIES = {
    # US Major Cities
    "san francisco", "new york", "los angeles", "chicago", "seattle", "austin", "boston",
    "denver", "atlanta", "miami", "dallas", "houston", "phoenix", "philadelphia",
    "san jose", "san diego", "portland", "baltimore", "washington", "detroit",
    "kansas city", "minneapolis", "tampa", "nashville", "cleveland", "pittsburgh",
    "cincinnati", "indianapolis", "columbus", "milwaukee", "salt lake city",
    "charlotte", "raleigh", "sacramento", "las vegas", "orlando", "st. louis",
    "richmond", "jacksonville", "memphis", "omaha", "new orleans", "buffalo",
    # International
    "london", "paris", "berlin", "amsterdam", "tokyo", "singapore", "sydney",
    "toronto", "vancouver", "dublin", "tel aviv", "bangalore", "mumbai", "beijing",
    "shanghai", "hong kong", "seoul", "lyon", "munich", "zurich", "stockholm",
    "melbourne", "auckland", "copenhagen", "oslo", "helsinki", "prague", "vienna"
}

# Map nationality adjectives to countries (for "American company" → "United States")
NATIONALITY_TO_COUNTRY = {
    "american": "United States",
    "french": "France", 
    "german": "Germany",
    "british": "United Kingdom",
    "canadian": "Canada",
    "australian": "Australia",
    "japanese": "Japan",
    "chinese": "China",
    "indian": "India",
    "brazilian": "Brazil",
    "mexican": "Mexico",
    "spanish": "Spain",
    "italian": "Italy",
    "dutch": "Netherlands",
    "swiss": "Switzerland",
    "swedish": "Sweden",
    "norwegian": "Norway",
    "danish": "Denmark",
    "finnish": "Finland",
    "irish": "Ireland",
    "singaporean": "Singapore",
    "korean": "South Korea",
}

def _is_valid_location(location: str) -> bool:
    """Check if extracted text is a valid location (not garbage)."""
    if not location:
        return False
    
    location_lower = location.lower()
    
    # Check if it's in "City, State/Country" format (e.g., "Media, US")
    # If so, skip garbage pattern checks for city names that might match garbage words
    is_city_state_format = bool(re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,}$', location))
    
    # Reject obvious garbage patterns (but not if it's a valid City, State format)
    garbage_patterns = [
        # Business/company terms
        'products', 'competitors', 'valuation', 'funding', 'revenue',
        'technology', 'entertainment', 'software', 'services', 'solutions',
        'company', 'corporation', 'enterprise', 'business', 'industry',
        'profile', 'overview', 'about', 'description', 'information',
        'employees', 'staff', 'team', 'board', 'members', 'contacts',
        'education', 'internet', 'partnerships', 'news',
        # NOTE: 'media' removed from here - it's a valid city name (Media, PA)
        'silicon', 'bay area',  # Too generic - reject "Silicon Valley" / "Bay Area"
        # Generic web terms
        'linkedin', 'crunchbase', 'wikipedia', 'facebook', 'twitter',
        # Too generic
        'global', 'worldwide', 'international', 'regional',
        'united states', 'usa',  # Too generic
        # Business departments/units (CRITICAL NEW FILTERS)
        'sales', 'marketing', 'operations', 'engineering', 'hr', 'finance',
        'accounting', 'legal', 'it', 'support', 'customer', 'business development',
        # Street address indicators (NOT locations)
        'street', 'avenue', 'boulevard', 'drive', 'road', 'lane', 'way',
        'court', 'circle', 'plaza', 'square', 'parkway', 'highway',
        'suite', 'floor', 'building', 'tower', 'complex', 'center',
        'crescent', 'block', 'terrace', 'mews', 'close', 'grove',
        'ste', 'apt', 'unit', 'room', 'no.', '#',
        # Product/material names
        'glass', 'steel', 'wood', 'metal', 'plastic', 'ceramic',
        'stained', 'colored', 'painted',
        # CRITICAL: Company suffixes (NOT locations!)
        ' inc', ' inc.', ' llc', ' corp', ' corp.', ' ltd', ' ltd.',
        ' co.', ' company', ' group', ' enterprises', ' holdings',
    ]
    
    # Only apply garbage filters if NOT in City, State format
    if not is_city_state_format:
        if any(garbage in location_lower for garbage in garbage_patterns):
            return False
    
    # Reject if it's just state codes without city
    if re.match(r'^[A-Z]{2}$', location):
        return False
    
    # CRITICAL: Reject duplicate words (e.g., "Modotech Modotech")
    # Check before comma (to catch "Modotech Modotech, Inc")
    first_part = location.split(',')[0].strip()
    words = first_part.split()
    if len(words) >= 2 and words[0].lower() == words[1].lower():
        return False
    
    # CRITICAL: Reject person name patterns (e.g., "Mike Shaughnessy, Mike")
    # Pattern: "FirstName LastName, FirstName" - person name with repeated first name
    if ',' in location:
        parts = location.split(',')
        before_comma = parts[0].strip()
        after_comma = parts[1].strip() if len(parts) > 1 else ""
        
        # Check if it looks like "FirstName LastName, FirstName"
        before_words = before_comma.split()
        after_words = after_comma.split() if after_comma else []
        
        if len(before_words) == 2 and len(after_words) >= 1:
            # Check if first word before comma matches first word after comma (e.g., "Mike Shaughnessy, Mike")
            if before_words[0].lower() == after_words[0].lower():
                return False
    
    # CRITICAL: Reject reversed city/state (e.g., "York, New" should be "New York")
    # Check if it's "Word, Word" where the second word is a common city prefix
    if ', ' in location:
        parts = location.split(', ')
        if len(parts) == 2:
            first, second = parts
            # Common city first-word patterns
            common_city_prefixes = ['new', 'san', 'los', 'fort', 'mount', 'saint', 'st.', 'port', 'lake']
            if second.lower() in common_city_prefixes:
                return False  # This is reversed! (e.g., "York, New" instead of "New York")
    
    # Reject if too long (likely a description, not a location)
    if len(location) > 50:
        return False
    
    # Reject if starts with articles or prepositions  
    if location_lower.startswith(('the ', 'a ', 'an ', 'in ', 'at ', 'on ')):
        return False
    
    # Must contain at least some location-like content
    # Either a known city, state, or comma-separated format
    has_comma = ',' in location
    has_known_state = any(state in location_lower for state in [
        # US States (comprehensive list)
        'california', 'new york', 'texas', 'florida', 'washington', 'massachusetts',
        'illinois', 'georgia', 'colorado', 'oregon', 'pennsylvania', 'ohio',
        'virginia', 'north carolina', 'michigan', 'arizona', 'maryland', 'tennessee',
        'alabama', 'alaska', 'arkansas', 'connecticut', 'delaware', 'hawaii', 'idaho',
        'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire',
        'new jersey', 'new mexico', 'north dakota', 'oklahoma', 'rhode island',
        'south carolina', 'south dakota', 'utah', 'vermont', 'west virginia',
        'wisconsin', 'wyoming',
        # International Countries/Regions
        'canada', 'united kingdom', 'france', 'germany', 'australia', 'singapore',
        'south africa', 'ireland', 'leinster', 'denmark', 'sweden', 'norway',
        'finland', 'netherlands', 'belgium', 'switzerland', 'austria', 'spain',
        'italy', 'portugal', 'japan', 'china', 'india', 'brazil', 'mexico',
        'argentina', 'chile', 'colombia'
    ])
    has_known_city = any(city in location_lower for city in MAJOR_CITIES)
    
    return has_comma or has_known_state or has_known_city


def extract_person_location_from_linkedin_snippet(snippet: str) -> Optional[str]:
    """
    Extract person's location from LinkedIn search result snippet.
    
    LinkedIn snippets typically show the profile header location in formats like:
    - End of snippet: "...School of Business. New York, New York, United States."
    - Middle of snippet: "...10 months. Manhattan, New York, United States..."
    - Directory format: "New York, NY. Nasdaq, +3 more."
    - Location prefix: "Location: New York"
    
    This extracts the PERSON's location (from their profile header),
    NOT the company headquarters.
    
    Returns:
        Location string if found, None otherwise
    """
    if not snippet:
        return None
    
    # Known countries for validation
    COUNTRIES = {
        'united states', 'united kingdom', 'canada', 'australia', 'germany', 
        'france', 'spain', 'italy', 'netherlands', 'india', 'singapore',
        'japan', 'china', 'brazil', 'mexico', 'ireland', 'switzerland',
        'sweden', 'norway', 'denmark', 'finland', 'belgium', 'austria',
        'new zealand', 'south africa', 'israel', 'uae', 'united arab emirates',
        'hong kong', 'taiwan', 'south korea', 'poland', 'czech republic',
        'portugal', 'greece', 'argentina', 'chile', 'colombia', 'peru',
        'russia', 'turkey', 'egypt', 'nigeria', 'kenya', 'indonesia',
        'malaysia', 'thailand', 'vietnam', 'philippines'
    }
    
    # US state abbreviations for "City, ST" format
    US_ABBREVS = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
        'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
        'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
        'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
        'WI', 'WY', 'DC'
    }
    
    # Pattern 1: Full location at END of snippet with country
    # Matches: "...School of Business. New York, New York, United States."
    pattern_full_end = r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\.?\s*$'
    match = re.search(pattern_full_end, snippet)
    if match:
        location = match.group(1).strip().rstrip('.')
        parts = [p.strip() for p in location.split(',')]
        if len(parts) >= 2 and parts[-1].lower() in COUNTRIES:
            return location
    
    # Pattern 2: Full location in MIDDLE of snippet with country
    # Matches: "...10 months. Manhattan, New York, United States..."
    pattern_full_middle = r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)(?:\s*[·\.\|]|\s+\d)'
    match = re.search(pattern_full_middle, snippet)
    if match:
        location = match.group(1).strip()
        parts = [p.strip() for p in location.split(',')]
        if len(parts) >= 2 and parts[-1].lower() in COUNTRIES:
            return location
    
    # Pattern 3: Abbreviated US location (City, ST) anywhere in snippet
    # Matches: "New York, NY" or "San Francisco, CA"
    pattern_abbrev = r'([A-Z][a-zA-Z\s]+,\s*(' + '|'.join(US_ABBREVS) + r'))\b'
    match = re.search(pattern_abbrev, snippet)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: Location with "Location:" prefix (from LinkedIn directory pages)
    # Matches: "Location: New York" or "Location: 600039"
    pattern_prefix = r'Location:\s*([A-Z][a-zA-Z\s,]+?)(?:\s*[·\|]|\s+\d|\s*$)'
    match = re.search(pattern_prefix, snippet)
    if match:
        location = match.group(1).strip()
        # Skip numeric-only locations (postal codes)
        if not location.isdigit():
            return location
    
    # Pattern 5: Metro areas
    # Matches: "San Francisco Bay Area", "Greater New York City Area"
    pattern_metro = r'((?:Greater\s+)?[A-Z][a-zA-Z\s]+(?:Bay\s+Area|Metro(?:politan)?\s+Area|City\s+Area))'
    match = re.search(pattern_metro, snippet)
    if match:
        return match.group(1).strip()
    
    # Pattern 6: Two-part location at end (City, Country) - no state
    # Matches: "...profile. London, United Kingdom."
    pattern_two_part = r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\.?\s*$'
    match = re.search(pattern_two_part, snippet)
    if match:
        location = match.group(1).strip().rstrip('.')
        parts = [p.strip() for p in location.split(',')]
        if len(parts) == 2 and parts[-1].lower() in COUNTRIES:
            return location
    
    return None


def extract_location_from_text(text: str) -> Optional[str]:
    """Extract location from text using regex patterns."""
    if not text:
        return None
    
    # Try case-insensitive patterns first (headquartered in, based in, located in)
    for pattern in LOCATION_PATTERNS_IGNORECASE:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            location = re.sub(r'\s*\|.*$', '', location)
            location = re.sub(r'\s*-.*$', '', location)
            # Validate: reject garbage
            if not _is_valid_location(location):
                continue
            return location
    
    # Try case-sensitive patterns (City, ST format)
    for pattern in LOCATION_PATTERNS_CASESENSITIVE:
        match = re.search(pattern, text)  # No IGNORECASE
        if match:
            location = match.group(1).strip()
            if _is_valid_location(location):
                return location
    
    # Try nationality patterns (e.g., "American company" → "United States")
    text_lower = text.lower()
    for nationality, country in NATIONALITY_TO_COUNTRY.items():
        if re.search(rf'\b{nationality}\b', text_lower):
            # Make sure it's in context of company description
            if any(ctx in text_lower for ctx in ['company', 'corporation', 'firm', 'business', 'enterprise', 'multinational']):
                return country
    
    # Last resort: Look for major tech hub cities mentioned in text
    for city in MAJOR_CITIES:
        # Match city as whole word with possible state/country after
        pattern = rf'\b({re.escape(city)}(?:,?\s*[A-Z]{{2}})?)\b'
        match = re.search(pattern, text_lower)
        if match:
            # Find the actual case-preserved text from original
            start = match.start(1)
            end = match.end(1)
            original_match = text[start:end]
            # Only return if it looks like a location reference (not part of company name)
            # Check context: should have location-related context nearby
            context_start = max(0, start - 30)
            context_end = min(len(text), end + 30)
            context = text[context_start:context_end].lower()
            location_context_words = ['based', 'headquarter', 'located', 'office', 'hq', 'from', 'in', 'city', 'area']
            if any(word in context for word in location_context_words):
                return original_match.title()
    
    return None


def fuzzy_pre_verification_stage5(
    claimed_role: str,
    claimed_region: str,
    claimed_industry: str,
    role_search_results: List[Dict],
    region_search_results: List[Dict],
    industry_search_results: List[Dict],
    full_name: str = "",
    company: str = "",
    role_only: bool = False,
    role_verified_stage4: bool = False
) -> Dict:
    """
    Pre-verify ROLE and REGION using fuzzy matching BEFORE sending to LLM.
    INDUSTRY is ALWAYS sent to LLM.
    
    Args:
        role_only: If True, only check role and suppress region/industry messages.
                   Used for early exit check before region/industry ScrapingDog searches.
        role_verified_stage4: If True, role was already verified in Stage 4 (don't print confusing warnings).
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
    if role_search_results and claimed_role:
        # Check if role was VERIFIED by fallback search (name+company+role confirmed)
        for r in role_search_results[:3]:  # Check first few results
            if r.get("role_verified") and r.get("verified_role"):
                verified_role = r["verified_role"]
                print(f"   ✅ ROLE PRE-VERIFIED: '{verified_role}' (from fallback verification)")
                result["role_verified"] = True
                result["role_extracted"] = verified_role
                result["role_confidence"] = 1.0
                result["role_reason"] = f"Role verified via fallback search (name+company+role confirmed)"
                result["needs_llm"] = [x for x in result["needs_llm"] if x != "role"]  # Don't need LLM
                # Still continue to check region/industry
                break
        
        if not result["role_verified"]:  # Only do fuzzy matching if not already verified
            name_lower = full_name.lower() if full_name else ""
            first_name = name_lower.split()[0] if name_lower else ""
            last_name = name_lower.split()[-1] if name_lower else ""
            
            best_extracted_role = None
            best_match = False
            best_confidence = 0.0
            best_reason = "No role found in ScrapingDog results"
            
            # DEBUG: Show what we're trying to extract from
            print(f"   🔍 DEBUG ROLE: Attempting to extract role from {len(role_search_results)} search results")
            print(f"      Claimed role: '{claimed_role}'")
            print(f"      Person: '{full_name}' at '{company}'")
            
            # Look at up to 15 results to include fallback results (5 primary + 5 fallback1 + 5 fallback2)
            for idx, r in enumerate(role_search_results[:15], 1):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))
                link = r.get("href", r.get("link", ""))
                
                # DEBUG: Show what text we're processing
                print(f"      Result {idx}: Title: '{title[:80]}{'...' if len(title) > 80 else ''}'")
                print(f"                 Snippet: '{snippet[:80]}{'...' if len(snippet) > 80 else ''}'")
                
                # CRITICAL: Filter out LinkedIn POSTS - they contain garbage text
                # Posts have URLs like: linkedin.com/posts/username/activity-123456
                # We only want PROFILES: linkedin.com/in/username
                if "/posts/" in link or "/feed/" in link or "/pulse/" in link:
                    print(f"                 SKIPPED (LinkedIn post/feed)")
                    continue  # Skip posts, only process profiles and company pages
                
                title_lower = title.lower()
                
                # Check if title contains the person's name OR is a job posting/company role listing
                # Job postings like "Company hiring Role" don't contain the person's name but prove the role exists
                is_job_posting = "hiring" in title_lower
                is_company_role = company and company.lower() in title_lower
                
                if first_name and last_name and not is_job_posting:
                    first_pattern = rf'\b{re.escape(first_name)}\b'
                    last_pattern = rf'\b{re.escape(last_name)}\b'
                    has_name_in_title = re.search(first_pattern, title_lower) and re.search(last_pattern, title_lower)
                    
                    # ALSO check if name is in snippet (for directory-style listings)
                    # e.g., Title: "Steven Rimpici - VP Sales", Snippet: "Ty Horner. Senior VP, Marketing. Email..."
                    snippet_lower = snippet.lower() if snippet else ""
                    has_name_in_snippet = snippet_lower and re.search(first_pattern, snippet_lower) and re.search(last_pattern, snippet_lower)
                    
                    has_name = has_name_in_title or has_name_in_snippet
                    
                    # Allow if: has person's name (in title OR snippet), OR is a job posting, OR is a company role listing
                    if not has_name and not is_company_role:
                        print(f"                 SKIPPED (name '{full_name}' not found in title or snippet)")
                        continue
                
                extracted = extract_role_from_search_title(title, snippet, company_name=company, full_name=full_name)
                
                if extracted:
                    print(f"                 📝 EXTRACTED: '{extracted}'")
                else:
                    print(f"                 ❌ NO ROLE extracted (filters rejected or no role found)")
                
                if extracted:
                    extracted_lower = extracted.lower()
                    
                    # Pre-compute: Check for strong role keywords (used in multiple filters)
                    role_keywords_quick = ["ceo", "cto", "cfo", "coo", "founder", "president", "director", 
                                           "manager", "head", "lead", "vp", "chief", "officer", "attorney",
                                           "engineer", "accountant", "physician", "partner", "principal"]
                    has_strong_role_keyword = any(kw in extracted_lower for kw in role_keywords_quick)
                    
                    # Filter 1: Known invalid site names and domains
                    invalid_extractions = ["wikipedia", "linkedin", "facebook", "twitter", "crunchbase", 
                                           "glassdoor", "indeed", "zoominfo", "bloomberg", "forbes", "reuters",
                                           "craft.co", "theorg.com", "the org", "contactout", "rocketreach",
                                           "apollo.io", "leadiq", "lusha", "seamless.ai", "hunter.io",
                                           "clearbit", "datanyze", "discoverorg", "insideview", "owler"]
                    if extracted_lower in invalid_extractions:
                        print(f"                 REJECTED (invalid site name: '{extracted}')")
                        continue
                    # Filter website domains (anything ending in .com, .co, .io, etc.)
                    if re.match(r'^[\w\-]+\.(com|co|io|org|net)$', extracted_lower):
                        print(f"                 REJECTED (looks like domain: '{extracted}')")
                        continue
                    
                    # Filter 1b: Too short/generic extractions
                    # Common role abbreviations that are valid even if ≤2 chars
                    common_abbreviations = ['vp', 'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cpo', 'svp', 'evp']
                    is_common_abbrev = extracted_lower in common_abbreviations
                    
                    too_short_generic = ["lead", "head", "manager", "director", "partner", "officer", 
                                        "engineer", "analyst", "the org", "the company", "org", "inc", "llc"]
                    if extracted_lower in too_short_generic:
                        print(f"                 REJECTED (too generic/short: '{extracted}')")
                        continue
                    if len(extracted) < 5 and not is_common_abbrev:
                        print(f"                 REJECTED (too short: {len(extracted)} chars)")
                        continue
                    
                    # Filter 1c: Truncated/garbage extractions
                    if "..." in extracted or extracted_lower.endswith("- linkedin") or extracted_lower.endswith("| linkedin"):
                        print(f"                 REJECTED (truncated/garbage)")
                        continue
                    
                    # Filter 2: Garbage patterns that contain role keywords but aren't roles
                    # Be specific to avoid false positives (e.g., "email example" in titles shouldn't block)
                    garbage_patterns = [
                        "work history", "executive bio", "company profile", "contact info",
                        "phone number", "email address", "company overview", "about us",
                        "company headquarters", "company website", "biography of",
                        "practice on the", "focuses on the", "specializes in the",
                        "expertise in the", "experience in the", "works in the",
                        "involved in the", "engaged in the", "active in the",
                        "known for the", "recognized for", "awarded for",
                        "session details", "read more", "click here", "learn more",
                        "view profile", "see the complete", "view full", "show more"
                    ]
                    # Only filter if these patterns appear AND no strong role keyword exists
                    has_garbage = any(pattern in extracted_lower for pattern in garbage_patterns)
                    if has_garbage and not has_strong_role_keyword:
                        print(f"                 REJECTED (garbage pattern found, no role keyword)")
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
                        print(f"                 REJECTED (looks like location, no role keyword)")
                        continue
                    
                    # Filter 4: Too long to be a job title (likely garbage)
                    # But allow longer strings if they contain clear role indicators
                    # (Yahoo sometimes concatenates multiple results which still contain valid roles)
                    # Note: Yahoo can produce VERY long concatenated titles (300+ chars) but still contain valid roles
                    if len(extracted) > 500:
                        print(f"                 REJECTED (too long: {len(extracted)} chars)")
                        continue
                    if len(extracted) > 100 and not has_strong_role_keyword:
                        print(f"                 REJECTED (long but no role keyword: {len(extracted)} chars)")
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
                    
                    # Check if this result mentions the target person's name
                    has_target_name = False
                    if first_name and last_name and title_lower:
                        first_pattern = rf'\b{re.escape(first_name)}\b'
                        last_pattern = rf'\b{re.escape(last_name)}\b'
                        has_target_name = re.search(first_pattern, title_lower) and re.search(last_pattern, title_lower)
                    
                    # Prioritize results that:
                    # 1. Have higher confidence
                    # 2. Mention the target person's name (avoids extracting from wrong person)
                    # 3. Have strong role keywords
                    # CRITICAL: Don't accept low-confidence extractions without name match
                    should_update = False
                    
                    if confidence > best_confidence:
                        # Always update if confidence is better
                        should_update = True
                    elif not best_extracted_role and extracted:
                        # Accept first extraction ONLY if:
                        # - It has the target person's name, OR
                        # - It has strong role keywords AND reasonable confidence
                        if has_target_name:
                            should_update = True
                        elif has_strong_role_keyword and confidence >= 0.3:
                            should_update = True
                        # If no strong role keyword and no name match, reject
                    elif has_target_name and not best_match:
                        # Prefer results with target name over generic extractions
                        should_update = True
                    
                    if should_update:
                        print(f"                 ✅ ACCEPTED: '{extracted}' (confidence: {confidence:.0%})")
                        best_extracted_role = extracted
                        best_match = is_match
                        best_confidence = confidence
                        best_reason = reason
                    else:
                        if extracted:
                            print(f"                 REJECTED (not better than current best)")
            
            # DEBUG: Summary of role extraction
            if not best_extracted_role:
                print(f"      ❌ FINAL: No valid role extracted from {len(role_search_results)} results")
            else:
                print(f"      ✅ FINAL: Best role = '{best_extracted_role}' (confidence: {best_confidence:.0%})")
            
            if best_extracted_role:
                result["role_extracted"] = best_extracted_role
                result["role_confidence"] = best_confidence
                result["role_reason"] = best_reason
                
                if best_match and best_confidence >= 0.8:
                    result["role_verified"] = True
                    print(f"   ✅ FUZZY ROLE MATCH: '{claimed_role}' ≈ '{best_extracted_role}'")
                    print(f"      Confidence: {best_confidence:.0%} | Reason: {best_reason}")
                elif best_match:
                    result["needs_llm"].append("role")
                    print(f"   ⚠️ FUZZY ROLE: Low confidence match ({best_confidence:.0%}), sending to LLM")
                else:
                    # Match failed - check if extraction looks like a valid role
                    result["needs_llm"].append("role")
                    print(f"   ❌ FUZZY ROLE: No match - '{claimed_role}' vs '{best_extracted_role}' ({best_confidence:.0%})")
            else:
                # No role extracted at all from search results
                result["needs_llm"].append("role")
                print(f"   ⚠️ FUZZY ROLE: No role extracted from ScrapingDog results")
        else:
            result["needs_llm"].append("role")
            result["role_reason"] = "Could not extract role from ScrapingDog results"
            print(f"   ⚠️ FUZZY ROLE: Could not extract role from search results, sending to LLM")
    else:
        # This triggers when role_search_results is empty (intentionally not searched)
        # OR when claimed_role is empty
        if role_verified_stage4:
            # Role was already verified in Stage 4 - don't print any warning
            print(f"   ℹ️  FUZZY ROLE: Skipped (already verified by Stage 4)")
        elif not role_search_results and claimed_role:
            # Role search was intentionally skipped for some other reason
            print(f"   ℹ️  FUZZY ROLE: Skipped (role search not performed)")
        elif not claimed_role:
            print(f"   ℹ️  FUZZY ROLE: Skipped (no role claimed by miner)")
        else:
            print(f"   ⚠️ FUZZY ROLE: No ScrapingDog results")
        
        # Don't add to needs_llm if already verified by Stage 4
        if not role_verified_stage4:
            result["needs_llm"].append("role")
    
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
            # e.g., "Arkansas" should not match "kansas" (use word boundaries)
            pattern = r'\b' + re.escape(state) + r'\b'
            if re.search(pattern, claimed_lower):
                states_found.add(state)
        
        # Special case: "west virginia" should not also count "virginia"
        if 'west virginia' in states_found and 'virginia' in states_found:
            states_found.discard('virginia')
        
        if len(states_found) >= 2:
            result["region_verified"] = False
            result["region_hard_fail"] = True
            result["region_confidence"] = 0.0
            result["region_reason"] = f"HARD FAIL: Multiple US states in claimed region: {states_found}"
            result["region_extracted"] = "REJECTED - multiple states detected"
            print(f"   ❌ ANTI-GAMING HARD FAIL: Multiple states detected in region: {states_found}")
            print(f"      Claimed region contains {len(states_found)} different US states - HARD FAIL")
            print(f"      This lead will FAIL regardless of LLM verification")
            region_search_results = None
        
        # ANTI-GAMING: Check for multiple major US cities (comma-separated)
        # Pattern: "Los Angeles, Chicago, Houston" - clearly gaming
        # CAREFUL: Don't catch "San Francisco" (one city with space)
        # CAREFUL: Don't catch "New York, NY" (city + state abbreviation)
        MAJOR_US_CITIES = {
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
            'fort worth', 'columbus', 'charlotte', 'san francisco', 'indianapolis',
            'seattle', 'denver', 'washington', 'boston', 'el paso', 'nashville',
            'detroit', 'oklahoma city', 'portland', 'las vegas', 'memphis', 'louisville',
            'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
            'kansas city', 'mesa', 'atlanta', 'omaha', 'colorado springs', 'raleigh',
            'miami', 'cleveland', 'tulsa', 'oakland', 'minneapolis', 'wichita',
            'arlington', 'tampa', 'new orleans', 'bakersfield', 'honolulu', 'anaheim',
            'aurora', 'santa ana', 'st louis', 'riverside', 'pittsburgh', 'cincinnati'
        }
        
        cities_found = set()
        for city in MAJOR_US_CITIES:
            # Use word boundary to avoid "Portland" matching in "Portlandville"
            if re.search(rf'\b{re.escape(city)}\b', claimed_lower):
                cities_found.add(city)
        
        # Only fail if 2+ DIFFERENT major cities found (gaming pattern)
        # This catches: "Los Angeles, Chicago, Houston"
        # But allows: "New York, NY" (only 1 city)
        if len(cities_found) >= 2 and not result.get("region_hard_fail"):
            result["region_verified"] = False
            result["region_hard_fail"] = True
            result["region_confidence"] = 0.0
            result["region_reason"] = f"HARD FAIL: Multiple major cities in claimed region: {cities_found}"
            result["region_extracted"] = "REJECTED - multiple cities detected"
            print(f"   ❌ ANTI-GAMING HARD FAIL: Multiple cities detected in region: {cities_found}")
            print(f"      Claimed region contains {len(cities_found)} different major cities - HARD FAIL")
            print(f"      This lead will FAIL regardless of LLM verification")
            region_search_results = None
    
    # If role_only mode, skip region GSE-based matching and industry checks
    # (Anti-gaming check above still runs for early exit detection)
    if role_only:
        return result
    
    # REGION FUZZY MATCHING
    if region_search_results and claimed_region:
        company_lower = company.lower() if company else ""
        extracted_region = None
        
        # DEBUG: Show what we're trying to extract from
        print(f"   🔍 DEBUG REGION: Attempting to extract location from {len(region_search_results)} search results")
        print(f"      Claimed region: '{claimed_region}'")
        
        for idx, r in enumerate(region_search_results[:5], 1):
            link = r.get("href", r.get("link", ""))
            
            # Filter out LinkedIn posts for region extraction too
            if "/posts/" in link or "/feed/" in link or "/pulse/" in link:
                print(f"      Result {idx}: SKIPPED (LinkedIn post/feed)")
                continue
            
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            combined = title + " " + snippet
            
            # DEBUG: Show what text we're extracting from
            print(f"      Result {idx}: Title: '{title[:80]}{'...' if len(title) > 80 else ''}'")
            print(f"                Snippet: '{snippet[:80]}{'...' if len(snippet) > 80 else ''}'")
            
            if company_lower and company_lower not in combined.lower():
                print(f"                SKIPPED (company '{company}' not in text)")
                continue
            
            loc = extract_location_from_text(combined)
            if loc:
                print(f"                ✅ EXTRACTED: '{loc}'")
                extracted_region = loc
                break
            else:
                print(f"                ❌ NO LOCATION extracted (filters rejected or no location found)")
        
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
                result["region_reason"] = "Could not extract region from ScrapingDog results"
                print(f"   ⚠️ FUZZY REGION: Could not extract location, sending to LLM")
    else:
        if not result.get("region_hard_fail"):
            result["needs_llm"].append("region")
            if not region_search_results and claimed_region:
                print(f"   ℹ️  FUZZY REGION: Skipped (no search results available)")
            elif not claimed_region:
                print(f"   ℹ️  FUZZY REGION: Skipped (no region claimed by miner)")
            else:
                print(f"   ⚠️ FUZZY REGION: No ScrapingDog results")
    
    print(f"   🤖 INDUSTRY: Always verified by LLM (too subjective for fuzzy match)")
    
    return result


# ========================================================================
# Employee Count Verification Functions
# ========================================================================

# LinkedIn employee count ranges (standardized)
LINKEDIN_EMPLOYEE_RANGES = [
    (0, 1, "0-1"),
    (2, 10, "2-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1,000"),
    (1001, 5000, "1,001-5,000"),
    (5001, 10000, "5,001-10,000"),
    (10001, float('inf'), "10,001+"),
]


def parse_employee_count(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse employee count from various text formats.
    
    Returns (min, max) tuple or None if not parseable.
    
    Handles formats like:
    - "2-10 employees"
    - "11-50"
    - "Company size: 51-200 employees"
    - "1,001-5,000"
    - "10001+"
    - "500+"
    - "Self-employed"
    - "50"
    """
    if not text:
        return None
    
    text = text.strip().lower()
    
    # Handle "self-employed"
    if "self-employed" in text or "self employed" in text:
        return (1, 1)
    
    # Remove commas from numbers
    text = text.replace(",", "")
    
    # Handle "10001+" or "500+" format
    plus_match = re.search(r'(\d+)\+', text)
    if plus_match:
        min_val = int(plus_match.group(1))
        return (min_val, 100000)  # Assume large upper bound
    
    # Handle range format: "X-Y" or "X - Y"
    range_match = re.search(r'(\d+)\s*[-–—]\s*(\d+)', text)
    if range_match:
        min_val = int(range_match.group(1))
        max_val = int(range_match.group(2))
        return (min_val, max_val)
    
    # Handle single number
    single_match = re.search(r'(\d+)', text)
    if single_match:
        val = int(single_match.group(1))
        # If it's a single number, treat it as exact
        return (val, val)
    
    return None


def is_valid_employee_count_extraction(extracted: str) -> bool:
    """
    Post-extraction validation to filter out invalid employee counts.
    
    This prevents bugs where regex patterns partially match numbers and extract
    invalid values like "000" from "2000 employees" or years like "2024".
    
    Rejects:
    - "000" or "00" (partial matches from years like 2000)
    - Single years like "2000", "2024" (not employee counts)
    - "0" or values that would parse to 0
    
    Accepts:
    - "2,000" (comma indicates it's a formatted number, not a year)
    - "51-200" (ranges are valid)
    - "500", "5000", "10000+" (counts outside year range or with + suffix)
    """
    if not extracted or not extracted.strip():
        return False
    
    extracted = extracted.strip()
    
    # Remove commas for parsing
    clean = extracted.replace(",", "").replace("+", "").strip()
    
    # Handle ranges like "51-200"
    if "-" in clean or "–" in clean:
        parts = re.split(r'[-–]', clean)
        if len(parts) == 2:
            try:
                min_val = int(parts[0].strip())
                max_val = int(parts[1].strip())
                # Valid if both parts are reasonable employee counts (not zero)
                return min_val > 0 and max_val >= min_val
            except ValueError:
                return False
    
    # Single value
    try:
        val = int(clean)
        
        # Reject 0 or values like "000"
        if val == 0:
            return False
        
        # Reject partial matches like "001" or "001+" (from "10,001+")
        # These are clearly partial extractions when regex doesn't handle commas
        # Real employee counts with "+" are always >= 10,001 (LinkedIn's largest bucket)
        if "+" in extracted and val < 1000:
            return False
        
        # Reject numbers with leading zeros (e.g., "001" parsed as 1)
        # Unless it's a single digit (which is valid like "2" from "2-10")
        if clean.startswith("0") and len(clean) > 1:
            return False
        
        # Reject likely years (1900-2099) UNLESS formatted with comma
        # Employee counts of 1900-2099 are valid only if written as "1,900" or "2,000"
        if 1900 <= val <= 2099:
            # If original has comma (like "2,000"), it's employee count
            if "," in extracted:
                return True
            # If original is 4 digits without comma, likely a year
            if len(clean) == 4:
                return False
        
        return True
    except ValueError:
        return False


def extract_employee_count_from_results(search_results: List[Dict], company: str = "", company_slug: str = "") -> Optional[str]:
    """
    Extract employee count from LinkedIn company page search results.
    
    IMPORTANT: Only extracts from results that match the EXACT company slug to prevent
    false positives from subsidiaries or wrong companies.
    
    Looks for patterns like:
    - "Company size: 2-10 employees"
    - "11-50 employees"
    - "Company size: 51-200"
    
    Args:
        search_results: List of search result dicts with 'title', 'body/snippet', 'href'
        company: Company name for validation
        company_slug: The LinkedIn company slug (e.g., "bp") to verify exact page match
        
    Returns:
        Extracted employee range string (e.g., "11-50") or None
    """
    if not search_results:
        return None
    
    company_lower = company.lower() if company else ""
    
    # Patterns to look for (ordered by specificity)
    patterns = [
        # "Company size: 2-10 employees" or "Company size: 1,001-5,000 employees"
        r'company\s*size[:\s]+(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*employees?',
        # "Company size: 10,001+" or "Company size: 2-10"
        r'company\s*size[:\s]+(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*|\d+\+)',
        # "1,001-5,000 employees" with commas
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*employees?',
        # "2-10 employees" simple range
        r'(\d+[\s,-–—]+\d+)\s*employees?',
        # "10,001+ employees" with comma
        r'(\d{1,3}(?:,\d{3})*\+)\s*employees?',
        # "10001+ employees" or "2-10 employees"
        r'(\d+\+|\d+[\s,-–—]+\d+)\s*employees?',
        # "· 2-10 employees" (after followers on LinkedIn)
        r'·\s*(\d{1,2}[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*employees?',
        # International: German "Mitarbeiter"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*mitarbeiter',
        r'(\d+[\s,-–—]+\d+)\s*mitarbeiter',
        # International: French "employés"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*employés',
        # International: Spanish "empleados"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*empleados',
        # International: Italian "dipendenti"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*dipendenti',
        # International: Dutch "werknemers" / "medewerkers"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*(?:werknemers|medewerkers)',
        # International: Portuguese "funcionários"
        r'(\d{1,3}(?:,\d{3})*[\s,-–—]+\d{1,3}(?:,\d{3})*)\s*funcionários',
        # LinkedIn standard ranges after separator (match exact standard ranges)
        r'·\s*(2[-–]10|11[-–]50|51[-–]200|201[-–]500|501[-–]1,?000|1,?001[-–]5,?000|5,?001[-–]10,?000|10,?001\+)',
    ]
    
    for result in search_results:
        title = result.get("title", "")
        snippet = result.get("body", result.get("snippet", ""))
        href = result.get("href", "").lower()
        
        combined = f"{title} {snippet}".lower()
        
        # Only consider LinkedIn company pages
        if "linkedin.com" not in href:
            continue
        
        # CRITICAL: Verify this result is from the EXACT company slug
        # e.g., for slug "bp", accept "/company/bp/" but NOT "/company/bp-america/"
        if company_slug:
            expected_url_patterns = [
                f'/company/{company_slug}/',  # With trailing slash
                f'/company/{company_slug}?',  # With query params
                f'/company/{company_slug}#',  # With hash
            ]
            
            is_exact_match = False
            if f'/company/{company_slug}' in href:
                for pattern in expected_url_patterns:
                    if pattern in href:
                        is_exact_match = True
                        break
                
                # Also accept if it ends with the slug
                if href.endswith(f'/company/{company_slug}'):
                    is_exact_match = True
            
            if not is_exact_match:
                continue  # Skip - not from exact company page
        
        # Verify it's about the right company (if company name provided)
        if company_lower:
            company_words = [word for word in company_lower.split()[:2] if len(word) > 3]
            company_match = company_lower in combined or any(word in combined for word in company_words)
            if not company_match:
                continue
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Normalize the range format
                extracted = re.sub(r'[\s]+', '', extracted)  # Remove spaces
                extracted = re.sub(r'[–—]', '-', extracted)  # Normalize dashes
                # Validate extraction to prevent bugs like "2000 employees" -> "000"
                if is_valid_employee_count_extraction(extracted):
                    return extracted
                # Invalid extraction - try next pattern
    
    return None


def normalize_to_linkedin_range(min_val: int, max_val: int) -> Optional[str]:
    """
    Normalize a parsed (min, max) range to the standard LinkedIn range string.
    
    LinkedIn has these standard ranges:
    - 0-1, 2-10, 11-50, 51-200, 201-500, 501-1,000, 1,001-5,000, 5,001-10,000, 10,001+
    """
    # Standard LinkedIn ranges with their boundaries
    LINKEDIN_RANGES = [
        ((0, 1), "0-1"),
        ((2, 10), "2-10"),
        ((11, 50), "11-50"),
        ((51, 200), "51-200"),
        ((201, 500), "201-500"),
        ((501, 1000), "501-1,000"),
        ((1001, 5000), "1,001-5,000"),
        ((5001, 10000), "5,001-10,000"),
        ((10001, 100000), "10,001+"),  # 10,001+ uses high upper bound
    ]
    
    # Check if the range falls within a standard LinkedIn range
    for (range_min, range_max), range_str in LINKEDIN_RANGES:
        # For exact match of range boundaries
        if min_val == range_min and (max_val == range_max or (range_str == "10,001+" and max_val >= 10001)):
            return range_str
        # For single values (min == max), check if they fall within a range
        if min_val == max_val:
            if range_min <= min_val <= range_max:
                return range_str
    
    return None


def fuzzy_match_employee_count(claimed: str, extracted: str) -> Tuple[bool, str]:
    """
    STRICT match employee count ranges - requires exact LinkedIn range match.
    
    Args:
        claimed: Miner's claimed employee count (e.g., "51-200")
        extracted: Extracted from company LinkedIn (e.g., "51-200")
    
    Returns:
        (match: bool, reason: str)
    """
    if not claimed or not extracted:
        return False, "Missing data for comparison"
    
    claimed_range = parse_employee_count(claimed)
    extracted_range = parse_employee_count(extracted)
    
    if not claimed_range:
        return False, f"Could not parse claimed employee count: '{claimed}'"
    
    if not extracted_range:
        return False, f"Could not parse extracted employee count: '{extracted}'"
    
    claimed_min, claimed_max = claimed_range
    extracted_min, extracted_max = extracted_range
    
    # Normalize both to standard LinkedIn ranges
    claimed_linkedin = normalize_to_linkedin_range(claimed_min, claimed_max)
    extracted_linkedin = normalize_to_linkedin_range(extracted_min, extracted_max)
    
    if not claimed_linkedin:
        return False, f"Claimed value '{claimed}' doesn't map to standard LinkedIn range"
    
    if not extracted_linkedin:
        return False, f"Extracted value '{extracted}' doesn't map to standard LinkedIn range"
    
    # STRICT: Require same LinkedIn range
    if claimed_linkedin == extracted_linkedin:
        return True, f"LinkedIn range match: '{claimed_linkedin}'"
    
    # No match - different LinkedIn ranges
    return False, f"Different LinkedIn ranges: claimed '{claimed_linkedin}' vs extracted '{extracted_linkedin}'"


def _gse_search_employee_count_sync(company: str, company_linkedin_slug: str = None, max_results: int = 5) -> List[Dict]:
    """
    Search for company employee count on LinkedIn using ScrapingDog.
    
    Uses the miner's provided company LinkedIn URL to ensure we only get data
    from that specific company page, not other sources.
    
    Args:
        company: Company name to search
        company_linkedin_slug: The slug from the miner's company_linkedin URL (e.g., "brivo-inc")
        max_results: Maximum results to return
        
    Returns:
        List of search results
    """
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        print(f"   ⚠️ SCRAPINGDOG_API_KEY not set - skipping employee count search")
        return []
    
    if not company:
        return []
    
    # If we have the company LinkedIn slug, search specifically on that page
    # This ensures we only get data from the miner's provided company LinkedIn
    if company_linkedin_slug:
        queries = [
            f'site:linkedin.com/company/{company_linkedin_slug} company size',  # Primary - includes "company size" for better extraction
            f'site:linkedin.com/company/{company_linkedin_slug} employees',  # Fallback 1
            # IMPORTANT: For smaller companies, the site: restriction may not return employee count
            # in the snippet. This broader query returns better metadata. The URL validation in
            # extract_employee_count_from_results ensures we ONLY extract from the exact company slug,
            # preventing false positives from other companies with similar names.
            f'"{company}" linkedin company size employees',  # Fallback 2 - broader search
        ]
    else:
        # Fallback to generic search if no slug provided (shouldn't happen)
        queries = [
            f'{company} linkedin company size',
            f'"{company}" linkedin employees',
        ]
    
    for query in queries:
        print(f"   🔍 GSE Employee Count: {query}")
        
        try:
            url = "https://api.scrapingdog.com/google"
            params = {
                "api_key": api_key,
                "query": query,
                "results": max_results
            }
            
            response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)
            
            if response.status_code == 200:
                data = response.json()
                query_results = []
                
                for item in data.get("organic_results", []):
                    result = {
                        "title": item.get("title", ""),
                        "href": item.get("link", ""),
                        "body": item.get("snippet", "")
                    }
                    query_results.append(result)
                
                if query_results:
                    # Try extraction immediately - only return if we find the RIGHT company's data
                    extracted = extract_employee_count_from_results(query_results, company, company_linkedin_slug)
                    if extracted:
                        print(f"   ✅ Found employee count: {extracted}")
                        return query_results
                    else:
                        print(f"   ⚠️ Query returned results but couldn't extract for '{company}' - trying next query...")
            else:
                print(f"   ⚠️ ScrapingDog API error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Employee count search failed: {e}")
    
    print(f"   ❌ All queries exhausted - could not find employee count for '{company}'")
    return []


async def _gse_search_employee_count(company: str, company_linkedin_slug: str = None, max_results: int = 3) -> List[Dict]:
    """Async wrapper for employee count search."""
    try:
        return await asyncio.to_thread(
            _gse_search_employee_count_sync,
            company,
            company_linkedin_slug,
            max_results
        )
    except Exception as e:
        print(f"   ⚠️ Employee count search thread failed: {e}")
        return []


def _gse_search_stage5_sync(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """
    Stage 5 search helper using ScrapingDog GSE.
    MODIFIED: Replaced ScrapingDog with ScrapingDog for consistency.
    """
    
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return []
    
    if search_type == "role":
        linkedin_url = kwargs.get("linkedin_url", "")
        
        # MANDATE: Role verification MUST use the miner's LinkedIn URL
        # This ensures we extract role from the SPECIFIC person's profile.
        # We search multiple queries for different snippets, but ONLY use results
        # where the URL matches the miner's LinkedIn profile URL.
        # NO FALLBACK to name+company searches.
        
        role_simplified = re.split(r'[,&/]', role)[0].strip() if role else ""
        
        queries = []
        fallback_queries = []  # Empty - no fallback allowed
        
        # Extract profile slug for URL matching
        profile_slug = None
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower()
        
        if linkedin_url and "linkedin.com/in/" in linkedin_url and profile_slug:
            # All queries search for different aspects, but we ONLY use results
            # where the URL contains the miner's LinkedIn profile slug
            
            # Query 1: LinkedIn URL alone (returns profile with role in title)
            queries.append(f'"{linkedin_url}"')
            
            # Query 2: LinkedIn URL + company (focuses on work history)
            queries.append(f'"{linkedin_url}" "{company}"')
            
            # Query 3: LinkedIn URL + claimed role (verification mode)
            if role_simplified:
                queries.append(f'"{linkedin_url}" "{role_simplified}"')
            
            # Query 4: Name + role + company (may return profile in results)
            # We still search this but ONLY use results matching the LinkedIn URL
            if role_simplified:
                queries.append(f'"{full_name}" "{role_simplified}" "{company}"')
            
            # Query 5: site: operator + claimed role (BEST for extracting role from Experience section)
            # Uses site: to search INSIDE the profile page content, returns snippet with
            # full Experience section like: "Paralegal. Brewe Layman. Aug 2005 - Present"
            if role_simplified:
                queries.append(f'site:linkedin.com/in/{profile_slug} "{role_simplified}"')
            
            # NO FALLBACK - only use results matching the miner's LinkedIn URL
        else:
            # NO LinkedIn URL provided - FAIL role verification
            # Stage 4 requires LinkedIn URL, so this should never happen
            print(f"   ❌ No valid LinkedIn URL for role search - cannot verify role")
            return []  # Return empty - role verification will fail
    elif search_type == "person_location":
        # Search for PERSON's location using their LinkedIn URL
        # This is called ONLY when Stage 4 didn't extract location from its search results.
        # Stage 4 already searches for the profile - we just need targeted location queries.
        #
        # Query 1: "{linkedin_url}" location
        # Example result: "Location: New York City Metropolitan Area · 500+ connections"
        #
        # Query 2: "{linkedin_url}" {claimed_region}
        # Example: "{linkedin_url}" United States, New York, Manhattan
        #
        # Query 3: "{linkedin_url}" "{role}" (forces experience section with location)
        # Example: "{linkedin_url}" "Technical Director" → returns work history with location
        linkedin_url = kwargs.get("linkedin_url", "")
        region_hint = kwargs.get("region_hint", "")  # country, state, city combined
        role = kwargs.get("role", "")
        
        queries = []
        fallback_queries = []
        
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            # Primary: LinkedIn URL + "location"
            queries.append(f'"{linkedin_url}" location')
            
            # Secondary: LinkedIn URL + miner's claimed region (country, state, city)
            if region_hint:
                queries.append(f'"{linkedin_url}" {region_hint}')
            
            # Tertiary: LinkedIn URL + role (forces Google to return experience section with location)
            if role:
                queries.append(f'"{linkedin_url}" "{role}"')
    elif search_type == "industry":
        region_hint = kwargs.get("region_hint", "")
        if region_hint:
            queries = [f'{company} {region_hint} company industry', f'{company} company industry {region_hint}']
        else:
            queries = [f'{company} company industry']
        fallback_queries = []
    
    def gse_search_with_fallback(query, max_results, company_name=None):
        """GSE search with company verification"""
        try:
            url = "https://api.scrapingdog.com/google"
            params = {
                "api_key": api_key,
                "query": query,
                "results": max_results
            }
            
            response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Convert to standard format
                for item in data.get("organic_results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "href": item.get("link", ""),
                        "body": item.get("snippet", "")
                    })
                
                # For industry: verify company mentioned
                if company_name and search_type == "industry":
                    company_normalized = re.sub(r'\s*-\s*', '-', company_name.lower())
                    company_words = [w for w in company_normalized.split() if len(w) > 3][:2]
                    
                    for r in results:
                        text = f"{r.get('title', '')} {r.get('body', '')}".lower()
                        if company_name.lower() in text or any(word in text for word in company_words):
                            return results  # Company mentioned, good results
                    
                    return []  # Company not mentioned
                
                return results
        except Exception:
            return []
        
        return []
    
    # Try primary queries first
    all_results = []
    role_found = False
    
    # For role searches: extract profile slug for URL matching
    role_profile_slug = None
    if search_type == "role":
        linkedin_url = kwargs.get("linkedin_url", "")
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            role_profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower()
    
    for query in queries:
        results = gse_search_with_fallback(query, max_results, company_name=company if search_type != "role" else None)
        if results:
            # For role searches: ONLY keep results matching the miner's LinkedIn profile URL
            if search_type == "role" and role_profile_slug:
                matching_results = []
                for r in results:
                    result_url = r.get("href", "").lower()
                    # Check if the result URL contains the miner's profile slug
                    if f"/in/{role_profile_slug}" in result_url:
                        matching_results.append(r)
                        print(f"   ✅ URL MATCH: {result_url[:60]}...")
                    else:
                        # Log non-matching URLs (for debugging)
                        pass  # Don't spam logs with non-matches
                
                if matching_results:
                    all_results.extend(matching_results)
                    # Check if we found a role from matching results
                    for r in matching_results:
                        title = r.get("title", "")
                        snippet = r.get("body", "")
                        extracted = extract_role_from_search_title(title, snippet, company_name=company, full_name=full_name)
                        if extracted and len(extracted) > 3:
                            role_found = True
                            break
            elif search_type == "person_location":
                # For person_location: only return if we actually extracted a location
                # Otherwise continue to next query (e.g., Query 2 or 3)
                all_results.extend(results)
                location_found = False
                for r in results:
                    snippet = r.get("body", r.get("snippet", ""))
                    title = r.get("title", "")
                    combined = title + " " + snippet
                    extracted_loc = extract_person_location_from_linkedin_snippet(combined)
                    if extracted_loc:
                        location_found = True
                        print(f"   ✅ Found location in query: '{extracted_loc}'")
                        break
                if location_found:
                    return results
                else:
                    print(f"   ⚠️ Query returned results but no location extracted - trying next query...")
            else:
                # Industry or other search types
                all_results.extend(results)
                return results
    
    # For role searches: NO FALLBACK - we only use results matching the miner's LinkedIn URL
    # All matching results are already collected in all_results
    if search_type == "role":
        if all_results:
            print(f"   📊 Role search: Found {len(all_results)} result(s) matching miner's LinkedIn URL")
        else:
            print(f"   ⚠️ Role search: No results matching miner's LinkedIn URL found")
            print(f"   ⚠️ All search results had different URLs - cannot verify role from different profiles")
        # Return what we have (may be empty if no URL matches)
        return all_results
    
    # For non-role searches: return collected results or try fallback queries
    if all_results:
        return all_results
    
    # Try fallback queries for non-role searches
    for query in fallback_queries:
        results = gse_search_with_fallback(query, max_results)
        if results:
            return results
    
    return []


async def _gse_search_stage5(
    search_type: str,
    full_name: str = "",
    company: str = "",
    role: str = "",
    max_results: int = 5,
    **kwargs
) -> List[Dict]:
    """Async wrapper for Stage 5 ScrapingDog search."""
    try:
        return await asyncio.to_thread(
            _gse_search_stage5_sync,
            search_type,
            full_name,
            company,
            role,
            max_results,
            **kwargs
        )
    except Exception as e:
        print(f"⚠️ ScrapingDog {search_type} search thread failed: {e}")
        return []


# ========================================================================
# COMPANY LINKEDIN VERIFICATION
# ========================================================================
# Validates company_linkedin URL, scrapes company data, and uses it to verify
# industry, sub_industry, description, and employee count.
# ========================================================================

def validate_company_linkedin_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that a URL is a valid LinkedIn company page (not a profile page).
    
    Args:
        url: The company_linkedin URL to validate
        
    Returns:
        (is_valid, reason, company_slug)
        - is_valid: True if URL is a valid company page
        - reason: Description of why validation passed/failed
        - company_slug: Extracted company slug (e.g., "microsoft" from linkedin.com/company/microsoft)
    """
    if not url or not url.strip():
        return False, "No company_linkedin URL provided", None
    
    url = url.strip().lower()
    
    # Must contain linkedin.com
    if "linkedin.com" not in url:
        return False, "URL is not a LinkedIn URL", None
    
    # Must be a company page, NOT a profile page
    if "/in/" in url:
        return False, "URL is a personal profile (/in/), not a company page (/company/)", None
    
    # Must contain /company/
    if "/company/" not in url:
        return False, "URL is not a company page (missing /company/)", None
    
    # Extract company slug
    try:
        # Handle various formats:
        # - linkedin.com/company/microsoft
        # - linkedin.com/company/microsoft/
        # - linkedin.com/company/microsoft/about
        # - https://www.linkedin.com/company/microsoft?param=value
        parts = url.split("/company/")
        if len(parts) < 2:
            return False, "Could not extract company slug from URL", None
        
        slug_part = parts[1]
        # Remove trailing slashes and query params
        slug = slug_part.split("/")[0].split("?")[0].strip()
        
        if not slug or len(slug) < 2:
            return False, "Company slug is too short or empty", None
        
        return True, f"Valid company page: /company/{slug}", slug
        
    except Exception as e:
        return False, f"Error parsing URL: {str(e)}", None


def _scrape_company_linkedin_gse_sync(company_slug: str, company_name: str, max_results: int = 3) -> Dict:
    """
    Scrape company LinkedIn page data using ScrapingDog GSE.
    
    Uses site:linkedin.com/company/{slug} to get company page data from search results.
    
    Args:
        company_slug: The company slug from the LinkedIn URL
        company_name: The company name claimed by the miner (for verification)
        max_results: Max results to fetch
        
    Returns:
        Dict with:
        - success: bool
        - company_name_from_linkedin: str (extracted company name)
        - company_name_match: bool (does it match miner's company?)
        - industry: str (if found)
        - description: str (if found)
        - employee_count: str (if found, e.g., "1,001-5,000 employees")
        - location: str (if found)
        - raw_results: list (original search results)
        - error: str (if any)
    """
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "SCRAPINGDOG_API_KEY not set",
            "raw_results": []
        }
    
    result = {
        "success": False,
        "company_name_from_linkedin": None,
        "company_name_match": False,
        "industry": None,
        "description": None,
        "employee_count": None,
        "location": None,
        "raw_results": [],
        "error": None
    }
    
    # Search for the company LinkedIn page
    query = f'site:linkedin.com/company/{company_slug}'
    
    try:
        url = "https://api.scrapingdog.com/google"
        params = {
            "api_key": api_key,
            "query": query,
            "results": max_results
        }
        
        print(f"   🔍 COMPANY LINKEDIN: Searching for {query}")
        
        response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG)
        
        if response.status_code != 200:
            result["error"] = f"GSE API returned status {response.status_code}"
            return result
        
        data = response.json()
        organic_results = data.get("organic_results", [])
        
        if not organic_results:
            result["error"] = "No search results found for company LinkedIn page"
            return result
        
        # Store raw results
        result["raw_results"] = [
            {
                "title": r.get("title", ""),
                "href": r.get("link", ""),
                "snippet": r.get("snippet", "")
            }
            for r in organic_results
        ]
        
        # Extract data from the first result (main company page)
        first_result = organic_results[0]
        title = first_result.get("title", "")
        snippet = first_result.get("snippet", "")
        link = first_result.get("link", "")
        
        # CRITICAL: Verify the URL matches the exact slug provided by the miner
        # This prevents accepting similar company names from different LinkedIn pages
        link_lower = link.lower()
        expected_url_patterns = [
            f'/company/{company_slug}/',  # With trailing slash
            f'/company/{company_slug}?',  # With query params
            f'/company/{company_slug}#',  # With hash
        ]
        
        url_matches = False
        if f'/company/{company_slug}' in link_lower:
            # Check if it's an exact match (not a longer slug that contains our slug)
            for pattern in expected_url_patterns:
                if pattern in link_lower:
                    url_matches = True
                    break
            # Also accept if URL ends with the slug
            if link_lower.endswith(f'/company/{company_slug}'):
                url_matches = True
        
        if not url_matches:
            result["error"] = f"Search result URL '{link}' does not match expected slug '/company/{company_slug}'"
            result["company_name_match"] = False
            print(f"   ❌ COMPANY LINKEDIN: URL mismatch - Expected /company/{company_slug}, got {link}")
            return result
        
        print(f"   ✅ COMPANY LINKEDIN: URL verified - {link}")
        
        # Combine all text for extraction
        all_text = f"{title} {snippet}"
        for r in organic_results[1:]:
            all_text += f" {r.get('snippet', '')}"
        
        # Extract company name from title
        # Format: "Company Name | LinkedIn" or "Company Name - LinkedIn" or "Company Name: Overview | LinkedIn"
        company_name_from_linkedin = None
        
        if "|" in title:
            company_name_from_linkedin = title.split("|")[0].strip()
        elif " - LinkedIn" in title:
            # Handle "Company Name - LinkedIn" format
            company_name_from_linkedin = title.replace(" - LinkedIn", "").strip()
        elif " LinkedIn" in title and title.endswith("LinkedIn"):
            # Handle "Company Name LinkedIn" format (no separator)
            company_name_from_linkedin = title.replace(" LinkedIn", "").strip()
        
        if company_name_from_linkedin:
            # Remove "Overview", "About", etc.
            for suffix in [": Overview", " - Overview", ": About", " - About", ": Jobs", " - Jobs"]:
                if suffix in company_name_from_linkedin:
                    company_name_from_linkedin = company_name_from_linkedin.replace(suffix, "").strip()
            result["company_name_from_linkedin"] = company_name_from_linkedin
        
        # Verify company name matches
        if result["company_name_from_linkedin"] and company_name:
            linkedin_name = result["company_name_from_linkedin"].lower().strip()
            claimed_name = company_name.lower().strip()
            
            # Direct match or one contains the other
            if linkedin_name == claimed_name:
                result["company_name_match"] = True
            elif linkedin_name in claimed_name or claimed_name in linkedin_name:
                result["company_name_match"] = True
            else:
                # Try fuzzy matching - extract key words
                linkedin_words = set(re.sub(r'[^\w\s]', '', linkedin_name).split())
                claimed_words = set(re.sub(r'[^\w\s]', '', claimed_name).split())
                # Remove common words
                common_words = {'inc', 'llc', 'corp', 'corporation', 'company', 'co', 'ltd', 'limited', 'the', 'group'}
                linkedin_words -= common_words
                claimed_words -= common_words
                
                if linkedin_words and claimed_words:
                    # Check if main words overlap
                    overlap = linkedin_words & claimed_words
                    if overlap and len(overlap) >= min(len(linkedin_words), len(claimed_words)) * 0.5:
                        result["company_name_match"] = True
            
            # ADDITIONAL CHECK: If name still doesn't match, check if claimed name appears 
            # anywhere in the company LinkedIn snippet/description (handles abbreviations like OWI Inc. = Old World Industries)
            if not result["company_name_match"]:
                all_text_lower = all_text.lower()
                # Check if the full claimed name appears in the snippet
                if claimed_name in all_text_lower:
                    result["company_name_match"] = True
                    result["company_name_match_source"] = "snippet_contains_claimed_name"
                else:
                    # Check if key words from claimed name appear together in snippet
                    claimed_key_words = claimed_words  # Already computed above, minus common words
                    if claimed_key_words and len(claimed_key_words) >= 2:
                        # Check if at least 2 key words from claimed name appear in snippet
                        words_found = sum(1 for w in claimed_key_words if w in all_text_lower)
                        if words_found >= min(2, len(claimed_key_words)):
                            result["company_name_match"] = True
                            result["company_name_match_source"] = "snippet_contains_key_words"
        
        # Extract employee count - ONLY from results that match the EXACT company slug
        # This prevents extracting employee counts from subsidiaries or wrong companies
        employee_patterns = [
            # English patterns
            r'company\s+size[:\s]+(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*employees',  # "Company size: X employees"
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*employees',  # "X employees" or "X-Y employees"
            r'(\d+(?:,\d{3})*\+?)\s+employees',  # "X+ employees"
            r'employees[:\s]+(\d{1,3}(?:,\d{3})*(?:\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)',  # "employees: X"
            r'·\s*(\d{1,2}[-–]\d{1,3}(?:,\d{3})*)\s*employees',  # "· 2-10 employees" (after followers)
            r'(\d{1,2}\s*(?:to|bis|à|a)\s*\d{1,3})\s*employees',  # "2 to 10 employees"
            # International patterns (German, French, Spanish, Italian, Dutch, Portuguese)
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*mitarbeiter',  # German
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*employés',  # French
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*empleados',  # Spanish
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*dipendenti',  # Italian
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*werknemers',  # Dutch
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*funcionários',  # Portuguese
            r'(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*medewerkers',  # Dutch alt
            # LinkedIn standard ranges - only match exact standard ranges for safety
            r'·\s*(2[-–]10|11[-–]50|51[-–]200|201[-–]500|501[-–]1,?000|1,?001[-–]5,?000|5,?001[-–]10,?000|10,?001\+)',
        ]
        
        # Check each result individually to ensure we're getting data from the CORRECT company page
        for r in organic_results:
            result_link = r.get("link", "").lower()
            result_snippet = r.get("snippet", "")
            
            # CRITICAL: Only extract employee count if this result is from the EXACT company slug
            # e.g., for slug "bp", accept "/company/bp" or "/company/bp/" but NOT "/company/bp-america"
            expected_url_patterns = [
                f'/company/{company_slug}/',  # With trailing slash
                f'/company/{company_slug}?',  # With query params
                f'/company/{company_slug}#',  # With hash
            ]
            
            # Check if this result is from the exact company page (not a subsidiary)
            is_exact_match = False
            if f'/company/{company_slug}' in result_link:
                # Check it's not a longer slug (e.g., "bp-america" when we want "bp")
                for pattern in expected_url_patterns:
                    if pattern in result_link:
                        is_exact_match = True
                        break
                
                # Also accept if it ends with the slug (e.g., "linkedin.com/company/bp")
                if result_link.endswith(f'/company/{company_slug}'):
                    is_exact_match = True
            
            if not is_exact_match:
                continue  # Skip this result - not from the exact company page
            
            # Try to extract employee count from THIS result's snippet
            for pattern in employee_patterns:
                match = re.search(pattern, result_snippet.lower())
                if match:
                    extracted = match.group(1).strip()
                    # Validate extraction to prevent bugs like "2000 employees" -> "000"
                    if is_valid_employee_count_extraction(extracted):
                        result["employee_count"] = extracted
                        break
                    else:
                        print(f"      ⚠️ Rejected invalid employee count extraction: '{extracted}'")
            
            if result["employee_count"]:
                break  # Found employee count from correct company page
        
        # Extract industry from snippet
        # Often appears after company name or in description
        industry_patterns = [
            r'(?:industry|sector|in the)\s*[:\s]*([A-Z][a-zA-Z\s&]+?)(?:\.|,|\||employees|founded|location)',
            r'\|\s*([A-Z][a-zA-Z\s&]+?)\s*\|',
        ]
        for pattern in industry_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                potential_industry = match.group(1).strip()
                # Filter out non-industry text
                if len(potential_industry) < 50 and potential_industry.lower() not in ['linkedin', 'overview', 'about']:
                    result["industry"] = potential_industry
                    break
        
        # Extract location/headquarters
        location_patterns = [
            r'(?:headquarters|headquartered|based|located)\s*(?:in|at)?\s*[:\s]*([A-Z][a-zA-Z\s,]+?)(?:\.|,|\||employees)',
            r'([A-Z][a-z]+(?:,\s*[A-Z]{2})?)\s*(?:area|region|metropolitan)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, all_text)
            if match:
                result["location"] = match.group(1).strip()
                break
        
        # Extract description - look through ALL results for the best company description
        # Filter out job postings, non-English content, and updates
        best_description = None
        best_score = 0
        
        for r in organic_results:
            candidate = r.get("snippet", "").strip()
            if not candidate or len(candidate) < 30:
                continue
            
            # Skip job postings and updates
            job_posting_indicators = [
                "i'm hiring", "we're hiring", "looking for", "job opening",
                "big news", "my team is growing", "join us", "apply now",
                "we are looking", "open position", "career opportunity"
            ]
            if any(indicator in candidate.lower() for indicator in job_posting_indicators):
                continue
            
            # Skip non-English content (check for common non-ASCII patterns)
            non_english_indicators = [
                "·", "sobre nós", "sobre nosotros", "о нас", "über uns",
                "עוקבים", "網站", "会社概要", "관하여"
            ]
            if any(indicator in candidate.lower() for indicator in non_english_indicators):
                continue
            
            # Score the snippet - prefer ones that describe the company
            score = 0
            description_patterns = [
                r'\bis the\b', r'\bis a\b', r'\bprovides\b', r'\boffers\b',
                r'\bspecializes\b', r'\bleader in\b', r'\bfocuses on\b',
                r'\bhelps\b', r'\benables\b', r'\bpowers\b', r'\bbuilds\b'
            ]
            for pattern in description_patterns:
                if re.search(pattern, candidate, re.IGNORECASE):
                    score += 10
            
            # Prefer longer, more descriptive snippets
            score += min(len(candidate) / 20, 10)
            
            if score > best_score:
                best_score = score
                best_description = candidate
        
        # Fallback to first snippet if no good description found
        if not best_description and snippet:
            best_description = snippet
        
        if best_description:
            # Clean up the description
            description = best_description.strip()
            description = re.sub(r'\d{1,3}(?:,\d{3})*(?:\+|\s*-\s*\d{1,3}(?:,\d{3})*)?\s*employees', '', description)
            description = re.sub(r'\s+', ' ', description).strip()
            if len(description) > 20:
                result["description"] = description
        
        result["success"] = True
        return result
        
    except Exception as e:
        result["error"] = f"Exception during scraping: {str(e)}"
        return result


async def scrape_company_linkedin_gse(company_slug: str, company_name: str, max_results: int = 3) -> Dict:
    """Async wrapper for company LinkedIn GSE scraping."""
    try:
        return await asyncio.to_thread(
            _scrape_company_linkedin_gse_sync,
            company_slug,
            company_name,
            max_results
        )
    except Exception as e:
        print(f"⚠️ Company LinkedIn scraping thread failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "raw_results": []
        }


def verify_company_linkedin_data(
    scraped_data: Dict,
    claimed_company: str,
    claimed_industry: str,
    claimed_sub_industry: str,
    claimed_description: str,
    claimed_employee_count: str,
    sub_industry_definition: str = ""
) -> Dict:
    """
    Verify miner's claims against scraped company LinkedIn data.
    
    Returns:
        Dict with verification results for each field
    """
    result = {
        "company_name_verified": False,
        "company_name_reason": "",
        "has_useful_data": False,
        "industry_from_linkedin": None,
        "description_from_linkedin": None,
        "employee_count_from_linkedin": None,
        "location_from_linkedin": None,
    }
    
    if not scraped_data.get("success"):
        result["company_name_reason"] = scraped_data.get("error", "Scraping failed")
        return result
    
    # Verify company name
    result["company_name_verified"] = scraped_data.get("company_name_match", False)
    linkedin_company = scraped_data.get("company_name_from_linkedin", "")
    
    if result["company_name_verified"]:
        result["company_name_reason"] = f"Company name matches: '{linkedin_company}' ≈ '{claimed_company}'"
    else:
        result["company_name_reason"] = f"Company name mismatch: LinkedIn shows '{linkedin_company}' but miner claimed '{claimed_company}'"
    
    # Store extracted data for verification
    if scraped_data.get("industry"):
        result["industry_from_linkedin"] = scraped_data["industry"]
        result["has_useful_data"] = True
    
    if scraped_data.get("description"):
        result["description_from_linkedin"] = scraped_data["description"]
        result["has_useful_data"] = True
    
    if scraped_data.get("employee_count"):
        result["employee_count_from_linkedin"] = scraped_data["employee_count"]
        result["has_useful_data"] = True
    
    if scraped_data.get("location"):
        result["location_from_linkedin"] = scraped_data["location"]
        result["has_useful_data"] = True
    
    return result


async def check_stage5_unified(lead: dict) -> Tuple[bool, dict]:
    """
    Stage 5: Unified verification of role, region, employee count, and industry.
    
    Uses ScrapingDog searches + fuzzy matching + LLM verification.
    Called AFTER Stage 4 LinkedIn verification passes.
    
    Order of checks: Role → Region → Employee Count → Industry
    
    Returns:
        (passed: bool, rejection_reason: dict or None)
    """
    full_name = get_field(lead, "full_name") or ""
    company = get_company(lead) or ""
    claimed_role = get_role(lead) or ""
    
    # Build claimed_region from country/state/city fields (new format from gateway)
    # Falls back to legacy "region" field for backward compatibility
    country = lead.get("country", "").strip()
    state = lead.get("state", "").strip()
    city = lead.get("city", "").strip()
    
    if country:
        # New format: Build region from components (no trailing commas)
        region_parts = [p for p in [country, state, city] if p]
        claimed_region = ", ".join(region_parts)
    else:
        # Fallback: Use legacy region field
        claimed_region = get_location(lead) or ""
    
    claimed_industry = get_industry(lead) or ""
    claimed_sub_industry = lead.get("sub_industry", "") or lead.get("Sub_industry", "") or ""
    claimed_employee_count = get_employee_count(lead) or ""
    linkedin_url = get_linkedin(lead) or ""
    website = get_website(lead) or ""
    claimed_description = lead.get("description", "") or ""
    company_linkedin = lead.get("company_linkedin", "") or ""
    
    if not company:
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": "No company name provided",
            "failed_fields": ["company"]
        }
    
    # ========================================================================
    # ROLE FORMAT VALIDATION (ANTI-GAMING)
    # ========================================================================
    # Check role format BEFORE any content matching to catch stuffed/malformed roles:
    # - Person's name in role (e.g., "Jones - Associate Director")
    # - Company name in role (e.g., "CEO at CloudFactory")
    # - Marketing taglines (e.g., "CEO. Unlocking the potential of AI...")
    # - Geographic locations at end (e.g., "VP Sales - Vietnam, Cambodia")
    # - Excessively long roles (> 80 chars)
    # ========================================================================
    
    if claimed_role:
        role_format_valid, role_format_reason = validate_role_format(claimed_role, full_name, company)
        if not role_format_valid:
            print(f"   ❌ ROLE FORMAT INVALID: {role_format_reason}")
            return False, {
                "stage": "Stage 5: Role Format",
                "check_name": "check_stage5_unified",
                "message": f"Role format invalid: {role_format_reason}",
                "failed_fields": ["role"],
                "claimed_role": claimed_role,
                "anti_gaming": "role_format"
            }
        print(f"   ✅ ROLE FORMAT: Valid format for '{claimed_role}'")
    
    # ========================================================================
    # INDUSTRY TAXONOMY VALIDATION (EXACT MATCH REQUIRED)
    # ========================================================================
    # Miners must submit industry and sub_industry that EXACTLY match industry taxonomy.
    # This happens BEFORE any LLM verification to fail fast on invalid submissions.
    # ========================================================================
    
    print(f"   🔍 TAXONOMY VALIDATION: Checking exact matches...")
    
    # Step 1: Validate industry is an exact match to valid industry
    industry_valid, industry_reason, matched_industry = validate_exact_industry_match(claimed_industry)
    if not industry_valid:
        print(f"   ❌ INDUSTRY EXACT MATCH FAILED: {industry_reason}")
        return False, {
            "stage": "Stage 5: Industry Taxonomy",
            "check_name": "check_stage5_unified",
            "message": f"Industry '{claimed_industry}' is not a valid industry. Must be exact match.",
            "failed_fields": ["industry"],
            "valid_industries": sorted(get_all_valid_industries())
        }
    print(f"   ✅ INDUSTRY: '{matched_industry}' is valid")
    
    # Step 2: Validate sub_industry is an exact match to valid sub-industry
    sub_industry_valid, sub_industry_reason, matched_sub_industry, taxonomy_entry = validate_exact_sub_industry_match(claimed_sub_industry)
    if not sub_industry_valid:
        print(f"   ❌ SUB-INDUSTRY EXACT MATCH FAILED: {sub_industry_reason}")
        return False, {
            "stage": "Stage 5: Industry Taxonomy",
            "check_name": "check_stage5_unified",
            "message": f"Sub-industry '{claimed_sub_industry}' is not a valid sub-industry. Must be exact match.",
            "failed_fields": ["sub_industry"]
        }
    print(f"   ✅ SUB-INDUSTRY: '{matched_sub_industry}' is valid")
    
    # Step 3: Validate industry ↔ sub_industry pairing
    pairing_valid, pairing_reason = validate_industry_sub_industry_exact_pairing(matched_industry, matched_sub_industry)
    if not pairing_valid:
        print(f"   ❌ INDUSTRY/SUB-INDUSTRY PAIRING FAILED: {pairing_reason}")
        valid_groups = taxonomy_entry.get("industries", []) if taxonomy_entry else []
        return False, {
            "stage": "Stage 5: Industry Taxonomy",
            "check_name": "check_stage5_unified",
            "message": f"Industry '{matched_industry}' is not valid for sub-industry '{matched_sub_industry}'. {pairing_reason}",
            "failed_fields": ["industry", "sub_industry"],
            "valid_industries_for_sub_industry": valid_groups
        }
    print(f"   ✅ PAIRING: '{matched_industry}' is valid for '{matched_sub_industry}'")
    
    # Store taxonomy validation results
    lead["taxonomy_industry_valid"] = True
    lead["taxonomy_matched_industry"] = matched_industry
    lead["taxonomy_sub_industry_valid"] = True
    lead["taxonomy_matched_sub_industry"] = matched_sub_industry
    lead["taxonomy_pairing_valid"] = True
    sub_industry_definition = taxonomy_entry.get("definition", "") if taxonomy_entry else ""
    
    # ========================================================================
    # COMPANY LINKEDIN DATA (FROM STAGE 4 CACHE)
    # ========================================================================
    # Stage 4 already validated company_linkedin URL, verified company name,
    # and cached the data. We just retrieve it here and determine what
    # additional GSE queries (if any) are needed.
    # ========================================================================
    
    # Get cached company LinkedIn data from Stage 4
    company_linkedin_data = lead.get("company_linkedin_data")
    company_linkedin_verified = lead.get("company_linkedin_verified", False)
    company_linkedin_from_cache = lead.get("company_linkedin_from_cache", False)
    
    # Determine what data is available from company LinkedIn
    has_industry_description = False
    has_employee_count = False
    use_company_linkedin_for_verification = False
    
    if company_linkedin_data:
        has_industry_description = bool(
            company_linkedin_data.get("industry") or 
            company_linkedin_data.get("description")
        )
        has_employee_count = bool(company_linkedin_data.get("employee_count"))
        use_company_linkedin_for_verification = has_industry_description or has_employee_count
        
        cache_status = "from global cache" if company_linkedin_from_cache else "freshly scraped"
        print(f"   📦 COMPANY LINKEDIN DATA ({cache_status}):")
        print(f"      Has industry/description: {has_industry_description}")
        print(f"      Has employee count: {has_employee_count}")
    else:
        print(f"   ⚠️ COMPANY LINKEDIN: No data available from Stage 4 - will use fallback GSE searches")
    
    # PRIORITY: Check if Stage 4 extracted a role from the confirmed LinkedIn profile
    stage4_role = lead.get("stage4_extracted_role")
    role_verified_by_stage4 = False
    
    if stage4_role and claimed_role:
        print(f"   📝 Stage 4 provided role: '{stage4_role}'")
        print(f"   📝 Miner claimed role: '{claimed_role}'")
        
        # Try fuzzy matching Stage 4's role against miner's claimed role
        match, confidence, reason = fuzzy_match_role(claimed_role, stage4_role)
        
        if match:
            print(f"   ✅ ROLE VERIFIED by Stage 4 profile: '{stage4_role}' ≈ '{claimed_role}'")
            print(f"      Confidence: {int(confidence*100)}% | Reason: {reason}")
            role_verified_by_stage4 = True
        else:
            print(f"   ⚠️ Stage 4 role mismatch: '{stage4_role}' ≠ '{claimed_role}' (Confidence: {int(confidence*100)}%)")
            print(f"   🔍 Falling back to Stage 5 GSE searches for independent verification...")
    
    # No delay needed between Stage 4 and Stage 5 (ScrapingDog GSE has no rate limiting)
    
    # STEP 1: GSE SEARCH FOR ROLE (only if Stage 4 didn't verify it)
    role_results = []
    if not role_verified_by_stage4:
        print(f"   🔍 GSE: Searching for {full_name}'s role at {company}...")
        role_results = await _gse_search_stage5("role", full_name, company, claimed_role, linkedin_url=linkedin_url)
        if role_results:
            print(f"   ✅ Found {len(role_results)} role search results")
        else:
            print(f"   ⚠️ No role results found")
    
    # EARLY EXIT CHECK: Do quick role + region anti-gaming check BEFORE region/industry GSE searches
    # This saves 6+ seconds and 2 GSE API calls when role is definitively wrong OR region is gaming
    # Skip this if role was already verified by Stage 4
    if not role_verified_by_stage4:
        print(f"   🔍 QUICK CHECK: Verifying role and region anti-gaming before continuing...")
        quick_result = fuzzy_pre_verification_stage5(
            claimed_role=claimed_role,
            claimed_region=claimed_region,  # Pass real region for anti-gaming check
            claimed_industry="",  # Skip industry check
            role_search_results=role_results,
            region_search_results=[],  # Empty - just checking anti-gaming on claimed_region string
            industry_search_results=[],  # Empty - not checking yet
            full_name=full_name,
            company=company,
            role_only=True  # Skip GSE-based region/industry matching, but anti-gaming still runs
        )
        
        # EARLY EXIT: Role definitively failed - skip region/industry GSE searches entirely
        if quick_result.get("role_definitive_fail"):
            print(f"   ❌ EARLY EXIT: Role check failed - SKIPPING region and industry GSE searches")
            return False, {
                "stage": "Stage 5: Role/Region/Industry",
                "check_name": "check_stage5_unified",
                "message": f"Role FAILED: Found '{quick_result.get('role_extracted')}' but miner claimed '{claimed_role}'",
                "failed_fields": ["role"],
                "early_exit": "role_failed_before_region_industry",
                "extracted_role": quick_result.get("role_extracted"),
                "claimed_role": claimed_role,
                "gse_searches_skipped": ["region", "industry"]
            }
    else:
        # Role already verified by Stage 4 - just check region anti-gaming
        print(f"   🔍 QUICK CHECK: Verifying region anti-gaming (role already verified by Stage 4)...")
        quick_result = fuzzy_pre_verification_stage5(
            claimed_role="",  # Skip role check
            claimed_region=claimed_region,
            claimed_industry="",
            role_search_results=[],
            region_search_results=[],
            industry_search_results=[],
            full_name=full_name,
            company=company,
            role_only=True,  # Just anti-gaming checks
            role_verified_stage4=True  # Don't print confusing warnings about role
        )
    
    # EARLY EXIT: Region anti-gaming (multiple states) - skip region/industry GSE searches
    if quick_result.get("region_hard_fail"):
        print(f"   ❌ EARLY EXIT: Region anti-gaming - SKIPPING region and industry GSE searches")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Region FAILED (anti-gaming): {quick_result.get('region_reason')}",
            "failed_fields": ["region"],
            "early_exit": "region_anti_gaming_before_gse",
            "gse_searches_skipped": ["region", "industry"]
        }
    
    # ========================================================================
    # SMART CONDITIONAL GSE QUERIES
    # ========================================================================
    # Only run GSE queries for data NOT available from company LinkedIn cache.
    # Order: Region (always) → Industry/Description (if needed) → Employee Count (if needed)
    # Employee count GSE search uses the miner's company LinkedIn URL specifically
    # ========================================================================
    
    industry_results = []
    employee_count_results = []
    
    # Determine which GSE queries to run
    need_industry_gse = not has_industry_description
    need_employee_count_gse = not has_employee_count  # Need GSE if Stage 4 didn't get employee count
    
    # Get company LinkedIn slug for targeted employee count search
    company_linkedin_slug = None
    company_linkedin = lead.get("company_linkedin", "") or ""
    if company_linkedin:
        # Extract slug from URL like "linkedin.com/company/brivo-inc/"
        import re
        match = re.search(r'linkedin\.com/company/([^/]+)', company_linkedin)
        if match:
            company_linkedin_slug = match.group(1)
    
    # ========================================================================
    # REGION: Use PERSON location (NOT company HQ)
    # ========================================================================
    # Priority:
    # 1. Stage 4 extracted location from LinkedIn snippets → use it
    # 2. Stage 5 searches for person location using LinkedIn URL
    # 3. NEVER fall back to company HQ (that's not accurate for the person)
    # ========================================================================
    stage4_location = lead.get("stage4_extracted_location")
    use_stage4_location = bool(stage4_location)
    need_person_location_search = not use_stage4_location  # Need to search for person location
    
    # Get LinkedIn URL for person location search
    linkedin_url = lead.get("linkedin", "")
    
    print(f"   🔍 GSE: Starting conditional searches...")
    if use_stage4_location:
        print(f"      Person location: SKIP (using Stage 4: '{stage4_location}')")
    elif linkedin_url:
        print(f"      Person location: RUN (searching via LinkedIn URL)")
    else:
        print(f"      Person location: SKIP (no LinkedIn URL for search)")
        need_person_location_search = False
    print(f"      Industry/description search: {'SKIP (have from company LinkedIn)' if has_industry_description else 'RUN (need fallback)'}")
    print(f"      Employee count search: {'SKIP (have from company LinkedIn)' if has_employee_count else f'RUN (targeting {company_linkedin_slug})'}")
    
    # Build task list based on what we need
    tasks = []
    task_names = []
    
    # Person location search: only if Stage 4 didn't find it
    if need_person_location_search and linkedin_url:
        # Use new "person_location" search type that searches the LinkedIn URL
        person_location_task = _gse_search_stage5(
            "person_location",
            full_name=full_name,
            company=company,
            linkedin_url=linkedin_url,
            region_hint=claimed_region,
            role=claimed_role
        )
        tasks.append(person_location_task)
        task_names.append("person_location")
    
    # Industry search only if we don't have data from company LinkedIn
    if need_industry_gse:
        industry_task = _gse_search_stage5("industry", company=company, region_hint=claimed_region)
        tasks.append(industry_task)
        task_names.append("industry")
    
    # Employee count search using the miner's company LinkedIn URL
    if need_employee_count_gse and company_linkedin_slug:
        employee_count_task = _gse_search_employee_count(company=company, company_linkedin_slug=company_linkedin_slug)
        tasks.append(employee_count_task)
        task_names.append("employee_count")
    
    # Run all needed searches in parallel
    results = await asyncio.gather(*tasks) if tasks else []
    
    # Parse results based on task order
    result_idx = 0
    person_location_results = []
    region_results = []  # Keep for backward compatibility in fuzzy_pre_verification
    
    if need_person_location_search and linkedin_url:
        # Person location search was run
        if result_idx < len(results):
            person_location_results = results[result_idx]
            result_idx += 1
    
    if need_industry_gse:
        if result_idx < len(results):
            industry_results = results[result_idx]
            result_idx += 1
    
    if need_employee_count_gse and company_linkedin_slug:
        if result_idx < len(results):
            employee_count_results = results[result_idx]
            result_idx += 1
    
    # Store all search results in lead for test access
    lead["_stage5_search_results"] = {
        "person_location_results": person_location_results,
        "role_results": role_results,
        "industry_results": industry_results,
        "employee_count_results": employee_count_results,
        "stage4_location": stage4_location,
        "use_stage4_location": use_stage4_location
    }
    
    # Extract person location from search results if Stage 4 didn't find it
    # IMPORTANT: Only extract from results that match the miner's LinkedIn URL
    stage5_extracted_location = None
    if not use_stage4_location and person_location_results:
        print(f"   🔍 Extracting person location from {len(person_location_results)} results...")
        
        # Extract profile slug from miner's provided LinkedIn URL for matching
        profile_slug = linkedin_url.split("/in/")[-1].strip("/").split("?")[0].lower() if linkedin_url and "/in/" in linkedin_url else None
        
        for r in person_location_results[:5]:
            result_url = r.get("link", r.get("href", r.get("url", ""))).lower()
            snippet = r.get("body", r.get("snippet", ""))
            
            # ENFORCE: Only extract from results that match the profile slug
            if profile_slug and "linkedin.com/in/" in result_url:
                # Extract slug from result URL
                result_slug = result_url.split("/in/")[-1].strip("/").split("?")[0]
                
                # Normalize for comparison (handle hyphens, underscores)
                profile_slug_norm = profile_slug.replace("-", "").replace("_", "")
                result_slug_norm = result_slug.replace("-", "").replace("_", "")
                
                if profile_slug_norm != result_slug_norm:
                    # URL doesn't match miner's profile - skip this result
                    continue
            
            if snippet:
                location = extract_person_location_from_linkedin_snippet(snippet)
                if location:
                    stage5_extracted_location = location
                    print(f"   📍 Stage 5: Extracted person location from VERIFIED profile URL: '{location}'")
                    break
        
        if not stage5_extracted_location:
            print(f"   ⚠️ Could not extract person location from search results (no matching profile URLs)")
    
    # Log results
    if use_stage4_location:
        print(f"   📍 Using Stage 4 person location: '{stage4_location}'")
    elif stage5_extracted_location:
        print(f"   📍 Using Stage 5 person location: '{stage5_extracted_location}'")
    elif person_location_results:
        print(f"   ⚠️ Found {len(person_location_results)} person location results but no location extracted")
    else:
        print(f"   ⚠️ No person location found (will rely on LLM)")
    
    if need_industry_gse:
        if industry_results:
            print(f"   ✅ Found {len(industry_results)} industry search results (fallback)")
        else:
            print(f"   ⚠️ No industry results found (fallback)")
    else:
        print(f"   📦 Using company LinkedIn data for industry/description")
    
    if need_employee_count_gse:
        if employee_count_results:
            print(f"   ✅ Found {len(employee_count_results)} employee count results (from company LinkedIn GSE)")
        else:
            print(f"   ⚠️ No employee count found in company LinkedIn GSE")
    else:
        print(f"   📦 Using company LinkedIn data for employee count (from Stage 4)")
    
    # STEP 4: FULL FUZZY PRE-VERIFICATION (now with all results)
    print(f"   🔍 FUZZY: Full pre-verification before LLM...")
    
    fuzzy_result = fuzzy_pre_verification_stage5(
        claimed_role=claimed_role if not role_verified_by_stage4 else "",  # Skip role fuzzy check if Stage 4 verified
        claimed_region=claimed_region,
        claimed_industry=claimed_industry,
        role_search_results=role_results if not role_verified_by_stage4 else [],  # Empty if Stage 4 verified
        region_search_results=region_results,  # Empty if using Stage 4 location
        industry_search_results=industry_results,
        full_name=full_name,
        company=company
    )
    
    # If Stage 4 verified role, mark it as verified in fuzzy_result
    if role_verified_by_stage4:
        fuzzy_result["role_verified"] = True
        fuzzy_result["role_extracted"] = stage4_role
        fuzzy_result["role_reason"] = "Verified by Stage 4 profile title"
        # Remove "role" from needs_llm if Stage 4 already verified it
        if "role" in fuzzy_result.get("needs_llm", []):
            fuzzy_result["needs_llm"].remove("role")
    
    # ========================================================================
    # REGION: Use extracted person location for verification
    # ========================================================================
    # Priority: Stage 4 location > Stage 5 location > LLM fallback
    # This verifies the PERSON's location, not company HQ.
    # ========================================================================
    extracted_person_location = stage4_location or stage5_extracted_location
    location_source = "Stage 4" if stage4_location else ("Stage 5" if stage5_extracted_location else None)
    
    if extracted_person_location and claimed_region:
        print(f"   🔍 REGION: Comparing person location vs claimed region...")
        print(f"      {location_source} extracted: '{extracted_person_location}'")
        print(f"      Miner claimed: '{claimed_region}'")
        
        # Use GeoPy to compare locations
        geo_match, geo_reason = locations_match_geopy(claimed_region, extracted_person_location)
        
        fuzzy_result["region_extracted"] = extracted_person_location
        fuzzy_result["region_confidence"] = 0.95 if geo_match else 0.3
        fuzzy_result["region_reason"] = f"[{location_source} person location] {geo_reason}"
        
        if geo_match:
            fuzzy_result["region_verified"] = True
            print(f"   ✅ REGION MATCH: '{claimed_region}' ≈ '{extracted_person_location}'")
            print(f"      Reason: {geo_reason}")
            # Remove region from LLM verification if fuzzy matched
            if "region" in fuzzy_result.get("needs_llm", []):
                fuzzy_result["needs_llm"].remove("region")
        else:
            # GeoPy says no match - still send to LLM for final verification
            if not fuzzy_result.get("region_hard_fail"):
                if "region" not in fuzzy_result.get("needs_llm", []):
                    fuzzy_result["needs_llm"].append("region")
                print(f"   ⚠️ REGION: GeoPy says no match, sending to LLM for verification")
                print(f"      Claimed: {claimed_region} | Extracted: {extracted_person_location}")
    elif not extracted_person_location and claimed_region:
        # No person location found - send to LLM with whatever region results we have
        print(f"   ⚠️ REGION: No person location extracted, sending to LLM for verification")
        if "region" not in fuzzy_result.get("needs_llm", []):
            fuzzy_result["needs_llm"].append("region")
    
    # Note: role_definitive_fail already checked above (before region/industry GSE)
    # so we only check region anti-gaming here
    
    # EARLY EXIT: Region anti-gaming AND role already verified
    if fuzzy_result.get("region_hard_fail") and fuzzy_result.get("role_verified"):
        print(f"   ❌ EARLY EXIT: Region anti-gaming triggered - skipping employee count and industry checks")
        return False, {
            "stage": "Stage 5: Role/Region/Employee Count/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Region FAILED (anti-gaming): {fuzzy_result.get('region_reason')}",
            "failed_fields": ["region"],
            "early_exit": "region_anti_gaming",
            "role_passed": True,
            "extracted_role": fuzzy_result.get("role_extracted")
        }
    
    # STEP: EMPLOYEE COUNT VERIFICATION (after region, before industry)
    # Sources: (1) Stage 4 company LinkedIn data, (2) GSE search of company LinkedIn
    # Employee count MUST match exactly (same LinkedIn range)
    employee_count_match = False
    extracted_employee_count = None
    employee_count_reason = ""
    employee_count_source = None
    
    if claimed_employee_count:
        print(f"   🔍 EMPLOYEE COUNT: Verifying claimed '{claimed_employee_count}'...")
        
        # PRIORITY 1: Company LinkedIn data (from Stage 4 scraping)
        if use_company_linkedin_for_verification and company_linkedin_data:
            linkedin_employee_count = company_linkedin_data.get("employee_count")
            if linkedin_employee_count:
                print(f"   📊 EMPLOYEE COUNT: Using Stage 4 company LinkedIn data: '{linkedin_employee_count}'")
                extracted_employee_count = linkedin_employee_count
                employee_count_source = "company_linkedin_stage4"
        
        # PRIORITY 2: GSE search of company LinkedIn (if Stage 4 didn't have it)
        if not extracted_employee_count and employee_count_results:
            gse_employee_count = extract_employee_count_from_results(employee_count_results, company, company_linkedin_slug)
            if gse_employee_count:
                print(f"   📊 EMPLOYEE COUNT: Using GSE search of company LinkedIn: '{gse_employee_count}'")
                extracted_employee_count = gse_employee_count
                employee_count_source = "company_linkedin_gse"
        
        if extracted_employee_count:
            # STRICT: Require exact range match
            employee_count_match, employee_count_reason = fuzzy_match_employee_count(
                claimed_employee_count, 
                extracted_employee_count
            )
            
            if employee_count_match:
                print(f"   ✅ EMPLOYEE COUNT MATCH: {employee_count_reason} (source: {employee_count_source})")
            else:
                print(f"   ❌ EMPLOYEE COUNT MISMATCH: {employee_count_reason}")
                # EARLY EXIT: Employee count failed - skip industry check
                return False, {
                    "stage": "Stage 5: Employee Count Verification",
                    "check_name": "check_stage5_unified",
                    "message": f"Employee Count FAILED: Miner claimed '{claimed_employee_count}' but company LinkedIn shows '{extracted_employee_count}'",
                    "failed_fields": ["employee_count"],
                    "early_exit": "employee_count_mismatch",
                    "claimed_employee_count": claimed_employee_count,
                    "extracted_employee_count": extracted_employee_count,
                    "match_reason": employee_count_reason,
                    "data_source": employee_count_source
                }
        else:
            # Could not extract employee count from company LinkedIn (both Stage 4 and GSE) - FAIL
            print(f"   ❌ EMPLOYEE COUNT: Could not extract from company LinkedIn - verification failed")
            return False, {
                "stage": "Stage 5: Employee Count Verification",
                "check_name": "check_stage5_unified",
                "message": f"Employee Count verification failed: Could not extract employee count from company LinkedIn page",
                "failed_fields": ["employee_count", "company_linkedin"],
                "early_exit": "employee_count_not_found",
                "claimed_employee_count": claimed_employee_count,
                "extracted_employee_count": None,
                "match_reason": "Company LinkedIn page did not contain employee count data (tried Stage 4 and GSE search)",
                "data_source": None
            }
    else:
        # No employee count claimed - this is required, should fail
        print(f"   ❌ EMPLOYEE COUNT: No claim provided - this field is required")
        return False, {
            "stage": "Stage 5: Employee Count Verification",
            "check_name": "check_stage5_unified",
            "message": "Employee Count verification failed: Miner did not provide employee_count",
            "failed_fields": ["employee_count"],
            "early_exit": "employee_count_missing",
            "claimed_employee_count": None,
            "extracted_employee_count": None,
            "match_reason": "No employee count provided by miner",
            "data_source": None
        }
    
    # Store employee count results on lead
    lead["stage5_employee_count_match"] = employee_count_match
    lead["stage5_claimed_employee_count"] = claimed_employee_count
    lead["stage5_extracted_employee_count"] = extracted_employee_count
    
    # Check if all fields were fuzzy-matched
    if not fuzzy_result["needs_llm"]:
        print(f"   ✅ FUZZY: All fields matched - skipping LLM!")
        lead["stage5_role_match"] = True
        lead["stage5_region_match"] = True
        lead["stage5_industry_match"] = True
        lead["stage5_extracted_role"] = fuzzy_result["role_extracted"]
        lead["stage5_extracted_region"] = fuzzy_result["region_extracted"]
        # Use miner's original country/state/city fields (submitted via gateway)
        lead["region_country"] = lead.get("country", "")
        lead["region_state"] = lead.get("state", "")
        lead["region_city"] = lead.get("city", "")
        return True, None
    
    # STEP 5: LLM VERIFICATION for remaining fields
    needs_llm = fuzzy_result["needs_llm"]
    print(f"   🤖 LLM: Need to verify: {needs_llm}")
    
    # Show what extracted values are being passed to LLM
    if "role" in needs_llm:
        extracted_role = fuzzy_result.get("role_extracted", "NOT_EXTRACTED")
        print(f"      📝 ROLE → Passing to LLM: Claimed='{claimed_role}' | Extracted='{extracted_role}'")
    
    if "region" in needs_llm:
        extracted_region = fuzzy_result.get("region_extracted", "NOT_EXTRACTED")
        print(f"      📝 REGION → Passing to LLM: Claimed='{claimed_region}' | Extracted='{extracted_region}'")
    
    if "industry" in needs_llm:
        # Industry is always sent to LLM (too subjective for fuzzy matching)
        print(f"      📝 INDUSTRY → Passing to LLM: Claimed='{claimed_industry}' | Search results will be analyzed")
    
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
            snippet = result.get("snippet", result.get("body", ""))
            region_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    # COMPANY LINKEDIN DATA (if available)
    company_linkedin_context = ""
    if use_company_linkedin_for_verification and company_linkedin_data:
        company_linkedin_context = "\nCOMPANY LINKEDIN DATA (from miner's provided company_linkedin URL):\n"
        if company_linkedin_data.get("company_name_from_linkedin"):
            company_linkedin_context += f"- Company Name: {company_linkedin_data['company_name_from_linkedin']}\n"
        if company_linkedin_data.get("industry"):
            company_linkedin_context += f"- Industry: {company_linkedin_data['industry']}\n"
        if company_linkedin_data.get("description"):
            company_linkedin_context += f"- Description: {company_linkedin_data['description'][:300]}\n"
        if company_linkedin_data.get("employee_count"):
            company_linkedin_context += f"- Employee Count: {company_linkedin_data['employee_count']}\n"
        if company_linkedin_data.get("location"):
            company_linkedin_context += f"- Location: {company_linkedin_data['location']}\n"
    
    industry_context = ""
    # ALWAYS include industry search results - industry is ALWAYS verified by LLM
    # BUG FIX: Previously this had "if 'industry' in needs_llm" but 'industry' is NEVER in needs_llm
    # This caused LLM to verify industry without any search context, leading to false rejections
    if industry_results:
        industry_context = "\nINDUSTRY SEARCH RESULTS:\n"
        for i, result in enumerate(industry_results[:4], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("body", ""))
            industry_context += f"{i}. {title}\n   {snippet[:150]}\n"
    
    all_search_context = role_context + region_context + company_linkedin_context + industry_context
    
    # AUTO-FAIL if role needs LLM but no context
    if "role" in needs_llm and not role_context.strip():
        print(f"   ❌ AUTO-FAIL: No ScrapingDog data for role verification")
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": "No search results found to verify role",
            "failed_fields": ["role"]
        }
    
    if not all_search_context.strip():
        print(f"   ❌ AUTO-FAIL: No ScrapingDog search results at all")
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
   - If you cannot find ANY location info → region_match=false, extracted_region="NOT_FOUND"
   - FAIL if you cannot verify the claimed region from search results
""")
    else:
        claims_to_verify.append(f'2. REGION: "{claimed_region}" ✅ (Already verified by fuzzy match)')
    
    # Always verify industry + sub_industry + description together (exact matches already validated)
    claims_to_verify.append(f'3. INDUSTRY: "{claimed_industry}" (taxonomy-validated)')
    claims_to_verify.append(f'4. SUB-INDUSTRY: "{claimed_sub_industry}" (taxonomy-validated)')
    claims_to_verify.append(f'   Definition: "{sub_industry_definition[:200]}..."' if len(sub_industry_definition) > 200 else f'   Definition: "{sub_industry_definition}"')
    if claimed_description:
        claims_to_verify.append(f'5. DESCRIPTION: "{claimed_description[:200]}..."' if len(claimed_description) > 200 else f'5. DESCRIPTION: "{claimed_description}"')
    
    # Add company LinkedIn context note if available
    linkedin_note = ""
    if use_company_linkedin_for_verification:
        linkedin_note = """
   - PRIORITY: Use COMPANY LINKEDIN DATA section above (from miner's provided company_linkedin URL) as primary source
   - Company LinkedIn data is authoritative - if it shows industry/description, weight it highly"""
    
    verification_rules.append(f"""
3. INDUSTRY & SUB-INDUSTRY VERIFICATION:
   - The industry "{claimed_industry}" and sub-industry "{claimed_sub_industry}" have been validated as exact taxonomy matches
   - Your job is to verify the COMPANY actually operates in this industry/sub-industry{linkedin_note}
   - Look at the search results and COMPANY LINKEDIN DATA to determine if the company's business matches:
     * Industry: "{claimed_industry}"
     * Sub-industry: "{claimed_sub_industry}"
     * Definition: "{sub_industry_definition}"
   - IMPORTANT: The definition may include examples, but examples are NOT exhaustive - other similar products/services also qualify
   - Verify the company fits the industry "{claimed_industry}" AND sub-industry "{claimed_sub_industry}"
   - PASS if the company operates in a space that reasonably fits the industry AND sub-industry category
   - FAIL only if the company operates in a completely unrelated field
""")
    
    if claimed_description:
        desc_linkedin_note = ""
        if use_company_linkedin_for_verification and company_linkedin_data and company_linkedin_data.get("description"):
            desc_linkedin_note = f"\n   - PRIORITY: Compare with COMPANY LINKEDIN DATA description: \"{company_linkedin_data['description'][:200]}...\""
        
        verification_rules.append(f"""
4. DESCRIPTION VERIFICATION:
   - Compare the miner's description to what you find in COMPANY LINKEDIN DATA about the company{desc_linkedin_note}
   - KEY QUESTION: Do both descriptions describe a company in the SAME INDUSTRY doing the SAME type of work?
   - FOCUS ON: Does the company's CORE BUSINESS match? (not exact wording)
   - description_match = true if:
     * Both describe the same industry sector (e.g., both describe a financial services company)
     * Both describe the same type of product/service (e.g., both describe B2B software)
     * Wording differs but the fundamental business is the same
   - description_match = false ONLY if:
     * Completely DIFFERENT industry (e.g., "software company" vs "construction firm")
     * Fundamentally different product type (e.g., "SaaS platform" vs "consulting services")
   - IMPORTANT: The industry "{matched_industry}" has already been verified - use this as a guide
""")
    
    claims_section = "\n".join(claims_to_verify)
    rules_section = "\n".join(verification_rules)
    
    response_fields = []
    if "role" in needs_llm:
        response_fields.append('"role_match": true/false,\n    "extracted_role": "role found in search results"')
    if "region" in needs_llm:
        response_fields.append('"region_match": true/false,\n    "extracted_region": "company HQ from search"')
    # Always include industry + sub_industry verification (exact matches already validated)
    response_fields.append('"industry_match": true/false,\n    "extracted_industry": "industry from search"')
    response_fields.append('"sub_industry_match": true/false,\n    "sub_industry_reasoning": "does company match the sub-industry definition?"')
    if claimed_description:
        response_fields.append('"description_match": true/false,\n    "description_reasoning": "is description accurate?"')
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
    # REMOVE THE FOLLOWING 5 PRINTS
    # DEBUG: Log full LLM prompt for diagnosis
    print(f"\n{'='*80}")
    print(f"🤖 LLM PROMPT FOR {company}:")
    print(f"{'='*80}")
    print(prompt)
    print(f"{'='*80}\n")

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
                
                # Industry - always verified by LLM now (exact match already validated)
                industry_match = result.get("industry_match", False)
                extracted_industry = result.get("extracted_industry", "")
                
                if not industry_match:
                    print(f"   ❌ INDUSTRY LLM FAILED: Company does not match industry '{claimed_industry}'")
                    return False, {
                        "stage": "Stage 5: Industry Verification",
                        "check_name": "check_stage5_unified",
                        "message": f"Industry verification failed: Company does not appear to operate in '{claimed_industry}'",
                        "failed_fields": ["industry"],
                        "extracted_industry": extracted_industry
                    }
                
                # Sub-industry - verified by LLM (exact match already validated)
                sub_industry_match = result.get("sub_industry_match", False)
                sub_industry_reasoning = result.get("sub_industry_reasoning", "")
                
                if not sub_industry_match:
                    print(f"   ❌ SUB-INDUSTRY LLM FAILED: Company does not match sub-industry '{claimed_sub_industry}'")
                    return False, {
                        "stage": "Stage 5: Sub-Industry Verification",
                        "check_name": "check_stage5_unified",
                        "message": f"Sub-industry verification failed: Company does not match '{claimed_sub_industry}' definition",
                        "failed_fields": ["sub_industry"],
                        "sub_industry_reasoning": sub_industry_reasoning,
                        "sub_industry_definition": sub_industry_definition
                    }
                
                # Description verification (if description was provided)
                description_match = True
                description_reasoning = "No description provided"
                if claimed_description:
                    description_match = result.get("description_match", True)
                    description_reasoning = result.get("description_reasoning", "")
                    
                    if not description_match:
                        # TARGETED OVERRIDE: If verified sub-industry keywords appear in description,
                        # the description is consistent with the verified business type.
                        # This is SAFE because sub-industry was already verified by LLM above.
                        # Example: sub_industry="Lending", description="fix and flip lending company"
                        #          → "lending" found in description → consistent with verified business
                        desc_lower = claimed_description.lower()
                        sub_lower = claimed_sub_industry.lower() if claimed_sub_industry else ""
                        industry_lower = claimed_industry.lower() if claimed_industry else ""
                        
                        # Get significant words from verified sub-industry and industry (skip short words)
                        sub_words = [w for w in sub_lower.split() if len(w) > 3]
                        industry_words = [w for w in industry_lower.split() if len(w) > 3]
                        
                        # Check if any sub-industry keywords appear in description
                        matching_sub = [w for w in sub_words if w in desc_lower]
                        matching_industry = [w for w in industry_words if w in desc_lower]
                        
                        if matching_sub:
                            # Sub-industry keyword found in description - override LLM
                            description_match = True
                            description_reasoning = f"Override: Verified sub-industry '{claimed_sub_industry}' keywords {matching_sub} found in description"
                            print(f"   🔄 DESCRIPTION OVERRIDE: Sub-industry keywords {matching_sub} found in description")
                        elif matching_industry:
                            # Industry keyword found in description - override LLM
                            description_match = True
                            description_reasoning = f"Override: Verified industry '{claimed_industry}' keywords {matching_industry} found in description"
                            print(f"   🔄 DESCRIPTION OVERRIDE: Industry keywords {matching_industry} found in description")
                        else:
                            # No keyword match - LLM decision stands
                            print(f"   ❌ DESCRIPTION FAILED: Description does not match company")
                            return False, {
                                "stage": "Stage 5: Description Verification",
                                "check_name": "check_stage5_unified",
                                "message": f"Description verification failed: Description does not accurately describe the company",
                                "failed_fields": ["description"],
                                "description_reasoning": description_reasoning
                            }
                
                # Store sub_industry results on lead
                lead["stage5_sub_industry_match"] = sub_industry_match
                lead["stage5_claimed_sub_industry"] = claimed_sub_industry
                lead["stage5_matched_sub_industry"] = matched_sub_industry
                lead["stage5_sub_industry_reason"] = sub_industry_reasoning
                lead["stage5_description_match"] = description_match
                lead["stage5_description_reasoning"] = description_reasoning
                
                all_match = role_match and region_match and industry_match and sub_industry_match and description_match
                
                # Store results on lead
                lead["stage5_role_match"] = role_match
                lead["stage5_region_match"] = region_match
                lead["stage5_industry_match"] = industry_match
                lead["stage5_extracted_role"] = extracted_role
                lead["stage5_extracted_region"] = extracted_region
                lead["stage5_extracted_industry"] = extracted_industry
                
                # Use miner's original country/state/city fields (submitted via gateway)
                # These are 100% accurate since miner explicitly provided them
                if region_match:
                    lead["region_country"] = lead.get("country", "")
                    lead["region_state"] = lead.get("state", "")
                    lead["region_city"] = lead.get("city", "")
                
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
                    if not sub_industry_match:
                        failed_fields.append("sub_industry")
                    
                    return False, {
                        "stage": "Stage 5: Role/Region/Industry/Sub-Industry",
                        "check_name": "check_stage5_unified",
                        "message": f"Stage 5 verification failed for: {', '.join(failed_fields)}",
                        "failed_fields": failed_fields,
                        "role_match": role_match,
                        "region_match": region_match,
                        "industry_match": industry_match,
                        "sub_industry_match": sub_industry_match,
                        "sub_industry_reason": sub_industry_reason
                    }
                
    except Exception as e:
        return False, {
            "stage": "Stage 5: Role/Region/Industry",
            "check_name": "check_stage5_unified",
            "message": f"Stage 5 verification failed: {str(e)}",
            "failed_fields": ["exception"]
        }


# ========================================================================
# ICP (Ideal Customer Profile) Multiplier Determination
# ========================================================================

def determine_icp_multiplier(lead: dict) -> float:
    """
    LEGACY FUNCTION - Kept for backwards compatibility.
    
    Determine if a lead matches our ICP (Ideal Customer Profile) criteria.
    
    Uses the ICP_DEFINITIONS table (defined at top of file) to check if a lead matches
    any target customer profile based on:
    - Sub-Industry (e.g., "Gas Stations", "AI Startups")
    - Role Type (e.g., "Operations", "Technology", "Leadership")
    - Role Details (specific titles like "CEO", "CTO", "VP of Operations")
    - Region (optional - e.g., "Africa" for streaming/broadcast ICP)
    
    Returns:
        Custom multiplier if defined in ICP (e.g., 5.0 for Africa)
        1.5 if lead matches ICP criteria (default)
        1.0 if lead is standard (non-ICP)
        
    NOTE: This function is deprecated. Use calculate_icp_adjustment() instead.
    """
    # Extract lead fields (case-insensitive)
    sub_industry = lead.get("sub_industry", "").strip().lower()
    role = lead.get("role", "").strip().lower()
    region = lead.get("region", "").strip().lower()
    
    # Helper function to check if any keyword matches in text
    def matches_any(text: str, keywords: list) -> bool:
        """Check if any keyword from the list is found in the text (case-insensitive)"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    # Iterate through all ICP definitions
    for icp in ICP_DEFINITIONS:
        # Step 1: Check if sub_industry matches
        if not matches_any(sub_industry, icp["sub_industries"]):
            continue  # No match, try next ICP
        
        # Step 2: Check region if specified in ICP definition
        # If "regions" is defined, lead must be from one of those regions
        if "regions" in icp:
            if not matches_any(region, icp["regions"]):
                continue  # Region doesn't match, try next ICP
        
        # Step 3: Check if role contains role_details (specific titles)
        # Role details are the most specific check (e.g., "CEO", "CTO", "VP of Operations")
        if matches_any(role, icp["role_details"]):
            # Return custom multiplier if defined, otherwise default 1.5x
            return icp.get("multiplier", 1.5)
    
    # No ICP match found
    return 1.0


def _matches_icp_definitions(lead: dict) -> bool:
    """
    Check if a lead matches any ICP definition (without returning multiplier value).
    
    Returns:
        True if lead matches any ICP definition
        False otherwise
    """
    sub_industry = lead.get("sub_industry", "").strip().lower()
    role = lead.get("role", "").strip().lower()
    region = lead.get("region", "").strip().lower()
    
    def matches_any(text: str, keywords: list) -> bool:
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    for icp in ICP_DEFINITIONS:
        if not matches_any(sub_industry, icp["sub_industries"]):
            continue
        if "regions" in icp:
            if not matches_any(region, icp["regions"]):
                continue
        if matches_any(role, icp["role_details"]):
            return True
    
    return False


def calculate_icp_adjustment(lead: dict) -> int:
    """
    Calculate ICP adjustment points (NEW SYSTEM - replaces multiplier).
    
    This function calculates an absolute point adjustment based on:
    1. ICP Definition Match: +50 points
    2. Small Company in Major Hub Bonus: +50 points
       - ≤10 employees AND in major hub (NYC, SF, LA, Austin, Chicago, etc.)
    3. Small Company Bonus:
       - ≤50 employees: +20 points
    4. Large Company Penalty:
       - >1,000 employees: -10 points
       - >5,000 or >10,000 employees: -15 points
    
    MAX POSITIVE BONUS: +50 (ICP and small company bonuses do NOT stack beyond 50)
    PENALTIES STACK: Penalties are applied AFTER capping the bonus
    
    Args:
        lead: Lead dictionary with employee_count, city, and ICP-relevant fields
        
    Returns:
        Integer adjustment (bonus capped at +50, then penalties applied)
        
    Examples:
        - ICP match only = +50
        - ICP + ≤50 employees = +70 → capped to +50
        - ICP + >1k employees = +50 - 10 = +40
        - ICP + >5k employees = +50 - 15 = +35
        - Small hub (≤10 + NYC) = +50
        - Small hub + >1k employees = +50 - 10 = +40
        - Non-ICP + ≤50 employees = +20
        - Non-ICP + >5k employees = -15
    """
    bonus = 0
    penalty = 0
    breakdown = {"icp_match": 0, "major_hub_bonus": 0, "employee_bonus": 0, "employee_penalty": 0}
    
    # ========================================================================
    # MAJOR HUBS BY COUNTRY (city + country must BOTH match)
    # ========================================================================
    # Uses CANONICAL city names from geo_lookup_fast.json (post-gateway normalization)
    # Gateway normalizes: "NYC" -> "New York City", "SF" -> "San Francisco", etc.
    # So we only need the canonical names here - no aliases needed!
    # 
    # Country names MUST match gateway/api/submit.py VALID_COUNTRIES (lowercase)
    MAJOR_HUBS_BY_COUNTRY = {
        # ----------------------------------------------------------------
        # NORTH AMERICA (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "united states": {
            # NYC area (manhattan/brooklyn are separate cities in JSON)
            "new york city", "manhattan", "brooklyn",
            # West Coast
            "san francisco", "los angeles", "san diego", "san jose", "seattle", "portland",
            # Texas
            "austin", "dallas", "houston",
            # Other major hubs
            "chicago", "boston", "denver", "miami", "washington", "atlanta", "phoenix",
        },
        "canada": {
            "toronto", "vancouver", "montréal",  # Note: "montréal" is canonical (not "montreal")
        },
        # ----------------------------------------------------------------
        # EUROPE (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "united kingdom": {
            "london", "manchester", "edinburgh", "cambridge", "oxford",
        },
        "germany": {
            "berlin", "münchen", "frankfurt am main", "hamburg",  # "münchen" is canonical
        },
        "france": {
            "paris",
        },
        "netherlands": {
            "amsterdam", "rotterdam",
        },
        "switzerland": {
            "zürich", "genève",  # Canonical names with accents
        },
        "ireland": {
            "dublin",
        },
        "sweden": {
            "stockholm",
        },
        "spain": {
            "barcelona", "madrid",
        },
        # ----------------------------------------------------------------
        # ASIA-PACIFIC (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "hong kong": {
            "hong kong",
        },
        "singapore": {
            "singapore",
        },
        "japan": {
            "tokyo", "osaka",
        },
        "south korea": {
            "seoul",
        },
        "china": {
            "shanghai", "beijing", "shenzhen",
        },
        "india": {
            "bengaluru", "mumbai", "new delhi", "hyderabad", "pune",  # "bengaluru" is canonical
        },
        "australia": {
            "sydney", "melbourne",
        },
        "new zealand": {
            "auckland",
        },
        # ----------------------------------------------------------------
        # MIDDLE EAST (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "israel": {
            "tel aviv",
        },
        "united arab emirates": {
            "dubai", "abu dhabi",
        },
        # ----------------------------------------------------------------
        # SOUTH AMERICA (canonical names from geo_lookup_fast.json)
        # ----------------------------------------------------------------
        "brazil": {
            "são paulo",  # Canonical name with accent
        },
    }
    
    # Get city and country for major hub check
    city = lead.get("city", "").strip().lower()
    country = lead.get("country", "").strip().lower()
    
    # Check if BOTH country AND city match a major hub
    # Simple exact matching - gateway already normalized cities to canonical form
    is_major_hub = False
    matched_hub = None
    
    if country in MAJOR_HUBS_BY_COUNTRY:
        hub_cities = MAJOR_HUBS_BY_COUNTRY[country]  # This is now a set
        if city in hub_cities:
            is_major_hub = True
            matched_hub = f"{city} ({country})"
    
    # ========================================================================
    # STEP 1: ICP Definition Match (+50 points)
    # ========================================================================
    if _matches_icp_definitions(lead):
        bonus += 50
        breakdown["icp_match"] = 50
        print(f"   🎯 ICP MATCH: +50 points")
    
    # ========================================================================
    # STEP 2: Employee Count Bonuses and Penalties
    # ========================================================================
    employee_count_str = get_employee_count(lead) or ""
    
    if employee_count_str:
        parsed = parse_employee_count(employee_count_str)
        
        if parsed:
            emp_min, emp_max = parsed
            
            # Small company in major hub bonus (+50 points)
            if emp_max <= 10 and is_major_hub:
                bonus += 50
                breakdown["major_hub_bonus"] = 50
                print(f"   🌆 SMALL COMPANY IN MAJOR HUB (≤10 + {matched_hub}): +50 points")
            # Small company bonus (+20 points for ≤50 employees)
            elif emp_max <= 50:
                bonus += 20
                breakdown["employee_bonus"] = 20
                print(f"   🏢 SMALL COMPANY (≤50): +20 points")
            
            # Large company penalty (stacks with capped bonus)
            # Note: Uses emp_min to determine the MINIMUM company size
            if emp_min > 5000:
                # 5,001+ employees (includes 10,001+)
                penalty = 15
                breakdown["employee_penalty"] = -15
                print(f"   🏭 LARGE COMPANY (>5k): -15 points")
            elif emp_min > 1000:
                # 1,001-5,000 employees
                penalty = 10
                breakdown["employee_penalty"] = -10
                print(f"   🏭 LARGE COMPANY (>1k): -10 points")
    else:
        print(f"   📋 No employee count available - no size adjustment")
    
    # ========================================================================
    # STEP 3: Cap bonus at +50, then apply penalties
    # ========================================================================
    if bonus > 50:
        print(f"   ⚠️  Bonus {bonus} exceeds cap, capping at +50")
        bonus = 50
    
    # Penalties stack with capped bonus
    adjustment = bonus - penalty
    
    print(f"   📊 FINAL ICP ADJUSTMENT: {adjustment:+d} points")
    print(f"      Bonus (capped at 50): {min(bonus, 50):+d} = ICP:{breakdown['icp_match']:+d} + Hub:{breakdown['major_hub_bonus']:+d} + Size:{breakdown['employee_bonus']:+d}")
    print(f"      Penalty: {-penalty:+d}")
    
    return adjustment


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
    
    # OPTIMIZATION: Run instant checks first, then overlap HEAD request with Stage 1 DNS checks
    # Instant checks (run sequentially - they're <0.01s each anyway)
    checks_stage0_instant = [
        check_required_fields,      # Required fields validation (HARD)
        check_email_regex,          # RFC-5322 regex validation (HARD)
        check_name_email_match,     # Name in email check (HARD) - NEW
        check_general_purpose_email,# General purpose email filter (HARD) - NEW
        check_free_email_domain,    # Reject free email domains (HARD) - NEW
        check_disposable,           # Filter throwaway email providers (HARD)
    ]

    for check_func in checks_stage0_instant:
        passed, rejection_reason = await check_func(lead)
        if not passed:
            msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
            print(f"   ❌ Stage 0 failed: {msg}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = rejection_reason
            return False, automated_checks_data

    # Collect Stage 0 data after successful instant checks
    automated_checks_data["stage_0_hardcoded"]["name_in_email"] = True  # Passed name-email match
    automated_checks_data["stage_0_hardcoded"]["is_general_purpose_email"] = False  # Not general purpose

    print("   ✅ Stage 0 instant checks passed")
    
    # OPTIMIZATION: Start HEAD request as background task (will check result after Stage 1)
    # This overlaps the 5-10s HEAD request with 1-3s Stage 1 DNS checks
    head_request_task = asyncio.create_task(check_head_request(lead))

    # ========================================================================
    # Stage 1: DNS Layer (MIXED)
    # - Domain Age, MX Record (HARD)
    # - SPF/DMARC (SOFT - always passes, appends data)
    # ========================================================================
    print(f"🔍 Stage 1: DNS layer checks for {email} @ {company}")
    
    # OPTIMIZATION: Run all Stage 1 DNS checks in parallel to save time
    # Old: Sequential execution = 2-5s total
    # New: Parallel execution = 1-3s (time of slowest check)
    results = await asyncio.gather(
        check_domain_age(lead),
        check_mx_record(lead),
        check_spf_dmarc(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )
    
    # Check results
    check_names = ["check_domain_age", "check_mx_record", "check_spf_dmarc"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exception
            print(f"   ❌ Stage 1 failed: {str(result)}")
            automated_checks_data["passed"] = False
            automated_checks_data["rejection_reason"] = {
                "stage": "Stage 1: DNS Layer",
                "check_name": check_names[i],
                "message": f"Check failed: {str(result)}",
                "failed_fields": ["domain"]
            }
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
        
        passed, rejection_reason = result
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
    # Stage 0 (continued): HEAD Request Check
    # Check result of background HEAD request task that was started before Stage 1
    # ========================================================================
    print(f"🔍 Stage 0: Website HEAD request check for {email} @ {company}")
    passed, rejection_reason = await head_request_task
    if not passed:
        msg = rejection_reason.get("message", "Unknown error") if rejection_reason else "Unknown error"
        print(f"   ❌ Stage 0 (HEAD request) failed: {msg}")
        automated_checks_data["passed"] = False
        automated_checks_data["rejection_reason"] = rejection_reason
        return False, automated_checks_data
    
    print("   ✅ Stage 0 (HEAD request) passed")

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
    # Stage 3: Email Verification (DEPRECATED - use run_batch_automated_checks instead)
    # ========================================================================
    # NOTE: Single-email validation has been removed. Email validation is now
    # handled by TrueList BATCH API in run_batch_automated_checks().
    # This function is kept for backwards compatibility but should not be used.
    print(f"⚠️  Stage 3: DEPRECATED - use run_batch_automated_checks() for email validation")
    print(f"   Skipping single-email validation for {email}")
    
    # Mark Stage 3 as skipped (not verified)
    automated_checks_data["stage_3_email"]["email_status"] = "skipped"
    automated_checks_data["stage_3_email"]["email_score"] = 0
    automated_checks_data["stage_3_email"]["is_disposable"] = False
    automated_checks_data["stage_3_email"]["is_role_based"] = False
    automated_checks_data["stage_3_email"]["is_free"] = False
    
    print("   ⏭️  Stage 3 skipped (use batch validation)")

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
    # - Uses ScrapingDog search + fuzzy matching + LLM to verify role, region, industry
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
    print(f"📊 Rep Score: Running soft checks for {email} @ {company} (parallel execution)")
    
    # OPTIMIZATION: Run all rep score checks in parallel to save time
    # Old: Sequential execution = 6-12s total
    # New: Parallel execution = 3-4s total (time of slowest API)
    results = await asyncio.gather(
        check_wayback_machine(lead),
        check_sec_edgar(lead),
        check_whois_dnsbl_reputation(lead),
        check_gdelt_mentions(lead),
        check_companies_house(lead),
        return_exceptions=True  # Don't fail entire batch if one check fails
    )
    
    # Unpack results (handle exceptions gracefully)
    wayback_score, wayback_data = results[0] if not isinstance(results[0], Exception) else (0, {"error": str(results[0])})
    sec_score, sec_data = results[1] if not isinstance(results[1], Exception) else (0, {"error": str(results[1])})
    whois_dnsbl_score, whois_dnsbl_data = results[2] if not isinstance(results[2], Exception) else (0, {"error": str(results[2])})
    gdelt_score, gdelt_data = results[3] if not isinstance(results[3], Exception) else (0, {"error": str(results[3])})
    companies_house_score, companies_house_data = results[4] if not isinstance(results[4], Exception) else (0, {"error": str(results[4])})
    
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
    
    # ========================================================================
    # ICP Adjustment Calculation (NEW SYSTEM - Absolute Points)
    # Replaces the old multiplier system with absolute point adjustments
    # ========================================================================
    icp_adjustment = calculate_icp_adjustment(lead)
    # Store in is_icp_multiplier field for backwards compatibility
    # Values: -15 to +20 (new format) vs 1.0/1.5/5.0 (old format)
    lead["is_icp_multiplier"] = float(icp_adjustment)
    automated_checks_data["is_icp_multiplier"] = float(icp_adjustment)

    # ========================================================================
    # Company Name Standardization (only on approval)
    # ========================================================================
    # Use the company LinkedIn slug to get/set the standardized company name.
    # This ensures all leads with the same company_linkedin URL have the same
    # standardized company name, regardless of how the miner submitted it.
    # ========================================================================
    company_slug = lead.get("company_linkedin_slug")
    company_linkedin_data = lead.get("company_linkedin_data")

    if company_slug:
        # Check cache first
        standardized_name = get_standardized_company_name(company_slug)

        if standardized_name:
            # Cache hit - use cached standardized name
            print(f"   📦 Company name from cache: '{company_slug}' → '{standardized_name}'")
        else:
            # Cache miss - get from Stage 4 scraped data and save to cache
            if company_linkedin_data and company_linkedin_data.get("company_name_from_linkedin"):
                standardized_name = company_linkedin_data["company_name_from_linkedin"]
                set_standardized_company_name(company_slug, standardized_name)
            else:
                # Fallback to miner's submitted company name if no scraped data
                standardized_name = company
                print(f"   ⚠️ No scraped company name available, using submitted: '{standardized_name}'")

        # Set on lead and automated_checks_data
        lead["company_standardized"] = standardized_name
        automated_checks_data["company_standardized"] = standardized_name
        print(f"   ✅ Company standardized: '{company}' → '{standardized_name}'")
    else:
        # No company_linkedin_slug - use submitted company name
        lead["company_standardized"] = company
        automated_checks_data["company_standardized"] = company
        print(f"   ⚠️ No company LinkedIn slug, using submitted name: '{company}'")

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

# EMAIL_CACHE = asyncio.run(load_email_cache())  # Disabled to avoid event loop issues
EMAIL_CACHE = {}

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
    """
    Verify company website is accessible.
    
    Strategy: Try HEAD first (lightweight), fall back to GET if HEAD fails.
    Many enterprise sites (Intuit, 3M, etc.) block HEAD requests but work with GET.
    Uses browser User-Agent to avoid anti-bot blocking.
    Uses custom SSL context with broader cipher support for enterprise sites (Hartford, etc.)
    """
    import ssl
    
    if not company_domain:
        return False, "No domain provided"
    if not company_domain.startswith(("http://", "https://")):
        company_domain = f"https://{company_domain}"
    
    # Status codes that indicate website exists (pass immediately)
    # 429 = Too Many Requests (rate limiting/bot protection) - proves site exists, just blocking automated requests
    PASS_STATUS_CODES = {200, 301, 302, 307, 308, 401, 403, 405, 429, 500, 502, 503}
    
    # Browser User-Agent to avoid anti-bot blocking (3M, etc.)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Create custom SSL context with broader cipher support
    # Some enterprise sites (Hartford, etc.) have strict SSL configs that reject default ciphers
    ssl_context = ssl.create_default_context()
    # Allow older TLS versions for compatibility with enterprise sites
    ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')
    # Add additional options for maximum compatibility
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Create connector with custom SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    try:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            # Try HEAD request first (lightweight)
            head_status = None
            head_error = None
            try:
                async with session.head(company_domain, timeout=10, allow_redirects=True) as response:
                    head_status = response.status
                    if head_status in PASS_STATUS_CODES:
                        return True, f"Website accessible (HEAD: {head_status})"
            except aiohttp.ClientError as e:
                head_error = str(e) or "connection_error"
                # Handle large enterprise headers - pass immediately
                if "Header value is too long" in head_error or "Got more than" in head_error:
                    return True, "Website accessible (large enterprise headers detected)"
            except asyncio.TimeoutError:
                head_error = "timeout"
            except Exception as e:
                head_error = str(e) or type(e).__name__
            
            # HEAD failed or returned non-pass status - try GET as fallback
            # Many enterprise sites (Intuit, 3M) block HEAD but allow GET
            try:
                async with session.get(company_domain, timeout=10, allow_redirects=True) as response:
                    get_status = response.status
                    if get_status in PASS_STATUS_CODES:
                        return True, f"Website accessible (GET fallback: {get_status})"
                    else:
                        # Both HEAD and GET returned non-pass status
                        return False, f"Website not accessible (HEAD: {head_status}, GET: {get_status})"
            except aiohttp.ClientError as e:
                get_error = str(e) or "connection_error"
                # Handle large enterprise headers on GET too
                if "Header value is too long" in get_error or "Got more than" in get_error:
                    return True, "Website accessible (large enterprise headers detected)"
                # Both HEAD and GET failed
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: {get_error})"
            except asyncio.TimeoutError:
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: timeout)"
            except Exception as e:
                return False, f"Website inaccessible (HEAD: {head_error or head_status}, GET: {str(e) or type(e).__name__})"
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
