"""
Stage 5: Unified Company Verification
Separated from automated_checks.py for modularity.

Verification Order (MUST CHECK - strict):
1. HQ - Company Location (3-query cascade: Q1/Q2/Q3)
2. Name - Company Name (from LinkedIn title)
3. Employee Count (exact range match)
4. Website (strict domain match)
5. Description - must pass validation
6. Industry/Sub-industry pair - must be from top 3 taxonomy

NOT validated (used as input only):
- Extracted industry from LinkedIn → used for description check & refined description

Note: Role and person location verification is handled by Stage 4.
Stage 5 trusts Stage 4's verification and handles company-level checks.
"""

import requests
import asyncio
import os
import re
import json
import gzip
import unicodedata
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from urllib.parse import urlparse

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

# Load environment variables (explicit path to ensure .env is found regardless of working directory)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

# Proxy configuration for containerized validators
HTTP_PROXY_URL = os.environ.get('HTTP_PROXY')
HTTPS_PROXY_URL = os.environ.get('HTTPS_PROXY', HTTP_PROXY_URL)
PROXY_CONFIG = None
if HTTP_PROXY_URL:
    PROXY_CONFIG = {
        'http': HTTP_PROXY_URL,
        'https': HTTPS_PROXY_URL or HTTP_PROXY_URL
    }


# ========================================================================
# TAXONOMY EMBEDDINGS FOR INDUSTRY CLASSIFICATION
# ========================================================================
# Pre-computed embeddings for 725 sub-industries from INDUSTRY_TAXONOMY.
# Used to find top 3 industry/sub-industry matches for new companies.
#
# Pipeline (same as tested in classify_pipeline.py):
# 1. Stage 1: Validate miner description + Generate refined description
# 2. Stage 2: Embed refined description (Qwen3-Embedding-8B)
# 3. Stage 3: Find top 30 candidates, LLM ranks top 3
# ========================================================================

_TAXONOMY_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), 'taxonomy_embeddings.json.gz'
)
_TAXONOMY_CACHE = None  # Cached taxonomy data
EMBED_MODEL = "qwen/qwen3-embedding-8b"
CLASSIFY_LLM_MODEL = "google/gemini-2.5-flash-lite"
TOP_K_CANDIDATES = 30

# Prompts (same as tested in classify_pipeline.py)
VALIDATE_REFINE_PROMPT = """Do these describe the SAME TYPE of business? Ignore company name, location, and exact wording. Focus only on core business activities.

If YES (same type of business): Output a 50-80 word refined description focusing on products/services/target market.
If NO (completely different industries or business types): Output only: INVALID

Miner: {miner_description}

Website: {extracted_content}"""

CLASSIFY_PROMPT = """Pick top 3 most relevant options by number only.

Description: {refined_description}

Options:
{candidates_list}

Output only 3 numbers in order of relevance, e.g.: 1, 5, 12"""


def _load_taxonomy_embeddings():
    """Load pre-computed taxonomy embeddings from gzip (cached)."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE

    try:
        with gzip.open(_TAXONOMY_EMBEDDINGS_PATH, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        names = list(data.keys())
        embeddings = np.array([data[n]['embedding'] for n in names])
        industries = {n: data[n]['industries'] for n in names}

        # Build valid pairs set
        valid_pairs = set()
        for sub, ind_list in industries.items():
            for ind in ind_list:
                valid_pairs.add((ind, sub))

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # NumPy 2.x: in-place modification of computed arrays; use explicit copy to avoid
        # copy-on-write CoW issues with views. norms here is already a fresh array so this is safe.
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        _TAXONOMY_CACHE = {
            'names': names,
            'embeddings': embeddings,
            'industries': industries,
            'valid_pairs': valid_pairs
        }
        print(f"   ✅ Loaded {len(names)} taxonomy embeddings")
        return _TAXONOMY_CACHE

    except Exception as e:
        print(f"   ⚠️ Failed to load taxonomy embeddings: {e}")
        return None


def _get_embedding_sync(text: str) -> Optional[np.ndarray]:
    """Get embedding for text via OpenRouter (synchronous)."""
    if not OPENROUTER_KEY:
        print("   ⚠️ OPENROUTER_KEY not set")
        return None
    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/embeddings',
            headers={
                'Authorization': f'Bearer {OPENROUTER_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': EMBED_MODEL,
                'input': [text],
            },
            timeout=60,
            proxies=PROXY_CONFIG if PROXY_CONFIG else None
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and len(data['data']) > 0:
                return np.array(data['data'][0]['embedding'])
    except Exception as e:
        print(f"   ⚠️ Embedding API error: {e}")
    return None


def _call_llm_sync(prompt: str) -> Optional[str]:
    """Call Gemini 2.5 Flash Lite via OpenRouter (synchronous)."""
    if not OPENROUTER_KEY:
        print("   ⚠️ OPENROUTER_KEY not set")
        return None
    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': CLASSIFY_LLM_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0,
                'max_tokens': 500,
            },
            timeout=60,
            proxies=PROXY_CONFIG if PROXY_CONFIG else None
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('choices') and len(data['choices']) > 0:
                return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"   ⚠️ LLM API error: {e}")
    return None


def _find_top_candidates(query_embedding: np.ndarray, taxonomy_cache: dict, k: int = 30) -> List[dict]:
    """Find top k candidates by cosine similarity."""
    names = taxonomy_cache['names']
    embeddings = taxonomy_cache['embeddings']
    industries = taxonomy_cache['industries']

    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        return []
    query_norm = query_embedding / norm

    similarities = embeddings @ query_norm
    similarities = np.nan_to_num(similarities, nan=0.0)

    top_indices = np.argsort(similarities)[-k:][::-1]

    candidates = []
    for idx in top_indices:
        name = names[idx]
        candidates.append({
            'sub_industry': name,
            'industries': industries[name],
            'similarity': float(similarities[idx])
        })

    return candidates


def _format_candidates_for_prompt(
    candidates: List[dict],
    miner_industry: str = "",
    miner_sub_industry: str = "",
    valid_pairs: set = None
) -> Tuple[str, List[dict]]:
    """
    Format candidates as numbered list with industry/sub_industry pairs.

    If miner_industry and miner_sub_industry are provided and form a valid
    taxonomy pair, they are added to the candidate list so the LLM can
    consider them even if embeddings didn't rank them highly.
    """
    pairs = []
    for c in candidates:
        sub = c['sub_industry']
        for ind in c['industries']:
            pairs.append({'industry': ind, 'sub_industry': sub})

    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p['industry'], p['sub_industry'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    unique_pairs = unique_pairs[:40]  # Limit to 40

    # Add miner's claimed pair if valid and not already in list
    if miner_industry and miner_sub_industry and valid_pairs:
        # Check case-insensitively against valid_pairs
        miner_pair_lower = (miner_industry.lower(), miner_sub_industry.lower())
        valid_pairs_lower = {(ind.lower(), sub.lower()) for ind, sub in valid_pairs}

        if miner_pair_lower in valid_pairs_lower:
            # Check if already in list (case-insensitive)
            existing_pairs = {(p['industry'].lower(), p['sub_industry'].lower()) for p in unique_pairs}
            if miner_pair_lower not in existing_pairs:
                unique_pairs.append({'industry': miner_industry, 'sub_industry': miner_sub_industry})
                print(f"   ℹ️ Added miner's claimed pair to candidates: {miner_industry} / {miner_sub_industry}")

    prompt_str = '\n'.join([f"{i+1}. {p['industry']} / {p['sub_industry']}" for i, p in enumerate(unique_pairs)])

    return prompt_str, unique_pairs


def _clean_refined_description(response: str) -> Optional[str]:
    """Clean and validate refined description from LLM."""
    if not response:
        return None

    text = response.strip()

    # Check if INVALID
    if text.upper() == 'INVALID' or text.upper().startswith('INVALID'):
        return None

    # Remove YES/SAME prefix (LLM sometimes prefixes with "YES" or "SAME")
    text = re.sub(r'^(YES|SAME)\s*\n*', '', text, flags=re.I).strip()

    # Check minimum quality (at least 30 chars, 5 words)
    if len(text) < 30 or len(text.split()) < 5:
        return None

    return text


def _parse_classification_response(response: str, candidates_list: List[dict]) -> List[dict]:
    """Parse number-based LLM response and look up pairs from candidates."""
    if not response:
        return []

    # Extract numbers from response
    numbers = re.findall(r'\d+', response)
    if not numbers:
        return []

    # Convert to integers and filter valid range
    classifications = []
    for n in numbers[:3]:  # Take max 3
        idx = int(n) - 1  # Convert to 0-based
        if 0 <= idx < len(candidates_list):
            pair = candidates_list[idx]
            classifications.append({
                'rank': len(classifications) + 1,
                'industry': pair['industry'],
                'sub_industry': pair['sub_industry']
            })

    return classifications


async def classify_company_industry(
    miner_description: str,
    extracted_content: str,
    extracted_industry: str = "",
    company_name: str = "",
    miner_industry: str = "",
    miner_sub_industry: str = ""
) -> Tuple[List[dict], str, str]:
    """
    Classify company into top 3 industry/sub-industry pairs using embeddings.

    Full pipeline (same as tested in classify_pipeline.py):
    1. Stage 1: Validate miner description + Generate refined description
    2. Stage 2: Embed refined description
    3. Stage 3: Find top 30 candidates, LLM ranks top 3

    If miner_industry and miner_sub_industry are provided, they are added to
    the candidate list so the LLM can evaluate them even if embeddings didn't
    rank them in top 30. This gives miners a fair chance when their claim is
    valid but uses terminology that doesn't match embedding similarity.

    Args:
        miner_description: Miner's claimed company description
        extracted_content: Extracted content from LinkedIn/website
        extracted_industry: Extracted industry from LinkedIn (optional)
        company_name: Company name for context
        miner_industry: Miner's claimed industry (optional)
        miner_sub_industry: Miner's claimed sub_industry (optional)

    Returns:
        (classifications, refined_description, error_message)
        classifications: List of {rank, industry, sub_industry} dicts
        refined_description: The generated refined description
        error_message: Empty string if success, error message if failed
    """
    # Load taxonomy embeddings
    taxonomy = _load_taxonomy_embeddings()
    if not taxonomy:
        return [], "", "taxonomy_embeddings_not_loaded"

    # Check inputs
    if not miner_description or not extracted_content:
        return [], "", "missing_description"

    # ========================================================================
    # Stage 1: Validate + Generate refined description
    # ========================================================================
    print(f"   🔍 CLASSIFY Stage 1: Validating and refining description...")

    # Add extracted industry context if available
    content_with_industry = extracted_content
    if extracted_industry:
        content_with_industry = f"[Industry: {extracted_industry}] {extracted_content}"

    prompt1 = VALIDATE_REFINE_PROMPT.format(
        miner_description=miner_description[:2000],
        extracted_content=content_with_industry[:3000]
    )

    response1 = _call_llm_sync(prompt1)
    if not response1:
        return [], "", "stage1_llm_failed"

    # Clean and validate refined description
    refined = _clean_refined_description(response1)
    if not refined:
        print(f"   ⚠️ CLASSIFY Stage 1: Description validation failed (INVALID or low quality)")
        return [], "", "stage1_invalid_description"

    print(f"   ✅ CLASSIFY Stage 1: Refined description generated ({len(refined)} chars)")

    # ========================================================================
    # Stage 2: Embed refined description
    # ========================================================================
    print(f"   🔍 CLASSIFY Stage 2: Embedding refined description...")

    query_emb = _get_embedding_sync(refined)
    if query_emb is None:
        return [], refined, "stage2_embedding_failed"

    # Find top 30 candidates
    candidates = _find_top_candidates(query_emb, taxonomy, k=TOP_K_CANDIDATES)
    if not candidates:
        return [], refined, "stage2_no_candidates"

    print(f"   ✅ CLASSIFY Stage 2: Found {len(candidates)} candidates, top similarity: {candidates[0]['similarity']:.3f}")

    # ========================================================================
    # Stage 3: LLM ranks top 3
    # ========================================================================
    print(f"   🔍 CLASSIFY Stage 3: LLM ranking top 3...")

    # Include miner's claimed pair in candidates so LLM can evaluate it
    candidates_str, candidates_list = _format_candidates_for_prompt(
        candidates,
        miner_industry=miner_industry,
        miner_sub_industry=miner_sub_industry,
        valid_pairs=taxonomy.get('valid_pairs')
    )

    # Include LinkedIn industry in description for LLM context (not saved to company table)
    desc_for_prompt = f"{refined}. LinkedIn Industry: {extracted_industry}" if extracted_industry else refined
    prompt2 = CLASSIFY_PROMPT.format(
        refined_description=desc_for_prompt,
        candidates_list=candidates_str,
    )

    response2 = _call_llm_sync(prompt2)
    if not response2:
        return [], refined, "stage3_llm_failed"

    classifications = _parse_classification_response(response2, candidates_list)

    if classifications:
        print(f"   ✅ CLASSIFY Stage 3: Top 3 = {[(c['industry'], c['sub_industry']) for c in classifications]}")
        return classifications, refined, ""
    else:
        return [], refined, "stage3_parse_failed"


# ========================================================================
# HELPER: Accent Normalization
# ========================================================================

def normalize_accents(text: str) -> str:
    """
    Remove accents/diacritics from text for name matching.
    e.g., "José" -> "Jose", "François" -> "Francois"
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')


# ========================================================================
# HELPER: Website Extraction and Validation
# ========================================================================

def _normalize_domain(url: str) -> str:
    """Normalize URL to domain only (lowercase, no www prefix)."""
    if not url:
        return ''
    url = url.strip().lower()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ''


def _extract_website_from_snippet(text: str) -> str:
    """Extract website URL from LinkedIn snippet.

    Looks for patterns like:
    - Website: https://example.com
    - Website: www.example.com
    - Website: example.com
    """
    if not text:
        return ''

    # Pattern 1: Website: followed by full URL
    m = re.search(r'Website\s*:\s*(https?://[^\s,;|]+|www\.[^\s,;|]+)', text, re.I)
    if m:
        url = m.group(1).strip().rstrip('.')
        # Skip LinkedIn URLs
        if 'lnkd.in' not in url and 'linkedin.com' not in url:
            return url

    # Pattern 2: Website: followed by domain without protocol
    m2 = re.search(r'Website\s*:\s*([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})+(?:/[^\s,;|]*)?)', text, re.I)
    if m2:
        url = m2.group(1).strip().rstrip('.')
        if 'lnkd.in' not in url and 'linkedin.com' not in url:
            return url

    return ''


# ========================================================================
# COMPANY LOCATION & NAME VERIFICATION (3-Query Cascade)
# ========================================================================
# Validates miner-submitted company city, state, country against
# LinkedIn search results using 3 Google Search queries:
#   Q1: site:linkedin.com/company/{slug} Locations
#   Q2: "{company_name}" headquarter location {country}
#   Q3: site:linkedin.com/company/{slug} "{city}"
# All queries require exact slug match in results.
# ========================================================================

# Load geo data for city/state validation
_GEO_LOOKUP_PATH = os.path.join(os.path.dirname(__file__), '..', 'gateway', 'utils', 'geo_lookup_fast.json')
_GEO_CACHE = None

# Load English word cities (cities that are also common English words — need strict validation)
_ENGLISH_WORD_CITIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'gateway', 'utils', 'english_word_cities.txt')
_ENGLISH_WORD_CITIES = set()
try:
    with open(_ENGLISH_WORD_CITIES_PATH, 'r') as _f:
        _ENGLISH_WORD_CITIES = {line.strip().lower() for line in _f if line.strip()}
except Exception:
    pass

def _load_geo():
    """Load geo lookup data (cached)."""
    global _GEO_CACHE
    if _GEO_CACHE is not None:
        return _GEO_CACHE
    try:
        with open(_GEO_LOOKUP_PATH, 'r') as f:
            _GEO_CACHE = json.load(f)
    except Exception as e:
        print(f"   ⚠️ Could not load geo_lookup_fast.json: {e}")
        _GEO_CACHE = {}
    return _GEO_CACHE


def _build_us_city_to_states():
    """Build city -> [states] lookup from geo data."""
    geo = _load_geo()
    city_to_states = {}
    for state, cities in geo.get('us_states', {}).items():
        for city in cities:
            cl = city.lower()
            if cl not in city_to_states:
                city_to_states[cl] = []
            city_to_states[cl].append(state.title())
    return city_to_states


def _get_state_abbrev_map():
    """Get state abbreviation -> full name map."""
    geo = _load_geo()
    return {k.upper(): v.title() for k, v in geo.get('state_abbr', {}).items()}


def _get_us_states_set():
    """Get set of full US state names."""
    geo = _load_geo()
    return {v.title() for v in geo.get('state_abbr', {}).values()}


# Constants for location extraction
_FOREIGN_CITIES = {'toronto', 'vancouver', 'montreal', 'sydney', 'melbourne', 'brisbane',
                   'london', 'edinburgh', 'manchester', 'birmingham', 'dublin', 'berlin',
                   'paris', 'amsterdam', 'stockholm', 'copenhagen', 'oslo', 'tokyo', 'seoul',
                   'tel aviv', 'tel-aviv', 'singapore', 'hong kong', 'beijing', 'shanghai'}

_SKIP_WORDS = {'suite', 'floor', 'unit', 'the', 'see', 'follow', 'greater', 'about',
               'company', 'industry', 'technology', 'information', 'services', 'solutions',
               'global', 'international', 'enterprise', 'business', 'capital', 'media',
               'digital', 'network', 'group', 'partners', 'consulting', 'management',
               'inside', 'click', 'view', 'contact', 'email', 'phone', 'building'}

_MAJOR_CITIES = {
    'atlanta': 'Georgia', 'bismarck': 'North Dakota', 'boston': 'Massachusetts',
    'boulder': 'Colorado', 'brooklyn': 'New York', 'dallas': 'Texas',
    'houston': 'Texas', 'miami': 'Florida', 'nashville': 'Tennessee',
    'new haven': 'Connecticut', 'palo alto': 'California',
    'philadelphia': 'Pennsylvania', 'raleigh': 'North Carolina',
    'san diego': 'California', 'tampa': 'Florida',
    'st louis': 'Missouri', 'st. louis': 'Missouri', 'saint louis': 'Missouri',
    'new york': 'New York', 'new york city': 'New York',
    'washington dc': 'District Of Columbia', 'wilmington': 'Delaware',
    'washington d.c.': 'District Of Columbia',
}

_CITY_ABBREVS = {'sf': 'san francisco', 'nyc': 'new york city', 'la': 'los angeles', 'dc': 'washington'}

_COMMON_STATE_ABBREVS = {
    'Pa': 'Pennsylvania', 'Ca': 'California', 'Ga': 'Georgia',
    'Va': 'Virginia', 'Ma': 'Massachusetts', 'Wa': 'Washington',
    'Fl': 'Florida', 'Tx': 'Texas', 'Il': 'Illinois', 'Oh': 'Ohio',
    'Nc': 'North Carolina', 'Nj': 'New Jersey', 'Ny': 'New York',
}

# ---- Robust HQ parsing constants (ported from Company_check/Company_industry/parse_hq_to_columns.py) ----

# Primary state for ambiguous US cities (well-known business/metro association)
_HQ_US_CITY_PRIMARY_STATE = {
    'atlanta': 'georgia',
    'ashland': 'kentucky',
    'aurora': 'colorado',
    'baltimore': 'maryland',
    'boston': 'massachusetts',
    'boulder': 'colorado',
    'brooklyn': 'new york',
    'burlingame': 'california',
    'charlotte': 'north carolina',
    'cincinnati': 'ohio',
    'clearwater': 'florida',
    'collierville': 'tennessee',
    'columbus': 'ohio',
    'crystal lake': 'illinois',
    'dallas': 'texas',
    'darien': 'connecticut',
    'denver': 'colorado',
    'franklin park': 'illinois',
    'fredericksburg': 'virginia',
    'frisco': 'texas',
    'henderson': 'nevada',
    'houston': 'texas',
    'huntingdon': 'pennsylvania',
    'huntsville': 'alabama',
    'indianapolis': 'indiana',
    'jacksonville': 'florida',
    'knoxville': 'tennessee',
    'lafayette': 'louisiana',
    'las vegas': 'nevada',
    'lexington': 'kentucky',
    'littleton': 'colorado',
    'louisville': 'kentucky',
    'madison': 'wisconsin',
    'medford': 'massachusetts',
    'memphis': 'tennessee',
    'miami': 'florida',
    'midland': 'texas',
    'minneapolis': 'minnesota',
    'nashville': 'tennessee',
    'new york city': 'new york',
    'new haven': 'connecticut',
    'newport': 'rhode island',
    'orange': 'california',
    'oregon': 'ohio',
    'palo alto': 'california',
    'park city': 'utah',
    'philadelphia': 'pennsylvania',
    'phoenix': 'arizona',
    'pittsburgh': 'pennsylvania',
    'pleasanton': 'california',
    'portland': 'oregon',
    'raleigh': 'north carolina',
    'reading': 'pennsylvania',
    'robinson': 'texas',
    'rockville': 'maryland',
    'san antonio': 'texas',
    'san diego': 'california',
    'santa fe': 'new mexico',
    'santa rosa': 'california',
    'spring lake': 'new jersey',
    'spring valley': 'new york',
    'tampa': 'florida',
    'union city': 'new jersey',
    'west springfield': 'massachusetts',
    'wilmington': 'delaware',
    'woodside': 'california',
}

_HQ_US_CITY_ALIASES = {
    'new york': 'new york city',
    'nyc': 'new york city',
    'ny': 'new york city',
    'sf': 'san francisco',
    'san fransisco': 'san francisco',
    'la': 'los angeles',
    'dc': 'washington',
    'washington dc': 'washington',
    'washington d.c.': 'washington',
    'st louis': 'st. louis',
    'st. louis': 'st. louis',
    'st paul': 'st. paul',
    'st. paul': 'st. paul',
    'ft worth': 'fort worth',
    'ft. worth': 'fort worth',
    'cranberry twp': 'cranberry township',
    'o fallon': "o'fallon",
    'gulf port': 'gulfport',
    'sandy spring': 'ashton-sandy spring',
    'winston salem': 'winston-salem',
}

_HQ_STATE_TYPOS = {
    'new youk': 'new york',
    'lousiana': 'louisiana',
    'virgina': 'virginia',
    'maharshtra': 'maharashtra',
    'californa': 'california',
}

_HQ_INTL_STATE_TO_COUNTRY = {
    'ontario': 'canada', 'british columbia': 'canada', 'alberta': 'canada',
    'quebec': 'canada', 'manitoba': 'canada', 'saskatchewan': 'canada',
    'nsw': 'australia', 'new south wales': 'australia', 'victoria': 'australia',
    'queensland': 'australia', 'western australia': 'australia',
    'south australia': 'australia', 'act': 'australia', 'tasmania': 'australia',
    'maharashtra': 'india', 'haryana': 'india', 'karnataka': 'india',
    'tamil nadu': 'india', 'delhi': 'india', 'maharshtra': 'india',
    'telangana': 'india', 'uttar pradesh': 'india', 'west bengal': 'india',
    'hampshire': 'united kingdom', 'west sussex': 'united kingdom',
    'surrey': 'united kingdom', 'england': 'united kingdom',
    'scotland': 'united kingdom', 'wales': 'united kingdom',
    'lancashire': 'united kingdom', 'kent': 'united kingdom',
    'selangor': 'malaysia',
    'center': 'israel',
    'abu dhabi emirate': 'united arab emirates',
}

_HQ_KNOWN_FOREIGN_SINGLE = {
    'dubai': ('Dubai', '', 'United Arab Emirates'),
    'abu dhabi': ('Abu Dhabi', '', 'United Arab Emirates'),
    'tel': ('Tel Aviv', '', 'Israel'),
    'tel aviv': ('Tel Aviv', '', 'Israel'),
    'singapore': ('Singapore', '', 'Singapore'),
    'hong kong': ('Hong Kong', '', 'Hong Kong'),
    'london': ('London', '', 'United Kingdom'),
    'toronto': ('Toronto', 'Ontario', 'Canada'),
}

# ---- End robust HQ parsing constants ----


def _normalize_state(state_str: str) -> str:
    """Normalize state string to full state name."""
    if not state_str:
        return ''
    state_str = unicodedata.normalize('NFKD', state_str.strip()).encode('ascii', 'ignore').decode('ascii')
    abbrev_map = _get_state_abbrev_map()
    states_set = _get_us_states_set()
    if len(state_str) == 2:
        return abbrev_map.get(state_str.upper(), '')
    title = state_str.title()
    if title in states_set:
        return title
    if title in _COMMON_STATE_ABBREVS:
        return _COMMON_STATE_ABBREVS[title]
    return ''


def _validate_city_state(city: str, state: str) -> Tuple[bool, Optional[str]]:
    """Validate city-state pair against geo lookup. Returns (is_valid, matched_city_name)."""
    if not city or not state:
        return False, None
    city_lower = city.lower().strip()
    if city_lower in _SKIP_WORDS:
        return False, None

    city_to_states = _build_us_city_to_states()

    if city_lower == 'new york' and state == 'New York':
        return True, 'New York'

    variations = [
        city_lower,
        city_lower.replace('.', ''),
        city_lower.replace('st ', 'saint ').replace('st.', 'saint'),
        city_lower.replace('ft ', 'fort ').replace('ft.', 'fort'),
        city_lower.replace('mt ', 'mount ').replace('mt.', 'mount'),
        city_lower.replace(' corner', ''),
        city_lower.replace(' township', ''),
        city_lower.replace(' twp', ''),
    ]
    if city_lower.startswith('st '):
        variations.append('st. ' + city_lower[3:])
    nospace = city_lower.replace(' ', '')
    if nospace != city_lower:
        variations.append(nospace)

    for var in variations:
        var = var.strip()
        if var in city_to_states and state in city_to_states[var]:
            return True, var.title()

    # Typo tolerance: 1-char edit distance
    if state and len(city_lower) >= 5:
        geo = _load_geo()
        state_cities = geo.get('us_states', {}).get(state.lower(), [])
        for geo_city in state_cities:
            if abs(len(geo_city) - len(city_lower)) > 1:
                continue
            diffs = 0
            if len(geo_city) == len(city_lower):
                for a, b in zip(geo_city, city_lower):
                    if a != b:
                        diffs += 1
                    if diffs > 1:
                        break
                if diffs == 1:
                    return True, geo_city.title()

    if city_lower in _FOREIGN_CITIES:
        return False, None
    return False, None


def _extract_usa_location(snippet: str) -> Tuple[str, str, bool]:
    """Extract USA city/state from LinkedIn snippet. Returns (city, state, valid)."""
    if not snippet:
        return '', '', False

    state_abbrev_map = _get_state_abbrev_map()
    us_states = _get_us_states_set()
    city_to_states = _build_us_city_to_states()
    unique_cities = {c: ss[0] for c, ss in city_to_states.items() if len(ss) == 1}

    STATE_NAMES = '|'.join(sorted(us_states, key=len, reverse=True))
    STATE_ABBREVS = '|'.join(state_abbrev_map.keys())
    cp_simple = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})'

    # Normalize accented characters
    snippet = unicodedata.normalize('NFKD', snippet).encode('ascii', 'ignore').decode('ascii')

    # City abbreviations: "based in SF", "based in NYC"
    abbrev_m = re.search(r'[Bb]ased\s+in\s+(SF|NYC|LA|DC)\b', snippet)
    if abbrev_m:
        abbrev = abbrev_m.group(1).lower()
        if abbrev in _CITY_ABBREVS:
            city_full = _CITY_ABBREVS[abbrev]
            if city_full in unique_cities:
                return city_full.title(), unique_cities[city_full], True

    # "City-based" pattern
    st_based_m = re.search(r'((?:St\.?|Ft\.?|Mt\.?)\s+[A-Z][a-z]+)-based', snippet)
    if st_based_m:
        city = st_based_m.group(1).strip()
        cl = city.lower()
        for var in [cl, cl.replace('.', ''), cl.replace('st ', 'saint ').replace('st.', 'saint'),
                    cl.replace('ft ', 'fort ').replace('ft.', 'fort'), cl.replace('mt ', 'mount ').replace('mt.', 'mount')]:
            var = var.strip()
            if var in unique_cities:
                return var.title(), unique_cities[var], True

    city_based_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\s-]based', snippet)
    if city_based_m:
        city = city_based_m.group(1).strip()
        cl = city.lower()
        if cl not in _SKIP_WORDS and cl not in us_states and cl.lower() not in {'silicon valley', 'blockchain'}:
            if cl in unique_cities:
                return city.title(), unique_cities[cl], True

    # St./Ft./Mt. prefix with state
    for pattern in [
        r'[Ll]ocated\s+in\s+((?:St\.?|Ft\.?|Mt\.?)\s+[A-Z][a-z]+)\s*,\s*(' + STATE_ABBREVS + r')',
        r'[Bb]ased\s+in\s+((?:St\.?|Ft\.?|Mt\.?)\s+[A-Z][a-z]+)\s*,\s*(' + STATE_ABBREVS + r')',
        r'\bin\s+((?:St\.?|Ft\.?|Mt\.?)\s+[A-Z][a-z]+)\s*,\s*(' + STATE_ABBREVS + r')',
    ]:
        m = re.search(pattern, snippet, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
            state = _normalize_state(m.group(2).strip())
            if state:
                valid, matched = _validate_city_state(city, state)
                if valid:
                    return matched, state, True

    # Main patterns: City, State (full/abbrev) with various contexts
    patterns = [
        # City, ST ZIP, US
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})\s+\d{{5}}(?:-\d{{4}})?\s*,?\s*(?:US|USA)', 'abbrev'),
        (rf'{cp_simple}\s*,\s*({STATE_NAMES})\s+\d{{5}}(?:-\d{{4}})?\s*,?\s*(?:US|USA)', 'full'),
        # City, State followers/·
        (rf'{cp_simple}\s*,\s*({STATE_NAMES})\s+[\d,]+\s*followers', 'full'),
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})\s+[\d,]+\s*followers', 'abbrev'),
        (rf'{cp_simple}\s*,\s*({STATE_NAMES})\s*·', 'full'),
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})\s*·', 'abbrev'),
        # Headquarters
        (rf'Headquarters?:?\s*{cp_simple}\s*,\s*({STATE_NAMES})', 'full'),
        (rf'Headquarters?:?\s*{cp_simple}\s*,\s*({STATE_ABBREVS})', 'abbrev'),
        # headquartered/based/located in
        (rf'[Hh]eadquarter(?:ed|s)?\s+in\s+{cp_simple}\s*,?\s*({STATE_NAMES})', 'full'),
        (rf'[Hh]eadquarter(?:ed|s)?\s+in\s+{cp_simple}\s*,?\s*({STATE_ABBREVS})', 'abbrev'),
        (rf'[Bb]ased\s+(?:in|out\s+of)\s+{cp_simple}\s*,?\s*({STATE_NAMES})', 'full'),
        (rf'[Bb]ased\s+(?:in|out\s+of)\s+{cp_simple}\s*,?\s*({STATE_ABBREVS})', 'abbrev'),
        (rf'[Ll]ocated\s+(?:in|near)\s+{cp_simple}\s*,?\s*({STATE_NAMES})', 'full'),
        (rf'[Ll]ocated\s+(?:in|near)\s+{cp_simple}\s*,?\s*({STATE_ABBREVS})', 'abbrev'),
        # Primary. City, State
        (rf'Primary\.?\s+{cp_simple}\s*,\s*({STATE_NAMES})', 'full'),
        (rf'Primary\.?\s+{cp_simple}\s*,\s*({STATE_ABBREVS})', 'abbrev'),
        # City, ST ZIP
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})\s+\d{{5}}(?:-\d{{4}})?', 'abbrev'),
        (rf'{cp_simple}\s*,\s*({STATE_NAMES})\s+\d{{5}}(?:-\d{{4}})?', 'full'),
        # City, ST, US
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})\s*,?\s*(?:US|USA)', 'abbrev'),
        # in City, State
        (rf'\bin\s+{cp_simple}\s*,?\s*({STATE_NAMES})', 'full'),
        (rf'\bin\s+{cp_simple}\s*,?\s*({STATE_ABBREVS})', 'abbrev'),
        # Generic City, State
        (rf'{cp_simple}\s*,\s*({STATE_NAMES})(?:\s|,|$|\.)', 'full'),
        (rf'{cp_simple}\s*,\s*({STATE_ABBREVS})(?:\s|,|$|\.)', 'abbrev'),
        # City State without comma
        (rf'{cp_simple}\s+({STATE_ABBREVS})(?:\s|,|$|\.)', 'abbrev'),
    ]

    for pattern, _ in patterns:
        for m in re.finditer(pattern, snippet, re.IGNORECASE):
            city = m.group(1).strip()
            state_raw = m.group(2).strip()
            if city.replace('.', '').lower() in _SKIP_WORDS:
                continue
            state = _normalize_state(state_raw)
            if not state:
                continue
            valid, matched = _validate_city_state(city, state)
            if valid:
                return matched, state, True
            # Try dropping leading/trailing words
            words = city.split()
            if len(words) > 1:
                for i in range(1, len(words)):
                    sub = ' '.join(words[i:])
                    valid, matched = _validate_city_state(sub, state)
                    if valid:
                        return matched, state, True
                for i in range(len(words) - 1, 0, -1):
                    sub = ' '.join(words[:i])
                    valid, matched = _validate_city_state(sub, state)
                    if valid:
                        return matched, state, True

    # Unique/major city fallbacks
    for pattern in [
        r'Locations?\.?\s+(?:Primary\.?\s+)?[^·]*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'[Hh]eadquarter(?:ed|s)?\s+(?:is\s+)?in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'[Bb]ased\s+(?:in|out\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'[Ll]ocated\s+(?:in|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        r'office\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
    ]:
        for m in re.finditer(pattern, snippet):
            city = m.group(1).strip()
            cl = city.lower()
            if cl in _SKIP_WORDS or cl in _FOREIGN_CITIES:
                continue
            if cl in unique_cities:
                return city.title(), unique_cities[cl], True
            if cl in _MAJOR_CITIES:
                return city.title(), _MAJOR_CITIES[cl], True

    # Fallback: city and state both in snippet but not adjacent
    company_part = snippet.split(' | ')[0].lower() if ' | ' in snippet else ''
    found_states = []
    for state_full in sorted(us_states, key=len, reverse=True):
        if re.search(r'\b' + re.escape(state_full) + r'\b', snippet, re.IGNORECASE):
            found_states.append(state_full)
    for abbr, state_full in state_abbrev_map.items():
        if re.search(r'\b' + abbr + r'\b', snippet):
            if state_full not in found_states:
                found_states.append(state_full)

    if found_states:
        for m in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', snippet):
            c = m.group(1).strip()
            cl = c.lower()
            if cl in _SKIP_WORDS or cl in _FOREIGN_CITIES:
                continue
            if company_part and cl in company_part:
                continue
            for state in found_states:
                valid, matched = _validate_city_state(c, state)
                if valid:
                    return matched, state, True

        # State-only fallback: if we found a US state but couldn't match a city,
        # return state-only (city is optional for US validation)
        return '', found_states[0], True

    return '', '', False


def _extract_uae_location(snippet: str) -> Tuple[str, bool]:
    """Extract UAE city (Dubai/Abu Dhabi) from LinkedIn snippet. Returns (city, valid)."""
    if not snippet:
        return '', False

    # Check structured location doesn't point to non-Dubai/Abu Dhabi
    primary_loc = re.search(r'Locations?[.\s·]+Primary[.\s]+([^·]+?)(?:·|Get directions|Employees)', snippet)
    if primary_loc:
        loc_text = primary_loc.group(1).strip()
        country_match = re.search(r',\s*([A-Z]{2})\s*\.?\s*$', loc_text)
        if country_match and country_match.group(1) != 'AE':
            return '', False
        if country_match and country_match.group(1) == 'AE':
            if not re.search(r'Dubai|Abu Dhabi', loc_text, re.IGNORECASE):
                return '', False

    hq_match = re.search(r'Headquarters?:\s*([^.]+?)(?:\.|Type:|$)', snippet)
    if hq_match:
        hq_text = hq_match.group(1).strip()
        if not re.search(r'Dubai|Abu Dhabi', hq_text, re.IGNORECASE):
            if re.search(r'Sharjah|Ajman|Fujairah|Ras Al Khaimah|Jeddah|Istanbul|Delhi|London|Tokyo|Sydney|Singapore', hq_text, re.IGNORECASE):
                return '', False

    patterns = [
        r'Locations?\.?\s+(?:Primary\.?\s+)?(Dubai|Abu Dhabi)',
        r'(Dubai|Abu Dhabi)(?:\s+Emirate)?[\s,]+(?:\d+[\s,]+)?(?:AE|United Arab Emirates)\b',
        r'Headquarters?:?\s*(Dubai|Abu Dhabi)',
        r'[Hh]eadquarter(?:ed|s)?\s+in\s+(Dubai|Abu Dhabi)',
        r'[Bb]ased\s+in\s+(Dubai|Abu Dhabi)',
        r'[Ll]ocated\s+in\s+(?:\w+\s+){0,4}(Dubai|Abu Dhabi)',
        r'office\s+in\s+(Dubai|Abu Dhabi)',
        r'(Dubai|Abu Dhabi)\s*·\s*[\d,]+\s*followers',
        r'(Dubai|Abu Dhabi)\s+[\d,]+\s*followers',
    ]
    for pattern in patterns:
        m = re.search(pattern, snippet, re.IGNORECASE)
        if m:
            city = 'Dubai' if 'dubai' in m.group(1).lower() else 'Abu Dhabi'
            return city, True

    m = re.search(r'(Dubai|Abu Dhabi),\s*UAE\b', snippet)
    if m:
        city = 'Dubai' if m.group(1) == 'Dubai' else 'Abu Dhabi'
        return city, True

    return '', False


def _extract_location_from_snippet(snippet: str, region: str) -> Tuple[str, str, bool]:
    """
    Extract location from LinkedIn snippet based on region.
    Returns (city, state, valid) for USA or (city, '', valid) for UAE.
    Also handles Remote.
    """
    if not snippet:
        return '', '', False

    # Remote check - check original snippet (capital R)
    if 'Locations. Primary. Remote' in snippet or snippet.strip().startswith('Remote'):
        return 'Remote', '', True
    # Also check lowercase patterns
    sl = snippet.lower().strip()
    if sl.startswith('remote') or '. remote.' in sl:
        return 'Remote', '', True

    if region == 'USA':
        return _extract_usa_location(snippet)
    elif region == 'UAE':
        city, valid = _extract_uae_location(snippet)
        return city, '', valid
    else:
        # Try USA first, then UAE
        city, state, valid = _extract_usa_location(snippet)
        if valid:
            return city, state, valid
        city, valid = _extract_uae_location(snippet)
        return city, '', valid


def _extract_company_name_from_title(title: str) -> str:
    """Extract company name from LinkedIn search result title (e.g., 'CompanyName | LinkedIn')."""
    if not title:
        return ''
    m = re.search(r'^(.+?)\s*[|\-–—]\s*(?:LinkedIn|领英)', title)
    if m:
        name = m.group(1).strip()
        # Remove junk suffixes LinkedIn adds
        name = re.sub(
            r':\s*(Jobs|Overview|People|Life|Posts|Insights|About|Culture|Empleos|Cultura|'
            r'Jobangebote|Pekerjaan|vacatures|Mga Trabaho|Kültür|Budaya|Key Highlights|'
            r'カルチャー|Unternehmenskultur|корпоративная культура|조직문화|ثقافة الشركة|culture)$',
            '', name
        ).strip()
        return name
    if ' - ' in title:
        return title.split(' - ')[0].strip()
    return title.strip()


def _normalize_company_name(name: str) -> str:
    """Normalize company name for comparison - only clean punctuation and whitespace."""
    if not name:
        return ''
    n = name.lower().strip()
    n = re.sub(r'[.,\-&\'\"()]+', ' ', n)
    return ' '.join(n.split()).strip()


def _validate_company_name(expected: str, found: str) -> Optional[bool]:
    """Check if expected company name matches found name. Returns True/False/None."""
    if not expected or not found:
        return None
    exp = _normalize_company_name(expected)
    fnd = _normalize_company_name(found)
    if not exp or not fnd:
        return None
    if exp == fnd:
        return True
    if exp in fnd or fnd in exp:
        return True
    exp_words = set(exp.split())
    fnd_words = set(fnd.split())
    shorter, longer = (exp_words, fnd_words) if len(exp_words) <= len(fnd_words) else (fnd_words, exp_words)
    if shorter and shorter.issubset(longer):
        return True
    return False


def _check_exact_slug_match(link: str, slug: str) -> bool:
    """Check if a URL contains an exact slug match (not a longer slug)."""
    if not link or not slug:
        return False
    link_lower = link.lower()
    slug_lower = slug.lower()
    # Extract slug from URL
    m = re.search(r'linkedin\.com/company/([^/?#]+)', link_lower)
    if not m:
        return False
    link_slug = m.group(1).rstrip('.')
    # Exact match or near-match (trailing dot/dash)
    return link_slug == slug_lower or link_slug.rstrip('-') == slug_lower


def _gse_company_location_q1_sync(slug: str) -> dict:
    """Q1: site:linkedin.com/company/{slug} Locations"""
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {'query': '', 'results': [], 'error': 'SCRAPINGDOG_API_KEY not set'}

    query = f'site:linkedin.com/company/{slug} Locations'
    try:
        resp = requests.get('https://api.scrapingdog.com/google', params={
            'api_key': api_key, 'query': query, 'results': 10
        }, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)

        if resp.status_code == 200:
            raw = resp.json()
            return {'query': query, 'results': raw.get('organic_results', []), 'error': None}
        return {'query': query, 'results': [], 'error': f'Status {resp.status_code}'}
    except Exception as e:
        return {'query': query, 'results': [], 'error': str(e)}


def _gse_company_location_q2_sync(company_name: str, country: str) -> dict:
    """Q2: "{company_name}" headquarter location {country}"""
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {'query': '', 'results': [], 'error': 'SCRAPINGDOG_API_KEY not set'}

    country_keyword = 'USA' if country.upper() in ('US', 'USA', 'UNITED STATES') else country
    query = f'"{company_name}" headquarter location {country_keyword}'
    try:
        resp = requests.get('https://api.scrapingdog.com/google', params={
            'api_key': api_key, 'query': query, 'results': 10
        }, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)

        if resp.status_code == 200:
            raw = resp.json()
            return {'query': query, 'results': raw.get('organic_results', []), 'error': None}
        return {'query': query, 'results': [], 'error': f'Status {resp.status_code}'}
    except Exception as e:
        return {'query': query, 'results': [], 'error': str(e)}


def _gse_company_location_q3_sync(slug: str, city: str) -> dict:
    """Q3: site:linkedin.com/company/{slug} "{city}" """
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {'query': '', 'results': [], 'error': 'SCRAPINGDOG_API_KEY not set'}

    query = f'site:linkedin.com/company/{slug} "{city}"'
    try:
        resp = requests.get('https://api.scrapingdog.com/google', params={
            'api_key': api_key, 'query': query, 'results': 10
        }, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)

        if resp.status_code == 200:
            raw = resp.json()
            return {'query': query, 'results': raw.get('organic_results', []), 'error': None}
        return {'query': query, 'results': [], 'error': f'Status {resp.status_code}'}
    except Exception as e:
        return {'query': query, 'results': [], 'error': str(e)}


def _find_exact_slug_data(results: list, slug: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From GSE results, find the first exact slug match and return (snippet, title).
    Returns (None, None) if no exact slug match found.
    """
    for r in results:
        link = r.get('link', '')
        if _check_exact_slug_match(link, slug):
            return r.get('snippet', ''), r.get('title', '')
    return None, None


async def verify_company_location_and_name(
    slug: str,
    company_name: str,
    claimed_city: str,
    claimed_state: str,
    claimed_country: str,
) -> dict:
    """
    Run 3-query cascade to verify company location, name, and website.

    Returns dict with:
        - passed: bool
        - company_name_matched: bool or None
        - extracted_city: str
        - extracted_state: str
        - extracted_company_name: str
        - extracted_website: str
        - location_matched: bool
        - query_used: str (q1/q2/q3)
        - failure_reason: str or None
    """
    result = {
        'passed': False,
        'company_name_matched': None,
        'extracted_city': '',
        'extracted_state': '',
        'extracted_company_name': '',
        'extracted_website': '',
        'location_matched': False,
        'query_used': '',
        'failure_reason': None,
    }

    if not slug:
        result['failure_reason'] = 'no_company_linkedin_slug'
        return result

    # Determine region for extraction
    country_upper = claimed_country.upper().strip() if claimed_country else ''
    if country_upper in ('US', 'USA', 'UNITED STATES', 'UNITED STATES OF AMERICA'):
        region = 'USA'
    elif country_upper in ('AE', 'UAE', 'UNITED ARAB EMIRATES'):
        region = 'UAE'
    else:
        region = 'USA'  # Default

    # Run Q1
    print(f"      🔍 Q1: site:linkedin.com/company/{slug} Locations")
    q1_data = await asyncio.to_thread(_gse_company_location_q1_sync, slug)
    snippet, title = _find_exact_slug_data(q1_data['results'], slug)

    if snippet is not None:
        city, state, valid = _extract_location_from_snippet(snippet, region)
        company_from_title = _extract_company_name_from_title(title)
        website = _extract_website_from_snippet(snippet)
        if valid:
            result['query_used'] = 'q1'
            result['extracted_city'] = city
            result['extracted_state'] = state
            result['extracted_company_name'] = company_from_title
            result['extracted_website'] = website
            result['company_name_matched'] = _validate_company_name(company_name, company_from_title)
            # Check location match
            result['location_matched'] = _check_location_match(
                claimed_city, claimed_state, city, state, region
            )
            result['passed'] = result['location_matched']
            if not result['passed']:
                result['failure_reason'] = _get_location_mismatch_reason(
                    claimed_city, claimed_state, city, state, region
                )
            return result

    # Run Q2
    print(f"      🔍 Q2: \"{company_name}\" headquarter location {claimed_country}")
    q2_data = await asyncio.to_thread(_gse_company_location_q2_sync, company_name, claimed_country)
    snippet, title = _find_exact_slug_data(q2_data['results'], slug)

    if snippet is not None:
        city, state, valid = _extract_location_from_snippet(snippet, region)
        company_from_title = _extract_company_name_from_title(title)
        website = _extract_website_from_snippet(snippet)
        if valid:
            result['query_used'] = 'q2'
            result['extracted_city'] = city
            result['extracted_state'] = state
            result['extracted_company_name'] = company_from_title
            result['extracted_website'] = website
            result['company_name_matched'] = _validate_company_name(company_name, company_from_title)
            result['location_matched'] = _check_location_match(
                claimed_city, claimed_state, city, state, region
            )
            result['passed'] = result['location_matched']
            if not result['passed']:
                result['failure_reason'] = _get_location_mismatch_reason(
                    claimed_city, claimed_state, city, state, region
                )
            return result

    # Run Q3
    search_city = claimed_city if claimed_city and claimed_city.lower() != 'remote' else ''
    if search_city:
        print(f"      🔍 Q3: site:linkedin.com/company/{slug} \"{search_city}\"")
        q3_data = await asyncio.to_thread(_gse_company_location_q3_sync, slug, search_city)
        snippet, title = _find_exact_slug_data(q3_data['results'], slug)

        if snippet is not None:
            city, state, valid = _extract_location_from_snippet(snippet, region)
            company_from_title = _extract_company_name_from_title(title)
            website = _extract_website_from_snippet(snippet)
            if valid:
                result['query_used'] = 'q3'
                result['extracted_city'] = city
                result['extracted_state'] = state
                result['extracted_company_name'] = company_from_title
                result['extracted_website'] = website
                result['company_name_matched'] = _validate_company_name(company_name, company_from_title)
                result['location_matched'] = _check_location_match(
                    claimed_city, claimed_state, city, state, region
                )
                result['passed'] = result['location_matched']
                if not result['passed']:
                    result['failure_reason'] = _get_location_mismatch_reason(
                        claimed_city, claimed_state, city, state, region
                    )
                return result

    # No exact slug match found in any query
    result['failure_reason'] = 'no_exact_slug_match'
    return result


def _check_location_match(
    claimed_city: str, claimed_state: str,
    extracted_city: str, extracted_state: str,
    region: str
) -> bool:
    """Check if claimed location matches extracted location.

    STRICT MATCHING: Miner must claim exactly what was extracted.
    - If extracted city + state → miner must claim city + state
    - If extracted state only → miner must claim state only (no city)
    """
    # Remote: both must be Remote
    if claimed_city.lower().strip() == 'remote':
        return extracted_city == 'Remote'
    if extracted_city == 'Remote':
        return claimed_city.lower().strip() == 'remote'

    # USA: strict matching - claimed must match extracted exactly
    if region == 'USA':
        claimed_city_clean = claimed_city.lower().strip() if claimed_city else ''
        extracted_city_clean = extracted_city.lower().strip() if extracted_city else ''

        # State is always required
        if not extracted_state:
            return False

        # Check state match
        state_match = (
            _normalize_state(claimed_state) == _normalize_state(extracted_state)
            or claimed_state.lower().strip() == extracted_state.lower().strip()
        ) if claimed_state else False

        if not state_match:
            return False

        # STRICT: If extracted has city, claimed must have city (and match)
        # If extracted has no city, claimed must have no city
        if extracted_city_clean and not claimed_city_clean:
            # Extracted has city but miner didn't claim one - REJECT
            return False
        if not extracted_city_clean and claimed_city_clean:
            # Extracted has no city but miner claimed one - REJECT
            return False

        # Both have no city - state match is enough
        if not extracted_city_clean and not claimed_city_clean:
            return True

        # Both have city - must match
        city_match = claimed_city_clean == extracted_city_clean
        if not city_match:
            # Try normalized comparison (punctuation, spacing differences)
            city_match = _normalize_company_name(claimed_city) == _normalize_company_name(extracted_city)
        if not city_match:
            # Try typo tolerance: St. Louis vs Saint Louis, Ft. Worth vs Fort Worth
            variations_claimed = [
                claimed_city_clean, claimed_city_clean.replace('.', ''),
                claimed_city_clean.replace('st ', 'saint ').replace('st.', 'saint'),
                claimed_city_clean.replace('ft ', 'fort ').replace('ft.', 'fort'),
            ]
            variations_extracted = [
                extracted_city_clean, extracted_city_clean.replace('.', ''),
                extracted_city_clean.replace('st ', 'saint ').replace('st.', 'saint'),
                extracted_city_clean.replace('ft ', 'fort ').replace('ft.', 'fort'),
            ]
            for vc in variations_claimed:
                for ve in variations_extracted:
                    if vc.strip() == ve.strip():
                        city_match = True
                        break
                if city_match:
                    break

        return city_match

    # UAE: match city only (Dubai or Abu Dhabi required)
    if region == 'UAE':
        if not extracted_city:
            return False
        return claimed_city.lower().strip() == extracted_city.lower().strip()

    return False


def _get_location_mismatch_reason(
    claimed_city: str, claimed_state: str,
    extracted_city: str, extracted_state: str,
    region: str
) -> str:
    """Get descriptive mismatch reason.

    STRICT MATCHING: Miner must claim exactly what was extracted.
    """
    if claimed_city.lower().strip() == 'remote' and extracted_city != 'Remote':
        return f"remote_mismatch: Miner claimed Remote but LinkedIn shows {extracted_city}, {extracted_state}"
    if extracted_city == 'Remote' and claimed_city.lower().strip() != 'remote':
        return f"remote_mismatch: LinkedIn shows Remote but miner claimed {claimed_city}, {claimed_state}"

    if region == 'USA':
        claimed_st = _normalize_state(claimed_state) if claimed_state else ''
        extracted_st = _normalize_state(extracted_state) if extracted_state else ''
        claimed_city_clean = claimed_city.lower().strip() if claimed_city else ''
        extracted_city_clean = extracted_city.lower().strip() if extracted_city else ''

        if not extracted_st:
            return f"no_state_extracted: Could not extract state from LinkedIn"

        if claimed_st and extracted_st and claimed_st != extracted_st:
            return f"state_mismatch: Miner claimed '{claimed_st}' but LinkedIn shows '{extracted_st}'"

        # STRICT: City presence must match
        if extracted_city_clean and not claimed_city_clean:
            return f"missing_city: LinkedIn shows '{extracted_city}' but miner did not claim a city"
        if not extracted_city_clean and claimed_city_clean:
            return f"extra_city: Miner claimed '{claimed_city}' but LinkedIn only shows state '{extracted_st}'"

        if claimed_city_clean and extracted_city_clean and claimed_city_clean != extracted_city_clean:
            return f"city_mismatch: Miner claimed '{claimed_city}' but LinkedIn shows '{extracted_city}'"

    if region == 'UAE':
        if not extracted_city:
            return f"no_city_extracted: Could not extract Dubai/Abu Dhabi from LinkedIn"
        if claimed_city.lower().strip() != extracted_city.lower().strip():
            return f"city_mismatch: Miner claimed '{claimed_city}' but LinkedIn shows '{extracted_city}'"

    return f"location_mismatch: claimed=({claimed_city}, {claimed_state}) vs extracted=({extracted_city}, {extracted_state})"


# ========================================================================
# AREA-CITY MAPPINGS FOR LOCATION VERIFICATION
# ========================================================================

AREA_CITY_MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'gateway', 'utils', 'area_city_mappings.json')
_AREA_CITY_MAPPINGS_CACHE = None


def load_area_city_mappings() -> dict:
    """Load area-city mappings from JSON file (cached)."""
    global _AREA_CITY_MAPPINGS_CACHE
    if _AREA_CITY_MAPPINGS_CACHE is not None:
        return _AREA_CITY_MAPPINGS_CACHE

    try:
        with open(AREA_CITY_MAPPINGS_PATH, 'r') as f:
            data = json.load(f)
            _AREA_CITY_MAPPINGS_CACHE = data.get('mappings', {})
            return _AREA_CITY_MAPPINGS_CACHE
    except Exception as e:
        print(f"⚠️ Could not load area_city_mappings.json: {e}")
        _AREA_CITY_MAPPINGS_CACHE = {}
        return _AREA_CITY_MAPPINGS_CACHE


def normalize_area_name(area: str) -> str:
    """Normalize area name for matching.
    e.g., "Greater Seattle Area" → "seattle"
    """
    area = area.lower().strip()
    area = area.replace("greater ", "").replace(" metropolitan", "").replace(" metro", "").replace(" area", "")
    return area.strip()



# ============================================================================
# STAGE 5: UNIFIED VERIFICATION (Role, Region, Industry)
# ============================================================================
# Verifies role, region, and industry in ONE LLM call after Stage 4 passes
# Uses ScrapingDog search results + rule-based matching + LLM verification
# 
# Flow:
# 1. ScrapingDog search for ROLE (name + company + linkedin)
# 2. ScrapingDog search for REGION (company headquarters)
# 3. ScrapingDog search for INDUSTRY (what company does)
# 4. Rule-based pre-verification (deterministic matching)
# 5. LLM verification (only for fields that need it)
# 6. Early exit if role fails → skip region/industry
# 7. Early exit if region fails → skip industry
# ============================================================================



# NOTE: GeoPy geocoding code was removed - now using city-in-snippet + area mapping approach

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


def rule_based_match_employee_count(claimed: str, extracted: str) -> Tuple[bool, str]:
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



def _gse_search_industry_sync(company: str, region_hint: str = "", max_results: int = 5) -> List[Dict]:
    """
    Search for company industry information using ScrapingDog GSE.
    Simplified from _gse_search_stage5_sync - only handles industry searches.
    """
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return []
    
    # Build industry search queries
    if region_hint:
        queries = [
            f'{company} {region_hint} company industry',
            f'{company} company industry {region_hint}'
        ]
    else:
        queries = [f'{company} company industry']
    
    def gse_search(query, max_results, company_name):
        """GSE search with company verification"""
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
                results = []
                
                # Convert to standard format
                for item in data.get("organic_results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "href": item.get("link", ""),
                        "body": item.get("snippet", "")
                    })
                
                # Verify company mentioned in results
                if company_name:
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

    # Try queries
    for query in queries:
        results = gse_search(query, max_results, company)
        if results:
            return results
    
    return []


async def _gse_search_industry(company: str, region_hint: str = "", max_results: int = 5) -> List[Dict]:
    """Async wrapper for industry search."""
    try:
        return await asyncio.to_thread(
            _gse_search_industry_sync,
            company,
            region_hint,
            max_results
        )
    except Exception as e:
        print(f"⚠️ ScrapingDog industry search thread failed: {e}")
        return []

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
        
        response = requests.get(url, params=params, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)
        
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
        
        # Iterate through all results to find exact slug match and extract company name
        # This handles cases where localized results (tw.linkedin.com, jp.linkedin.com)
        # appear before the English result with a parseable title
        url_verified = False
        all_text = ""
        company_name_from_linkedin = None

        for r in organic_results:
            link = r.get("link", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")

            if not _check_exact_slug_match(link, company_slug):
                continue

            if not url_verified:
                url_verified = True
                print(f"   ✅ COMPANY LINKEDIN: URL verified - {link}")

            all_text += f" {title} {snippet}"

            # Try to extract company name from this result's title (first successful extraction wins)
            if not company_name_from_linkedin and title:
                extracted_name = None
                if "|" in title:
                    extracted_name = title.split("|")[0].strip()
                elif re.search(r'\s+[-–—]\s+LinkedIn', title):
                    extracted_name = re.split(r'\s+[-–—]\s+LinkedIn', title)[0].strip()
                elif " LinkedIn" in title and title.endswith("LinkedIn"):
                    extracted_name = title.replace(" LinkedIn", "").strip()

                if extracted_name:
                    # Remove "Overview", "About", etc.
                    for suffix in [": Overview", " - Overview", ": About", " - About", ": Jobs", " - Jobs", ": Life", " - Life"]:
                        if suffix in extracted_name:
                            extracted_name = extracted_name.replace(suffix, "").strip()
                    if extracted_name:
                        company_name_from_linkedin = extracted_name
                        result["company_name_from_linkedin"] = company_name_from_linkedin

        if not url_verified:
            result["error"] = f"No search results match expected slug '/company/{company_slug}'"
            result["company_name_match"] = False
            print(f"   ❌ COMPANY LINKEDIN: No exact slug match found for /company/{company_slug}")
            return result

        all_text = all_text.strip()
        
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
                # Try rule-based matching - extract key words
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
        
        # Fallback to first exact-slug snippet if no good description found
        if not best_description:
            for r in organic_results:
                if _check_exact_slug_match(r.get("link", ""), company_slug):
                    fallback_snippet = r.get("snippet", "").strip()
                    if fallback_snippet:
                        best_description = fallback_snippet
                        break
        
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


# ========================================================================
# EFFICIENT VALIDATE-AS-YOU-GO QUERY PIPELINE
# ========================================================================
# Query Templates (from tested Company_check pipeline):
# Q1: site:linkedin.com/company/{slug} "Industry" "Company size" "Headquarters"
# Q2: {name} linkedin company size industry headquarters
# Q3: linkedin.com/company/{slug} Industry Company size
# S4: site:linkedin.com/company/{slug} "{miner_size}" (size confirmation)
# H4: site:linkedin.com/company/{slug} "{city}" "{state}" (HQ confirmation)
# H5: site:linkedin.com/company/{slug} "{city}" "{state_abbr}" (HQ fallback)
# H7: {company_name} linkedin location (no site: restriction)
# H6: site:linkedin.com/company/{slug} "{city}" (non-ambiguous city only)
# W4: site:linkedin.com/company/{slug} "{domain}" (website confirmation)
# W5: linkedin.com/company/{slug} "{domain}" (fallback)
# W2: linkedin.com/company/{slug} "Website" (extract Website: url)
# ========================================================================

def _gse_search_sync(query: str, max_results: int = 10) -> dict:
    """Run a single GSE query via ScrapingDog."""
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {'query': query, 'results': [], 'error': 'SCRAPINGDOG_API_KEY not set'}

    try:
        resp = requests.get('https://api.scrapingdog.com/google', params={
            'api_key': api_key, 'query': query, 'results': max_results
        }, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)

        if resp.status_code == 200:
            raw = resp.json()
            results = []
            for item in raw.get('organic_results', []):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            return {'query': query, 'results': results, 'error': None}
        return {'query': query, 'results': [], 'error': f'Status {resp.status_code}'}
    except Exception as e:
        return {'query': query, 'results': [], 'error': str(e)}


def _extract_industry_from_snippet(text: str) -> str:
    """Extract industry from LinkedIn snippet."""
    if not text:
        return ''
    m = re.search(r'Industry[:\s;]+([^;.\n·]+?)(?:\s*[·;.]|\s*Company\s*size|\s*$)', text)
    if m:
        val = m.group(1).strip()
        if 2 < len(val) < 80:
            return val
    m2 = re.search(r'Industry\s+([A-Z][A-Za-z &,/\'-]+?)\s+Company\s*size', text)
    if m2:
        val = m2.group(1).strip()
        if 2 < len(val) < 80:
            return val
    return ''


def _extract_company_size_from_snippet(text: str) -> str:
    """Extract company size from LinkedIn snippet."""
    if not text:
        return ''
    patterns = [
        re.compile(r'Company\s*size[:\s;]+(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*employees?', re.I),
        re.compile(r'Company\s*size\s+(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)\s*employees?', re.I),
        re.compile(r'Company\s*size[:\s;]+(\d{1,3}(?:,\d{3})*(?:\+|\s*[-–]\s*\d{1,3}(?:,\d{3})*)?)(?:\s|$|\.)', re.I),
        re.compile(r'(\d{1,3}(?:,\d{3})*\s*[-–]\s*\d{1,3}(?:,\d{3})*)\s*employees?', re.I),
        re.compile(r'(\d{1,3}(?:,\d{3})*\+)\s*employees?', re.I),
    ]
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1).strip()
    return ''


def _extract_headquarters_from_snippet(text: str) -> str:
    """Extract headquarters location from LinkedIn snippet."""
    if not text:
        return ''
    m = re.search(r'Headquarters[:\s;]+([^.;·\n]+?)(?:\s*[.;·]|\s*Type|\s*Founded|\s*$)', text, re.I)
    if m:
        val = m.group(1).strip()
        if 2 < len(val) < 100:
            return val
    m2 = re.search(r'Headquarters\s+([A-Z][A-Za-z ,]+?)(?:\s+Type|\s+Founded|\s*$)', text)
    if m2:
        val = m2.group(1).strip().rstrip('.')
        if 2 < len(val) < 100:
            return val
    return ''


def _extract_title_company_name(title: str) -> str:
    """Extract company name from LinkedIn title like 'Name | LinkedIn'."""
    if not title:
        return ''
    name = re.sub(r'\s*[\|–—-]\s*LinkedIn.*$', '', title).strip()
    name = re.sub(r'\s*[\|–—-]\s*领英\s*$', '', name).strip()
    parts = name.split(' - ')
    if len(parts) > 1 and len(parts[0]) > 2:
        name = parts[0].strip()
    return name if name else ''


def _extract_fields_from_results(results: list, slug: str) -> dict:
    """
    Extract all fields from GSE results matching exact slug.
    Returns dict with: title_company_name, industry, company_size, headquarters, website
    """
    extracted = {
        'title_company_name': '',
        'industry': '',
        'company_size': '',
        'headquarters': '',
        'website': '',
        'exact_slug_found': False
    }

    for r in results:
        link = r.get('link', '')
        if not _check_exact_slug_match(link, slug):
            continue

        extracted['exact_slug_found'] = True
        snippet = r.get('snippet', '')
        title = r.get('title', '')
        combined = f"{title} {snippet}"

        # Extract title company name (first match wins)
        if not extracted['title_company_name'] and title:
            name = _extract_title_company_name(title)
            if name:
                extracted['title_company_name'] = name

        # Extract industry
        if not extracted['industry']:
            ind = _extract_industry_from_snippet(combined)
            if ind:
                extracted['industry'] = ind

        # Extract company size
        if not extracted['company_size']:
            size = _extract_company_size_from_snippet(combined)
            if size:
                extracted['company_size'] = size

        # Extract headquarters
        if not extracted['headquarters']:
            hq = _extract_headquarters_from_snippet(combined)
            if hq:
                extracted['headquarters'] = hq

        # Extract website
        if not extracted['website']:
            website = _extract_website_from_snippet(combined)
            if website:
                extracted['website'] = website

    return extracted


def _check_domain_in_results(results: list, slug: str, domain: str) -> bool:
    """Check if miner's domain appears in snippet for exact slug match."""
    if not domain:
        return False

    for r in results:
        link = r.get('link', '')
        if not _check_exact_slug_match(link, slug):
            continue
        snippet = r.get('snippet', '')
        title = r.get('title', '')
        combined = f"{title} {snippet}".lower()
        if domain.lower() in combined:
            return True
    return False


def _clean_hq_part(s: str) -> str:
    """Remove trailing junk like (Houston area), [CA], zip codes from HQ parts."""
    s = re.sub(r'\s*\(.*?\)\s*$', '', s).strip()
    s = re.sub(r'\s*\[.*?\]\s*$', '', s).strip()
    s = re.sub(r'\s+\d{5}.*$', '', s).strip()
    s = re.sub(r'\s+Year$', '', s).strip()
    return s


def _resolve_hq_us_city(city_raw: str, state_name: str) -> Tuple[str, bool]:
    """Resolve a city name for a given US state, applying aliases."""
    geo = _load_geo()
    us_states_map = geo.get('us_states', {})
    city_lower = city_raw.lower().strip()
    city_lookup = _HQ_US_CITY_ALIASES.get(city_lower, city_lower)
    city_lookup = re.sub(r'\s*(metro|bay area|area|region)$', '', city_lookup, flags=re.I).strip()

    cities_in_state = us_states_map.get(state_name, [])
    if city_lookup in cities_in_state:
        return city_lookup.title(), True

    # Try "St" -> "Saint" and "St." variants
    if ' st ' in city_lookup or ' st. ' in city_lookup or city_lookup.startswith(('st ', 'st. ')):
        for old, new in [('st ', 'saint '), ('st. ', 'saint '), ('st ', 'st. '), ('st. ', 'st. ')]:
            candidate = city_lookup.replace(old, new) if (' ' + old) in (' ' + city_lookup) else city_lookup
            if candidate != city_lookup and candidate in cities_in_state:
                return candidate.title(), True

    return city_raw.strip().title(), False


def _resolve_hq_state(state_raw: str) -> Tuple[str, bool]:
    """Try to resolve state_raw to a US state name. Returns (state_name, is_us)."""
    geo = _load_geo()
    us_state_names = set(geo.get('us_states', {}).keys())
    state_abbr_map = {k.upper(): v for k, v in geo.get('state_abbr', {}).items()}

    s = state_raw.strip()
    s_upper = s.upper()
    s_lower = s.lower()

    # Fix typos first
    s_lower = _HQ_STATE_TYPOS.get(s_lower, s_lower)

    if s_upper in state_abbr_map:
        return state_abbr_map[s_upper], True
    if s_lower in us_state_names:
        return s_lower, True
    return s_lower, False


def _parse_hq_to_location(hq_raw: str) -> Tuple[str, str, str, bool]:
    """
    Parse headquarters string to (city, state, country, is_foreign).
    Robust parser with US city-to-state mapping, aliases, international support.
    Ported from Company_check/Company_industry/parse_hq_to_columns.py.
    """
    if not hq_raw:
        return '', '', '', False

    raw = hq_raw.strip()

    # Remote
    if raw.lower() in ('remote', 'global - remote - online'):
        return 'Remote', '', '', False

    # Junk
    if raw.lower() in ('nationwide', 'worldwide', 'anywhere', 'officer', 'plaza'):
        return '', '', '', False
    if any(j in raw for j in ['Retail Operations', 'Services (WHS)', 'Facilities Services']):
        return '', '', '', False

    # Strip trailing country suffix like "- USA", "- US", "- United States"
    raw = re.sub(r'\s*-\s*(USA|US|United States|UK|India|Canada|Australia)\s*$', '', raw, flags=re.I).strip()

    # Strip "Metro" / "Greater" prefix
    raw = re.sub(r'^(Metro|Greater)\s+', '', raw, flags=re.I).strip()

    # Slash separator: take first part (e.g., "Los Angeles / Orange" -> "Los Angeles")
    if '/' in raw and ',' not in raw:
        raw = raw.split('/')[0].strip()

    # "Metro Area - City, ST" -> keep "City, ST"
    m_metro = re.match(r'^.+\s+-\s+(.+,.+)$', raw)
    if m_metro:
        raw = m_metro.group(1).strip()

    raw_lower = raw.lower().strip()

    # Dubai, Dubai
    if raw_lower in ('dubai, dubai', 'dubai, u'):
        return 'Dubai', '', 'United Arab Emirates', True
    # Any "X, Dubai" → normalize to Dubai (districts like Business Bay, DIFC, JLT, Marina)
    parts_check = [p.strip().lower() for p in raw.split(',')]
    if len(parts_check) >= 2 and parts_check[-1] == 'dubai':
        return 'Dubai', '', 'United Arab Emirates', True
    # Any "X, Abu Dhabi" → normalize to Abu Dhabi
    if len(parts_check) >= 2 and parts_check[-1] == 'abu dhabi':
        return 'Abu Dhabi', '', 'United Arab Emirates', True

    # Hong Kong
    if 'hong kong' in raw_lower:
        return 'Hong Kong', '', 'Hong Kong', True

    # Dublin N (postal districts)
    if re.match(r'^dublin\s+\d+$', raw_lower):
        return 'Dublin', '', 'Ireland', True

    # Known foreign single word
    if raw_lower in _HQ_KNOWN_FOREIGN_SINGLE:
        c, s, co = _HQ_KNOWN_FOREIGN_SINGLE[raw_lower]
        return c, s, co, True

    # Washington DC variants
    if re.match(r'^washington\s*d\.?c\.?', raw_lower) or raw_lower == 'washington dc metro':
        return 'Washington', 'District Of Columbia', 'United States', False

    # Split by comma
    parts = [_clean_hq_part(p) for p in raw.split(',') if p.strip()]
    parts = [p for p in parts if p]

    if len(parts) == 0:
        return '', '', '', False

    # Load geo data
    geo = _load_geo()
    us_states_map = geo.get('us_states', {})
    us_state_names = set(us_states_map.keys())
    state_abbr_map = {k.upper(): v for k, v in geo.get('state_abbr', {}).items()}
    countries_set = set(geo.get('countries', []))
    intl_cities = geo.get('cities', {})

    # Build city-to-states lookup
    us_city_to_states = {}
    for state, cities in us_states_map.items():
        for city in cities:
            us_city_to_states.setdefault(city, []).append(state)

    # ---- 3+ parts: scan from right to find city + state ----
    if len(parts) >= 3:
        for i in range(len(parts) - 1, 0, -1):
            candidate_state = parts[i].strip()
            candidate_city = parts[i - 1].strip()
            state_name, is_us = _resolve_hq_state(candidate_state)
            if is_us:
                city_display, _ = _resolve_hq_us_city(candidate_city, state_name)
                return city_display, state_name.title(), 'United States', False
            state_lower = _HQ_STATE_TYPOS.get(candidate_state.lower(), candidate_state.lower())
            if state_lower in _HQ_INTL_STATE_TO_COUNTRY:
                country = _HQ_INTL_STATE_TO_COUNTRY[state_lower]
                return candidate_city.strip().title(), '', country.title(), True
            if state_lower in countries_set:
                return candidate_city.strip().title(), '', candidate_state.strip().title(), True

    # ---- 2 parts: city, state/country ----
    if len(parts) >= 2:
        city_raw = parts[0]
        state_raw = parts[1]

        # Fix "Washington, D" or "Washington, D.C."
        if city_raw.lower().strip() == 'washington' and state_raw.strip().lower().startswith('d'):
            return 'Washington', 'District Of Columbia', 'United States', False

        # Fix reversed: "MA, Boston" -> "Boston, MA"
        if state_raw.strip().upper() not in state_abbr_map and city_raw.strip().upper() in state_abbr_map:
            city_raw, state_raw = state_raw, city_raw

        # Puerto Rico (US territory)
        if state_raw.strip().upper() == 'PR':
            return city_raw.strip().title(), 'Puerto Rico', 'United States', False

        # Try US state
        state_name, is_us = _resolve_hq_state(state_raw)
        if is_us:
            city_display, _ = _resolve_hq_us_city(city_raw, state_name)
            return city_display, state_name.title(), 'United States', False

        # International state/province
        state_lower = _HQ_STATE_TYPOS.get(state_raw.lower().strip(), state_raw.lower().strip())
        if state_lower in _HQ_INTL_STATE_TO_COUNTRY:
            country = _HQ_INTL_STATE_TO_COUNTRY[state_lower]
            return city_raw.strip().title(), '', country.title(), True

        # State is a country name
        if state_lower in countries_set:
            return city_raw.strip().title(), '', state_raw.strip().title(), True

        # Truncated state like "New" (from "New York" cut off)
        if state_raw.strip().lower() in ('new', 'north', 'south', 'west'):
            city_lower = _HQ_US_CITY_ALIASES.get(city_raw.lower().strip(), city_raw.lower().strip())
            if city_lower in _HQ_US_CITY_PRIMARY_STATE:
                best_state = _HQ_US_CITY_PRIMARY_STATE[city_lower]
                return city_lower.title(), best_state.title(), 'United States', False
            if city_lower in us_city_to_states:
                return city_lower.title(), us_city_to_states[city_lower][0].title(), 'United States', False
            return city_raw.strip().title(), '', '', False

        # Zip code as state (e.g., "Jupiter, 33458")
        if re.match(r'^\d{4,5}$', state_raw.strip()):
            city_lower = _HQ_US_CITY_ALIASES.get(city_raw.lower().strip(), city_raw.lower().strip())
            for sn, cities in us_states_map.items():
                if city_lower in cities:
                    return city_lower.title(), sn.title(), 'United States', False
            return city_raw.strip().title(), '', 'United States', False

        # Try city in international cities DB
        city_lower = city_raw.lower().strip()
        for country_name, city_list in intl_cities.items():
            if city_lower in city_list:
                return city_raw.strip().title(), '', country_name.title(), True

        # Repeated city name (e.g., "Alexandria, Alexandria")
        if city_raw.lower().strip() == state_raw.lower().strip():
            city_lower = city_raw.lower().strip()
            for sn, cities in us_states_map.items():
                if city_lower in cities:
                    return city_lower.title(), sn.title(), 'United States', False

        return city_raw.strip().title(), state_raw.strip().title(), '', False

    # ---- 1 part: single city/state name ----
    elif len(parts) == 1:
        val = parts[0].strip()
        val_lower = val.lower()

        # Known foreign
        if val_lower in _HQ_KNOWN_FOREIGN_SINGLE:
            c, s, co = _HQ_KNOWN_FOREIGN_SINGLE[val_lower]
            return c, s, co, True

        # Apply alias (e.g., "New York" -> "new york city")
        alias = _HQ_US_CITY_ALIASES.get(val_lower, val_lower)

        # US state name only (like "Delaware", "Oregon", "Wisconsin")
        # But NOT if alias mapped it to a city
        if val_lower in us_state_names and alias == val_lower:
            return '', val.title(), 'United States', False
        alias = re.sub(r'\s*(metro|bay area|area|region)$', '', alias, flags=re.I).strip()

        # Check if city exists in intl DB
        has_intl = any(alias in city_list for co, city_list in intl_cities.items() if co != 'united states')

        # Search US cities
        if alias in us_city_to_states:
            states = us_city_to_states[alias]
            if alias in _HQ_US_CITY_PRIMARY_STATE:
                best_state = _HQ_US_CITY_PRIMARY_STATE[alias]
                return alias.title(), best_state.title(), 'United States', False
            elif len(states) == 1 and not has_intl:
                return alias.title(), states[0].title(), 'United States', False
            else:
                # Multi-state or intl duplicate without explicit mapping
                return alias.title(), '', '', False

        # Search international
        for country_name, city_list in intl_cities.items():
            if val_lower in city_list:
                return val.strip().title(), '', country_name.title(), True

        return val.strip().title(), '', '', False

    return '', '', '', False


def _validate_size_match(claimed: str, extracted: str) -> Tuple[bool, str]:
    """Validate if claimed size matches extracted size."""
    if not claimed or not extracted:
        return False, "Missing size data"

    # Normalize sizes
    claimed_norm = claimed.strip().replace(',', '').lower()
    extracted_norm = extracted.strip().replace(',', '').lower()

    # Exact match
    if claimed_norm == extracted_norm:
        return True, "Exact match"

    # Use existing rule_based_match_employee_count if available
    return rule_based_match_employee_count(claimed, extracted)


_TRADEMARK_SYMBOLS_RE = re.compile(r'[®™©℠]+')


def _validate_name_match(claimed: str, extracted: str) -> Tuple[bool, str]:
    """Validate if claimed company name matches extracted (case-sensitive).
    Trademark symbols (®™©℠) are stripped from both sides before comparing."""
    if not claimed or not extracted:
        return False, "Missing name data"

    claimed_clean = ' '.join(_TRADEMARK_SYMBOLS_RE.sub('', claimed.strip()).split())
    extracted_clean = ' '.join(_TRADEMARK_SYMBOLS_RE.sub('', extracted.strip()).split())

    if claimed_clean == extracted_clean:
        return True, "Exact match"

    return False, f"Name mismatch: claimed '{claimed.strip()}' vs extracted '{extracted.strip()}'"


# ========================================================================
# WEBSITE SCRAPING FOR CLASSIFICATION
# ========================================================================
# Scrapes website for description content used in industry classification.
# Priority order: meta description > homepage extra > about page > context
# If < 2 sources have content, GSE LinkedIn fallback is triggered.
# ========================================================================

def _scrape_website_sync(url: str) -> dict:
    """Scrape a website page using ScrapingDog API."""
    api_key = os.getenv("SCRAPINGDOG_API_KEY")
    if not api_key:
        return {'success': False, 'error': 'SCRAPINGDOG_API_KEY not set', 'html': ''}

    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        url = url.rstrip('/')

        resp = requests.get('https://api.scrapingdog.com/scrape', params={
            'api_key': api_key,
            'url': url,
            'dynamic': 'false'
        }, timeout=30, proxies=PROXY_CONFIG if PROXY_CONFIG else None)

        if resp.status_code == 200:
            return {'success': True, 'html': resp.text, 'url': url, 'error': None}
        return {'success': False, 'error': f'HTTP {resp.status_code}', 'html': '', 'url': url}
    except Exception as e:
        return {'success': False, 'error': str(e), 'html': '', 'url': url}


def _extract_meta_description(html: str) -> str:
    """Extract meta description from HTML."""
    if not html:
        return ''

    # Try meta name="description"
    m = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if not m:
        m = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']', html, re.I)
    if m:
        desc = m.group(1).strip()
        if len(desc) > 30:
            return desc[:1000]

    # Try og:description
    m = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if not m:
        m = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\']', html, re.I)
    if m:
        desc = m.group(1).strip()
        if len(desc) > 30:
            return desc[:1000]

    return ''


def _extract_homepage_extra(html: str) -> str:
    """Extract additional description content from homepage."""
    if not html:
        return ''

    # Try extracting from first substantial paragraph
    # NOTE: Previous regex had nested quantifiers causing catastrophic backtracking on some HTML
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.I | re.S)
    for p in paragraphs[:10]:  # Check first 10 paragraphs
        text = re.sub(r'<[^>]+>', ' ', p)
        text = re.sub(r'\s+', ' ', text).strip()
        # Skip short or navigation-like text
        skip_words = ['cookie', 'privacy', 'terms', 'subscribe', 'newsletter', 'sign up', 'login', 'register']
        if len(text) > 80 and not any(skip in text.lower() for skip in skip_words):
            return text[:500]

    return ''


def _extract_about_description(html: str) -> str:
    """Extract description from about page HTML."""
    if not html:
        return ''

    # Try meta description first
    meta_desc = _extract_meta_description(html)
    if meta_desc and len(meta_desc) > 50:
        return meta_desc

    # Try first substantial paragraph
    paragraphs = re.findall(r'<p[^>]*>([^<]+(?:<[^/p][^>]*>[^<]*</[^p][^>]*>)*[^<]*)</p>', html, re.I | re.S)
    for p in paragraphs:
        text = re.sub(r'<[^>]+>', ' ', p)
        text = re.sub(r'\s+', ' ', text).strip()
        skip_words = ['cookie', 'privacy', 'terms', 'subscribe', 'newsletter', 'sign up']
        if len(text) > 100 and not any(skip in text.lower() for skip in skip_words):
            return text[:1000]

    return ''


async def _scrape_website_content(website_url: str) -> dict:
    """
    Scrape website for classification content.

    Returns dict with:
    - meta_description: From homepage meta tags
    - homepage_extra: Additional content from homepage
    - about_description: Content from /about page
    - sources: List of non-empty source names
    - combined_description: Combined content from all sources
    """
    result = {
        'meta_description': '',
        'homepage_extra': '',
        'about_description': '',
        'sources': [],
        'combined_description': '',
        'error': None
    }

    if not website_url:
        result['error'] = 'No website URL provided'
        return result

    # Normalize URL
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    base_url = website_url.rstrip('/')

    # Scrape homepage
    homepage_result = await asyncio.to_thread(_scrape_website_sync, base_url)

    if homepage_result.get('success'):
        html = homepage_result.get('html', '')
        result['meta_description'] = _extract_meta_description(html)
        result['homepage_extra'] = _extract_homepage_extra(html)

        if result['meta_description']:
            result['sources'].append('meta')
        if result['homepage_extra']:
            result['sources'].append('extra')
    else:
        result['error'] = f"Homepage scrape failed: {homepage_result.get('error', 'unknown')}"

    # Scrape about page
    about_url = base_url + '/about'
    about_result = await asyncio.to_thread(_scrape_website_sync, about_url)

    if about_result.get('success'):
        about_html = about_result.get('html', '')
        result['about_description'] = _extract_about_description(about_html)

        if result['about_description']:
            result['sources'].append('about')

    # Build combined description (same logic as classify_pipeline.py)
    sources_content = [s for s in [result['meta_description'], result['homepage_extra'], result['about_description']] if s]

    if len(sources_content) >= 2:
        result['combined_description'] = ' '.join(sources_content)
    elif sources_content:
        result['combined_description'] = sources_content[0]

    return result


async def _gse_linkedin_fallback(slug: str, company_name: str, domain: str = "") -> str:
    """
    GSE LinkedIn fallback - extracts description from LinkedIn search.

    Only called when website scrape returns < 2 sources.

    Query: linkedin.com/company/{slug}
    (Same as tested pipeline in scrape_linkedin_fallback.py)

    Merges snippets from:
    - Exact LinkedIn slug matches (linkedin.com/company/{slug})
    - Domain matches (link contains domain)
    """
    query = f'linkedin.com/company/{slug}'
    print(f"   🔍 GSE LinkedIn fallback: {query}")

    result = await asyncio.to_thread(_gse_search_sync, query, 10)

    if result.get('error'):
        print(f"   ⚠️ GSE LinkedIn fallback error: {result['error']}")
        return ''

    # Junk patterns to filter (same as tested pipeline)
    junk_patterns = [
        'cookie', 'privacy policy', 'terms of use', 'terms of service',
        'legal notice', 'disclaimer', 'anti-bribery', 'code of conduct',
        'continuous disclosure', 'audit committee', 'see who you know',
    ]

    # Collect and merge snippets from all matching results
    snippets = []
    slug_lower = slug.lower().rstrip('.-')
    domain_lower = domain.lower() if domain else ""

    for item in result.get('results', []):
        link = item.get('link', '').lower()
        snippet = item.get('snippet', '').strip()

        if not snippet:
            continue

        # Check for exact slug match OR domain match (same as tested pipeline)
        is_slug_match = f'linkedin.com/company/{slug_lower}' in link
        is_domain_match = domain_lower and domain_lower in link

        if not (is_slug_match or is_domain_match):
            continue

        # Filter junk
        if any(junk in snippet.lower() for junk in junk_patterns):
            continue

        # Clean the snippet
        clean = re.sub(r'\s+', ' ', snippet).strip()
        clean = re.sub(r'^.*?\|\s*LinkedIn\s*', '', clean)
        clean = clean.strip()

        if len(clean) > 20:
            snippets.append(clean)

    if snippets:
        # Merge all snippets (up to 3)
        merged = ' '.join(snippets[:3])
        print(f"   ✅ GSE LinkedIn fallback: Merged {len(snippets[:3])} snippets ({len(merged)} chars)")
        return merged[:1000]

    print(f"   ⚠️ GSE LinkedIn fallback: No description found")
    return ''


async def check_stage5_unified(lead: dict) -> Tuple[bool, dict]:
    """
    Stage 5: Efficient Validate-As-You-Go Company Verification.

    Pipeline (from tested Company_check):
    Q1 → Q2 → Q3 (merge with validate) → S4 → H4 → H5 → H7 → H6 → W4 → W5 → W2 → Classification

    Validates immediately after each query - fails fast on mismatch.

    Returns:
        (passed: bool, rejection_reason: dict or None)
    """
    # Extract lead data
    full_name = get_field(lead, "full_name") or ""
    company = get_company(lead) or ""
    claimed_role = get_role(lead) or ""
    country = lead.get("country", "").strip()
    state = lead.get("state", "").strip()
    city = lead.get("city", "").strip()
    claimed_hq_country = (lead.get("hq_country") or "").strip()
    claimed_industry = get_industry(lead) or ""
    claimed_sub_industry = lead.get("sub_industry", "") or lead.get("Sub_industry", "") or ""
    claimed_employee_count = get_employee_count(lead) or ""
    linkedin_url = get_linkedin(lead) or ""
    website = get_website(lead) or ""
    claimed_description = lead.get("description", "") or ""
    company_linkedin = lead.get("company_linkedin", "") or ""

    if not company:
        return False, {
            "stage": "Stage 5: Pre-check",
            "check_name": "check_stage5_unified",
            "message": "No company name provided",
            "failed_fields": ["company"]
        }
    # ========================================================================
    # ROLE & LOCATION: Trust Stage 4 verification
    # ========================================================================
    role_verified_by_stage4 = lead.get("role_verified", False)
    role_method = lead.get("role_method", "")
    location_verified_by_stage4 = lead.get("location_verified", False)

    if role_verified_by_stage4:
        print(f"   ✅ ROLE: Already verified in Stage 4 (method: {role_method})")
    elif role_method == "no_role_provided":
        print(f"   ⚠️ ROLE: No role provided, skipping")
    else:
        print(f"   ❌ ROLE: Not verified in Stage 4")
        return False, {
            "stage": "Stage 5: Pre-check",
            "check_name": "check_stage5_unified",
            "message": "Role was not verified in Stage 4",
            "failed_fields": ["role"]
        }

    if location_verified_by_stage4:
        print(f"   ✅ LOCATION: Already verified in Stage 4")
    else:
        print(f"   ❌ LOCATION: Not verified in Stage 4")
        return False, {
            "stage": "Stage 5: Pre-check",
            "check_name": "check_stage5_unified",
            "message": "Location was not verified in Stage 4",
            "failed_fields": ["region"]
        }

    # ========================================================================
    # CACHED COMPANY CHECK (Fast-path)
    # ========================================================================
    company_exists_in_table = lead.get("_company_exists", False)
    skip_stage5_validation = lead.get("_skip_stage5_validation", False)
    validate_employee_count_only = lead.get("_validate_employee_count_only", False)

    # CASE 1: Fresh data (<30 days) - skip all
    if company_exists_in_table and skip_stage5_validation:
        print(f"   ⚡ CACHED COMPANY (FRESH): Skipping Stage 5 (data <30 days old)")
        lead["stage5_skipped"] = True
        lead["stage5_skip_reason"] = "company_data_fresh"
        return True, None

    # ========================================================================
    # EXTRACT COMPANY LINKEDIN SLUG
    # ========================================================================
    slug = None
    if company_linkedin:
        slug_match = re.search(r'linkedin\.com/company/([^/?#]+)', company_linkedin.lower())
        if slug_match:
            slug = slug_match.group(1).rstrip('.-')

    if not slug:
        return False, {
            "stage": "Stage 5: Pre-check",
            "check_name": "check_stage5_unified",
            "message": "Could not extract company LinkedIn slug",
            "failed_fields": ["company_linkedin"]
        }

    print(f"   🔍 STAGE 5: Starting efficient validate-as-you-go pipeline for '{company}' (slug: {slug})")

    # Normalize miner's domain for website validation
    miner_domain = _normalize_domain(website) if website else ""

    # ========================================================================
    # CASE 2: Stale data (>30 days) - validate employee count only
    # ========================================================================
    if company_exists_in_table and validate_employee_count_only:
        print(f"   ⚡ CACHED COMPANY (STALE): Validating employee count only")

        if not claimed_employee_count:
            return False, {
                "stage": "Stage 5: Employee Count",
                "check_name": "check_stage5_unified",
                "message": "No employee count provided",
                "failed_fields": ["employee_count"]
            }

        # Run S4: size confirmation query
        s4_query = f'site:linkedin.com/company/{slug} "{claimed_employee_count}"'
        print(f"   🔍 S4: {s4_query}")
        s4_result = await asyncio.to_thread(_gse_search_sync, s4_query, 10)

        if s4_result.get('error'):
            print(f"   ⚠️ S4 error: {s4_result['error']}")

        # Check if miner's size appears in exact slug match
        size_confirmed = False
        for r in s4_result.get('results', []):
            if _check_exact_slug_match(r.get('link', ''), slug):
                combined = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if claimed_employee_count.lower().replace(',', '') in combined.replace(',', ''):
                    size_confirmed = True
                    break

        if size_confirmed:
            print(f"   ✅ S4: Employee count '{claimed_employee_count}' confirmed")
            lead["_update_employee_count"] = True
            lead["_new_employee_count"] = claimed_employee_count
            return True, None
        else:
            print(f"   ❌ S4: Employee count '{claimed_employee_count}' not confirmed")
            return False, {
                "stage": "Stage 5: Employee Count",
                "check_name": "check_stage5_unified",
                "message": f"Employee count '{claimed_employee_count}' not found in LinkedIn",
                "failed_fields": ["employee_count"]
            }

    # ========================================================================
    # CASE 3: NEW COMPANY - Full validate-as-you-go pipeline
    # ========================================================================
    print(f"   📝 NEW COMPANY: Running full validation pipeline")

    # Merged fields (filled by Q1 → Q2 → Q3)
    merged = {
        'title_company_name': '',
        'industry': '',
        'company_size': '',
        'headquarters': '',
        'website': ''
    }
    sources = {}

    # ========================================================================
    # Q1: site:linkedin.com/company/{slug} "Industry" "Company size" "Headquarters"
    # ========================================================================
    q1_query = f'site:linkedin.com/company/{slug} "Industry" "Company size" "Headquarters"'
    print(f"   🔍 Q1: {q1_query}")
    q1_result = await asyncio.to_thread(_gse_search_sync, q1_query, 10)

    if q1_result.get('error'):
        print(f"   ⚠️ Q1 error: {q1_result['error']}")
    else:
        extracted = _extract_fields_from_results(q1_result.get('results', []), slug)

        if extracted['exact_slug_found']:
            # Fill merged fields
            for field in ['title_company_name', 'industry', 'company_size', 'headquarters', 'website']:
                if extracted.get(field) and not merged.get(field):
                    merged[field] = extracted[field]
                    sources[field] = 'q1'

            # VALIDATE IMMEDIATELY: Company name
            if merged['title_company_name']:
                name_match, name_reason = _validate_name_match(company, merged['title_company_name'])
                if not name_match:
                    print(f"   ❌ Q1 NAME MISMATCH: {name_reason}")
                    return False, {
                        "stage": "Stage 5: Company Name",
                        "check_name": "check_stage5_unified",
                        "message": name_reason,
                        "failed_fields": ["company_name"],
                        "claimed": company,
                        "extracted": merged['title_company_name']
                    }
                print(f"   ✅ Q1 NAME: '{merged['title_company_name']}' matches")

            # VALIDATE IMMEDIATELY: Employee count (if extracted)
            if merged['company_size'] and claimed_employee_count:
                size_match, size_reason = _validate_size_match(claimed_employee_count, merged['company_size'])
                if not size_match:
                    print(f"   ❌ Q1 SIZE MISMATCH: claimed '{claimed_employee_count}' vs extracted '{merged['company_size']}'")
                    return False, {
                        "stage": "Stage 5: Employee Count",
                        "check_name": "check_stage5_unified",
                        "message": f"Size mismatch: claimed '{claimed_employee_count}' vs LinkedIn '{merged['company_size']}'",
                        "failed_fields": ["employee_count"],
                        "claimed": claimed_employee_count,
                        "extracted": merged['company_size']
                    }
                print(f"   ✅ Q1 SIZE: '{merged['company_size']}' matches")

            # VALIDATE IMMEDIATELY: HQ location (if extracted)
            if merged['headquarters']:
                ext_city, ext_state, ext_country, is_foreign = _parse_hq_to_location(merged['headquarters'])
                # Check if miner's claimed country matches (only when both have country)
                if ext_country and claimed_hq_country and ext_country.lower() != claimed_hq_country.lower():
                    print(f"   ❌ Q1 HQ COUNTRY MISMATCH: claimed '{claimed_hq_country}' vs extracted '{ext_country}'")
                    return False, {
                        "stage": "Stage 5: HQ Location",
                        "check_name": "check_stage5_unified",
                        "message": f"HQ country mismatch: claimed '{claimed_hq_country}' vs LinkedIn '{ext_country}'",
                        "failed_fields": ["hq_country"],
                        "claimed": claimed_hq_country,
                        "extracted": ext_country
                    }
                if ext_country or ext_city:
                    print(f"   ✅ Q1 HQ: {ext_city}, {ext_state}, {ext_country}")
                    lead["extracted_hq_city"] = ext_city
                    lead["extracted_hq_state"] = ext_state
                    lead["extracted_hq_country"] = ext_country

            print(f"   📊 Q1 extracted: name={bool(merged['title_company_name'])}, size={bool(merged['company_size'])}, hq={bool(merged['headquarters'])}, industry={bool(merged['industry'])}")

    # Check if we have all required fields after Q1
    all_fields_filled = all([merged['title_company_name'], merged['company_size'], merged['headquarters']])

    # ========================================================================
    # Q2: {name} linkedin company size industry headquarters (if missing fields)
    # ========================================================================
    if not all_fields_filled:
        q2_query = f'{company} linkedin company size industry headquarters'
        print(f"   🔍 Q2: {q2_query}")
        q2_result = await asyncio.to_thread(_gse_search_sync, q2_query, 10)

        if not q2_result.get('error'):
            extracted = _extract_fields_from_results(q2_result.get('results', []), slug)

            if extracted['exact_slug_found']:
                # Fill gaps only
                for field in ['title_company_name', 'industry', 'company_size', 'headquarters', 'website']:
                    if extracted.get(field) and not merged.get(field):
                        merged[field] = extracted[field]
                        sources[field] = 'q2'

                # VALIDATE NEW FIELDS
                if sources.get('title_company_name') == 'q2' and merged['title_company_name']:
                    name_match, name_reason = _validate_name_match(company, merged['title_company_name'])
                    if not name_match:
                        print(f"   ❌ Q2 NAME MISMATCH: {name_reason}")
                        return False, {
                            "stage": "Stage 5: Company Name",
                            "check_name": "check_stage5_unified",
                            "message": name_reason,
                            "failed_fields": ["company_name"]
                        }
                    print(f"   ✅ Q2 NAME: '{merged['title_company_name']}' matches")

                if sources.get('company_size') == 'q2' and merged['company_size'] and claimed_employee_count:
                    size_match, size_reason = _validate_size_match(claimed_employee_count, merged['company_size'])
                    if not size_match:
                        print(f"   ❌ Q2 SIZE MISMATCH: claimed '{claimed_employee_count}' vs extracted '{merged['company_size']}'")
                        return False, {
                            "stage": "Stage 5: Employee Count",
                            "check_name": "check_stage5_unified",
                            "message": f"Size mismatch: claimed '{claimed_employee_count}' vs LinkedIn '{merged['company_size']}'",
                            "failed_fields": ["employee_count"]
                        }
                    print(f"   ✅ Q2 SIZE: '{merged['company_size']}' matches")

                if sources.get('headquarters') == 'q2' and merged['headquarters']:
                    ext_city, ext_state, ext_country, is_foreign = _parse_hq_to_location(merged['headquarters'])
                    if ext_country and claimed_hq_country and ext_country.lower() != claimed_hq_country.lower():
                        print(f"   ❌ Q2 HQ COUNTRY MISMATCH: claimed '{claimed_hq_country}' vs extracted '{ext_country}'")
                        return False, {
                            "stage": "Stage 5: HQ Location",
                            "check_name": "check_stage5_unified",
                            "message": f"HQ country mismatch: claimed '{claimed_hq_country}' vs LinkedIn '{ext_country}'",
                            "failed_fields": ["hq_country"]
                        }
                    if ext_country or ext_city:
                        print(f"   ✅ Q2 HQ: {ext_city}, {ext_state}, {ext_country}")
                        lead["extracted_hq_city"] = ext_city
                        lead["extracted_hq_state"] = ext_state
                        lead["extracted_hq_country"] = ext_country

                print(f"   📊 Q2 filled gaps: {[k for k, v in sources.items() if v == 'q2']}")

        all_fields_filled = all([merged['title_company_name'], merged['company_size'], merged['headquarters']])

    # ========================================================================
    # Q3: linkedin.com/company/{slug} Industry Company size (if still missing)
    # ========================================================================
    if not all_fields_filled:
        q3_query = f'linkedin.com/company/{slug} Industry Company size'
        print(f"   🔍 Q3: {q3_query}")
        q3_result = await asyncio.to_thread(_gse_search_sync, q3_query, 10)

        if not q3_result.get('error'):
            extracted = _extract_fields_from_results(q3_result.get('results', []), slug)

            if extracted['exact_slug_found']:
                for field in ['title_company_name', 'industry', 'company_size', 'headquarters', 'website']:
                    if extracted.get(field) and not merged.get(field):
                        merged[field] = extracted[field]
                        sources[field] = 'q3'

                # VALIDATE NEW FIELDS (same logic as Q2)
                if sources.get('title_company_name') == 'q3' and merged['title_company_name']:
                    name_match, name_reason = _validate_name_match(company, merged['title_company_name'])
                    if not name_match:
                        print(f"   ❌ Q3 NAME MISMATCH: {name_reason}")
                        return False, {
                            "stage": "Stage 5: Company Name",
                            "check_name": "check_stage5_unified",
                            "message": name_reason,
                            "failed_fields": ["company_name"]
                        }
                    print(f"   ✅ Q3 NAME matches")

                if sources.get('company_size') == 'q3' and merged['company_size'] and claimed_employee_count:
                    size_match, size_reason = _validate_size_match(claimed_employee_count, merged['company_size'])
                    if not size_match:
                        print(f"   ❌ Q3 SIZE MISMATCH")
                        return False, {
                            "stage": "Stage 5: Employee Count",
                            "check_name": "check_stage5_unified",
                            "message": f"Size mismatch: claimed '{claimed_employee_count}' vs LinkedIn '{merged['company_size']}'",
                            "failed_fields": ["employee_count"]
                        }
                    print(f"   ✅ Q3 SIZE matches")

                if sources.get('headquarters') == 'q3' and merged['headquarters']:
                    ext_city, ext_state, ext_country, is_foreign = _parse_hq_to_location(merged['headquarters'])
                    if ext_country and claimed_hq_country and ext_country.lower() != claimed_hq_country.lower():
                        print(f"   ❌ Q3 HQ COUNTRY MISMATCH: claimed '{claimed_hq_country}' vs extracted '{ext_country}'")
                        return False, {
                            "stage": "Stage 5: HQ Location",
                            "check_name": "check_stage5_unified",
                            "message": f"HQ country mismatch: claimed '{claimed_hq_country}' vs LinkedIn '{ext_country}'",
                            "failed_fields": ["hq_country"]
                        }
                    if ext_country or ext_city:
                        print(f"   ✅ Q3 HQ: {ext_city}, {ext_state}, {ext_country}")
                        lead["extracted_hq_city"] = ext_city
                        lead["extracted_hq_state"] = ext_state
                        lead["extracted_hq_country"] = ext_country

                print(f"   📊 Q3 filled gaps: {[k for k, v in sources.items() if v == 'q3']}")

    # ========================================================================
    # S4: Size confirmation (if size still missing after Q1-Q3)
    # ========================================================================
    if not merged['company_size'] and claimed_employee_count:
        s4_query = f'site:linkedin.com/company/{slug} "{claimed_employee_count}"'
        print(f"   🔍 S4: {s4_query}")
        s4_result = await asyncio.to_thread(_gse_search_sync, s4_query, 10)

        size_confirmed = False
        for r in s4_result.get('results', []):
            if _check_exact_slug_match(r.get('link', ''), slug):
                combined = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if claimed_employee_count.lower().replace(',', '') in combined.replace(',', ''):
                    size_confirmed = True
                    merged['company_size'] = claimed_employee_count
                    sources['company_size'] = 's4'
                    break

        if size_confirmed:
            print(f"   ✅ S4: Size '{claimed_employee_count}' confirmed")
        else:
            print(f"   ❌ S4: Size '{claimed_employee_count}' not found - FAIL")
            return False, {
                "stage": "Stage 5: Employee Count",
                "check_name": "check_stage5_unified",
                "message": f"Employee count '{claimed_employee_count}' not found in LinkedIn after Q1-Q3 and S4",
                "failed_fields": ["employee_count"]
            }

    # ========================================================================
    # CHECK REQUIRED FIELDS
    # ========================================================================
    if not merged['title_company_name']:
        print(f"   ❌ No company name found after Q1-Q3")
        return False, {
            "stage": "Stage 5: Company Name",
            "check_name": "check_stage5_unified",
            "message": "Could not extract company name from LinkedIn",
            "failed_fields": ["company_name"]
        }

    if not merged['headquarters']:
        print(f"   ⚠️ No HQ found after Q1-Q3 — attempting H4/H5 confirmation")

    # ========================================================================
    # H4/H5/H7/H6: HQ extraction fallback (if HQ still missing after Q1-Q3)
    # ========================================================================
    # Same extraction pipeline as Q1-Q3 but with targeted queries.
    # H4: site:linkedin.com/company/{slug} "{city}" "{state_full_name}"
    # H5: site:linkedin.com/company/{slug} "{city}" "{state_abbreviation}"
    # H7: {company_name}+linkedin+location (no site: restriction, verify slug)
    # H6: site:linkedin.com/company/{slug} "{city}" (non-ambiguous, non-English-word only)
    # Extracts HQ from snippet → parses → sets extracted values for downstream validation.
    # Only for US leads with city and state.
    # ========================================================================
    if not lead.get("extracted_hq_city") and not lead.get("extracted_hq_state"):
        _claimed_city = (lead.get("hq_city") or "").strip()
        _claimed_state = (lead.get("hq_state") or "").strip()

        if _claimed_city and _claimed_state and _claimed_city.lower() != "remote":
            geo = _load_geo()
            state_full_to_abbr = {v.lower(): k.upper() for k, v in geo.get('state_abbr', {}).items()}
            state_abbr = state_full_to_abbr.get(_claimed_state.lower(), "")

            def _try_extract_hq_from_results(results: list) -> bool:
                """Try to extract HQ from search results. Returns True if extracted."""
                for r in results:
                    if not _check_exact_slug_match(r.get('link', ''), slug):
                        continue
                    combined = f"{r.get('title', '')} {r.get('snippet', '')}"

                    # Try "Headquarters:" extraction first
                    hq_str = _extract_headquarters_from_snippet(combined)
                    if hq_str:
                        ext_city, ext_state, ext_country, _ = _parse_hq_to_location(hq_str)
                        if ext_city or ext_state:
                            lead["extracted_hq_city"] = ext_city
                            lead["extracted_hq_state"] = ext_state
                            lead["extracted_hq_country"] = ext_country
                            return True

                    # Try broader City, State extraction
                    city, state, valid = _extract_usa_location(combined)
                    if valid and city and state:
                        lead["extracted_hq_city"] = city
                        lead["extracted_hq_state"] = state
                        lead["extracted_hq_country"] = "United States"
                        return True

                return False

            # H4: full state name
            h4_query = f'site:linkedin.com/company/{slug} "{_claimed_city}" "{_claimed_state}"'
            print(f"   🔍 H4: {h4_query}")
            h4_result = await asyncio.to_thread(_gse_search_sync, h4_query, 10)
            if _try_extract_hq_from_results(h4_result.get('results', [])):
                print(f"   ✅ H4: Extracted HQ — {lead.get('extracted_hq_city')}, {lead.get('extracted_hq_state')}")
            else:
                # H5: state abbreviation fallback
                if state_abbr:
                    h5_query = f'site:linkedin.com/company/{slug} "{_claimed_city}" "{state_abbr}"'
                    print(f"   🔍 H5: {h5_query}")
                    h5_result = await asyncio.to_thread(_gse_search_sync, h5_query, 10)
                    if _try_extract_hq_from_results(h5_result.get('results', [])):
                        print(f"   ✅ H5: Extracted HQ — {lead.get('extracted_hq_city')}, {lead.get('extracted_hq_state')}")

            # H7: company name + linkedin + location (no site: restriction)
            if not lead.get("extracted_hq_city") and not lead.get("extracted_hq_state"):
                h7_query = f'{company}+linkedin+location'
                print(f"   🔍 H7: {h7_query}")
                h7_result = await asyncio.to_thread(_gse_search_sync, h7_query, 10)
                if _try_extract_hq_from_results(h7_result.get('results', [])):
                    print(f"   ✅ H7: Extracted HQ — {lead.get('extracted_hq_city')}, {lead.get('extracted_hq_state')}")

            # H6: city-only fallback (non-ambiguous, non-English-word cities)
            if not lead.get("extracted_hq_city") and not lead.get("extracted_hq_state"):
                city_lo_check = _claimed_city.lower().strip()
                us_city_to_states = {}
                for _st, _cities in geo.get('us_states', {}).items():
                    for _c in _cities:
                        us_city_to_states.setdefault(_c.lower(), []).append(_st)

                is_ambiguous = len(us_city_to_states.get(city_lo_check, [])) > 1
                is_english_word = city_lo_check in _ENGLISH_WORD_CITIES

                if not is_ambiguous and not is_english_word and city_lo_check in us_city_to_states:
                    h6_query = f'site:linkedin.com/company/{slug} "{_claimed_city}"'
                    print(f"   🔍 H6: {h6_query}")
                    h6_result = await asyncio.to_thread(_gse_search_sync, h6_query, 10)

                    # Try standard extraction first
                    if _try_extract_hq_from_results(h6_result.get('results', [])):
                        print(f"   ✅ H6: Extracted HQ — {lead.get('extracted_hq_city')}, {lead.get('extracted_hq_state')}")
                    else:
                        # Fallback: find city in snippet, extract actual text, resolve state from geo_lookup
                        city_boundary = re.compile(r'(?<![a-zA-Z])(' + re.escape(city_lo_check) + r')(?=\s*[,.\d]|\s*$)', re.I)
                        for r in h6_result.get('results', []):
                            if _check_exact_slug_match(r.get('link', ''), slug):
                                combined = f"{r.get('title', '')} {r.get('snippet', '')}"
                                m = city_boundary.search(combined)
                                if m:
                                    snippet_city = m.group(1).strip()
                                    resolved_state = us_city_to_states[city_lo_check][0]
                                    lead["extracted_hq_city"] = snippet_city.title()
                                    lead["extracted_hq_state"] = resolved_state.title()
                                    lead["extracted_hq_country"] = "United States"
                                    print(f"   ✅ H6: City '{snippet_city}' extracted, resolved to {resolved_state} (non-ambiguous)")
                                    break

            if not lead.get("extracted_hq_city") and not lead.get("extracted_hq_state"):
                print(f"   ❌ H4/H5/H7/H6: Could not extract HQ for '{_claimed_city}, {_claimed_state}'")

    # ========================================================================
    # HQ VALIDATION: Compare miner's claimed HQ against LinkedIn-extracted HQ
    # ========================================================================
    # Miner submits hq_city, hq_state, hq_country via gateway.
    # Q1-Q3 extracted real HQ into lead["extracted_hq_city/state/country"].
    # Reject if miner claims "Remote" but LinkedIn shows real location (or vice versa).
    # ========================================================================
    claimed_hq_city = (lead.get("hq_city") or "").strip()
    claimed_hq_state = (lead.get("hq_state") or "").strip()
    claimed_hq_country = (lead.get("hq_country") or "").strip()
    extracted_hq_city = (lead.get("extracted_hq_city") or "").strip()
    extracted_hq_state = (lead.get("extracted_hq_state") or "").strip()
    extracted_hq_country = (lead.get("extracted_hq_country") or "").strip()

    miner_claims_remote = claimed_hq_city.lower() == "remote"
    linkedin_shows_remote = extracted_hq_city == "Remote"

    if extracted_hq_country or extracted_hq_city:
        # We have extracted HQ data from LinkedIn — validate against miner's claim

        if miner_claims_remote and not linkedin_shows_remote:
            print(f"   ❌ HQ MISMATCH: Miner claimed Remote but LinkedIn shows {extracted_hq_city}, {extracted_hq_state}, {extracted_hq_country}")
            return False, {
                "stage": "Stage 5: HQ Location",
                "check_name": "check_stage5_unified",
                "message": f"HQ mismatch: Miner claimed Remote but LinkedIn shows {extracted_hq_city}, {extracted_hq_state}, {extracted_hq_country}",
                "failed_fields": ["hq_city"],
                "claimed": "Remote",
                "extracted": f"{extracted_hq_city}, {extracted_hq_state}, {extracted_hq_country}"
            }

        if linkedin_shows_remote and not miner_claims_remote:
            print(f"   ❌ HQ MISMATCH: LinkedIn shows Remote but miner claimed {claimed_hq_city}, {claimed_hq_state}, {claimed_hq_country}")
            return False, {
                "stage": "Stage 5: HQ Location",
                "check_name": "check_stage5_unified",
                "message": f"HQ mismatch: LinkedIn shows Remote but miner claimed {claimed_hq_city}, {claimed_hq_state}, {claimed_hq_country}",
                "failed_fields": ["hq_city"],
                "claimed": f"{claimed_hq_city}, {claimed_hq_state}, {claimed_hq_country}",
                "extracted": "Remote"
            }

        if not miner_claims_remote and not linkedin_shows_remote:
            # Both have real locations — extracted values are the source of truth
            if extracted_hq_country:
                if not claimed_hq_country or claimed_hq_country.lower() != extracted_hq_country.lower():
                    print(f"   ❌ HQ COUNTRY MISMATCH: claimed '{claimed_hq_country}' vs LinkedIn '{extracted_hq_country}'")
                    return False, {
                        "stage": "Stage 5: HQ Location",
                        "check_name": "check_stage5_unified",
                        "message": f"HQ country mismatch: claimed '{claimed_hq_country}' vs LinkedIn '{extracted_hq_country}'",
                        "failed_fields": ["hq_country"],
                        "claimed": claimed_hq_country or "(not provided)",
                        "extracted": extracted_hq_country
                    }

            if extracted_hq_state:
                if not claimed_hq_state or claimed_hq_state.lower() != extracted_hq_state.lower():
                    print(f"   ❌ HQ STATE MISMATCH: claimed '{claimed_hq_state}' vs LinkedIn '{extracted_hq_state}'")
                    return False, {
                        "stage": "Stage 5: HQ Location",
                        "check_name": "check_stage5_unified",
                        "message": f"HQ state mismatch: claimed '{claimed_hq_state}' vs LinkedIn '{extracted_hq_state}'",
                        "failed_fields": ["hq_state"],
                        "claimed": claimed_hq_state or "(not provided)",
                        "extracted": extracted_hq_state
                    }

            if extracted_hq_city:
                if not claimed_hq_city or claimed_hq_city.lower() != extracted_hq_city.lower():
                    print(f"   ❌ HQ CITY MISMATCH: claimed '{claimed_hq_city}' vs LinkedIn '{extracted_hq_city}'")
                    return False, {
                        "stage": "Stage 5: HQ Location",
                        "check_name": "check_stage5_unified",
                        "message": f"HQ city mismatch: claimed '{claimed_hq_city}' vs LinkedIn '{extracted_hq_city}'",
                        "failed_fields": ["hq_city"],
                        "claimed": claimed_hq_city or "(not provided)",
                        "extracted": extracted_hq_city
                    }

        print(f"   ✅ HQ validation passed: claimed=({claimed_hq_city}, {claimed_hq_state}, {claimed_hq_country}), extracted=({extracted_hq_city}, {extracted_hq_state}, {extracted_hq_country})")
    else:
        print(f"   ❌ No HQ data extracted from LinkedIn — cannot verify miner's claimed HQ")
        return False, {
            "stage": "Stage 5: HQ Location",
            "check_name": "check_stage5_unified",
            "message": f"Could not extract HQ from LinkedIn to verify miner's claim ({claimed_hq_city}, {claimed_hq_state}, {claimed_hq_country})",
            "failed_fields": ["hq_city", "hq_country"]
        }

    # ========================================================================
    # W4: site:linkedin.com/company/{slug} "{domain}" (website confirmation)
    # ========================================================================
    website_confirmed = False
    website_source = None

    if miner_domain:
        w4_query = f'site:linkedin.com/company/{slug} "{miner_domain}"'
        print(f"   🔍 W4: {w4_query}")
        w4_result = await asyncio.to_thread(_gse_search_sync, w4_query, 10)

        if _check_domain_in_results(w4_result.get('results', []), slug, miner_domain):
            website_confirmed = True
            website_source = 'w4'
            print(f"   ✅ W4: Domain '{miner_domain}' confirmed")

    # ========================================================================
    # W5: linkedin.com/company/{slug} "{domain}" (fallback, no site:)
    # ========================================================================
    if not website_confirmed and miner_domain:
        w5_query = f'linkedin.com/company/{slug} "{miner_domain}"'
        print(f"   🔍 W5: {w5_query}")
        w5_result = await asyncio.to_thread(_gse_search_sync, w5_query, 10)

        if _check_domain_in_results(w5_result.get('results', []), slug, miner_domain):
            website_confirmed = True
            website_source = 'w5'
            print(f"   ✅ W5: Domain '{miner_domain}' confirmed")

    # ========================================================================
    # W2: linkedin.com/company/{slug} (check if domain appears in snippet)
    # ========================================================================
    if not website_confirmed and miner_domain:
        w2_query = f'linkedin.com/company/{slug}'
        print(f"   🔍 W2: {w2_query}")
        w2_result = await asyncio.to_thread(_gse_search_sync, w2_query, 10)

        for r in w2_result.get('results', []):
            if _check_exact_slug_match(r.get('link', ''), slug):
                combined = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                # Simply check if miner's domain appears in the snippet (works for any language)
                if miner_domain in combined:
                    website_confirmed = True
                    website_source = 'w2'
                    print(f"   ✅ W2: Domain '{miner_domain}' found in snippet")
                    break

    if not website_confirmed and miner_domain:
        print(f"   ❌ Website '{miner_domain}' not confirmed after W4-W5-W2")
        return False, {
            "stage": "Stage 5: Website",
            "check_name": "check_stage5_unified",
            "message": f"Website domain '{miner_domain}' not found in LinkedIn",
            "failed_fields": ["website"]
        }

    if website_confirmed:
        lead["website_confirmed"] = True
        lead["website_source"] = website_source

    # ========================================================================
    # STORE EXTRACTION RESULTS
    # ========================================================================
    lead["stage5_extracted_name"] = merged['title_company_name']
    lead["stage5_extracted_size"] = merged['company_size']
    lead["stage5_extracted_hq"] = merged['headquarters']
    lead["stage5_extracted_industry"] = merged['industry']
    lead["stage5_sources"] = sources

    # Set match flags (all True since validation passed to reach here)
    lead["stage5_name_match"] = True
    lead["stage5_size_match"] = True
    lead["stage5_hq_match"] = True
    lead["stage5_industry_match"] = True

    # ========================================================================
    # CLASSIFICATION PIPELINE (For new companies)
    # ========================================================================
    # Uses website content (not LinkedIn description) for industry classification.
    # Scrapes: meta description, homepage extra, about page
    # If < 2 sources have content → GSE LinkedIn fallback
    # ========================================================================
    print(f"   📝 NEW COMPANY: Running classification pipeline...")

    industry_top3 = {"industry_match1": "", "industry_match2": "", "industry_match3": ""}
    sub_industry_top3 = {"sub_industry_match1": "", "sub_industry_match2": "", "sub_industry_match3": ""}
    refined_description = ""
    extracted_industry = merged.get('industry', '')

    # ========================================================================
    # STEP 1: Scrape website for classification content
    # ========================================================================
    website_content = None
    extracted_content = ""
    content_sources = []

    if website:
        print(f"   🌐 SCRAPE: Scraping website '{website}'...")
        website_content = await _scrape_website_content(website)

        content_sources = website_content.get('sources', [])
        extracted_content = website_content.get('combined_description', '')

        if content_sources:
            print(f"   ✅ SCRAPE: Found {len(content_sources)} sources: {content_sources}")
            if extracted_content:
                print(f"   📝 SCRAPE: Combined description ({len(extracted_content)} chars)")
        else:
            print(f"   ⚠️ SCRAPE: No content sources found")
            if website_content.get('error'):
                print(f"      Error: {website_content['error']}")

    # ========================================================================
    # STEP 2: GSE LinkedIn fallback (only if < 2 sources from website)
    # ========================================================================
    if len(content_sources) < 2:
        print(f"   📍 len(sources) = {len(content_sources)} < 2 → Running GSE LinkedIn fallback")
        linkedin_fallback_desc = await _gse_linkedin_fallback(slug, company, miner_domain)

        if linkedin_fallback_desc:
            # Add LinkedIn description to extracted content
            if extracted_content:
                extracted_content = f"{extracted_content} {linkedin_fallback_desc}"
            else:
                extracted_content = linkedin_fallback_desc
            content_sources.append('linkedin_gse')
            print(f"   ✅ GSE LinkedIn fallback added to sources")
    else:
        print(f"   📍 len(sources) = {len(content_sources)} >= 2 → Skipping GSE LinkedIn fallback")

    # Store website content on lead
    lead["website_meta_description"] = website_content.get('meta_description', '') if website_content else ''
    lead["website_homepage_extra"] = website_content.get('homepage_extra', '') if website_content else ''
    lead["website_about_description"] = website_content.get('about_description', '') if website_content else ''
    lead["classification_sources"] = content_sources

    # ========================================================================
    # STEP 3: Run 3-stage classification pipeline
    # ========================================================================
    if claimed_description and extracted_content:
        print(f"   🔍 CLASSIFY: Running 3-stage classification pipeline...")
        try:
            classifications, refined_description, classify_error = await classify_company_industry(
                miner_description=claimed_description,
                extracted_content=extracted_content,
                extracted_industry=extracted_industry,
                company_name=company,
                miner_industry=claimed_industry,
                miner_sub_industry=claimed_sub_industry
            )

            if classifications and len(classifications) >= 1:
                for i, c in enumerate(classifications[:3], 1):
                    industry_top3[f"industry_match{i}"] = c['industry']
                    sub_industry_top3[f"sub_industry_match{i}"] = c['sub_industry']
                print(f"   ✅ CLASSIFY: Top 3 = {[(c['industry'], c['sub_industry']) for c in classifications[:3]]}")

                # Validate miner's claimed pair is in top 3
                if claimed_industry and claimed_sub_industry:
                    claimed_pair = (claimed_industry.lower(), claimed_sub_industry.lower())
                    top3_pairs = [(c['industry'].lower(), c['sub_industry'].lower()) for c in classifications[:3]]
                    if claimed_pair not in top3_pairs:
                        print(f"   ❌ CLASSIFY: Miner claimed pair {claimed_pair} not in top 3 - REJECT but store company")
                        # Store company with correct top 3 for future validation
                        return False, {
                            "stage": "Stage 5: Industry",
                            "check_name": "check_stage5_unified",
                            "message": f"Claimed industry/sub-industry pair not in top 3 classifications",
                            "failed_fields": ["industry", "sub_industry"],
                            "claimed": f"{claimed_industry} / {claimed_sub_industry}",
                            "top3": top3_pairs,
                            # Include company data so gateway can still insert it
                            "company_table_action": "insert",
                            "company_refined_description": refined_description,
                            "company_industry_top3": industry_top3,
                            "company_sub_industry_top3": sub_industry_top3,
                            "company_verified_employee_count": claimed_employee_count
                        }
                    print(f"   ✅ CLASSIFY: Miner claimed pair matches top 3")
            else:
                # REJECT if miner description is invalid (doesn't match website)
                if classify_error == "stage1_invalid_description":
                    print(f"   ❌ CLASSIFY: Miner description does not match website content - REJECT")
                    return False, {
                        "stage": "Stage 5: Description",
                        "check_name": "check_stage5_unified",
                        "message": "Miner description does not match website content (INVALID)",
                        "failed_fields": ["description"]
                    }
                print(f"   ❌ CLASSIFY: Failed ({classify_error}) - REJECT")
                return False, {
                    "stage": "Stage 5: Classification",
                    "check_name": "check_stage5_unified",
                    "message": f"Classification failed: {classify_error}",
                    "failed_fields": ["industry", "sub_industry"]
                }
        except Exception as e:
            print(f"   ❌ CLASSIFY: Exception - {e} - REJECT")
            return False, {
                "stage": "Stage 5: Classification",
                "check_name": "check_stage5_unified",
                "message": f"Classification failed with exception: {e}",
                "failed_fields": ["industry", "sub_industry"]
            }
    elif extracted_content:
        # No miner description but have extracted content from website/GSE
        print(f"   ⚠️ No miner description, using extracted content for classification...")
        try:
            classifications, refined_description, classify_error = await classify_company_industry(
                miner_description=extracted_content,  # Use extracted as miner description
                extracted_content=extracted_content,
                extracted_industry=extracted_industry,
                company_name=company,
                miner_industry=claimed_industry,
                miner_sub_industry=claimed_sub_industry
            )
            if classifications and len(classifications) >= 1:
                for i, c in enumerate(classifications[:3], 1):
                    industry_top3[f"industry_match{i}"] = c['industry']
                    sub_industry_top3[f"sub_industry_match{i}"] = c['sub_industry']
                print(f"   ✅ CLASSIFY: Top 3 stored")
            else:
                print(f"   ❌ CLASSIFY: No classifications returned - REJECT")
                return False, {
                    "stage": "Stage 5: Classification",
                    "check_name": "check_stage5_unified",
                    "message": "Classification failed: no results returned",
                    "failed_fields": ["industry", "sub_industry"]
                }
        except Exception as e:
            print(f"   ❌ CLASSIFY: Exception - {e} - REJECT")
            return False, {
                "stage": "Stage 5: Classification",
                "check_name": "check_stage5_unified",
                "message": f"Classification failed with exception: {e}",
                "failed_fields": ["industry", "sub_industry"]
            }
    else:
        print(f"   ❌ No descriptions available - REJECT")
        return False, {
            "stage": "Stage 5: Classification",
            "check_name": "check_stage5_unified",
            "message": "No description available for classification (miner description and website content both empty)",
            "failed_fields": ["description"]
        }

    # Store for gateway to insert (automated_checks.py reads these keys)
    lead["_insert_new_company"] = True
    lead["_company_refined_description"] = refined_description
    lead["_company_industry_top3"] = industry_top3
    lead["_company_sub_industry_top3"] = sub_industry_top3
    lead["_company_verified_employee_count"] = claimed_employee_count

    print(f"   ✅ STAGE 5 PASSED: All validations complete")
    return True, None
