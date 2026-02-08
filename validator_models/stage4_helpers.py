"""
Stage 4 Helper Functions
========================
Helper functions for Stage 4 person verification.

Contains:
- Location extraction and matching (extract_location_from_text, check_locations_match)
- Q3 location fallback query (check_q3_location_fallback)
- Role extraction and matching (extract_role_from_result, validate_role_rule_based)
- Role LLM verification (validate_role_with_llm)
- Name/Company matching helpers
- Area mapping utilities (is_city_in_area_approved, is_area_in_mappings)

Usage:
    from validator_models.stage4_helpers import (
        extract_location_from_text,
        check_locations_match,
        check_q3_location_fallback,
    )
"""

import re
import json
import time
import unicodedata
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load JSON config files from gateway/utils folder
VALIDATOR_PATH = Path(__file__).parent
PROJECT_ROOT = VALIDATOR_PATH.parent
GATEWAY_UTILS_PATH = PROJECT_ROOT / 'gateway' / 'utils'

# Area city mappings for metro area validation
AREA_MAPPINGS_PATH = GATEWAY_UTILS_PATH / 'area_city_mappings.json'
if AREA_MAPPINGS_PATH.exists():
    with open(AREA_MAPPINGS_PATH, 'r') as f:
        AREA_MAPPINGS = json.load(f).get('mappings', {})
else:
    AREA_MAPPINGS = {}


def _normalize_area_name_simple(area: str) -> str:
    """Normalize area name for comparison (used at load time).

    Note: We keep 'metropolitan' to distinguish areas like:
    - 'Greater Vancouver Metropolitan Area' (Canada)
    - 'Greater Vancouver Area' (Washington, USA)
    """
    area = area.lower().strip()
    area = area.replace("greater ", "").replace(" metro ", " ").replace(" area", "")
    # Keep 'metropolitan' but normalize ' metro ' (with spaces) to maintain distinction
    return area.strip()


def _strip_accents_simple(s: str) -> str:
    """Strip accent marks from string (used at load time)."""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# Pre-compute normalized area names and cities for O(1) lookup
AREA_NORMALIZED_CACHE = {}  # {normalized_area_name: original_area_key}
AREA_CITIES_CACHE = {}  # {original_area_key: set of normalized city names}

for _area_key, _area_data in AREA_MAPPINGS.items():
    _norm_key = _normalize_area_name_simple(_area_key)
    AREA_NORMALIZED_CACHE[_norm_key] = _area_key
    if isinstance(_area_data, dict):
        _cities = _area_data.get('cities', [])
    else:
        _cities = _area_data if isinstance(_area_data, list) else []
    AREA_CITIES_CACHE[_area_key] = {_strip_accents_simple(c.lower()) for c in _cities}

# Geo lookup for state validation
GEO_LOOKUP_PATH = GATEWAY_UTILS_PATH / 'geo_lookup_fast.json'
if GEO_LOOKUP_PATH.exists():
    with open(GEO_LOOKUP_PATH, 'r') as f:
        GEO_LOOKUP = json.load(f)
else:
    GEO_LOOKUP = {}

def _build_duplicate_city_sets():
    """Build sets of cities that need geographic validation."""
    us_city_states = {}
    for state, cities in GEO_LOOKUP.get('us_states', {}).items():
        for city in cities:
            city_lower = city.lower().strip()
            if city_lower not in us_city_states:
                us_city_states[city_lower] = []
            us_city_states[city_lower].append(state)
    us_duplicates = {city for city, states in us_city_states.items() if len(states) > 1}

    # Find US-International overlap (accent-aware)
    # e.g., US "san jose" matches intl "san josé" when accents are stripped
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    intl_all = set()
    for country, cities in GEO_LOOKUP.get('cities', {}).items():
        for city in cities:
            intl_all.add(city.lower().strip())

    us_all = set(us_city_states.keys())
    # Exact match overlap
    us_intl_overlap = us_all & intl_all
    # Accent-stripped overlap: find US cities that match intl cities when accents removed
    us_stripped = {_strip_accents(c): c for c in us_all}
    intl_stripped = {_strip_accents(c) for c in intl_all}
    for stripped, us_city in us_stripped.items():
        if stripped in intl_stripped and us_city not in us_intl_overlap:
            us_intl_overlap.add(us_city)

    return us_duplicates, us_intl_overlap

US_DUPLICATE_CITIES, US_INTL_OVERLAP_CITIES = _build_duplicate_city_sets()
AMBIGUOUS_CITIES = US_DUPLICATE_CITIES | US_INTL_OVERLAP_CITIES

ENGLISH_WORD_CITIES_PATH = GATEWAY_UTILS_PATH / 'english_word_cities.txt'
if ENGLISH_WORD_CITIES_PATH.exists():
    with open(ENGLISH_WORD_CITIES_PATH, 'r') as f:
        ENGLISH_WORD_CITIES = {line.strip().lower() for line in f if line.strip()}
else:
    ENGLISH_WORD_CITIES = set()


def is_english_word_city(city: str) -> bool:
    """
    Check if a city name requires additional validation.

    Args:
        city: The city name to check

    Returns:
        True if city is an English word that needs strict validation
    """
    if not city:
        return False
    return city.lower().strip() in ENGLISH_WORD_CITIES

# Common words filtered from city matching
INVALID_CITY_NAMES = {
    # Job titles and roles
    'researcher', 'senior', 'junior', 'associate', 'assistant',
    'executive', 'director', 'manager', 'analyst', 'developer',
    'consultant', 'specialist', 'coordinator', 'administrator',
    'president', 'founder', 'partner', 'principal', 'lead', 'head', 'chief',
    'supervisor', 'advisor', 'adviser', 'trainer', 'instructor', 'teacher',
    'professor', 'lecturer', 'scientist', 'architect', 'designer', 'writer',
    'editor', 'producer', 'creator', 'builder', 'maker', 'planner', 'strategist',
    'accountant', 'auditor', 'lawyer', 'attorney', 'counsel', 'paralegal',
    'nurse', 'doctor', 'physician', 'therapist', 'technician', 'mechanic',
    'operator', 'pilot', 'agent', 'broker', 'trader', 'buyer',
    'seller', 'vendor', 'supplier', 'representative', 'ambassador', 'advocate',

    # Academic and institutional
    'university', 'college', 'school', 'academy', 'institute', 'institution',
    'faculty', 'campus', 'student', 'graduate', 'undergraduate', 'alumni',
    'dean', 'chancellor', 'rector', 'fellow', 'scholar',

    # Geographic/directional (not actual cities)
    'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
    'northern', 'southern', 'eastern', 'western', 'central', 'middle',
    'upper', 'lower', 'inner', 'downtown', 'uptown', 'midtown',
    'middle east', 'far east', 'near east',  # Regions, not cities

    # Size and order descriptors
    'great', 'greater', 'little', 'big', 'small', 'large', 'medium',
    'high', 'low', 'top', 'bottom', 'first', 'second', 'third', 'fourth', 'fifth',
    'main', 'primary', 'secondary', 'minor', 'major', 'super', 'mini',

    # Time-related
    'new', 'old', 'ancient', 'modern', 'current', 'former', 'future', 'early', 'late',

    # Scope and jurisdiction
    'national', 'international', 'global', 'worldwide', 'regional', 'local',
    'state', 'county', 'municipal', 'metropolitan', 'urban', 'rural',
    'domestic', 'foreign', 'overseas', 'continental',

    # Business and organization
    'company', 'business', 'enterprise', 'corporation', 'firm', 'agency',
    'organization', 'association', 'foundation', 'charity', 'nonprofit',
    'startup', 'venture', 'holding', 'subsidiary', 'affiliate', 'branch',
    'headquarters', 'office', 'studio', 'lab', 'laboratory', 'workshop',
    'factory', 'plant', 'warehouse', 'store', 'shop', 'outlet', 'depot',

    # Team and structure
    'group', 'team', 'squad', 'crew', 'staff', 'workforce', 'personnel',
    'division', 'department', 'unit', 'section', 'segment',
    'center', 'centre', 'hub', 'base', 'station', 'post', 'site', 'location',

    # Industry terms
    'industry', 'market', 'field', 'domain', 'area', 'space', 'energy',
    'service', 'services', 'solutions', 'products', 'goods', 'supplies',
    'technology', 'technologies', 'systems', 'software', 'hardware', 'platform',
    'network', 'infrastructure', 'operations', 'logistics', 'supply',

    # Digital and modern work
    'remote', 'virtual', 'online', 'digital', 'mobile', 'cloud', 'hybrid',
    'freelance', 'contract', 'temporary', 'permanent', 'fulltime', 'parttime',

    # Descriptive adjectives
    'corporate', 'professional', 'commercial', 'industrial', 'residential',
    'private', 'public', 'independent', 'joint', 'shared', 'common',
    'general', 'specific', 'special', 'custom', 'standard', 'premium',
    'basic', 'advanced', 'expert', 'master', 'elite', 'select', 'preferred',

    # Common nouns
    'home', 'house', 'building', 'plaza', 'park', 'garden',
    'place', 'point', 'view', 'vista', 'hill', 'valley', 'ridge', 'creek',
    'river', 'lake', 'bay', 'beach', 'coast', 'shore', 'island', 'forest',
    'wood', 'woods', 'grove', 'meadow', 'field', 'farm', 'ranch', 'estate',

    # Action words that appear in titles
    'marketing', 'finance', 'accounting', 'legal', 'compliance',
    'support', 'growth', 'development', 'training', 'learning',
    'quality', 'safety', 'security', 'risk', 'audit', 'control', 'assurance',
    'strategy', 'planning', 'operations', 'production', 'manufacturing',
    'engineering', 'design', 'creative', 'content', 'media', 'communications',
    'relations', 'affairs', 'resources', 'talent', 'people', 'culture',

    # Other false positives
    'city', 'town', 'village', 'county', 'district', 'province', 'territory',
    'region', 'ward', 'borough', 'parish', 'township', 'municipality',
    'capital', 'suburban', 'exurban',

    # Foreign language equivalents (common false positives)
    'universidad', 'universidade', 'université', 'universität', 'università',  # University
    'industrie', 'industria',  # Industry
    'energie', 'energia', 'énergie',  # Energy
    'centro', 'zentrum',  # Center
    'ville', 'ciudad', 'città', 'stadt',  # City
}

# ============================================================================
# CONSTANTS
# ============================================================================

ABBREVIATIONS = {
    'sr': 'senior', 'sr.': 'senior', 'jr': 'junior', 'jr.': 'junior',
    'vp': 'vice president', 'svp': 'senior vice president',
    'evp': 'executive vice president', 'avp': 'assistant vice president',
    'ceo': 'chief executive officer', 'cfo': 'chief financial officer',
    'cto': 'chief technology officer', 'coo': 'chief operating officer',
    'cmo': 'chief marketing officer', 'cio': 'chief information officer',
    'mgr': 'manager', 'dir': 'director', 'eng': 'engineer', 'engr': 'engineer',
    'dev': 'developer', 'admin': 'administrator', 'exec': 'executive',
    'asst': 'assistant', 'assoc': 'associate', 'coord': 'coordinator',
    'rep': 'representative', 'spec': 'specialist', 'tech': 'technician',
    'acct': 'accountant', 'hr': 'human resources', 'it': 'information technology',
    'qa': 'quality assurance', 'pm': 'project manager', 'ops': 'operations',
    'mktg': 'marketing', 'svc': 'service', 'svcs': 'services',
}

INVALID_ROLE_PATTERNS = [
    'job title', 'n/a', 'na', 'none', 'unknown', 'not available', 'tbd', 'tba',
    'position', 'role', 'title', 'employee', 'staff', 'worker', 'team member'
]

US_STATES = r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s*Hampshire|New\s*Jersey|New\s*Mexico|New\s*York|North\s*Carolina|North\s*Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s*Island|South\s*Carolina|South\s*Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s*Virginia|Wisconsin|Wyoming'
US_ABBREV = r'AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY'
INDIA_STATES = r'Karnataka|Maharashtra|Tamil\s*Nadu|Telangana|Delhi|Gujarat|Rajasthan|West\s*Bengal|Uttar\s*Pradesh|Kerala|Andhra\s*Pradesh|Punjab|Haryana|Madhya\s*Pradesh|Bihar|Odisha|Jharkhand'
FRANCE_REGIONS = r"Île-de-France|Ile-de-France|Auvergne-Rhône-Alpes|Hauts-de-France|Nouvelle-Aquitaine|Occitanie|Grand Est|Provence-Alpes-Côte d'Azur|Pays de la Loire|Bretagne|Normandie|Bourgogne-Franche-Comté|Centre-Val de Loire"

CITY_EQUIVALENTS = {
    'bangalore': 'bengaluru', 'bombay': 'mumbai',
    'madras': 'chennai', 'calcutta': 'kolkata',
    'new york city': 'new york',
}

# Consolidated country aliases - used for matching country names across all functions
# Format: {canonical_name: [aliases]} - all lowercase
COUNTRY_ALIASES = {
    'united states': ['usa', 'us', 'u.s.', 'u.s.a.', 'america'],
    'united kingdom': ['uk', 'u.k.', 'britain', 'england', 'scotland', 'wales', 'great britain'],
    'united arab emirates': ['uae', 'u.a.e.', 'emirates'],
    'china': ['cn', 'prc'],
    'south korea': ['korea'],
}

LOCATION_PATTERNS = [
    (100, re.compile(rf'([A-Z][a-zA-Z\s]+,\s*(?:{US_STATES}),?\s*(?:United\s*States|USA?))', re.IGNORECASE)),
    (99, re.compile(rf'([A-Z][a-zA-Z\s]+,\s*(?:{US_STATES}),?\s*United\s*\.{{0,3}})', re.IGNORECASE)),
    (98, re.compile(rf'([A-Z][a-zA-Z\s]+,\s*({US_ABBREV}),?\s*(?:United\s*States|USA))', re.IGNORECASE)),
    (97, re.compile(rf'([A-Z][a-zA-Zéèêëàâäôöùûüç\s-]+,\s*(?:{FRANCE_REGIONS})(?:,\s*France)?)', re.IGNORECASE)),
    (96, re.compile(r'Location[:\s]+([A-Z][a-zA-Z\s,]+?)(?:\s*[·•|]|\s+\d+|\.\s)', re.IGNORECASE)),
    (95, re.compile(r'((?:Dallas[-\s]Fort\s+Worth|Miami[-\s]Fort\s+Lauderdale|Salt\s+Lake\s+City|San\s+Francisco\s+Bay)\s*(?:Area|Metroplex)?)', re.IGNORECASE)),
    (94, re.compile(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+Metropolitan\s+Area)', re.IGNORECASE)),
    (93, re.compile(r'(Greater\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+Area)?)', re.IGNORECASE)),
    (92, re.compile(rf'([A-Z][a-zA-Z\s]+,\s*(?:{US_STATES}))', re.IGNORECASE)),
    (91, re.compile(rf'([A-Z][a-zA-Z]+,\s*(?:{INDIA_STATES}),?\s*India)', re.IGNORECASE)),
    (90, re.compile(rf'((?:Bengaluru|Bangalore|Mumbai|Delhi|Hyderabad|Chennai|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Nagpur|Indore|Gurgaon|Noida),\s*(?:{INDIA_STATES})(?:,?\s*India)?)', re.IGNORECASE)),
    (89, re.compile(r'((?:Bengaluru|Bangalore|Mumbai|Delhi|Hyderabad|Chennai|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Nagpur|Indore|Gurgaon|Noida)(?:,?\s*India)?)', re.IGNORECASE)),
    (88, re.compile(r'([A-Z][a-zA-Z\s]+,\s*(?:England|Scotland|Wales|Northern Ireland)(?:,\s*(?:United Kingdom|UK))?)', re.IGNORECASE)),
    (87, re.compile(r'(United\s+Kingdom)', re.IGNORECASE)),
    (86, re.compile(r'((?:London|Manchester|Birmingham|Leeds|Glasgow|Liverpool|Bristol|Sheffield|Edinburgh)(?:,?\s*(?:Area|England|UK))?)', re.IGNORECASE)),
    (85, re.compile(r'((?:Paris|Berlin|Munich|Frankfurt|Amsterdam|Brussels|Dublin|Vienna|Prague|Warsaw|Stockholm|Copenhagen|Oslo|Helsinki|Zurich|Geneva|Madrid|Barcelona|Rome|Milan)(?:,?\s*[A-Za-z]+)?)', re.IGNORECASE)),
    (84, re.compile(r'(Dubai,?\s*(?:United Arab Emirates|UAE)?)', re.IGNORECASE)),
    (83, re.compile(r'(Singapore)', re.IGNORECASE)),
    (82, re.compile(r'((?:Sydney|Melbourne|Brisbane|Perth|Toronto|Vancouver|Montreal|Calgary)(?:,?\s*(?:Area|Australia|Canada))?)', re.IGNORECASE)),
    (81, re.compile(rf'([A-Z][a-zA-Z\s]+,\s*({US_ABBREV}))\b', re.IGNORECASE)),
    (80, re.compile(r'(Columbus)\s+Crew', re.IGNORECASE)),
    (79, re.compile(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+Area)', re.IGNORECASE)),
    (70, re.compile(r'(Alexandria|Arlington|Fairfax|Reston|Herndon|Bethesda|Rockville|McLean|Vienna|Tysons|Ashburn|Sterling|Leesburg|Manassas|Woodbridge|Springfield|Annandale|Falls\s*Church|Centreville|Chantilly),\s*\.{2,}', re.IGNORECASE)),
]

INVALID_SMALL = {
    'sion', 'ston', 'burg', 'ville', 'ton', 'ford', 'port', 'land', 'wood',
    'dale', 'view', 'hill', 'mont', 'field', 'ley', 'ham', 'chester', 'shire',
    'borough', 'ing', 'lin', 'son', 'ber', 'den', 'ter', 'ner', 'don', 'gan',
    'ian', 'van', 'agra', 'chur', 'co', 'way', 'greater', 'et', 'lynn', 'change', 'lodi'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_linkedin_id(url: Optional[str]) -> Optional[str]:
    """Extract LinkedIn ID from URL."""
    if not url:
        return None
    m = re.search(r'linkedin\.com/in/([^/?]+)', str(url), re.IGNORECASE)
    return m.group(1).lower().rstrip('/') if m else None


# LinkedIn subdomain to country mapping
LINKEDIN_SUBDOMAIN_COUNTRIES = {
    'cn': 'china',
    'uk': 'united kingdom',
    'de': 'germany',
    'fr': 'france',
    'es': 'spain',
    'it': 'italy',
    'br': 'brazil',
    'mx': 'mexico',
    'jp': 'japan',
    'kr': 'south korea',
    'au': 'australia',
    'ca': 'canada',
    'nl': 'netherlands',
    'be': 'belgium',
    'ch': 'switzerland',
    'at': 'austria',
    'se': 'sweden',
    'no': 'norway',
    'dk': 'denmark',
    'fi': 'finland',
    'pl': 'poland',
    'pt': 'portugal',
    'ru': 'russia',
    'za': 'south africa',
    'sg': 'singapore',
    'hk': 'hong kong',
    'tw': 'taiwan',
    'th': 'thailand',
    'my': 'malaysia',
    'id': 'indonesia',
    'ph': 'philippines',
    'vn': 'vietnam',
    'ae': 'united arab emirates',
    'sa': 'saudi arabia',
    'il': 'israel',
    'tr': 'turkey',
    'ar': 'argentina',
    'cl': 'chile',
    'co': 'colombia',
    'nz': 'new zealand',
    'ie': 'ireland',
    'cz': 'czech republic',
    'hu': 'hungary',
    'ro': 'romania',
    'gr': 'greece',
    'in': 'india',  # Note: in.linkedin.com for India
}


def get_linkedin_url_country(url: Optional[str]) -> Optional[str]:
    """
    Extract country from LinkedIn URL subdomain.

    Examples:
        cn.linkedin.com → 'china'
        uk.linkedin.com → 'united kingdom'
        www.linkedin.com → None (no country indicator)

    Returns:
        Country name (lowercase) or None if no country subdomain
    """
    if not url:
        return None

    # Match subdomain pattern: xx.linkedin.com
    m = re.search(r'https?://([a-z]{2})\.linkedin\.com', str(url).lower())
    if m:
        subdomain = m.group(1)
        if subdomain != 'www':
            return LINKEDIN_SUBDOMAIN_COUNTRIES.get(subdomain)

    return None


def check_linkedin_url_country_match(url: str, claimed_country: str) -> Tuple[bool, Optional[str]]:
    """
    Check if LinkedIn URL country matches claimed country.

    Returns:
        (is_valid, rejection_reason)
        - (True, None) if match or no country in URL
        - (False, 'url_country_mismatch') if mismatch
    """
    url_country = get_linkedin_url_country(url)

    if not url_country:
        return True, None

    # Normalize claimed country
    claimed_lower = claimed_country.lower().strip() if claimed_country else ''

    # Direct match
    if url_country == claimed_lower:
        return True, None

    # Check if claimed country is an alias of URL country (use centralized COUNTRY_ALIASES)
    if url_country in COUNTRY_ALIASES:
        if claimed_lower in COUNTRY_ALIASES[url_country] or claimed_lower == url_country:
            return True, None

    # Check reverse - if claimed country has URL country as alias
    for main_country, aliases in COUNTRY_ALIASES.items():
        if claimed_lower == main_country or claimed_lower in aliases:
            if url_country == main_country or url_country in aliases:
                return True, None

    # Mismatch
    return False, 'url_country_mismatch'


def is_valid_state(state_part: str) -> bool:
    """Check if state is valid using geo lookup."""
    if not state_part:
        return False
    state_lower = state_part.lower().strip().rstrip('.')
    if state_lower in GEO_LOOKUP.get('state_abbr', {}):
        return True
    if state_lower in GEO_LOOKUP.get('us_states', {}):
        return True
    return False


def normalize_accents(text: str) -> str:
    """Remove accents from text."""
    if not text:
        return ""
    return unicodedata.normalize('NFD', str(text)).encode('ascii', 'ignore').decode('utf-8')


def strip_accents(s: str) -> str:
    """Strip accent marks from string."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_area_name(area: str) -> str:
    """Normalize area name for comparison.

    Note: We keep 'metropolitan' to distinguish areas like:
    - 'Greater Vancouver Metropolitan Area' (Canada)
    - 'Greater Vancouver Area' (Washington, USA)
    """
    area = area.lower().strip()
    area = area.replace("greater ", "").replace(" metro ", " ").replace(" area", "")
    # Keep 'metropolitan' but normalize ' metro ' (with spaces) to maintain distinction
    return area.strip()


def is_area_in_mappings(area: str) -> bool:
    """Check if area exists in our mappings. Uses cached normalized names for O(1) lookup."""
    area_norm = normalize_area_name(area)
    return area_norm in AREA_NORMALIZED_CACHE


def is_city_in_area_approved(city: str, area: str, claimed_state: str = "", claimed_country: str = "") -> bool:
    """
    Check if city is approved for given area AND state/country matches.
    Uses cached normalized data for O(1) lookups.

    Args:
        city: City name to check
        area: Metro area name (e.g., "Greater Seattle Area")
        claimed_state: Miner's claimed state (for US locations)
        claimed_country: Miner's claimed country

    Returns:
        True if city is in area AND state/country matches, False otherwise
    """
    area_norm = normalize_area_name(area)
    city_norm = strip_accents(city.lower().strip())
    claimed_state_lower = claimed_state.lower().strip() if claimed_state else ""
    claimed_country_lower = claimed_country.lower().strip() if claimed_country else ""

    # O(1) lookup for area key
    area_key = AREA_NORMALIZED_CACHE.get(area_norm)
    if not area_key:
        return False

    # O(1) lookup for city in area's cities
    cities_set = AREA_CITIES_CACHE.get(area_key, set())
    if city_norm not in cities_set:
        return False

    # City found in area - now check state/country
    area_data = AREA_MAPPINGS.get(area_key, {})
    area_state = area_data.get("state", "") if isinstance(area_data, dict) else ""
    area_country = area_data.get("country", "") if isinstance(area_data, dict) else ""

    if not claimed_state_lower and not claimed_country_lower:
        if is_ambiguous_city(city):
            return False
        return True

    area_state_lower = area_state.lower().strip()
    area_country_lower = area_country.lower().strip()

    # State abbreviation mapping for normalization
    state_abbr = GEO_LOOKUP.get('state_abbr', {})

    def normalize_state(s):
        """Normalize state name - convert abbreviation to full name."""
        s_lower = s.lower().strip()
        if s_lower in state_abbr:
            return state_abbr[s_lower].lower()  # "ca" -> "california"
        return s_lower

    # For US areas, verify state matches
    if area_country_lower == "united states":
        if claimed_state_lower and area_state_lower:
            # Normalize both states before comparison (handles abbreviations)
            claimed_state_norm = normalize_state(claimed_state_lower)
            area_state_norm = normalize_state(area_state_lower)
            if claimed_state_norm == area_state_norm:
                return True
        # If no state to check, country match is enough
        elif claimed_country_lower == "united states":
            return True
    else:
        # For international areas, verify country matches
        if claimed_country_lower and area_country_lower:
            if claimed_country_lower == area_country_lower:
                return True
            # Check country aliases (use centralized COUNTRY_ALIASES)
            if area_country_lower in COUNTRY_ALIASES:
                if claimed_country_lower in COUNTRY_ALIASES[area_country_lower]:
                    return True

    return False


def is_city_in_area_with_matching_state(city: str, claimed_state: str, claimed_country: str) -> bool:
    """
    Check if a city belongs to any area mapping where the state/country matches.

    This allows approving ambiguous cities like "Boston" when:
    1. Boston is in "Greater Boston Area"
    2. Greater Boston Area has state = "Massachusetts"
    3. Claimed state = "Massachusetts"

    Args:
        city: City name to check
        claimed_state: The state from the lead data
        claimed_country: The country from the lead data

    Returns:
        True if city is in an area with matching state/country
    """
    if not city:
        return False

    city_norm = _strip_accents_simple(city.lower().strip())
    claimed_state_lower = claimed_state.lower().strip() if claimed_state else ""
    claimed_country_lower = claimed_country.lower().strip() if claimed_country else ""

    # State abbreviation mapping for normalization
    state_abbr = GEO_LOOKUP.get('state_abbr', {})

    def normalize_state(s):
        s_lower = s.lower().strip()
        if s_lower in state_abbr:
            return state_abbr[s_lower].lower()
        return s_lower

    claimed_state_norm = normalize_state(claimed_state_lower) if claimed_state_lower else ""

    # Check each area to see if city is in it and state matches
    for area_key, cities_set in AREA_CITIES_CACHE.items():
        if city_norm in cities_set:
            # City is in this area - check if state/country matches
            area_data = AREA_MAPPINGS.get(area_key, {})
            if isinstance(area_data, dict):
                area_state = area_data.get("state", "").lower().strip()
                area_country = area_data.get("country", "").lower().strip()

                # For US areas, check state match
                if area_country == "united states" and claimed_country_lower in ["united states", "usa", "us", "u.s.", "u.s.a."]:
                    if claimed_state_norm and area_state:
                        area_state_norm = normalize_state(area_state)
                        if claimed_state_norm == area_state_norm:
                            return True
                # For international areas, check country match
                elif area_country and claimed_country_lower:
                    if claimed_country_lower == area_country:
                        return True
                    if area_country in COUNTRY_ALIASES:
                        if claimed_country_lower in COUNTRY_ALIASES[area_country]:
                            return True

    return False


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = normalize_accents(text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def normalize_role(role: str) -> str:
    """Normalize role title with abbreviation expansion."""
    if not role:
        return ""
    role = str(role).lower()
    role = re.sub(r'[^\w\s]', ' ', role)
    words = role.split()
    expanded = [ABBREVIATIONS.get(w.strip(), w) for w in words]
    return ' '.join(expanded).strip()


def remove_filler_words(text: str) -> str:
    """Remove common filler words from text."""
    filler = {'of', 'the', 'and', 'for', 'at', 'in', 'to', 'a', 'an'}
    words = text.split()
    return ''.join(w for w in words if w not in filler)


def extract_company_from_email(email: str) -> str:
    """Extract company name from email domain."""
    if not email or '@' not in str(email):
        return ""
    domain = str(email).split('@')[1].lower()
    domain = re.sub(r'\.(com|org|net|io|co|edu|gov|us|uk|ca|au|de|fr|in)$', '', domain)
    domain = re.sub(r'^(mail|email|smtp|info|contact)\.', '', domain)
    return domain.replace('-', '').replace('_', '').replace('.', '')


def is_valid_location(loc: str) -> bool:
    """Check if extracted location is valid."""
    if not loc:
        return False
    loc_lower = loc.lower().strip()
    if loc_lower in INVALID_SMALL:
        return False
    if 'locationunited' in loc_lower.replace(' ', ''):
        return False
    if ' is an ' in loc_lower or ' is a ' in loc_lower:
        return False
    if 'whitepages' in loc_lower or 'people search' in loc_lower:
        return False
    invalid_terms = [
        'linkedin', 'profile', 'view', 'email', 'graphic', 'experience',
        'followers', 'connections', 'senior', 'manager', 'director',
        'associate', 'specialist', 'coordinator', 'analyst', 'at city',
        'at ', ' at ', 'based in', 'headquartered', 'university', 'college', 'school'
    ]
    if any(x in loc_lower for x in invalid_terms):
        return False
    if len(loc) > 80 or len(loc) < 3:
        return False
    return True


def normalize_location(s: str) -> str:
    """Normalize location string for comparison."""
    if not s:
        return ''
    # Strip accents first (e.g., Montréal → Montreal)
    s = strip_accents(str(s))
    # Remove punctuation and lowercase
    s = re.sub(r'[^\w\s]', '', s.lower()).strip()
    # Apply city equivalents
    for old, new in CITY_EQUIVALENTS.items():
        s = s.replace(old, new)
    return s


def is_city_only_in_institution_context(city: str, text: str) -> bool:
    """
    Check if city appears ONLY in institution/org context.

    SIMPLE RULE:
    - If city is followed by SPACE + WORD (that's not state/country) → institution
    - If city is followed by punctuation, comma, or end of text → legitimate location

    Example:
        "Boston University" → True (space + "University")
        "Boston." → False (punctuation, no space + word)
        "Boston, MA" → False (comma + valid state)
        "at Boston." → False (city not followed by space + word)
    """
    if not text or not city:
        return False

    text_lower = text.lower()
    city_lower = city.lower().strip()
    city_esc = re.escape(city_lower)

    # Check if city appears at all (with word boundary)
    city_pattern = rf'\b{city_esc}\b'
    if not re.search(city_pattern, text_lower):
        return False

    # Get valid states and countries from geo_lookup
    valid_states = set(GEO_LOOKUP.get('us_states', {}).keys())
    state_abbr = set(GEO_LOOKUP.get('state_abbr', {}).keys())
    state_abbr.update({'d.c.', 'dc', 'd.c'})
    valid_countries = set(c.lower() for c in GEO_LOOKUP.get('countries', []))

    # Add country aliases from centralized COUNTRY_ALIASES
    for canonical, aliases in COUNTRY_ALIASES.items():
        valid_countries.add(canonical)
        valid_countries.update(aliases)

    # Valid location suffixes
    valid_location_suffixes = {
        'area', 'bay', 'metro', 'metropolitan', 'greater', 'city', 'county', 'region', 'district'
    }

    institution_prefixes = {
        'university', 'college', 'school', 'institute', 'academy',
        'hospital', 'bank', 'church', 'museum', 'library', 'port'
    }

    # Find all occurrences of city in text
    has_legitimate_location = False

    for match in re.finditer(city_pattern, text_lower):
        start_pos = match.start()
        end_pos = match.end()

        # Check what comes BEFORE the city
        text_before = text_lower[:start_pos].rstrip()

        education_prefixes = ('education:', 'studied at', 'alumni of', 'alumnus of', 'graduated from')
        before_trimmed = text_before.rstrip(' ·\t-')
        if any(before_trimmed.endswith(ep) for ep in education_prefixes):
            continue

        # Check for "[Education prefix] [School Name]. City" pattern
        # e.g., "Education: Trinity College. Burlington" - city follows school name
        # e.g., "Studied at UCLA. Los Angeles" - city follows school name
        # e.g., "Alumni of Boston College. Boston" - city follows school name
        edu_pattern = r'(education:|studied at|alumni of|alumnus of|graduated from)\s*[^·]+$'
        education_match = re.search(edu_pattern, before_trimmed)
        if education_match:
            # Text between education prefix and city - check if it looks like a school name
            edu_text = education_match.group(0)
            # If ends with period, it's likely "[prefix] School Name." pattern
            if edu_text.rstrip().endswith('.'):
                continue  # Skip - city is part of education location

        if text_before.endswith(' of'):
            before_of = text_before[:-3].rstrip()
            words_before = before_of.split()
            if words_before:
                last_word = words_before[-1].lower()
                if last_word in institution_prefixes:
                    continue

        if text_before.endswith(','):
            before_comma = text_before[:-1].rstrip()
            if re.search(r'\b(' + '|'.join(institution_prefixes) + r')\s+of\s+[\w\s]+$', before_comma):
                continue

        # Get what comes immediately after the city
        text_after = text_lower[end_pos:]

        # CASE 1: Nothing after city (end of text) → legitimate
        if not text_after or not text_after.strip():
            has_legitimate_location = True
            break

        # CASE 2: Followed by punctuation (not space) → check further
        # e.g., "Boston." "Boston," "Boston!" "Boston?"
        first_char = text_after[0] if text_after else ''
        if first_char and first_char not in ' \t':
            # It's punctuation or something else, not a space
            # Check if it's comma followed by state/country
            if first_char == ',':
                after_comma = text_after[1:].strip()
                words_after = after_comma.split()
                if words_after:
                    first_word = words_after[0].rstrip('.,;:').lower()
                    two_words = ' '.join(words_after[:2]).rstrip('.,;:').lower() if len(words_after) >= 2 else ''

                    # Check if it's a valid state
                    if first_word in valid_states or first_word in state_abbr:
                        has_legitimate_location = True
                        break
                    # Check if it's a valid country
                    if first_word in valid_countries or two_words in valid_countries:
                        has_legitimate_location = True
                        break
                    # Check if it's an institution keyword (FP prevention)
                    institution_keywords = {
                        'university', 'college', 'school', 'institute', 'academy',
                        'hospital', 'medical', 'clinic', 'health',
                        'consulting', 'group', 'inc', 'corp', 'llc', 'ltd',
                        'tribune', 'times', 'post', 'news'
                    }
                    if first_word in institution_keywords:
                        continue  # Skip this occurrence - institution format
                # If comma but not state/country/institution, allow (likely international region)
                has_legitimate_location = True
                break
            # Any other punctuation (period, etc.) → legitimate standalone city
            has_legitimate_location = True
            break

        # CASE 3: Followed by space - check what word comes after
        if first_char == ' ' or first_char == '\t':
            after_space = text_after.strip()
            words_after = after_space.split()

            if not words_after:
                # Just trailing space → legitimate
                has_legitimate_location = True
                break

            first_word = words_after[0].rstrip('.,;:!?').lower()
            two_words = ' '.join(words_after[:2]).rstrip('.,;:!?').lower() if len(words_after) >= 2 else ''

            # Check if followed by valid state
            if first_word in valid_states or first_word in state_abbr:
                has_legitimate_location = True
                break

            # Check if followed by valid country
            if first_word in valid_countries or two_words in valid_countries:
                has_legitimate_location = True
                break

            # Check if followed by location suffix (area, metro, etc.)
            if first_word in valid_location_suffixes:
                has_legitimate_location = True
                break

            # Check if followed by institution word (specific pattern)
            # Look at first 4 words to catch patterns like "Pontiac Protestant High School"
            # where the institution word ("school") isn't the immediate next word
            institution_words = {
                'university', 'college', 'school', 'institute', 'academy',
                'hospital', 'medical', 'clinic', 'health', 'healthcare',
                'consulting', 'group', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'company',
                'bank', 'tribune', 'times', 'post', 'news', 'journal', 'gazette',
                'dynamics', 'international', 'associates', 'partners', 'foundation',
                'museum', 'library', 'church', 'symphony', 'philharmonic', 'orchestra',
                'zoo', 'aquarium', 'garden', 'gardens', 'park', 'stadium', 'arena',
                'exchange', 'authority', 'commission', 'board', 'council', 'committee'
            }
            # Check first 4 words (handles "[City] [Adj] [Adj] School" patterns)
            nearby_words = [w.rstrip('.,;:!?').lower() for w in words_after[:4]]
            if any(w in institution_words for w in nearby_words):
                # This is institution context, continue checking other occurrences
                continue

            # If followed by a non-institution word (like job titles), it's legitimate
            has_legitimate_location = True
            break

    # If we found at least one legitimate location reference, allow it
    if has_legitimate_location:
        return False  # City appears in legitimate location context

    return True  # City ONLY appeared in institution/org context


def is_ambiguous_city(city: str) -> bool:
    """
    Check if a city is ambiguous (appears in multiple states/countries or both US and International).
    These cities require state/country validation in addition to city name match.
    """
    city_lower = city.lower().strip()
    return city_lower in AMBIGUOUS_CITIES


def verify_state_or_country_in_text(city: str, state: str, country: str, text: str) -> bool:
    """Verify geographic context in text."""
    city_lower = city.lower().strip()

    if city_lower not in AMBIGUOUS_CITIES:
        return True

    text_lower = text.lower()
    claimed_country_lower = country.lower().strip() if country else ''
    claimed_state_lower = state.lower().strip() if state else ''

    us_states_lower = {s.lower() for s in GEO_LOOKUP.get('us_states', {}).keys()}
    us_country_aliases = {'united states', 'us', 'usa', 'u.s.', 'u.s.a.', 'united states of america'}
    is_us_location = (
        claimed_country_lower in us_country_aliases or
        (not claimed_country_lower and claimed_state_lower in us_states_lower)
    )

    if is_us_location:
        if claimed_state_lower:
            escaped = re.escape(claimed_state_lower).replace(r'\ ', r'\s+')
            pattern = r'\b' + escaped + r'(?![a-z-])'
            if re.search(pattern, text_lower):
                return True
        return False
    else:
        if claimed_country_lower:
            escaped = re.escape(claimed_country_lower).replace(r'\ ', r'\s+')
            pattern = r'\b' + escaped + r'(?![a-z-])'
            if re.search(pattern, text_lower):
                return True
        if claimed_country_lower in COUNTRY_ALIASES:
            for alias in COUNTRY_ALIASES[claimed_country_lower]:
                if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                    return True
        return False


def is_city_matching_person_name(city: str, full_name: str, full_text: str) -> bool:
    """
    Check if city appears in text only as part of person's name (false positive).

    Example: City "Roberts" matching "Eric Roberts" in text.

    Args:
        city: City name (lowercase)
        full_name: Person's full name
        full_text: The text to check

    Returns:
        True if city only appears as part of person's name (should reject)
    """
    if not city or not full_name or not full_text:
        return False

    city_lower = city.lower().strip()
    name_lower = full_name.lower().strip()
    text_lower = full_text.lower()

    # Check if city is part of person's name
    name_parts = name_lower.split()
    city_in_name = any(city_lower == part or city_lower in part or part in city_lower
                       for part in name_parts if len(part) > 2)

    if not city_in_name:
        return False  # City is not part of person's name

    other_name_parts = [p for p in name_parts if p != city_lower and len(p) > 2]

    if other_name_parts:
        any_other_in_text = any(
            re.search(r'\b' + re.escape(p) + r'\b', text_lower)
            for p in other_name_parts
        )
        if not any_other_in_text:
            # Check if any occurrence looks like a location (city followed by state/country)
            state_abbr = GEO_LOOKUP.get('state_abbr', {})
            us_states = set(GEO_LOOKUP.get('us_states', {}).keys())
            valid_countries = {c.lower() for c in GEO_LOOKUP.get('countries', [])}
            city_pattern = r'\b' + re.escape(city_lower) + r'\b'
            has_location_context = False
            for match in re.finditer(city_pattern, text_lower):
                after = text_lower[match.end():match.end() + 30]
                # Must be followed by delimiter then state/country
                m = re.match(r'^[\s]*[,\-|:]\s*(\S+)', after)
                if m:
                    token = m.group(1).rstrip('.,;:').lower()
                    if token in state_abbr or token in us_states or token in valid_countries:
                        has_location_context = True
                        break
            if not has_location_context:
                return True  # No location context; city is clearly the person's name

    # Find all positions where city appears
    city_pattern = r'\b' + re.escape(city_lower) + r'\b'
    for match in re.finditer(city_pattern, text_lower):
        start, end = match.start(), match.end()

        # Check if any other name part is immediately adjacent (within 2 chars, allowing for space)
        is_part_of_name = False
        for other_part in other_name_parts:
            other_pattern = r'\b' + re.escape(other_part) + r'\b'
            for other_match in re.finditer(other_pattern, text_lower):
                other_start, other_end = other_match.start(), other_match.end()
                # Check if adjacent: other_part ends near city start, or city ends near other_part start
                if (other_end >= start - 2 and other_end <= start + 1) or \
                   (end >= other_start - 2 and end <= other_start + 1):
                    is_part_of_name = True
                    break
            if is_part_of_name:
                break

        if not is_part_of_name:
            # Found an independent occurrence of city (not part of name)
            return False

    return True  # City only appears as part of person's name - reject


def _has_contradicting_state_or_province(city: str, state: str, country: str, full_text: str, linkedin_url: str = "") -> bool:
    """
    Check if text contains a state/province/country that contradicts the claimed location.

    This catches cases like:
    - Claimed: Vancouver, Washington, USA
    - Text shows: "Vancouver, British Columbia" → CONTRADICTION (BC is Canada, not WA)
    - URL is ca.linkedin.com when claiming US → CONTRADICTION

    Returns:
        True if a contradicting state/province/country is found
    """
    if not city or not full_text:
        return False

    claimed_country_lower = country.lower().strip() if country else ""

    if linkedin_url:
        country_domain_match = re.search(r'https?://([a-z]{2})\.linkedin\.com', linkedin_url.lower())
        if country_domain_match:
            url_country_code = country_domain_match.group(1)
            if url_country_code not in ('ww', 'www'[:2]):
                country_to_code = {
                    'united states': None, 'us': None, 'usa': None, 'u.s.': None, 'u.s.a.': None,  # US uses www
                    'canada': 'ca', 'uk': 'uk', 'united kingdom': 'uk', 'great britain': 'uk', 'england': 'uk',
                    'australia': 'au', 'germany': 'de', 'france': 'fr', 'spain': 'es', 'italy': 'it',
                    'brazil': 'br', 'india': 'in', 'japan': 'jp', 'china': 'cn', 'mexico': 'mx',
                    'netherlands': 'nl', 'belgium': 'be', 'sweden': 'se', 'norway': 'no', 'denmark': 'dk',
                    'finland': 'fi', 'switzerland': 'ch', 'austria': 'at', 'poland': 'pl', 'ireland': 'ie',
                    'portugal': 'pt', 'greece': 'gr', 'turkey': 'tr', 'russia': 'ru', 'ukraine': 'ua',
                    'south africa': 'za', 'egypt': 'eg', 'nigeria': 'ng', 'kenya': 'ke', 'morocco': 'ma',
                    'algeria': 'dz', 'tunisia': 'tn', 'israel': 'il', 'saudi arabia': 'sa', 'uae': 'ae',
                    'united arab emirates': 'ae', 'qatar': 'qa', 'kuwait': 'kw', 'bahrain': 'bh',
                    'pakistan': 'pk', 'bangladesh': 'bd', 'indonesia': 'id', 'malaysia': 'my',
                    'singapore': 'sg', 'thailand': 'th', 'vietnam': 'vn', 'philippines': 'ph',
                    'south korea': 'kr', 'korea': 'kr', 'taiwan': 'tw', 'hong kong': 'hk',
                    'new zealand': 'nz', 'argentina': 'ar', 'chile': 'cl', 'colombia': 'co', 'peru': 'pe',
                    'venezuela': 've', 'czech republic': 'cz', 'czechia': 'cz', 'hungary': 'hu',
                    'romania': 'ro', 'bulgaria': 'bg', 'croatia': 'hr', 'serbia': 'rs', 'slovakia': 'sk',
                    # Additional countries
                    'afghanistan': 'af', 'chad': 'td', 'ethiopia': 'et', 'ghana': 'gh', 'cameroon': 'cm',
                    'ivory coast': 'ci', 'senegal': 'sn', 'uganda': 'ug', 'tanzania': 'tz', 'zimbabwe': 'zw',
                    'jordan': 'jo', 'lebanon': 'lb', 'iraq': 'iq', 'iran': 'ir', 'oman': 'om', 'yemen': 'ye',
                    'nepal': 'np', 'sri lanka': 'lk', 'myanmar': 'mm', 'cambodia': 'kh', 'laos': 'la',
                    'luxembourg': 'lu', 'slovenia': 'si', 'estonia': 'ee', 'latvia': 'lv', 'lithuania': 'lt',
                    'iceland': 'is', 'malta': 'mt', 'cyprus': 'cy', 'bosnia': 'ba', 'north macedonia': 'mk',
                    'ecuador': 'ec', 'bolivia': 'bo', 'paraguay': 'py', 'uruguay': 'uy', 'costa rica': 'cr',
                    'panama': 'pa', 'guatemala': 'gt', 'honduras': 'hn', 'el salvador': 'sv', 'nicaragua': 'ni',
                    'dominican republic': 'do', 'puerto rico': 'pr', 'jamaica': 'jm', 'cuba': 'cu',
                }
                expected_code = country_to_code.get(claimed_country_lower)

                # US claims should have www.linkedin.com (no country code)
                if claimed_country_lower in ['united states', 'us', 'usa', 'u.s.', 'u.s.a.']:
                    return True  # Contradiction - non-US domain for US claim

                # For other countries, check if URL domain matches expected
                if expected_code and url_country_code != expected_code:
                    return True  # Contradiction - different country domain

                # If country not in our mapping but we see a specific domain, be suspicious
                # Build reverse lookup from domain code to country
                if not expected_code and claimed_country_lower:
                    code_to_country = {}
                    for cname, ccode in country_to_code.items():
                        if ccode and ccode not in code_to_country:
                            code_to_country[ccode] = cname
                    # If URL domain code maps to a known country that's different from claimed
                    if url_country_code in code_to_country:
                        return True  # URL shows known country but claimed country is different/unknown

    city_lower = city.lower().strip()
    text_lower = full_text.lower()
    claimed_state_lower = state.lower().strip() if state else ""
    claimed_country_lower = country.lower().strip() if country else ""

    # Canadian provinces that could conflict with US locations
    canadian_provinces = {
        'british columbia', 'bc', 'b.c.',
        'alberta', 'ab',
        'ontario', 'on',
        'quebec', 'qc',
        'manitoba', 'mb',
        'saskatchewan', 'sk',
        'nova scotia', 'ns',
        'new brunswick', 'nb',
        'newfoundland', 'nl',
        'prince edward island', 'pei',
        'northwest territories', 'nt',
        'yukon', 'yt',
        'nunavut', 'nu'
    }

    # UK regions that could conflict
    uk_regions = {
        'england', 'scotland', 'wales', 'northern ireland',
        'greater london', 'greater manchester', 'west midlands',
        'lancashire', 'yorkshire', 'kent', 'essex', 'surrey'
    }

    # Find city in text and check what follows
    city_pattern = re.compile(r'\b' + re.escape(city_lower) + r'\b', re.IGNORECASE)

    for match in city_pattern.finditer(text_lower):
        end_pos = match.end()
        text_after = text_lower[end_pos:end_pos + 50]  # Look at next 50 chars

        # Check for ", Province/State" pattern
        comma_match = re.match(r'\s*,\s*([a-z][a-z\s\.]+?)(?:\s*,|\s*\.|\s*$)', text_after)
        if comma_match:
            found_region = comma_match.group(1).strip().rstrip('.')

            # Check if it's a Canadian province when claiming US
            if claimed_country_lower in ['united states', 'us', 'usa', 'u.s.', 'u.s.a.']:
                if found_region in canadian_provinces:
                    return True  # Contradiction - Canadian province for US claim

            # Check if it's UK region when claiming US
            if claimed_country_lower in ['united states', 'us', 'usa', 'u.s.', 'u.s.a.']:
                if found_region in uk_regions:
                    return True  # Contradiction - UK region for US claim

            # Check if found region is a different US state
            if claimed_state_lower:
                state_abbr = GEO_LOOKUP.get('state_abbr', {})
                found_region_full = state_abbr.get(found_region, found_region)
                claimed_state_full = state_abbr.get(claimed_state_lower, claimed_state_lower)

                # Get all valid US states for comparison
                valid_states = set(s.lower() for s in GEO_LOOKUP.get('states', []))
                valid_states.update(state_abbr.keys())

                if found_region in valid_states or found_region_full.lower() in valid_states:
                    # Found a US state - check if it matches
                    if found_region_full.lower() != claimed_state_full.lower():
                        if found_region != claimed_state_lower and found_region_full.lower() != claimed_state_lower:
                            return True  # Contradiction - different US state

    return False


def should_reject_city_match(city: str, state: str, country: str, full_text: str, full_name: str = "", city_only_fallback: bool = True, linkedin_url: str = "", role: str = "", company: str = "") -> bool:
    """
    Validate city match against false positive patterns and geographic constraints.

    Returns:
        True if the match should be rejected, False if it should be accepted
    """
    if not city or not full_text:
        return False
    # Check if city is a common word that shouldn't be treated as a city
    city_lower = city.lower().strip()
    if city_lower in INVALID_CITY_NAMES:
        return True  # Reject - not a real city name
    # Check if city only appears as part of person's name
    if full_name and is_city_matching_person_name(city, full_name, full_text):
        return True  # Reject - city matches person's name, not a real location
    if is_city_only_in_institution_context(city, full_text):
        return True  # Reject - city appears only in institution name

    # Check if city only appears as part of company name (e.g., "Merlin" in "Merlin Cyber")
    if company and city_lower in company.lower():
        text_without_company = full_text.lower().replace(company.lower(), '')
        if city_lower not in text_without_company:
            return True  # Reject - city only appears in company name, not as location

    # Check if city only appears in role/job title context (e.g., "Distribution Centre Supervisor")
    if role and city_lower in role.lower():
        # Check if city appears anywhere else in text besides the role
        text_without_role = full_text.lower().replace(role.lower(), '')
        if city_lower not in text_without_role:
            return True  # Reject - city only appears in role, not as location

    if linkedin_url and _has_contradicting_state_or_province(city, state, country, full_text, linkedin_url):
        return True

    is_ambiguous = is_ambiguous_city(city)
    is_english_word = is_english_word_city(city) and city_only_fallback
    needs_strict_validation = is_ambiguous or is_english_word

    if needs_strict_validation:
        if _verify_state_or_country_for_strict_validation(city, state, country, full_text, linkedin_url):
            return False

        greater_m = re.search(r'Greater\s+' + re.escape(city_lower) + r'\b', full_text, re.IGNORECASE)
        if greater_m:
            greater_text = greater_m.group(0).strip()
            if is_city_in_area_approved(city_lower, greater_text, state, country) or \
               is_city_in_area_approved(city_lower, greater_text + " Area", state, country):
                return False

        if _has_contradicting_state_or_province(city, state, country, full_text, ""):
            return True

        return True

    return False


def _verify_state_or_country_for_strict_validation(city: str, state: str, country: str, text: str, linkedin_url: str = '') -> bool:
    """
    Verify state (for US) or country (for international) appears AFTER the city in text.

    This is used for cities that need strict validation:
    - Ambiguous cities (appear in multiple states/countries)
    - English word cities (could be false positives from job titles, etc.)

    The state/country must appear AFTER the city to match location patterns like:
    - "Research, Australia"
    - "Reading, Pennsylvania" or "Reading, PA"
    - "Nice, France"

    For international locations, if country cannot be found in text, the LinkedIn
    profile URL domain (e.g., uk.linkedin.com) can be used to verify country.

    Args:
        city: The city name
        state: The claimed state (for US locations)
        country: The claimed country
        text: The text to search in
        linkedin_url: Optional LinkedIn URL for domain-based country verification

    Returns:
        True if state/country is verified AFTER city in text, False otherwise
    """
    text_lower = text.lower()
    city_lower = city.lower().strip()
    claimed_country_lower = country.lower().strip() if country else ''
    claimed_state_lower = state.lower().strip() if state else ''

    # Find all positions where city appears in text (as whole word)
    city_pattern = r'\b' + re.escape(city_lower) + r'\b'
    city_matches = list(re.finditer(city_pattern, text_lower))

    if not city_matches:
        return False

    us_states_lower = {s.lower() for s in GEO_LOOKUP.get('us_states', {}).keys()}
    us_country_aliases = {'united states', 'us', 'usa', 'u.s.', 'u.s.a.', 'united states of america'}
    is_us_location = (
        claimed_country_lower in us_country_aliases or
        (not claimed_country_lower and claimed_state_lower in us_states_lower)
    )

    MAX_DISTANCE = 30

    COMPOUND_CITY_PREFIXES = {
        'new', 'san', 'los', 'las', 'el', 'fort', 'saint', 'st', 'port',
        'west', 'east', 'north', 'south', 'mount', 'mt', 'grand', 'palm',
        'long', 'salt', 'little', 'santa', 'cape', 'baton', 'corpus',
        'virginia', 'college', 'del', 'la', 'le'
    }

    if is_us_location:
        if claimed_state_lower:
            for match in city_matches:
                city_start_pos = match.start()
                city_end_pos = match.end()
                text_after_city = text_lower[city_end_pos:city_end_pos + MAX_DISTANCE]

                if city_start_pos > 0:
                    pre_text = text_lower[:city_start_pos].rstrip()
                    if pre_text:
                        preceding_word = pre_text.split()[-1] if pre_text.split() else ''
                        if preceding_word in COMPOUND_CITY_PREFIXES:
                            continue

                if not re.match(r'^[\s]*[,\-|:]', text_after_city):
                    continue

                escaped = re.escape(claimed_state_lower).replace(r'\ ', r'\s+')
                pattern = r'\b' + escaped + r'(?![a-z-])'
                if re.search(pattern, text_after_city):
                    return True

                state_abbr = GEO_LOOKUP.get('state_abbr', {})
                for abbr, full_name in state_abbr.items():
                    if full_name.lower() == claimed_state_lower:
                        abbr_pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
                        if re.search(abbr_pattern, text_after_city):
                            return True
                        break
                    elif abbr.lower() == claimed_state_lower:
                        full_escaped = re.escape(full_name.lower()).replace(r'\ ', r'\s+')
                        full_pattern = r'\b' + full_escaped + r'(?![a-z-])'
                        if re.search(full_pattern, text_after_city):
                            return True
                        break
        return False
    else:
        if claimed_country_lower:
            if linkedin_url:
                url_country = get_linkedin_url_country(linkedin_url)
                if url_country:
                    return url_country == claimed_country_lower

            for match in city_matches:
                city_start_pos = match.start()
                city_end_pos = match.end()
                text_after_city = text_lower[city_end_pos:city_end_pos + MAX_DISTANCE]

                if city_start_pos > 0:
                    pre_text = text_lower[:city_start_pos].rstrip()
                    if pre_text:
                        preceding_word = pre_text.split()[-1] if pre_text.split() else ''
                        if preceding_word in COMPOUND_CITY_PREFIXES:
                            continue

                if not re.match(r'^[\s]*[,\-|:]', text_after_city):
                    continue

                escaped = re.escape(claimed_country_lower).replace(r'\ ', r'\s+')
                pattern = r'\b' + escaped + r'(?![a-z-])'
                if re.search(pattern, text_after_city):
                    return True

            if claimed_country_lower == 'united kingdom':
                ENGLISH_CITIES = {
                    'london', 'manchester', 'birmingham', 'liverpool', 'leeds', 'sheffield',
                    'bristol', 'newcastle', 'nottingham', 'leicester', 'southampton', 'portsmouth',
                    'brighton', 'plymouth', 'coventry', 'reading', 'derby', 'wolverhampton',
                    'milton keynes', 'york', 'oxford', 'cambridge', 'norwich', 'exeter',
                    'bournemouth', 'chester', 'bath', 'canterbury', 'hull', 'stoke',
                    'sunderland', 'middlesbrough', 'blackpool', 'luton', 'watford', 'ipswich',
                    'peterborough', 'slough', 'warrington', 'huddersfield', 'blackburn',
                    'bolton', 'oldham', 'rochdale', 'wigan', 'stockport', 'salford',
                }
                if city_lower in ENGLISH_CITIES:
                    for match in city_matches:
                        city_end_pos = match.end()
                        text_after_city = text_lower[city_end_pos:city_end_pos + MAX_DISTANCE]

                        if not re.match(r'^[\s]*[,\-|:]', text_after_city):
                            continue

                        if re.search(r'\bengland\b', text_after_city):
                            return True
        return False


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_name_in_result(full_name: str, result: Dict, linkedin_url: Optional[str] = None) -> bool:
    """Check if name appears in search result."""
    if not full_name:
        return False
    title = result.get('title', '')
    snippet = result.get('snippet', '')
    combined = f"{title} {snippet}"
    combined_norm = normalize_accents(combined.lower())
    name_norm = normalize_accents(str(full_name).lower())
    name_parts = name_norm.split()
    if not name_parts:
        return False
    first = name_parts[0]
    last = name_parts[-1] if len(name_parts) > 1 else first
    if first in combined_norm or last in combined_norm:
        return True
    if linkedin_url:
        lid = get_linkedin_id(linkedin_url)
        if lid:
            lid_clean = normalize_accents(lid.replace('-', ' ').replace('%', ' '))
            if first in lid_clean or last in lid_clean:
                return True
    return False


def check_company_in_result(company: str, result: Dict, email: Optional[str] = None) -> bool:
    """Check if company appears in search result."""
    if not company or not result:
        return False
    title = result.get('title', '')
    snippet = result.get('snippet', '')
    combined = f"{title} {snippet}"
    combined_clean = normalize_text(combined)
    company_clean = normalize_text(str(company))
    if not company_clean:
        return False
    if company_clean in combined_clean:
        return True
    skip_words = {'the', 'and', 'for', 'at', 'of', 'in', 'to', 'a', 'an'}
    company_words = [w for w in company_clean.split() if len(w) > 2 and w not in skip_words]
    if company_words:
        if len(company_words) > 3:
            if all(w in combined_clean for w in company_words[:3]):
                return True
        else:
            if all(w in combined_clean for w in company_words):
                return True
    if company_words and len(company_words[0]) >= 5:
        if company_words[0] in combined_clean:
            return True
    if email:
        email_company = extract_company_from_email(email)
        if email_company and len(email_company) >= 3:
            company_no_filler = remove_filler_words(company_clean)
            if email_company in company_no_filler or company_no_filler in email_company:
                return True
            if company_words:
                for word in reversed(company_words):
                    if len(word) >= 4 and email_company.endswith(word):
                        return True
            if 2 <= len(email_company) <= 5:
                pos = 0
                matched = True
                for char in email_company:
                    found = company_no_filler.find(char, pos)
                    if found == -1:
                        matched = False
                        break
                    pos = found + 1
                if matched:
                    return True
    return False


def extract_role_from_result(result: Dict, full_name: str = "", company: str = "") -> Optional[str]:
    """
    Extract role/title from LinkedIn search result.

    Looks for patterns like:
    - "John Smith - Senior Manager - Company | LinkedIn"
    - "John Smith | Senior Manager at Company"
    - Snippet: "Senior Manager at Company..."

    Args:
        result: Search result dict with title and snippet
        full_name: Person's full name (to exclude from role)
        company: Company name (to exclude from role)

    Returns:
        Extracted role string or None
    """
    title = result.get('title', '')
    snippet = result.get('snippet', '')

    # Normalize inputs
    name_lower = full_name.lower() if full_name else ""
    company_lower = company.lower() if company else ""

    # Try extracting from title first (most reliable)
    # Pattern: "Name - Role - Company | LinkedIn"
    title_match = re.search(r'^([^|]+)\|', title)
    if title_match:
        title_part = title_match.group(1).strip()
        # Split by " - " to get parts
        parts = [p.strip() for p in title_part.split(' - ')]

        for part in parts:
            part_lower = part.lower()
            # Skip if it's the name
            if name_lower and (name_lower in part_lower or part_lower in name_lower):
                continue
            # Skip if it's the company
            if company_lower and (company_lower in part_lower or part_lower in company_lower):
                continue
            # Skip if it's "LinkedIn" or similar
            if part_lower in ['linkedin', 'profile', 'professional profile']:
                continue
            # Skip if too short or too long
            if len(part) < 3 or len(part) > 80:
                continue
            # Skip if contains "..." (truncated)
            if '...' in part or ' ... ' in part:
                continue
            # This might be the role
            return part

    # Try extracting from snippet
    # Pattern: "Role at Company" or "Company | Role"
    if snippet:
        # Try "Role at Company" pattern
        at_match = re.search(r'^([A-Z][^.]+?)\s+at\s+', snippet)
        if at_match:
            potential_role = at_match.group(1).strip()
            if len(potential_role) >= 3 and len(potential_role) <= 80:
                # Make sure it's not the name
                if not (name_lower and name_lower in potential_role.lower()):
                    return potential_role

        # Try "Company | Role" pattern in snippet
        pipe_match = re.search(r'\|\s*([A-Z][^|.]+)', snippet)
        if pipe_match:
            potential_role = pipe_match.group(1).strip()
            if len(potential_role) >= 3 and len(potential_role) <= 80:
                if not (name_lower and name_lower in potential_role.lower()):
                    if not (company_lower and company_lower in potential_role.lower()):
                        return potential_role

    return None


def extract_location_from_text(text: str) -> str:
    """Extract location from search result text."""
    if not text:
        return ""
    # Try follower count pattern first
    match = re.search(r'([A-Z][a-z]+(?:[\s\-]+[A-Z][a-z]+)*(?:,\s*[A-Z][a-zA-Z\s]+)*)\.\s*\d+[KMk]?\s*(?:followers?|connections?|volgers?|collegamenti)', text)
    if match:
        loc = match.group(1).strip()
        if not any(x.lower() in loc.lower() for x in ['University', 'College', 'Institute', 'School', 'Inc', 'LLC', 'Ltd', 'Corp']):
            if is_valid_location(loc):
                return loc
    # Try other patterns
    for priority, pattern in sorted(LOCATION_PATTERNS, key=lambda x: -x[0]):
        match = pattern.search(text)
        if match:
            loc = match.group(1).strip()
            loc = re.sub(r'\s*\.{2,}\s*$', '', loc)
            loc = re.sub(r',\s*United\s*$', '', loc)
            if is_valid_location(loc):
                return loc
    return ""


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


def check_locations_match(extracted: str, ground_truth: str, full_text: str = "", linkedin_url: str = "") -> Tuple[bool, str]:
    """
    Check if extracted location matches ground truth.

    Args:
        extracted: Extracted location string
        ground_truth: Claimed location (city, state, country)
        full_text: Original full text (for institution context check)
        linkedin_url: LinkedIn URL for domain-based country detection (ca.linkedin.com = Canada)

    Returns:
        (is_match, match_method)
    """
    if not extracted or not ground_truth:
        return False, 'missing_data'
    ext_norm = normalize_location(extracted)
    gt_norm = normalize_location(ground_truth)
    if not ext_norm or not gt_norm:
        return False, 'empty_norm'
    if ext_norm == gt_norm:
        return True, 'exact'
    if gt_norm in ext_norm:
        return True, 'direct'

    # Check if extracted is structured (has comma/state) or city-only
    extracted_is_city_only = ',' not in str(extracted) and len(ext_norm.split()) <= 2

    # Parse ground_truth into city, state, country (needed for ambiguous city check)
    # Format: "City, State, Country" (3 parts) or "City, Country" (2 parts for international)
    gt_parts = [p.strip() for p in str(ground_truth).split(',')]
    gt_city_parsed = gt_parts[0].lower() if gt_parts else ''

    # Determine if 2-part location is (city, state) or (city, country)
    us_states_lower = {s.lower() for s in GEO_LOOKUP.get('us_states', {}).keys()}
    if len(gt_parts) == 3:
        gt_state_parsed = gt_parts[1]
        gt_country_parsed = gt_parts[2]
    elif len(gt_parts) == 2:
        part1_lower = gt_parts[1].lower()
        if part1_lower in us_states_lower:
            # "Austin, Texas" - US location with state only
            gt_state_parsed = gt_parts[1]
            gt_country_parsed = 'United States'  # Infer country
        else:
            # "London, United Kingdom" - International location
            gt_state_parsed = ''
            gt_country_parsed = gt_parts[1]
    else:
        gt_state_parsed = ''
        gt_country_parsed = ''

    if ext_norm.startswith(gt_norm) or gt_norm.startswith(ext_norm):
        # If extracted is city-only, check for institution context and ambiguous cities
        city_for_check = ext_norm.split()[0] if ext_norm else ''
        if extracted_is_city_only and should_reject_city_match(city_for_check, gt_state_parsed, gt_country_parsed, full_text, linkedin_url=linkedin_url):
            pass  # Don't return, continue to other strategies
        else:
            return True, 'startswith'

    def strip_area_suffix(s):
        s = re.sub(r'\s+metropolitan\s+area$', '', s).strip()
        s = re.sub(r'\s+area$', '', s).strip()
        return s

    ext_stripped = strip_area_suffix(ext_norm)
    gt_stripped = strip_area_suffix(gt_norm)
    if ext_stripped and gt_stripped:
        city_for_check = ext_stripped.split()[0] if ext_stripped else ''
        if ext_stripped == gt_stripped:
            if extracted_is_city_only and should_reject_city_match(city_for_check, gt_state_parsed, gt_country_parsed, full_text, linkedin_url=linkedin_url):
                pass  # Don't return, continue to other strategies
            else:
                return True, 'suffix_match'
        if ext_stripped in gt_stripped or gt_stripped in ext_stripped:
            if extracted_is_city_only and should_reject_city_match(city_for_check, gt_state_parsed, gt_country_parsed, full_text, linkedin_url=linkedin_url):
                pass  # Don't return, continue to other strategies
            else:
                return True, 'suffix_match'

    gt_city = CITY_EQUIVALENTS.get(gt_city_parsed, gt_city_parsed)
    gt_state = gt_state_parsed
    gt_country = gt_country_parsed

    is_area_extraction = any(x in ext_norm for x in ['area', 'metropolitan', 'metro', 'greater'])
    if gt_city and len(gt_city) > 2 and gt_city in ext_norm and not is_area_extraction:
        if should_reject_city_match(gt_city, gt_state, gt_country, full_text, linkedin_url=linkedin_url):
            pass  # Skip city_extract, continue to next strategy
        else:
            return True, 'city_extract'

    gt_city_orig = str(ground_truth).split(',')[0].strip() if ground_truth else ''
    if gt_city_orig and is_city_in_area_approved(gt_city_orig, extracted, gt_state, gt_country):
        city_for_check = gt_city_orig.lower()
        if extracted_is_city_only and should_reject_city_match(city_for_check, gt_state, gt_country, full_text, linkedin_url=linkedin_url):
            pass  # Don't return, continue to other strategies
        else:
            return True, 'area_mapping'

    if extracted:
        ext_city = str(extracted).split(',')[0].strip().lower()
        ext_city = CITY_EQUIVALENTS.get(ext_city, ext_city)
        if ext_city and len(ext_city) > 2 and ext_city in gt_city:
            if should_reject_city_match(ext_city, gt_state, gt_country, full_text, linkedin_url=linkedin_url):
                pass  # Skip ext_city_in_gt, continue to next strategy
            else:
                return True, 'ext_city_in_gt'

    if not is_area_extraction:
        ext_words = set(ext_norm.replace(',', ' ').split())
        # Only use city (first part) from ground truth for word overlap
        gt_city_for_overlap = ground_truth.split(',')[0].strip().lower() if ground_truth else ''
        gt_words = set(gt_city_for_overlap.split()) if gt_city_for_overlap and len(gt_city_for_overlap) > 2 else set()
        if gt_words and gt_words.issubset(ext_words):
            if should_reject_city_match(gt_city_for_overlap, gt_state, gt_country, full_text, linkedin_url=linkedin_url):
                pass  # Skip word_overlap
            else:
                return True, 'word_overlap'

    # Saint/St prefix normalization fallback
    # Handles: "Saint Paul" vs "St Paul", "St. Louis" vs "Saint Louis", etc.
    _saint_re = re.compile(r'^(?:saint|st\.?)\s+', re.IGNORECASE)
    ext_city_raw = str(extracted).split(',')[0].strip().lower() if extracted else ''
    gt_city_raw = gt_city_parsed  # already lowercased
    if _saint_re.match(ext_city_raw) and _saint_re.match(gt_city_raw):
        ext_remainder = _saint_re.sub('', ext_city_raw).strip()
        gt_remainder = _saint_re.sub('', gt_city_raw).strip()
        if ext_remainder and ext_remainder == gt_remainder:
            if verify_state_or_country_in_text(gt_city_raw, gt_state, gt_country, full_text):
                return True, 'saint_st_match'

    return False, 'no_match'


def check_role_matches(gt_role: str, text: str) -> bool:
    """Check if ground truth role matches text."""
    if not gt_role or not text:
        return False
    gt_norm = normalize_role(str(gt_role))
    text_norm = normalize_role(text)
    if not gt_norm or not text_norm:
        return False
    skip_words = {'the', 'and', 'for', 'at', 'of', 'in', 'to', 'a', 'an'}
    gt_words = [w for w in gt_norm.split() if len(w) > 2 and w not in skip_words]
    if not gt_words:
        gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return False
    if len(gt_words) <= 2:
        return all(w in text_norm for w in gt_words)
    else:
        return all(w in text_norm for w in gt_words[:3])


def validate_role_rule_based(
    gt_role: str,
    search_results: List[Dict],
    linkedin_url: str,
    full_name: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate role using rule-based matching.

    Returns:
        (is_valid, method) - method is None if not valid
    """
    if not gt_role or not search_results:
        return False, None
    expected_lid = get_linkedin_id(linkedin_url)

    # First try URL-matched result
    for result in search_results:
        result_lid = get_linkedin_id(result.get('link', ''))
        if result_lid and expected_lid and result_lid == expected_lid:
            combined = f"{result.get('title', '')} {result.get('snippet', '')}"
            if check_role_matches(gt_role, combined):
                return True, 'url_match'

    # Then try name-matched results
    for result in search_results:
        if check_name_in_result(full_name, result, linkedin_url):
            combined = f"{result.get('title', '')} {result.get('snippet', '')}"
            if check_role_matches(gt_role, combined):
                return True, 'name_match'

    return False, None


def validate_role_with_llm(
    name: str,
    company: str,
    claimed_role: str,
    exact_url_result: Optional[str],
    other_results: List[str],
    openrouter_api_key: str,
    model: str = 'google/gemini-2.5-flash-lite'
) -> Dict[str, Any]:
    """
    Validate role using LLM.

    Returns:
        {
            'success': bool,
            'role_pass': bool (if success),
            'role_found': str (if success),
            'error': str (if not success),
            'raw': str (raw LLM response)
        }
    """
    other_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(other_results[:5])]) if other_results else "None"

    prompt = f'''"{name}" at "{company}" - Role: "{claimed_role}"

[LINKEDIN - PRIORITY]
{exact_url_result or "Not found"}

[OTHER - fallback]
{other_text}

RULES:
1. FAIL if "{claimed_role}" contains company name
2. FAIL if "{claimed_role}" is invalid (e.g., "Job Title", "N/A", "Title", "Employee")
3. FAIL if LinkedIn shows different company than "{company}"
4. FAIL if different function (Sales≠Product, Engineer≠Marketing)
5. PASS if semantically same function:
   - Ignore seniority (Manager≈Sr.Manager, Engineer≈Senior Engineer)
   - Match synonyms (Developer≈Engineer, VP≈Vice President)
   - Match abbreviations (Dev≈Developer, Mgr≈Manager, Dir≈Director)
6. Use OTHER only if role not found in LinkedIn
7. NOT FOUND: role_pass=false, role_found=""

JSON only: {{"role_pass": bool, "role_found": ""}}'''

    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {openrouter_api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 200,
                'temperature': 0
            },
            timeout=30
        )

        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content']
            # Parse JSON
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return {
                        'success': True,
                        'role_pass': parsed.get('role_pass', False),
                        'role_found': parsed.get('role_found', ''),
                        'raw': content
                    }
                except:
                    pass
            return {'success': False, 'error': 'parse_error', 'raw': content}
        else:
            return {'success': False, 'error': f'HTTP {resp.status_code}'}
    except Exception as e:
        return {'success': False, 'error': str(e)[:100]}


def check_q3_location_fallback(
    name: str,
    company: str,
    city: str,
    linkedin_url: str,
    scrapingdog_api_key: str,
    state: str = '',
    country: str = '',
    role: str = ''
) -> Dict[str, Any]:
    """
    Q3 Location fallback search.

    Returns:
        {
            'success': bool,
            'passed': bool,
            'snippet': str,
            'results': list,
            'error': str (if failed)
        }
    """
    expected_lid = get_linkedin_id(linkedin_url)
    if not expected_lid:
        return {'success': False, 'passed': False, 'error': 'No LinkedIn ID'}

    query = f'"{name}" "{company}" "{city}" "{linkedin_url}"'

    try:
        # Retry loop for transient failures (timeout, 502, 503, 429)
        resp = None
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.get('https://api.scrapingdog.com/google', params={
                    'api_key': scrapingdog_api_key,
                    'query': query,
                    'results': 3,
                    'country': 'us'
                }, timeout=45)

                if resp.status_code in (502, 503, 429):
                    last_error = f'HTTP {resp.status_code}'
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # 1s, 2s
                        continue
                break  # Success or non-retryable HTTP error
            except requests.exceptions.Timeout:
                last_error = 'Read timed out'
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {'success': False, 'passed': False, 'error': f'Timeout after 3 retries', 'results': []}
            except requests.exceptions.ConnectionError:
                last_error = 'Connection error'
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {'success': False, 'passed': False, 'error': f'Connection error after 3 retries', 'results': []}

        if resp is None:
            return {'success': False, 'passed': False, 'error': last_error or 'No response', 'results': []}

        if resp.status_code != 200:
            return {'success': False, 'passed': False, 'error': f'HTTP {resp.status_code}', 'results': []}

        data = resp.json()
        results = data.get('organic_results', [])

        # Format results for storage
        formatted_results = [{
            'title': r.get('title', ''),
            'snippet': r.get('snippet', ''),
            'link': r.get('link', ''),
            'missing': r.get('missing', [])
        } for r in results]

        for r in results:
            result_lid = get_linkedin_id(r.get('link', ''))
            if result_lid == expected_lid:
                missing = r.get('missing', [])
                snippet = r.get('snippet', '')
                title = r.get('title', '')
                full_text = f"{title} {snippet}"

                if not missing:  # All search terms found including city
                    city_lower = city.lower()
                    city_equiv = CITY_EQUIVALENTS.get(city_lower, city_lower)
                    result_url = r.get('link', '')

                    text_lower = full_text.lower()
                    _saint_re = re.compile(r'^(?:saint|st\.?)\s+', re.IGNORECASE)
                    city_saint_variants = []
                    if _saint_re.match(city_lower):
                        remainder = _saint_re.sub('', city_lower).strip()
                        city_saint_variants = [f'saint {remainder}', f'st {remainder}', f'st. {remainder}']
                    city_in_text = (
                        re.search(r'\b' + re.escape(city_lower) + r'\b', text_lower) or
                        (city_equiv != city_lower and re.search(r'\b' + re.escape(city_equiv) + r'\b', text_lower)) or
                        any(re.search(r'\b' + re.escape(v) + r'\b', text_lower) for v in city_saint_variants)
                    )
                    if not city_in_text:
                        return {
                            'success': True,
                            'passed': False,
                            'snippet': snippet[:200],
                            'results': formatted_results,
                            'error': f'City "{city}" not literally in text despite missing=[]'
                        }

                    if _has_contradicting_state_or_province(city_lower, state, country, full_text, result_url):
                        return {
                            'success': True,
                            'passed': False,
                            'snippet': snippet[:200],
                            'results': formatted_results,
                            'error': 'Contradicting location in text or URL domain'
                        }

                    if is_city_only_in_institution_context(city_lower, full_text):
                        return {
                            'success': True,
                            'passed': False,
                            'snippet': snippet[:200],
                            'results': formatted_results,
                            'error': 'City only in institution context'
                        }

                    if name and is_city_matching_person_name(city_lower, name, full_text):
                        return {
                            'success': True,
                            'passed': False,
                            'snippet': snippet[:200],
                            'results': formatted_results,
                            'error': 'City only appears as part of person name'
                        }

                    if role and city_lower in role.lower():
                        text_without_role = full_text.lower().replace(role.lower(), '')
                        if city_lower not in text_without_role:
                            return {
                                'success': True,
                                'passed': False,
                                'snippet': snippet[:200],
                                'results': formatted_results,
                                'error': 'City only in role/title context'
                            }

                    is_ambiguous = is_ambiguous_city(city_lower)
                    is_english_word = is_english_word_city(city_lower)
                    needs_strict_validation = is_ambiguous or is_english_word

                    if needs_strict_validation:
                        has_state_country = _verify_state_or_country_for_strict_validation(city_lower, state, country, full_text, linkedin_url)

                        if not has_state_country:
                            greater_m = re.search(r'Greater\s+' + re.escape(city_lower), full_text, re.IGNORECASE)
                            if greater_m:
                                greater_text = greater_m.group(0).strip()
                                if is_city_in_area_approved(city_lower, greater_text, state, country) or \
                                   is_city_in_area_approved(city_lower, greater_text + " Area", state, country):
                                    has_state_country = True

                        if not has_state_country:
                            error_type = 'English word city' if is_english_word else 'Ambiguous city'
                            return {
                                'success': True,
                                'passed': False,
                                'snippet': snippet[:200],
                                'results': formatted_results,
                                'error': f'{error_type} without state/country verification'
                            }

                    return {
                        'success': True,
                        'passed': True,
                        'snippet': snippet[:200],
                        'results': formatted_results,
                        'error': None
                    }
                else:
                    return {
                        'success': True,
                        'passed': False,
                        'snippet': snippet[:200],
                        'results': formatted_results,
                        'error': f'Missing: {missing}'
                    }

        return {
            'success': True,
            'passed': False,
            'snippet': None,
            'results': formatted_results,
            'error': 'No URL match in results'
        }

    except Exception as e:
        return {'success': False, 'passed': False, 'error': str(e)[:100], 'results': []}


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_lead(
    lead: Dict[str, Any],
    search_results: List[Dict],
    url_matched_result: Optional[Dict] = None,
    openrouter_api_key: Optional[str] = None,
    scrapingdog_api_key: Optional[str] = None,
    use_llm: bool = True,
    use_q3: bool = True
) -> Dict[str, Any]:
    """
    Validate a single lead.

    Args:
        lead: Lead data with keys: full_name, business, linkedin, city, state, country, role, email
        search_results: List of search results from Google
        url_matched_result: The specific result that matched the LinkedIn URL (optional)
        openrouter_api_key: API key for LLM validation (optional if use_llm=False)
        scrapingdog_api_key: API key for Q3 fallback search (optional if use_q3=False)
        use_llm: Whether to use LLM for role verification when rule-based fails
        use_q3: Whether to use Q3 location fallback when location not found

    Returns:
        {
            'valid': bool,
            'rejection_reason': str or None,
            'checks': {
                'role_validity': {'passed': bool, 'reason': str},
                'url': {'passed': bool, 'reason': str},
                'name': {'passed': bool},
                'company': {'passed': bool},
                'location': {'passed': bool, 'method': str, 'extracted': str, 'q3_called': bool, 'q3_result': str},
                'role': {'passed': bool, 'method': str, 'llm_used': bool}
            }
        }
    """
    result = {
        'valid': False,
        'rejection_reason': None,
        'checks': {
            'role_validity': {'passed': False, 'reason': None},
            'url': {'passed': False, 'reason': None},
            'name': {'passed': False},
            'company': {'passed': False},
            'location': {'passed': False, 'method': None, 'extracted': None, 'q3_called': False, 'q3_result': None},
            'role': {'passed': False, 'method': None, 'llm_used': False}
        }
    }

    # Extract lead data
    name = str(lead.get('full_name', '')).strip()
    company = str(lead.get('business', '')).strip()
    linkedin = str(lead.get('linkedin', '')).strip()
    city = str(lead.get('city', '')).strip()
    state = str(lead.get('state', '')).strip()
    country = str(lead.get('country', '')).strip()
    role = str(lead.get('role', '')).strip()
    email = str(lead.get('email', '')).strip()

    # STEP 0: Role validity check - SKIPPED (already done in gateway via check_role_sanity)
    result['checks']['role_validity']['passed'] = True
    result['checks']['role_validity']['reason'] = None

    # STEP 1: URL check
    expected_lid = get_linkedin_id(linkedin)
    if not url_matched_result:
        for r in search_results:
            if get_linkedin_id(r.get('link', '')) == expected_lid:
                url_matched_result = r
                break

    if not url_matched_result:
        result['checks']['url']['reason'] = 'not_found'
        result['rejection_reason'] = 'url_not_found'
        return result

    result['checks']['url']['passed'] = True

    # STEP 2: Name check
    if not check_name_in_result(name, url_matched_result, linkedin):
        result['rejection_reason'] = 'name_not_found'
        return result

    result['checks']['name']['passed'] = True

    # STEP 3: Company check
    if not check_company_in_result(company, url_matched_result, email):
        result['rejection_reason'] = 'company_not_found'
        return result

    result['checks']['company']['passed'] = True

    # STEP 4: Location check
    full_text = f"{url_matched_result.get('title', '')} {url_matched_result.get('snippet', '')}"

    # STEP 4a: Check LinkedIn URL country matches claimed country
    result_url = url_matched_result.get('link', '')
    url_country_valid, url_country_error = check_linkedin_url_country_match(result_url, country)
    if not url_country_valid:
        result['checks']['location']['method'] = 'url_country_mismatch'
        result['checks']['location']['extracted'] = get_linkedin_url_country(result_url)
        result['rejection_reason'] = 'url_country_mismatch'
        return result

    extracted_loc = extract_location_from_text(full_text)

    # Validate extracted state for structured locations
    structured_loc_valid = False
    structured_loc_type = None  # 'us' or 'international'
    extracted_parts = {}  # Store parsed parts for matching

    if extracted_loc and ',' in extracted_loc:
        parts = [p.strip() for p in extracted_loc.split(',')]
        if len(parts) >= 2:
            state_part = parts[1]

            # Check if US location (valid US state in parts[1])
            if state_part and is_valid_state(state_part):
                structured_loc_valid = True
                structured_loc_type = 'us'
                extracted_parts = {'city': parts[0], 'state': state_part}

            # Check if international location (valid country in parts[2] or parts[1])
            elif len(parts) >= 3:
                country_part = parts[2].lower().strip()
                # Check against geo_lookup countries (excluding US)
                valid_countries = GEO_LOOKUP.get('countries', [])
                if country_part in [c.lower() for c in valid_countries] and country_part != 'united states':
                    structured_loc_valid = True
                    structured_loc_type = 'international'
                    extracted_parts = {'city': parts[0], 'region': parts[1], 'country': parts[2]}

            # Check if 2-part international (City, Country)
            elif len(parts) == 2:
                country_part = parts[1].lower().strip()
                valid_countries = GEO_LOOKUP.get('countries', [])
                if country_part in [c.lower() for c in valid_countries] and country_part != 'united states':
                    structured_loc_valid = True
                    structured_loc_type = 'international'
                    extracted_parts = {'city': parts[0], 'country': parts[1]}

            # If still not valid, clear extraction
            if not structured_loc_valid:
                # Check for metro areas which are still valid
                is_metro = any(x.lower() in extracted_loc.lower() for x in ['Area', 'Metropolitan', 'Greater'])
                if not is_metro:
                    extracted_loc = ''

    result['checks']['location']['extracted'] = extracted_loc

    # Structured location check (US or International)
    location_passed = False
    location_method = None

    if structured_loc_valid and extracted_loc and city:
        ext_city = extracted_parts.get('city', '').lower()
        claimed_city = city.lower().strip()
        city_match = (claimed_city in ext_city or ext_city in claimed_city)

        if structured_loc_type == 'us':
            # US: Match BOTH city AND state
            ext_state = extracted_parts.get('state', '').lower()
            claimed_state = state.lower().strip() if state else ''

            # Normalize state abbreviations for comparison
            state_abbr = GEO_LOOKUP.get('state_abbr', {})
            # Convert abbreviation to full name if needed
            if ext_state in state_abbr:
                ext_state_full = state_abbr[ext_state].lower()
            else:
                ext_state_full = ext_state
            if claimed_state in state_abbr:
                claimed_state_full = state_abbr[claimed_state].lower()
            else:
                claimed_state_full = claimed_state

            state_match = (claimed_state_full in ext_state_full or ext_state_full in claimed_state_full or
                          claimed_state in ext_state or ext_state in claimed_state)

            if city_match and state_match:
                location_passed = True
                location_method = 'structured_us_match'
            elif not city_match:
                result['checks']['location']['method'] = 'city_mismatch'
                result['rejection_reason'] = 'city_mismatch'
                return result
            elif not state_match:
                result['checks']['location']['method'] = 'state_mismatch'
                result['rejection_reason'] = 'state_mismatch'
                return result

        elif structured_loc_type == 'international':
            # International: Match BOTH city AND country
            ext_country = extracted_parts.get('country', '').lower()
            claimed_country = country.lower().strip() if country else ''
            country_match = (claimed_country in ext_country or ext_country in claimed_country)

            # Also check country aliases (e.g., "USA" vs "United States")
            if not country_match:
                for canonical, aliases in COUNTRY_ALIASES.items():
                    # Check if both are aliases of the same canonical country
                    claimed_is_canonical = (claimed_country == canonical)
                    claimed_is_alias = (claimed_country in aliases)
                    ext_is_canonical = (ext_country == canonical)
                    ext_is_alias = (ext_country in aliases)

                    if (claimed_is_canonical or claimed_is_alias) and (ext_is_canonical or ext_is_alias):
                        country_match = True
                        break

            if city_match and country_match:
                location_passed = True
                location_method = 'structured_intl_match'
            elif not city_match:
                result['checks']['location']['method'] = 'city_mismatch'
                result['rejection_reason'] = 'city_mismatch'
                return result
            elif not country_match:
                result['checks']['location']['method'] = 'country_mismatch'
                result['rejection_reason'] = 'country_mismatch'
                return result

    # Other location checks if not structured
    if not location_passed and city:
        gt_location = f"{city}, {state}, {country}".strip(', ')

        if not structured_loc_valid and extracted_loc:
            loc_match, loc_method = check_locations_match(extracted_loc, gt_location, full_text)
            if loc_match:
                location_passed = True
                location_method = loc_method

        if not location_passed:
            city_lower = city.lower().strip()
            if re.search(r'\b' + re.escape(city_lower) + r'\b', full_text.lower()):
                # Get the result URL for domain check
                result_url = url_matched_result.get('link', '') if url_matched_result else linkedin
                if should_reject_city_match(city_lower, state, country, full_text, name, linkedin_url=result_url, role=role):
                    pass  # Skip - institution context or ambiguous city
                else:
                    location_passed = True
                    location_method = 'city_fallback'

        # Area check
        if not location_passed:
            # Normalize "St." to "St " (same length) so regex can match St. Paul, St. Louis etc.
            area_search_text = re.sub(r'\bSt\.\s', 'St  ', full_text)
            area_match = re.search(r'(Greater\s+[\w\s\-]+|[\w\s\-]+\s+Metropolitan|[\w\s\-]+\s+Bay|[\w\s\-]+\s+Metro)\s*Area', area_search_text, re.IGNORECASE)
            if area_match:
                area_found = full_text[area_match.start():area_match.end()].strip()
                if city_lower not in area_found.lower():
                    # Check if city appears right before area name in text
                    area_start = area_match.start()
                    text_before_area = full_text[:area_start]
                    city_before_area = re.search(r'\b' + re.escape(city_lower) + r'\b\s*,\s*$', text_before_area.lower())
                    if city_before_area:
                        location_passed = True
                        location_method = 'area_approved'
                        result['checks']['location']['extracted'] = f"{city}, {area_found}"
                    elif is_city_in_area_approved(city, area_found, state, country):
                        location_passed = True
                        location_method = 'area_approved'
                        result['checks']['location']['extracted'] = area_found
                    elif is_area_in_mappings(area_found):
                        result['checks']['location']['method'] = 'area_mismatch'
                        result['rejection_reason'] = 'area_mismatch'
                        return result

        # Non-LinkedIn fallback
        if not location_passed:
            for r in search_results[:5]:
                if get_linkedin_id(r.get('link', '')):
                    continue
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}"
                r_loc = extract_location_from_text(r_text)
                if r_loc:
                    loc_match, loc_method = check_locations_match(r_loc, gt_location, r_text)
                    if loc_match:
                        location_passed = True
                        location_method = f'non_linkedin_{loc_method}'
                        result['checks']['location']['extracted'] = r_loc
                        break

    # Q3 location fallback when not found
    if not location_passed and use_q3 and scrapingdog_api_key and city and linkedin:
        result['checks']['location']['q3_called'] = True
        q3_result = check_q3_location_fallback(name, company, city, linkedin, scrapingdog_api_key, state, country, role)

        if q3_result.get('passed'):
            location_passed = True
            location_method = 'q3_fallback'
            result['checks']['location']['q3_result'] = 'pass'
        else:
            result['checks']['location']['q3_result'] = 'fail'

    if not location_passed:
        result['checks']['location']['method'] = 'not_found'
        result['rejection_reason'] = 'location_not_found'
        return result

    result['checks']['location']['passed'] = True
    result['checks']['location']['method'] = location_method

    # STEP 5: Role rule-based check
    role_passed, role_method = validate_role_rule_based(role, search_results, linkedin, name)

    if role_passed:
        result['checks']['role']['passed'] = True
        result['checks']['role']['method'] = role_method
        result['valid'] = True
        return result

    # STEP 6: Role LLM check (if enabled)
    if use_llm and openrouter_api_key:
        result['checks']['role']['llm_used'] = True

        # Prepare LLM input
        exact_url_text = None
        if url_matched_result:
            exact_url_text = f"Title: {url_matched_result.get('title', '')}\nSnippet: {url_matched_result.get('snippet', '')}"

        other_results_text = []
        for r in search_results[:10]:
            if 'linkedin.com' not in r.get('link', '').lower():
                other_results_text.append(f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}")

        llm_result = validate_role_with_llm(
            name, company, role, exact_url_text, other_results_text[:5], openrouter_api_key
        )

        if llm_result.get('success') and llm_result.get('role_pass'):
            result['checks']['role']['passed'] = True
            result['checks']['role']['method'] = 'llm'
            result['valid'] = True
            return result
        elif llm_result.get('success'):
            result['checks']['role']['method'] = 'llm'
            result['rejection_reason'] = 'role_llm_fail'
            return result
        else:
            result['checks']['role']['method'] = 'llm_error'
            result['rejection_reason'] = 'llm_error'
            return result

    # No LLM - rule-based failed
    result['checks']['role']['method'] = 'rule_failed'
    result['rejection_reason'] = 'role_rule_fail'
    return result


# ============================================================================
# VALIDATOR CLASS
# ============================================================================

class LeadValidator:
    """
    Lead validation class with built-in search capability.

    Usage:
        validator = LeadValidator(
            scrapingdog_api_key='xxx',
            openrouter_api_key='xxx'
        )
        result = validator.validate(lead_data)
    """

    def __init__(
        self,
        scrapingdog_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        use_llm: bool = True,
        use_q3: bool = True
    ):
        self.scrapingdog_api_key = scrapingdog_api_key
        self.openrouter_api_key = openrouter_api_key
        self.use_llm = use_llm
        self.use_q3 = use_q3

    def search_google(self, query: str, max_results: int = 10) -> Tuple[List[Dict], Optional[str]]:
        """Search Google via ScrapingDog."""
        if not self.scrapingdog_api_key:
            return [], "No ScrapingDog API key"

        for attempt in range(3):
            try:
                resp = requests.get('https://api.scrapingdog.com/google', params={
                    'api_key': self.scrapingdog_api_key,
                    'query': query,
                    'results': max_results
                }, timeout=45)
                if resp.status_code in (502, 503, 429):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return [], f"HTTP {resp.status_code} after 3 retries"
                if resp.status_code == 200:
                    data = resp.json()
                    return [{
                        'title': r.get('title', ''),
                        'snippet': r.get('snippet', ''),
                        'link': r.get('link', ''),
                        'missing': r.get('missing', [])
                    } for r in data.get('organic_results', [])], None
                else:
                    return [], f"HTTP {resp.status_code}"
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return [], f"Timeout after 3 retries"
            except Exception as e:
                return [], str(e)[:100]
        return [], "Max retries exhausted"

    def validate(self, lead: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a lead with automatic search.

        Args:
            lead: Lead data with keys: full_name, business, linkedin, city, state, country, role, email

        Returns:
            Validation result dict (see validate_lead for structure)
        """
        name = str(lead.get('full_name', '')).strip()
        company = str(lead.get('business', '')).strip()
        linkedin = str(lead.get('linkedin', '')).strip()

        # Search Q4: name + company + linkedin location
        q4_query = f'"{name}" "{company}" linkedin location'
        q4_results, q4_error = self.search_google(q4_query)

        # Find URL-matched result
        expected_lid = get_linkedin_id(linkedin)
        url_matched = None

        for r in q4_results:
            if get_linkedin_id(r.get('link', '')) == expected_lid:
                url_matched = r
                break

        # Q1 fallback if URL not found
        all_results = q4_results.copy()
        if not url_matched and expected_lid:
            q1_query = f'site:linkedin.com/in/{expected_lid}'
            q1_results, q1_error = self.search_google(q1_query)
            all_results.extend(q1_results)

            for r in q1_results:
                if get_linkedin_id(r.get('link', '')) == expected_lid:
                    url_matched = r
                    break

        # Run validation
        result = validate_lead(
            lead=lead,
            search_results=all_results,
            url_matched_result=url_matched,
            openrouter_api_key=self.openrouter_api_key,
            scrapingdog_api_key=self.scrapingdog_api_key,
            use_llm=self.use_llm,
            use_q3=self.use_q3
        )

        # Add search results to output
        result['search_results'] = all_results

        return result
