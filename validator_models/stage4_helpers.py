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

Location check order:
1. Structured city match (City, State format) → city_mismatch = direct fail
2. Flexible location match (word overlap, etc.)
3. City fallback (city in text)
4. Area mapping check → area_mismatch = direct fail
5. Non-LinkedIn fallback
6. Q3 fallback query: "{name}" "{company}" "{city}" "{url}"

Usage:
    from validator_models.stage4_helpers import (
        extract_location_from_text,
        check_locations_match,
        check_q3_location_fallback,
    )
"""

import re
import json
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

# Geo lookup for state validation
GEO_LOOKUP_PATH = GATEWAY_UTILS_PATH / 'geo_lookup_fast.json'
if GEO_LOOKUP_PATH.exists():
    with open(GEO_LOOKUP_PATH, 'r') as f:
        GEO_LOOKUP = json.load(f)
else:
    GEO_LOOKUP = {}

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
    """Normalize area name for comparison."""
    area = area.lower().strip()
    area = area.replace("greater ", "").replace(" metropolitan", "").replace(" metro", "").replace(" area", "")
    return area.strip()


def is_area_in_mappings(area: str) -> bool:
    """Check if area exists in our mappings."""
    area_norm = normalize_area_name(area)
    for area_key in AREA_MAPPINGS.keys():
        key_norm = normalize_area_name(area_key)
        if key_norm == area_norm or area_norm in key_norm or key_norm in area_norm:
            return True
    return False


def is_city_in_area_approved(city: str, area: str) -> bool:
    """Check if city is approved for given area."""
    area_norm = normalize_area_name(area)
    city_norm = strip_accents(city.lower().strip())
    for area_key, cities in AREA_MAPPINGS.items():
        key_norm = normalize_area_name(area_key)
        if key_norm == area_norm or area_norm in key_norm or key_norm in area_norm:
            if city_norm in [strip_accents(c.lower()) for c in cities]:
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
    s = re.sub(r'[^\w\s]', '', str(s).lower()).strip()
    for old, new in CITY_EQUIVALENTS.items():
        s = s.replace(old, new)
    return s


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
# NOTE: Role format validation (check_role_validity) has been moved to gateway
# (submit.py check_role_sanity function) to reject bad roles earlier and save API costs.
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
    match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-zA-Z\s]+)*)\.\s*\d+[KMk]?\s*(?:followers?|connections?|volgers?|collegamenti)', text)
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


def check_locations_match(extracted: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Check if extracted location matches ground truth.

    Returns:
        (is_match, match_method)
    """
    if not extracted and not ground_truth:
        return True, 'empty'
    if not extracted or not ground_truth:
        return False, 'no_extraction'
    ext_norm = normalize_location(extracted)
    gt_norm = normalize_location(ground_truth)
    if not ext_norm or not gt_norm:
        return False, 'empty_norm'
    if ext_norm == gt_norm:
        return True, 'exact'
    if gt_norm in ext_norm:
        return True, 'direct'
    if ext_norm.startswith(gt_norm) or gt_norm.startswith(ext_norm):
        return True, 'startswith'

    def strip_area_suffix(s):
        s = re.sub(r'\s+metropolitan\s+area$', '', s).strip()
        s = re.sub(r'\s+area$', '', s).strip()
        return s

    ext_stripped = strip_area_suffix(ext_norm)
    gt_stripped = strip_area_suffix(gt_norm)
    if ext_stripped and gt_stripped:
        if ext_stripped == gt_stripped:
            return True, 'suffix_match'
        if ext_stripped in gt_stripped or gt_stripped in ext_stripped:
            return True, 'suffix_match'

    if ground_truth:
        gt_city = str(ground_truth).split(',')[0].strip().lower()
        gt_city = CITY_EQUIVALENTS.get(gt_city, gt_city)
        if gt_city and len(gt_city) > 2 and gt_city in ext_norm:
            return True, 'city_extract'

    gt_city = str(ground_truth).split(',')[0].strip() if ground_truth else ''
    if gt_city and is_city_in_area_approved(gt_city, extracted):
        return True, 'area_mapping'

    if extracted:
        ext_city = str(extracted).split(',')[0].strip().lower()
        ext_city = CITY_EQUIVALENTS.get(ext_city, ext_city)
        if ext_city and len(ext_city) > 2 and ext_city in gt_norm:
            return True, 'ext_city_in_gt'

    ext_words = set(ext_norm.replace(',', ' ').split())
    # Only use city (first part) from ground truth for word overlap
    gt_city = ground_truth.split(',')[0].strip().lower() if ground_truth else ''
    gt_words = set(gt_city.split()) if gt_city and len(gt_city) > 2 else set()
    if gt_words and gt_words.issubset(ext_words):
        return True, 'word_overlap'

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
    scrapingdog_api_key: str
) -> Dict[str, Any]:
    """
    Q3 Location fallback query: "{name}" "{company}" "{city}" "{url}"

    Only checks exact URL match with missing=[] (all terms found).

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
        resp = requests.get('https://api.scrapingdog.com/google', params={
            'api_key': scrapingdog_api_key,
            'query': query,
            'results': 3,
            'country': 'us'
        }, timeout=30)

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

                if not missing:  # All search terms found including city
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
    extracted_loc = extract_location_from_text(full_text)

    # Validate extracted state for structured locations
    structured_loc_valid = False
    if extracted_loc and ',' in extracted_loc:
        parts = extracted_loc.split(',')
        if len(parts) >= 2:
            state_part = parts[1].strip()
            if state_part and not is_valid_state(state_part):
                non_us_valid = any(x.lower() in extracted_loc.lower() for x in [
                    'India', 'UK', 'Canada', 'Australia', 'Germany', 'France',
                    'England', 'Scotland', 'Karnataka', 'Maharashtra', 'Tamil Nadu',
                    'Area', 'Metropolitan', 'Greater'
                ])
                if not non_us_valid:
                    extracted_loc = ''
            else:
                structured_loc_valid = True

    result['checks']['location']['extracted'] = extracted_loc

    # Structured city check
    location_passed = False
    location_method = None

    if structured_loc_valid and extracted_loc and city:
        ext_city = extracted_loc.split(',')[0].strip().lower()
        claimed_city = city.lower().strip()
        city_match = (claimed_city in ext_city or ext_city in claimed_city)

        if city_match:
            location_passed = True
            location_method = 'structured_city_match'
        else:
            result['checks']['location']['method'] = 'city_mismatch'
            result['rejection_reason'] = 'city_mismatch'
            return result

    # Other location checks if not structured
    if not location_passed and city:
        gt_location = f"{city}, {state}, {country}".strip(', ')

        if not structured_loc_valid and extracted_loc:
            loc_match, loc_method = check_locations_match(extracted_loc, gt_location)
            if loc_match:
                location_passed = True
                location_method = loc_method

        if not location_passed:
            city_lower = city.lower().strip()
            if city_lower in full_text.lower():
                location_passed = True
                location_method = 'city_fallback'

        # Area check
        if not location_passed:
            area_match = re.search(r'(Greater\s+[\w\s]+|[\w\s]+\s+Metropolitan|[\w\s]+\s+Bay|[\w\s]+\s+Metro)\s*Area', full_text, re.IGNORECASE)
            if area_match:
                area_found = area_match.group(0).strip()
                if city_lower not in area_found.lower():
                    if is_city_in_area_approved(city, area_found):
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
                    loc_match, loc_method = check_locations_match(r_loc, gt_location)
                    if loc_match:
                        location_passed = True
                        location_method = f'non_linkedin_{loc_method}'
                        result['checks']['location']['extracted'] = r_loc
                        break
                # Removed: non-LinkedIn city fallback was too loose
                # It matched company HQ locations instead of person locations

    # Q3 location fallback when not found
    if not location_passed and use_q3 and scrapingdog_api_key and city and linkedin:
        result['checks']['location']['q3_called'] = True
        q3_result = check_q3_location_fallback(name, company, city, linkedin, scrapingdog_api_key)

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

        try:
            resp = requests.get('https://api.scrapingdog.com/google', params={
                'api_key': self.scrapingdog_api_key,
                'query': query,
                'results': max_results
            }, timeout=30)
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
        except Exception as e:
            return [], str(e)[:100]

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
