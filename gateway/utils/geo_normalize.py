"""
Geographic Field Normalization
==============================

Standardizes city, state, and country fields for consistent storage.

Features:
- Infers country from state if missing (e.g., "CA" -> "United States")
- Uses geo_lookup_fast.json for all lookups (no external libraries)
- Validates against VALID_COUNTRIES_SET (199 countries from JSON)

All data is loaded from geo_lookup_fast.json at import time (one-time load).
"""

from typing import Tuple
from pathlib import Path
import json


# ============================================================
# Load geo_lookup_fast.json (one-time load at import)
# ============================================================

_geo_path = Path(__file__).parent / "geo_lookup_fast.json"
with open(_geo_path, 'r', encoding='utf-8') as _f:
    _geo_data = json.load(_f)

# Build sets for O(1) validation lookups
VALID_COUNTRIES_SET = set(_geo_data['countries'])
US_STATES_SET = set(_geo_data['us_states'].keys())
US_CITIES_BY_STATE = {state: set(cities) for state, cities in _geo_data['us_states'].items()}
CITIES_BY_COUNTRY = {country: set(cities) for country, cities in _geo_data['cities'].items()}
STATE_ABBR_TO_NAME = _geo_data['state_abbr']

# Build proper case mapping for US states (lowercase -> Proper Case)
US_STATE_PROPER = {}
for _state in US_STATES_SET:
    if _state == 'district of columbia':
        US_STATE_PROPER[_state] = 'District of Columbia'
    else:
        US_STATE_PROPER[_state] = ' '.join(w.capitalize() for w in _state.split())

# Build abbreviation -> Proper Case mapping
STATE_ABBR_TO_PROPER = {abbr: US_STATE_PROPER[name] for abbr, name in STATE_ABBR_TO_NAME.items()}

del _geo_path, _f, _geo_data, _state


# ============================================================
# City aliases for validation - SPLIT INTO US AND INTERNATIONAL
# 
# US aliases are applied for US validation only.
# International aliases are applied for non-US validation only.
# This prevents aliases like 'bogota' -> 'bogotá' from breaking
# US cities like Bogota, NJ.
# ============================================================

# US-only city aliases (applied only for United States validation)
US_CITY_ALIASES = {
    # Common abbreviations
    'new york': 'new york city',
    'nyc': 'new york city',
    'city of new york': 'new york city',  # Official name variant
    'la': 'los angeles',
    'sf': 'san francisco',
    'dc': 'washington',
    'washington dc': 'washington',
    'washington d.c.': 'washington',
    'philly': 'philadelphia',
    'vegas': 'las vegas',

    # McLean, Virginia (space variant)
    'mc lean': 'mclean',

    # Saint/St variations
    # NOTE: 'saint louis' NOT aliased - Michigan has 'saint louis', Missouri has 'st. louis'
    # Adding an alias would break one or the other state's validation
    'st louis': 'st. louis',  # For Missouri (St. Louis without period - missing the dot)
    'saint peters': 'st. peters',
    'st peters': 'st. peters',
    'saint petersburg': 'st. petersburg',
    'st. augustine': 'saint augustine',
    'st augustine': 'saint augustine',
    'port st. lucie': 'port saint lucie',
    'port st lucie': 'port saint lucie',
    'st. paul': 'saint paul',
    'st paul': 'saint paul',

    # Fort/Ft variations
    'ft. lauderdale': 'fort lauderdale',
    'ft lauderdale': 'fort lauderdale',
    'ft. worth': 'fort worth',
    'ft worth': 'fort worth',
    'ft. myers': 'fort myers',
    'ft myers': 'fort myers',

    # Hawaii (with okina U+2018 or macron)
    'kihei': 'kīhei',
    'mililani': 'mililani town',
    'aiea': '\u2018aiea',
    'ewa beach': '\u2018ewa beach',
    'kalaheo': 'kalaheo hillside',
    'haleiwa': 'hale\u2018iwa',
    'pāhoa': 'pahoa',  # Macron variant -> JSON has non-macron

    # Township variations
    'lakewood township': 'lakewood',
    'woodbridge township': 'woodbridge',
    'springfield township': 'springfield',
    'freehold township': 'freehold',
    'marlboro township': 'marlboro',
    'plymouth township': 'plymouth',

    # Unique cities with suffixes
    'simsbury': 'simsbury center',
    'killingly': 'killingly center',
    'suffield': 'suffield depot',
    'deep river': 'deep river center',
    'harpswell': 'harpswell center',
    'stratham': 'stratham station',
    'north attleborough': 'north attleborough center',
    'loxahatchee': 'loxahatchee groves',
    'plainsboro': 'plainsboro center',

    # City name variations (CITIES only - no towns/villages/townships)
    'salt lake': 'salt lake city',  # Utah - common abbreviation for the city
    'st. augustine south': 'saint augustine south',  # FL - St. -> Saint formatting
}

# International city aliases (applied only for non-US validation)
# These may conflict with US city names, but that's OK because
# they're only applied when country != United States
INTERNATIONAL_CITY_ALIASES = {
    # === Switzerland ===
    'zurich': 'zürich',
    'zuerich': 'zürich',
    'geneve': 'genève',
    'lucerne': 'luzern',
    'st. gallen': 'sankt gallen',
    'st gallen': 'sankt gallen',
    'saint gallen': 'sankt gallen',
    'staefa': 'stäfa',
    'duebendorf': 'dübendorf',
    'rueschlikon': 'rüschlikon',
    'neuchatel': 'neuchâtel',
    'kuettigen': 'küttigen',
    'fuellinsdorf': 'füllinsdorf',
    'koeniz': 'köniz',
    'kuessnacht': 'küssnacht',
    'wadenswil': 'wädenswil',
    'waedenswil': 'wädenswil',

    # === Sweden ===
    'gothenburg': 'göteborg',
    'goteborg': 'göteborg',
    'lidingo': 'lidingö',
    'lidingoe': 'lidingö',
    'malmo': 'malmö',
    'malmoe': 'malmö',
    'soedertaelje': 'södertälje',
    'sodertälje': 'södertälje',
    'taeby': 'täby',
    'jonkoping': 'jönköping',
    'jonkoeping': 'jönköping',
    'norrkoping': 'norrköping',
    'norrkoeping': 'norrköping',
    'linkoping': 'linköping',
    'linkoeping': 'linköping',
    'gaevle': 'gävle',
    'gavle': 'gävle',
    'orebro': 'örebro',
    'oerebro': 'örebro',
    'vasteras': 'västerås',
    'vaesteras': 'västerås',
    'umea': 'umeå',
    'lulea': 'luleå',

    # === Denmark ===
    'kobenhavn': 'københavn',
    'koebenhavn': 'københavn',
    'copenhagen': 'københavn',

    # === Turkey ===
    'izmir': 'i\u0307zmir',
    'atasehir': 'ataşehir',
    'kadikoy': 'kadıköy',
    'kadikoey': 'kadıköy',
    'eskisehir': 'eskişehir',
    'sisli': 'şişli',
    'uskudar': 'üsküdar',

    # === India ===
    'bangalore': 'bengaluru',
    'bombay': 'mumbai',
    'gurgaon': 'gurugram',
    'ernakulam': 'kochi',
    'calcutta': 'kolkata',  # Old name

    # === Italy (English -> Italian names in JSON) ===
    'milano': 'milan',
    'roma': 'rome',
    'firenze': 'florence',
    'venezia': 'venice',
    'napoli': 'naples',
    'torino': 'turin',

    # === Belgium ===
    'antwerp': 'antwerpen',
    'bruges': 'brugge',
    'brussel': 'brussels',
    'ghent': 'gent',

    # === Germany ===
    'munich': 'münchen',
    'muenchen': 'münchen',
    'munchen': 'münchen',
    'nuremberg': 'nürnberg',
    'nuernberg': 'nürnberg',
    'dusseldorf': 'düsseldorf',
    'duesseldorf': 'düsseldorf',
    'cologne': 'köln',
    'koeln': 'köln',
    'frankfurt': 'frankfurt am main',
    'wurzburg': 'würzburg',
    'wuerzburg': 'würzburg',
    'tubingen': 'tübingen',
    'tuebingen': 'tübingen',
    'gottingen': 'göttingen',
    'goettingen': 'göttingen',
    'lubeck': 'lübeck',
    'luebeck': 'lübeck',
    'hanover': 'hannover',

    # === Austria ===
    'wien': 'vienna',

    # === Netherlands ===
    'den haag': 'the hague',

    # === Canada ===
    'montreal': 'montréal',

    # === UK ===
    'saint helens': 'st helens',

    # === Australia ===
    'brisbane city': 'brisbane',

    # === Brazil ===
    'sao paulo': 'são paulo',
    'brasilia': 'brasília',

    # === Colombia ===
    'bogota': 'bogotá',
    'puerto bogota': 'puerto bogotá',
    'medellin': 'medellín',

    # === Argentina ===
    'cordoba': 'córdoba',

    # === Spain ===
    'coruna': 'a coruña',
    'la coruna': 'a coruña',
    'malaga': 'málaga',
    'almeria': 'almería',
    'leon': 'león',
    'seville': 'sevilla',

    # === Poland ===
    'gdansk': 'gdańsk',
    'wroclaw': 'wrocław',
    'poznan': 'poznań',
    'lodz': 'łódź',
    'krakow': 'kraków',
    'cracow': 'kraków',

    # === Mexico ===
    'mexico city': 'ciudad de méxico',
    'ciudad de mexico': 'ciudad de méxico',
    'cdmx': 'ciudad de méxico',
    'cancun': 'cancún',
    'merida': 'mérida',
    'queretaro': 'querétaro',

    # === Chile ===
    'valparaiso': 'valparaíso',

    # === Morocco ===
    'marrakech': 'marrakesh',

    # === Iceland ===
    'reykjavik': 'reykjavík',

    # === Portugal ===
    'evora': 'évora',

    # === Vietnam ===
    'saigon': 'ho chi minh city',

    # === Philippines ===
    'cebu': 'cebu city',

    # === China (old names) ===
    'peking': 'beijing',
    'canton': 'guangzhou',

    # === Ukraine ===
    'kiev': 'kyiv',

    # === Romania ===
    'bucuresti': 'bucharest',

    # === Serbia ===
    'beograd': 'belgrade',

    # === Russia ===
    'st petersburg': 'saint petersburg',

    # === Saudi Arabia ===
    'mecca': 'makkah',
    'medina': 'madinah',

    # === UAE ===
    'ajman': 'ajman city',

    # === Cuba ===
    'la habana': 'havana',

    # === Egypt ===
    'sharm el sheikh': 'sharm el-sheikh',

    # === Costa Rica ===
    'san jose': 'san josé',

    # === Panama ===
    'panama city': 'panamá',
}


# ============================================================
# Country Data
# ============================================================

COUNTRY_ALIASES = {
    # USA
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "america": "united states",
    "united states of america": "united states",
    # UK
    "uk": "united kingdom", "u.k.": "united kingdom", "gb": "united kingdom",
    "great britain": "united kingdom", "britain": "united kingdom",
    "england": "united kingdom", "scotland": "united kingdom",
    "wales": "united kingdom", "northern ireland": "united kingdom",
    # UAE
    "uae": "united arab emirates", "u.a.e.": "united arab emirates",
    "emirates": "united arab emirates",
    # Korea
    "korea": "south korea", "republic of korea": "south korea", "rok": "south korea",
    # Others
    "holland": "netherlands", "the netherlands": "netherlands", "nederland": "netherlands",
    "deutschland": "germany", "brasil": "brazil",
    "espana": "spain", "españa": "spain", "italia": "italy",
    "nippon": "japan", "nihon": "japan",
    "prc": "china", "peoples republic of china": "china", "people's republic of china": "china",
    "roc": "taiwan", "republic of china": "taiwan",
    "russia": "russia", "russian federation": "russia",
    "czech": "czech republic", "czechia": "czech republic",
    "vatican": "vatican city", "holy see": "vatican city",
    "burma": "myanmar", "persia": "iran", "swaziland": "eswatini",
    "the gambia": "gambia", "eesti": "estonia",
    "côte d'ivoire": "ivory coast", "cote d'ivoire": "ivory coast",
    "congo": "republic of the congo",
    "drc": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",
    "zaire": "democratic republic of the congo",
}

# VALID_COUNTRIES is now VALID_COUNTRIES_SET (loaded from JSON above)

# Country name -> city lookup key mapping
# Used when cities are stored under a different key than the country name
COUNTRY_CITY_KEY_MAP = {
    'czech republic': 'czechia',  # Cities stored under 'czechia' not 'czech republic'
}

# SPECIAL_CAPITALIZATION removed - just use title()


# ============================================================
# State Data
# ============================================================

# US state aliases (abbreviations not in us library)
US_STATE_ALIASES = {
    "calif": "California", "cali": "California",
    "tex": "Texas", "penn": "Pennsylvania", "penna": "Pennsylvania",
    "mass": "Massachusetts", "wash": "Washington", "mich": "Michigan",
    "minn": "Minnesota", "conn": "Connecticut", "ariz": "Arizona",
    "tenn": "Tennessee", "wisc": "Wisconsin", "okla": "Oklahoma",
    "colo": "Colorado", "fla": "Florida",
    # DC is a territory, not handled by us library
    "dc": "District of Columbia", "d c": "District of Columbia",
    "district of columbia": "District of Columbia",
}

# Non-US states/provinces
INTERNATIONAL_STATES = {
    # Canada (13)
    "ab": "Alberta", "bc": "British Columbia", "mb": "Manitoba",
    "nb": "New Brunswick", "nl": "Newfoundland And Labrador",
    "ns": "Nova Scotia", "nt": "Northwest Territories", "nu": "Nunavut",
    "on": "Ontario", "pe": "Prince Edward Island", "qc": "Quebec",
    "sk": "Saskatchewan", "yt": "Yukon",
    # Australia (8)
    "nsw": "New South Wales", "vic": "Victoria", "qld": "Queensland",
    "wa": "Western Australia", "sa": "South Australia", "tas": "Tasmania",
    "act": "Australian Capital Territory",
    # Note: "nt" is both Canada (Northwest Territories) and Australia (Northern Territory)
    # Canada takes precedence since it's more common
    # UK (4)
    "eng": "England", "sco": "Scotland", "wal": "Wales", "nir": "Northern Ireland",
}

# Non-US state -> Country mapping (for inference)
NON_US_STATE_TO_COUNTRY = {
    # Canada
    "alberta": "Canada", "british columbia": "Canada", "manitoba": "Canada",
    "new brunswick": "Canada", "newfoundland and labrador": "Canada",
    "nova scotia": "Canada", "northwest territories": "Canada", "nunavut": "Canada",
    "ontario": "Canada", "prince edward island": "Canada", "quebec": "Canada",
    "saskatchewan": "Canada", "yukon": "Canada",
    # Australia
    "new south wales": "Australia", "victoria": "Australia", "queensland": "Australia",
    "western australia": "Australia", "south australia": "Australia", "tasmania": "Australia",
    "australian capital territory": "Australia", "northern territory": "Australia",
    # UK
    "england": "United Kingdom", "scotland": "United Kingdom",
    "wales": "United Kingdom", "northern ireland": "United Kingdom",
}


# CITY_ALIASES removed - use US_CITY_ALIASES + INTERNATIONAL_CITY_ALIASES + title case for display


# ============================================================
# Normalization Functions
# ============================================================

def normalize_country(country: str) -> str:
    """Normalize country name to canonical form."""
    if not country:
        return ""

    country_lower = country.strip().lower()

    # Check aliases
    if country_lower in COUNTRY_ALIASES:
        country_lower = COUNTRY_ALIASES[country_lower]

    # Return title case (works for all countries)
    return country_lower.title()


def normalize_state(state: str, country: str = "") -> str:
    """Normalize state/province name to canonical form."""
    if not state:
        return ""

    cleaned = state.strip().lower().replace(".", "")
    country_lower = country.lower().replace(".", "") if country else ""

    # Check US aliases first (e.g., "calif" -> "California")
    if cleaned in US_STATE_ALIASES:
        return US_STATE_ALIASES[cleaned]

    # Check if US (or unknown country - assume US for state lookup)
    US_COUNTRY_INDICATORS = [
        "united states", "us", "usa", "america", "u s", "u s a"
    ]
    is_us = not country_lower or any(ind in country_lower for ind in US_COUNTRY_INDICATORS)

    if is_us:
        # Check abbreviation first (e.g., "CA" -> "California")
        if cleaned in STATE_ABBR_TO_PROPER:
            return STATE_ABBR_TO_PROPER[cleaned]
        # Check full state name (e.g., "california" -> "California")
        if cleaned in US_STATE_PROPER:
            return US_STATE_PROPER[cleaned]

    # Check international states/provinces
    if cleaned in INTERNATIONAL_STATES:
        return INTERNATIONAL_STATES[cleaned]

    # Fallback to title case
    return state.strip().title()


def normalize_city(city: str) -> str:
    """
    Normalize city name using:
    1. US_CITY_ALIASES for US variations (nyc, sf, etc.)
    2. INTERNATIONAL_CITY_ALIASES for international variations
    3. Title case fallback
    
    Note: This is for display normalization. Validation uses separate
    alias dictionaries based on country to avoid conflicts.
    """
    if not city:
        return ""

    cleaned = city.strip().lower().replace(".", "")

    # Check US aliases first, then international
    if cleaned in US_CITY_ALIASES:
        return US_CITY_ALIASES[cleaned].title()
    if cleaned in INTERNATIONAL_CITY_ALIASES:
        return INTERNATIONAL_CITY_ALIASES[cleaned].title()

    # Fallback to title case
    return city.strip().title()


def infer_country_from_state(norm_state: str) -> str:
    """
    Infer country from normalized state name.

    Uses:
    - US_STATES_SET for US states
    - NON_US_STATE_TO_COUNTRY for Canada/Australia/UK
    """
    if not norm_state:
        return ""

    # Check if it's a US state
    if norm_state.lower() in US_STATES_SET:
        return "United States"

    # Check non-US states
    return NON_US_STATE_TO_COUNTRY.get(norm_state.lower(), "")


def normalize_location(city: str, state: str, country: str) -> Tuple[str, str, str]:
    """
    Normalize city and state fields. Country is passed through as-is.
    
    NOTE: Country normalization is NOT done here - submit.py has its own
    COUNTRY_ALIASES and VALID_COUNTRIES_SET validation. Miners MUST submit
    exact country names from the 199-country list (or valid aliases).
    
    Flow:
    1. Normalize state
    2. If country empty, infer from normalized state
    3. Normalize city
    4. Return country as-is (submit.py handles country validation)
    
    Args:
        city: Raw city input (e.g., "SF", "nyc", "Bombay")
        state: Raw state input (e.g., "CA", "calif", "ON")
        country: Raw country input - passed through unchanged
    
    Returns:
        Tuple of (normalized_city, normalized_state, country_as_is)
    
    Examples:
        ("SF", "CA", "USA") -> ("San Francisco", "California", "USA")  # country unchanged
        ("nyc", "ny", "") -> ("New York City", "New York", "United States")  # country inferred
        ("tor", "on", "") -> ("Toronto", "Ontario", "Canada")  # country inferred
    """
    # Step 1: Normalize state
    norm_state = normalize_state(state, country)
    
    # Step 2: Infer country if empty (useful feature - don't reject leads missing country)
    # NOTE: We infer the FULL country name here since miner didn't provide one
    if not country.strip() and norm_state:
        inferred = infer_country_from_state(norm_state)
        if inferred.lower() in VALID_COUNTRIES_SET:
            country = inferred
    
    # Step 3: Normalize city only
    norm_city = normalize_city(city)
    
    # Step 4: Return country as-is (submit.py handles validation/normalization)
    # Only strip whitespace, don't change the value
    return (norm_city, norm_state, country.strip())


# ============================================================
# Location Validation
# ============================================================

def _normalize_for_validation(city: str, is_us: bool = False) -> str:
    """
    Normalize city for validation lookup.
    
    Args:
        city: City name to normalize
        is_us: If True, apply US_CITY_ALIASES (for US validation).
               If False, apply INTERNATIONAL_CITY_ALIASES (for non-US).
               
    This split prevents international aliases (e.g., 'bogota' -> 'bogotá')
    from breaking US city validation (e.g., Bogota, NJ exists in the US).
    """
    if not city:
        return ""
    city_lower = city.lower().strip()
    if is_us:
        return US_CITY_ALIASES.get(city_lower, city_lower)
    else:
        return INTERNATIONAL_CITY_ALIASES.get(city_lower, city_lower)


def _normalize_state_for_validation(state: str) -> str:
    """Normalize state for validation lookup."""
    if not state:
        return ""
    state_lower = state.lower().strip().replace('.', '')
    # Filter out invalid state values
    if state_lower in ['us', 'usa', 'unknown', 'united states']:
        return ""
    # Check abbreviation mapping
    return STATE_ABBR_TO_NAME.get(state_lower, state_lower)


def validate_location(city: str, state: str, country: str) -> Tuple[bool, str]:
    """
    Validate that location exists in geo_lookup_fast.json.

    Checks:
    1. Country is in 199 valid countries
    2. For US: state is valid AND city exists in that state
    3. For international: city exists in that country

    Args:
        city: City name (will be normalized)
        state: State/province name (will be normalized)
        country: Country name (will be normalized)

    Returns:
        Tuple of (is_valid, rejection_reason)
        - (True, None) if valid
        - (False, "reason") if invalid

    Examples:
        ("Los Angeles", "CA", "USA") -> (True, None)
        ("LA", "California", "United States") -> (True, None)
        ("FakeCity", "CA", "USA") -> (False, "city_invalid_for_state")
        ("Berlin", "", "Germany") -> (True, None)
        ("FakeCity", "", "Germany") -> (False, "city_invalid_for_country")
        ("NYC", "NY", "Narnia") -> (False, "country_invalid")
    """
    # Normalize country
    if not country:
        return False, "country_empty"

    country_lower = country.lower().strip()
    # Apply country aliases
    country_lower = COUNTRY_ALIASES.get(country_lower, country_lower)

    # Check country is valid
    if country_lower not in VALID_COUNTRIES_SET:
        return False, "country_invalid"

    # US validation: check state AND city
    # Uses US_CITY_ALIASES (nyc -> new york city, sf -> san francisco, etc.)
    if country_lower == 'united states':
        city_norm = _normalize_for_validation(city, is_us=True)
        if not city_norm:
            return False, "city_empty"
            
        state_norm = _normalize_state_for_validation(state)

        if not state_norm:
            return False, "state_empty_for_usa"

        if state_norm not in US_STATES_SET:
            return False, "state_invalid"

        # Check city exists in that state
        state_cities = US_CITIES_BY_STATE.get(state_norm, set())
        if city_norm not in state_cities:
            return False, "city_invalid_for_state"

    # International validation: check city exists in country
    # Uses INTERNATIONAL_CITY_ALIASES (bogota -> bogotá, munich -> münchen, etc.)
    else:
        city_norm = _normalize_for_validation(city, is_us=False)
        if not city_norm:
            return False, "city_empty"
            
        # Handle country name -> city lookup key mapping (e.g., 'czech republic' -> 'czechia')
        city_lookup_key = COUNTRY_CITY_KEY_MAP.get(country_lower, country_lower)
        country_cities = CITIES_BY_COUNTRY.get(city_lookup_key, set())
        if city_norm not in country_cities:
            return False, "city_invalid_for_country"

    return True, None

