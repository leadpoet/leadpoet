"""
Geographic Field Normalization
==============================

Standardizes city, state, and country fields for consistent storage.

Features:
- Infers country from state if missing (e.g., "CA" -> "United States")
- Uses `us` library for US states (no hardcoding)
- Uses `geonamescache` for cities (100k+ cities with alternate names)
- Validates against VALID_COUNTRIES list

Libraries used (NO rate limits - both are offline):
- us: Pure Python US state data
- geonamescache: Offline city database with 786k+ name variations
"""

from typing import Tuple

import us
import geonamescache

# ============================================================
# Initialize geonamescache (one-time load at import)
# ============================================================

_gc = geonamescache.GeonamesCache(min_city_population=1000)
_all_cities = _gc.get_cities()

# Build city lookup: alternate names -> canonical name
# Handles aliases like "Bombay" -> "Mumbai", "Peking" -> "Beijing"
CITY_LOOKUP = {}
for _city_id, _city_data in _all_cities.items():
    _canonical = _city_data['name']
    # Add canonical name
    CITY_LOOKUP[_canonical.lower()] = _canonical
    # Add alternate names if available
    _alt_names = _city_data.get('alternatenames', [])
    # Handle both list and string formats (version compatibility)
    if isinstance(_alt_names, str):
        _alt_names = _alt_names.split(',')
    for _alt in _alt_names:
        _alt_clean = _alt.strip().lower()
        if _alt_clean and len(_alt_clean) >= 2:
            # Don't overwrite if canonical exists
            if _alt_clean not in CITY_LOOKUP:
                CITY_LOOKUP[_alt_clean] = _canonical

# Clean up temporary variables
del _gc, _all_cities, _city_id, _city_data, _canonical, _alt_names, _alt, _alt_clean


# ============================================================
# Country Data
# ============================================================

COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "america": "united states",
    "united states of america": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom",
    "great britain": "united kingdom", "britain": "united kingdom",
    "england": "united kingdom", "scotland": "united kingdom",
    "wales": "united kingdom", "northern ireland": "united kingdom",
    "uae": "united arab emirates", "u.a.e.": "united arab emirates",
    "emirates": "united arab emirates",
    "korea": "south korea", "republic of korea": "south korea",
    "holland": "netherlands", "the netherlands": "netherlands",
    "deutschland": "germany", "brasil": "brazil",
    "espana": "spain", "españa": "spain", "italia": "italy",
    "nippon": "japan", "nihon": "japan",
    "prc": "china", "russia": "russia", "russian federation": "russia",
    "czech": "czech republic", "vatican": "vatican city", "holy see": "vatican city",
    "burma": "myanmar", "persia": "iran", "swaziland": "eswatini",
    "congo": "republic of the congo", "drc": "democratic republic of the congo",
}

VALID_COUNTRIES = {
    "united states", "canada", "mexico",
    "guatemala", "belize", "honduras", "el salvador", "nicaragua", "costa rica", "panama",
    "cuba", "jamaica", "haiti", "dominican republic", "bahamas", "barbados", "trinidad and tobago",
    "saint lucia", "grenada", "saint vincent and the grenadines", "antigua and barbuda",
    "dominica", "saint kitts and nevis",
    "brazil", "argentina", "colombia", "peru", "venezuela", "chile", "ecuador", "bolivia",
    "paraguay", "uruguay", "guyana", "suriname",
    "united kingdom", "germany", "france", "italy", "spain", "portugal", "netherlands",
    "belgium", "switzerland", "austria", "ireland", "luxembourg", "monaco", "andorra",
    "liechtenstein", "san marino", "vatican city",
    "sweden", "norway", "denmark", "finland", "iceland",
    "poland", "czech republic", "czechia", "hungary", "romania", "bulgaria", "ukraine",
    "belarus", "moldova", "slovakia", "slovenia", "croatia", "serbia", "bosnia and herzegovina",
    "montenegro", "north macedonia", "albania", "kosovo", "lithuania", "latvia", "estonia",
    "greece", "cyprus", "malta",
    "russia", "kazakhstan", "uzbekistan", "turkmenistan", "tajikistan", "kyrgyzstan",
    "georgia", "armenia", "azerbaijan",
    "turkey", "israel", "palestine", "lebanon", "jordan", "syria", "iraq", "iran",
    "saudi arabia", "united arab emirates", "qatar", "kuwait", "bahrain", "oman", "yemen",
    "india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives", "afghanistan",
    "indonesia", "malaysia", "singapore", "thailand", "vietnam", "philippines", "myanmar",
    "cambodia", "laos", "brunei", "timor-leste",
    "china", "japan", "south korea", "taiwan", "mongolia", "hong kong", "macau",
    "australia", "new zealand", "fiji", "papua new guinea", "solomon islands", "vanuatu",
    "samoa", "tonga", "kiribati", "micronesia", "palau", "marshall islands", "nauru", "tuvalu",
    "egypt", "libya", "tunisia", "algeria", "morocco",
    "nigeria", "ghana", "senegal", "ivory coast", "mali", "burkina faso", "niger", "guinea",
    "benin", "togo", "sierra leone", "liberia", "mauritania", "gambia", "guinea-bissau", "cape verde",
    "democratic republic of the congo", "cameroon", "central african republic", "chad",
    "republic of the congo", "gabon", "equatorial guinea", "sao tome and principe",
    "kenya", "ethiopia", "tanzania", "uganda", "rwanda", "burundi", "south sudan", "sudan",
    "eritrea", "djibouti", "somalia", "comoros", "mauritius", "seychelles", "madagascar",
    "south africa", "namibia", "botswana", "zimbabwe", "zambia", "malawi", "mozambique",
    "angola", "lesotho", "eswatini"
}

SPECIAL_CAPITALIZATION = {
    "united states": "United States",
    "united kingdom": "United Kingdom",
    "united arab emirates": "United Arab Emirates",
    "south korea": "South Korea",
    "south africa": "South Africa",
    "south sudan": "South Sudan",
    "new zealand": "New Zealand",
    "saudi arabia": "Saudi Arabia",
    "sri lanka": "Sri Lanka",
    "hong kong": "Hong Kong",
    "costa rica": "Costa Rica",
    "el salvador": "El Salvador",
    "dominican republic": "Dominican Republic",
    "czech republic": "Czech Republic",
    "north macedonia": "North Macedonia",
    "san marino": "San Marino",
    "vatican city": "Vatican City",
    "papua new guinea": "Papua New Guinea",
    "solomon islands": "Solomon Islands",
    "marshall islands": "Marshall Islands",
    "burkina faso": "Burkina Faso",
    "sierra leone": "Sierra Leone",
    "cape verde": "Cape Verde",
    "ivory coast": "Ivory Coast",
    "central african republic": "Central African Republic",
    "equatorial guinea": "Equatorial Guinea",
    "trinidad and tobago": "Trinidad And Tobago",
    "saint lucia": "Saint Lucia",
    "saint vincent and the grenadines": "Saint Vincent And The Grenadines",
    "antigua and barbuda": "Antigua And Barbuda",
    "saint kitts and nevis": "Saint Kitts And Nevis",
    "bosnia and herzegovina": "Bosnia And Herzegovina",
    "sao tome and principe": "Sao Tome And Principe",
    "democratic republic of the congo": "Democratic Republic Of The Congo",
    "republic of the congo": "Republic Of The Congo",
}


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


# ============================================================
# City Aliases (for abbreviations not in geonamescache)
# ============================================================

CITY_ALIASES = {
    # Common US abbreviations
    "sf": "San Francisco", "san fran": "San Francisco",
    "la": "Los Angeles", "l.a.": "Los Angeles",
    "nyc": "New York City", "n.y.c.": "New York City", 
    "new york": "New York City", "new york city": "New York City",  # Override geonamescache "Niu-York"
    "dc": "Washington", "d.c.": "Washington", "washington dc": "Washington",
    "philly": "Philadelphia", "vegas": "Las Vegas",
    "chi": "Chicago", "chi-town": "Chicago",
    "atl": "Atlanta", "bos": "Boston", "dal": "Dallas",
    "hou": "Houston", "htx": "Houston", "mia": "Miami",
    "phx": "Phoenix", "det": "Detroit", "den": "Denver",
    "sea": "Seattle", "pdx": "Portland", "sj": "San Jose",
    "sd": "San Diego", "sac": "Sacramento", "stl": "St. Louis",
    "nola": "New Orleans", "slc": "Salt Lake City",
    "indy": "Indianapolis", "kc": "Kansas City", "msp": "Minneapolis",
    # Common international abbreviations
    "ldn": "London", "lon": "London", "hk": "Hong Kong", "sg": "Singapore",
    "syd": "Sydney", "mel": "Melbourne", "tor": "Toronto",
    "van": "Vancouver", "mtl": "Montreal", "tyo": "Tokyo", "osa": "Osaka",
    "sel": "Seoul", "pek": "Beijing", "sha": "Shanghai",
    "bkk": "Bangkok", "sin": "Singapore", "kul": "Kuala Lumpur",
    "del": "Delhi", "bom": "Mumbai", "blr": "Bengaluru",
    "dxb": "Dubai", "auh": "Abu Dhabi",
    "par": "Paris", "ber": "Berlin", "muc": "Munich", "ffm": "Frankfurt",
    "ams": "Amsterdam", "bcn": "Barcelona", "mad": "Madrid",
    "rom": "Rome", "mil": "Milan", "dub": "Dublin",
    # Override geonamescache errors (actual city names mapping to wrong cities)
    "sao paulo": "Sao Paulo", "são paulo": "Sao Paulo",  # Was mapping to "Diamante"
    "cairo": "Cairo",  # Ensure Cairo maps correctly
    "geneva": "Geneva", "geneve": "Geneva", "genève": "Geneva",
    "montreal": "Montreal", "montréal": "Montreal",
    "zurich": "Zurich", "zürich": "Zurich",
    "bogota": "Bogota", "bogotá": "Bogota",
    # Critical: actual city names were mapping to completely wrong cities
    "cancun": "Cancun", "cancún": "Cancun",  # Was "Changchun"
    "brasilia": "Brasilia", "brasília": "Brasilia",  # Was "Porecatu"
    # German cities - standardize to English spellings
    "cologne": "Cologne", "köln": "Cologne", "koln": "Cologne",
    "dusseldorf": "Dusseldorf", "düsseldorf": "Dusseldorf",
    "munich": "Munich", "münchen": "Munich",
    # Other standardizations (actual city names, not airport codes)
    "thessaloniki": "Thessaloniki", "thessaloníki": "Thessaloniki",
    "medellin": "Medellin", "medellín": "Medellin",
    # BLOCK airport codes from being mapped to wrong cities by geonamescache
    # These pass through as-is (bad data stays bad, doesn't silently corrupt)
    "brs": "BRS", "arn": "ARN", "eze": "EZE", "cgn": "CGN",
    "gva": "GVA", "zrh": "ZRH", "cai": "CAI", "muc": "MUC",
    "dus": "DUS", "skg": "SKG", "mde": "MDE", "bog": "BOG",
    "cun": "CUN", "bsb": "BSB",
}


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
    
    # Validate against known countries
    if country_lower not in VALID_COUNTRIES:
        # Unknown country - just title case it
        return country.strip().title()
    
    # Special capitalization for multi-word countries
    if country_lower in SPECIAL_CAPITALIZATION:
        return SPECIAL_CAPITALIZATION[country_lower]
    
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
    # Include common aliases that submit.py also recognizes
    US_COUNTRY_INDICATORS = [
        "united states", "us", "usa", "america", "u s", "u s a"
    ]
    is_us = not country_lower or any(ind in country_lower for ind in US_COUNTRY_INDICATORS)
    
    if is_us:
        # Use us library for official US states
        found = us.states.lookup(state.strip())
        if found:
            return found.name
    
    # Check international states/provinces
    if cleaned in INTERNATIONAL_STATES:
        return INTERNATIONAL_STATES[cleaned]
    
    # Fallback to title case
    return state.strip().title()


def normalize_city(city: str) -> str:
    """
    Normalize city name using:
    1. Fallback aliases (for abbreviations like SF, NYC)
    2. geonamescache lookup (786k+ variations, handles Bombay->Mumbai, etc.)
    3. Title case fallback
    """
    if not city:
        return ""
    
    cleaned = city.strip().lower().replace(".", "")
    
    # 1. Check fallback aliases first (abbreviations)
    if cleaned in CITY_ALIASES:
        return CITY_ALIASES[cleaned]
    
    # 2. Check geonamescache lookup (includes alternate names)
    if cleaned in CITY_LOOKUP:
        return CITY_LOOKUP[cleaned]
    
    # 3. Fallback to title case
    return city.strip().title()


def infer_country_from_state(norm_state: str) -> str:
    """
    Infer country from normalized state name.
    
    Uses:
    - us library for US states (no hardcoding)
    - NON_US_STATE_TO_COUNTRY for Canada/Australia/UK
    """
    if not norm_state:
        return ""
    
    # Check US states using library
    if us.states.lookup(norm_state):
        return "United States"
    
    # Check non-US states
    return NON_US_STATE_TO_COUNTRY.get(norm_state.lower(), "")


def normalize_location(city: str, state: str, country: str) -> Tuple[str, str, str]:
    """
    Normalize city and state fields. Country is passed through as-is.
    
    NOTE: Country normalization is NOT done here - submit.py has its own
    COUNTRY_ALIASES and VALID_COUNTRIES validation. Miners MUST submit
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
        if inferred.lower() in VALID_COUNTRIES:
            country = inferred
    
    # Step 3: Normalize city only
    norm_city = normalize_city(city)
    
    # Step 4: Return country as-is (submit.py handles validation/normalization)
    # Only strip whitespace, don't change the value
    return (norm_city, norm_state, country.strip())

