"""Arena company prechecks and shared country matching.

Company fit and contact verification run separately. Sourcing costs and
execution deadlines are enforced by the Arena broker and runner.
"""

import re
import logging
from typing import Any, Tuple, Optional, Set, NamedTuple, List, Dict

from gateway.qualification.config import CONFIG
from gateway.qualification.models import CompanyOutput
from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult,
    company_fit_match,
    company_fit_mismatch,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ValidationResult(NamedTuple):
    """Result of a validation check."""
    passed: bool
    reason: Optional[str] = None


# =============================================================================
# Configuration Constants
# =============================================================================

# Placeholder text patterns that indicate fake/test data
PLACEHOLDER_PATTERNS: List[str] = [
    "test", "asdf", "xxx", "sample", "example", "lorem", "ipsum",
    "foo", "bar", "baz", "qwerty", "dummy", "fake", "placeholder",
    "demo", "temp", "null", "undefined", "n/a", "na", "none",
    "tbd", "todo", "fixme", "testing", "aaa", "bbb", "ccc",
]

# Markup and template delimiters that should not appear in company names.
# A vertical bar is a common separator in legitimate display names.
SUSPICIOUS_CHAR_PATTERN = re.compile(r'[<>{}\\\^~`\[\]]')


# =============================================================================
# Main Validation Function — Company-Mode
# =============================================================================


async def run_company_zero_checks(
    company: CompanyOutput,
    run_time_seconds: float,
    seen_companies: Set[str],
    gate_receipts: Optional[List[Dict[str, Any]]] = None,
) -> CompanyFitDecisionResult:
    """Check company runtime, output quality, and duplicate identity."""
    def _mismatch(reason: Optional[str]) -> CompanyFitDecisionResult:
        decision = company_fit_mismatch(reason)
        if gate_receipts is not None:
            gate_receipts.append(decision.receipt("company_pre_checks"))
        return decision

    # Retain the existing verifier runtime safety limit.
    result = check_hard_time_limit(run_time_seconds)
    if not result.passed:
        logger.info(f"Company failed hard time limit: {result.reason}")
        return _mismatch(result.reason)

    # Reject placeholder company names and websites before evidence checks.
    quality_valid, quality_reason = _check_company_data_quality(company)
    if not quality_valid:
        logger.info(f"Company failed data quality check: {quality_reason}")
        return _mismatch(f"Data quality issue: {quality_reason}")

    # Check 6: Duplicate company tracking — first surface wins.
    result = check_duplicate_company(company.company_name, seen_companies)
    if not result.passed:
        logger.info(f"Company failed duplicate check: {result.reason}")
        return _mismatch(result.reason)

    logger.debug(f"Company passed all company-mode pre-checks: {company.company_name}")
    return company_fit_match()


def _check_company_data_quality(
    company: CompanyOutput,
) -> Tuple[bool, Optional[str]]:
    """Light data-quality check for CompanyOutput.

    We only verify the two free-form text fields that get used in
    downstream prompts: company name and company website.  Both must:
      * be non-empty,
      * not match obvious placeholder text ('test', 'foo', etc.),
      * not contain suspicious characters that indicate templated junk.

    Country, industry, and other company-fit requirements are checked by
    the shared company verifier.
    """
    name = (company.company_name or "").strip()
    if not name:
        return False, "Missing company_name"
    name_lower = name.lower()
    for placeholder in PLACEHOLDER_PATTERNS:
        # Whole-word match so a real name containing 'na' (e.g.
        # 'Nautical Inc') isn't tripped up.  Cheaper than re.search
        # because placeholders are simple identifiers.
        if name_lower == placeholder or name_lower.startswith(placeholder + " "):
            return False, f"company_name looks like placeholder text: {name!r}"
    if SUSPICIOUS_CHAR_PATTERN.search(name):
        return False, f"company_name contains suspicious characters: {name!r}"

    website = (company.company_website or "").strip()
    if not website:
        return False, "Missing company_website"
    # Pydantic already URL-normalized the website at parse time; we
    # only need to guard against placeholder-y URLs slipping past
    # parsing (e.g. 'https://example.com').
    if "example.com" in website.lower() or "example.org" in website.lower():
        return False, f"company_website is example/placeholder: {website!r}"

    return True, None


# =============================================================================
# Individual Check Functions
# =============================================================================

def check_hard_time_limit(run_time_seconds: float) -> ValidationResult:
    """Apply the existing company verifier runtime safety limit."""
    if run_time_seconds > CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS:
        return ValidationResult(
            passed=False,
            reason=f"Exceeded HARD time limit: {run_time_seconds:.1f}s > {CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS}s (instant fail)"
        )
    return ValidationResult(passed=True)


# Colloquial spellings that the ISO country database does not carry. This is
# alias glue only — the authoritative country list (names, alpha-2/alpha-3
# codes, continents) comes from geonamescache at runtime, never hand-typed.
# Targets are geonamescache canonical names, lowercased.
_COUNTRY_ALIASES: dict = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "united states of america": "united states",
    "america": "united states",
    "uk": "united kingdom", "great britain": "united kingdom",
    "england": "united kingdom", "u.k.": "united kingdom",
    "scotland": "united kingdom", "wales": "united kingdom",
    "northern ireland": "united kingdom",
    "uae": "united arab emirates",
    "korea": "south korea", "republic of korea": "south korea",
    "korea, republic of": "south korea",
    "russian federation": "russia",
    "taiwan, province of china": "taiwan",
    "czech republic": "czechia",
    "holland": "the netherlands", "netherlands": "the netherlands",
    "deutschland": "germany",
    "turkiye": "turkey", "türkiye": "turkey",
    "viet nam": "vietnam",
    "cote d'ivoire": "ivory coast", "côte d'ivoire": "ivory coast",
}

# The seven continents are a closed set; the member countries per continent
# come from geonamescache continent codes at runtime.
_CONTINENT_CODES: dict = {
    "europe": "EU", "north america": "NA", "south america": "SA",
    "asia": "AS", "africa": "AF", "oceania": "OC", "antarctica": "AN",
}

_GEO_LOOKUP_CACHE: dict = {}


def _country_lookup() -> dict:
    """name/alias/ISO-code -> canonical country name.

    Built once per process from the vendored ISO-3166 table
    (qualification/scoring/country_data.py, generated by
    scripts/generate_country_data.py — no geo library in the hot path).
    Alpha-2/alpha-3 codes are stored uppercase and only matched against
    uppercase input, so lowercase English words in geography prose
    ("in", "it", "no", "and") never resolve as ISO codes.
    """
    cached = _GEO_LOOKUP_CACHE.get("countries")
    if cached is not None:
        return cached
    from qualification.scoring.country_data import COUNTRIES

    lookup: dict = {}
    continents: dict = {}
    for name, iso2, iso3, continent in COUNTRIES:
        canonical = name.lower()
        lookup[canonical] = canonical
        if iso2:
            lookup[iso2.upper()] = canonical
        if iso3:
            lookup[iso3.upper()] = canonical
        if continent:
            continents.setdefault(continent.upper(), set()).add(canonical)
    for alias, target in _COUNTRY_ALIASES.items():
        lookup[alias] = target
    _GEO_LOOKUP_CACHE["countries"] = lookup
    _GEO_LOOKUP_CACHE["continents"] = {
        code: frozenset(members) for code, members in continents.items()
    }
    return lookup


def _continent_members(code: str) -> frozenset:
    _country_lookup()
    return _GEO_LOOKUP_CACHE["continents"].get(code, frozenset())


def _resolve_country(value: str) -> Optional[str]:
    """Resolve free text to a canonical country name, or None."""
    token = str(value or "").strip()
    if not token:
        return None
    lookup = _country_lookup()
    resolved = lookup.get(token.lower())
    if resolved is not None:
        return resolved
    # Official-style "The X" forms (The Bahamas, The Gambia, The Netherlands)
    # resolve to the same canonical country as their bare names.
    lowered = token.lower()
    if lowered.startswith("the ") and len(lowered) > 4:
        resolved = lookup.get(lowered[4:].strip())
        if resolved is not None:
            return resolved
    # ISO alpha-2/alpha-3 codes must be uppercase in the source text so
    # ordinary words in geography prose cannot masquerade as codes.
    if len(token) in (2, 3) and token.isupper():
        return lookup.get(token)
    return None


def _normalize_country(name: str) -> str:
    """Normalize a country string to its canonical form for comparison."""
    return _resolve_country(name) or str(name or "").strip().lower()


_GEO_TOKEN_SPLIT = re.compile(r",|/|&|\bor\b|\band\b", re.IGNORECASE)


def _allowed_countries_from_icp_geography(value: str) -> frozenset:
    """The set of canonical countries an ICP geography string permits.

    Handles the shapes production ICPs and client ICPs actually use:
    ``"United States, West Coast"`` (country + region),
    ``"London, United Kingdom"`` (city first), ``"United States or
    Canada"`` (multi-country), ``"Europe"`` (continent), and
    ``"Georgia, United States"`` (US state names that collide with
    country names — any US state in the string also permits the US).
    Returns an empty set when nothing resolves; the caller then defers
    geography to the ICP-fit scorer instead of hard-zeroing on a string
    that can never equal a country.
    """
    allowed: set = set()
    for raw_token in _GEO_TOKEN_SPLIT.split(str(value or "")):
        token = raw_token.strip()
        if not token:
            continue
        country = _resolve_country(token)
        if country is not None:
            allowed.add(country)
        continent_code = _CONTINENT_CODES.get(token.lower()) or (
            "EU" if token == "EU" else None
        )
        if continent_code:
            allowed.update(_continent_members(continent_code))
        us_state = _lookup_us_state(token)
        if us_state is not None:
            allowed.add("united states")
    return frozenset(allowed)


def _lookup_us_state(token: str):
    """US state name for a token, or None; abbreviations must be uppercase."""
    from qualification.scoring.country_data import US_STATES

    text = token.strip()
    if not text:
        return None
    if len(text) == 2:
        # USPS abbreviations only match uppercase so prose words never do.
        return US_STATES.get(text) if text.isupper() else None
    return US_STATES.get(text.lower())


def check_country_match(lead_country: str, icp_country: str) -> ValidationResult:
    """
    Verify lead's country matches ICP requirement.
    
    Uses case-insensitive matching with common alias normalization
    (e.g., USA = United States, UK = United Kingdom).
    If the ICP doesn't specify a country, any country is accepted.
    """
    if not icp_country or not icp_country.strip():
        return ValidationResult(passed=True)
    
    if not lead_country or not lead_country.strip():
        return ValidationResult(
            passed=False,
            reason=f"Missing country (ICP requires '{icp_country}')"
        )

    allowed = _allowed_countries_from_icp_geography(icp_country)
    if not allowed:
        geography_token = str(icp_country or "").strip()
        if (
            len(geography_token) in (2, 3)
            and geography_token.isalpha()
            and geography_token.isupper()
        ):
            # A bare code-shaped token (e.g. "ZZ") that resolves to NOTHING is
            # a broken requirement, not free prose — failing open here would
            # accept every company against an unknown code (PR-28 audit).
            return ValidationResult(
                passed=False,
                reason=(
                    f"Country mismatch: ICP geography '{icp_country}' is not "
                    "a recognized ISO country code"
                ),
            )
        # The ICP geography does not resolve to any recognized country or
        # continent (business regions like "EMEA", free prose). Hard-zeroing
        # here would reject every company, so geography nuance is scored by
        # the ICP-fit LLM instead — mirroring the industry-gate contract.
        logger.info(
            "Country pre-check deferred to ICP-fit scorer: ICP geography %r "
            "does not resolve to a recognized country",
            icp_country,
        )
        return ValidationResult(passed=True)
    if _normalize_country(lead_country) not in allowed:
        return ValidationResult(
            passed=False,
            reason=f"Country mismatch: '{lead_country}' vs ICP '{icp_country}'"
        )
    return ValidationResult(passed=True)


def check_duplicate_company(company_name: str, seen_companies: Set[str]) -> ValidationResult:
    """
    Check 8: Verify this company hasn't already been scored in this evaluation.
    
    First lead per company wins - subsequent leads for the same company are rejected.
    This prevents models from gaming the system by returning multiple leads
    for the same company.
    
    Args:
        company_name: Company name from the lead
        seen_companies: Set of company names already scored
    
    Returns:
        ValidationResult with pass/fail and reason
    """
    if not company_name:
        return ValidationResult(
            passed=False,
            reason="Missing company/business field"
        )
    
    company_key = company_name.lower().strip()
    
    if company_key in seen_companies:
        return ValidationResult(
            passed=False,
            reason=f"Duplicate company: '{company_name}' already scored this evaluation"
        )
    
    return ValidationResult(passed=True)
