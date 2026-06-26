"""
Qualification System: Pre-Score Validation (Automatic Zero Checks)

This module implements deterministic pre-checks that run BEFORE any
LLM scoring for the company-mode model competition.  If a company
output fails any of these checks, it automatically receives a score
of 0 and the failure reason is recorded.

These checks are designed to be:
- Fast (no external API calls)
- Deterministic (same input always produces same output)
- Configurable (thresholds from CONFIG)

Company-mode pre-checks (``run_company_zero_checks``):
1. Per-lead HARD time limit (30 seconds) — instant fail safety net
2. Industry fuzzy match (80% threshold)
3. Sub-industry fuzzy match (70% threshold) — only if both sides set
4. Country match
5. Company-shaped data quality (placeholder text, suspicious chars)
6. Duplicate company handling (first surface per company wins)

NOTE: Cost and time VARIABILITY is handled via penalties in
lead_scorer.py.  Role / seniority / email validation are intentionally
ABSENT here — see module-level comment at the top of lead_scorer.py for
the May 2026 company-mode cutover rationale.

CRITICAL: This is validator-side qualification logic.
Do NOT modify any existing validation in validator_models/automated_checks.py
(which is fulfillment-side).
"""

import re
import logging
from typing import Tuple, Optional, Set, NamedTuple, List, Dict

try:
    from rapidfuzz import fuzz
except ImportError:
    # Fallback to basic fuzzy matching if rapidfuzz not installed
    import difflib
    class FuzzFallback:
        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            # Simple partial match approximation
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            return difflib.SequenceMatcher(None, s1, s2).ratio() * 100
    fuzz = FuzzFallback()

from gateway.qualification.config import CONFIG
from gateway.qualification.models import LeadOutput, ICPPrompt, CompanyOutput

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

# Fuzzy matching thresholds
INDUSTRY_MATCH_THRESHOLD = 80  # 80% for industry
SUB_INDUSTRY_MATCH_THRESHOLD = 70  # 70% for sub-industry


def _norm_industry(label: str) -> str:
    """Normalize LinkedIn/ICP industry labels for deterministic broad buckets."""
    normalized = (label or "").lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_INDUSTRY_BUCKETS: Dict[str, set[str]] = {
    "software": {
        "software", "information technology", "artificial intelligence",
        "data and analytics", "privacy and security",
        "software development", "computer software",
        "it services and it consulting", "information technology and services",
        "information services", "internet", "internet publishing",
        "technology information and internet", "technology information and media",
        "computer and network security", "network security",
        "computer networking", "computer networking products", "computer games",
        "cloud computing", "data infrastructure and analytics",
        "mobile computing software products", "embedded software products",
        "machine learning", "e-learning providers",
    },
    "finance": {
        "financial services", "lending and investments", "payments",
        "banking", "capital markets", "investment management",
        "investment banking", "venture capital and private equity principals",
        "insurance", "fintech",
    },
    "health_bio": {
        "health care", "biotechnology",
        "hospitals and health care", "biotechnology research",
        "pharmaceutical manufacturing", "pharmaceuticals", "medical device",
        "medical devices", "medical equipment manufacturing", "medical practices",
        "mental health care", "wellness and fitness services",
    },
    "hardware_mfg": {
        "hardware", "manufacturing",
        "computer hardware", "semiconductor manufacturing", "semiconductors",
        "electrical and electronic manufacturing", "electronics",
        "appliances electrical and electronics manufacturing",
        "industrial machinery manufacturing", "machinery", "robotics",
        "automation machinery manufacturing", "nanotechnology research",
        "defense and space manufacturing",
        "aviation and aerospace component manufacturing",
        "computers and electronics manufacturing",
    },
    "commerce_retail": {
        "commerce and shopping", "retail", "e-commerce",
        "consumer goods", "wholesale", "retail apparel and fashion",
        "food and beverage services", "consumer services",
    },
    "prof_services": {
        "professional services", "business consulting and services",
        "management consulting", "legal services", "accounting",
        "staffing and recruiting", "outsourcing and offshoring consulting",
    },
    "marketing_ad": {
        "advertising", "sales and marketing", "advertising services",
        "marketing services", "marketing and advertising",
        "public relations and communications services",
    },
    "real_estate": {
        "real estate", "leasing real estate", "commercial real estate",
        "real estate and equipment rental services",
    },
    "energy": {
        "energy", "oil and gas", "utilities", "renewables and environment",
        "renewable energy semiconductor manufacturing",
        "electric power generation", "services for renewable energy",
    },
    "transportation": {
        "transportation", "transportation logistics supply chain and storage",
        "truck transportation", "airlines and aviation", "freight and package transportation",
        "logistics and supply chain",
    },
    "education": {
        "education", "education administration programs", "higher education",
        "primary and secondary education",
    },
}

_INDUSTRY_TO_BUCKET: Dict[str, str] = {
    _norm_industry(name): bucket
    for bucket, names in _INDUSTRY_BUCKETS.items()
    for name in names
}


def _industry_bucket(label: str) -> Optional[str]:
    return _INDUSTRY_TO_BUCKET.get(_norm_industry(label))

# Placeholder text patterns that indicate fake/test data
PLACEHOLDER_PATTERNS: List[str] = [
    "test", "asdf", "xxx", "sample", "example", "lorem", "ipsum",
    "foo", "bar", "baz", "qwerty", "dummy", "fake", "placeholder",
    "demo", "temp", "null", "undefined", "n/a", "na", "none",
    "tbd", "todo", "fixme", "testing", "aaa", "bbb", "ccc",
]

# Suspicious characters that shouldn't appear in professional data
SUSPICIOUS_CHAR_PATTERN = re.compile(r'[<>{}|\\\^~`\[\]]')


# =============================================================================
# Main Validation Function — Company-Mode
# =============================================================================

async def run_company_zero_checks(
    company: CompanyOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
) -> Tuple[bool, Optional[str]]:
    """Deterministic pre-checks that run BEFORE any LLM scoring in
    the company-mode model competition.

    Checks (in order):

      * Hard time limit (30s safety net)
      * Industry fuzzy match (80%)
      * Sub-industry fuzzy match (70%) — only enforced if the ICP
        provides a sub-industry AND the model provides one
        (sub-industry is optional on CompanyOutput).
      * Country match
      * Company-shaped data quality (placeholder text, suspicious
        chars in company name / website)
      * Duplicate company tracking, keyed off ``company.company_name``

    There is intentionally no role / seniority / email / contact
    check — the model competition surfaces companies, not contacts.

    Returns ``(passed, failure_reason)``.
    """
    # Check 1: HARD time limit (30s safety net) — unchanged from lead-mode.
    result = check_hard_time_limit(run_time_seconds)
    if not result.passed:
        logger.info(f"Company failed hard time limit: {result.reason}")
        return False, result.reason

    # Check 2: Industry fuzzy match (80% threshold) — unchanged.
    result = check_industry_match(company.industry, icp.industry)
    if not result.passed:
        logger.info(f"Company failed industry check: {result.reason}")
        return False, result.reason

    # Check 3: Sub-industry fuzzy match — relaxed.  CompanyOutput marks
    # sub_industry as optional; if either side is empty we skip the check
    # rather than failing (industry fit is the primary filter).
    if (icp.sub_industry and icp.sub_industry.strip()
            and company.sub_industry and company.sub_industry.strip()):
        result = check_sub_industry_match(company.sub_industry, icp.sub_industry)
        if not result.passed:
            logger.info(f"Company failed sub-industry check: {result.reason}")
            return False, result.reason

    # Check 4: Country match — unchanged.
    result = check_country_match(company.country, icp.country)
    if not result.passed:
        logger.info(f"Company failed country check: {result.reason}")
        return False, result.reason

    # Check 5: Company-shaped data quality — light check for placeholder
    # text in the company name and website fields.  We do NOT call
    # check_data_quality(lead) here because that function reads
    # role / seniority / etc. that don't exist on CompanyOutput.
    quality_valid, quality_reason = _check_company_data_quality(company)
    if not quality_valid:
        logger.info(f"Company failed data quality check: {quality_reason}")
        return False, f"Data quality issue: {quality_reason}"

    # Check 6: Duplicate company tracking — first surface wins.
    result = check_duplicate_company(company.company_name, seen_companies)
    if not result.passed:
        logger.info(f"Company failed duplicate check: {result.reason}")
        return False, result.reason

    logger.debug(f"Company passed all company-mode pre-checks: {company.company_name}")
    return True, None


def _check_company_data_quality(
    company: CompanyOutput,
) -> Tuple[bool, Optional[str]]:
    """Light data-quality check for CompanyOutput.

    We only verify the two free-form text fields that get used in
    downstream prompts: company name and company website.  Both must:
      * be non-empty,
      * not match obvious placeholder text ('test', 'foo', etc.),
      * not contain suspicious characters that indicate templated junk.

    Industry / sub-industry / country are already enforced by their own
    fuzzy / exact-match checks elsewhere in run_company_zero_checks, so
    we don't re-validate them here.
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
    """
    Check 1: Verify lead didn't exceed HARD time limit (30s safety net).
    
    This is the ONLY automatic-zero check for time. The 30-second limit
    is a safety net to prevent runaway processes.
    
    Cost and time VARIABILITY penalties are handled separately in lead_scorer.py:
    - NO penalty if within budget (cost ≤ $0.05, time ≤ 8s)
    - 5-point penalty if cost > 2× budget or time > 2× budget
    
    Args:
        run_time_seconds: Total processing time for this lead
    
    Returns:
        ValidationResult with pass/fail and reason
    """
    if run_time_seconds > CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS:
        return ValidationResult(
            passed=False,
            reason=f"Exceeded HARD time limit: {run_time_seconds:.1f}s > {CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS}s (instant fail)"
        )
    return ValidationResult(passed=True)


# DEPRECATED - kept for backwards compatibility
def check_cost_limit(run_cost_usd: float) -> ValidationResult:
    """
    DEPRECATED: Cost limits are now handled via variability penalties in lead_scorer.py.
    
    This function always returns passed=True. Cost variability is penalized
    with 5 points if cost > 2× MAX_COST_PER_LEAD_USD.
    """
    # No longer enforced as automatic zero - variability penalties handle this
    return ValidationResult(passed=True)


def check_time_limit(run_time_seconds: float) -> ValidationResult:
    """
    DEPRECATED: Soft time limits are now handled via variability penalties in lead_scorer.py.
    
    Use check_hard_time_limit() for the 30s instant-fail safety net.
    Time variability is penalized with 5 points if time > 2× MAX_TIME_PER_LEAD_SECONDS.
    """
    # Delegate to hard time limit check
    return check_hard_time_limit(run_time_seconds)


def check_industry_match(lead_industry: str, icp_industry: str) -> ValidationResult:
    """
    Check 3: Verify lead's industry matches ICP (80% fuzzy match threshold).
    
    Args:
        lead_industry: Industry from the lead
        icp_industry: Target industry from the ICP
    
    Returns:
        ValidationResult with pass/fail and reason
    """
    if not lead_industry or not icp_industry:
        return ValidationResult(
            passed=False,
            reason="Missing industry field"
        )
    
    lead_bucket = _industry_bucket(lead_industry)
    icp_bucket = _industry_bucket(icp_industry)
    if lead_bucket is not None and lead_bucket == icp_bucket:
        return ValidationResult(passed=True)

    score = fuzz.ratio(lead_industry.lower().strip(), icp_industry.lower().strip())
    
    if score < INDUSTRY_MATCH_THRESHOLD:
        return ValidationResult(
            passed=False,
            reason=f"Industry mismatch: '{lead_industry}' vs '{icp_industry}' (score: {score:.0f}%, threshold: {INDUSTRY_MATCH_THRESHOLD}%)"
        )
    return ValidationResult(passed=True)


def check_sub_industry_match(lead_sub_industry: str, icp_sub_industry: str) -> ValidationResult:
    """
    Check 4: Verify lead's sub-industry matches ICP (70% fuzzy match threshold).
    
    More lenient than industry since sub-industry naming varies more.
    
    Args:
        lead_sub_industry: Sub-industry from the lead
        icp_sub_industry: Target sub-industry from the ICP
    
    Returns:
        ValidationResult with pass/fail and reason
    """
    if not lead_sub_industry or not icp_sub_industry:
        return ValidationResult(
            passed=False,
            reason="Missing sub-industry field"
        )
    
    score = fuzz.ratio(lead_sub_industry.lower().strip(), icp_sub_industry.lower().strip())
    
    if score < SUB_INDUSTRY_MATCH_THRESHOLD:
        return ValidationResult(
            passed=False,
            reason=f"Sub-industry mismatch: '{lead_sub_industry}' vs '{icp_sub_industry}' (score: {score:.0f}%, threshold: {SUB_INDUSTRY_MATCH_THRESHOLD}%)"
        )
    return ValidationResult(passed=True)


# --- Role / seniority / email checks REMOVED (May 2026 company-mode cutover) ---
# ``check_role_match``, ``check_seniority_match``, ``validate_email``,
# ``validate_email_sync``, ``check_data_quality`` (lead-shaped),
# ``validate_lead_batch``, ``get_check_names`` and
# ``summarize_validation_results`` have all been removed.  The
# company-mode pipeline does not have contact-level fields to check.
# Industry / sub-industry / country / duplicate-company checks below
# are still used (by ``run_company_zero_checks``).


_COUNTRY_ALIASES: dict = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "united states of america": "united states",
    "uk": "united kingdom", "great britain": "united kingdom",
    "england": "united kingdom", "u.k.": "united kingdom",
    "uae": "united arab emirates",
    "south korea": "korea, republic of", "republic of korea": "korea, republic of",
    "russia": "russian federation",
    "taiwan": "taiwan, province of china",
    "czech republic": "czechia",
    "holland": "netherlands",
}


def _normalize_country(name: str) -> str:
    """Normalize country name to handle common aliases."""
    stripped = name.strip().lower()
    return _COUNTRY_ALIASES.get(stripped, stripped)


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
    
    if _normalize_country(lead_country) != _normalize_country(icp_country):
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


# =============================================================================
# Email validation / lead-shaped data quality / batch validation REMOVED
# =============================================================================
# Removed in the May 2026 company-mode cutover:
#   * ``validate_email`` / ``validate_email_sync`` (no email on CompanyOutput)
#   * ``check_data_quality(lead)``                (lead-shaped, replaced by
#                                                  ``_check_company_data_quality``
#                                                  above)
#   * ``validate_lead_batch``                     (no per-lead pre-check
#                                                  batch — the scorer loops
#                                                  CompanyOutput rows directly)
#   * ``get_check_names`` / ``summarize_validation_results``
#                                                 (only callers were
#                                                  validate_lead_batch /
#                                                  ad-hoc test harnesses)
# =============================================================================
