"""
Qualification System: Pre-Score Validation (Automatic Zero Checks)

Phase 5.0 from tasks10.md

This module implements deterministic pre-checks that run BEFORE any LLM
scoring. If a lead fails any of these checks, it automatically receives
a score of 0 and the failure reason is recorded.

These checks are designed to be:
- Fast (no external API calls except email validation)
- Deterministic (same input always produces same output)
- Configurable (thresholds from CONFIG)

The 7 Automatic Zero Checks:
1. Per-lead HARD time limit (30 seconds) - instant fail safety net
2. Industry fuzzy match (80% threshold)
3. Sub-industry fuzzy match (70% threshold)
4. Role fuzzy match (60% threshold)
5. Email validation (format, disposable domains)
6. Data quality (placeholder text, suspicious chars, case issues)
7. Duplicate company handling (first lead per company wins)

NOTE: Cost and time VARIABILITY is handled via penalties in lead_scorer.py:
- NO penalty if cost ≤ average ($0.05) or time ≤ average (8s)
- 5-point penalty if cost > 2× average or time > 2× average
- This allows high-variability models to still succeed

CRITICAL: This is NEW validation logic for qualification only.
Do NOT modify any existing validation in validator_models/automated_checks.py.
"""

import re
import logging
from typing import Tuple, Optional, Set, NamedTuple, List

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
from gateway.qualification.models import LeadOutput, ICPPrompt

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
ROLE_MATCH_THRESHOLD = 60  # 60% for role (more lenient for variations)

# Known disposable email domains
DISPOSABLE_EMAIL_DOMAINS: Set[str] = {
    # Common disposable providers
    "tempmail.com", "throwaway.com", "mailinator.com", "10minutemail.com",
    "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "sharklasers.com", "spam4.me", "grr.la", "guerrillamailblock.com",
    "pokemail.net", "spam.la", "maildrop.cc", "yopmail.com", "yopmail.fr",
    "trashmail.com", "trashmail.net", "fakeinbox.com", "tempinbox.com",
    "discard.email", "throwawaymail.com", "getnada.com", "emailondeck.com",
    "tempail.com", "tempmailaddress.com", "burnermail.io", "mytrashmail.com",
    # Temporary mail services
    "33mail.com", "temp-mail.org", "10minemail.com", "dropmail.me",
    "mohmal.com", "guerrilla-mail.com", "crazymailing.com", "tempr.email",
    "dispostable.com", "fakemail.net", "inboxkitten.com", "minutemail.com",
}

# Placeholder text patterns that indicate fake/test data
PLACEHOLDER_PATTERNS: List[str] = [
    "test", "asdf", "xxx", "sample", "example", "lorem", "ipsum",
    "foo", "bar", "baz", "qwerty", "dummy", "fake", "placeholder",
    "demo", "temp", "null", "undefined", "n/a", "na", "none",
    "tbd", "todo", "fixme", "testing", "aaa", "bbb", "ccc",
]

# Suspicious characters that shouldn't appear in professional data
SUSPICIOUS_CHAR_PATTERN = re.compile(r'[<>{}|\\\^~`\[\]]')

# Email format regex
EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)


# =============================================================================
# Main Validation Function
# =============================================================================

async def run_automatic_zero_checks(
    lead: LeadOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str]
) -> Tuple[bool, Optional[str]]:
    """
    Run automatic-zero validation checks BEFORE LLM scoring.
    
    These checks are fast, deterministic, and must all pass for a lead
    to proceed to LLM-based scoring. If any check fails, the lead
    receives a score of 0.
    
    Args:
        lead: The lead output from the model
        icp: The ICP prompt that was used
        run_cost_usd: Total API cost for this lead so far
        run_time_seconds: Total time spent processing this lead
        seen_companies: Set of company names already scored in this evaluation
    
    Returns:
        Tuple of (passed: bool, failure_reason: Optional[str])
        If passed=False, the lead should score 0 with the given reason.
    """
    
    # Check 1: Per-lead HARD time limit (30s safety net)
    # NOTE: Cost is NOT checked here - variability penalties handle cost in lead_scorer.py
    result = check_hard_time_limit(run_time_seconds)
    if not result.passed:
        logger.info(f"Lead failed hard time limit: {result.reason}")
        return False, result.reason
    
    # Check 2: Industry fuzzy match (80% threshold)
    result = check_industry_match(lead.industry, icp.industry)
    if not result.passed:
        logger.info(f"Lead failed industry check: {result.reason}")
        return False, result.reason
    
    # Check 3: Sub-industry fuzzy match (70% threshold)
    result = check_sub_industry_match(lead.sub_industry, icp.sub_industry)
    if not result.passed:
        logger.info(f"Lead failed sub-industry check: {result.reason}")
        return False, result.reason
    
    # Check 4: Role fuzzy match (60% threshold) - uses target_roles array
    result = check_role_match(lead.role, icp.target_roles)
    if not result.passed:
        logger.info(f"Lead failed role check: {result.reason}")
        return False, result.reason
    
    # Check 5: Country match — ICP says "United States", lead must be in US
    result = check_country_match(lead.country, icp.country)
    if not result.passed:
        logger.info(f"Lead failed country check: {result.reason}")
        return False, result.reason
    
    # Check 6: Seniority level — lead must be within 1 level of ICP target
    result = check_seniority_match(
        lead.seniority.value if hasattr(lead.seniority, 'value') else str(lead.seniority),
        icp.target_seniority
    )
    if not result.passed:
        logger.info(f"Lead failed seniority check: {result.reason}")
        return False, result.reason
    
    # Check 7: Data quality (placeholder text, suspicious chars, case issues)
    quality_valid, quality_reason = check_data_quality(lead)
    if not quality_valid:
        logger.info(f"Lead failed data quality check: {quality_reason}")
        return False, f"Data quality issue: {quality_reason}"
    
    # Check 8: Duplicate company handling (first lead per company wins)
    result = check_duplicate_company(lead.business, seen_companies)
    if not result.passed:
        logger.info(f"Lead failed duplicate check: {result.reason}")
        return False, result.reason
    
    # All checks passed
    logger.debug(f"Lead passed all pre-checks: {lead.business} / {lead.role}")
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


def check_role_match(lead_role: str, icp_target_roles: list) -> ValidationResult:
    """
    Check 5: Verify lead's role matches ANY ICP target role (60% partial match threshold).
    
    Uses partial_ratio for more lenient matching since role titles vary widely.
    Examples: "VP of Sales" matches "VP Sales" or "Vice President of Sales"
    
    Args:
        lead_role: Role from the lead
        icp_target_roles: List of target roles from the ICP (can be empty)
    
    Returns:
        ValidationResult with pass/fail and reason
    """
    if not lead_role:
        return ValidationResult(
            passed=False,
            reason="Missing lead role field"
        )
    
    # If no target roles specified in ICP, any role is acceptable
    if not icp_target_roles or len(icp_target_roles) == 0:
        return ValidationResult(passed=True)
    
    # Check if lead role matches ANY of the target roles
    lead_role_lower = lead_role.lower().strip()
    best_score = 0
    best_match = ""
    
    for target_role in icp_target_roles:
        if not target_role:
            continue
        target_role_lower = target_role.lower().strip()
        score = fuzz.partial_ratio(lead_role_lower, target_role_lower)
        if score > best_score:
            best_score = score
            best_match = target_role
    
    if best_score >= ROLE_MATCH_THRESHOLD:
        return ValidationResult(passed=True)
    
    return ValidationResult(
        passed=False,
        reason=f"Role mismatch: '{lead_role}' vs {icp_target_roles} (best: {best_score:.0f}%, threshold: {ROLE_MATCH_THRESHOLD}%)"
    )


def check_country_match(lead_country: str, icp_country: str) -> ValidationResult:
    """
    Verify lead's country matches ICP requirement.
    
    Uses case-insensitive exact match. Both must be non-empty.
    If the ICP doesn't specify a country, any country is accepted.
    """
    if not icp_country or not icp_country.strip():
        return ValidationResult(passed=True)
    
    if not lead_country or not lead_country.strip():
        return ValidationResult(
            passed=False,
            reason=f"Missing country (ICP requires '{icp_country}')"
        )
    
    if lead_country.strip().lower() != icp_country.strip().lower():
        return ValidationResult(
            passed=False,
            reason=f"Country mismatch: '{lead_country}' vs ICP '{icp_country}'"
        )
    return ValidationResult(passed=True)


# Seniority hierarchy — index 0 is highest
_SENIORITY_LEVELS = ["C-Suite", "VP", "Director", "Manager", "Individual Contributor"]

def check_seniority_match(lead_seniority: str, icp_seniority: str) -> ValidationResult:
    """
    Verify lead's seniority is within 1 level of ICP target.
    
    Hierarchy: C-Suite > VP > Director > Manager > Individual Contributor
    
    Tolerance of 1 level below the target:
      ICP=C-Suite → accept C-Suite, VP
      ICP=VP      → accept C-Suite, VP, Director
      ICP=Director→ accept C-Suite, VP, Director, Manager
      ICP=Manager → accept C-Suite through Manager
      ICP=IC      → accept all
    
    Leads ABOVE the target always pass (a CEO is fine for a VP ICP).
    """
    if not icp_seniority or not icp_seniority.strip():
        return ValidationResult(passed=True)
    
    if not lead_seniority or not lead_seniority.strip():
        return ValidationResult(passed=True)
    
    lead_s = lead_seniority.strip()
    icp_s = icp_seniority.strip()
    
    try:
        lead_idx = _SENIORITY_LEVELS.index(lead_s)
    except ValueError:
        return ValidationResult(passed=True)
    
    try:
        icp_idx = _SENIORITY_LEVELS.index(icp_s)
    except ValueError:
        return ValidationResult(passed=True)
    
    # lead_idx <= icp_idx means lead is at or above target (always OK)
    # lead_idx == icp_idx + 1 means 1 level below (tolerated)
    # lead_idx > icp_idx + 1 means too junior
    if lead_idx > icp_idx + 1:
        return ValidationResult(
            passed=False,
            reason=f"Seniority too low: '{lead_s}' for ICP target '{icp_s}' (min: {_SENIORITY_LEVELS[min(icp_idx + 1, len(_SENIORITY_LEVELS) - 1)]})"
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
# Email Validation
# =============================================================================

async def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Check 6: Validate email is not catchall/invalid.
    
    Performs multiple checks:
    1. Email format validation (basic regex)
    2. Disposable domain detection
    3. Professional domain check (optional)
    
    Args:
        email: Email address to validate
    
    Returns:
        Tuple of (is_valid: bool, failure_reason: Optional[str])
    """
    if not email:
        return False, "Missing email field"
    
    email = email.strip().lower()
    
    # Check 1: Basic format validation
    if not EMAIL_REGEX.match(email):
        return False, "Invalid email format"
    
    # Check 2: Extract domain
    try:
        domain = email.split('@')[1]
    except IndexError:
        return False, "Invalid email format (no @ symbol)"
    
    # Check 3: Disposable domain detection
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        return False, f"Disposable email domain: {domain}"
    
    # Check 4: Check for numeric-only local part (often fake)
    local_part = email.split('@')[0]
    if local_part.isdigit():
        return False, "Numeric-only email local part"
    
    # Check 5: Check for very short local parts (often fake)
    if len(local_part) < 2:
        return False, "Email local part too short"
    
    # Check 6: Check for suspicious patterns in local part
    suspicious_local_patterns = ['noreply', 'no-reply', 'donotreply', 'nobody', 'admin', 'info@']
    if any(pattern in local_part.lower() for pattern in suspicious_local_patterns):
        return False, f"Generic/system email address"
    
    # TODO: Add TrueList API check for catchall detection in production
    # This would be an async call to TrueList API to verify the email
    # is not a catchall or risky address.
    
    return True, None


def validate_email_sync(email: str) -> Tuple[bool, Optional[str]]:
    """
    Synchronous version of email validation (for use in non-async contexts).
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, can't use run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, validate_email(email))
                return future.result()
        else:
            return loop.run_until_complete(validate_email(email))
    except RuntimeError:
        return asyncio.run(validate_email(email))


# =============================================================================
# Data Quality Checks
# =============================================================================

def check_data_quality(lead: LeadOutput) -> Tuple[bool, Optional[str]]:
    """
    Check 7: Verify lead data quality - no placeholder text, suspicious chars, etc.
    
    Checks for:
    1. Placeholder text patterns (test, asdf, lorem ipsum, etc.)
    2. Suspicious characters (<>{}|\\^~`)
    3. ALL CAPS or all lowercase names (likely bad data)
    4. Excessive special characters
    5. Numeric-only fields
    
    Args:
        lead: The lead to validate
    
    Returns:
        Tuple of (is_valid: bool, failure_reason: Optional[str])
    """
    
    # Fields to check for quality
    # NOTE: full_name is NOT allowed in schema anymore - models cannot submit PII
    fields_to_check = [
        ("business", lead.business),
        ("role", lead.role),
    ]
    
    # Check 1: Placeholder text patterns
    for field_name, field_value in fields_to_check:
        if field_value:
            field_lower = field_value.lower().strip()
            for pattern in PLACEHOLDER_PATTERNS:
                # Check if the pattern is a whole word or significant part
                if pattern == field_lower or f" {pattern} " in f" {field_lower} ":
                    return False, f"Placeholder text detected in {field_name}: '{field_value}'"
    
    # Check 2: Suspicious characters
    for field_name, field_value in fields_to_check:
        if field_value and SUSPICIOUS_CHAR_PATTERN.search(field_value):
            return False, f"Suspicious characters in {field_name}: '{field_value}'"
    
    # NOTE: Name case check REMOVED - full_name field no longer exists in schema
    
    # Check 3: Numeric-only fields (except where expected)
    for field_name, field_value in [("business", lead.business)]:
        if field_value and field_value.strip().isdigit():
            return False, f"Numeric-only {field_name}: '{field_value}'"
    
    # Check 4: Excessive repetition (e.g., "aaaaaaa" or "1111111")
    for field_name, field_value in fields_to_check:
        if field_value and len(field_value) >= 5:
            # Check if any character is repeated more than 70% of the string
            for char in set(field_value.lower()):
                if field_value.lower().count(char) / len(field_value) > 0.7:
                    return False, f"Excessive character repetition in {field_name}: '{field_value}'"
    
    # Check 6: Empty or whitespace-only after strip
    for field_name, field_value in fields_to_check:
        if field_value is not None and not field_value.strip():
            return False, f"Empty {field_name} field"
    
    # Check 7: Business/company name sanity
    if lead.business:
        business = lead.business.strip()
        # Check for obviously fake company names
        fake_company_patterns = [
            r'^company\s*\d*$',  # "Company" or "Company123"
            r'^corp(oration)?\s*\d*$',  # "Corp" or "Corporation123"
            r'^inc(orporated)?\s*\d*$',  # "Inc" or "Incorporated"
            r'^llc\s*\d*$',  # Just "LLC"
            r'^business\s*\d*$',  # Just "Business"
        ]
        for pattern in fake_company_patterns:
            if re.match(pattern, business.lower()):
                return False, f"Generic company name: '{lead.business}'"
    
    # Check 8: Role sanity
    if lead.role:
        role = lead.role.strip()
        # Check for obviously fake roles
        fake_role_patterns = [
            r'^employee\s*$',  # Just "Employee"
            r'^worker\s*$',  # Just "Worker"
            r'^staff\s*$',  # Just "Staff"
            r'^person\s*$',  # Just "Person"
        ]
        for pattern in fake_role_patterns:
            if re.match(pattern, role.lower()):
                return False, f"Generic role: '{lead.role}'"
    
    return True, None


# =============================================================================
# Batch Validation
# =============================================================================

async def validate_lead_batch(
    leads: List[LeadOutput],
    icp: ICPPrompt,
    costs: List[float],
    times: List[float]
) -> List[Tuple[bool, Optional[str]]]:
    """
    Validate a batch of leads against the same ICP.
    
    Tracks seen companies across the batch to detect duplicates.
    
    Args:
        leads: List of leads to validate
        icp: The ICP prompt used
        costs: List of costs per lead
        times: List of times per lead
    
    Returns:
        List of (passed, reason) tuples, one per lead
    """
    results = []
    seen_companies: Set[str] = set()
    
    for i, lead in enumerate(leads):
        cost = costs[i] if i < len(costs) else 0.0
        time = times[i] if i < len(times) else 0.0
        
        passed, reason = await run_automatic_zero_checks(
            lead, icp, cost, time, seen_companies
        )
        
        results.append((passed, reason))
        
        # Add company to seen set if passed (first one wins)
        if passed and lead.business:
            seen_companies.add(lead.business.lower().strip())
    
    return results


# =============================================================================
# Summary Statistics
# =============================================================================

def get_check_names() -> List[str]:
    """Get names of all automatic zero checks."""
    return [
        "hard_time_limit",  # 30s safety net (instant fail)
        "industry_match",
        "sub_industry_match",
        "role_match",
        "email_validation",
        "data_quality",
        "duplicate_company",
    ]


def summarize_validation_results(
    results: List[Tuple[bool, Optional[str]]]
) -> dict:
    """
    Summarize validation results for a batch.
    
    Args:
        results: List of (passed, reason) tuples
    
    Returns:
        Dict with summary statistics
    """
    total = len(results)
    passed = sum(1 for p, _ in results if p)
    failed = total - passed
    
    # Group failures by reason prefix
    failure_reasons = {}
    for p, reason in results:
        if not p and reason:
            # Extract check type from reason
            check_type = reason.split(":")[0] if ":" in reason else reason[:50]
            failure_reasons[check_type] = failure_reasons.get(check_type, 0) + 1
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "failure_breakdown": failure_reasons,
    }
