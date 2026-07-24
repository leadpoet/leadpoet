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
2. Industry sanity check — exact/fuzzy/broad-bucket matches pass;
   plausible mismatches are deferred to the LLM ICP-fit scorer
3. Sub-industry sanity check — exact/fuzzy matches pass; plausible
   mismatches are deferred to the LLM ICP-fit scorer
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

import os
import re
import logging
from typing import Any, Tuple, Optional, Set, NamedTuple, List, Dict

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
        "machine learning",
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
        "computer hardware", "computer hardware manufacturing",
        "semiconductor manufacturing", "semiconductors",
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
        "property management software", "proptech",
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
        "primary and secondary education", "e-learning providers",
        "e learning providers", "e-learning", "e learning", "edtech",
        "educational technology",
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

def _taxonomy_industry_gate_mode() -> str:
    """Resolve the taxonomy industry gate mode (disabled | shadow | enforce)."""
    value = str(
        os.environ.get("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE") or "shadow"
    ).strip().lower()
    if value not in {"disabled", "shadow", "enforce"}:
        logger.warning(
            "taxonomy_industry_gate_invalid_mode value=%r falling back to shadow",
            value,
        )
        return "shadow"
    return value


def _resolved_semantic_mode() -> str:
    """Resolve VERIFIER_SEMANTIC_GATES_MODE, defaulting to disabled on ANY error."""
    try:
        from leadpoet_verifier.semantic_gates import semantic_gate_mode

        return semantic_gate_mode()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "semantic_gate_mode_unresolvable error=%s", str(exc)[:200]
        )
        return "disabled"


async def _semantic_industry_rescue(
    company: Any, icp: Any, detail: Dict[str, Any], semantic_mode: str
) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """Optional source-grounded semantic judge for AMBIGUOUS industry labels.

    Site-faithful composition: the semantic judge may only rescue labels the
    canonical taxonomy is SILENT about — a canonical taxonomy conflict is
    final and never consulted.  Enabled via VERIFIER_SEMANTIC_GATES_MODE
    (disabled by default; a real OpenRouter + fetch pipeline when on).

    Returns ``(verdict, semantic_mode, receipt)``:
      * verdict — True (semantic match), False (semantic no-match), or None
        when the judge is disabled, ineligible (canonical conflict), or
        UNAVAILABLE.  Unavailability is never a verdict.
      * semantic_mode — the resolved VERIFIER_SEMANTIC_GATES_MODE.
      * receipt — durable audit document (mode, model, input hash, source
        hashes, judgment, or the unavailability error class); None when the
        judge was disabled or ineligible.
    """
    from leadpoet_verifier.semantic_gates import SemanticGateEvaluator

    if semantic_mode == "disabled":
        return None, None
    taxonomy = detail.get("leadpoet_taxonomy") or {}
    if taxonomy.get("decision") == "rejected":
        # Canonical conflict: authoritative, never semantically rescued.
        return None, None
    try:
        # DeepLine evidence repair is part of the ported gate: enable_repair
        # is itself env-gated (VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED +
        # DEEPLINE_API_KEY), so from_env returns a repair-less evaluator when
        # the operator has not opted in.
        evaluator = SemanticGateEvaluator.from_env(enable_repair=True)
        result = await evaluator.evaluate_industry(
            company_name=str(getattr(company, "company_name", "") or ""),
            company_website=str(getattr(company, "company_website", "") or ""),
            requested_industry=str(getattr(icp, "industry", "") or ""),
            candidate_industry=str(getattr(company, "industry", "") or ""),
            candidate_subindustry=str(getattr(company, "sub_industry", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 — judge unavailability is not a verdict
        logger.warning(
            "semantic_industry_gate_unavailable error=%s", str(exc)[:200]
        )
        return None, {
            "status": "unavailable",
            "error_class": type(exc).__name__,
        }
    receipt: Optional[Dict[str, Any]] = None
    receipt_fn = getattr(result, "receipt", None)
    if callable(receipt_fn):
        try:
            receipt = receipt_fn()
        except Exception:  # noqa: BLE001 — receipts must never break the gate
            receipt = None
    outcome = getattr(result, "outcome", None) or (
        result.get("outcome") if isinstance(result, dict) else None
    )
    return outcome == "passed", receipt


async def _taxonomy_industry_gate(
    company: Any, icp: Any
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Canonical taxonomy/concept industry fit, ported from the site verifier.

    Deterministic core is pure compute (no network, no LLM).  Decision table
    (taxonomy mode x semantic mode), fixed after the PR-28 audit:

      * taxonomy ``shadow``  — outcome is ALWAYS pass; deterministic and (if
        consulted) semantic verdicts are logged + receipted only.
      * taxonomy ``enforce``:
          - canonical taxonomy conflict            -> zero (judge not consulted)
          - ambiguous + semantic ``disabled``      -> zero
          - ambiguous + semantic ``shadow``        -> zero (verdict receipted
            only — semantic shadow NEVER changes the outcome)
          - ambiguous + semantic ``enforce`` match -> pass (rescued)
          - ambiguous + semantic ``enforce`` no-match -> zero
          - ambiguous + semantic ``enforce`` UNAVAILABLE -> pass (fail-open:
            judge unavailability is never a verdict)

    Returns ``(passed, failure_reason, receipt)``; the receipt is a durable
    audit document (gate modes, deterministic detail, semantic receipt, and
    the final scoring effect).  Any internal failure logs a WARNING and fails
    open — this gate must never take down scoring availability.
    """
    mode = _taxonomy_industry_gate_mode()
    if mode == "disabled":
        return True, None, None
    try:
        from leadpoet_verifier.industry_fit import industry_fit

        passed, detail = industry_fit(
            getattr(icp, "industry", None),
            getattr(company, "industry", None),
            getattr(company, "sub_industry", None),
            candidate_description=getattr(company, "description", None),
        )
    except Exception as exc:  # noqa: BLE001 — availability over strictness
        logger.warning(
            "taxonomy_industry_gate_error mode=%s error=%s", mode, str(exc)[:200]
        )
        return True, None, None
    deterministic_receipt = {
        "passed": passed,
        "match_strategy": detail.get("match_strategy"),
        "candidate": detail.get("candidate"),
        "requested": detail.get("requested"),
        "matched_concepts": detail.get("matched_concepts"),
        "leadpoet_taxonomy": detail.get("leadpoet_taxonomy"),
    }

    def _receipt(final_effect: str, semantic_mode: str = "disabled",
                 semantic_receipt: Optional[Dict[str, Any]] = None,
                 semantic_verdict: Optional[bool] = None) -> Dict[str, Any]:
        return {
            "gate": "taxonomy_industry",
            "taxonomy_mode": mode,
            "semantic_mode": semantic_mode,
            "deterministic": deterministic_receipt,
            "semantic_verdict": semantic_verdict,
            "semantic": semantic_receipt,
            "final_effect": final_effect,
        }

    if passed:
        return True, None, _receipt("passed_deterministic")
    semantic_verdict: Optional[bool] = None
    semantic_mode = _resolved_semantic_mode()
    semantic_receipt: Optional[Dict[str, Any]] = None
    try:
        semantic_verdict, semantic_receipt = await _semantic_industry_rescue(
            company, icp, detail, semantic_mode
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "semantic_industry_gate_error error=%s", str(exc)[:200]
        )
        if semantic_mode == "enforce":
            # A configured judge that cannot even start is UNAVAILABLE, not a
            # verdict — receipt it so the fail-open branch below applies.
            semantic_receipt = {
                "status": "unavailable",
                "error_class": type(exc).__name__,
            }
    if mode == "shadow":
        logger.warning(
            "taxonomy_industry_gate_shadow_mismatch company=%r icp_industry=%r "
            "strategy=%s semantic_verdict=%s detail=%s",
            getattr(company, "company_name", None),
            getattr(icp, "industry", None),
            detail.get("match_strategy"),
            semantic_verdict,
            {k: detail.get(k) for k in ("candidate", "requested", "matched_concepts")},
        )
        return True, None, _receipt(
            "shadow_pass", semantic_mode, semantic_receipt, semantic_verdict
        )
    # taxonomy enforce: only a semantic-ENFORCE judge may change the outcome.
    if semantic_mode == "enforce":
        if semantic_verdict is True:
            logger.info(
                "taxonomy_industry_gate_semantic_rescue company=%r icp_industry=%r",
                getattr(company, "company_name", None),
                getattr(icp, "industry", None),
            )
            return True, None, _receipt(
                "rescued_semantic_enforce", semantic_mode, semantic_receipt, True
            )
        if semantic_verdict is None and semantic_receipt is not None:
            # The judge was consulted but UNAVAILABLE (provider outage or
            # internal error): unavailability is never a verdict, so the
            # documented fail-open contract passes the company.
            logger.warning(
                "taxonomy_industry_gate_semantic_unavailable_failopen company=%r",
                getattr(company, "company_name", None),
            )
            return True, None, _receipt(
                "failed_open_semantic_unavailable",
                semantic_mode,
                semantic_receipt,
                None,
            )
    receipt = _receipt("zeroed", semantic_mode, semantic_receipt, semantic_verdict)
    return False, (
        "Industry outside canonical taxonomy fit: "
        f"'{getattr(company, 'industry', '')}' does not match ICP industry "
        f"'{getattr(icp, 'industry', '')}'"
    ), receipt


async def run_company_zero_checks(
    company: CompanyOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
    gate_receipts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """Deterministic pre-checks that run BEFORE any LLM scoring in
    the company-mode model competition.

    Checks (in order):

      * Hard time limit (30s safety net)
      * Industry sanity check.  Exact/fuzzy/broad-bucket matches pass,
        but semantic mismatches are not hard-zeroed here because the
        downstream ICP-fit LLM scores industry/sub-industry nuance.
      * Sub-industry sanity check.  Exact/fuzzy matches pass, but
        mismatches are deferred to the downstream ICP-fit scorer.
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

    # Check 2: Industry sanity check.  Do not hard-zero merely because
    # LinkedIn and ICP labels use adjacent names; the ICP-fit scorer
    # has the context needed to score this properly.
    result = check_industry_match(company.industry, icp.industry)
    if not result.passed:
        logger.info(f"Company failed industry check: {result.reason}")
        return False, result.reason

    # Check 3: Sub-industry sanity check — relaxed.  CompanyOutput marks
    # sub_industry as optional; if either side is empty we skip the check
    # rather than failing (industry fit is the primary filter).
    if (icp.sub_industry and icp.sub_industry.strip()
            and company.sub_industry and company.sub_industry.strip()):
        result = check_sub_industry_match(company.sub_industry, icp.sub_industry)
        if not result.passed:
            logger.info(f"Company failed sub-industry check: {result.reason}")
            return False, result.reason

    # Check 3b: Canonical taxonomy industry fit (ported from the site
    # verifier).  The fuzzy checks above are presence-only sanity gates; this
    # evaluates the authoritative Leadpoet taxonomy + bounded canonical
    # concepts.  Mode via RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE:
    #   disabled — skip entirely;
    #   shadow (default) — compute + log mismatches only, outcome unchanged;
    #   enforce — hard-zero on a canonical mismatch (operator opt-in only,
    #   because it changes benchmark scores).
    taxonomy_passed, taxonomy_reason, taxonomy_receipt = (
        await _taxonomy_industry_gate(company, icp)
    )
    if gate_receipts is not None and taxonomy_receipt is not None:
        gate_receipts.append(taxonomy_receipt)
    if not taxonomy_passed:
        logger.info(f"Company failed taxonomy industry gate: {taxonomy_reason}")
        return False, taxonomy_reason

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

    Country is enforced by its own exact-match check elsewhere in
    run_company_zero_checks.  Industry / sub-industry are sanity checks
    and detailed fit is scored by the ICP-fit LLM, so
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
    Check 3: Verify lead's industry is present and not obviously impossible.

    Exact/fuzzy/broad-bucket matches pass immediately.  If labels are
    adjacent but not an exact deterministic match, defer to the ICP-fit
    scorer instead of hard-zeroing the company.  This mirrors the private
    model contract: deterministic industry logic may accept obvious
    matches, but nuanced company/ICP fit belongs in the LLM scorer.
    
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
        logger.info(
            "Company industry label deferred to ICP-fit scorer: "
            "'%s' vs '%s' (score: %.0f%%, threshold: %s%%)",
            lead_industry,
            icp_industry,
            score,
            INDUSTRY_MATCH_THRESHOLD,
        )
    return ValidationResult(passed=True)


def check_sub_industry_match(lead_sub_industry: str, icp_sub_industry: str) -> ValidationResult:
    """
    Check 4: Verify sub-industry is present and defer nuanced mismatch
    handling to the ICP-fit scorer.
    
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
        logger.info(
            "Company sub-industry label deferred to ICP-fit scorer: "
            "'%s' vs '%s' (score: %.0f%%, threshold: %s%%)",
            lead_sub_industry,
            icp_sub_industry,
            score,
            SUB_INDUSTRY_MATCH_THRESHOLD,
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
