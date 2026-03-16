"""
Qualification System: Lead Scoring

Phase 5.2 from tasks10.md

This module implements the lead scoring system for the Lead Qualification
Agent competition. It combines:

1. Automatic-zero pre-checks (deterministic validation)
2. LLM-based scoring for three components:
   - ICP Fit (0-20 points)
   - Decision Maker (0-30 points)
   - Intent Signal (0-50 points, with time decay)
3. Penalties for cost and time
4. Final score calculation

Scoring Flow:
1. Run pre-checks → If fail, score = 0
2. Mark company as seen (first lead per company wins)
3. Score ICP fit via LLM
4. Score decision maker via LLM
5. Verify intent signal and score relevance via LLM
6. Apply time decay to intent signal
7. Calculate penalties
8. Compute final score (floor at 0)

Max Score per Lead: 100 points (20 + 30 + 50)

CRITICAL: This is NEW scoring for qualification models only.
Do NOT modify any existing scoring or reputation calculation in the
sourcing workflow.
"""

import re
import logging
from datetime import date, datetime
from typing import Set, Optional, Tuple, List
from collections import Counter
from urllib.parse import urlparse

from gateway.qualification.config import CONFIG
from gateway.qualification.models import LeadOutput, ICPPrompt, LeadScoreBreakdown
from qualification.scoring.pre_checks import run_automatic_zero_checks
from qualification.scoring.db_verification import verify_leads_batch
from qualification.scoring.intent_verification import (
    verify_intent_signal,
    openrouter_chat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Score component maximums
MAX_ICP_FIT_SCORE = 20
MAX_DECISION_MAKER_SCORE = 30
MAX_INTENT_SIGNAL_SCORE = 50
MAX_TOTAL_SCORE = MAX_ICP_FIT_SCORE + MAX_DECISION_MAKER_SCORE + MAX_INTENT_SIGNAL_SCORE

# LLM temperature for scoring (slightly higher for nuanced scoring)
SCORING_TEMPERATURE = 0.4


# =============================================================================
# Main Scoring Function
# =============================================================================

async def score_lead(
    lead: LeadOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
    force_fail_reason: Optional[str] = None
) -> LeadScoreBreakdown:
    """
    Score a lead against an ICP.
    
    This is the main entry point for lead scoring. The flow is:
    
    1. Run automatic-zero pre-checks (BEFORE any LLM calls)
    2. If pre-checks fail → return score 0 immediately with failure_reason
    3. If pre-checks pass → mark company as seen, run LLM scoring
    
    Scoring Components:
    - ICP Fit: 0-20 pts (how well lead matches ICP criteria)
    - Decision Maker: 0-30 pts (is this person a buyer/decision maker)
    - Intent Signal: 0-50 pts (quality and relevance of intent signal)
    
    Time Decay:
    - ≤2 months old: 100% (multiplier 1.0)
    - ≤12 months old: 50% (multiplier 0.5)
    - >12 months old: 25% (multiplier 0.25)
    
    Variability Penalties (NEW - replaces old linear penalties):
    - NO penalty if cost ≤ average ($0.05) and time ≤ average (8s)
    - 5-point penalty if cost > 2× average ($0.10)
    - 5-point penalty if time > 2× average (16s)
    - Thresholds are DYNAMIC based on CONFIG settings
    
    Args:
        lead: The lead to score
        icp: The ICP prompt used for evaluation
        run_cost_usd: Total API cost for this lead
        run_time_seconds: Total processing time for this lead
        seen_companies: Set of companies already scored (for duplicate detection)
        force_fail_reason: If set, skip pre-checks and return 0 with this reason
    
    Returns:
        LeadScoreBreakdown with all scoring components
    """
    logger.info(f"Scoring lead: {lead.business} / {lead.role} for ICP {icp.icp_id}")
    
    # Handle forced failure (from validator when pre-check already failed)
    if force_fail_reason:
        logger.info(f"Lead forced to fail: {force_fail_reason}")
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=force_fail_reason
        )
    
    # =========================================================================
    # STEP 1: Automatic-zero pre-checks (deterministic, no LLM)
    # =========================================================================
    passes, failure_reason = await run_automatic_zero_checks(
        lead, icp, run_cost_usd, run_time_seconds, seen_companies
    )
    
    if not passes:
        logger.info(f"Lead failed pre-checks: {failure_reason}")
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=failure_reason
        )
    
    # =========================================================================
    # STEP 2: Mark company as seen (first lead per company wins)
    # =========================================================================
    if lead.business:
        seen_companies.add(lead.business.lower().strip())
    
    # =========================================================================
    # STEP 3: LLM-based scoring (only if pre-checks passed)
    # =========================================================================
    try:
        # Score ICP Fit (0-20 pts)
        icp_fit = await score_icp_fit(lead, icp)
        logger.debug(f"ICP fit score: {icp_fit}")
        
        # Score Decision Maker (0-30 pts)
        decision_maker = await score_decision_maker(lead, icp)
        logger.debug(f"Decision maker score: {decision_maker}")
        
        # Score Intent Signals (0-50 pts) — scores ALL signals, averages after per-signal time decay
        intent_raw, intent_final, decay_multiplier, max_confidence, all_fabricated = await score_intent_signal(lead, icp)
        logger.debug(f"Intent signal avg_raw={intent_raw:.1f}, avg_final={intent_final:.1f}, decay={decay_multiplier:.2f}")
        
        # =====================================================================
        # CRITICAL: Zero ENTIRE lead score if ALL intent signals are fabricated
        # =====================================================================
        if all_fabricated:
            logger.warning(f"❌ ALL INTENT SIGNALS FABRICATED - zeroing entire lead score")
            return LeadScoreBreakdown(
                icp_fit=0,
                decision_maker=0,
                intent_signal_raw=0,
                time_decay_multiplier=1.0,
                intent_signal_final=0,
                cost_penalty=0,
                time_penalty=0,
                final_score=0,
                failure_reason="Intent fabrication detected (hardcoded date or generic claim)"
            )
        
    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=f"LLM scoring error: {str(e)[:100]}"
        )
    
    # =========================================================================
    # STEP 4: Calculate variability penalties
    # =========================================================================
    # NEW SYSTEM: No penalty if within budget, 5-point penalty for high variability
    #
    # - NO penalty if cost ≤ MAX_COST_PER_LEAD_USD (e.g., $0.05)
    # - NO penalty if time ≤ MAX_TIME_PER_LEAD_SECONDS (e.g., 8s)
    # - 5-point penalty if cost > 2× MAX_COST_PER_LEAD_USD (e.g., $0.10)
    # - 5-point penalty if time > 2× MAX_TIME_PER_LEAD_SECONDS (e.g., 16s)
    #
    # This allows models with high variability (some leads expensive/slow)
    # to still succeed as long as the TOTAL stays within budget.
    
    cost_penalty = 0.0
    time_penalty = 0.0
    
    # Cost variability penalty
    cost_penalty_threshold = CONFIG.get_cost_penalty_threshold()
    if run_cost_usd > cost_penalty_threshold:
        cost_penalty = float(CONFIG.VARIABILITY_PENALTY_POINTS)
        logger.debug(
            f"Cost variability penalty applied: ${run_cost_usd:.4f} > "
            f"${cost_penalty_threshold:.4f} (2× ${CONFIG.MAX_COST_PER_LEAD_USD:.4f})"
        )
    
    # Time variability penalty - DISABLED
    # Models should optimize for quality first, speed later.
    # The 30s hard stop (RUNNING_MODEL_TIMEOUT_SECONDS) is the only time constraint.
    
    logger.debug(f"Variability penalties - cost: {cost_penalty:.0f} pts, time: {time_penalty:.0f} pts")
    
    # =========================================================================
    # STEP 5: Calculate final score (floor at 0)
    # =========================================================================
    total_raw = icp_fit + decision_maker + intent_final
    final_score = max(0.0, total_raw - cost_penalty - time_penalty)
    
    total_penalty = cost_penalty + time_penalty
    if total_penalty > 0:
        logger.info(
            f"Lead scored: {final_score:.2f} (ICP: {icp_fit}, DM: {decision_maker}, "
            f"Intent: {intent_final:.2f}, Variability penalty: -{total_penalty:.0f} pts)"
        )
    else:
        logger.info(
            f"Lead scored: {final_score:.2f} (ICP: {icp_fit}, DM: {decision_maker}, "
            f"Intent: {intent_final:.2f}, No variability penalty)"
        )
    
    return LeadScoreBreakdown(
        icp_fit=icp_fit,
        decision_maker=decision_maker,
        intent_signal_raw=intent_raw,
        time_decay_multiplier=decay_multiplier,
        intent_signal_final=intent_final,
        cost_penalty=cost_penalty,
        time_penalty=time_penalty,
        final_score=final_score,
        failure_reason=None
    )


# =============================================================================
# ICP Fit Scoring
# =============================================================================

async def score_icp_fit(lead: LeadOutput, icp: ICPPrompt) -> float:
    """
    Score how well the lead matches the ICP criteria.
    
    Evaluates:
    - Industry/sub-industry match
    - Role/seniority match
    - Company size match
    - Geographic match
    
    Args:
        lead: The lead to score
        icp: The ICP prompt
    
    Returns:
        Score from 0-20
    """
    icp_product = icp.product_service or ""
    icp_prompt_text = icp.prompt or ""

    prompt = f"""Score how well this lead matches the Ideal Customer Profile (ICP) on a scale of 0-20.

ICP CRITERIA:
- Industry: {icp.industry}
- Sub-industry: {icp.sub_industry}
- Product/Service buyer is selling: {icp_product}
- Target roles: {', '.join(icp.target_roles) if icp.target_roles else 'Any'}
- Target seniority: {icp.target_seniority}
- Employee count: {icp.employee_count}
- Geography: {icp.geography}
- Full buyer request: "{icp_prompt_text}"

LEAD DATA:
- Industry: {lead.industry}
- Sub-industry: {lead.sub_industry}
- Role: {lead.role}
- Seniority: {lead.seniority.value if hasattr(lead.seniority, 'value') else lead.seniority}
- Employee count: {lead.employee_count}
- Company: {lead.business}
- Location: {lead.city}, {lead.state}, {lead.country}

SCORING GUIDELINES (check EACH criterion):

1. INDUSTRY FIT (0-8 points):
   - Exact industry + sub-industry match: 7-8
   - Same industry, different sub-industry: 4-6
   - Related industry: 2-3
   - Unrelated industry: 0-1

2. PRODUCT/SERVICE RELEVANCE (0-6 points):
   - Company would clearly be a buyer of "{icp_product}": 5-6
   - Company might use this type of product: 3-4
   - Weak connection to the product: 1-2
   - No connection: 0

3. STRUCTURAL FIT (0-6 points):
   - Role matches target roles, right seniority, right geography, right size: 5-6
   - Most structural criteria match: 3-4
   - Some match: 1-2
   - Poor structural fit: 0

Add the three sub-scores together (max 20).

Respond with ONLY a single number (0-20):"""

    response = await openrouter_chat(prompt, model="gpt-4o-mini")
    score = extract_score(response, max_score=MAX_ICP_FIT_SCORE)
    return score


# =============================================================================
# Decision Maker Scoring
# =============================================================================

async def score_decision_maker(lead: LeadOutput, icp: ICPPrompt) -> float:
    """
    Score whether this person is a decision-maker for the product/service.
    
    Evaluates:
    - Role authority level
    - Seniority and purchasing power
    - Relevance to the product/service being sold
    
    Args:
        lead: The lead to score
        icp: The ICP prompt (contains product_service info)
    
    Returns:
        Score from 0-30
    """
    target_roles_str = ', '.join(icp.target_roles) if icp.target_roles else 'Any senior role'

    prompt = f"""Score whether this lead's role matches what the buyer is looking for AND has decision-making authority, on a scale of 0-30.

BUYER IS LOOKING FOR THESE SPECIFIC ROLES: {target_roles_str}
BUYER IS SELLING: "{icp.product_service}"

LEAD:
- Role: {lead.role}
- Seniority: {lead.seniority.value if hasattr(lead.seniority, 'value') else lead.seniority}
- Company: {lead.business}
- Industry: {lead.industry}

SCORING (two dimensions — BOTH matter):

1. ROLE MATCH (0-15 points):
   - Lead's role is one of the buyer's target roles or very close equivalent: 13-15
     (e.g., buyer wants "CTO" and lead IS the CTO)
   - Lead's role is in the same department/function as target roles: 8-12
     (e.g., buyer wants "VP of Engineering" and lead is "Director of Engineering")
   - Lead's role is senior but in a different function: 3-7
     (e.g., buyer wants "Supply Chain Director" and lead is "CEO" — senior but different function)
   - Lead's role is unrelated to any target role: 0-2
     (e.g., buyer wants "CTO" and lead is "HR Director")

2. DECISION-MAKING AUTHORITY (0-15 points):
   - C-suite or VP with clear budget authority for this product type: 13-15
   - Director-level with likely purchasing influence for this product: 8-12
   - Manager-level, may influence but not decide: 4-7
   - Individual contributor or unrelated authority: 0-3

Add both sub-scores together (max 30).

CRITICAL: A CEO always scores high on authority (13-15) but should score LOW on role match (3-7) if the buyer specifically asked for a technical role like "CTO" or "Head of DevOps". The buyer knows who they want to reach — don't override their targeting.

Respond with ONLY a single number (0-30):"""

    response = await openrouter_chat(prompt, model="gpt-4o-mini")
    score = extract_score(response, max_score=MAX_DECISION_MAKER_SCORE)
    return score


# =============================================================================
# Intent Signal Scoring
# =============================================================================

# Source type quality multipliers - high-value sources get full credit
# Low-value or vague sources get penalized
SOURCE_TYPE_MULTIPLIERS = {
    "linkedin": 1.0,           # High-value: professional network
    "job_board": 1.0,          # High-value: explicit hiring intent
    "github": 1.0,             # High-value: technical activity
    "news": 0.9,               # Good: public announcements
    "company_website": 0.85,   # Medium: could be generic content
    "social_media": 0.8,       # Medium: less reliable intent signals
    "review_site": 0.75,       # Medium-low: indirect signal
    "wikipedia": 0.6,          # Low-medium: reliable company info but indirect intent
    "other": 0.3,              # LOW: catch-all category indicates fallback
}


def _apply_signal_time_decay(
    raw_score: float,
    signal_date: Optional[str],
    date_status: str,
    source_str: str,
) -> Tuple[float, float]:
    """
    Apply time decay to a single signal's raw score.
    
    Returns:
        Tuple of (after_decay_score, decay_multiplier)
    """
    NO_DATE_DECAY_MULTIPLIER = 0.5
    source_lower = (source_str or "").lower().strip()

    if date_status == "no_date":
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return raw_score, 1.0
        else:
            return raw_score * NO_DATE_DECAY_MULTIPLIER, NO_DATE_DECAY_MULTIPLIER

    try:
        parsed_date = date.fromisoformat(signal_date) if signal_date else None
    except (ValueError, AttributeError):
        parsed_date = None

    if parsed_date is None:
        return 0.0, 0.0

    age_months = calculate_age_months(parsed_date)
    decay = calculate_time_decay_multiplier(age_months)
    return raw_score * decay, decay


def _extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL (e.g. 'www.bloomberg.com' → 'bloomberg.com').
    
    Handles miner variability: missing schemes, www prefixes, mixed casing.
    URLs are normalized at the Pydantic layer, but this is defensive.
    """
    try:
        clean = url.strip()
        if not clean.lower().startswith(('http://', 'https://')):
            clean = 'https://' + clean
        hostname = urlparse(clean).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname
    except Exception:
        return url.lower().strip()


async def score_intent_signal(lead: LeadOutput, icp: ICPPrompt) -> Tuple[float, float, float, int, bool]:
    """
    Score ALL intent signals submitted by the model.
    
    Each signal is verified, scored, and time-decayed independently.
    The final intent score is the AVERAGE of all after-decay scores.
    This rewards models that submit multiple high-quality signals and
    penalizes padding with fabricated or weak signals (they drag the average down).
    
    URL deduplication: only the first signal from a given domain is scored
    normally — subsequent signals pointing to the same domain score 0
    (they're the same evidence repackaged).
    
    Args:
        lead: The lead to score (has intent_signals: List[IntentSignal])
        icp: The ICP prompt
    
    Returns:
        Tuple of (avg_raw, avg_final, avg_decay_multiplier, max_confidence, all_fabricated)
    """
    icp_criteria = None

    seen_domains: set = set()
    signal_results = []
    for signal in lead.intent_signals:
        domain = _extract_domain(signal.url)
        if domain in seen_domains:
            logger.warning(
                f"  ⚠ Duplicate domain '{domain}' — signal scores 0 (URL dedup)"
            )
            signal_results.append({
                "raw": 0.0,
                "after_decay": 0.0,
                "decay": 0.0,
                "confidence": 0,
                "date_status": "fabricated",
            })
            continue
        seen_domains.add(domain)

        score, confidence, date_status = await _score_single_intent_signal(
            signal, icp, icp_criteria, lead.business, lead.company_website
        )
        source_str = (signal.source.value if hasattr(signal.source, 'value') else str(signal.source))
        after_decay, decay_mult = _apply_signal_time_decay(
            score, signal.date, date_status, source_str
        )
        signal_results.append({
            "raw": score,
            "after_decay": after_decay,
            "decay": decay_mult,
            "confidence": confidence,
            "date_status": date_status,
        })
        logger.info(
            f"  Signal {len(signal_results)}: raw={score:.1f}, decay={decay_mult:.2f}, "
            f"final={after_decay:.1f}, confidence={confidence}, date_status={date_status}"
        )

    if not signal_results:
        return 0.0, 0.0, 1.0, 0, True

    n = len(signal_results)
    avg_raw = sum(r["raw"] for r in signal_results) / n
    avg_final = sum(r["after_decay"] for r in signal_results) / n
    avg_decay = sum(r["decay"] for r in signal_results) / n
    max_confidence = max(r["confidence"] for r in signal_results)
    all_fabricated = all(r["confidence"] == 0 for r in signal_results)

    logger.info(
        f"Scored {n} intent signal(s) — avg_raw={avg_raw:.1f}, avg_final={avg_final:.1f}, "
        f"avg_decay={avg_decay:.2f}, max_confidence={max_confidence}, all_fabricated={all_fabricated}"
    )

    return avg_raw, avg_final, avg_decay, max_confidence, all_fabricated


# Source-dependent date requirements:
# - Some sources (tech stack, company info) don't need dates — they're ongoing signals
# - Other sources (job postings, news, announcements) NEED dates — recency matters
SOURCES_DATE_NOT_REQUIRED = frozenset({
    "github",           # Tech stack is ongoing — no date needed
    "company_website",  # About pages, tech stack pages — ongoing
    "wikipedia",        # Company info is ongoing — no date needed
    "review_site",      # Reviews are ongoing signals
})

SOURCES_DATE_REQUIRED = frozenset({
    "linkedin",         # Posts/updates need dates — recency matters
    "job_board",        # Job postings need dates — could be stale
    "news",             # News articles need dates — recency is everything
    "social_media",     # Social posts need dates — could be old
})

MAX_INTENT_NO_DATE_REQUIRED = 15   # Cap for undated signals where date IS required
MAX_INTENT_NO_DATE_UNKNOWN = 40   # Cap for undated signals from unrecognized source types
MAX_INTENT_NO_DATE_OPTIONAL = 50  # Full score for undated signals where date is NOT required

async def _score_single_intent_signal(
    signal: "IntentSignal",
    icp: ICPPrompt,
    icp_criteria: Optional[str],
    company_name: str,
    company_website: str = ""
) -> Tuple[float, int, str]:
    """
    Verify and score a single intent signal.
    
    Returns:
        Tuple of (score 0-50, verification_confidence 0-100, date_status)
    """
    # Verify the signal is real AND provides evidence of ICP fit
    verified, confidence, reason, date_status = await verify_intent_signal(
        signal,
        icp_industry=icp.industry,
        icp_criteria=icp_criteria,
        company_name=company_name,
        company_website=company_website
    )
    
    if not verified:
        logger.info(f"Intent signal not verified: {reason}")
        return 0.0, confidence, date_status
    
    # Get source as string
    source_str = signal.source.value if hasattr(signal.source, 'value') else str(signal.source)
    source_lower = source_str.lower()
    
    # Get source type multiplier (penalize low-value sources like "other")
    source_multiplier = SOURCE_TYPE_MULTIPLIERS.get(source_lower, 0.5)
    
    # Use both: full prompt for complete buyer context, product_service for what's being sold
    buyer_request = icp.prompt or icp.product_service
    
    icp_intent_list = ", ".join(str(s) for s in (icp.intent_signals or [])) if hasattr(icp, 'intent_signals') and icp.intent_signals else ""
    icp_intent_section = f"\nBUYER'S EXPECTED INTENT SIGNALS: \"{icp_intent_list}\"\n" if icp_intent_list else ""

    prompt = f"""Score how relevant this intent signal is to the buyer's request on a scale of 0-50.

BUYER IS SELLING: "{icp.product_service}"

BUYER'S FULL REQUEST:
"{buyer_request}"
{icp_intent_section}
INTENT SIGNAL FOUND:
- Source: {source_str}
- Description: {signal.description}
- Date: {signal.date}
- Snippet: {signal.snippet}

SCORING GUIDELINES:
- 40-50: Signal directly proves the company matches the buyer's request AND matches the buyer's expected intent signals (e.g., buyer expects "hiring for specific roles" and signal shows actual job postings)
- 30-39: Signal strongly suggests the company fits AND is related to the expected intent type
- 20-29: Signal is somewhat relevant but the intent TYPE doesn't match what the buyer asked for (e.g., buyer expected "hiring signals" but signal shows "product launch")
- 10-19: Signal is tangentially related — generic company activity that doesn't match the specific intent the buyer described
- 0-9: Signal has no meaningful connection to the buyer's request OR is generic marketing copy rephrased to sound like intent

CRITICAL: Score 0-10 if the description is just rephrased website marketing copy. Examples:
- "Company launched advanced [product category]" when the source is just an About page → 0-5
- "Company is committed to innovative solutions" → 0-5
- "Company announced [generic capability]" when there's no actual announcement → 0-5
- Description uses action words (launched, announced, hiring) but the source content is just a static company page → 0-10

IMPORTANT: The description must reflect a SPECIFIC, TIMELY action (a real event that happened) — not a restatement of what the company does in general.

Consider:
1. Does this signal match the buyer's SPECIFIC expected intent signals?
2. Is the described action a REAL EVENT or just rephrased marketing copy from the company website?
3. Would a salesperson use this signal to pitch "{icp.product_service}" to this company TODAY?
4. Is the description specific enough to be verifiable (names, dates, amounts) or vague enough to apply to any company?

Respond with ONLY a single number (0-50):"""

    response = await openrouter_chat(prompt, model="gpt-4o-mini")
    raw_score = extract_score(response, max_score=MAX_INTENT_SIGNAL_SCORE)
    
    # Apply source-dependent date requirements
    if date_status == "no_date":
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            # Tech stack, company info, etc. — date not needed, full score allowed
            logger.info(f"Undated {source_str} signal — date not required for this source type")
        elif source_lower in SOURCES_DATE_REQUIRED:
            # Job postings, news, etc. — date IS required, cap the score
            raw_score = min(raw_score, MAX_INTENT_NO_DATE_REQUIRED)
            logger.info(f"Undated {source_str} signal — date required, capped at {MAX_INTENT_NO_DATE_REQUIRED}")
        else:
            # Unknown source type — moderate cap (more lenient than date-required)
            raw_score = min(raw_score, MAX_INTENT_NO_DATE_UNKNOWN)
            logger.info(f"Undated {source_str} signal (unknown source) — capped at {MAX_INTENT_NO_DATE_UNKNOWN}")
    
    # Weight by verification confidence AND source type quality
    weighted_score = raw_score * (confidence / 100) * source_multiplier
    
    if source_multiplier < 1.0:
        logger.info(f"Applied source type penalty: {source_str} -> {source_multiplier}x")
    
    return weighted_score, confidence, date_status


# =============================================================================
# Time Decay Calculation
# =============================================================================

def calculate_age_months(signal_date: date) -> float:
    """
    Calculate the age of a signal in months.
    
    Args:
        signal_date: The date of the intent signal
    
    Returns:
        Age in months (can be fractional)
    """
    today = date.today()
    days_old = (today - signal_date).days
    return days_old / 30.0  # Approximate months


def calculate_time_decay_multiplier(age_months: float) -> float:
    """
    Calculate the time decay multiplier for an intent signal.
    
    Decay tiers:
    - ≤2 months: 100% (1.0x)
    - ≤12 months: 50% (0.5x)
    - >12 months: 25% (0.25x)
    
    Args:
        age_months: Age of the signal in months
    
    Returns:
        Decay multiplier (1.0, 0.5, or 0.25)
    """
    if age_months <= CONFIG.INTENT_SIGNAL_DECAY_50_PCT_MONTHS:
        return 1.0
    elif age_months <= CONFIG.INTENT_SIGNAL_DECAY_25_PCT_MONTHS:
        return 0.5
    else:
        return 0.25


# =============================================================================
# Helper Functions
# =============================================================================

def extract_score(response: str, max_score: int) -> float:
    """
    Extract numeric score from LLM response.
    
    Handles various response formats:
    - Just a number: "15"
    - With text: "Score: 15"
    - With decimal: "15.5"
    
    Args:
        response: The LLM response text
        max_score: Maximum allowed score
    
    Returns:
        Extracted score (capped at max_score), or 0.0 if not found
    """
    response = response.strip()
    
    # Try to find a number in the response
    # Look for patterns like "15", "15.5", "Score: 15", etc.
    patterns = [
        r'^(\d+(?:\.\d+)?)\s*$',  # Just a number
        r'(?:score|rating)[:=\s]+(\d+(?:\.\d+)?)',  # "Score: 15"
        r'(\d+(?:\.\d+)?)\s*(?:out of|\/)',  # "15 out of" or "15/"
        r'(\d+(?:\.\d+)?)',  # Any number (fallback)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Cap at max score
                return min(score, float(max_score))
            except ValueError:
                continue
    
    logger.warning(f"Could not extract score from response: {response[:100]}")
    return 0.0


# =============================================================================
# Structural Similarity Detection
# =============================================================================

def _normalize_for_similarity(text: str) -> str:
    """Normalize text for similarity comparison - remove company-specific details."""
    if not text:
        return ""
    # Lowercase and remove extra whitespace
    text = " ".join(text.lower().split())
    # Remove common variable parts (company names, dates, numbers)
    text = re.sub(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', '[DATE]', text)  # ISO dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]', text)  # Other dates
    text = re.sub(r'\b\d+\s*(employees?|people|staff|workers)\b', '[EMPLOYEE_COUNT]', text)
    text = re.sub(r'\$\d+[\d,]*\.?\d*\s*(million|m|billion|b|k)?\b', '[MONEY]', text)
    text = re.sub(r'\b\d{3,}\b', '[NUMBER]', text)  # Large numbers
    return text


def detect_structural_similarity(leads: List[LeadOutput], threshold: float = 0.7) -> List[int]:
    """
    Detect leads with structurally similar intent signals.
    
    This catches gaming where models use templated responses with minor variations.
    Gaming typically occurs in intent_signal.description and intent_signal.snippet.
    
    Args:
        leads: List of leads to analyze
        threshold: Similarity ratio threshold (0.7 = 70% similar)
    
    Returns:
        List of indices of leads flagged for structural similarity
    """
    if len(leads) < 3:
        return []  # Need at least 3 leads to detect patterns
    
    flagged_indices = []
    
    # Extract normalized intent descriptions and snippets (from first/primary signal)
    # Gaming typically occurs here - models use templated intent signals
    intent_descs = [
        _normalize_for_similarity(lead.intent_signals[0].description if lead.intent_signals else "")
        for lead in leads
    ]
    intent_snippets = [
        _normalize_for_similarity(lead.intent_signals[0].snippet if lead.intent_signals else "")
        for lead in leads
    ]
    
    # Count similar patterns in intent descriptions
    intent_desc_patterns = Counter()
    for intent in intent_descs:
        if len(intent) > 20:  # Only count substantial descriptions
            # Create a simplified pattern (first 50 chars)
            pattern = intent[:50]
            intent_desc_patterns[pattern] += 1
    
    # Count similar patterns in intent snippets
    intent_snippet_patterns = Counter()
    for snippet in intent_snippets:
        if len(snippet) > 20:
            pattern = snippet[:50]
            intent_snippet_patterns[pattern] += 1
    
    # Flag leads that match repeated patterns
    for i, lead in enumerate(leads):
        intent_desc_normalized = _normalize_for_similarity(
            lead.intent_signals[0].description if lead.intent_signals else ""
        )
        intent_snippet_normalized = _normalize_for_similarity(
            lead.intent_signals[0].snippet if lead.intent_signals else ""
        )
        
        # Check if intent matches a repeated pattern
        intent_desc_pattern = intent_desc_normalized[:50] if len(intent_desc_normalized) > 20 else ""
        intent_snippet_pattern = intent_snippet_normalized[:50] if len(intent_snippet_normalized) > 20 else ""
        
        # If same pattern appears 3+ times, it's likely templated
        intent_desc_repeated = intent_desc_patterns.get(intent_desc_pattern, 0) >= 3
        intent_snippet_repeated = intent_snippet_patterns.get(intent_snippet_pattern, 0) >= 3
        
        if intent_desc_repeated or intent_snippet_repeated:
            flagged_indices.append(i)
            logger.warning(
                f"Lead {i} flagged for structural similarity: "
                f"intent_desc_repeated={intent_desc_repeated}, intent_snippet_repeated={intent_snippet_repeated}"
            )
    
    # If more than 50% of leads are flagged, this is likely gaming
    if len(flagged_indices) >= len(leads) * 0.5:
        logger.error(
            f"❌ STRUCTURAL GAMING DETECTED: {len(flagged_indices)}/{len(leads)} leads "
            f"show templated patterns"
        )
    
    return flagged_indices


# =============================================================================
# Batch Scoring
# =============================================================================

async def score_leads_batch(
    leads: list[LeadOutput],
    icp: ICPPrompt,
    costs: list[float],
    times: list[float],
    apply_similarity_detection: bool = True
) -> list[LeadScoreBreakdown]:
    """
    Score a batch of leads against the same ICP.
    
    Tracks seen companies across the batch to enforce first-wins rule.
    Also detects structural similarity (templated responses) across leads.
    
    IMPORTANT: DB field verification runs FIRST as a batch query.
    Leads with tampered fields get instant score=0 (no further checks).
    This verification time is NOT counted against the model's execution time.
    
    Args:
        leads: List of leads to score
        icp: The ICP prompt
        costs: List of API costs per lead
        times: List of processing times per lead
        apply_similarity_detection: Whether to detect and penalize templated leads
    
    Returns:
        List of LeadScoreBreakdown objects
    """
    # =========================================================================
    # STEP 0: DB field verification (ONE batch query, then local comparison)
    # This catches gaming where models modify fields to better match the ICP.
    # Time for this check is NOT counted against the model.
    # =========================================================================
    try:
        db_failures = await verify_leads_batch(leads)
    except Exception as e:
        logger.error(f"DB verification failed with exception (proceeding without): {e}")
        db_failures = {}
    
    results = []
    seen_companies: Set[str] = set()
    
    for i, lead in enumerate(leads):
        # If DB verification failed for this lead, instant zero — skip all other checks
        if i in db_failures:
            logger.info(f"Lead {i} ({lead.business}) failed DB verification — instant zero")
            results.append(LeadScoreBreakdown(
                icp_fit=0,
                decision_maker=0,
                intent_signal_raw=0,
                time_decay_multiplier=1.0,
                intent_signal_final=0,
                cost_penalty=0,
                time_penalty=0,
                final_score=0,
                failure_reason=db_failures[i]
            ))
            continue
        
        cost = costs[i] if i < len(costs) else 0.0
        time = times[i] if i < len(times) else 0.0
        
        score = await score_lead(lead, icp, cost, time, seen_companies)
        results.append(score)
    
    # Apply structural similarity detection
    if apply_similarity_detection and len(leads) >= 3:
        flagged_indices = detect_structural_similarity(leads)
        
        # Zero out scores for flagged leads
        if flagged_indices:
            for idx in flagged_indices:
                if idx < len(results) and results[idx].failure_reason is None:
                    # Create new breakdown with zeroed score
                    results[idx] = LeadScoreBreakdown(
                        icp_fit=0,
                        decision_maker=0,
                        intent_signal_raw=0,
                        time_decay_multiplier=1.0,
                        intent_signal_final=0,
                        cost_penalty=0,
                        time_penalty=0,
                        final_score=0,
                        failure_reason="Structural similarity detected (templated response)"
                    )
    
    return results


# =============================================================================
# Scoring Summary
# =============================================================================

def summarize_scores(scores: list[LeadScoreBreakdown]) -> dict:
    """
    Summarize scoring results for a batch.
    
    Args:
        scores: List of LeadScoreBreakdown objects
    
    Returns:
        Summary statistics
    """
    if not scores:
        return {
            "total_leads": 0,
            "scored_leads": 0,
            "failed_leads": 0,
            "total_score": 0.0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
        }
    
    scored = [s for s in scores if s.failure_reason is None]
    failed = [s for s in scores if s.failure_reason is not None]
    
    all_final_scores = [s.final_score for s in scores]
    scored_final_scores = [s.final_score for s in scored] if scored else [0.0]
    
    return {
        "total_leads": len(scores),
        "scored_leads": len(scored),
        "failed_leads": len(failed),
        "total_score": sum(all_final_scores),
        "avg_score": sum(all_final_scores) / len(scores) if scores else 0.0,
        "avg_score_scored_only": sum(scored_final_scores) / len(scored) if scored else 0.0,
        "max_score": max(all_final_scores),
        "min_score": min(all_final_scores),
        "failure_reasons": [s.failure_reason for s in failed],
        "score_breakdown": {
            "avg_icp_fit": sum(s.icp_fit for s in scored) / len(scored) if scored else 0,
            "avg_decision_maker": sum(s.decision_maker for s in scored) / len(scored) if scored else 0,
            "avg_intent_signal": sum(s.intent_signal_final for s in scored) / len(scored) if scored else 0,
            "leads_with_cost_penalty": sum(1 for s in scored if s.cost_penalty > 0),
            "leads_with_time_penalty": sum(1 for s in scored if s.time_penalty > 0),
            "total_variability_penalty_pts": sum(s.cost_penalty + s.time_penalty for s in scored),
        }
    }
