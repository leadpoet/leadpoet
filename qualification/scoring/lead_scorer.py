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
from typing import Set, Optional, Tuple

from gateway.qualification.config import CONFIG
from gateway.qualification.models import LeadOutput, ICPPrompt, LeadScoreBreakdown
from qualification.scoring.pre_checks import run_automatic_zero_checks
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
        
        # Score Intent Signal (0-50 pts) - includes verification
        intent_raw, verification_confidence = await score_intent_signal(lead, icp)
        logger.debug(f"Intent signal raw score: {intent_raw} (confidence: {verification_confidence})")
        
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
    # STEP 4: Apply time decay to intent signal
    # =========================================================================
    try:
        signal_date = date.fromisoformat(lead.intent_signal.date)
    except (ValueError, AttributeError):
        # Invalid date - use current date (no decay)
        signal_date = date.today()
        logger.warning(f"Invalid signal date, using today: {lead.intent_signal.date}")
    
    age_months = calculate_age_months(signal_date)
    decay_multiplier = calculate_time_decay_multiplier(age_months)
    
    intent_final = intent_raw * decay_multiplier
    logger.debug(f"Intent after decay: {intent_final} (age: {age_months:.1f} months, multiplier: {decay_multiplier})")
    
    # =========================================================================
    # STEP 5: Calculate variability penalties
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
    
    # Time variability penalty
    time_penalty_threshold = CONFIG.get_time_penalty_threshold()
    if run_time_seconds > time_penalty_threshold:
        time_penalty = float(CONFIG.VARIABILITY_PENALTY_POINTS)
        logger.debug(
            f"Time variability penalty applied: {run_time_seconds:.1f}s > "
            f"{time_penalty_threshold:.1f}s (2× {CONFIG.MAX_TIME_PER_LEAD_SECONDS:.1f}s)"
        )
    
    logger.debug(f"Variability penalties - cost: {cost_penalty:.0f} pts, time: {time_penalty:.0f} pts")
    
    # =========================================================================
    # STEP 6: Calculate final score (floor at 0)
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
    prompt = f"""Score how well this lead matches the Ideal Customer Profile (ICP) on a scale of 0-20.

ICP CRITERIA:
- Industry: {icp.industry}
- Sub-industry: {icp.sub_industry}
- Target roles: {', '.join(icp.target_roles) if icp.target_roles else 'Any'}
- Target seniority: {icp.target_seniority}
- Company size: {icp.company_size}
- Geography: {icp.geography}

LEAD DATA:
- Industry: {lead.industry}
- Sub-industry: {lead.sub_industry}
- Role: {lead.role}
- Seniority: {lead.seniority.value if hasattr(lead.seniority, 'value') else lead.seniority}
- Employee count: {lead.employee_count}
- Company: {lead.business}
- Location: {lead.city}, {lead.state}, {lead.country}

SCORING GUIDELINES:
- 18-20: Perfect or near-perfect match on all criteria
- 14-17: Strong match with minor deviations
- 10-13: Good match but some criteria don't align
- 5-9: Partial match, significant gaps
- 0-4: Poor match, most criteria don't align

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
    prompt = f"""Score whether this role is likely a decision-maker for purchasing "{icp.product_service}" on a scale of 0-30.

LEAD:
- Role: {lead.role}
- Seniority: {lead.seniority.value if hasattr(lead.seniority, 'value') else lead.seniority}
- Company: {lead.business}
- Industry: {lead.industry}

SCORING GUIDELINES:
- 25-30: Definitely a decision-maker (C-suite, VP with relevant authority, budget owner)
- 18-24: Likely a decision-maker or strong influencer
- 10-17: May influence the decision but unlikely to have final authority
- 5-9: Has some involvement but limited decision power
- 0-4: Unlikely to be involved in purchasing decisions

Consider:
1. Is this role typically involved in purchasing this type of product?
2. Does the seniority level suggest budget authority?
3. Is this someone a sales team should prioritize reaching out to?

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
    "other": 0.3,              # LOW: catch-all category indicates fallback
}


async def score_intent_signal(lead: LeadOutput, icp: ICPPrompt) -> Tuple[float, int]:
    """
    Score the quality and relevance of the intent signal.
    
    First verifies the signal is real AND provides ICP evidence, then scores relevance.
    The final score is weighted by verification confidence AND source type quality.
    
    Args:
        lead: The lead to score
        icp: The ICP prompt
    
    Returns:
        Tuple of (score 0-50, verification_confidence 0-100)
    """
    # Build ICP criteria string for verification
    icp_criteria_parts = []
    if getattr(icp, 'company_size', None):
        icp_criteria_parts.append(f"Company size: {icp.company_size}")
    if getattr(icp, 'company_stage', None):
        icp_criteria_parts.append(f"Company stage: {icp.company_stage}")
    if getattr(icp, 'geography', None):
        icp_criteria_parts.append(f"Geography: {icp.geography}")
    icp_criteria = "; ".join(icp_criteria_parts) if icp_criteria_parts else None
    
    # Verify the signal is real AND provides evidence of ICP fit
    # This uses ScrapingDog to fetch the URL and verify the content
    verified, confidence, reason = await verify_intent_signal(
        lead.intent_signal,
        icp_industry=icp.industry,
        icp_criteria=icp_criteria,
        company_name=lead.business
    )
    
    if not verified:
        logger.info(f"Intent signal not verified: {reason}")
        return 0.0, confidence
    
    # Get source as string
    source_str = lead.intent_signal.source.value if hasattr(lead.intent_signal.source, 'value') else str(lead.intent_signal.source)
    
    # Get source type multiplier (penalize low-value sources like "other")
    source_multiplier = SOURCE_TYPE_MULTIPLIERS.get(source_str.lower(), 0.5)
    
    # Score the relevance of the verified signal
    prompt = f"""Score how relevant this verified intent signal is for selling "{icp.product_service}" on a scale of 0-50.

INTENT SIGNAL:
- Source: {source_str}
- Description: {lead.intent_signal.description}
- Date: {lead.intent_signal.date}
- Snippet: {lead.intent_signal.snippet or 'N/A'}

TARGET PRODUCT/SERVICE: {icp.product_service}

SCORING GUIDELINES:
- 40-50: Strong buying intent directly related to this product category
- 30-39: Clear interest in the problem this product solves
- 20-29: Relevant activity that suggests potential need
- 10-19: Tangentially related signal, weak buying intent
- 0-9: Signal exists but not relevant to this product

IMPORTANT: Penalize generic descriptions. Examples of LOW scores:
- "Company is actively operating in X industry" → 0-5 (too generic)
- "Visible market activity" → 0-5 (no specific intent)
- Vague descriptions without specific actions → max 10

Consider:
1. Does this signal indicate explicit buying intent for this type of product?
2. How strong is the signal (job posting, public statement, hiring activity)?
3. Is this actionable for a sales team?
4. Is the description specific or generic/templated?

Respond with ONLY a single number (0-50):"""

    response = await openrouter_chat(prompt, model="gpt-4o-mini")
    raw_score = extract_score(response, max_score=MAX_INTENT_SIGNAL_SCORE)
    
    # Weight by verification confidence AND source type quality
    weighted_score = raw_score * (confidence / 100) * source_multiplier
    
    if source_multiplier < 1.0:
        logger.info(f"Applied source type penalty: {source_str} -> {source_multiplier}x")
    
    return weighted_score, confidence


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
# Batch Scoring
# =============================================================================

async def score_leads_batch(
    leads: list[LeadOutput],
    icp: ICPPrompt,
    costs: list[float],
    times: list[float]
) -> list[LeadScoreBreakdown]:
    """
    Score a batch of leads against the same ICP.
    
    Tracks seen companies across the batch to enforce first-wins rule.
    
    Args:
        leads: List of leads to score
        icp: The ICP prompt
        costs: List of API costs per lead
        times: List of processing times per lead
    
    Returns:
        List of LeadScoreBreakdown objects
    """
    results = []
    seen_companies: Set[str] = set()
    
    for i, lead in enumerate(leads):
        cost = costs[i] if i < len(costs) else 0.0
        time = times[i] if i < len(times) else 0.0
        
        score = await score_lead(lead, icp, cost, time, seen_companies)
        results.append(score)
    
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
