"""
Qualification System: Model Competition Scoring

This module implements the validator-side scoring for the Lead
Qualification Agent competition (a.k.a. the model competition).

As of May 2026 the competition surfaces COMPANIES from the open web
that match an ICP and carry verifiable intent signals — NOT contacts.
The historical lead-mode pipeline (DB row equality, role / seniority /
decision-maker LLM, email validation) has been removed in favor of a
single-path company-mode pipeline.  Rationale: cleanly finding
contacts requires Apify / LinkedIn scraping, which we do not want
baked into the base miner model.  Fulfillment miners can layer their
own contact enrichment on top of a license-clean base model.

Scoring flow:
  1. ``run_company_zero_checks`` — deterministic gates (industry +
     sub-industry + country match, dup-company tracking, hard time
     limit).  No role / seniority / email checks.
  2. ``verify_company_exists`` — HTTP fetch of the company website;
     fail → score 0.  Plays the anti-fabrication role that DB row
     equality used to play in the old lead-mode pipeline.
  3. ``score_company_icp_fit`` — single LLM call, 0-40 (industry,
     product fit, structural fit, intent-class fit; no role).
  4. ``score_company_intent_signal`` — per-signal verification via
     ``verify_intent_signal`` + URL dedup + time decay, 0-60.
  5. Cost variability penalty.
  6. Final score = max(0, icp_fit + intent_final - cost_penalty).

Max Score: MAX_COMPANY_TOTAL_SCORE = 100.

Cross-module dependencies kept for fulfillment compatibility:
  * ``_score_single_intent_signal``, ``_apply_signal_time_decay``,
    ``_extract_domain``, ``detect_structural_similarity`` are
    imported by ``gateway/fulfillment/scoring.py``.  Do not rename
    or move them.

CRITICAL: This module is the validator-side model-competition scorer
ONLY.  It must not import from or be coupled to fulfillment-side
verification (Stage 4 person verification, etc.).
"""

import os
import re
import logging
from datetime import date, datetime
from typing import Set, Optional, Tuple, List
from collections import Counter
from urllib.parse import urlparse

from gateway.qualification.config import CONFIG
from gateway.qualification.models import (
    LeadOutput,        # re-exported for fulfillment imports via this module
    ICPPrompt,
    LeadScoreBreakdown,
    CompanyOutput,
)
from qualification.scoring.pre_checks import run_company_zero_checks
from qualification.scoring.verification_helpers import (
    is_generic_intent_description,
    check_future_date,
    openrouter_chat,
)
from qualification.scoring.intent_signal_gate import judge_intent_signal
from qualification.scoring.company_verification import verify_company_exists

# Feature flag for the strict LLM judge (Layer 4 of intent_signal_gate).
# On by default.  Set INTENT_GATE_STRICT_JUDGE_ENABLED=false to disable
# the Layer 4 LLM judge; Layers 1-3 (anti-bot, structural URL/category,
# freshness window, self-published bias) still run inside
# verify_intent_signal regardless.
INTENT_GATE_STRICT_JUDGE_ENABLED = (
    os.getenv("INTENT_GATE_STRICT_JUDGE_ENABLED", "true").strip().lower()
    in ("true", "1", "yes", "on")
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Score component maximums
# No decision-maker / role / contact dimension: there is no contact
# in the model competition.  The 40-point ICP-fit budget covers
# industry + product + structural + intent-class fit; intent signals
# carry the other 60.
MAX_COMPANY_ICP_FIT_SCORE = 40
MAX_COMPANY_INTENT_SIGNAL_SCORE = 60
MAX_COMPANY_TOTAL_SCORE = MAX_COMPANY_ICP_FIT_SCORE + MAX_COMPANY_INTENT_SIGNAL_SCORE  # = 100
MAX_AUTORESEARCH_INTENT_SCORE = 100
AUTORESEARCH_INTENT_CAP_BY_SIGNAL_COUNT = {
    1: 60.0,
    2: 80.0,
    3: 88.0,
    4: 92.0,
    5: 96.0,
    6: 100.0,
}

# Per-signal LLM score cap (each individual intent signal scores 0-60
# inside ``_score_single_intent_signal``).  Kept as an alias for the
# previous lead-mode name because ``_score_single_intent_signal`` is
# also imported directly by ``gateway/fulfillment/scoring.py``.
MAX_INTENT_SIGNAL_SCORE = MAX_COMPANY_INTENT_SIGNAL_SCORE

# LLM temperature for scoring (slightly higher for nuanced scoring)
SCORING_TEMPERATURE = 0.4


# =============================================================================
# Main Scoring Function — Company-Mode Model Competition
# =============================================================================
#
# Single-path scorer.  Lead-mode (DB-row equality + role / seniority /
# decision-maker LLM + email validation) was removed when the model
# competition was retargeted to surface high-intent COMPANIES from the
# open web (see module docstring).  The historical lead-mode helpers
# (_score_single_intent_signal, _apply_signal_time_decay,
# _extract_domain, detect_structural_similarity, time-bound ICP
# regex, etc.) remain in this module — they are reused by
# gateway/fulfillment/scoring.py for fulfillment-side ranking, which
# DOES still need contact-aware scoring.  Do not move them.
#
# Total max score = MAX_COMPANY_TOTAL_SCORE = 100 (40 ICP + 60 intent),
# so the existing champion thresholds in CONFIG
# (MINIMUM_CHAMPION_SCORE, CHAMPION_DETHRONING_THRESHOLD_POINTS) carry
# over unchanged.


async def score_company(
    company: CompanyOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
    force_fail_reason: Optional[str] = None,
    is_reference_model: bool = False,
) -> LeadScoreBreakdown:
    """Score a CompanyOutput against an ICP.

    Returns a ``LeadScoreBreakdown`` with the historical four-field
    shape (``icp_fit``, ``decision_maker``, ``intent_signal_*``,
    penalties, ``final_score``) so the validator's aggregation,
    transparency logging, and champion-status reporting can stay
    unchanged.  ``decision_maker`` is always 0 (there is no contact
    in this model competition); the 40-point ICP-fit budget covers
    industry + product + structural + intent-class fit.

    Pipeline:

      0. Forced-fail short-circuit (e.g. structural-templating
         detection from the caller's per-batch dedup pass).
      1. ``run_company_zero_checks`` — country/geo match, duplicate
         company tracking, cost / time hard limits.  Skips role /
         seniority / DB-row checks.
      2. ``verify_company_exists`` — HTTP fetch of the company's
         website to confirm it's a real page that mentions the
         claimed company name and isn't a parked / for-sale domain.
         Hard gate: failure -> score 0.
      3. ``score_company_icp_fit`` — single LLM call, 0-40 score
         (richer prompt than lead-mode ICP fit).
      4. ``score_company_intent_signal`` — per-signal verification +
         time decay, identical algorithm to lead-mode.  Fabrication
         detection: if all signals are fabricated, zero entire score.
      5. Cost variability penalty (same rules as lead-mode).
      6. ``final_score = max(0, icp_fit + intent_final - cost_penalty)``.
    """
    if force_fail_reason:
        logger.info(
            f"Company forced to fail (company-mode): {force_fail_reason}"
        )
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=force_fail_reason,
        )

    # -----------------------------------------------------------------
    # STEP 1: Company-mode pre-checks (deterministic, no LLM)
    # -----------------------------------------------------------------
    passes, failure_reason = await run_company_zero_checks(
        company, icp, run_cost_usd, run_time_seconds, seen_companies
    )
    if not passes:
        logger.info(f"Company failed company-mode pre-checks: {failure_reason}")
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=failure_reason,
        )

    # -----------------------------------------------------------------
    # STEP 2: Company-existence verification (web fetch — hard gate)
    # -----------------------------------------------------------------
    try:
        co_verified, co_reason = await verify_company_exists(
            company.company_name, company.company_website
        )
    except Exception as e:
        # Never let a transient web error crash the scorer; log and
        # treat as a soft failure so the model just gets a 0 on this
        # ICP (rather than wedging the whole evaluation batch).
        logger.error(f"Company verification raised: {e}")
        co_verified, co_reason = False, f"company verification error: {str(e)[:120]}"
    if not co_verified:
        logger.info(
            f"Company failed existence check: {co_reason} "
            f"(name={company.company_name!r}, url={company.company_website!r})"
        )
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=f"Company verification failed: {co_reason}",
        )
    logger.debug(f"Company existence verified: {co_reason}")

    # -----------------------------------------------------------------
    # STEP 3: Mark company as seen (first lead per company wins)
    # -----------------------------------------------------------------
    if company.company_name:
        seen_companies.add(company.company_name.lower().strip())

    # -----------------------------------------------------------------
    # STEP 4: LLM-based scoring
    # -----------------------------------------------------------------
    try:
        icp_fit = await score_company_icp_fit(company, icp)
        logger.debug(f"Company ICP fit score: {icp_fit}")

        intent_raw, intent_final, decay_multiplier, _max_confidence, all_fabricated = (
            await score_company_intent_signal(company, icp)
        )
        logger.debug(
            f"Company intent signal avg_raw={intent_raw:.1f}, "
            f"avg_final={intent_final:.1f}, decay={decay_multiplier:.2f}"
        )

        # Fabrication zeroing — same rule as lead-mode.
        if all_fabricated:
            logger.warning(
                f"❌ ALL INTENT SIGNALS FABRICATED for company "
                f"{company.company_name!r} — zeroing entire score"
            )
            return LeadScoreBreakdown(
                icp_fit=0,
                decision_maker=0,
                intent_signal_raw=0,
                time_decay_multiplier=1.0,
                intent_signal_final=0,
                cost_penalty=0,
                time_penalty=0,
                final_score=0,
                failure_reason=(
                    "Intent fabrication detected (hardcoded date or "
                    "generic claim)"
                ),
            )
    except Exception as e:
        logger.error(f"Company-mode LLM scoring failed: {e}")
        return LeadScoreBreakdown(
            icp_fit=0,
            decision_maker=0,
            intent_signal_raw=0,
            time_decay_multiplier=1.0,
            intent_signal_final=0,
            cost_penalty=0,
            time_penalty=0,
            final_score=0,
            failure_reason=f"LLM scoring error: {str(e)[:100]}",
        )

    # -----------------------------------------------------------------
    # STEP 5: Cost variability penalty (same rules as lead-mode)
    # -----------------------------------------------------------------
    # The reference / baseline model that the validator runs daily to set
    # the per-day champion floor is exempt from the cost variability
    # penalty — its purpose is to set a fair ceiling on what's achievable,
    # not to compete on cost.  Miner submissions remain subject to the
    # penalty as before.  The actual cost / time are logged in either
    # case so the value is fully traceable independent of the penalty.
    cost_penalty = 0.0
    time_penalty = 0.0
    cost_penalty_threshold = CONFIG.get_cost_penalty_threshold()
    cost_over = run_cost_usd > cost_penalty_threshold
    if is_reference_model:
        # Trace: the cost still must be visible even though no penalty applies.
        logger.info(
            f"[reference_model] cost_recorded=${run_cost_usd:.4f} "
            f"time_recorded={run_time_seconds:.1f}s  "
            f"threshold=${cost_penalty_threshold:.4f}  "
            f"would_have_penalized={cost_over}  "
            f"penalty_applied=False (exempt)"
        )
    else:
        if cost_over:
            cost_penalty = float(CONFIG.VARIABILITY_PENALTY_POINTS)
            logger.info(
                f"[miner] cost variability penalty applied: "
                f"${run_cost_usd:.4f} > ${cost_penalty_threshold:.4f}  "
                f"penalty=-{cost_penalty:.0f} pts"
            )
        else:
            logger.debug(
                f"[miner] cost ${run_cost_usd:.4f} within threshold "
                f"${cost_penalty_threshold:.4f}, no penalty"
            )

    # -----------------------------------------------------------------
    # STEP 6: Final score (floor at 0, ceiling at MAX_COMPANY_TOTAL_SCORE)
    # -----------------------------------------------------------------
    total_raw = icp_fit + intent_final
    final_score = max(0.0, total_raw - cost_penalty - time_penalty)
    final_score = min(final_score, float(MAX_COMPANY_TOTAL_SCORE))

    total_penalty = cost_penalty + time_penalty
    role_tag = "reference" if is_reference_model else "miner"
    if total_penalty > 0:
        logger.info(
            f"Company scored [{role_tag}]: {final_score:.2f} "
            f"(ICP: {icp_fit}, Intent: {intent_final:.2f}, "
            f"cost=${run_cost_usd:.4f}, time={run_time_seconds:.1f}s, "
            f"Variability penalty: -{total_penalty:.0f} pts)"
        )
    else:
        logger.info(
            f"Company scored [{role_tag}]: {final_score:.2f} "
            f"(ICP: {icp_fit}, Intent: {intent_final:.2f}, "
            f"cost=${run_cost_usd:.4f}, time={run_time_seconds:.1f}s, "
            f"No variability penalty)"
        )

    return LeadScoreBreakdown(
        icp_fit=icp_fit,
        decision_maker=0,
        intent_signal_raw=intent_raw,
        time_decay_multiplier=decay_multiplier,
        intent_signal_final=intent_final,
        cost_penalty=cost_penalty,
        time_penalty=time_penalty,
        final_score=final_score,
        failure_reason=None,
    )


async def score_company_autoresearch_intent_v2(
    company: CompanyOutput,
    icp: ICPPrompt,
    run_cost_usd: float,
    run_time_seconds: float,
    seen_companies: Set[str],
    force_fail_reason: Optional[str] = None,
    is_reference_model: bool = False,
) -> LeadScoreBreakdown:
    """Opt-in Research Lab scorer: binary fit gates, 0-100 intent-only score.

    This intentionally leaves ``score_company`` unchanged for the public model
    competition and fulfillment-adjacent imports.  The Research Lab private
    evaluator opts into this function explicitly.
    """
    if force_fail_reason:
        logger.info(
            f"Autoresearch company forced to fail: {force_fail_reason}"
        )
        return _zero_company_breakdown(force_fail_reason)

    passes, failure_reason = await run_company_zero_checks(
        company, icp, run_cost_usd, run_time_seconds, seen_companies
    )
    if not passes:
        logger.info(f"Autoresearch company failed pre-checks: {failure_reason}")
        return _zero_company_breakdown(failure_reason)

    binary_passed, binary_reason = _run_autoresearch_binary_fit_checks(company, icp)
    if not binary_passed:
        logger.info(f"Autoresearch company failed binary fit: {binary_reason}")
        return _zero_company_breakdown(binary_reason)

    try:
        co_verified, co_reason = await verify_company_exists(
            company.company_name, company.company_website
        )
    except Exception as e:
        logger.error(f"Autoresearch company verification raised: {e}")
        co_verified, co_reason = False, f"company verification error: {str(e)[:120]}"
    if not co_verified:
        return _zero_company_breakdown(f"Company verification failed: {co_reason}")

    if company.company_name:
        seen_companies.add(company.company_name.lower().strip())

    try:
        intent_raw, intent_final, decay_multiplier, _max_confidence, all_fabricated = (
            await score_company_autoresearch_intent_signal(company, icp)
        )
        if all_fabricated:
            logger.warning(
                f"❌ ALL AUTORESEARCH INTENT SIGNALS FABRICATED for company "
                f"{company.company_name!r} — zeroing entire score"
            )
            return _zero_company_breakdown(
                "Intent fabrication detected (hardcoded date or generic claim)"
            )
    except Exception as e:
        logger.error(f"Autoresearch intent scoring failed: {e}")
        return _zero_company_breakdown(f"LLM scoring error: {str(e)[:100]}")

    final_score = max(0.0, min(float(MAX_AUTORESEARCH_INTENT_SCORE), intent_final))
    role_tag = "reference" if is_reference_model else "miner"
    logger.info(
        f"Autoresearch company scored [{role_tag}]: {final_score:.2f} "
        f"(IntentV2:{intent_final:.2f}, cost=${run_cost_usd:.4f}, "
        f"time={run_time_seconds:.1f}s)"
    )
    return LeadScoreBreakdown(
        icp_fit=0,
        decision_maker=0,
        intent_signal_raw=intent_raw,
        time_decay_multiplier=decay_multiplier,
        intent_signal_final=final_score,
        cost_penalty=0,
        time_penalty=0,
        final_score=final_score,
        failure_reason=None,
    )


def _zero_company_breakdown(reason: Optional[str]) -> LeadScoreBreakdown:
    return LeadScoreBreakdown(
        icp_fit=0,
        decision_maker=0,
        intent_signal_raw=0,
        time_decay_multiplier=1.0,
        intent_signal_final=0,
        cost_penalty=0,
        time_penalty=0,
        final_score=0,
        failure_reason=reason,
    )


def _run_autoresearch_binary_fit_checks(
    company: CompanyOutput, icp: ICPPrompt
) -> Tuple[bool, Optional[str]]:
    company_bucket = _normalize_linkedin_employee_bucket(company.employee_count)
    icp_buckets = _normalize_icp_employee_buckets(icp.employee_count)
    if icp_buckets:
        if not company_bucket:
            return False, "Missing employee_count bucket"
        if company_bucket not in icp_buckets:
            return (
                False,
                f"Employee count mismatch: '{company.employee_count}' not in {sorted(icp_buckets)}",
            )

    icp_stage = _normalize_company_stage(icp.company_stage)
    if icp_stage:
        company_stage = _normalize_company_stage(company.company_stage)
        if not company_stage:
            return False, f"Missing company_stage (ICP requires '{icp.company_stage}')"
        if company_stage != icp_stage:
            return (
                False,
                f"Company stage mismatch: '{company.company_stage}' vs '{icp.company_stage}'",
            )
    return True, None


def _normalize_linkedin_employee_bucket(value) -> str:
    try:
        from research_lab.employee_buckets import normalize_employee_count_bucket

        return normalize_employee_count_bucket(value, default=None)
    except Exception as e:
        logger.warning(
            "autoresearch employee bucket normalization failed: %s: %s",
            type(e).__name__, e,
        )
        return ""


def _normalize_icp_employee_buckets(value) -> set:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"any", "all", "unknown", "n/a", "na"}:
        return set()
    pieces = [
        item.strip()
        for item in re.split(r"\s*(?:\||;|,|\bor\b)\s*", raw, flags=re.I)
        if item.strip()
    ]
    buckets = {
        _normalize_linkedin_employee_bucket(piece)
        for piece in pieces
    }
    return {bucket for bucket in buckets if bucket}


def _normalize_company_stage(value) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"any", "all", "unknown", "n/a", "na", "not specified"}:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


async def score_company_icp_fit(
    company: CompanyOutput, icp: ICPPrompt, api_key: str = ""
) -> float:
    """Company-mode ICP-fit scorer (0-40).

    Replaces the lead-mode trio of ``score_icp_fit`` (industry +
    product + structural fit, max 20) + ``score_decision_maker``
    (role + authority, max 20).  In company-mode there is no
    contact, so the decision-maker dimension is removed and the
    ICP-fit budget is widened to 40 with four sub-scores:

      1. Industry / sub-industry fit            (0-10)
      2. Product / service buying fit           (0-10)
      3. Structural fit (size, geo, stage)      (0-10)
      4. ICP intent-class alignment             (0-10)
         (Does this company plausibly carry the
          *kind* of intent the buyer asked for?
          Verifying individual signals is the
          job of score_company_intent_signal —
          here we just check fit.)
    """
    icp_product = icp.product_service or ""
    icp_prompt_text = icp.prompt or ""
    icp_signals_str = (
        "; ".join(icp.intent_signals)
        if icp.intent_signals
        else "Any verifiable buying intent"
    )

    prompt = f"""You are scoring how well a company matches a buyer's Ideal Customer Profile on a 0-40 scale.

ICP CRITERIA:
- Industry: {icp.industry}
- Sub-industry: {icp.sub_industry}
- Employee count: {icp.employee_count}
- Company stage: {icp.company_stage}
- Geography: {icp.geography}
- Product/service the buyer is selling: {icp_product}
- Intent signals the buyer wants the company to be showing: {icp_signals_str}
- Full buyer request: "{icp_prompt_text}"

COMPANY DATA:
- Company: {company.company_name}
- Website: {company.company_website}
- Industry: {company.industry}
- Sub-industry: {company.sub_industry}
- Employee count: {company.employee_count}
- Company stage: {company.company_stage}
- Location: {company.country} ({company.state or 'state unspecified'})
- Description: {company.description or '(none provided)'}

SCORING — give a sub-score for EACH dimension then sum.  All dimensions are 0-10.

1. INDUSTRY FIT (0-10):
   - Exact industry + sub-industry match: 9-10
   - Same industry, different sub-industry: 6-8
   - Adjacent/related industry: 3-5
   - Unrelated: 0-2

2. PRODUCT-FIT (0-10):
   - Company is clearly a likely buyer of "{icp_product}" given its
     business model: 8-10
   - Company plausibly uses this kind of product: 5-7
   - Weak product fit: 2-4
   - No connection: 0-1

3. STRUCTURAL FIT (0-10):
   - Employee count, company stage, AND geography all match the ICP: 9-10
   - 2 of 3 structural criteria match: 6-8
   - 1 of 3 structural criteria match: 3-5
   - None match: 0-2

4. INTENT-CLASS FIT (0-10):
   This is about whether the *type* of company is consistent with the
   buyer's intent class, not whether individual intent signals are
   verified (that's done separately).
   - The company is the kind of company that would plausibly show
     the buyer's expected intent signals AND its description /
     industry is consistent with those signals: 8-10
   - Plausible match but mixed signals: 5-7
   - Tenuous: 2-4
   - Clearly inconsistent: 0-1

Sum the four sub-scores.  Final score is in [0, 40].

CRITICAL: Be conservative.  If the company's industry / sub-industry
does NOT match the ICP, even high product-fit and structural-fit
shouldn't push the total above 20.  The buyer told us their industry.

Respond with ONLY a single integer 0-40."""

    response = await openrouter_chat(prompt, model="gpt-4o-mini", api_key=api_key)
    score = extract_score(response, max_score=MAX_COMPANY_ICP_FIT_SCORE)
    return score


async def score_company_intent_signal(
    company: CompanyOutput, icp: ICPPrompt, api_key: str = ""
) -> Tuple[float, float, float, int, bool]:
    """Score ALL intent signals on a CompanyOutput.

    Identical algorithm to ``score_intent_signal`` (lead-mode) but
    parameterized over CompanyOutput fields.  Reuses
    ``_score_single_intent_signal`` so every per-signal rule
    (verification via ``verify_intent_signal``, source multipliers,
    time decay, dedup, fabrication marker) is shared.

    Returns ``(avg_raw, avg_final, avg_decay, max_confidence, all_fabricated)``
    — same tuple shape as ``score_intent_signal``.
    """
    icp_criteria = None  # Same as score_intent_signal — built inside _score_single
    seen_domains: set = set()
    signal_results = []

    for signal in company.intent_signals:
        domain = _extract_domain(signal.url)
        if domain in seen_domains:
            logger.warning(
                f"  ⚠ Duplicate domain {domain!r} on company "
                f"{company.company_name!r} — signal scores 0 (URL dedup)"
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

        score, confidence, date_status, content_found_date, _matched_idx = (
            await _score_single_intent_signal(
                signal,
                icp,
                icp_criteria,
                company.company_name,
                company.company_website,
                api_key=api_key,
            )
        )

        after_decay, decay = _apply_signal_time_decay(
            score, signal.date, date_status,
            signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
            content_found_date=content_found_date,
        )
        signal_results.append({
            "raw": score,
            "after_decay": after_decay,
            "decay": decay,
            "confidence": confidence,
            "date_status": date_status,
        })

    if not signal_results:
        return 0.0, 0.0, 0.0, 0, True

    raw_scores = [r["raw"] for r in signal_results]
    decayed_scores = [r["after_decay"] for r in signal_results]
    decays = [r["decay"] for r in signal_results if r["decay"] > 0]
    confidences = [r["confidence"] for r in signal_results]

    avg_raw = sum(raw_scores) / len(raw_scores)
    avg_final = sum(decayed_scores) / len(decayed_scores)
    avg_decay = sum(decays) / len(decays) if decays else 0.0
    max_confidence = max(confidences) if confidences else 0
    # Fabrication marker: every signal was either fabricated, a domain
    # dup, or otherwise scored 0.  Matches lead-mode semantics.
    all_fabricated = all(r["raw"] == 0.0 for r in signal_results)

    return avg_raw, avg_final, avg_decay, max_confidence, all_fabricated


async def score_company_autoresearch_intent_signal(
    company: CompanyOutput, icp: ICPPrompt, api_key: str = ""
) -> Tuple[float, float, float, int, bool]:
    """Score CompanyOutput intent signals with capped-sum breadth rewards."""
    icp_criteria = None
    seen_domains: set = set()
    signal_results = []

    for signal in company.intent_signals:
        domain = _extract_domain(signal.url)
        if domain in seen_domains:
            logger.warning(
                f"  ⚠ Duplicate domain {domain!r} on autoresearch company "
                f"{company.company_name!r} — signal scores 0 (URL dedup)"
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

        score, confidence, date_status, content_found_date, _matched_idx = (
            await _score_single_intent_signal(
                signal,
                icp,
                icp_criteria,
                company.company_name,
                company.company_website,
                api_key=api_key,
                company_linkedin=getattr(company, "company_linkedin", "") or "",
                product_service_context=getattr(icp, "product_service", "") or "",
            )
        )
        after_decay, decay = _apply_signal_time_decay(
            score, signal.date, date_status,
            signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
            content_found_date=content_found_date,
        )
        signal_results.append({
            "raw": score,
            "after_decay": after_decay,
            "decay": decay,
            "confidence": confidence,
            "date_status": date_status,
        })

    if not signal_results:
        return 0.0, 0.0, 0.0, 0, True

    raw_scores = [r["raw"] for r in signal_results]
    decayed_scores = [r["after_decay"] for r in signal_results]
    decays = [r["decay"] for r in signal_results if r["decay"] > 0]
    confidences = [r["confidence"] for r in signal_results]
    raw_total = aggregate_autoresearch_intent_scores(raw_scores)
    final_total = aggregate_autoresearch_intent_scores(decayed_scores)
    avg_decay = sum(decays) / len(decays) if decays else 0.0
    max_confidence = max(confidences) if confidences else 0
    all_fabricated = all(r["raw"] == 0.0 for r in signal_results)
    return raw_total, final_total, avg_decay, max_confidence, all_fabricated


def aggregate_autoresearch_intent_scores(signal_scores: List[float]) -> float:
    """Capped sum over top verified signals, with monotonic breadth caps."""
    positives = sorted(
        [max(0.0, float(score or 0.0)) for score in signal_scores if float(score or 0.0) > 0.0],
        reverse=True,
    )[:6]
    if not positives:
        return 0.0
    cap = AUTORESEARCH_INTENT_CAP_BY_SIGNAL_COUNT[len(positives)]
    return min(sum(positives), cap)


# =============================================================================
# Lead-mode ICP-fit / decision-maker / intent scorers REMOVED (May 2026)
# =============================================================================
# When the model competition was retargeted from leads-with-contacts to
# companies-from-the-open-web, three lead-mode entry points
# (``score_icp_fit(lead, icp)``, ``score_decision_maker(lead, icp)``,
# and the lead-mode ``score_intent_signal(lead, icp)``) became dead
# code and were deleted.  Their company-mode equivalents are
# ``score_company_icp_fit(company, icp)`` and
# ``score_company_intent_signal(company, icp)`` defined above.  There
# is no decision-maker dimension in company-mode (there's no contact).
#
# The lead-mode helpers ``_score_single_intent_signal``,
# ``_apply_signal_time_decay``, ``_extract_domain``,
# ``_parse_intent_score_response``, ``SOURCE_TYPE_MULTIPLIERS``,
# ``SOURCES_DATE_*``, ``_TIME_BOUND_ICP_PHRASES`` /
# ``_icp_signal_is_time_bound`` are KEPT because they are imported
# directly by ``gateway/fulfillment/scoring.py`` for fulfillment-side
# lead ranking, which still operates on contacts.
# =============================================================================

# =============================================================================
# Intent Signal Scoring  (shared helpers used by company-mode AND fulfillment)
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
    content_found_date: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Apply time decay to a single signal's raw score.
    
    Returns:
        Tuple of (after_decay_score, decay_multiplier)
    """
    NO_DATE_DECAY_MULTIPLIER = 0.5
    source_lower = (source_str or "").lower().strip()

    if date_status == "date_omitted" and content_found_date:
        # Model submitted date=null but our re-scrape found a date in the content.
        # Apply time decay based on the date we found — the model shouldn't get to
        # hide a real date to avoid decay.
        # EXCEPTION: sources that don't require dates (company_website, review_site,
        # etc.) are exempt — their pages often contain old dates in footers,
        # copyright notices, or unrelated content that shouldn't penalize the signal.
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return raw_score, 1.0
        try:
            parsed_date = date.fromisoformat(content_found_date)
        except (ValueError, AttributeError):
            parsed_date = None
        if parsed_date is not None:
            age_months = calculate_age_months(parsed_date)
            decay = calculate_time_decay_multiplier(age_months)
            logger.info(
                f"⚠️ Date omission: applying time decay from content date "
                f"{content_found_date} (age={age_months:.1f}mo, decay={decay:.2f}x)"
            )
            return raw_score * decay, decay
        return raw_score, 1.0

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
        if source_lower in SOURCES_DATE_NOT_REQUIRED:
            return raw_score, 1.0
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

MAX_INTENT_NO_DATE_REQUIRED = 18   # Cap for undated signals where date IS required
MAX_INTENT_NO_DATE_UNKNOWN = 48   # Cap for undated signals from unrecognized source types
MAX_INTENT_NO_DATE_OPTIONAL = 60  # Full score for undated signals where date is NOT required


# Compiled regex patterns that identify time-bound ICP intent signals. These
# are claims whose meaning depends on recency — submitting an undated source
# for a claim like "Raised seed funding in the last few weeks" defeats the
# purpose of the claim regardless of how trustworthy the source category is.
# Used by ``_icp_signal_is_time_bound`` below.
_TIME_BOUND_ICP_PHRASES = re.compile(
    r"\b("
    r"in the (last|past) (\d+ )?(few |couple of )?(day|week|month|quarter|year)s?"
    r"|in the last \d+"
    r"|this (week|month|quarter|year)"
    r"|last (week|month|quarter|year)"
    r"|past (week|month|quarter|year)"
    r"|recent(?:ly)?"
    r"|just (raised|secured|closed|launched|announced|hired|acquired|partnered)"
    r"|new(?:ly)? (funded|hired|launched|opened)"
    r"|(\d+\+? )?days? ago"
    r"|\bytd\b|year[- ]to[- ]date"
    r")\b",
    re.IGNORECASE,
)


def _icp_signal_is_time_bound(icp_signal_text: str) -> bool:
    """Return True when the ICP signal phrase encodes a recency requirement.

    Examples that should match (recency is the whole point):
      - "Raised Seed funding in the last few weeks"
      - "Hired a CTO this quarter"
      - "Recently posted 10+ engineering roles"
      - "Just announced Series B"

    Examples that should NOT match (state, not event):
      - "Uses Procore"
      - "Has 50-200 employees"
      - "Headquartered in Mexico"

    Keep this function pure — no API calls, no LLM. The matched-claim cap
    runs on every undated signal score, so this is on the hot path.
    """
    if not icp_signal_text:
        return False
    return bool(_TIME_BOUND_ICP_PHRASES.search(icp_signal_text))

def _parse_intent_score_response(
    response: str,
    max_score: int,
    num_icp_signals: int,
) -> Tuple[float, int]:
    """Parse the LLM response into ``(raw_score, matched_icp_signal_idx)``.

    Prefers strict JSON (``{"score": N, "matched_icp_signal_idx": I}``) but
    falls back to regex number extraction for score if JSON parsing fails.
    ``matched_icp_signal_idx`` is clamped to ``[-1, num_icp_signals - 1]``
    and defaults to ``-1`` (no match) on any parse failure.
    """
    import json as _json
    import re as _re

    if not response:
        return 0.0, -1

    text = response.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text)
    match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    json_str = match.group(0) if match else text

    try:
        obj = _json.loads(json_str)
        score = float(obj.get("score", 0))
        idx = int(obj.get("matched_icp_signal_idx", -1))
    except Exception:
        score = float(extract_score(response, max_score=max_score))
        idx = -1

    score = max(0.0, min(score, float(max_score)))
    if num_icp_signals <= 0 or idx < 0 or idx >= num_icp_signals:
        idx = -1
    return score, idx


async def _score_single_intent_signal(
    signal: "IntentSignal",
    icp: ICPPrompt,
    icp_criteria: Optional[str],
    company_name: str,
    company_website: str = "",
    api_key: str = "",
    company_linkedin: str = "",
    product_service_context: str = "",
) -> Tuple[float, int, str, Optional[str], int]:
    """
    Verify and score a single intent signal.

    Returns:
        Tuple of (score 0-60, verification_confidence 0-100, date_status,
                  content_found_date, matched_icp_signal_idx)

        ``matched_icp_signal_idx`` is the 0-based index into
        ``icp.intent_signals`` of the client-requested signal this miner
        signal satisfies, or ``-1`` if none. When ``icp.intent_signals``
        is empty, this is always ``-1``.
    """
    # ── Gate 0: miner-asserted matched_icp_signal must be set and in range ──
    # Each intent signal a miner submits MUST be tagged with the index of
    # the client-listed signal that this evidence is meant to satisfy.  A
    # value of -1 means the miner did not declare a target signal — we
    # reject those at scoring time rather than letting them silently fall
    # back to LLM-guessed matching.  Out-of-range values are also rejected
    # (defends against off-by-one from miner code that doesn't read the
    # request's icp_details list correctly).
    icp_signals_for_gate = list(getattr(icp, "intent_signals", None) or [])
    miner_asserted_idx = getattr(signal, "matched_icp_signal", -1)
    if not isinstance(miner_asserted_idx, int) or miner_asserted_idx < 0:
        logger.info(
            f"Intent signal rejected: matched_icp_signal not set "
            f"(value={miner_asserted_idx!r}).  Miner must declare which "
            f"client intent signal this evidence proves."
        )
        return 0.0, 0, "fabricated", None, -1
    if not icp_signals_for_gate or miner_asserted_idx >= len(icp_signals_for_gate):
        logger.info(
            f"Intent signal rejected: matched_icp_signal={miner_asserted_idx} "
            f"out of range (request has {len(icp_signals_for_gate)} listed signals)."
        )
        return 0.0, 0, "fabricated", None, -1

    # ── Cheap pre-checks BEFORE the three-stage LLM pipeline ────────────
    # These are deterministic rejects that don't require URL fetching:
    #   1. Generic / templated descriptions (cached blocklist patterns)
    #   2. "other" source type with a too-short description
    #   3. Future-dated signals (obviously fabricated)
    #
    # We intentionally do NOT call the older verify_intent_signal() Layer-1
    # gate here.  That function fetches via Scrapingdog only (no Exa
    # fallback), so on anti-bot / JS-heavy URLs it returns False and would
    # short-circuit the entire scoring before three-stage gets a chance to
    # use its SD + Exa fallback to crack the same URL.  Skipping the
    # Layer-1 fetch means three-stage is the sole content-aware verifier,
    # and its Exa fallback now reaches the URLs that need it most.
    confidence = 0
    content_found_date: Optional[str] = None
    date_status = "verified"

    source_str = signal.source.value if hasattr(signal.source, "value") else str(signal.source)
    source_lower = source_str.lower()

    is_generic, generic_reason = is_generic_intent_description(signal.description or "")
    if is_generic:
        logger.warning(f"Intent signal rejected: generic/templated — {generic_reason}")
        return 0.0, 5, "fabricated", None, -1
    if source_lower == "other" and len(signal.description or "") < 100:
        logger.warning("Intent signal rejected: 'other' source with short description")
        return 0.0, 10, "fabricated", None, -1
    future_err = check_future_date(signal.date)
    if future_err:
        logger.warning(f"Intent signal rejected: future date — {future_err}")
        return 0.0, 0, "fabricated", None, -1

    # ── Self-contradicting evidence guard ────────────────────────────────
    # The three-stage verifier's Stage 1 (Sonar with native web search)
    # makes an "approve" decision based on broad web searches for the
    # target signal — not the miner's specific URL. That means a miner can
    # submit a dead/closed-job URL ("Remote Jobs at Conversica · 0 Open
    # Positions", "The job you are looking for is no longer open") and
    # Stage 1 still approves because Sonar found OTHER references to job
    # postings on the company elsewhere. Observed 2026-05-19 on multiple
    # winning leads.
    #
    # Belt-and-suspenders: if the miner's own snippet/description text
    # contains an explicit negation phrase relative to the claim, reject
    # the signal regardless of the LLM verdict. The miner is literally
    # telling us the evidence URL doesn't support the claim.
    _NEGATION_PATTERNS = (
        r"\b0\s+(open|available|current|listed|active)\b",
        r"\bno\s+(open|current|active|listed|available)\s+(position|opening|job|hire|role)",
        r"\bno\s+longer\s+(open|available|accepting|listed|active)\b",
        r"\bnot\s+(currently|available|accepting|open|listed|hiring)\b",
        r"\bjob\s+(no\s+longer|is\s+(no\s+longer|not)\s+(open|available))",
        r"\b(page|posting|position)\s+(not\s+found|no\s+longer\s+exists|expired|removed)\b",
        r"\bunable\s+to\s+(verify|find|access|locate)\b",
        r"\bno\s+evidence\b",
        r"\b404\b",
    )
    _negation_re = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)
    _evidence_text = " ".join(filter(None, [
        signal.description or "", signal.snippet or "",
    ]))
    _neg_match = _negation_re.search(_evidence_text)
    if _neg_match:
        logger.warning(
            f"Intent signal rejected: miner's own description/snippet "
            f"contains a negation phrase ({_neg_match.group(0)!r}) — "
            f"evidence URL appears to NOT support the claim"
        )
        return 0.0, 0, "fabricated", None, -1

    # Get source type multiplier (penalize low-value sources like "other")
    source_multiplier = SOURCE_TYPE_MULTIPLIERS.get(source_lower, 0.5)

    # ── Intent verification — three-stage sonar → SD/Exa → sonar-pro ──
    # qualification/scoring/intent_verification_three_stage.py is the sole
    # intent verifier.  Pipeline:
    #   STAGE 1: perplexity/sonar with native web search verifies the claim.
    #            approve / reject -> STOP.
    #   STAGE 2: on review, SD (hardened) + Exa fallback fetches supplied URL.
    #   STAGE 3: perplexity/sonar-pro re-judges using the extracted content.
    # Production binary mapping: approve -> accept; reject/review -> reject
    # (flip with INTENT_VERIFIER_REVIEW_AS_ACCEPT=on for more recall).
    # Fail-closed: any unhandled exception rejects the signal.
    from qualification.scoring.intent_verification_three_stage import (
        verify_three_stage,
    )
    target_signal_raw = icp_signals_for_gate[miner_asserted_idx]
    target_signal_text = (
        target_signal_raw.get("text")
        if isinstance(target_signal_raw, dict)
        else str(target_signal_raw)
    )
    if product_service_context:
        target_signal_text = (
            f"{target_signal_text}\n"
            f"Product/service buying-fit context: the evidence should indicate "
            f"credible buying need or relevance for {product_service_context}."
        )
    # Pull spec.evidence_type off the matched ICP signal so the verifier's
    # prompt dispatcher routes to the per-type module (PART D for
    # SOCIAL_POSTING, PART E for TECHSTACK, PART F for
    # PODCAST_APPEARANCE).  None passes through and the verifier falls
    # back to the default builder — fail-open so signals whose
    # evidence_type couldn't be classified still get a generic substance
    # check rather than being silently dropped.
    #
    # PRIMARY source: ``icp.intent_signal_evidence_types`` (sibling list
    # indexed alongside ``intent_signals``).  Populated by
    # ``FulfillmentICP.to_icp_prompt`` from the structured spec.  This
    # exists because ``to_icp_prompt`` collapses ``intent_signals`` to a
    # plain ``List[str]`` for back-compat with the qualification LLM
    # prompt, so we can't reach back through the now-stringified entry
    # to find the structured ``evidence_type`` field.
    target_evidence_type = None
    icp_ets = getattr(icp, "intent_signal_evidence_types", None) or []
    if isinstance(icp_ets, list) and miner_asserted_idx < len(icp_ets):
        target_evidence_type = icp_ets[miner_asserted_idx]
    # Legacy fallback: if the sibling list isn't populated, attempt to
    # read from the raw entry — handles the rare case where lead_scorer
    # was called with a structured spec list directly (e.g. unit tests).
    if target_evidence_type is None:
        if isinstance(target_signal_raw, dict):
            target_evidence_type = target_signal_raw.get("evidence_type")
        else:
            target_evidence_type = getattr(
                target_signal_raw, "evidence_type", None,
            )
    import httpx
    try:
        async with httpx.AsyncClient() as http_client:
            three_stage_result = await verify_three_stage(
                http_client,
                company_name=company_name,
                company_linkedin=company_linkedin,
                company_website=company_website,
                source_url=signal.url,
                miner_claim=signal.description,
                target_signal_text=target_signal_text,
                miner_signal_date=(str(signal.date) if signal.date else None),
                evidence_type=target_evidence_type,
            )
    except Exception as three_stage_error:
        logger.error(
            "three-stage verifier raised: %s: %s — "
            "rejecting signal (no fallback)  source=%s",
            type(three_stage_error).__name__, three_stage_error,
            signal.url[:60],
        )
        return 0.0, confidence, "verified", content_found_date, -1

    s1_status = (three_stage_result.get("stage1") or {}).get("status")
    s3_status = (three_stage_result.get("stage3") or {}).get("status")
    scrape_summary = three_stage_result.get("scrape") or {}
    pipeline_decision = three_stage_result.get("decision")
    if not three_stage_result.get("client_ready"):
        logger.info(
            "Intent signal three-stage REJECT  reason=%s  "
            "decision=%s  s1_status=%s  s3_status=%s  "
            "scrape_results=%s  source=%s  target[%d]=%r",
            three_stage_result.get("rejection_reason"),
            pipeline_decision, s1_status, s3_status,
            scrape_summary.get("result_count"),
            signal.url[:60],
            miner_asserted_idx, target_signal_text[:60],
        )
        return 0.0, confidence, "verified", content_found_date, -1

    miner_date_match = (
        (three_stage_result.get("stage3") or {}).get("claim_matches_miner_date")
    )
    if miner_date_match == "contradicted":
        logger.info(
            "Intent signal three-stage REJECT  reason=miner_date_contradicted  "
            "miner_date=%s  source=%s",
            (str(signal.date) if signal.date else None), signal.url[:60],
        )
        return 0.0, confidence, "fabricated", content_found_date, -1

    logger.info(
        "Intent signal three-stage ACCEPT  decision=%s  "
        "s1_status=%s  s3_status=%s  miner_date_match=%s  scrape_results=%s  "
        "source=%s  target[%d]=%r",
        pipeline_decision, s1_status, s3_status, miner_date_match,
        scrape_summary.get("result_count"),
        signal.url[:60],
        miner_asserted_idx, target_signal_text[:60],
    )
    return (
        60.0 * source_multiplier,
        max(confidence, 90),
        "verified",
        content_found_date,
        miner_asserted_idx,
    )


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
# Batch Scoring + Summary  (lead-mode only — REMOVED May 2026)
# =============================================================================
#
# ``score_leads_batch`` (which orchestrated DB row equality verification
# via ``verify_leads_batch`` from ``qualification/scoring/db_verification.py``
# and then per-lead ``score_lead`` calls) and ``summarize_scores`` were
# part of the old leads-with-contacts pipeline.  Both have been removed
# in the company-mode cutover; the validator now loops over
# ``CompanyOutput`` instances directly and calls ``score_company`` per
# row (see ``neurons/validator.py::process_qualification_models``).
# Per-batch structural-similarity detection still lives in
# ``detect_structural_similarity`` above and is invoked by the
# validator before per-row scoring.
# =============================================================================
