"""
Fulfillment scoring pipeline: three-tier gate-then-score architecture.

Tier 1: ICP Fit Gate ($0 — free exact-match checks)
Tier 2: Data Accuracy Gate (Stage 0-2 + Stage 3 email + Stage 4 person + Stage 5 company)
Tier 3: Intent Scoring ($moderate — LLM calls, peak-weighted aggregation)
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from gateway.fulfillment.config import (
    get_fulfillment_api_key,
    FULFILLMENT_MIN_INTENT_SCORE,
    FULFILLMENT_INTENT_QUALITY_FLOOR,
    FULFILLMENT_INTENT_BREADTH_WEIGHT,
)
from gateway.fulfillment.models import (
    FulfillmentLead,
    FulfillmentICP,
    FulfillmentScoreResult,
    VALID_ROLE_TYPES,
)
from gateway.qualification.models import LeadOutput, ICPPrompt
from validator_models.fulfillment_person_verification import fulfillment_person_verification
from validator_models.fulfillment_company_verification import fulfillment_company_verification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak-weighted intent aggregation
# ---------------------------------------------------------------------------

def aggregate_intent_scores(signal_scores: List[float]) -> float:
    """
    Peak-weighted aggregation: best signal dominates, quality signals
    add diminishing breadth bonus, noise is ignored.
    """
    if not signal_scores:
        return 0.0

    sorted_desc = sorted(signal_scores, reverse=True)
    best = sorted_desc[0]

    bonus = 0.0
    for i, score in enumerate(sorted_desc[1:], start=1):
        if score < FULFILLMENT_INTENT_QUALITY_FLOOR:
            break
        bonus += score * FULFILLMENT_INTENT_BREADTH_WEIGHT * (1 / i)

    return min(best + bonus, 60.0)


# ---------------------------------------------------------------------------
# Employee count range helpers
# ---------------------------------------------------------------------------

def _parse_employee_range(val: str) -> Tuple[int, int]:
    """Parse employee count string into (min, max). Returns (0, 0) on failure."""
    if not val:
        return (0, 0)
    val = val.strip()
    if re.match(r"^\d+$", val):
        n = int(val)
        return (n, n)
    m = re.match(r"^(\d+)-(\d+)$", val)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d+)\+$", val)
    if m:
        return (int(m.group(1)), 10_000_000)
    return (0, 0)


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# ---------------------------------------------------------------------------
# Country normalization (reuses existing logic)
# ---------------------------------------------------------------------------

def _normalize_country(c: str) -> str:
    """Simple country alias normalization."""
    aliases = {
        "us": "united states", "usa": "united states", "u.s.": "united states",
        "u.s.a.": "united states", "uk": "united kingdom",
        "gb": "united kingdom", "great britain": "united kingdom",
    }
    c = c.strip().lower()
    return aliases.get(c, c)


# ---------------------------------------------------------------------------
# Fuzzy role matching for ICP Tier 1
# ---------------------------------------------------------------------------

_ROLE_TITLE_EQUIVALENTS = {
    "vp": ["vp", "vice president", "v.p."],
    "svp": ["svp", "senior vice president", "senior vp"],
    "evp": ["evp", "executive vice president"],
    "director": ["director", "dir"],
    "head": ["head", "head of"],
    "cro": ["cro", "chief revenue officer"],
    "coo": ["coo", "chief operating officer"],
    "cmo": ["cmo", "chief marketing officer"],
    "cto": ["cto", "chief technology officer"],
    "cfo": ["cfo", "chief financial officer"],
    "ceo": ["ceo", "chief executive officer"],
    "cio": ["cio", "chief information officer"],
    "manager": ["manager", "mgr"],
    "gm": ["gm", "general manager"],
    "md": ["md", "managing director"],
}

_ROLE_FUNCTION_EQUIVALENTS = {
    "sales": ["sales", "revenue", "commercial", "business development", "gtm", "go-to-market", "go to market"],
    "marketing": ["marketing", "growth", "demand generation", "brand"],
    "engineering": ["engineering", "software engineering", "development", "r&d"],
    "product": ["product", "product management"],
    "operations": ["operations", "ops"],
    "hr": ["hr", "human resources", "people", "talent"],
    "finance": ["finance", "financial"],
    "it": ["it", "information technology"],
    "customer success": ["customer success", "client success", "cx"],
    "partnerships": ["partnerships", "alliances", "channel"],
}


def _normalize_role_tokens(role: str) -> set:
    """Break a role into normalized tokens, expanding equivalents."""
    role_lower = role.lower().strip()

    # Handle slash/comma separated roles: "CRO / VP Sales" → ["cro", "vp", "sales"]
    role_lower = re.sub(r'[/,&]+', ' ', role_lower)
    role_lower = re.sub(r'\s+of\s+', ' ', role_lower)
    role_lower = re.sub(r'\s+', ' ', role_lower).strip()

    tokens = set(role_lower.split())

    # Expand title equivalents
    expanded = set()
    for token in tokens:
        for canonical, equivalents in _ROLE_TITLE_EQUIVALENTS.items():
            if token in equivalents:
                expanded.update(equivalents)
                break
        for canonical, equivalents in _ROLE_FUNCTION_EQUIVALENTS.items():
            if token in equivalents:
                expanded.update(equivalents)
                break

    return tokens | expanded


def _fuzzy_role_match(lead_role: str, target_roles: list) -> bool:
    """Check if lead_role is a fuzzy match for any target role.

    Handles: "CRO / VP Sales" matching "VP of Sales",
    "Director, Revenue" matching "Director of Sales",
    "Head of GTM" matching "Head of Revenue", etc.
    """
    if not lead_role or not target_roles:
        return False

    lead_tokens = _normalize_role_tokens(lead_role)

    for target in target_roles:
        target_tokens = _normalize_role_tokens(target)

        # Check overlap: if both roles share a title token AND a function token, it's a match
        lead_titles = set()
        lead_functions = set()
        target_titles = set()
        target_functions = set()

        for token in lead_tokens:
            for _, equivs in _ROLE_TITLE_EQUIVALENTS.items():
                if token in equivs:
                    lead_titles.add(token)
            for _, equivs in _ROLE_FUNCTION_EQUIVALENTS.items():
                if token in equivs:
                    lead_functions.add(token)

        for token in target_tokens:
            for _, equivs in _ROLE_TITLE_EQUIVALENTS.items():
                if token in equivs:
                    target_titles.add(token)
            for _, equivs in _ROLE_FUNCTION_EQUIVALENTS.items():
                if token in equivs:
                    target_functions.add(token)

        # Match if: shared title level AND shared function area
        title_overlap = bool(lead_titles & target_titles)
        function_overlap = bool(lead_functions & target_functions)

        if title_overlap and function_overlap:
            return True

        # Fallback: high token overlap (>= 50% of smaller set)
        overlap = lead_tokens & target_tokens
        min_size = min(len(lead_tokens), len(target_tokens))
        if min_size > 0 and len(overlap) / min_size >= 0.5:
            return True

    return False


# ---------------------------------------------------------------------------
# Tier 1: ICP Fit Gate (free, deterministic)
# ---------------------------------------------------------------------------

def _tier1_check(
    lead: FulfillmentLead,
    lead_output: LeadOutput,
    icp: FulfillmentICP,
    seen_companies: Set[str],
) -> Optional[str]:
    """
    Return failure_reason string if the lead fails any ICP check, else None.
    """
    # ``icp.industry`` and ``icp.sub_industry`` are lists of taxonomy values.
    # The Tier 1 gate matches a lead if its single industry/sub_industry
    # tag is present in the corresponding ICP list (set-membership), so a
    # client can target multiple verticals in one request (e.g.
    # restaurants + gyms + medspas) without splitting it into N separate
    # requests.  Empty list means "any industry accepted" — same semantics
    # as the legacy empty-string sentinel.
    #
    # Defensive ``isinstance(..., list)`` guards: an old icp_details row
    # that pre-dates the multi-value migration may still be a plain str
    # if any code path bypasses the model's ``mode="before"`` validator.
    if icp.industry:
        allowed_inds = icp.industry if isinstance(icp.industry, list) else [icp.industry]
        if lead.industry not in allowed_inds:
            return "industry_mismatch"

    if icp.sub_industry:
        allowed_subs = icp.sub_industry if isinstance(icp.sub_industry, list) else [icp.sub_industry]
        if lead.sub_industry not in allowed_subs:
            return "sub_industry_mismatch"

    # Excluded companies: leads whose company EXACTLY matches any entry
    # in the ICP's excluded_companies list (case-insensitive only, no
    # suffix stripping, no punctuation normalization) are hard-rejected
    # at Tier 1.  Populated by the gateway at create_request time from
    # the client's prior FULFILLED requests unless the client supplied
    # an explicit list in the create payload.
    #
    # Exact-match is intentional: the lead's `business` field gets
    # coerced to LinkedIn's canonical company name at Stage 5 of the
    # original submit flow, so two submissions for the same company
    # will have byte-identical `business` strings.  If the client
    # manually types "Microsoft" in their excluded list and LinkedIn
    # returns "Microsoft Corporation", those are treated as DIFFERENT
    # companies here by design — fuzzy suffix-stripping would be too
    # eager and could incorrectly block legitimate parent / subsidiary
    # distinctions (e.g. "Meta" vs "Meta Platforms, Inc.").
    if icp.excluded_companies and lead.business:
        excluded_keys = {c.strip().lower() for c in icp.excluded_companies if c and c.strip()}
        if lead.business.strip().lower() in excluded_keys:
            return "company_excluded"

    if icp.target_role_types and lead.role_type not in icp.target_role_types:
        return "role_type_mismatch"

    if icp.target_roles and lead.role not in icp.target_roles:
        if not _fuzzy_role_match(lead.role, icp.target_roles):
            return "role_mismatch"

    if icp.target_seniority:
        try:
            lead_sen = lead_output.seniority.value if hasattr(lead_output.seniority, "value") else str(lead_output.seniority)
            if lead_sen.lower() != icp.target_seniority.lower():
                return "seniority_mismatch"
        except Exception:
            return "seniority_mismatch"

    if icp.country and lead.company_hq_country:
        if _normalize_country(lead.company_hq_country) != _normalize_country(icp.country):
            return "country_mismatch"

    # Exact-bucket match: ``icp.employee_count`` is a list of canonical
    # buckets (e.g. ``["201-500", "501-1,000", "1,001-5,000"]``).  Miners
    # submit leads using the same canonical vocabulary (enforced in
    # gateway/api/submit.py), so this is a pure set-membership check.
    # Prior implementation used a range-overlap test which let leads
    # slip through when their bucket touched the ICP range at a single
    # endpoint (e.g. a ``"51-200"`` lead against a ``"200-5000"`` ICP
    # matched via the shared 200-employee boundary even though the
    # client excluded companies below 200 employees).
    if icp.employee_count and lead.employee_count:
        allowed = icp.employee_count if isinstance(icp.employee_count, list) else [icp.employee_count]
        if lead.employee_count not in allowed:
            return "employee_count_mismatch"

    if icp.company_stage and lead_output.role:
        pass

    biz_lower = lead_output.business.strip().lower()
    if not biz_lower:
        return "data_quality"
    if biz_lower in seen_companies:
        return "duplicate_company"
    seen_companies.add(biz_lower)

    return None


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

async def score_fulfillment_lead(
    lead: FulfillmentLead,
    icp: FulfillmentICP,
    seen_companies: Set[str],
    email_result: Optional[dict] = None,
    use_apify: bool = False,
) -> FulfillmentScoreResult:
    """Score a single fulfillment lead through the full verification + scoring pipeline.

    Pipeline order:
      Tier 1  – ICP fit (free, deterministic)
      Tier 2  – Stage 0-2 data accuracy (DNS, DNSBL, basic checks)
              – Apify Stage 4 (if use_apify=True, replaces ScrapingDog Stage 4)
              – Stage 3   email verification (TrueList result)
              – Stage 4   person verification (LinkedIn/GSE) — skipped if Apify verified
              – Stage 5   company verification + rep score
      Tier 3  – Intent scoring (LLM, peak-weighted aggregation)
    """
    lead_output = lead.to_lead_output()
    icp_prompt = icp.to_icp_prompt()

    # --- Tier 1: ICP Fit ---
    t1_failure = _tier1_check(lead, lead_output, icp, seen_companies)
    if t1_failure:
        return FulfillmentScoreResult(
            tier1_passed=False,
            failure_reason=t1_failure,
        )

    # Build a mutable dict that validator check functions can annotate in-place
    # (they add domain_age_days, has_mx, gse_search_count, etc.)
    validator_dict = lead.to_validator_dict()

    # --- Tier 2a: Stage 0-2 data accuracy ---
    t2_failure, stage0_2_data = await _run_fulfillment_stage0_2(validator_dict)
    if t2_failure:
        return FulfillmentScoreResult(
            tier1_passed=True, tier2_passed=False,
            failure_reason=t2_failure,
        )

    # --- Tier 2b: Verification ---
    # Fulfillment flow: Email → Company → Person → Rep score
    # Old flow (use_apify=False): run_stage4_5_repscore (email + S4 + S5 + rep)
    scrapingdog_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_KEY", "")

    if scrapingdog_key:
        # --- Email check (from batch result, no API call) ---
        email_verified = False
        if email_result:
            batch_status = email_result.get("status", "unknown")
            if batch_status == "email_ok":
                email_verified = True
            else:
                return FulfillmentScoreResult(
                    tier1_passed=True, tier2_passed=False,
                    failure_reason=f"email_{batch_status}",
                )
        else:
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=False,
                failure_reason="email_verification_unavailable",
            )

        # --- Company verification (always uses ScrapingDog LinkedIn) ---
        s5_passed, s5_rejection = await fulfillment_company_verification(
            validator_dict, scrapingdog_key, openrouter_key,
        )
        if not s5_passed:
            reason = s5_rejection.get("check_name", "fulfillment_company_failed") if s5_rejection else "fulfillment_company_failed"
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=True,
                email_verified=email_verified,
                failure_reason=reason,
            )
        company_verified = True

        # --- Person verification ---
        person_verified = False
        skip_stage4 = False
        apify_token = os.environ.get("APIFY_API_TOKEN", "")
        if use_apify and apify_token:
            # New: Q1 → SD → Apify
            passed, rejection_reason = await fulfillment_person_verification(
                validator_dict, apify_token, openrouter_key, scrapingdog_key,
            )
            if passed:
                person_verified = True
                skip_stage4 = True
            elif rejection_reason and rejection_reason.get("check_name") == "fulfillment_person_fetch_failed":
                # Apify API error — fall back to old Stage 4
                print("   ⚠️ Apify fetch failed, falling back to ScrapingDog Stage 4")
            else:
                reason = rejection_reason.get("check_name", "fulfillment_person_verification_failed") if rejection_reason else "fulfillment_person_verification_failed"
                return FulfillmentScoreResult(
                    tier1_passed=True, tier2_passed=True,
                    email_verified=email_verified,
                    company_verified=company_verified,
                    failure_reason=reason,
                )

        # If person not yet verified (Apify off or fetch failed), use old Stage 4
        if not person_verified:
            verif_failure, verif_data = await _run_verification_stages(
                validator_dict, email_result, stage0_2_data,
                skip_stage4=False, skip_stage5=True,
            )
            if verif_failure:
                return FulfillmentScoreResult(
                    tier1_passed=True, tier2_passed=True,
                    email_verified=email_verified,
                    company_verified=company_verified,
                    failure_reason=verif_failure,
                )
            person_verified = verif_data.get("stage_4_linkedin", {}).get("linkedin_verified", False)
            rep_score_val = float(verif_data.get("rep_score", {}).get("total_score", 0))
        else:
            # Person verified via Apify — still need rep score
            verif_failure, verif_data = await _run_verification_stages(
                validator_dict, email_result, stage0_2_data,
                skip_stage4=True, skip_stage5=True,
            )
            rep_score_val = float(verif_data.get("rep_score", {}).get("total_score", 0))

    else:
        # No ScrapingDog key — use old pipeline entirely
        verif_failure, verif_data = await _run_verification_stages(
            validator_dict, email_result, stage0_2_data,
        )
        email_verified = verif_data.get("stage_3_email", {}).get("email_status") == "valid"
        person_verified = verif_data.get("stage_4_linkedin", {}).get("linkedin_verified", False)
        stage5 = verif_data.get("stage_5_verification", {})
        company_verified = stage5.get("company_name_verified", False)
        rep_score_val = float(verif_data.get("rep_score", {}).get("total_score", 0))

        if verif_failure:
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=True,
                email_verified=email_verified,
                person_verified=person_verified,
                company_verified=company_verified,
                rep_score=rep_score_val,
                failure_reason=verif_failure,
            )

    # --- Tier 3: Intent Scoring ---
    api_key = get_fulfillment_api_key()
    from qualification.scoring.lead_scorer import (
        _score_single_intent_signal,
        _apply_signal_time_decay,
        _extract_domain,
    )

    icp_criteria = None
    seen_domains: Set[str] = set()
    signal_results = []
    # Per-signal breakdown surfaced up to the score result so the gateway can
    # persist the miner-signal -> ICP-signal mapping alongside the aggregate
    # intent score. One dict per input miner signal, same order.
    signal_details: List[dict] = []
    icp_intent_signals_list = list(icp_prompt.intent_signals or [])

    lead_label = f"{lead.full_name} @ {lead_output.business}"
    print(f"🔍 Tier 3 Intent Scoring: {lead_label} — {len(lead.intent_signals)} signal(s)")

    for idx, signal in enumerate(lead.intent_signals):
        source_str = signal.source.value if hasattr(signal.source, "value") else str(signal.source)
        print(f"   Signal {idx+1}: source={source_str}, url={signal.url[:80]}, "
              f"date={signal.date}, desc={signal.description[:100]}")

        domain = _extract_domain(signal.url)
        if domain in seen_domains:
            print(f"   ⏭️  Duplicate domain '{domain}' — skipping")
            signal_results.append({"after_decay": 0.0, "decay_mult": 1.0, "confidence": 0})
            signal_details.append({
                "url": signal.url,
                "description": signal.description,
                "snippet": signal.snippet,
                "date": str(signal.date) if signal.date else None,
                "source": source_str,
                "raw_score": 0.0,
                "after_decay_score": 0.0,
                "decay_multiplier": 1.0,
                "confidence": 0,
                "date_status": "duplicate_domain",
                "matched_icp_signal_idx": -1,
                "matched_icp_signal": None,
            })
            continue
        seen_domains.add(domain)

        matched_idx = -1
        try:
            (
                score, confidence, date_status,
                content_found_date, matched_idx,
            ) = await _score_single_intent_signal(
                signal, icp_prompt, icp_criteria,
                lead_output.business, lead_output.company_website,
                api_key=api_key,
            )
            print(f"   📊 Raw score={score:.1f}, confidence={confidence}, "
                  f"date_status={date_status}, content_date={content_found_date}, "
                  f"matched_icp_idx={matched_idx}")
        except Exception as e:
            print(f"   ❌ Signal scoring error: {e}")
            score, confidence, date_status, content_found_date = 0.0, 0, "fabricated", None

        after_decay, decay_mult = _apply_signal_time_decay(
            score, signal.date, date_status, source_str, content_found_date
        )
        print(f"   📉 After decay={after_decay:.1f}, decay_mult={decay_mult:.2f}")

        signal_results.append({
            "after_decay": after_decay,
            "decay_mult": decay_mult,
            "confidence": confidence,
        })
        matched_str = (
            icp_intent_signals_list[matched_idx]
            if 0 <= matched_idx < len(icp_intent_signals_list)
            else None
        )
        signal_details.append({
            "url": signal.url,
            "description": signal.description,
            "snippet": signal.snippet,
            "date": str(signal.date) if signal.date else None,
            "source": source_str,
            "raw_score": float(score),
            "after_decay_score": float(after_decay),
            "decay_multiplier": float(decay_mult),
            "confidence": int(confidence),
            "date_status": date_status,
            "matched_icp_signal_idx": int(matched_idx),
            "matched_icp_signal": matched_str,
        })

    after_decay_scores = [r["after_decay"] for r in signal_results]
    intent_signal_final = aggregate_intent_scores(after_decay_scores)
    intent_signal_final = min(intent_signal_final, 60.0)

    print(f"   🎯 Intent final={intent_signal_final:.1f} (threshold={FULFILLMENT_MIN_INTENT_SCORE}), "
          f"decay_scores={[f'{s:.1f}' for s in after_decay_scores]}")

    all_fabricated = bool(signal_results) and all(r["confidence"] == 0 for r in signal_results)
    if all_fabricated:
        print(f"   ⚠️  All signals have confidence=0 (all_fabricated=True)")

    shared_fields = dict(
        tier1_passed=True,
        tier2_passed=True,
        email_verified=email_verified,
        person_verified=person_verified,
        company_verified=company_verified,
        rep_score=rep_score_val,
        intent_signal_raw=max(after_decay_scores) if after_decay_scores else 0.0,
        intent_signal_final=intent_signal_final,
        intent_decay_multiplier=_avg([r["decay_mult"] for r in signal_results]),
        all_fabricated=all_fabricated,
        intent_signals_detail=signal_details,
    )

    if intent_signal_final < FULFILLMENT_MIN_INTENT_SCORE:
        return FulfillmentScoreResult(
            **shared_fields,
            final_score=0.0,
            failure_reason="insufficient_intent",
        )

    return FulfillmentScoreResult(
        **shared_fields,
        final_score=intent_signal_final,
    )


async def score_fulfillment_batch(
    leads: List[FulfillmentLead],
    icp: FulfillmentICP,
    use_apify: bool = False,
) -> List[FulfillmentScoreResult]:
    """Score a batch of fulfillment leads with batch email verification."""
    seen_companies: Set[str] = set()

    # Run batch email verification up-front so individual leads get results
    email_results_map = await _run_batch_email_verification(leads)

    results: List[FulfillmentScoreResult] = []
    for lead in leads:
        per_lead = email_results_map.get(lead.email.lower())
        result = await score_fulfillment_lead(
            lead, icp, seen_companies, email_result=per_lead,
            use_apify=use_apify,
        )
        results.append(result)

    # Structural similarity detection on leads that passed all tiers
    try:
        from qualification.scoring.lead_scorer import detect_structural_similarity
        passing_outputs = []
        passing_indices = []
        for i, (lead, result) in enumerate(zip(leads, results)):
            if result.final_score > 0:
                passing_outputs.append(lead.to_lead_output())
                passing_indices.append(i)

        if len(passing_outputs) >= 2:
            flagged = detect_structural_similarity(passing_outputs)
            for local_idx in flagged:
                global_idx = passing_indices[local_idx]
                results[global_idx] = FulfillmentScoreResult(
                    **{
                        **results[global_idx].model_dump(),
                        "final_score": 0.0,
                        "failure_reason": "structural_similarity_detected",
                    }
                )
    except Exception as e:
        logger.warning(f"Structural similarity detection error: {e}")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_batch_email_verification(
    leads: List[FulfillmentLead],
) -> Dict[str, dict]:
    """Verify all emails in the batch via TrueList inline API.

    Returns a mapping of ``email (lowercase) -> result dict``.
    Each result dict contains at minimum ``status``, ``passed``, and
    ``rejection_reason`` keys compatible with ``run_stage4_5_repscore``.

    On failure the returned dict is empty, causing each lead to
    individually fail with ``email_verification_unavailable`` rather
    than crashing the entire validator scoring loop.
    """
    emails = list({lead.email.lower() for lead in leads if lead.email})
    if not emails:
        return {}

    try:
        from validator_models.checks_email import verify_emails_inline
        return await verify_emails_inline(emails)
    except Exception as e:
        logger.error(f"Batch email verification failed: {e}")
        return {}


async def _run_fulfillment_stage0_2(
    validator_dict: dict,
) -> Tuple[Optional[str], Optional[dict]]:
    """Run Stage 0, 1, 2 checks adapted for fulfillment leads.

    Identical to the validator pipeline's ``run_stage0_2_checks`` except
    source-provenance checks are skipped (fulfillment leads are submitted
    directly by miners, not scraped from a tracked source URL).

    ``validator_dict`` is mutated in-place by check functions — they add
    fields like ``domain_age_days``, ``has_mx``, etc. that Stage 4-5 and
    the rep-score pipeline read later.

    Returns ``(failure_reason, stage0_2_data)``.  ``failure_reason`` is
    ``None`` when all checks pass.
    """
    from validator_models.automated_checks import (
        check_required_fields, check_email_regex, check_name_email_match,
        check_general_purpose_email, check_free_email_domain, check_disposable,
        MAX_REP_SCORE,
    )
    from validator_models.checks_email import (
        check_domain_age, check_mx_record, check_spf_dmarc,
        check_head_request, check_dnsbl,
    )

    stage0_2_data: dict = {
        "stage_0_hardcoded": {
            "name_in_email": False,
            "is_general_purpose_email": False,
        },
        "stage_1_dns": {
            "has_mx": False, "has_spf": False,
            "has_dmarc": False, "dmarc_policy": None,
        },
        "stage_2_domain": {
            "dnsbl_checked": False, "dnsbl_blacklisted": False,
            "dnsbl_list": None, "domain_age_days": None,
            "domain_registrar": None, "domain_nameservers": None,
            "whois_updated_days_ago": None,
        },
        "stage_3_email": {
            "email_status": "unknown", "email_score": 0,
            "is_disposable": False, "is_role_based": False, "is_free": False,
        },
        "stage_4_linkedin": {
            "linkedin_verified": False, "gse_search_count": 0,
            "llm_confidence": "none",
        },
        "stage_5_verification": {
            "role_verified": False, "region_verified": False,
            "industry_verified": False, "extracted_role": None,
            "extracted_region": None, "extracted_industry": None,
            "early_exit": None,
        },
        "rep_score": {
            "total_score": 0, "max_score": MAX_REP_SCORE,
            "breakdown": {
                "wayback_machine": 0, "uspto_trademarks": 0,
                "sec_edgar": 0, "whois_dnsbl": 0,
                "gdelt": 0, "companies_house": 0,
            },
        },
        "passed": False,
        "rejection_reason": None,
    }

    def _collect_dns_data() -> None:
        stage0_2_data["stage_1_dns"]["has_mx"] = validator_dict.get("has_mx", False)
        stage0_2_data["stage_1_dns"]["has_spf"] = validator_dict.get("has_spf", False)
        stage0_2_data["stage_1_dns"]["has_dmarc"] = validator_dict.get("has_dmarc", False)
        stage0_2_data["stage_1_dns"]["dmarc_policy"] = (
            "strict" if validator_dict.get("dmarc_policy_strict") else "none"
        )
        stage0_2_data["stage_2_domain"]["domain_age_days"] = validator_dict.get("domain_age_days")
        stage0_2_data["stage_2_domain"]["domain_registrar"] = validator_dict.get("domain_registrar")
        stage0_2_data["stage_2_domain"]["domain_nameservers"] = validator_dict.get("domain_nameservers")
        stage0_2_data["stage_2_domain"]["whois_updated_days_ago"] = validator_dict.get("whois_updated_days_ago")

    def _fail(rejection: Optional[dict]) -> Tuple[str, dict]:
        stage0_2_data["passed"] = False
        stage0_2_data["rejection_reason"] = rejection
        rej = rejection or {}
        return rej.get("check_name", "stage0_2_failure"), stage0_2_data

    # -- Stage 0: instant checks --
    for check_func in [
        check_required_fields, check_email_regex, check_name_email_match,
        check_general_purpose_email, check_free_email_domain, check_disposable,
    ]:
        passed, rejection = await check_func(validator_dict)
        if not passed:
            return _fail(rejection)

    stage0_2_data["stage_0_hardcoded"]["name_in_email"] = True
    stage0_2_data["stage_0_hardcoded"]["is_general_purpose_email"] = False

    # -- Stage 0 (continued): HEAD request runs in background --
    head_task = asyncio.create_task(check_head_request(validator_dict))

    # -- Stage 1: DNS checks in parallel --
    # check_domain_age still RUNS (to populate WHOIS metadata used by the
    # rep-score check downstream) but its "<7 days old → reject" gate no
    # longer rejects fulfillment leads.  Stage 5 website confirmation
    # already verifies the miner's domain appears on the company's real
    # LinkedIn page, which is a stronger signal than raw registration
    # age.  A legitimately new startup (< 7 days old but already on
    # LinkedIn) used to be wrongly rejected here; ~7% of rejected leads
    # hit this path on 2026-04-21.  check_mx_record and check_spf_dmarc
    # stay hard-gating — a domain with no MX record genuinely can't
    # receive mail.
    CHECK_INDEX = {0: "check_domain_age", 1: "check_mx_record", 2: "check_spf_dmarc"}
    DOMAIN_AGE_INDEX = 0

    dns_results = await asyncio.gather(
        check_domain_age(validator_dict),
        check_mx_record(validator_dict),
        check_spf_dmarc(validator_dict),
        return_exceptions=True,
    )

    for idx, result in enumerate(dns_results):
        is_domain_age = (idx == DOMAIN_AGE_INDEX)

        if isinstance(result, Exception):
            if is_domain_age:
                # WHOIS flakiness shouldn't tank the lead — rep score
                # will reflect reduced confidence automatically.
                print(f"   ℹ️  check_domain_age errored (non-fatal, bypassed): {result}")
                continue
            _collect_dns_data()
            head_task.cancel()
            return _fail({
                "stage": "Stage 1: DNS Layer",
                "check_name": CHECK_INDEX.get(idx, "stage1_dns_failure"),
                "message": str(result),
                "failed_fields": ["domain"],
            })

        passed, rejection = result
        if not passed:
            if is_domain_age:
                msg = (rejection or {}).get("message", "domain too young")
                print(f"   ℹ️  check_domain_age failed but bypassed in fulfillment: {msg}")
                continue
            _collect_dns_data()
            head_task.cancel()
            return _fail(rejection)

    _collect_dns_data()

    # -- Stage 0 HEAD result --
    passed, rejection = await head_task
    if not passed:
        return _fail(rejection)

    # -- Stage 2: DNSBL --
    passed, rejection = await check_dnsbl(validator_dict)

    stage0_2_data["stage_2_domain"]["dnsbl_checked"] = validator_dict.get("dnsbl_checked", False)
    stage0_2_data["stage_2_domain"]["dnsbl_blacklisted"] = validator_dict.get("dnsbl_blacklisted", False)
    stage0_2_data["stage_2_domain"]["dnsbl_list"] = validator_dict.get("dnsbl_list")

    if not passed:
        return _fail(rejection)

    stage0_2_data["passed"] = True
    return None, stage0_2_data


async def _run_verification_stages(
    validator_dict: dict,
    email_result: Optional[dict],
    stage0_2_data: Optional[dict],
    skip_stage4: bool = False,
    skip_stage5: bool = False,
) -> Tuple[Optional[str], dict]:
    """Run Stage 3 (email) + Stage 4 (person) + Stage 5 (company) + rep score.

    Delegates to the validator pipeline's ``run_stage4_5_repscore`` which
    expects pre-computed TrueList email results and the Stage 0-2 data dict.

    When *skip_stage4* is True (Apify already verified the person), Stage 4
    inside ``run_stage4_5_repscore`` is skipped.  When *skip_stage5* is True
    (fulfillment Stage 5 already verified the company), Stage 5 is skipped.

    Returns ``(failure_reason, verification_data)``.
    """
    from validator_models.automated_checks import run_stage4_5_repscore

    if not email_result:
        return "email_verification_unavailable", stage0_2_data or {}

    if not stage0_2_data:
        return "stage0_2_data_missing", {}

    passed, full_data = await run_stage4_5_repscore(
        validator_dict, email_result, stage0_2_data,
        skip_stage4=skip_stage4, skip_stage5=skip_stage5,
    )
    if not passed:
        rej = full_data.get("rejection_reason") or {}
        reason = rej.get("check_name", "verification_failure")
        return reason, full_data

    return None, full_data


def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0
