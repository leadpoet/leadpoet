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
from gateway.fulfillment.icp_checks import (
    tier1_check,
    semantic_sub_industry_match,
    validate_lead_geography,
)
from validator_models.fulfillment_person_verification import fulfillment_person_verification
from validator_models.fulfillment_company_verification import fulfillment_company_verification
from validator_models.fulfillment_attribute_verification import verify_required_attributes
from validator_models.checks_zerobounce import (
    zerobounce_validate,
    is_truelist_catch_all,
)

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


def _build_failure_detail(
    failure_reason: str,
    lead: "FulfillmentLead" = None,
    lead_output: "LeadOutput" = None,
    icp: "FulfillmentICP" = None,
    s5_rejection: dict = None,
    person_rejection: dict = None,
    signal_details: list = None,
    intent_final: float = 0.0,
    missing_required_signals: list = None,
    attribute_rejection: dict = None,
) -> str:
    """Build a human-readable failure detail for the public dashboard.

    Shows miner's submitted values (non-PII) and generic reason.
    Never exposes: email, person name, company name, website, LinkedIn URL, ICP prompt.
    """
    r = failure_reason or ""

    # --- Tier 1 ---
    if r == "industry_mismatch" and lead:
        return f"Submitted industry '{lead.industry}' does not match target industry"
    if r == "sub_industry_mismatch" and lead:
        return f"Submitted sub-industry '{lead.sub_industry}' does not match target sub-industry"
    if r == "role_mismatch" and lead:
        return f"Submitted role '{lead.role}' does not match target roles"
    if r == "role_type_mismatch" and lead:
        return f"Submitted role type '{lead.role_type}' does not match target role types"
    if r == "seniority_mismatch" and lead_output:
        sen = lead_output.seniority.value if hasattr(lead_output.seniority, "value") else str(lead_output.seniority)
        return f"Submitted seniority '{sen}' does not match target seniority"
    if r == "country_mismatch" and lead:
        if icp and icp.country:
            targets = icp.country if isinstance(icp.country, list) else [icp.country]
            target_str = ", ".join(str(c) for c in targets if c)
            if target_str:
                label = "country" if len(targets) == 1 else "countries"
                return (
                    f"Submitted country '{lead.company_hq_country}' does not "
                    f"match target {label}: {target_str}"
                )
        return f"Submitted country '{lead.company_hq_country}' does not match target country"
    if r == "employee_count_mismatch" and lead:
        return f"Submitted employee count '{lead.employee_count}' does not match target range"
    if r == "company_excluded":
        return "Company is on the exclusion list for this request"
    if r == "duplicate_company":
        return "Another lead for the same company was already submitted in this batch"
    if r == "data_quality":
        return "Missing required field: company name"
    if r == "invalid_hq_location" and lead:
        return f"Submitted HQ '{lead.company_hq_city}, {lead.company_hq_state}, {lead.company_hq_country}' is not a valid location"
    if r == "geography_mismatch" and lead:
        return f"Submitted HQ '{lead.company_hq_city}, {lead.company_hq_state}, {lead.company_hq_country}' is not within target geography"
    if r == "geography_missing":
        return "No HQ location submitted but ICP requires specific geography"

    # --- Email ---
    if r.startswith("email_"):
        if "no_mailbox" in r:
            return "Email mailbox does not exist"
        if "greylisted" in r or "failed_greylisted" in r:
            return "Email server temporarily rejected verification"
        if "catch" in r or "accept_all" in r:
            return "Email domain accepts all addresses (catch-all) — cannot verify individual mailbox"
        if r == "email_verification_unavailable":
            return "Email verification service unavailable"
        return f"Email verification failed ({r.replace('email_', '')})"

    # --- Company verification ---
    if r.startswith("fulfillment_company_"):
        if s5_rejection:
            msg = s5_rejection.get("message", "")
            check = s5_rejection.get("check_name", "")
            if check == "fulfillment_company_size_mismatch":
                return msg if msg else "Submitted employee count does not match LinkedIn verified size"
            if check == "fulfillment_company_hq_mismatch":
                return msg if msg else "Submitted HQ location does not match LinkedIn HQ"
            if check == "fulfillment_company_website_mismatch":
                return "Submitted website does not match LinkedIn website"
            if check == "fulfillment_company_description_invalid":
                return "Submitted description does not match what the company actually does"
            if check == "fulfillment_company_industry_classification_mismatch":
                return msg if msg else "Company's verified industry does not match target industry"
            if check == "fulfillment_company_enrich_failed":
                return "Could not verify company details via web search"
            if check == "fulfillment_company_no_description":
                return "No company description available for verification"
            if check == "fulfillment_company_validation_failed":
                return "Company verification failed due to processing error"
        return "Company verification failed"

    # --- Person verification ---
    if r.startswith("fulfillment_person_"):
        if person_rejection:
            check = person_rejection.get("check_name", "")
            if check == "fulfillment_person_role_mismatch":
                submitted = person_rejection.get("claimed_role", "")
                actual = person_rejection.get("actual_role", "")
                if submitted and actual:
                    return f"Submitted role '{submitted}' does not match LinkedIn role '{actual}'"
                return "Submitted role does not match LinkedIn profile role"
            if check == "fulfillment_person_company_name_mismatch":
                return "Person's LinkedIn company does not match submitted company"
            if check == "fulfillment_person_company_url_mismatch":
                return "Person's LinkedIn company URL does not match submitted company"
            if check == "fulfillment_person_no_company_url":
                return "Person's LinkedIn profile has no company URL to verify"
            if check == "fulfillment_person_fetch_failed":
                return "Could not fetch person's LinkedIn profile for verification"
        return "Person verification failed"

    # --- Intent ---
    if r == "insufficient_intent":
        parts = [f"Intent score {intent_final:.1f} below threshold 5.0"]
        if signal_details:
            for i, sd in enumerate(signal_details):
                src = sd.get("source", "unknown")
                conf = sd.get("confidence", 0)
                date_status = sd.get("date_status", "")
                score = sd.get("after_decay_score", 0)
                if conf == 0 or score == 0:
                    if date_status == "fabricated":
                        parts.append(f"Signal {i+1} ({src}): content not verified against source")
                    elif date_status == "duplicate_domain":
                        parts.append(f"Signal {i+1} ({src}): duplicate source domain")
                    else:
                        parts.append(f"Signal {i+1} ({src}): could not verify (confidence={conf})")
        return " — ".join(parts)

    # Lead failed because a buyer-side intent_signal flagged
    # ``required=True`` was not satisfied by any verified miner signal.
    # We name the missing spec text(s) so miners and operators can see
    # exactly which requirement(s) the lead missed without leaking
    # other ICP context.
    if r == "missing_required_intent_signal":
        missing = missing_required_signals or []
        if missing:
            quoted = ", ".join(f'"{t}"' for t in missing)
            return (
                f"Lead missing required intent signal(s): {quoted}. "
                f"Required signals must have at least one miner-supplied "
                f"piece of evidence that the LLM matches to them AND "
                f"that passes URL verification."
            )
        return "Lead missing required intent signal(s)"

    # --- Required-attribute (Tier 2c) ---
    if r == "required_attribute_failed":
        if attribute_rejection:
            msg = attribute_rejection.get("message", "")
            if msg:
                return msg
        return "Lead failed one or more required-attribute checks"
    if r == "attribute_verification_no_api_key":
        return "Required-attribute verification unavailable (API key missing)"

    # --- Structural similarity ---
    if r == "structural_similarity_detected":
        return "Multiple leads have identical templated descriptions"

    # --- Stage 0-2 ---
    if r == "check_head_request":
        return "Company website URL is not reachable"
    if r.startswith("check_"):
        return f"Data accuracy check failed: {r.replace('check_', '').replace('_', ' ')}"

    # Fallback
    return r.replace("_", " ").replace("fulfillment ", "")


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

    # --- Tier 1: ICP Fit (deterministic, free) ---
    t1_failure = tier1_check(lead, lead_output, icp, seen_companies)
    if t1_failure and t1_failure != "sub_industry_needs_llm":
        return FulfillmentScoreResult(
            tier1_passed=False,
            failure_reason=t1_failure,
            failure_detail=_build_failure_detail(t1_failure, lead=lead, lead_output=lead_output, icp=icp),
        )

    # --- Tier 1.5: LLM-based checks (sub-industry semantic + location) ---
    # Call 1: Sub-industry semantic match (only if Tier 1 exact+containment failed)
    if t1_failure == "sub_industry_needs_llm":
        icp_subs = icp.sub_industry if isinstance(icp.sub_industry, list) else [icp.sub_industry]
        sub_matched, matched_sub = await semantic_sub_industry_match(
            lead.industry, lead.sub_industry, icp_subs,
        )
        if not sub_matched:
            return FulfillmentScoreResult(
                tier1_passed=False,
                failure_reason="sub_industry_mismatch",
                failure_detail=_build_failure_detail("sub_industry_mismatch", lead=lead, lead_output=lead_output, icp=icp),
            )
        print(f"   ✅ Tier 1.5: Sub-industry semantic match: '{lead.sub_industry}' → '{matched_sub}'")

    # Call 2: Location validation + geography match (when ICP has specific geography)
    # Skip when geography is just naming a country already in icp.country
    # (Tier 1 country gate handles that). icp.country is List[str] post the
    # multi-country change.
    icp_geo = (icp.geography or "").strip()
    _countries = icp.country if isinstance(icp.country, list) else ([icp.country] if icp.country else [])
    icp_countries_norm = {(c or "").strip().lower() for c in _countries if c}
    if icp_geo and icp_geo.lower() not in icp_countries_norm:
        # Pre-check: if miner submitted no location at all, reject before LLM
        if not lead.company_hq_city and not lead.company_hq_state and not lead.company_hq_country:
            return FulfillmentScoreResult(
                tier1_passed=False,
                failure_reason="geography_missing",
                failure_detail="No HQ location submitted but ICP requires specific geography",
            )
        loc_valid, geo_match = await validate_lead_geography(
            lead.company_hq_city, lead.company_hq_state, lead.company_hq_country,
            icp.geography,
        )
        if not loc_valid:
            return FulfillmentScoreResult(
                tier1_passed=False,
                failure_reason="invalid_hq_location",
                failure_detail=f"Submitted HQ '{lead.company_hq_city}, {lead.company_hq_state}, {lead.company_hq_country}' is not a valid location",
            )
        if not geo_match:
            return FulfillmentScoreResult(
                tier1_passed=False,
                failure_reason="geography_mismatch",
                failure_detail=f"Submitted HQ '{lead.company_hq_city}, {lead.company_hq_state}, {lead.company_hq_country}' is not within target geography",
            )
        print(f"   ✅ Tier 1.5: Geography match: '{lead.company_hq_city}, {lead.company_hq_state}' within '{icp.geography}'")

    # Build a mutable dict that validator check functions can annotate in-place
    # (they add domain_age_days, has_mx, gse_search_count, etc.)
    validator_dict = lead.to_validator_dict()

    # --- Tier 2a: Stage 0-2 data accuracy ---
    t2_failure, stage0_2_data = await _run_fulfillment_stage0_2(validator_dict)
    if t2_failure:
        return FulfillmentScoreResult(
            tier1_passed=True, tier2_passed=False,
            failure_reason=t2_failure,
            failure_detail=_build_failure_detail(t2_failure),
        )

    # --- Tier 2b: Verification ---
    # Fulfillment flow: Email → Company → Person → Rep score
    # Old flow (use_apify=False): run_stage4_5_repscore (email + S4 + S5 + rep)
    scrapingdog_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_KEY", "")

    if scrapingdog_key:
        # --- Email check (from batch result, no API call) ---
        # Path: TrueList batch email_ok               → pass
        #       TrueList batch accept_all/ok_for_all  → ZeroBounce fallback
        #         ZeroBounce status == "valid"        → pass (catch-all bypassed)
        #         anything else                        → keep TrueList reject
        #       any other TrueList status              → reject as before
        email_verified = False
        if email_result:
            batch_status = email_result.get("status", "unknown")
            if batch_status == "email_ok":
                email_verified = True
            elif is_truelist_catch_all(batch_status):
                zb_result = await zerobounce_validate(lead.email)
                zb_status = zb_result.get("status", "error")
                zb_sub = zb_result.get("sub_status", "")
                if zb_result.get("valid"):
                    print(
                        f"   📧 Email catch-all ({batch_status}) — ZeroBounce confirmed valid "
                        f"(sub_status={zb_sub or 'n/a'}); bypassing TrueList catch-all reject"
                    )
                    email_verified = True
                else:
                    print(
                        f"   📧 Email catch-all ({batch_status}) — ZeroBounce did not confirm "
                        f"(zb_status={zb_status}, sub={zb_sub or 'n/a'}); keeping reject"
                    )
                    _email_reason = f"email_{batch_status}_zb_{zb_status}"
                    return FulfillmentScoreResult(
                        tier1_passed=True, tier2_passed=False,
                        failure_reason=_email_reason,
                        failure_detail=_build_failure_detail(_email_reason),
                    )
            else:
                _email_reason = f"email_{batch_status}"
                return FulfillmentScoreResult(
                    tier1_passed=True, tier2_passed=False,
                    failure_reason=_email_reason,
                    failure_detail=_build_failure_detail(_email_reason),
                )
        else:
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=False,
                failure_reason="email_verification_unavailable",
                failure_detail=_build_failure_detail("email_verification_unavailable"),
            )

        # --- Company verification (always uses ScrapingDog LinkedIn) ---
        s5_passed, s5_rejection = await fulfillment_company_verification(
            validator_dict, scrapingdog_key, openrouter_key,
            icp_prompt=icp.prompt,
            icp_product_service=icp.product_service,
            icp_industry=icp.industry or [],
            icp_sub_industry=icp.sub_industry or [],
        )
        if not s5_passed:
            reason = s5_rejection.get("check_name", "fulfillment_company_failed") if s5_rejection else "fulfillment_company_failed"
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=True,
                email_verified=email_verified,
                failure_reason=reason,
                failure_detail=_build_failure_detail(reason, s5_rejection=s5_rejection),
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
                    failure_detail=_build_failure_detail(reason, person_rejection=rejection_reason),
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
                    failure_detail=_build_failure_detail(verif_failure),
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
                failure_detail=_build_failure_detail(verif_failure),
            )

    # --- Tier 2c: Required-Attribute Verification (Sonar) ---
    # Runs only when the buyer's ICP defines required_attributes. Each attribute
    # is checked in parallel via Perplexity Sonar. Negative attributes
    # ("Does not have X", "No Y in place") route through the positive-proxy
    # path (search for the positive opposite). Fail-closed: any NO → REJECT.
    # Contact-side prompts are enriched with the Apify-extracted LinkedIn data
    # stashed on the lead dict by fulfillment_person_verification.
    attribute_verification_result: Optional[dict] = None
    if icp.required_attributes and (
        icp.required_attributes.get("company") or icp.required_attributes.get("contact")
    ):
        apify_person_data = validator_dict.get("_apify_data")
        print(
            f"🔍 Tier 2c: Attribute verification — "
            f"{len(icp.required_attributes.get('company', []))} company / "
            f"{len(icp.required_attributes.get('contact', []))} contact attribute(s); "
            f"apify_data={'present' if apify_person_data else 'absent'}"
        )
        attr_passed, attribute_verification_result = await verify_required_attributes(
            lead=validator_dict,
            required_attributes=icp.required_attributes,
            apify_person_data=apify_person_data,
            openrouter_key=openrouter_key,
        )
        decision = attribute_verification_result.get("decision", "REJECT")
        counts = attribute_verification_result.get("counts", {})
        print(
            f"   📋 Tier 2c result: {decision} "
            f"(YES={counts.get('yes', 0)}, NO={counts.get('no', 0)}, "
            f"DEFERRED={counts.get('deferred', 0)}, "
            f"elapsed={attribute_verification_result.get('elapsed_s', 0):.1f}s)"
        )
        if not attr_passed:
            rejection = attribute_verification_result.get("rejection_reason") or {}
            reason = rejection.get("check_name", "required_attribute_failed")
            return FulfillmentScoreResult(
                tier1_passed=True, tier2_passed=True,
                email_verified=email_verified,
                person_verified=person_verified,
                company_verified=company_verified,
                attribute_verification=attribute_verification_result,
                rep_score=rep_score_val,
                failure_reason=reason,
                failure_detail=_build_failure_detail(
                    reason,
                    attribute_rejection=rejection,
                ),
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
    # Structured buyer-side specs (text + required). Indexed by position;
    # ``matched_icp_signal_idx`` from the LLM points into this list. The
    # shared lead_scorer LLM prompt receives the same list collapsed to
    # ``List[str]`` via icp_prompt.intent_signals, so both views are
    # positionally identical.
    icp_signal_specs = list(icp.intent_signals or [])
    icp_intent_signals_list = [s.text for s in icp_signal_specs]

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
                "matched_icp_signal_required": None,
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
                # company_linkedin only used by v2 verifier path (flag-gated
                # by INTENT_VERIFIER_V2 inside _score_single_intent_signal);
                # legacy path ignores it.
                company_linkedin=getattr(lead_output, "company_linkedin", "") or "",
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

        # Pull the matched spec's required flag so downstream surfaces
        # (CSV, winning-leads UI) know whether this miner signal satisfies
        # a buyer hard requirement.
        matched_str = None
        matched_required: Optional[bool] = None
        if 0 <= matched_idx < len(icp_signal_specs):
            spec = icp_signal_specs[matched_idx]
            matched_str = spec.text
            matched_required = spec.required

        signal_results.append({
            "after_decay": after_decay,
            "decay_mult": decay_mult,
            "confidence": confidence,
            "matched_icp_signal_idx": int(matched_idx),
        })
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
            "matched_icp_signal_required": matched_required,
        })

    after_decay_scores = [r["after_decay"] for r in signal_results]
    intent_signal_final = aggregate_intent_scores(after_decay_scores)
    intent_signal_final = min(intent_signal_final, 60.0)

    print(
        f"   🎯 Intent final={intent_signal_final:.1f} "
        f"(threshold={FULFILLMENT_MIN_INTENT_SCORE}), "
        f"decay={[f'{s:.1f}' for s in after_decay_scores]}"
    )

    all_fabricated = bool(signal_results) and all(r["confidence"] == 0 for r in signal_results)
    if all_fabricated:
        print(f"   ⚠️  All signals have confidence=0 (all_fabricated=True)")

    shared_fields = dict(
        tier1_passed=True,
        tier2_passed=True,
        email_verified=email_verified,
        person_verified=person_verified,
        company_verified=company_verified,
        attribute_verification=attribute_verification_result,
        rep_score=rep_score_val,
        intent_signal_raw=max(after_decay_scores) if after_decay_scores else 0.0,
        intent_signal_final=intent_signal_final,
        intent_decay_multiplier=_avg([r["decay_mult"] for r in signal_results]),
        all_fabricated=all_fabricated,
        intent_signals_detail=signal_details,
    )

    # ------------------------------------------------------------------
    # Required-signal gate
    # ------------------------------------------------------------------
    # For each buyer spec marked ``required=True``, the lead must
    # produce at least one miner signal that:
    #   (a) was matched to that spec by the LLM (matched_icp_signal_idx
    #       equals the spec's index), AND
    #   (b) actually verified — after_decay_score > 0. This rules out
    #       signals where the URL didn't fetch, confidence was 0, or the
    #       strict judge rejected the evidence as marketing copy.
    #
    # If any required spec is missing satisfying evidence the lead
    # fails with ``missing_required_intent_signal`` regardless of the
    # numeric intent total — required is a hard gate, not a weight.
    required_indices = [
        i for i, spec in enumerate(icp_signal_specs) if spec.required
    ]
    satisfied_required: Set[int] = set()
    if required_indices:
        for r in signal_results:
            idx = r.get("matched_icp_signal_idx", -1)
            if (
                idx in required_indices
                and float(r.get("after_decay", 0.0)) > 0
            ):
                satisfied_required.add(idx)
    missing_required = [i for i in required_indices if i not in satisfied_required]

    if missing_required:
        missing_texts = [icp_signal_specs[i].text for i in missing_required]
        print(
            f"   ❌ Lead missing required intent signal(s): "
            f"{missing_texts}"
        )
        return FulfillmentScoreResult(
            **shared_fields,
            final_score=0.0,
            failure_reason="missing_required_intent_signal",
            failure_detail=_build_failure_detail(
                "missing_required_intent_signal",
                signal_details=signal_details,
                intent_final=intent_signal_final,
                missing_required_signals=missing_texts,
            ),
        )

    # ------------------------------------------------------------------
    # Threshold gate
    # ------------------------------------------------------------------
    if intent_signal_final < FULFILLMENT_MIN_INTENT_SCORE:
        return FulfillmentScoreResult(
            **shared_fields,
            final_score=0.0,
            failure_reason="insufficient_intent",
            failure_detail=_build_failure_detail(
                "insufficient_intent",
                signal_details=signal_details,
                intent_final=intent_signal_final,
            ),
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
        try:
            result = await score_fulfillment_lead(
                lead, icp, seen_companies, email_result=per_lead,
                use_apify=use_apify,
            )
        except Exception as e:
            # Per-lead guard: one bad lead (e.g., LeadOutput Pydantic enum
            # rejection on an unknown seniority value, or any other crash
            # inside score_fulfillment_lead) must NOT take down the whole
            # batch.  Surface as a failed lead with a clear failure_reason
            # so the validator's results file stays complete and the
            # request reaches consensus on its remaining leads.
            err_type = type(e).__name__
            err_msg = str(e)
            # Detect the most common case so dashboards group it cleanly.
            reason = "invalid_seniority" if "Seniority" in err_msg or "seniority" in err_msg \
                else "scorer_crashed"
            logger.warning(
                f"score_fulfillment_lead crashed for {getattr(lead, 'full_name', '?')!r}"
                f" @ {getattr(lead, 'business', '?')!r}: {err_type}: {err_msg[:300]}"
            )
            result = FulfillmentScoreResult(
                lead_id=getattr(lead, "lead_id", "") or "",
                tier1_passed=False,
                tier2_passed=False,
                failure_reason=reason,
                failure_detail=f"{err_type}: {err_msg[:300]}",
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
                        "failure_detail": "Multiple leads have identical templated descriptions",
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
