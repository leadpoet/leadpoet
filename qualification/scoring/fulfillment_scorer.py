"""
Validator-side fulfillment lead scoring.

Receives revealed FulfillmentLead data from the gateway, runs the three-tier
scoring pipeline, applies cross-miner deduplication, and returns score
results ready for gateway submission.

This module calls into gateway/fulfillment/scoring.py for the actual Tier 1-3
pipeline but adds:
  - Batch orchestration across multiple miners
  - Cross-miner deduplication (highest score wins, ties split)
  - Entity verification via validator_models checks
  - Result formatting for gateway_submit_fulfillment_scores
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from gateway.fulfillment.models import (
    FulfillmentICP,
    FulfillmentLead,
    FulfillmentScoreResult,
)
from gateway.fulfillment.scoring import (
    score_fulfillment_batch,
    aggregate_intent_scores,
)

logger = logging.getLogger(__name__)


async def score_miner_submission(
    leads: List[dict],
    icp_details: dict,
) -> List[FulfillmentScoreResult]:
    """Score a single miner's revealed leads against the ICP.

    Args:
        leads: raw dicts from the gateway reveal payload
        icp_details: ICP details dict from the fulfillment request
    """
    icp = FulfillmentICP(**icp_details)

    parsed_leads: List[FulfillmentLead] = []
    parse_errors: List[FulfillmentScoreResult] = []

    for ld in leads:
        try:
            parsed_leads.append(FulfillmentLead(**ld))
            parse_errors.append(None)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Lead parse error: {e}")
            parsed_leads.append(None)  # type: ignore[arg-type]
            parse_errors.append(FulfillmentScoreResult(
                tier1_passed=False,
                failure_reason=f"parse_error: {e}",
            ))

    scoreable = [l for l in parsed_leads if l is not None]
    if not scoreable:
        return [pe if pe is not None else FulfillmentScoreResult(failure_reason="empty")
                for pe in parse_errors]

    # FULFILLMENT_USE_APIFY env var flips fulfillment Stage 4/5 from the
    # legacy check_linkedin_gse + check_stage5_unified pair to the new
    # Apify-based fulfillment_person_verification +
    # fulfillment_company_verification path.  Default is OFF — opt in by
    # setting FULFILLMENT_USE_APIFY=true (or 1/yes) in the container env.
    # Sourcing is unaffected by either value: that pipeline goes through
    # run_automated_checks in neurons/validator.py and never reaches this
    # function.
    #
    # Operator notes:
    #   * When ON, requires APIFY_API_TOKEN in the container env (wired
    #     in deploy_dynamic.sh -e flags); also requires OPENROUTER_KEY
    #     and SCRAPINGDOG_API_KEY which are already provisioned.
    #   * Graceful degradation: if APIFY_API_TOKEN is missing or invalid
    #     while the flag is on, fulfillment_person_verification returns
    #     "fulfillment_person_fetch_failed" and scoring.py falls back to
    #     the legacy ScrapingDog Stage 4 path automatically (see
    #     scoring.py L374-376).
    #   * Kill switch: set FULFILLMENT_USE_APIFY=false in .env and restart
    #     the validator — instantly reverts to legacy Stage 4/5.
    import os as _os
    _use_apify = _os.getenv("FULFILLMENT_USE_APIFY", "false").strip().lower() in (
        "true", "1", "yes", "on",
    )
    scored = await score_fulfillment_batch(scoreable, icp, use_apify=_use_apify)

    results: List[FulfillmentScoreResult] = []
    score_idx = 0
    for pe in parse_errors:
        if pe is not None:
            results.append(pe)
        else:
            results.append(scored[score_idx])
            score_idx += 1

    return results


def deduplicate_across_miners(
    scored_submissions: Dict[str, List[Tuple[dict, FulfillmentScoreResult]]],
) -> Dict[str, List[FulfillmentScoreResult]]:
    """Cross-miner deduplication: highest score wins per company.

    Args:
        scored_submissions: ``{miner_hotkey: [(lead_dict, score_result), ...]}``

    Returns:
        ``{miner_hotkey: [FulfillmentScoreResult]}`` — losers have
        ``final_score=0`` and ``failure_reason="dedup_lost"``.
    """
    company_best: Dict[str, List[Tuple[str, int, float]]] = {}

    for hotkey, entries in scored_submissions.items():
        for idx, (lead_dict, score) in enumerate(entries):
            if score.final_score <= 0:
                continue
            company = _normalize_company(lead_dict.get("business", ""))
            if not company:
                continue
            company_best.setdefault(company, []).append(
                (hotkey, idx, score.final_score)
            )

    winners: Set[Tuple[str, int]] = set()
    for company, candidates in company_best.items():
        best_score = max(c[2] for c in candidates)
        top = [(hk, idx) for hk, idx, sc in candidates if sc == best_score]
        for pair in top:
            winners.add(pair)

    output: Dict[str, List[FulfillmentScoreResult]] = {}
    for hotkey, entries in scored_submissions.items():
        out_list: List[FulfillmentScoreResult] = []
        for idx, (lead_dict, score) in enumerate(entries):
            if score.final_score > 0 and (hotkey, idx) not in winners:
                out_list.append(FulfillmentScoreResult(
                    **{
                        **score.model_dump(),
                        "final_score": 0.0,
                        "failure_reason": "dedup_lost",
                    }
                ))
            else:
                out_list.append(score)
        output[hotkey] = out_list

    return output


def format_scores_for_gateway(
    miner_hotkey: str,
    lead_ids: List[str],
    results: List[FulfillmentScoreResult],
    request_id: str = "",
    submission_id: str = "",
) -> List[dict]:
    """Build the JSON payload for ``gateway_submit_fulfillment_scores``.

    Each entry is a dict with ``lead_id``, ``request_id``, ``submission_id``
    plus the score breakdown fields expected by the RPC function.
    """
    if len(lead_ids) != len(results):
        raise ValueError(
            "fulfillment score cardinality mismatch: "
            f"{len(lead_ids)} lead IDs but {len(results)} results"
        )
    out: List[dict] = []
    for lid, result in zip(lead_ids, results):
        d = result.model_dump()
        d["lead_id"] = lid
        d["miner_hotkey"] = miner_hotkey
        d["request_id"] = request_id
        d["submission_id"] = submission_id
        out.append(d)
    return out


def _normalize_company(name: str) -> str:
    """Lowercase, strip common suffixes for dedup key."""
    n = name.strip().lower()
    for suffix in (" inc.", " inc", " llc", " ltd.", " ltd", " corp.", " corp",
                   " co.", " co", " gmbh", " s.a.", " plc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].rstrip()
    return n
