"""
Consensus calculation for distributed validator ranking.

Implements weighted consensus where each validator's ranking is weighted
by their trust value: S_lead = Σ(S_v * V_v) for all validators
"""
import bittensor as bt
from typing import List, Dict, Tuple
from collections import defaultdict
from Leadpoet.utils.utils_lead_extraction import get_company, get_email, get_field


def calculate_consensus_ranking(
        validator_rankings: List[Dict],
        num_leads_requested: int,
        min_validators: int = 1) -> Tuple[List[Dict], Dict]:
    """
    Calculate consensus ranking from multiple validator submissions.

    Formula: S_lead = (S_1 * V_1) + (S_2 * V_2) + ... + (S_N * V_N)
    Where:
        S_v = score from validator v for that specific lead
        V_v = trust value for validator v
        N = total number of validators

    Args:
        validator_rankings: List of validator ranking submissions
        num_leads_requested: Number of leads client requested
        min_validators: Minimum validators required for consensus (default 1)

    Returns:
        Tuple of (final_ranked_leads, consensus_metadata)
    """
    if not validator_rankings:
        bt.logging.warning("No validator rankings provided for consensus")
        return [], {"error": "No validator rankings", "num_validators": 0}

    if len(validator_rankings) < min_validators:
        bt.logging.warning(
            f"Only {len(validator_rankings)} validator(s) submitted rankings, "
            f"minimum required: {min_validators}")
        # Still proceed with available validators

    bt.logging.info(
        f"🔍 Calculating consensus from {len(validator_rankings)} validator(s)")

    # Build a mapping: lead_identifier → {validator: (score, trust)}
    # We'll use (business_name, email) as lead identifier
    lead_scores = defaultdict(lambda: {
        "scores": [],
        "trusts": [],
        "lead_data": None
    })

    total_trust = 0.0
    for ranking in validator_rankings:
        validator_hotkey = ranking.get("validator_hotkey", "unknown")
        validator_trust = float(ranking.get("validator_trust", 0.0))
        ranked_leads = ranking.get("ranked_leads", [])

        total_trust += validator_trust

        bt.logging.debug(
            f"  Validator {validator_hotkey[:10]}... (trust={validator_trust:.4f}) "
            f"ranked {len(ranked_leads)} leads")

        for lead_entry in ranked_leads:
            lead = lead_entry if isinstance(lead_entry, dict) else {
                "lead": lead_entry
            }

            # Extract lead data
            lead_data = lead.get("lead", lead)
            business = get_company(lead_data)
            email = get_email(lead_data)

            # Use (business, email) as unique identifier
            lead_id = (business.lower().strip(), email.lower().strip())

            # Get validator's score for this lead
            score = float(get_field(lead, "score", "intent_score", default=0.0))

            # Store this validator's score and trust for this lead
            lead_scores[lead_id]["scores"].append(score)
            lead_scores[lead_id]["trusts"].append(validator_trust)

            # Keep the lead data (use first occurrence)
            if lead_scores[lead_id]["lead_data"] is None:
                lead_scores[lead_id]["lead_data"] = lead_data

    # Calculate weighted consensus score for each lead
    consensus_leads = []
    for lead_id, data in lead_scores.items():
        scores = data["scores"]
        trusts = data["trusts"]
        lead_data = data["lead_data"]

        # Calculate: S_lead = Σ(S_v * V_v)
        weighted_score = sum(s * t for s, t in zip(scores, trusts))

        # Normalize by total trust (optional - helps with interpretation)
        if total_trust > 0:
            normalized_score = weighted_score / total_trust
        else:
            normalized_score = weighted_score

        consensus_leads.append({
            "lead": lead_data,
            "consensus_score": round(weighted_score, 6),
            "normalized_score": round(normalized_score, 6),
            "num_validators": len(scores),
            "validator_scores": scores,
            "validator_trusts": trusts,
        })

    # Sort by consensus score (descending)
    consensus_leads.sort(key=lambda x: x["consensus_score"], reverse=True)

    # Take top N requested
    final_leads = consensus_leads[:num_leads_requested]

    # Prepare final lead objects (remove metadata, keep only lead data + score)
    result_leads = []
    for item in final_leads:
        lead = item["lead"].copy()
        lead["consensus_score"] = item["consensus_score"]
        lead["num_validators_ranked"] = item["num_validators"]
        result_leads.append(lead)

    # Metadata about consensus
    metadata = {
        "num_validators":
        len(validator_rankings),
        "total_trust":
        round(total_trust, 4),
        "total_unique_leads":
        len(lead_scores),
        "leads_returned":
        len(result_leads),
        "validator_hotkeys": [
            r.get("validator_hotkey", "unknown")[:10]
            for r in validator_rankings
        ],
    }

    bt.logging.info(f"✅ Consensus calculated: {len(result_leads)} leads from "
                    f"{metadata['total_unique_leads']} unique leads, "
                    f"{metadata['num_validators']} validators")

    return result_leads, metadata


def get_active_validators() -> List[Dict]:
    """
    Get list of currently active validators on the subnet.

    Returns:
        List of validator info dicts with uid, hotkey, trust, is_serving, last_seen
    """
    try:
        from Leadpoet.utils.cloud_db import NETWORK, SUBNET_ID
        from datetime import datetime

        subtensor = bt.subtensor(network=NETWORK)
        mg = subtensor.metagraph(netuid=SUBNET_ID)
        current_block = subtensor.get_current_block()

        validators = []
        for uid in range(mg.n):
            if mg.validator_permit[uid].item():
                # Enhanced validator status monitoring

                # Calculate last_seen based on block activity
                # If validator is serving, they're currently active
                # Otherwise, estimate time since last update
                try:
                    last_update_block = mg.last_update[uid].item() if hasattr(
                        mg, 'last_update') else current_block
                    blocks_since_update = current_block - last_update_block
                    seconds_since_update = blocks_since_update * 12  # ~12 seconds per block
                    last_seen = datetime.utcnow().timestamp(
                    ) - seconds_since_update
                    last_seen_iso = datetime.fromtimestamp(
                        last_seen).isoformat() + "Z"
                except Exception:
                    # Fallback if we can't determine last_seen
                    last_seen_iso = datetime.utcnow().isoformat() + "Z"

                validators.append({
                    "uid":
                    uid,
                    "hotkey":
                    mg.hotkeys[uid],
                    "trust":
                    mg.validator_trust[uid].item(),
                    "is_serving":
                    mg.axons[uid].is_serving,
                    "stake":
                    mg.S[uid].item(),
                    "last_seen":
                    last_seen_iso,
                    "blocks_since_update":
                    blocks_since_update
                    if 'blocks_since_update' in locals() else 0,
                })

        bt.logging.debug(
            f"Found {len(validators)} active validators on subnet {SUBNET_ID}")
        return validators

    except Exception as e:
        bt.logging.error(f"Failed to get active validators: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return []


def should_use_direct_ranking(validator_rankings: List[Dict],
                              active_validators: List[Dict]) -> bool:
    """
    Determine if we should skip consensus and use direct ranking.

    Use direct ranking (no consensus) when:
    - Only 1 active validator on the subnet
    - Only 1 validator submitted ranking

    Args:
        validator_rankings: List of validator ranking submissions
        active_validators: List of active validators on subnet

    Returns:
        bool: True if should use direct ranking, False if should calculate consensus
    """
    num_active = len(active_validators)
    num_submitted = len(validator_rankings)

    # Single validator on subnet - use direct ranking
    if num_active == 1 and num_submitted == 1:
        bt.logging.info(
            "🔷 Single validator mode - using direct ranking (no consensus)")
        return True

    # Multiple validators exist but only one submitted (edge case)
    # Still calculate "consensus" to maintain consistent format
    return False


def get_direct_ranking(validator_ranking: Dict,
                       num_leads_requested: int) -> Tuple[List[Dict], Dict]:
    """
    Extract direct ranking from a single validator (no consensus calculation needed).

    This is used when only one validator is active on the subnet to avoid
    unnecessary consensus overhead.

    Args:
        validator_ranking: Single validator's ranking submission
        num_leads_requested: Number of leads client requested

    Returns:
        Tuple of (final_ranked_leads, metadata)
    """
    validator_hotkey = validator_ranking.get("validator_hotkey", "unknown")
    validator_trust = float(validator_ranking.get("validator_trust", 0.0))
    ranked_leads = validator_ranking.get("ranked_leads", [])

    bt.logging.info(
        f"🔷 Using direct ranking from validator {validator_hotkey[:10]}... "
        f"(trust={validator_trust:.4f})")

    # Take top N leads from this validator's ranking
    top_leads = ranked_leads[:num_leads_requested]

    # Format leads for response
    result_leads = []
    for item in top_leads:
        lead = item.get("lead", item)
        if isinstance(lead, dict):
            lead = lead.copy()
            # Use the validator's score as the consensus score
            lead["consensus_score"] = item.get("score", 0.0)
            lead["num_validators_ranked"] = 1
            result_leads.append(lead)

    # Metadata
    metadata = {
        "num_validators": 1,
        "total_trust": validator_trust,
        "total_unique_leads": len(ranked_leads),
        "leads_returned": len(result_leads),
        "validator_hotkeys": [validator_hotkey[:10]],
        "consensus_mode": "direct",  # Indicate this was direct, not consensus
    }

    bt.logging.info(f"✅ Direct ranking: {len(result_leads)} leads returned")

    return result_leads, metadata
