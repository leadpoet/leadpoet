"""
Weighted Consensus Utilities

Implements V-score weighted consensus for lead validation outcomes.

The consensus mechanism aggregates validator decisions using their V-scores
(validator scores from Bittensor) as weights. This ensures that validators
with higher stake/reputation have more influence on the final outcome.
"""

from typing import Dict, List
from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def compute_weighted_consensus(lead_id: str, epoch_id: int) -> Dict:
    """
    Compute v_score × stake weighted consensus for lead outcome.
    
    Aggregates all validator decisions for a lead using (v_score × stake) as weights.
    Only includes validators who validated THIS specific lead.
    
    Algorithm:
    1. Query all revealed validations for this lead (decision, rep_score, rejection_reason, v_score, stake)
    2. Calculate weight for each validator: v_score × stake
    3. Compute weighted rep_score (Σ rep_score × weight / Σ weight)
    4. Compute weighted approval (Σ weight for "approve" / Σ weight)
    5. Final decision: "approve" if approval_ratio > 50%, else "deny"
    6. Calculate primary_rejection_reason (most common among denials)
    
    Args:
        lead_id: Lead UUID
        epoch_id: Epoch number
    
    Returns:
        {
            "lead_id": str,
            "epoch_id": int,
            "final_decision": "approve" or "deny",
            "final_rep_score": float,
            "primary_rejection_reason": str,  # "pass" if approved, else most common reason
            "validator_count": int,
            "consensus_weight": float,
            "approval_ratio": float
        }
    
    Example:
        >>> consensus = await compute_weighted_consensus("uuid", 100)
        >>> consensus
        {
            "lead_id": "uuid",
            "epoch_id": 100,
            "final_decision": "approve",
            "final_rep_score": 0.87,
            "primary_rejection_reason": "pass",
            "validator_count": 5,
            "consensus_weight": 450.0,
            "approval_ratio": 0.6823
        }
    
    Notes:
        - Only considers revealed validations (decision != NULL)
        - Returns "deny" if no validators revealed
        - Weight = v_score × stake (both factors matter)
        - Approval requires >50% weighted votes
        - Only includes validators who validated THIS lead
    """
    
    # ========================================
    # Step 1: Query all revealed validations for this lead
    # ========================================
    try:
        result = supabase.table("validation_evidence_private") \
            .select("validator_hotkey, decision, rep_score, rejection_reason, v_score, stake") \
            .eq("lead_id", lead_id) \
            .eq("epoch_id", epoch_id) \
            .not_.is_("decision", "null") \
            .not_.is_("rep_score", "null") \
            .execute()
        
        validations = result.data
    
    except Exception as e:
        print(f"❌ Error querying validations for lead {lead_id}: {e}")
        return {
            "lead_id": lead_id,
            "epoch_id": epoch_id,
            "final_decision": "deny",
            "final_rep_score": 0.0,
            "primary_rejection_reason": "no_validations",
            "validator_count": 0,
            "consensus_weight": 0.0,
            "approval_ratio": 0.0,
            "error": str(e)
        }
    
    # ========================================
    # Step 2: Handle no validations case
    # ========================================
    if not validations:
        print(f"⚠️  No revealed validations for lead {lead_id} in epoch {epoch_id}")
        return {
            "lead_id": lead_id,
            "epoch_id": epoch_id,
            "final_decision": "deny",
            "final_rep_score": 0.0,
            "primary_rejection_reason": "no_validations",
            "validator_count": 0,
            "consensus_weight": 0.0,
            "approval_ratio": 0.0
        }
    
    # ========================================
    # Step 3: Compute weighted aggregates
    # ========================================
    total_weight = 0.0
    weighted_rep_score = 0.0
    weighted_approval = 0.0
    rejection_reasons = []
    
    for v in validations:
        v_score = float(v["v_score"])
        stake = float(v["stake"])
        weight = v_score * stake  # Weight = v_score × stake
        
        total_weight += weight
        
        # Accumulate weighted rep_score
        weighted_rep_score += float(v["rep_score"]) * weight
        
        # Accumulate weighted approval (1 = approve, 0 = deny)
        if v["decision"] == "approve":
            weighted_approval += weight
        else:  # decision == "deny"
            # Collect rejection reasons for denied leads
            rejection_reason = v.get("rejection_reason", "unknown")
            if rejection_reason:
                rejection_reasons.append(rejection_reason)
    
    # ========================================
    # Step 4: Calculate final values
    # ========================================
    if total_weight > 0:
        final_rep_score = weighted_rep_score / total_weight
        approval_ratio = weighted_approval / total_weight
    else:
        final_rep_score = 0.0
        approval_ratio = 0.0
    
    # Final decision: approve if >50% weighted approval
    final_decision = "approve" if approval_ratio > 0.5 else "deny"
    
    # Calculate primary rejection reason (most common among denials)
    from collections import Counter
    if final_decision == "approve":
        primary_rejection_reason = "pass"
    elif rejection_reasons:
        primary_rejection_reason = Counter(rejection_reasons).most_common(1)[0][0]
    else:
        primary_rejection_reason = "unknown"
    
    # ========================================
    # Step 5: Log consensus result
    # ========================================
    print(f"📊 Consensus for lead {lead_id[:8]}... (epoch {epoch_id}):")
    print(f"   Validators: {len(validations)}")
    print(f"   Total weight: {total_weight:.2f}")
    print(f"   Final rep score: {final_rep_score:.4f}")
    print(f"   Approval ratio: {approval_ratio:.4f} ({approval_ratio * 100:.2f}%)")
    print(f"   Final decision: {final_decision}")
    print(f"   Primary rejection reason: {primary_rejection_reason}")
    
    # ========================================
    # Step 6: Return consensus result
    # ========================================
    return {
        "lead_id": lead_id,
        "epoch_id": epoch_id,
        "final_decision": final_decision,
        "final_rep_score": round(final_rep_score, 4),
        "primary_rejection_reason": primary_rejection_reason,
        "validator_count": len(validations),
        "consensus_weight": round(total_weight, 2),
        "approval_ratio": round(approval_ratio, 4)
    }


async def compute_consensus_batch(lead_ids: List[str], epoch_id: int) -> List[Dict]:
    """
    Compute consensus for multiple leads in batch.
    
    More efficient than calling compute_weighted_consensus() individually.
    
    Args:
        lead_ids: List of lead UUIDs
        epoch_id: Epoch number
    
    Returns:
        List of consensus results (one per lead)
    
    Example:
        >>> results = await compute_consensus_batch(["uuid1", "uuid2"], 100)
        >>> len(results)
        2
    """
    results = []
    
    for lead_id in lead_ids:
        consensus = await compute_weighted_consensus(lead_id, epoch_id)
        results.append(consensus)
    
    return results


def get_consensus_stats(consensus_results: List[Dict]) -> Dict:
    """
    Get aggregate statistics for a batch of consensus results.
    
    Args:
        consensus_results: List of consensus dictionaries
    
    Returns:
        {
            "total_leads": int,
            "approved": int,
            "denied": int,
            "approval_rate": float,
            "avg_rep_score": float,
            "avg_validator_count": float
        }
    
    Example:
        >>> stats = get_consensus_stats(results)
        >>> stats
        {
            "total_leads": 100,
            "approved": 75,
            "denied": 25,
            "approval_rate": 0.75,
            "avg_rep_score": 0.85,
            "avg_validator_count": 5.2
        }
    """
    if not consensus_results:
        return {
            "total_leads": 0,
            "approved": 0,
            "denied": 0,
            "approval_rate": 0.0,
            "avg_rep_score": 0.0,
            "avg_validator_count": 0.0
        }
    
    total_leads = len(consensus_results)
    approved = sum(1 for r in consensus_results if r["final_decision"] == "approve")
    denied = total_leads - approved
    approval_rate = approved / total_leads if total_leads > 0 else 0.0
    
    avg_rep_score = sum(r["final_rep_score"] for r in consensus_results) / total_leads
    avg_validator_count = sum(r["validator_count"] for r in consensus_results) / total_leads
    
    return {
        "total_leads": total_leads,
        "approved": approved,
        "denied": denied,
        "approval_rate": round(approval_rate, 4),
        "avg_rep_score": round(avg_rep_score, 4),
        "avg_validator_count": round(avg_validator_count, 2)
    }


async def verify_consensus_determinism(lead_id: str, epoch_id: int, expected_consensus: Dict) -> bool:
    """
    Verify that consensus computation is deterministic.
    
    Useful for testing and auditing. Recomputes consensus and checks
    if it matches the expected result.
    
    Args:
        lead_id: Lead UUID
        epoch_id: Epoch number
        expected_consensus: Previously computed consensus
    
    Returns:
        True if computed consensus matches expected, False otherwise
    
    Example:
        >>> is_deterministic = await verify_consensus_determinism(
        ...     "uuid", 100, expected_consensus
        ... )
        >>> is_deterministic
        True
    """
    computed = await compute_weighted_consensus(lead_id, epoch_id)
    
    # Compare key fields
    matches = (
        computed["final_decision"] == expected_consensus["final_decision"] and
        computed["final_rep_score"] == expected_consensus["final_rep_score"] and
        computed["validator_count"] == expected_consensus["validator_count"]
    )
    
    if not matches:
        print(f"⚠️  Consensus mismatch for lead {lead_id}:")
        print(f"   Expected: {expected_consensus}")
        print(f"   Computed: {computed}")
    
    return matches

