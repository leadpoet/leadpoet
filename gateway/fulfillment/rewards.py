"""
Fulfillment reward calculation.

Phase 1: rewards are stored in the DB only (no emission weight impact).
Phase 2 will integrate with submit_weights_at_epoch_end().
"""

import logging

logger = logging.getLogger(__name__)


def _get_supabase():
    from gateway.db.client import get_write_client
    return get_write_client()


def calculate_lead_rewards(
    request_id: str,
    winners: list,
    z_pct: float,
    current_epoch: int,
    l_epochs: int = 100,
) -> None:
    """
    Assign reward_pct and reward_expires_epoch on fulfillment_score_consensus
    for winning leads.

    Each winner dict must include:
      - consensus_id or (request_id, submission_id, lead_id)
      - miner_hotkey
      - tie_count (1 if sole winner, >1 if tied)
    """
    supabase = _get_supabase()
    expires_epoch = current_epoch + l_epochs

    for w in winners:
        pct = z_pct / w.get("tie_count", 1)

        try:
            supabase.table("fulfillment_score_consensus").update({
                "is_winner": True,
                "reward_pct": pct,
                "reward_expires_epoch": expires_epoch,
            }).eq("request_id", w.get("request_id", request_id)) \
              .eq("submission_id", w["submission_id"]) \
              .eq("lead_id", w["lead_id"]) \
              .execute()
        except Exception as e:
            logger.error(f"Failed to assign reward for lead {w.get('lead_id')}: {e}")
