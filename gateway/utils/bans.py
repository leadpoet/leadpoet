"""
Ban Utilities (shared across qualification and fulfillment)

Extracted from gateway/qualification/api/submit.py (WP0b) so both
qualification and fulfillment endpoints can import ban functions
without cross-module coupling.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import bittensor as bt

logger = logging.getLogger(__name__)


def is_hotkey_banned_sync(hotkey: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a hotkey is banned from submitting models.

    Queries the public ``banned_hotkeys`` table in Supabase.
    Banned hotkeys cannot submit new models and lose champion status if they have it.

    Fail-open: returns ``(False, None)`` on any DB exception so legitimate
    miners are not blocked by transient DB errors.

    This is the blocking implementation — call it from worker threads.
    Event-loop callers must use ``is_hotkey_banned`` instead, which runs
    this in a thread so the loop keeps serving other requests meanwhile.
    """
    try:
        from gateway.db.client import get_write_client

        supabase = get_write_client()

        response = supabase.table("banned_hotkeys") \
            .select("hotkey, reason, banned_at, banned_by") \
            .eq("hotkey", hotkey) \
            .execute()

        if response.data and len(response.data) > 0:
            ban_record = response.data[0]
            reason = ban_record.get("reason", "Banned for gaming/hardcoding violations")
            banned_at = ban_record.get("banned_at", "unknown")
            logger.warning(f"🚫 Banned hotkey attempted submission: {hotkey[:16]}... (reason: {reason}, banned_at: {banned_at})")
            return True, reason

        return False, None

    except Exception as e:
        bt.logging.warning(f"Ban check failed for {hotkey[:16]}...: {e} — failing open (allowing request)")
        return False, None


async def is_hotkey_banned(hotkey: str) -> Tuple[bool, Optional[str]]:
    """Async wrapper for :func:`is_hotkey_banned_sync`.

    Runs the blocking Supabase lookup in a worker thread. The gateway is a
    single-event-loop process, so a blocking DB round-trip here would stall
    every in-flight request for its duration — this check runs on the
    hottest miner paths, where that stall compounds into a gateway-wide
    backlog.
    """
    return await asyncio.to_thread(is_hotkey_banned_sync, hotkey)


async def promote_next_champion() -> Optional[Dict[str, Any]]:
    """
    After a champion is dethroned (e.g. due to ban), find and promote the next
    highest-scoring eligible model from today.

    Eligibility rules:
    - Model was submitted after 12:00 AM UTC today
    - Score > 10.0
    - Status = 'evaluated'
    - Miner hotkey is NOT in banned_hotkeys table
    """
    try:
        from gateway.db.client import get_write_client
        supabase = get_write_client()

        today_midnight = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        banned_response = supabase.table("banned_hotkeys") \
            .select("hotkey") \
            .execute()
        banned_hotkeys = {r["hotkey"] for r in (banned_response.data or [])}

        candidates_response = supabase.table("qualification_models") \
            .select("id, model_name, miner_hotkey, score, champion_at, "
                    "evaluation_cost_usd, evaluation_time_seconds, code_content") \
            .eq("status", "evaluated") \
            .eq("is_champion", False) \
            .gt("score", 10.0) \
            .gte("created_at", today_midnight) \
            .order("score", desc=True) \
            .limit(50) \
            .execute()

        if not candidates_response.data:
            logger.info("📭 No eligible models found today for auto-promotion")
            return None

        promoted = None
        for candidate in candidates_response.data:
            if candidate["miner_hotkey"] in banned_hotkeys:
                continue
            promoted = candidate
            break

        if not promoted:
            logger.info("📭 All today's models belong to banned hotkeys - no promotion")
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("qualification_models").update({
            "is_champion": True,
            "champion_at": now_iso,
            "dethroned_at": None,
        }).eq("id", promoted["id"]).execute()

        logger.warning(
            f"👑 AUTO-PROMOTED new champion after ban: {promoted['model_name']} "
            f"(hotkey: {promoted['miner_hotkey'][:16]}..., score: {promoted['score']:.2f})"
        )
        print(f"\n{'='*60}")
        print(f"👑 AUTO-PROMOTED NEW CHAMPION (after ban dethronement)")
        print(f"   Model:  {promoted['model_name']}")
        print(f"   Miner:  {promoted['miner_hotkey'][:20]}...")
        print(f"   Score:  {promoted['score']:.2f}")
        print(f"   Source: Highest eligible model submitted today (score > 10)")
        print(f"{'='*60}\n")

        return {
            "model_id": promoted["id"],
            "model_name": promoted["model_name"],
            "miner_hotkey": promoted["miner_hotkey"],
            "score": promoted["score"],
            "champion_at": now_iso,
        }

    except Exception as e:
        logger.error(f"Error in promote_next_champion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def ban_hotkey(
    hotkey: str,
    reason: str,
    banned_by: str = "system"
) -> bool:
    """
    Ban a hotkey from submitting models.

    Also revokes champion status if the hotkey is the current champion,
    then auto-promotes the next eligible model from today.

    Uses upsert (ON CONFLICT) so banning the same hotkey twice updates
    the reason/timestamp instead of failing.
    """
    try:
        from gateway.db.client import get_write_client

        supabase = get_write_client()

        supabase.table("banned_hotkeys").upsert({
            "hotkey": hotkey,
            "reason": reason,
            "banned_by": banned_by,
            "banned_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="hotkey").execute()

        logger.warning(f"🚫 Hotkey banned: {hotkey[:16]}... (reason: {reason}, by: {banned_by})")

        was_champion = False
        try:
            dethrone_result = supabase.table("qualification_models") \
                .update({
                    "is_champion": False,
                    "dethroned_at": datetime.now(timezone.utc).isoformat(),
                }) \
                .eq("miner_hotkey", hotkey) \
                .eq("is_champion", True) \
                .execute()

            if dethrone_result.data and len(dethrone_result.data) > 0:
                was_champion = True
                model_name = dethrone_result.data[0].get("model_name", "unknown")
                logger.warning(f"👑➡️🚫 Champion dethroned due to ban: {model_name} (hotkey: {hotkey[:16]}...)")
        except Exception as dethrone_error:
            logger.error(f"Error dethroning banned champion: {dethrone_error}")

        if was_champion:
            await promote_next_champion()

        return True

    except Exception as e:
        logger.error(f"Error banning hotkey: {e}")
        return False


async def dethrone_banned_champions() -> int:
    """
    Check all banned hotkeys and dethrone any that are still champions.

    Cleanup function that can be called periodically or manually.
    After dethroning, auto-promotes the next eligible model from today.
    """
    try:
        from gateway.db.client import get_write_client

        supabase = get_write_client()

        banned_response = supabase.table("banned_hotkeys") \
            .select("hotkey") \
            .execute()

        if not banned_response.data:
            return 0

        banned_hotkeys = [r["hotkey"] for r in banned_response.data]

        dethroned_count = 0
        for hotkey in banned_hotkeys:
            result = supabase.table("qualification_models") \
                .update({
                    "is_champion": False,
                    "dethroned_at": datetime.now(timezone.utc).isoformat(),
                }) \
                .eq("miner_hotkey", hotkey) \
                .eq("is_champion", True) \
                .execute()

            if result.data and len(result.data) > 0:
                dethroned_count += 1
                model_name = result.data[0].get("model_name", "unknown")
                logger.warning(f"👑➡️🚫 Dethroned banned champion: {model_name} (hotkey: {hotkey[:16]}...)")

        if dethroned_count > 0:
            await promote_next_champion()

        return dethroned_count

    except Exception as e:
        logger.error(f"Error in dethrone_banned_champions: {e}")
        return 0
