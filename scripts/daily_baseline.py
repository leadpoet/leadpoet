"""Manual one-shot: run the reference model on today's ICP set and persist
the baseline_score to qualification_baselines. This is the bootstrapping
run for 20260530 — automation kicks in from tomorrow's 00:00 UTC rotation.

Logs every step + writes a 'failed' row on crash so it's visible in DB.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")

from dotenv import load_dotenv
load_dotenv(Path("/Users/tasnimul/Desktop/leadpoet/.env"), override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("baseline_bootstrap")

from supabase import create_client


async def main() -> int:
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # ── Step 1: fetch today's active ICP set ──────────────────────────
    today_set_id = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    logger.info(f"Bootstrapping baseline for set_id={today_set_id}")

    r = sb.table("qualification_private_icp_sets") \
        .select("set_id, icps, icp_set_hash") \
        .eq("set_id", today_set_id) \
        .eq("is_active", True) \
        .execute()
    if not r.data:
        logger.error(f"No active ICP set for set_id={today_set_id}; aborting")
        return 1
    active = r.data[0]
    icps = active["icps"]
    icp_set_hash = active["icp_set_hash"]
    logger.info(f"Loaded {len(icps)} ICPs (hash={icp_set_hash[:12]})")

    # ── Step 2: skip if a completed row already exists ────────────────
    b = sb.table("qualification_baselines") \
        .select("set_id, run_status") \
        .eq("set_id", today_set_id).execute()
    if b.data and b.data[0].get("run_status") == "completed":
        logger.info("Baseline already completed for today; nothing to do")
        return 0

    # ── Step 3: import + run reference model ──────────────────────────
    from miner_models.qualification_model import qualify
    from qualification.scoring.lead_scorer import score_company
    from qualification.scoring.baseline import (
        run_and_save_baseline,
        save_baseline_to_db,
        REFERENCE_MODEL_ID,
    )

    started = time.monotonic()
    logger.info(f"Starting reference-model evaluation against {len(icps)} ICPs")
    record = await run_and_save_baseline(
        set_id=today_set_id,
        icp_set=icps,
        qualify_fn=qualify,
        score_fn=score_company,
    )
    duration = time.monotonic() - started

    logger.info(
        f"Reference model evaluation complete: "
        f"baseline_score={record.baseline_score:.2f} duration={duration:.0f}s"
    )
    logger.info(f"per_icp_scores: {[f'{s:.2f}' for s in record.per_icp_scores]}")

    # ── Step 4: persist ───────────────────────────────────────────────
    save_baseline_to_db(
        record, sb,
        icp_set_hash=icp_set_hash,
        run_duration_seconds=duration,
        run_status="completed",
    )
    logger.info(f"✅ Baseline row written for set_id={today_set_id}")
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except Exception as e:
        logger.exception(f"Bootstrap FAILED: {e}")
        # Best-effort failure row so the missing baseline is visible
        try:
            sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
            today_set_id = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
            sb.table("qualification_baselines").upsert({
                "set_id": today_set_id,
                "baseline_score": 0.0,
                "per_icp_scores": [],
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "model_id": "reference:qualification_model:v1",
                "run_status": "failed",
            }, on_conflict="set_id").execute()
        except Exception:
            pass
        sys.exit(1)
