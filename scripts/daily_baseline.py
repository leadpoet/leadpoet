"""Manual one-shot: run configured baseline arms on today's ICP set and persist
their baseline_score rows to qualification_baselines.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("baseline_bootstrap")

from supabase import create_client
from qualification.scoring.baseline import (
    REFERENCE_MODEL_ID,
    run_and_save_baseline,
    save_baseline_failure_to_db,
    save_baseline_to_db,
)
from qualification.scoring.baseline_arms import (
    daily_baseline_arm_specs,
    resolve_qualify_fn,
)


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

    arms = daily_baseline_arm_specs()
    arm_model_ids = {arm.model_id for arm in arms}
    logger.info(
        "Configured baseline arms: "
        + ", ".join(f"{arm.label}={arm.model_id}" for arm in arms)
    )

    # ── Step 2: skip arms that already have completed rows ────────────
    b = sb.table("qualification_baselines") \
        .select("set_id, model_id, run_status") \
        .eq("set_id", today_set_id) \
        .eq("run_status", "completed") \
        .execute()
    completed_model_ids = {
        row.get("model_id") or REFERENCE_MODEL_ID
        for row in (b.data or [])
        if (row.get("model_id") or REFERENCE_MODEL_ID) in arm_model_ids
    }
    if arm_model_ids.issubset(completed_model_ids):
        logger.info("All configured baseline arms already completed for today")
        return 0

    # ── Step 3: import + run configured arms ──────────────────────────
    from qualification.scoring.lead_scorer import score_company

    failures = 0
    for arm in arms:
        if arm.model_id in completed_model_ids:
            logger.info(f"Skipping completed baseline arm {arm.model_id}")
            continue

        started = time.monotonic()
        try:
            qualify = resolve_qualify_fn(arm)
            logger.info(
                f"Starting baseline arm {arm.label} ({arm.model_id}) "
                f"against {len(icps)} ICPs"
            )
            record = await run_and_save_baseline(
                set_id=today_set_id,
                icp_set=icps,
                qualify_fn=qualify,
                score_fn=score_company,
                model_id=arm.model_id,
                score_cost_exempt=True,
            )
            duration = time.monotonic() - started

            logger.info(
                f"Baseline arm {arm.model_id} complete: "
                f"baseline_score={record.baseline_score:.2f} "
                f"duration={duration:.0f}s"
            )
            logger.info(
                f"{arm.model_id} per_icp_scores: "
                f"{[f'{s:.2f}' for s in record.per_icp_scores]}"
            )

            save_baseline_to_db(
                record, sb,
                icp_set_hash=icp_set_hash,
                run_duration_seconds=duration,
                run_status="completed",
            )
            logger.info(f"✅ Baseline row written for set_id={today_set_id} model_id={arm.model_id}")
        except Exception as e:
            failures += 1
            duration = time.monotonic() - started
            logger.exception(f"Baseline arm {arm.model_id} FAILED: {e}")
            try:
                save_baseline_failure_to_db(
                    today_set_id,
                    arm.model_id,
                    sb,
                    icp_set_hash=icp_set_hash,
                    run_duration_seconds=duration,
                )
            except Exception:
                logger.exception(f"Failed to persist failed row for arm {arm.model_id}")

    return 1 if failures else 0


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
            save_baseline_failure_to_db(today_set_id, REFERENCE_MODEL_ID, sb)
        except Exception:
            pass
        sys.exit(1)
