"""Small gateway worker for the daily public-baseline rebenchmark."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.daily_rebenchmark import run_daily_public_rebenchmark
from gateway.research_lab.scoring_control import get_scoring_control


logger = logging.getLogger(__name__)


class DailyPublicRebenchmarkWorker:
    """Run only the public baseline; miner competition runs in Lab Arena."""

    def __init__(
        self,
        config: Any,
        *,
        worker_ref: str | None = None,
    ) -> None:
        self.config = config
        self.worker_ref = (
            worker_ref or config.scoring_worker_id or "daily-public-rebenchmark"
        )

    async def run_once(self) -> dict[str, Any]:
        if not self.config.scoring_worker_enabled:
            return {"processed": False, "status": "disabled"}
        if not self.config.production_writes_enabled:
            return {"processed": False, "status": "writes_disabled"}
        if not self.config.public_baseline_rebenchmark_enabled:
            return {"processed": False, "status": "public_baseline_disabled"}
        maintenance = await get_scoring_control()
        if bool(maintenance.get("paused")):
            return {"processed": False, "status": "maintenance_paused"}
        if int(self.config.scoring_worker_index or 0) != 0:
            return {"processed": False, "status": "daily_rebenchmark_non_owner"}

        epoch, _block, _source = await resolve_research_lab_evaluation_epoch(
            self.config.evaluation_epoch
        )
        baseline = await run_daily_public_rebenchmark(
            config=self.config,
            worker_ref=self.worker_ref,
            evaluation_epoch=epoch,
        )
        status = str(baseline.get("status") or "daily_rebenchmark_failed")
        return {
            "processed": status == "completed",
            "status": status,
            "baseline": baseline,
        }

    async def run_forever(self) -> None:
        poll_seconds = max(1, int(self.config.scoring_worker_poll_seconds or 1))
        while True:
            try:
                result = await self.run_once()
                logger.info(
                    "daily_public_rebenchmark_pass worker=%s status=%s",
                    self.worker_ref,
                    result.get("status"),
                )
            except Exception:
                logger.exception(
                    "daily_public_rebenchmark_pass_failed worker=%s",
                    self.worker_ref,
                )
            await asyncio.sleep(poll_seconds)
