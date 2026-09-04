from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker


class PublicBaselineWorkerCutoverTests(unittest.IsolatedAsyncioTestCase):
    def _worker(self, *, index: int) -> ResearchLabGatewayScoringWorker:
        config = ResearchLabGatewayConfig(
            scoring_worker_enabled=True,
            production_writes_enabled=True,
            evaluation_bundles_enabled=True,
            public_baseline_rebenchmark_enabled=True,
            scoring_worker_index=index,
            scoring_worker_total_workers=2,
        )
        return ResearchLabGatewayScoringWorker(config, worker_ref=f"worker-{index}")

    async def test_owner_calls_only_public_daily_rebenchmark(self):
        worker = self._worker(index=0)
        worker._resolve_evaluation_epoch = AsyncMock(return_value=42)
        worker._run_lease_held_recovery_and_preflight = AsyncMock(
            side_effect=AssertionError("old provider preflight was called")
        )
        worker._run_private_baseline_contained = AsyncMock(
            side_effect=AssertionError("old model baseline was called")
        )
        expected = {
            "status": "completed",
            "baseline_run_id": "run-1",
            "completed_icp_count": 20,
        }
        with patch(
            "gateway.research_lab.scoring_worker.get_scoring_maintenance_state",
            new=AsyncMock(return_value={"paused": False}),
        ), patch(
            "gateway.research_lab.daily_rebenchmark.run_daily_public_rebenchmark",
            new=AsyncMock(return_value=expected),
        ) as run_public:
            result = await worker.run_once()

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["processed"])
        run_public.assert_awaited_once_with(
            config=worker.config,
            worker_ref="worker-0",
            evaluation_epoch=42,
        )
        worker._run_lease_held_recovery_and_preflight.assert_not_awaited()
        worker._run_private_baseline_contained.assert_not_awaited()

    async def test_non_owner_does_no_provider_or_model_work(self):
        worker = self._worker(index=1)
        worker._resolve_evaluation_epoch = AsyncMock(
            side_effect=AssertionError("epoch lookup was called")
        )
        with patch(
            "gateway.research_lab.scoring_worker.get_scoring_maintenance_state",
            new=AsyncMock(return_value={"paused": False}),
        ):
            result = await worker.run_once()

        self.assertEqual(result["status"], "daily_rebenchmark_non_owner")
        self.assertFalse(result["processed"])
        worker._resolve_evaluation_epoch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
