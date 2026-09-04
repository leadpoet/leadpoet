from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.daily_worker import DailyPublicRebenchmarkWorker
from gateway.research_lab.worker_autostart import (
    ResearchLabWorkerAutoStartPlan,
    ResearchLabWorkerFleetPlan,
    ResearchLabWorkerStartupError,
    ResearchLabWorkerSupervisor,
    build_research_lab_worker_autostart_plan,
)


class PublicBaselineWorkerCutoverTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _fleet(kind: str, count: int) -> ResearchLabWorkerFleetPlan:
        return ResearchLabWorkerFleetPlan(
            kind=kind,
            worker_count=count,
            worker_prefix=kind,
            log_level="INFO",
            proxy_refs=(),
            enabled=True,
        )

    def _worker(self, *, index: int) -> DailyPublicRebenchmarkWorker:
        config = ResearchLabGatewayConfig(
            scoring_worker_enabled=True,
            production_writes_enabled=True,
            evaluation_bundles_enabled=True,
            public_baseline_rebenchmark_enabled=True,
            scoring_worker_index=index,
            scoring_worker_total_workers=2,
        )
        return DailyPublicRebenchmarkWorker(config, worker_ref=f"worker-{index}")

    async def test_owner_calls_only_public_daily_rebenchmark(self):
        worker = self._worker(index=0)
        expected = {
            "status": "completed",
            "baseline_run_id": "run-1",
            "completed_icp_count": 20,
        }
        with patch(
            "gateway.research_lab.daily_worker.get_scoring_control",
            new=AsyncMock(return_value={"paused": False}),
        ), patch(
            "gateway.research_lab.daily_worker.resolve_research_lab_evaluation_epoch",
            new=AsyncMock(return_value=(42, 100, "test")),
        ), patch(
            "gateway.research_lab.daily_worker.run_daily_public_rebenchmark",
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

    async def test_non_owner_does_no_provider_or_model_work(self):
        worker = self._worker(index=1)
        with patch(
            "gateway.research_lab.daily_worker.get_scoring_control",
            new=AsyncMock(return_value={"paused": False}),
        ), patch(
            "gateway.research_lab.daily_worker.resolve_research_lab_evaluation_epoch",
            new=AsyncMock(side_effect=AssertionError("epoch lookup was called")),
        ) as resolve_epoch:
            result = await worker.run_once()

        self.assertEqual(result["status"], "daily_rebenchmark_non_owner")
        self.assertFalse(result["processed"])
        resolve_epoch.assert_not_awaited()

    async def test_supervisor_starts_only_daily_scoring_fleet(self):
        plan = ResearchLabWorkerAutoStartPlan(
            auto_start_enabled=True,
            hosted=self._fleet("hosted", 3),
            scoring=self._fleet("scoring", 2),
        )
        supervisor = ResearchLabWorkerSupervisor(plan)
        started: list[str] = []

        async def record_start(fleet, *, fleet_deadline):
            del fleet_deadline
            started.append(fleet.kind)

        with patch.dict("os.environ", {"GATEWAY_TEE_TOPOLOGY_MODE": "component"}), patch.object(
            supervisor,
            "_start_fleet_without_blocking_event_loop",
            side_effect=record_start,
        ):
            await supervisor.start_without_blocking_event_loop()

        self.assertEqual(started, ["scoring"])

    def test_hosted_worker_cannot_be_spawned_directly(self):
        plan = ResearchLabWorkerAutoStartPlan(
            auto_start_enabled=True,
            hosted=self._fleet("hosted", 1),
            scoring=self._fleet("scoring", 1),
        )
        supervisor = ResearchLabWorkerSupervisor(plan)

        with self.assertRaisesRegex(
            ResearchLabWorkerStartupError,
            "hosted Research Lab workers are retired",
        ):
            supervisor._spawn_child(plan.hosted, 0)

    def test_scoring_autostart_uses_public_baseline_gate(self):
        plan = build_research_lab_worker_autostart_plan(
            {
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "1",
                "RESEARCH_LAB_PUBLIC_BASELINE_REBENCHMARK_ENABLED": "true",
                "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "false",
            }
        )
        self.assertTrue(plan.scoring.enabled)
        self.assertEqual(plan.scoring.worker_count, 1)

        disabled = build_research_lab_worker_autostart_plan(
            {
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "1",
                "RESEARCH_LAB_PUBLIC_BASELINE_REBENCHMARK_ENABLED": "false",
                "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
            }
        )
        self.assertFalse(disabled.scoring.enabled)
        self.assertEqual(disabled.scoring.reason, "public_baseline_disabled")


if __name__ == "__main__":
    unittest.main()
