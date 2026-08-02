#!/usr/bin/env python3
"""Verify scoring workers claim only after measured provider preflight passes."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
import gateway.research_lab.scoring_worker as scoring_worker_module  # noqa: E402
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker  # noqa: E402


class ProbeScoringWorker(ResearchLabGatewayScoringWorker):
    def __init__(self, config: ResearchLabGatewayConfig):
        super().__init__(config, worker_ref="readiness-test-scorer")
        self.recover_calls = 0
        self.alert_calls = 0
        self.requeue_calls = 0
        self.claim_calls = 0

    async def _recover_stale_candidate_claims(self) -> int:
        self.recover_calls += 1
        return 0

    async def _claim_next_candidate(self) -> dict[str, Any] | None:
        self.claim_calls += 1
        return None

    async def _alert_stuck_candidates(self) -> None:
        self.alert_calls += 1

    async def _requeue_quarantined_candidates(self) -> int:
        self.requeue_calls += 1
        return 0

    async def _maybe_run_pending_confirmation(self) -> dict[str, Any] | None:
        return None

    async def _candidate_claim_capacity(self) -> dict[str, Any]:
        return {"available": True, "cap_disabled": True}

    async def _run_global_icp_queue_pass(self) -> list[str]:
        raise AssertionError("offline readiness verifier entered global ICP queue")


def _config() -> ResearchLabGatewayConfig:
    return ResearchLabGatewayConfig(
        production_writes_enabled=True,
        evaluation_bundles_enabled=True,
        scoring_worker_enabled=True,
        private_baseline_rebenchmark_enabled=False,
        auto_promotion_enabled=False,
        scoring_worker_max_candidates=1,
        scoring_worker_max_active_claims=0,
        scoring_worker_total_workers=2,
    )


async def _verify() -> None:
    originals = {
        name: getattr(scoring_worker_module, name)
        for name in (
            "get_scoring_maintenance_state",
            "try_acquire_maintenance_lease",
            "MaintenanceLeaseHeartbeat",
            "preflight_gate",
            "apply_preflight_control_result",
        )
    }
    original_global_queue_enabled = (
        scoring_worker_module.global_icp_queue.global_icp_queue_enabled
    )
    preflight_proceed = False
    preflight_worker_indices: list[int] = []
    control_results: list[dict[str, Any]] = []
    heartbeat_events: list[str] = []

    async def _not_paused() -> dict[str, Any]:
        return {"paused": False, "status": "active"}

    async def _lease_acquired(**_kwargs: Any) -> bool:
        return True

    class _LocalHeartbeat:
        def __init__(self, **_kwargs: Any):
            heartbeat_events.append("created")

        async def start(self) -> None:
            heartbeat_events.append("started")

        def ensure_held(self) -> None:
            heartbeat_events.append("checked")

        async def stop(self) -> None:
            heartbeat_events.append("stopped")

    async def _provider_preflight(**kwargs: Any) -> dict[str, Any]:
        worker_index = int(kwargs["worker_index"])
        preflight_worker_indices.append(worker_index)
        return {
            "proceed": preflight_proceed,
            "healthy": preflight_proceed,
            "pause_worthy": not preflight_proceed,
            "disabled": False,
            "verdicts": [
                {
                    "provider": f"measured-test-provider-{worker_index}",
                    "healthy": preflight_proceed,
                    "status": (
                        "healthy"
                        if preflight_proceed
                        else "credential_measurement_unavailable"
                    ),
                }
            ],
        }

    async def _record_control_result(**kwargs: Any) -> dict[str, Any]:
        control_results.append(dict(kwargs["result"]))
        return {"changed": False}

    scoring_worker_module.get_scoring_maintenance_state = _not_paused
    scoring_worker_module.try_acquire_maintenance_lease = _lease_acquired
    scoring_worker_module.MaintenanceLeaseHeartbeat = _LocalHeartbeat
    scoring_worker_module.preflight_gate = _provider_preflight
    scoring_worker_module.apply_preflight_control_result = _record_control_result
    scoring_worker_module.global_icp_queue.global_icp_queue_enabled = lambda: False
    try:
        blocked_worker = ProbeScoringWorker(_config())
        blocked_result = await blocked_worker.run_once()
        assert blocked_result["status"] == "provider_preflight_unhealthy"
        assert blocked_result["preflight"][0]["healthy"] is False
        assert blocked_worker.recover_calls == 1
        assert blocked_worker.alert_calls == 1
        assert blocked_worker.requeue_calls == 0
        assert blocked_worker.claim_calls == 0
        assert preflight_worker_indices == [0, 1]
        assert control_results[-1]["healthy"] is False
        assert heartbeat_events[-1] == "stopped"

        preflight_proceed = True
        preflight_worker_indices.clear()
        ready_worker = ProbeScoringWorker(_config())
        ready_result = await ready_worker.run_once()
        assert ready_result["status"] == "idle"
        assert ready_worker.recover_calls == 1
        assert ready_worker.alert_calls == 1
        assert ready_worker.requeue_calls == 1
        assert ready_worker.claim_calls == 1
        assert preflight_worker_indices == [0, 1]
        assert control_results[-1]["healthy"] is True
        assert heartbeat_events[-1] == "stopped"
    finally:
        for name, value in originals.items():
            setattr(scoring_worker_module, name, value)
        scoring_worker_module.global_icp_queue.global_icp_queue_enabled = (
            original_global_queue_enabled
        )


def main() -> int:
    asyncio.run(_verify())
    print(
        "Research Lab scoring worker readiness verified: measured provider "
        "preflight blocks claims until healthy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
