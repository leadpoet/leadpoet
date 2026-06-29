#!/usr/bin/env python3
"""Local contract checks for Research Lab recovery and scoring retry controls."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab import scoring_worker as scoring_mod
from gateway.research_lab import worker as hosted_mod


async def main() -> int:
    await _test_terminal_loop_projection()
    await _test_candidate_claim_skips_retry_cooldowns()
    _test_failure_classification()
    print("Research Lab recovery controls verified")
    return 0


async def _test_terminal_loop_projection() -> None:
    writes: list[dict[str, Any]] = []
    original_select_one = hosted_mod.select_one
    original_create_loop_event = hosted_mod.create_auto_research_loop_event

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        assert table == "research_lab_auto_research_loop_current"
        return {
            "run_id": "run-a",
            "current_loop_status": "running",
            "current_event_type": "source_inspection_resolved",
            "current_event_seq": 2,
            "current_event_hash": "sha256:" + "1" * 64,
        }

    async def fake_create_loop_event(**kwargs: Any) -> dict[str, Any]:
        writes.append(dict(kwargs))
        return {"seq": 3, "anchored_hash": "sha256:" + "2" * 64}

    try:
        hosted_mod.select_one = fake_select_one  # type: ignore[assignment]
        hosted_mod.create_auto_research_loop_event = fake_create_loop_event  # type: ignore[assignment]
        worker = hosted_mod.ResearchLabHostedWorker(ResearchLabGatewayConfig(), worker_ref="test-worker")
        context = hosted_mod.HostedRunContext(
            queue_row={"run_id": "run-a", "ticket_id": "ticket-a"},
            ticket={"ticket_id": "ticket-a", "miner_hotkey": "hotkey", "island": "generalist"},
            payment=None,
            receipt_id="receipt-a",
        )
        await worker._ensure_terminal_loop_projection(
            context,
            event_type="loop_failed",
            loop_status="failed",
            reason="test_failure",
            event_doc={"error": "fixture"},
        )
        assert len(writes) == 1
        assert writes[0]["event_type"] == "loop_failed"
        assert writes[0]["loop_status"] == "failed"
        assert writes[0]["event_doc"]["previous_loop_status"] == "running"
    finally:
        hosted_mod.select_one = original_select_one  # type: ignore[assignment]
        hosted_mod.create_auto_research_loop_event = original_create_loop_event  # type: ignore[assignment]


async def _test_candidate_claim_skips_retry_cooldowns() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "candidate_id": "candidate:" + "1" * 64,
            "run_id": "run-old",
            "ticket_id": "ticket-old",
            "current_candidate_status": "queued",
            "current_reason": "baseline_not_ready",
            "current_status_at": (now - timedelta(seconds=30)).isoformat(),
        },
        {
            "candidate_id": "candidate:" + "2" * 64,
            "run_id": "run-new",
            "ticket_id": "ticket-new",
            "current_candidate_status": "queued",
            "current_reason": "candidate_generated_by_gateway_worker",
            "current_status_at": now.isoformat(),
        },
    ]
    events: list[dict[str, Any]] = []
    assigned_hash = "sha256:" + "3" * 64
    original_select_many = scoring_mod.select_many
    original_select_one = scoring_mod.select_one
    original_create_event = scoring_mod.create_candidate_evaluation_event
    original_project = scoring_mod.safe_project_public_loop_activity

    async def fake_select_many(table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "research_lab_candidate_evaluation_current"
        return [dict(row) for row in rows]

    async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
        candidate_id = kwargs.get("filters", (("", ""),))[0][1]
        assigned = bool(events)
        return {
            "candidate_id": candidate_id,
            "current_candidate_status": "assigned" if assigned else "queued",
            "current_evaluator_ref": "scorer-test" if assigned else None,
            "current_event_hash": assigned_hash if assigned else None,
        }

    async def fake_create_event(**kwargs: Any) -> dict[str, Any]:
        events.append(dict(kwargs))
        return {"anchored_hash": assigned_hash}

    async def fake_project(*args: Any, **kwargs: Any) -> None:
        return None

    try:
        scoring_mod.select_many = fake_select_many  # type: ignore[assignment]
        scoring_mod.select_one = fake_select_one  # type: ignore[assignment]
        scoring_mod.create_candidate_evaluation_event = fake_create_event  # type: ignore[assignment]
        scoring_mod.safe_project_public_loop_activity = fake_project  # type: ignore[assignment]
        config = ResearchLabGatewayConfig(
            scoring_worker_baseline_not_ready_retry_seconds=900,
            scoring_worker_retryable_failure_retry_seconds=300,
        )
        worker = scoring_mod.ResearchLabGatewayScoringWorker(config, worker_ref="scorer-test")
        claimed = await worker._claim_next_candidate()
        assert claimed is not None
        assert claimed["candidate_id"] == "candidate:" + "2" * 64
        assert events[0]["event_type"] == "assigned"
    finally:
        scoring_mod.select_many = original_select_many  # type: ignore[assignment]
        scoring_mod.select_one = original_select_one  # type: ignore[assignment]
        scoring_mod.create_candidate_evaluation_event = original_create_event  # type: ignore[assignment]
        scoring_mod.safe_project_public_loop_activity = original_project  # type: ignore[assignment]


def _test_failure_classification() -> None:
    assert scoring_mod._candidate_scoring_failure_class(scoring_mod.CandidateBaselineNotReady("missing")) == (
        "baseline_not_ready",
        True,
    )
    assert scoring_mod._candidate_scoring_failure_class(TimeoutError("adapter timed out")) == (
        "adapter_timeout",
        True,
    )
    assert scoring_mod._candidate_scoring_failure_class(RuntimeError("ScrapingDog status=500")) == (
        "provider_http_5xx",
        True,
    )
    assert scoring_mod._candidate_scoring_failure_class(RuntimeError("plain coding bug")) == (
        "candidate_scoring_error",
        False,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
