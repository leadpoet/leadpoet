#!/usr/bin/env python3
"""Verify Research Lab auto-research loop queue-cap enforcement."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
import gateway.research_lab.api as research_lab_api  # noqa: E402


def _config(*, total_workers: int = 5, require_proxy: bool = False) -> ResearchLabGatewayConfig:
    return ResearchLabGatewayConfig(
        hosted_worker_total_workers=total_workers,
        hosted_worker_require_proxy=require_proxy,
    )


async def main() -> int:
    errors: list[str] = []
    original_select_all = research_lab_api.select_all
    original_select_one = research_lab_api.select_one
    original_env = dict(os.environ)

    async def run_case(
        name: str,
        *,
        queue_rows: list[dict[str, Any]],
        ticket_rows: list[dict[str, Any]],
        miner_hotkey: str,
        config: ResearchLabGatewayConfig,
        expected_detail: str | None,
        proxy_count: int = 5,
    ) -> None:
        for key in list(os.environ):
            if key.startswith("RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_"):
                os.environ.pop(key, None)
        for index in range(1, proxy_count + 1):
            os.environ[f"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{index}"] = f"http://proxy-{index}.example:8080"

        async def fake_select_all(table: str, **kwargs: Any) -> list[dict[str, Any]]:
            if table == "research_loop_run_queue_current":
                filters = dict(kwargs.get("filters") or ())
                status = str(filters.get("current_queue_status") or "").lower()
                return [
                    row
                    for row in queue_rows
                    if str(row.get("current_queue_status") or "").lower() == status
                ]
            raise AssertionError(f"unexpected select_all table: {table}")

        async def fake_select_one(table: str, **kwargs: Any) -> dict[str, Any] | None:
            if table == "research_loop_ticket_current":
                filters = dict(kwargs.get("filters") or ())
                ticket_id = str(filters.get("ticket_id") or "")
                for row in ticket_rows:
                    if str(row.get("ticket_id") or "") == ticket_id:
                        return row
                return None
            raise AssertionError(f"unexpected select_one table: {table}")

        research_lab_api.select_all = fake_select_all
        research_lab_api.select_one = fake_select_one
        try:
            await research_lab_api._enforce_autoresearch_loop_capacity(config, miner_hotkey)
            if expected_detail is not None:
                errors.append(f"{name}: expected {expected_detail!r}, got pass")
        except HTTPException as exc:
            if expected_detail is None:
                errors.append(f"{name}: unexpected HTTP {exc.status_code}: {exc.detail}")
            elif exc.status_code != 409 or exc.detail != expected_detail:
                errors.append(f"{name}: expected 409/{expected_detail!r}, got {exc.status_code}/{exc.detail!r}")

    try:
        now = datetime.now(timezone.utc)
        await run_case(
            "empty queue passes",
            queue_rows=[],
            ticket_rows=[],
            miner_hotkey="hotkey-new",
            config=_config(),
            expected_detail=None,
        )
        await run_case(
            "same hotkey blocked",
            queue_rows=[
                {
                    "run_id": "run-1",
                    "ticket_id": "ticket-1",
                    "current_queue_status": "started",
                    "current_status_at": now.isoformat(),
                }
            ],
            ticket_rows=[{"ticket_id": "ticket-1", "miner_hotkey": "hotkey-active"}],
            miner_hotkey="hotkey-active",
            config=_config(),
            expected_detail="autoresearch loop for this hotkey already running",
        )
        await run_case(
            "proxy capacity blocked",
            queue_rows=[
                {
                    "run_id": f"run-{index}",
                    "ticket_id": f"ticket-{index}",
                    "current_queue_status": "queued",
                    "current_status_at": now.isoformat(),
                }
                for index in range(5)
            ],
            ticket_rows=[
                {"ticket_id": f"ticket-{index}", "miner_hotkey": f"hotkey-{index}"}
                for index in range(5)
            ],
            miner_hotkey="hotkey-new",
            config=_config(),
            expected_detail="too many autoresearch loops right now, try again later",
        )
        await run_case(
            "below capacity passes",
            queue_rows=[
                {
                    "run_id": f"run-{index}",
                    "ticket_id": f"ticket-{index}",
                    "current_queue_status": "queued",
                    "current_status_at": now.isoformat(),
                }
                for index in range(4)
            ],
            ticket_rows=[
                {"ticket_id": f"ticket-{index}", "miner_hotkey": f"hotkey-{index}"}
                for index in range(4)
            ],
            miner_hotkey="hotkey-new",
            config=_config(),
            expected_detail=None,
        )
        await run_case(
            "required proxy missing blocks",
            queue_rows=[],
            ticket_rows=[],
            miner_hotkey="hotkey-new",
            config=_config(require_proxy=True),
            expected_detail="too many autoresearch loops right now, try again later",
            proxy_count=0,
        )
        await run_case(
            "stale same hotkey does not block",
            queue_rows=[
                {
                    "run_id": "run-stale",
                    "ticket_id": "ticket-stale",
                    "current_queue_status": "started",
                    "current_status_at": (now - timedelta(hours=3)).isoformat(),
                }
            ],
            ticket_rows=[{"ticket_id": "ticket-stale", "miner_hotkey": "hotkey-active"}],
            miner_hotkey="hotkey-active",
            config=_config(),
            expected_detail=None,
        )
    finally:
        research_lab_api.select_all = original_select_all
        research_lab_api.select_one = original_select_one
        os.environ.clear()
        os.environ.update(original_env)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Research Lab auto-research capacity enforcement verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
