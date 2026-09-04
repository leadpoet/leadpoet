"""Read the existing operator pause for public-baseline scoring."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping


logger = logging.getLogger(__name__)
SCORING_CONTROL_KEY = "scoring_maintenance"


async def get_scoring_control() -> dict[str, Any]:
    from gateway.db.client import get_write_client

    def _call() -> Any:
        return (
            get_write_client()
            .table("research_lab_gateway_control_current")
            .select("control_key,current_control_status,current_reason,current_status_at")
            .eq("control_key", SCORING_CONTROL_KEY)
            .limit(1)
            .execute()
        )

    try:
        response = await asyncio.to_thread(_call)
        rows = getattr(response, "data", None) or []
    except Exception as exc:
        logger.warning("public_baseline_scoring_control_unavailable type=%s", type(exc).__name__)
        return {"paused": True, "status": "unavailable_fail_closed"}
    row = rows[0] if rows and isinstance(rows[0], Mapping) else {}
    status = str(row.get("current_control_status") or "inactive")
    return {
        "paused": status == "active",
        "status": status,
        "reason": row.get("current_reason"),
        "status_at": row.get("current_status_at"),
    }
