"""Private Research Lab failure-funnel report loader.

This module stays independent of the large API/auth import graph so the
read-only reporting contract can be tested without initializing the gateway.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .store import call_rpc


logger = logging.getLogger(__name__)


def missing_failure_funnel(
    ticket_id: str,
    candidate_id: str | None,
    *,
    report_available: bool,
) -> dict[str, Any]:
    """Return an explicit no-data state; never turn missing telemetry into zero."""
    return {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": ticket_id,
        "candidate_id": candidate_id,
        "stages": [],
        "rejections": [],
        "model_revisions": [],
        "telemetry": {
            "status": "missing",
            "report_available": report_available,
        },
    }


async def build_ticket_failure_funnel(
    ticket_id: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Load the service-only aggregate without making diagnostics brittle.

    The report migration can be rolled out before or after the gateway. A
    missing function or temporary reporting read failure therefore degrades
    only this optional panel and never hides the existing loop diagnostics.
    """
    try:
        report = await call_rpc(
            "get_research_lab_failure_funnel",
            {
                "p_ticket_id": ticket_id,
                "p_candidate_id": candidate_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 - optional reporting projection
        logger.warning(
            "research_lab_failure_funnel_unavailable ticket_id=%s error_type=%s",
            ticket_id,
            type(exc).__name__,
        )
        return missing_failure_funnel(ticket_id, candidate_id, report_available=False)
    if isinstance(report, list) and len(report) == 1 and isinstance(report[0], Mapping):
        report = report[0]
    if (
        not isinstance(report, Mapping)
        or not isinstance(report.get("telemetry"), Mapping)
        or report["telemetry"].get("status") not in {"complete", "partial", "missing"}
    ):
        return missing_failure_funnel(ticket_id, candidate_id, report_available=True)
    return dict(report)
