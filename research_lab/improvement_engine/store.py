"""Supabase-backed storage helpers for Improvement Engine state."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from gateway.research_lab.store import insert_row, select_many, select_one

from .models import EngineIssue


async def fetch_recent_rows(
    table: str,
    *,
    columns: str = "*",
    filters: Iterable[tuple[str, Any]] = (),
    order_by: Iterable[tuple[str, bool]] = (("created_at", True),),
    limit: int = 1000,
) -> list[dict[str, Any]]:
    return await select_many(table, columns=columns, filters=filters, order_by=order_by, limit=limit)


async def get_issue(issue_key: str) -> dict[str, Any] | None:
    return await select_one("engine_issues", filters=(("issue_key", issue_key),))


async def persist_issue(issue: EngineIssue, *, dry_run: bool = True) -> dict[str, Any]:
    row = issue.to_row()
    existing = None if dry_run else await get_issue(issue.issue_key)
    if dry_run:
        return {"dry_run": True, **row}
    if existing:
        # Avoid UPDATE because issue history should remain append-auditable in
        # events. The current row is immutable enough for v1; recurrence gets
        # an event row instead of mutating existing issue evidence.
        await insert_row(
            "engine_issue_events",
            {
                "issue_id": existing["id"],
                "event_type": "recurrence_detected",
                "event_doc": row,
            },
        )
        return dict(existing)
    inserted = await insert_row("engine_issues", row)
    await insert_row(
        "engine_issue_events",
        {
            "issue_id": inserted["id"],
            "event_type": "opened",
            "event_doc": row,
        },
    )
    return inserted


def persist_issue_sync(issue: EngineIssue, *, dry_run: bool = True) -> dict[str, Any]:
    return asyncio.run(persist_issue(issue, dry_run=dry_run))
