#!/usr/bin/env python3
"""Backfill sanitized Research Lab public loop activity cards.

Dry-run is the default. Pass --write only after script 40 is applied.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.public_activity import (  # noqa: E402
    contains_secret_material,
    derive_public_loop_outcome,
    project_public_loop_activity,
    sanitize_public_text,
    topic_signature_hash,
    topic_tags_from_texts,
)
from gateway.research_lab.store import select_all, select_many  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Insert cards/events into Supabase")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum tickets to inspect")
    args = parser.parse_args()

    tickets = await select_many(
        "research_loop_ticket_current",
        filters=(),
        order_by=(("created_at", True),),
        limit=max(1, min(args.limit, 1000)),
    )
    config = ResearchLabGatewayConfig.from_env()
    inspected = 0
    unsafe = 0
    written = 0

    for ticket in tickets:
        inspected += 1
        ticket_id = str(ticket["ticket_id"])
        if args.write:
            await project_public_loop_activity(
                ticket_id,
                source_ref="backfill_research_lab_public_loop_activity:v1",
                reason="backfill",
                config=config,
            )
            written += 1
            continue

        preview = await _dry_run_preview(ticket)
        if contains_secret_material(preview):
            unsafe += 1
            print(f"UNSAFE ticket_id={ticket_id}")

    mode = "write" if args.write else "dry-run"
    print(
        "Research Lab public activity backfill "
        f"{mode}: inspected={inspected} written={written} unsafe_previews={unsafe}"
    )
    if not args.write:
        print("No rows were written. Re-run with --write after script 40 is applied.")
    return 1 if unsafe else 0


async def _dry_run_preview(ticket: Mapping[str, Any]) -> dict[str, Any]:
    ticket_id = str(ticket["ticket_id"])
    queue_rows = await select_many("research_loop_run_queue_current", filters=(("ticket_id", ticket_id),), limit=1000)
    receipt_rows = await select_many("research_loop_receipt_current", filters=(("ticket_id", ticket_id),), limit=1000)
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", ticket_id),),
        limit=1000,
    )
    score_bundle_rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("ticket_id", ticket_id),),
        limit=1000,
    )
    auto_loop_event_rows = await select_all(
        "research_lab_auto_research_loop_events",
        columns=(
            "event_id,run_id,ticket_id,event_type,loop_status,seq,elapsed_seconds,"
            "provider_usage,cost_ledger,event_doc,created_at"
        ),
        filters=(("ticket_id", ticket_id),),
        order_by=(("seq", False),),
        batch_size=1000,
        max_rows=10000,
        allow_partial=True,
    )
    promotion_rows = await _promotion_events(candidate_rows)
    ticket_doc = ticket.get("ticket_doc") if isinstance(ticket.get("ticket_doc"), Mapping) else {}
    research_area = str(ticket.get("island") or "generalist")
    focus_summary = sanitize_public_text(ticket_doc.get("brief_public_summary"), max_length=1200)
    candidate_summaries = [
        sanitize_public_text(row.get("redacted_public_summary"), max_length=800)
        for row in candidate_rows
        if row.get("redacted_public_summary")
    ]
    tags = topic_tags_from_texts(research_area, focus_summary, *candidate_summaries)
    outcome = derive_public_loop_outcome(
        ticket=ticket,
        queue_rows=queue_rows,
        receipt_rows=receipt_rows,
        candidate_rows=candidate_rows,
        score_bundle_rows=score_bundle_rows,
        promotion_event_rows=promotion_rows,
        auto_loop_event_rows=auto_loop_event_rows,
    )
    return {
        "ticket_id": ticket_id,
        "miner_hotkey": ticket.get("miner_hotkey"),
        "research_area": research_area,
        "research_focus_summary": focus_summary,
        "topic_tags": tags,
        "topic_signature_hash": topic_signature_hash(research_area, tags),
        "outcome_label": outcome.outcome_label,
        "outcome_band": outcome.outcome_band,
        "candidate_count": outcome.candidate_count,
        "scored_candidate_count": outcome.scored_candidate_count,
        "best_candidate_public_summary": outcome.best_candidate_public_summary,
    }


async def _promotion_events(candidate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids = {str(row.get("candidate_id") or "") for row in candidate_rows if row.get("candidate_id")}
    if not candidate_ids:
        return []
    rows = await select_many("research_lab_candidate_promotion_events", filters=(), limit=1000)
    return [row for row in rows if str(row.get("candidate_id") or "") in candidate_ids]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
