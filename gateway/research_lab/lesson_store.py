"""§9.5 lesson store: cross-run lesson retrieval over reflection loop events.

Lessons ARE the ``reflection_recorded`` events the code-edit lane emits
(§9.1 item 4 in code_loop_engine.py) — there is no new table or migration.
This module reads recent reflection events from
``research_lab_auto_research_loop_events``, filters them by lane/component
match, applies staleness demotion (lessons whose ``champion_base`` no longer
matches the active parent artifact hash are labeled ``stale_basis`` and rank
lower — mirroring ``engine_v1.mark_lesson_staleness`` semantics), and returns
compact lesson dicts for the planner + draft prompt context.

Retrieval is flag-gated OFF by default (``RESEARCH_LAB_LESSON_RETRIEVAL_ENABLED``,
§8.3 staged-enable discipline) and every caller treats it as best-effort: a
retrieval failure must never fail a paid run.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Sequence

from gateway.research_lab.code_loop_engine import _reflection_safe_text

logger = logging.getLogger(__name__)

LESSON_RETRIEVAL_ENABLED_ENV = "RESEARCH_LAB_LESSON_RETRIEVAL_ENABLED"

LOOP_EVENTS_TABLE = "research_lab_auto_research_loop_events"
REFLECTION_EVENT_TYPE = "reflection_recorded"

DEFAULT_LESSON_LIMIT = 5
DEFAULT_SCAN_LIMIT = 100
# ~1500 tokens at ~4 chars/token for the injected prompt section.
DEFAULT_LESSON_CONTEXT_MAX_CHARS = 6000

_LESSON_TEXT_FIELDS = ("worked", "failed", "why", "next_question")


def lesson_retrieval_enabled() -> bool:
    """§8.3 staged enable: lesson retrieval into prompts is OFF by default."""
    return os.environ.get(LESSON_RETRIEVAL_ENABLED_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class GatewayLessonStore:
    """Thin async facade over gateway.research_lab.store's select helpers.

    Exists so tests can inject fakes; retrieval only uses this one query shape.
    """

    async def select_many(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Any,
        order_by: Any = (),
        limit: int = 100,
    ):
        from gateway.research_lab import store

        return await store.select_many(
            table, columns=columns, filters=filters, order_by=order_by, limit=limit
        )


def _compact_lesson_from_event_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Parse one reflection_recorded loop-event row into a compact lesson dict."""
    doc = row.get("event_doc")
    if not isinstance(doc, Mapping):
        return None
    reflection = doc.get("reflection")
    if not isinstance(reflection, Mapping):
        return None
    lesson: dict[str, Any] = {
        "lesson_id": str(reflection.get("lesson_id") or "")[:120],
        "node_id": str(row.get("node_id") or reflection.get("node_id") or "")[:120],
        "recorded_at": str(row.get("created_at") or "")[:64],
        "lane": str(doc.get("lane") or "")[:80],
        "outcome": str(doc.get("outcome") or "")[:80],
        "component": str(reflection.get("component") or "")[:128],
        "target_files": [
            str(path)[:240]
            for path in (doc.get("target_files") or [])
            if isinstance(path, str)
        ][:10],
        "champion_base": str(reflection.get("champion_base") or "")[:120],
        "eval_version": str(reflection.get("eval_version") or "")[:120],
        "stale_basis": bool(reflection.get("stale_basis")),
    }
    for field in _LESSON_TEXT_FIELDS:
        # Emission already sanitizes; re-sanitize on read as defense-in-depth for
        # historical/foreign rows so a poisoned lesson cannot re-enter a prompt
        # (or, via the checkpoint/trajectory path, a corpus record).
        lesson[field] = _reflection_safe_text(str(reflection.get(field) or ""))
    if not any(lesson[field] for field in _LESSON_TEXT_FIELDS):
        return None
    return lesson


def _lesson_matches(
    lesson: Mapping[str, Any],
    *,
    lane: str | None,
    components: Sequence[str],
) -> bool:
    if lane:
        lesson_lane = str(lesson.get("lane") or "")
        if lesson_lane and lesson_lane != str(lane):
            return False
    if components:
        wanted = {str(item) for item in components if item}
        if wanted:
            have = {str(lesson.get("component") or "")} | {
                str(path) for path in (lesson.get("target_files") or [])
            }
            have.discard("")
            if not (wanted & have):
                return False
    return True


async def fetch_recent_lessons(
    *,
    lane: str | None = None,
    components: Sequence[str] = (),
    active_parent_hash: str = "",
    limit: int = DEFAULT_LESSON_LIMIT,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    store: Any | None = None,
) -> list[dict[str, Any]]:
    """Top-k recent lessons by lane/component match with staleness demotion.

    Reads recent ``reflection_recorded`` events (newest first), filters by
    lane/component when given, and marks lessons whose ``champion_base`` is not
    the active parent artifact hash as ``stale_basis`` — stale lessons rank
    after fresh ones but are not dropped (``mark_lesson_staleness`` semantics:
    demote, do not delete).
    """
    store = store or GatewayLessonStore()
    limit = max(1, int(limit))
    rows = await store.select_many(
        LOOP_EVENTS_TABLE,
        columns="event_id,run_id,node_id,event_type,event_doc,created_at",
        filters=[("event_type", REFLECTION_EVENT_TYPE)],
        order_by=[("created_at", True)],
        limit=max(limit, int(scan_limit)),
    )
    lessons: list[dict[str, Any]] = []
    seen_lesson_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lesson = _compact_lesson_from_event_row(row)
        if lesson is None:
            continue
        if lesson["lesson_id"] and lesson["lesson_id"] in seen_lesson_ids:
            continue
        if not _lesson_matches(lesson, lane=lane, components=components):
            continue
        if (
            active_parent_hash
            and lesson["champion_base"]
            and lesson["champion_base"] != str(active_parent_hash)
        ):
            lesson["stale_basis"] = True
        if lesson["lesson_id"]:
            seen_lesson_ids.add(lesson["lesson_id"])
        lessons.append(lesson)
    # Rows arrive newest-first; the stable sort keeps recency order within the
    # fresh and stale groups while demoting stale-basis lessons.
    lessons.sort(key=lambda lesson: bool(lesson.get("stale_basis")))
    kept = lessons[:limit]
    await _hydrate_score_enrichments(kept, store=store)
    return kept


async def _hydrate_score_enrichments(
    lessons: Sequence[dict[str, Any]], *, store: Any | None = None
) -> None:
    """Phase-5 score-aware lessons: merge realized deltas onto lessons in place.

    Reflection events are immutable (recorded pre-score); the backfill hook
    writes append-only calibration rows keyed by node_id instead, and this
    read-time join surfaces them as ``realized_delta`` / ``scored_outcome``.
    Strictly best-effort: with the backfill flag off, the calibration table
    absent (scripts/71 unapplied), or no rows yet, lessons stay score-blind.
    """
    node_ids = [str(lesson.get("node_id") or "") for lesson in lessons]
    if not any(node_ids):
        return
    try:
        from gateway.research_lab.score_backfill import (
            fetch_score_enrichments_by_node,
            score_backfill_enabled,
        )

        if not score_backfill_enabled():
            return
        # The lesson store's select facade and the calibration store share the
        # gateway store helpers; pass an explicit store through only when the
        # caller injected one that also serves the calibration table (tests).
        enrichments = await fetch_score_enrichments_by_node(node_ids, store=store)
    except Exception as exc:
        logger.debug("research_lab_lesson_score_enrichment_skipped error=%s", str(exc)[:200])
        return
    for lesson in lessons:
        enrichment = enrichments.get(str(lesson.get("node_id") or ""))
        if not enrichment:
            continue
        if enrichment.get("realized_delta") is not None:
            lesson["realized_delta"] = enrichment["realized_delta"]
        if enrichment.get("realized_delta_lcb") is not None:
            lesson["realized_delta_lcb"] = enrichment["realized_delta_lcb"]
        if enrichment.get("predicted_delta") is not None:
            lesson["predicted_delta"] = enrichment["predicted_delta"]
        if enrichment.get("scored_outcome"):
            lesson["scored_outcome"] = enrichment["scored_outcome"]


async def build_lesson_prompt_context(
    *,
    lane: str | None = None,
    components: Sequence[str] = (),
    active_parent_hash: str = "",
    limit: int = DEFAULT_LESSON_LIMIT,
    max_chars: int = DEFAULT_LESSON_CONTEXT_MAX_CHARS,
    store: Any | None = None,
    lessons: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Compact ``retrieved_lessons`` doc for the planner + draft prompt context.

    Returns ``None`` when there is nothing to inject. The serialized doc is
    capped at ``max_chars`` (~1500 tokens) by dropping the lowest-ranked
    lessons first.
    """
    if lessons is None:
        fetched = await fetch_recent_lessons(
            lane=lane,
            components=components,
            active_parent_hash=active_parent_hash,
            limit=limit,
            store=store,
        )
    else:
        fetched = [dict(item) for item in lessons if isinstance(item, Mapping)][
            : max(1, int(limit))
        ]
    if not fetched:
        return None

    def _doc(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "note": (
                "Lessons distilled from earlier runs' recorded reflections. Address the "
                "recorded failures instead of repeating them; lessons marked stale_basis "
                "reference a superseded parent artifact and carry less weight."
            ),
            "active_parent_hash": str(active_parent_hash or ""),
            "lesson_count": len(items),
            "lessons": [dict(item) for item in items],
        }

    kept = list(fetched)
    while len(kept) > 1 and len(json.dumps(_doc(kept), default=str)) > max(1, int(max_chars)):
        kept.pop()
    return _doc(kept)
