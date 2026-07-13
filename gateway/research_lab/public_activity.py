"""Sanitized public projection for Research Lab loop activity."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import re
from typing import Any, Iterable, Mapping, Sequence

from .candidate_diagnostics import (
    NO_BUILDABLE_CANDIDATE_EVENT_TYPE,
    build_candidate_generation_failure_summary,
    public_candidate_generation_failure_summary,
    public_run_summary_from_terminal_event,
)
from .config import ResearchLabGatewayConfig
from .store import (
    canonical_hash,
    create_public_loop_card,
    create_public_loop_card_event,
    public_loop_card_event_ref,
    public_loop_card_id,
    select_all,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)

PUBLIC_ACTIVITY_SCHEMA_VERSION = "1.0"
TOPIC_POLICY_ID = "research_lab_public_activity_topics:v1"
OUTCOME_POLICY_ID = "research_lab_public_activity_outcomes:v3"

# A projection write is small and idempotent (event_ref dedup), so one bounded retry
# recovers transient DB blips without hammering; a second failure escalates loudly so
# a stuck card is visible in ops logs instead of silently freezing (bug 34b).
PROJECTION_RETRY_ATTEMPTS = 1
PROJECTION_RETRY_BACKOFF_SECONDS = 0.5

# Reprojection sweep + list-cap knobs are read straight from the environment so ops can
# flip them without a config-module redeploy. Defaults preserve current behavior except
# the sweep itself, which is on by default because it only reuses the existing
# idempotent projection path (see reproject_stale_public_cards).
PUBLIC_REPROJECTION_SWEEP_ENABLED_ENV = "RESEARCH_LAB_PUBLIC_REPROJECTION_SWEEP_ENABLED"
PUBLIC_REPROJECTION_LOG_ONLY_ENV = "RESEARCH_LAB_PUBLIC_REPROJECTION_LOG_ONLY"
PUBLIC_LOOP_LIST_MAX_CARDS_ENV = "RESEARCH_LAB_PUBLIC_LOOP_LIST_MAX_CARDS"
DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS = 1000
DEFAULT_REPROJECTION_SWEEP_BATCH_SIZE = 25
PUBLIC_LOOP_TICKET_ID_PREFIX_MIN_LENGTH = 8
_FULL_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UUID_PREFIX_RE = re.compile(rf"^[0-9a-fA-F-]{{{PUBLIC_LOOP_TICKET_ID_PREFIX_MIN_LENGTH},36}}$")
_TICKET_CAP_CLOSING_OUTCOME_LABELS = frozenset(
    {
        "completed_no_candidate",
        "expired",
        "failed",
        NO_BUILDABLE_CANDIDATE_EVENT_TYPE,
        "promotion_passed",
        "promoted",
        "scored_no_gain",
        "scored_promising",
    }
)
_TICKET_CAP_CLOSING_OUTCOME_BANDS = frozenset(
    {"expired", "failed", "no_gain", "passed_threshold", "promoted", "small_gain"}
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _normalize_public_loop_ticket_lookup(value: Any) -> str:
    return str(value or "").strip().lower()


def public_loop_ticket_id_matches_lookup(ticket_id: Any, lookup: str) -> bool:
    normalized_ticket_id = _normalize_public_loop_ticket_lookup(ticket_id)
    normalized_lookup = _normalize_public_loop_ticket_lookup(lookup)
    if not normalized_ticket_id or not normalized_lookup:
        return False
    if _FULL_UUID_RE.fullmatch(normalized_lookup):
        return normalized_ticket_id == normalized_lookup
    if not _UUID_PREFIX_RE.fullmatch(normalized_lookup):
        return False
    return normalized_ticket_id.startswith(normalized_lookup)


async def resolve_public_loop_ticket_id(ticket_id_or_prefix: str) -> str | None:
    """Resolve public dashboard full or short ticket IDs before UUID-typed filters."""
    lookup = _normalize_public_loop_ticket_lookup(ticket_id_or_prefix)
    if not lookup:
        return None
    if _FULL_UUID_RE.fullmatch(lookup):
        return lookup
    if not _UUID_PREFIX_RE.fullmatch(lookup):
        return None

    rows = await select_many(
        "research_lab_public_loop_card_current",
        columns="ticket_id,current_last_activity_at,created_at",
        filters=(),
        order_by=(("current_last_activity_at", True), ("created_at", True)),
        limit=_env_positive_int(PUBLIC_LOOP_LIST_MAX_CARDS_ENV, DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS),
    )
    matches: list[str] = []
    seen: set[str] = set()
    for row in rows:
        ticket_id = _normalize_public_loop_ticket_lookup(row.get("ticket_id"))
        if not public_loop_ticket_id_matches_lookup(ticket_id, lookup) or ticket_id in seen:
            continue
        matches.append(ticket_id)
        seen.add(ticket_id)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "research_lab_public_loop_short_id_ambiguous lookup=%s match_count=%s",
            lookup,
            len(matches),
        )
    return None


def public_loop_outcome_closes_ticket(row: Mapping[str, Any] | None) -> bool:
    """Whether a public loop card represents a terminal result for ticket caps."""
    if not row:
        return False
    label = str(row.get("current_outcome_label") or row.get("outcome_label") or "").strip().lower()
    band = str(row.get("current_outcome_band") or row.get("outcome_band") or "").strip().lower()
    return label in _TICKET_CAP_CLOSING_OUTCOME_LABELS or band in _TICKET_CAP_CLOSING_OUTCOME_BANDS

TOPIC_TAGS = (
    "intent_quality",
    "evidence_freshness",
    "query_generation",
    "source_routing",
    "overbroad_matches",
    "role_targeting",
    "geography_targeting",
    "company_size_targeting",
    "scoring_calibration",
    "cost_efficiency",
    "unknown",
)

KEYWORDS: dict[str, tuple[str, ...]] = {
    "intent_quality": (
        "intent",
        "signal",
        "buying",
        "trigger",
        "qualification",
        "relevance",
        "fit",
        "pain",
    ),
    "evidence_freshness": (
        "fresh",
        "freshness",
        "recent",
        "recency",
        "latest",
        "current",
        "stale",
        "dated",
    ),
    "query_generation": (
        "query",
        "queries",
        "search",
        "keyword",
        "prompt",
        "template",
        "variant",
    ),
    "source_routing": (
        "source",
        "routing",
        "provider",
        "scrape",
        "scraper",
        "linkedin",
        "website",
        "web",
    ),
    "overbroad_matches": (
        "overbroad",
        "broad",
        "false positive",
        "noise",
        "noisy",
        "irrelevant",
        "filter",
    ),
    "role_targeting": (
        "role",
        "title",
        "persona",
        "decision maker",
        "decision-maker",
        "seniority",
        "executive",
    ),
    "geography_targeting": (
        "geo",
        "geography",
        "region",
        "country",
        "city",
        "location",
        "territory",
    ),
    "company_size_targeting": (
        "company size",
        "headcount",
        "employee",
        "enterprise",
        "mid-market",
        "smb",
        "startup",
    ),
    "scoring_calibration": (
        "score",
        "scoring",
        "calibrate",
        "calibration",
        "threshold",
        "ranking",
        "rank",
    ),
    "cost_efficiency": (
        "cost",
        "budget",
        "token",
        "tokens",
        "efficient",
        "efficiency",
        "cheap",
        "latency",
    ),
}

SECRET_TEXT_RE = re.compile(
    r"("
    r"sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext|\.dkr\.ecr\.|"
    r"image_digest|private_model_manifest_doc|candidate_patch_manifest|"
    r"proxy[_-]?url|://[^/]+:[^/@]+@"
    r")",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PublicLoopOutcome:
    event_type: str
    outcome_label: str
    outcome_band: str
    candidate_count: int
    scored_candidate_count: int
    best_candidate_public_summary: str
    run_id: str | None
    receipt_id: str | None
    last_activity_at: str
    event_doc: dict[str, Any]


def contains_secret_material(value: Any) -> bool:
    return bool(SECRET_TEXT_RE.search(str(value or "")))


def sanitize_public_text(value: Any, *, max_length: int = 800) -> str:
    text = WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if not text:
        return ""
    if contains_secret_material(text):
        return "[redacted]"
    return text[:max_length]


def normalize_research_area(value: Any) -> str:
    area = WORD_RE.sub("_", str(value or "generalist").strip().lower()).strip("_")
    return area or "generalist"


def topic_tags_from_texts(research_area: Any, *texts: Any) -> list[str]:
    haystack = " ".join(
        part
        for part in [
            normalize_research_area(research_area),
            *(sanitize_public_text(text, max_length=2000).lower() for text in texts),
        ]
        if part
    )
    tags = [
        tag
        for tag in TOPIC_TAGS
        if tag != "unknown" and any(keyword in haystack for keyword in KEYWORDS[tag])
    ]
    return tags or ["unknown"]


def topic_signature_hash(research_area: Any, topic_tags: Sequence[str]) -> str:
    normalized_tags = [tag for tag in TOPIC_TAGS if tag in set(topic_tags)]
    return canonical_hash(
        {
            "policy": TOPIC_POLICY_ID,
            "research_area": normalize_research_area(research_area),
            "topic_tags": normalized_tags or ["unknown"],
        }
    )


async def safe_project_public_loop_activity(
    ticket_id: str,
    *,
    source_ref: str,
    reason: str,
    config: ResearchLabGatewayConfig | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    effective_config = config or ResearchLabGatewayConfig.from_env()
    if not force and not (
        effective_config.api_enabled
        and effective_config.reports_enabled
        and effective_config.public_activity_enabled
    ):
        return None
    last_exc: Exception | None = None
    for attempt in range(1 + max(0, int(PROJECTION_RETRY_ATTEMPTS))):
        try:
            return await project_public_loop_activity(
                ticket_id,
                source_ref=source_ref,
                reason=reason,
                config=effective_config,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < PROJECTION_RETRY_ATTEMPTS:
                logger.warning(
                    "research_lab_public_activity_projection_failed_retrying ticket_id=%s reason=%s attempt=%s error=%s",
                    ticket_id,
                    reason,
                    attempt + 1,
                    str(exc)[:300],
                )
                await asyncio.sleep(PROJECTION_RETRY_BACKOFF_SECONDS)
    # Escalated: the retry also failed, so this card's public status is now stale
    # until the reprojection sweep (or a later lifecycle event) repairs it.
    logger.error(
        "research_lab_public_activity_projection_failed_after_retry ticket_id=%s reason=%s attempts=%s error=%s",
        ticket_id,
        reason,
        PROJECTION_RETRY_ATTEMPTS + 1,
        str(last_exc)[:300],
    )
    return None


async def _fetch_projection_inputs(ticket_id: str) -> dict[str, Any] | None:
    """Fetch the row groups the public outcome is derived from, or None if the
    ticket does not exist. Shared by the projection and the reprojection sweep so
    both always derive from identical inputs."""
    ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not ticket:
        return None
    queue_rows = await select_many(
        "research_loop_run_queue_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    receipt_rows = await select_many(
        "research_loop_receipt_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    score_bundle_rows = await select_many(
        "research_evaluation_score_bundle_current",
        filters=(("ticket_id", ticket_id),),
        order_by=(("current_status_at", True),),
        limit=1000,
    )
    promotion_event_rows = await _promotion_events_for_candidates(candidate_rows)
    latest_queue = _latest_row(queue_rows, "current_status_at")
    auto_loop_filters: tuple[tuple[str, Any], ...] = (("ticket_id", ticket_id),)
    if latest_queue and latest_queue.get("run_id"):
        auto_loop_filters = (
            ("ticket_id", ticket_id),
            ("run_id", str(latest_queue.get("run_id") or "")),
        )
    auto_loop_event_rows = await select_all(
        "research_lab_auto_research_loop_events",
        columns=(
            "event_id,run_id,ticket_id,event_type,loop_status,seq,elapsed_seconds,"
            "provider_usage,cost_ledger,event_doc,created_at"
        ),
        filters=auto_loop_filters,
        order_by=(("seq", False),),
        batch_size=1000,
        max_rows=10000,
        allow_partial=True,
    )
    return {
        "ticket": ticket,
        "queue_rows": queue_rows,
        "receipt_rows": receipt_rows,
        "candidate_rows": candidate_rows,
        "score_bundle_rows": score_bundle_rows,
        "promotion_event_rows": promotion_event_rows,
        "auto_loop_event_rows": auto_loop_event_rows,
    }


async def project_public_loop_activity(
    ticket_id: str,
    *,
    source_ref: str,
    reason: str,
    config: ResearchLabGatewayConfig | None = None,
) -> dict[str, Any]:
    effective_config = config or ResearchLabGatewayConfig.from_env()
    inputs = await _fetch_projection_inputs(ticket_id)
    if inputs is None:
        raise ValueError(f"Research Lab ticket not found: {ticket_id}")
    ticket = inputs["ticket"]
    candidate_rows = inputs["candidate_rows"]

    research_area = normalize_research_area(ticket.get("island"))
    ticket_doc = ticket.get("ticket_doc") if isinstance(ticket.get("ticket_doc"), Mapping) else {}
    focus_summary = sanitize_public_text(ticket_doc.get("brief_public_summary"), max_length=1200)
    candidate_summaries = [
        sanitize_public_text(row.get("redacted_public_summary"), max_length=800)
        for row in candidate_rows
        if row.get("redacted_public_summary")
    ]
    tags = topic_tags_from_texts(research_area, focus_summary, *candidate_summaries)
    signature_hash = topic_signature_hash(research_area, tags)

    card = await create_public_loop_card(
        ticket_id=ticket_id,
        miner_hotkey=str(ticket.get("miner_hotkey") or ""),
        research_area=research_area,
        research_focus_summary=focus_summary,
        topic_tags=tags,
        topic_signature_hash=signature_hash,
        card_doc={
            "schema_version": PUBLIC_ACTIVITY_SCHEMA_VERSION,
            "projection_policy": OUTCOME_POLICY_ID,
            "topic_policy": TOPIC_POLICY_ID,
            "source": "research_lab_public_activity_projection",
        },
    )

    outcome = derive_public_loop_outcome(
        ticket=ticket,
        queue_rows=inputs["queue_rows"],
        receipt_rows=inputs["receipt_rows"],
        candidate_rows=candidate_rows,
        score_bundle_rows=inputs["score_bundle_rows"],
        promotion_event_rows=inputs["promotion_event_rows"],
        auto_loop_event_rows=inputs["auto_loop_event_rows"],
        improvement_threshold_points=effective_config.improvement_threshold_points,
    )
    event_ref = public_loop_card_event_ref(ticket_id, source_ref, outcome.outcome_label, outcome.last_activity_at)
    event = await create_public_loop_card_event(
        event_ref=event_ref,
        card_id=str(card["card_id"]),
        ticket_id=ticket_id,
        run_id=outcome.run_id,
        receipt_id=outcome.receipt_id,
        event_type=outcome.event_type,
        outcome_label=outcome.outcome_label,
        outcome_band=outcome.outcome_band,
        topic_tags=tags,
        topic_signature_hash=signature_hash,
        candidate_count=outcome.candidate_count,
        scored_candidate_count=outcome.scored_candidate_count,
        best_candidate_public_summary=outcome.best_candidate_public_summary,
        last_activity_at=outcome.last_activity_at,
        event_doc={
            **outcome.event_doc,
            "schema_version": PUBLIC_ACTIVITY_SCHEMA_VERSION,
            "source_ref": sanitize_public_text(source_ref, max_length=240),
            "projection_reason": sanitize_public_text(reason, max_length=160),
            "projection_policy": OUTCOME_POLICY_ID,
            "topic_policy": TOPIC_POLICY_ID,
            "topic_tags": tags,
        },
    )
    return {"card": card, "event": event}


# Candidate statuses that are terminal: the candidate can never score without a new
# event moving it back into the queue. A loop whose only candidates are terminal (none
# scored, none still active, queue not running) is over and must surface as a terminal
# `failed` outcome, not the in-progress-looking `candidate_generation_complete` — no
# matter which rejection reason it died with (bug 33d replaced the old 3-reason
# allowlist, which let all-rejected loops keep showing as complete). Recoverable
# rejections (`stale_parent_needs_rescore`) are surfaced earlier as `needs_rescore`.
_TERMINAL_CANDIDATE_STATUSES = frozenset({"failed", "rejected", "tombstoned"})

# Promotion evidence that the candidate actually merged (or merged and is waiting on /
# received its champion reward). Scanned across the FULL promotion history, not just
# the latest event, so a later `champion_reward_pending_uid` (or any bookkeeping event)
# can never regress a merged champion back to `scored_promising` (bug 33c).
_MERGED_PROMOTION_STATUSES = frozenset({"merged", "reward_pending_uid", "reward_created"})
_MERGED_PROMOTION_EVENT_TYPES = frozenset(
    {
        "active_version_created",
        "champion_reward_pending_uid",
        "champion_reward_created",
    }
)

# Post-score side-effect failures (audit bundle, receipt finalize, projection) record
# that everything scoring-relevant succeeded but a bookkeeping step crashed. The
# scoring worker writes them as promotion_status/event_type
# `post_score_side_effect_failed` (legacy rows: status `failed` with
# event_doc.reason/`candidate_status_preserved` markers). They must never change the
# public status of a scored candidate (bug 33b / Chain B item 3).
_POST_SCORE_SIDE_EFFECT_MARKER = "post_score_side_effect_failed"


def _is_post_score_side_effect_event(row: Mapping[str, Any]) -> bool:
    if str(row.get("promotion_status") or "") == _POST_SCORE_SIDE_EFFECT_MARKER:
        return True
    if str(row.get("event_type") or "") == _POST_SCORE_SIDE_EFFECT_MARKER:
        return True
    doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    if str(doc.get("reason") or "") == _POST_SCORE_SIDE_EFFECT_MARKER:
        return True
    return str(doc.get("candidate_status_preserved") or "") == "scored"


def _has_merged_promotion(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(row.get("promotion_status") or "") in _MERGED_PROMOTION_STATUSES
        or str(row.get("event_type") or "") in _MERGED_PROMOTION_EVENT_TYPES
        for row in rows
    )


def _all_candidates_terminal(candidate_rows: Sequence[Mapping[str, Any]]) -> bool:
    if not candidate_rows:
        return False
    return all(
        str(row.get("current_candidate_status") or "") in _TERMINAL_CANDIDATE_STATUSES
        for row in candidate_rows
    )


def _is_stale_parent_tombstone(row: Mapping[str, Any]) -> bool:
    return str(row.get("current_candidate_status") or "") == "rejected" and str(
        row.get("current_reason") or row.get("reason") or ""
    ) in {
        "stale_parent_needs_rescore",
        "stale_parent_rebase_failed",
        "stale_parent_rebase_unavailable",
        "stale_parent_rebased_to_current",
    }


def _has_replacement_candidate(candidate_rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in candidate_rows:
        if _is_stale_parent_tombstone(row):
            continue
        if str(row.get("current_candidate_status") or "") in {"queued", "assigned", "evaluating", "scored"}:
            return True
    return False


def _activity_candidate_rows(candidate_rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    if not _has_replacement_candidate(candidate_rows):
        return candidate_rows
    return [row for row in candidate_rows if not _is_stale_parent_tombstone(row)]


def _is_stale_parent_promotion_tombstone(row: Mapping[str, Any]) -> bool:
    return str(row.get("event_type") or "") == "stale_parent_detected" or str(
        row.get("promotion_status") or ""
    ) == "rebase_required"


def _activity_promotion_rows(
    promotion_event_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    if not _has_replacement_candidate(candidate_rows):
        return promotion_event_rows
    return [row for row in promotion_event_rows if not _is_stale_parent_promotion_tombstone(row)]


def _public_run_summary(
    auto_loop_event_rows: Sequence[Mapping[str, Any]],
    run_id: str | None,
) -> dict[str, Any]:
    """Spend-free run summary from the loop's terminal event, if one exists.

    Zero-candidate and failed loops otherwise surface no public explanation of
    why the paid run stopped; the sanitized terminal projection (stop_reason,
    last stage, stage counts, call/iteration counts) is enum-and-count only.
    """
    terminal_row = None
    failure_classes: list[str] = []
    for row in auto_loop_event_rows:
        row_run_id = str(row.get("run_id") or "")
        if run_id and row_run_id != str(run_id):
            continue
        event_doc = row.get("event_doc")
        event_doc = event_doc if isinstance(event_doc, Mapping) else {}
        failure_class = str(event_doc.get("failure_class") or "")
        if failure_class and failure_class not in failure_classes:
            failure_classes.append(failure_class)
        if str(row.get("event_type") or "") in {"loop_completed", "loop_failed"}:
            if terminal_row is None or int(row.get("seq") or 0) > int(
                terminal_row.get("seq") or 0
            ):
                terminal_row = row
    if terminal_row is None:
        return {}
    terminal_doc = terminal_row.get("event_doc")
    terminal_doc = terminal_doc if isinstance(terminal_doc, Mapping) else {}
    return public_run_summary_from_terminal_event(
        str(terminal_row.get("run_id") or ""),
        str(terminal_row.get("loop_status") or ""),
        terminal_doc,
        failure_classes,
    )


def derive_public_loop_outcome(
    *,
    ticket: Mapping[str, Any],
    queue_rows: Sequence[Mapping[str, Any]],
    receipt_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    score_bundle_rows: Sequence[Mapping[str, Any]],
    promotion_event_rows: Sequence[Mapping[str, Any]],
    auto_loop_event_rows: Sequence[Mapping[str, Any]] = (),
    improvement_threshold_points: float = 1.0,
) -> PublicLoopOutcome:
    candidate_count = len(candidate_rows)
    status_counts = _status_counts(candidate_rows, "current_candidate_status")
    scored_candidate_count = int(status_counts.get("scored", 0))
    latest_queue = _latest_row(queue_rows, "current_status_at")
    latest_receipt = _latest_row(receipt_rows, "current_status_at")
    last_activity_candidate_rows = _activity_candidate_rows(candidate_rows)
    latest_candidate = _latest_row(last_activity_candidate_rows, "current_status_at")
    # Post-score side-effect failures are bookkeeping noise, not promotion outcomes;
    # they must not become the "latest" promotion event that drives the status.
    effective_promotion_rows = [
        row for row in promotion_event_rows if not _is_post_score_side_effect_event(row)
    ]
    latest_promotion = _latest_row(effective_promotion_rows, "created_at")

    run_id = _first_non_empty(
        latest_queue.get("run_id") if latest_queue else None,
        latest_candidate.get("run_id") if latest_candidate else None,
        latest_receipt.get("run_id") if latest_receipt else None,
    )
    receipt_id = _first_non_empty(
        latest_receipt.get("receipt_id") if latest_receipt else None,
        latest_candidate.get("receipt_id") if latest_candidate else None,
    )
    best_bundle = _best_score_bundle(score_bundle_rows)
    best_summary = _best_candidate_summary(candidate_rows, best_bundle)
    last_activity_promotion_rows = _activity_promotion_rows(promotion_event_rows, candidate_rows)
    last_activity_at = _latest_timestamp(
        [ticket],
        queue_rows,
        receipt_rows,
        last_activity_candidate_rows,
        score_bundle_rows,
        last_activity_promotion_rows,
    )

    queue_status = str(latest_queue.get("current_queue_status") or "") if latest_queue else ""
    queue_reason = str(latest_queue.get("current_reason") or "") if latest_queue else ""
    # The capacity-guard park (bug 28) writes `paused/requeue_capacity_conflict_parked`
    # and stays recoverable: the stale-paused reaper requeues the run once loop
    # capacity frees up. For the public projection it is a wait state, so it maps
    # onto the existing allowlisted `queued` outcome instead of falling through
    # the generic paused paths (which surfaced it as `failed` when every candidate
    # was terminal, or `running` via the ticket status).
    queue_parked_waiting = queue_status == "paused" and queue_reason == "requeue_capacity_conflict_parked"

    event_doc = {
        "queue_status": queue_status,
        "queue_reason": queue_reason,
        "receipt_status": str(latest_receipt.get("current_receipt_status") or "") if latest_receipt else "",
        "candidate_status_counts": status_counts,
        "score_bundle_count": len(score_bundle_rows),
        "promotion_event_count": len(promotion_event_rows),
    }
    run_summary = _public_run_summary(auto_loop_event_rows, run_id)
    if run_summary:
        event_doc["run_summary"] = run_summary
    candidate_generation_failure: dict[str, Any] = {}
    if (
        candidate_count == 0
        and queue_status not in {"queued", "started", "running", "paused"}
        and (queue_status == "failed" or auto_loop_event_rows)
    ):
        candidate_generation_failure = build_candidate_generation_failure_summary(
            auto_loop_event_rows,
            queue_reason=queue_reason,
            terminal_error=queue_reason,
            candidate_count=candidate_count,
        )
        public_candidate_generation_failure = public_candidate_generation_failure_summary(
            candidate_generation_failure
        )
        if public_candidate_generation_failure:
            event_doc["candidate_generation_failure"] = public_candidate_generation_failure
    if queue_parked_waiting:
        # Truthful detail for the public card: the run is parked waiting for a
        # free loop slot — neither actively running nor failed.
        event_doc["queue_parked"] = True
        event_doc["queue_parked_detail"] = "requeue parked until loop capacity frees; requeued automatically"

    def _result(event_type: str, outcome_label: str, outcome_band: str) -> PublicLoopOutcome:
        return PublicLoopOutcome(
            event_type,
            outcome_label,
            outcome_band,
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )

    promotion_status = str(latest_promotion.get("promotion_status") or "") if latest_promotion else ""
    promotion_type = str(latest_promotion.get("event_type") or "") if latest_promotion else ""
    # A merge anywhere in the promotion history wins: later bookkeeping events
    # (champion_reward_pending_uid, reward retries) must not regress the champion.
    if _has_merged_promotion(effective_promotion_rows):
        return PublicLoopOutcome(
            "promoted",
            "promoted",
            "promoted",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    # A promotion failure only makes the whole loop `failed` when nothing scored:
    # a genuinely scored candidate whose promotion step failed still surfaces its
    # scored outcome below (the failure stays visible in promotion_event_count).
    if (
        promotion_status == "failed" or promotion_type == "promotion_failed"
    ) and scored_candidate_count == 0:
        return PublicLoopOutcome(
            "failed",
            "failed",
            "failed",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if promotion_status == "passed" or promotion_type == "promotion_passed":
        return PublicLoopOutcome(
            "promotion_passed",
            "promotion_passed",
            "passed_threshold",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )

    ticket_status = str(ticket.get("current_ticket_status") or ticket.get("ticket_status") or "")
    if ticket_status == "expired":
        expired_event_doc = dict(event_doc)
        expired_event_doc["public_status"] = "expired"
        expired_event_doc["public_status_label"] = "Expired"
        expired_event_doc["payment_state"] = "expired"
        expired_event_doc["status_detail"] = "Payment window expired before the loop started."
        if ticket.get("unpaid_expires_at"):
            expired_event_doc["unpaid_expires_at"] = str(ticket.get("unpaid_expires_at"))
        return PublicLoopOutcome(
            "expired",
            "expired",
            "expired",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            expired_event_doc,
        )
    if queue_status in {"paused", "failed"} and _is_credit_block_reason(queue_reason):
        return _result("blocked_for_credit", "blocked_for_credit", "blocked")
    if (
        queue_status == "failed"
        and candidate_count == 0
        and candidate_generation_failure
        and not _is_credit_block_reason(queue_reason)
    ):
        no_candidate_event_doc = dict(event_doc)
        no_candidate_event_doc["public_status"] = NO_BUILDABLE_CANDIDATE_EVENT_TYPE
        no_candidate_event_doc["public_status_label"] = "No buildable candidate"
        return PublicLoopOutcome(
            "no_buildable_candidate",
            "no_buildable_candidate",
            "failed",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            no_candidate_event_doc,
        )
    if (
        _has_candidate_reason(candidate_rows, "stale_parent_needs_rescore")
        and not _has_active_candidate(status_counts)
        and scored_candidate_count == 0
    ):
        # Only surface needs_rescore while nothing else is progressing: once a derived
        # (rebased) candidate is actively scoring or has scored, the stale original —
        # which keeps its rejected/stale_parent_needs_rescore row forever — must not
        # mask the live outcome.
        return _result("needs_rescore", "needs_rescore", "blocked")
    if (
        queue_status not in {"queued", "started", "running"}
        and not queue_parked_waiting
        and _all_candidates_terminal(candidate_rows)
        and scored_candidate_count == 0
    ):
        # The loop is over (no run still active) and every candidate reached a terminal
        # status (failed/rejected/tombstoned) without a score. A fresh re-run, if any,
        # would make the latest queue status active and take precedence over this
        # terminal outcome — and a capacity-parked run has exactly such a re-run
        # pending, so it is excluded above.
        return _result("failed", "failed", "failed")
    if _has_candidate_reason(candidate_rows, "baseline_not_ready") or (
        queue_status == "queued" and queue_reason == "baseline_not_ready"
    ):
        return _result("waiting_for_baseline", "waiting_for_baseline", "pending")

    if (
        queue_status not in {"queued", "started", "running"}
        and not queue_parked_waiting
        and status_counts.get("failed", 0)
        and not _has_active_candidate(status_counts)
        and not scored_candidate_count
    ):
        # A failed sibling never outranks scored history (with any scored candidate
        # the loop reports its scored outcome below) nor an actively re-queued run
        # (which surfaces as queued/running further down).
        return PublicLoopOutcome(
            "failed",
            "failed",
            "failed",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if _has_active_candidate(status_counts):
        # Distinguish "waiting on the private baseline" from genuinely-scoring.
        baseline_waiting = any(
            str(row.get("current_candidate_status") or "") in {"queued", "assigned", "evaluating"}
            and str(row.get("current_reason") or "") == "baseline_not_ready"
            for row in candidate_rows
        )
        if baseline_waiting and scored_candidate_count == 0:
            return _result("waiting_for_baseline", "waiting_for_baseline", "pending")
        return _result("scoring", "scoring", "pending")
    if scored_candidate_count:
        mean_delta, _delta_lcb = _score_bundle_delta(best_bundle)
        if mean_delta >= float(improvement_threshold_points):
            return PublicLoopOutcome(
                "scored",
                "scored_promising",
                "passed_threshold",
                candidate_count,
                scored_candidate_count,
                best_summary,
                run_id,
                receipt_id,
                last_activity_at,
                event_doc,
            )
        if mean_delta > 0:
            return PublicLoopOutcome(
                "scored",
                "scored_promising",
                "small_gain",
                candidate_count,
                scored_candidate_count,
                best_summary,
                run_id,
                receipt_id,
                last_activity_at,
                event_doc,
            )
        return PublicLoopOutcome(
            "scored",
            "scored_no_gain",
            "no_gain",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )

    if queue_status in {"started", "running"}:
        # A re-queued/re-started run is live again: it must not fall through to the
        # completed outcomes below just because earlier candidates exist.
        return _result("running", "running", "pending")
    if queue_status == "queued" or queue_parked_waiting:
        # Capacity-parked runs surface as `queued` (allowlisted) with the parked
        # detail recorded in event_doc; the reaper requeues them for real.
        return _result("queued", "queued", "pending")
    if queue_status == "completed" and candidate_count == 0:
        # Queue completed but produced no candidate: ops should investigate,
        # and the public card must not show a successful candidate generation.
        return _result("completed_no_candidate", "completed_no_candidate", "pending")
    if candidate_count or queue_status == "completed":
        return PublicLoopOutcome(
            "candidate_generation_complete",
            "candidate_generation_complete",
            "pending",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if not queue_rows and not receipt_rows and not candidate_rows:
        # A ticket in `opened` with no run/queue/receipt/candidate never had its
        # loop-start payment: opening a ticket only declares intent; the loop is
        # launched by a separate paid loop-start call that writes the first
        # `paid_loop_queued` queue event. Surface this as its own stage so the
        # dashboard distinguishes "miner has not paid to launch" from a run the
        # platform simply has not picked up yet.
        if ticket_status in {"opened", "probe_created", "funding_pending"}:
            # Emit the canonical lifecycle fields the dashboard consumes
            # (public_status / payment_state) so it renders the "Awaiting payment"
            # stage from the canonical path rather than inferring it from the legacy
            # outcome label. `no_payment` is what maps to Awaiting payment downstream.
            awaiting_event_doc = dict(event_doc)
            awaiting_event_doc["public_status"] = "awaiting_payment"
            awaiting_event_doc["payment_state"] = "no_payment"
            if ticket.get("unpaid_expires_at"):
                awaiting_event_doc["unpaid_expires_at"] = str(ticket.get("unpaid_expires_at"))
            return PublicLoopOutcome(
                "awaiting_payment",
                "awaiting_payment",
                "pending",
                candidate_count,
                scored_candidate_count,
                best_summary,
                run_id,
                receipt_id,
                last_activity_at,
                awaiting_event_doc,
            )
        return PublicLoopOutcome(
            "not_started",
            "not_started",
            "pending",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if queue_status == "failed" or ticket_status in {"cancelled", "failed"}:
        return PublicLoopOutcome(
            "failed",
            "failed",
            "failed",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if queue_status == "started" or ticket_status == "running":
        return PublicLoopOutcome(
            "running",
            "running",
            "pending",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    if queue_status == "queued" or ticket_status == "queued":
        return PublicLoopOutcome(
            "queued",
            "queued",
            "pending",
            candidate_count,
            scored_candidate_count,
            best_summary,
            run_id,
            receipt_id,
            last_activity_at,
            event_doc,
        )
    return PublicLoopOutcome(
        "submitted",
        "submitted",
        "pending",
        candidate_count,
        scored_candidate_count,
        best_summary,
        run_id,
        receipt_id,
        last_activity_at,
        event_doc,
    )


def public_loop_api_item(row: Mapping[str, Any], *, similar_recent_loop_count: int = 0) -> dict[str, Any]:
    topic_tags = row.get("current_topic_tags") or row.get("topic_tags") or []
    if not isinstance(topic_tags, list):
        topic_tags = []
    event_doc = row.get("current_event_doc") if isinstance(row.get("current_event_doc"), Mapping) else {}
    outcome_label = row.get("current_outcome_label") or "submitted"
    outcome_display_label = str(event_doc.get("public_status_label") or "").strip()
    if not outcome_display_label and outcome_label == NO_BUILDABLE_CANDIDATE_EVENT_TYPE:
        outcome_display_label = "No buildable candidate"
    return {
        "card_id": row.get("card_id"),
        "ticket_id": row.get("ticket_id"),
        "run_id": row.get("current_run_id"),
        "receipt_id": row.get("current_receipt_id"),
        "miner_hotkey": row.get("miner_hotkey"),
        "submitted_at": row.get("created_at"),
        "last_activity_at": row.get("current_last_activity_at") or row.get("created_at"),
        "research_area": row.get("research_area"),
        "research_focus_summary": row.get("research_focus_summary"),
        "topic_tags": topic_tags,
        "topic_signature_hash": row.get("current_topic_signature_hash") or row.get("topic_signature_hash"),
        "outcome_label": outcome_label,
        "outcome_display_label": outcome_display_label or outcome_label,
        "outcome_band": row.get("current_outcome_band") or "pending",
        "candidate_count": int(row.get("current_candidate_count") or 0),
        "scored_candidate_count": int(row.get("current_scored_candidate_count") or 0),
        "best_candidate_public_summary": row.get("current_best_candidate_public_summary") or "",
        "similar_recent_loop_count": int(similar_recent_loop_count),
    }


def public_loop_event_api_item(row: Mapping[str, Any]) -> dict[str, Any]:
    event_doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    outcome_label = row.get("outcome_label")
    outcome_display_label = str(event_doc.get("public_status_label") or "").strip()
    if not outcome_display_label and outcome_label == NO_BUILDABLE_CANDIDATE_EVENT_TYPE:
        outcome_display_label = "No buildable candidate"
    return {
        "event_ref": row.get("event_ref"),
        "event_type": row.get("event_type"),
        "outcome_label": outcome_label,
        "outcome_display_label": outcome_display_label or outcome_label,
        "outcome_band": row.get("outcome_band"),
        "run_id": row.get("run_id"),
        "receipt_id": row.get("receipt_id"),
        "candidate_count": int(row.get("candidate_count") or 0),
        "scored_candidate_count": int(row.get("scored_candidate_count") or 0),
        "best_candidate_public_summary": row.get("best_candidate_public_summary") or "",
        "topic_tags": row.get("topic_tags") if isinstance(row.get("topic_tags"), list) else [],
        "topic_signature_hash": row.get("topic_signature_hash"),
        "last_activity_at": row.get("last_activity_at"),
        "created_at": row.get("created_at"),
    }


def topic_group_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = str(row.get("current_topic_signature_hash") or row.get("topic_signature_hash") or "")
        if not signature:
            continue
        group = groups.setdefault(
            signature,
            {
                "topic_signature_hash": signature,
                "topic_tags": row.get("current_topic_tags") or row.get("topic_tags") or [],
                "total": 0,
                "running": 0,
                "completed": 0,
                "scored": 0,
                "promising_or_promoted": 0,
                "no_gain_or_failed": 0,
                "latest_activity_at": None,
            },
        )
        label = str(row.get("current_outcome_label") or "submitted")
        band = str(row.get("current_outcome_band") or "pending")
        group["total"] += 1
        if label in {"queued", "running", "scoring", "waiting_for_baseline"}:
            group["running"] += 1
        if label in {
            "candidate_generation_complete",
            "scored_no_gain",
            "scored_promising",
            "promotion_passed",
            "promoted",
            "failed",
            NO_BUILDABLE_CANDIDATE_EVENT_TYPE,
            "blocked_for_credit",
            "needs_rescore",
        }:
            group["completed"] += 1
        if label in {"scored_no_gain", "scored_promising", "promotion_passed", "promoted"}:
            group["scored"] += 1
        if band in {"small_gain", "passed_threshold", "promoted"}:
            group["promising_or_promoted"] += 1
        if band in {"no_gain", "failed", "blocked"}:
            group["no_gain_or_failed"] += 1
        activity = row.get("current_last_activity_at") or row.get("created_at")
        if activity and (not group["latest_activity_at"] or str(activity) > str(group["latest_activity_at"])):
            group["latest_activity_at"] = activity
    return sorted(groups.values(), key=lambda item: str(item.get("latest_activity_at") or ""), reverse=True)


async def fetch_public_loop_rows(
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    topic: str | None = None,
    research_area: str | None = None,
    since_days: int = 14,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = await select_many(
        "research_lab_public_loop_card_current",
        filters=(),
        order_by=(("current_last_activity_at", True), ("created_at", True)),
        # Behavior-visible knob: cards beyond this cap fall off the public list and
        # topic groups are computed on the truncated set. Default preserves the
        # historical 1000-card cap; raise via env as the card count grows.
        limit=_env_positive_int(PUBLIC_LOOP_LIST_MAX_CARDS_ENV, DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS),
    )
    cutoff = 0.0 if since_days == 0 else datetime.now(timezone.utc).timestamp() - max(0, since_days) * 86400
    filtered = [
        row
        for row in rows
        if _row_matches_filters(
            row,
            status=status,
            topic=topic,
            research_area=research_area,
            cutoff_timestamp=cutoff,
        )
    ]
    filtered.sort(key=lambda row: str(row.get("current_last_activity_at") or row.get("created_at") or ""), reverse=True)
    groups = topic_group_items(filtered)
    signature_counts: dict[str, int] = {}
    for row in filtered:
        signature = str(row.get("current_topic_signature_hash") or row.get("topic_signature_hash") or "")
        if signature:
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
    page = filtered[max(0, offset) : max(0, offset) + min(max(1, limit), 100)]
    items = [
        public_loop_api_item(
            row,
            similar_recent_loop_count=signature_counts.get(
                str(row.get("current_topic_signature_hash") or row.get("topic_signature_hash") or ""),
                0,
            ),
        )
        for row in page
    ]
    return items, groups


async def fetch_public_loop_summary(
    *,
    status: str | None = None,
    topic: str | None = None,
    research_area: str | None = None,
    since_days: int = 14,
) -> dict[str, Any]:
    rows = await select_many(
        "research_lab_public_loop_card_current",
        filters=(),
        order_by=(("current_last_activity_at", True), ("created_at", True)),
        # Same behavior-visible cap as the list path: summary counts are computed
        # over at most this many cards, so both views stay consistent when ops
        # raise the cap as the card count grows.
        limit=_env_positive_int(PUBLIC_LOOP_LIST_MAX_CARDS_ENV, DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS),
    )
    cutoff = 0.0 if since_days == 0 else datetime.now(timezone.utc).timestamp() - max(0, since_days) * 86400
    filtered = [
        row
        for row in rows
        if _row_matches_filters(
            row,
            status=status,
            topic=topic,
            research_area=research_area,
            cutoff_timestamp=cutoff,
        )
    ]
    outcome_counts = Counter(str(row.get("current_outcome_label") or "submitted") for row in filtered)
    band_counts = Counter(str(row.get("current_outcome_band") or "pending") for row in filtered)
    miner_hotkeys = {str(row.get("miner_hotkey") or "") for row in filtered if row.get("miner_hotkey")}
    topic_signatures = {
        str(row.get("current_topic_signature_hash") or row.get("topic_signature_hash") or "")
        for row in filtered
        if row.get("current_topic_signature_hash") or row.get("topic_signature_hash")
    }
    return {
        "total_visible": len(filtered),
        "run_count": len(filtered),
        "miner_count": len(miner_hotkeys),
        "direction_count": len(topic_signatures),
        "candidate_count": sum(int(row.get("current_candidate_count") or 0) for row in filtered),
        "scored_candidate_count": sum(int(row.get("current_scored_candidate_count") or 0) for row in filtered),
        "outcome_label_counts": dict(sorted(outcome_counts.items())),
        "outcome_band_counts": dict(sorted(band_counts.items())),
        "topic_groups": topic_group_items(filtered),
    }


async def fetch_public_loop_detail(ticket_id: str) -> dict[str, Any] | None:
    resolved_ticket_id = await resolve_public_loop_ticket_id(ticket_id)
    if not resolved_ticket_id:
        return None
    row = await select_one(
        "research_lab_public_loop_card_current",
        filters=(("ticket_id", resolved_ticket_id),),
    )
    if not row:
        return None
    card_id = str(row.get("card_id") or public_loop_card_id(resolved_ticket_id))
    events = await select_many(
        "research_lab_public_loop_card_events",
        filters=(("card_id", card_id),),
        order_by=(("seq", False),),
        limit=1000,
    )
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", resolved_ticket_id),),
        order_by=(("created_at", False),),
        limit=1000,
    )
    candidate_summaries = [
        {
            "candidate_id": row.get("candidate_id"),
            "run_id": row.get("run_id"),
            "status": row.get("current_candidate_status"),
            "public_summary": sanitize_public_text(row.get("redacted_public_summary"), max_length=800),
            "created_at": row.get("created_at"),
            "last_status_at": row.get("current_status_at"),
        }
        for row in candidate_rows
        if row.get("redacted_public_summary") and not contains_secret_material(row.get("redacted_public_summary"))
    ]
    return {
        "card": public_loop_api_item(row),
        "candidate_public_summaries": candidate_summaries,
        "events": [public_loop_event_api_item(event) for event in events],
    }


async def reproject_stale_public_cards(
    *,
    config: ResearchLabGatewayConfig | None = None,
    batch_size: int = DEFAULT_REPROJECTION_SWEEP_BATCH_SIZE,
    max_cards: int = DEFAULT_PUBLIC_LOOP_LIST_MAX_CARDS,
    reason: str = "public_reprojection_sweep",
    force: bool = False,
) -> dict[str, Any]:
    """Best-effort sweep that re-projects public cards whose stored status drifted.

    Covers every projection a lifecycle path missed (recovery actions, terminal paths
    that skip projection, the historical awaiting_payment CHECK failures): a card is
    stale when its derived outcome label disagrees with the stored one, or when the
    derived outcome carries canonical lifecycle fields (public_status/payment_state)
    that the stored current event lacks. Reuses the existing idempotent projection
    path (event_ref dedup), so re-running it is safe.

    Gated by RESEARCH_LAB_PUBLIC_REPROJECTION_SWEEP_ENABLED (default true) with a
    RESEARCH_LAB_PUBLIC_REPROJECTION_LOG_ONLY mode (default false) for a log-only
    first rollout. At most ``batch_size`` cards are re-projected per sweep; the rest
    are picked up by later sweeps. Never raises: per-card failures are logged and
    counted.
    """
    result: dict[str, Any] = {
        "action": "reproject-stale-public-cards",
        "enabled": True,
        "log_only": _env_flag(PUBLIC_REPROJECTION_LOG_ONLY_ENV, default=False),
        "cards_checked": 0,
        "stale_found": 0,
        "reprojected": 0,
        "deferred_to_next_sweep": 0,
        "stale": [],
        "failed": [],
    }
    if not _env_flag(PUBLIC_REPROJECTION_SWEEP_ENABLED_ENV, default=True):
        result["enabled"] = False
        return result
    effective_config = config or ResearchLabGatewayConfig.from_env()
    if not force and not (
        effective_config.api_enabled
        and effective_config.reports_enabled
        and effective_config.public_activity_enabled
    ):
        result["enabled"] = False
        result["disabled_reason"] = "public_activity_disabled"
        return result

    cards = await select_many(
        "research_lab_public_loop_card_current",
        filters=(),
        order_by=(("current_last_activity_at", True), ("created_at", True)),
        limit=max(1, int(max_cards)),
    )
    for card in cards:
        ticket_id = str(card.get("ticket_id") or "")
        if not ticket_id:
            continue
        result["cards_checked"] += 1
        try:
            inputs = await _fetch_projection_inputs(ticket_id)
            if inputs is None:
                continue
            outcome = derive_public_loop_outcome(
                ticket=inputs["ticket"],
                queue_rows=inputs["queue_rows"],
                receipt_rows=inputs["receipt_rows"],
                candidate_rows=inputs["candidate_rows"],
                score_bundle_rows=inputs["score_bundle_rows"],
                promotion_event_rows=inputs["promotion_event_rows"],
                auto_loop_event_rows=inputs["auto_loop_event_rows"],
                improvement_threshold_points=effective_config.improvement_threshold_points,
            )
        except Exception as exc:  # noqa: BLE001 - sweep must never fail the worker pass
            result["failed"].append({"ticket_id": ticket_id, "stage": "derive", "error": str(exc)[:200]})
            continue

        stored_label = str(card.get("current_outcome_label") or "")
        stored_doc = card.get("current_event_doc") if isinstance(card.get("current_event_doc"), Mapping) else {}
        # Only flag canonical-field drift the projection can actually repair.
        # The projection writes public_status (it is part of the derived
        # event_doc) but never payment_state — that comes from the payment
        # lifecycle paths.  Requiring payment_state here marked cards stale
        # that reprojection could never satisfy, so every sweep found the
        # same cards again and, with the per-pass budget, spun on them
        # forever while genuinely repairable cards waited behind them.
        canonical_fields_missing = bool(outcome.event_doc.get("public_status")) and not stored_doc.get(
            "public_status"
        )
        if outcome.outcome_label == stored_label and not canonical_fields_missing:
            continue
        result["stale_found"] += 1
        stale_entry = {
            "ticket_id": ticket_id,
            "stored_outcome_label": stored_label,
            "derived_outcome_label": outcome.outcome_label,
            "canonical_fields_missing": canonical_fields_missing,
        }
        result["stale"].append(stale_entry)
        if result["log_only"]:
            logger.info(
                "research_lab_public_reprojection_sweep_stale_card ticket_id=%s stored=%s derived=%s canonical_missing=%s (log-only)",
                ticket_id,
                stored_label or "-",
                outcome.outcome_label,
                canonical_fields_missing,
            )
            continue
        if result["reprojected"] >= max(1, int(batch_size)):
            result["deferred_to_next_sweep"] += 1
            continue
        try:
            # Force the projection like the operator repair path does.  The
            # non-forced path dedups on event_ref and can decline to write,
            # which leaves the card's stored outcome unchanged — the sweep
            # then finds the same cards stale forever and, with the per-pass
            # budget, spins on the oldest ones while newer stale cards never
            # get a turn (observed: stale=94 reprojected=25 deferred=69 on
            # every pass, indefinitely).
            await safe_project_public_loop_activity(
                ticket_id,
                source_ref=f"{reason}:{ticket_id}",
                reason=reason,
                force=True,
            )
            result["reprojected"] += 1
        except Exception as exc:  # noqa: BLE001 - sweep must never fail the worker pass
            result["failed"].append({"ticket_id": ticket_id, "stage": "project", "error": str(exc)[:200]})
            logger.warning(
                "research_lab_public_reprojection_sweep_failed ticket_id=%s error=%s",
                ticket_id,
                str(exc)[:200],
            )
    if result["stale_found"]:
        logger.info(
            "research_lab_public_reprojection_sweep checked=%s stale=%s reprojected=%s deferred=%s failed=%s log_only=%s",
            result["cards_checked"],
            result["stale_found"],
            result["reprojected"],
            result["deferred_to_next_sweep"],
            len(result["failed"]),
            result["log_only"],
        )
    return result


async def _promotion_events_for_candidates(candidate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids = {str(row.get("candidate_id") or "") for row in candidate_rows if row.get("candidate_id")}
    if not candidate_ids:
        return []
    rows = await select_many(
        "research_lab_candidate_promotion_events",
        filters=(("candidate_id", "in", sorted(candidate_ids)),),
        order_by=(("created_at", True),),
        limit=1000,
    )
    return rows


def _status_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(field) or "")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts


def _has_active_candidate(status_counts: Mapping[str, int]) -> bool:
    return any(status_counts.get(status, 0) for status in ("assigned", "evaluating", "queued"))


def _has_candidate_reason(rows: Sequence[Mapping[str, Any]], reason: str) -> bool:
    return any(str(row.get("current_reason") or row.get("reason") or "") == reason for row in rows)


def _is_credit_block_reason(reason: str) -> bool:
    lowered = str(reason or "").lower()
    return any(
        marker in lowered
        for marker in (
            "blocked_for_credit",
            "insufficient_credit",
            "insufficient credits",
            "insufficient balance",
            "payment required",
            "openrouter_credit_blocked",
        )
    )


def _best_score_bundle(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _score_bundle_delta(row)[0])


def _best_candidate_summary(
    candidate_rows: Sequence[Mapping[str, Any]],
    best_bundle: Mapping[str, Any] | None,
) -> str:
    if best_bundle:
        candidate_artifact_hash = str(best_bundle.get("candidate_artifact_hash") or "")
        for row in candidate_rows:
            if str(row.get("candidate_artifact_hash") or "") == candidate_artifact_hash:
                return sanitize_public_text(row.get("redacted_public_summary"), max_length=800)
    for row in candidate_rows:
        summary = sanitize_public_text(row.get("redacted_public_summary"), max_length=800)
        if summary:
            return summary
    return ""


def _score_bundle_delta(row: Mapping[str, Any] | None) -> tuple[float, float]:
    if not row:
        return 0.0, 0.0
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
    gate = doc.get("private_holdout_gate") if isinstance(doc.get("private_holdout_gate"), Mapping) else {}
    if str(gate.get("decision") or "") == "rejected_before_private_holdout":
        return 0.0, 0.0
    daily_delta = gate.get("candidate_delta_vs_daily_baseline")
    if daily_delta is not None:
        return _float(daily_delta), 0.0
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
    return _float(aggregates.get("mean_delta")), _float(aggregates.get("delta_lcb"))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _latest_timestamp(*row_groups: Iterable[Mapping[str, Any]]) -> str:
    values: list[tuple[float, str]] = []
    for rows in row_groups:
        for row in rows:
            for field in ("current_status_at", "created_at", "last_activity_at"):
                value = row.get(field)
                if value:
                    values.append((_parse_timestamp(value), str(value)))
                    break
    if not values:
        return datetime.now(timezone.utc).isoformat()
    return max(values, key=lambda item: item[0])[1]


def _latest_row(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _parse_timestamp(row.get(field)))


def _parse_timestamp(value: Any) -> float:
    if not value:
        return 0.0
    if isinstance(value, datetime):
        return value.timestamp()
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value:
            return str(value)
    return None


def _row_matches_filters(
    row: Mapping[str, Any],
    *,
    status: str | None,
    topic: str | None,
    research_area: str | None,
    cutoff_timestamp: float,
) -> bool:
    if cutoff_timestamp and _parse_timestamp(row.get("current_last_activity_at") or row.get("created_at")) < cutoff_timestamp:
        return False
    if status and str(row.get("current_outcome_label") or "") != status:
        return False
    if research_area and normalize_research_area(row.get("research_area")) != normalize_research_area(research_area):
        return False
    if topic:
        tags = row.get("current_topic_tags") or row.get("topic_tags") or []
        if not isinstance(tags, list) or topic not in tags:
            return False
    return True
