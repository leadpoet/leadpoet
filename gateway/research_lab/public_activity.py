"""Sanitized public projection for Research Lab loop activity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
from typing import Any, Iterable, Mapping, Sequence

from .config import ResearchLabGatewayConfig
from .store import (
    canonical_hash,
    create_public_loop_card,
    create_public_loop_card_event,
    public_loop_card_event_ref,
    public_loop_card_id,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)

PUBLIC_ACTIVITY_SCHEMA_VERSION = "1.0"
TOPIC_POLICY_ID = "research_lab_public_activity_topics:v1"
OUTCOME_POLICY_ID = "research_lab_public_activity_outcomes:v1"

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
    try:
        return await project_public_loop_activity(
            ticket_id,
            source_ref=source_ref,
            reason=reason,
            config=effective_config,
        )
    except Exception as exc:
        logger.warning(
            "research_lab_public_activity_projection_failed ticket_id=%s reason=%s error=%s",
            ticket_id,
            reason,
            str(exc)[:300],
        )
        return None


async def project_public_loop_activity(
    ticket_id: str,
    *,
    source_ref: str,
    reason: str,
    config: ResearchLabGatewayConfig | None = None,
) -> dict[str, Any]:
    effective_config = config or ResearchLabGatewayConfig.from_env()
    ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not ticket:
        raise ValueError(f"Research Lab ticket not found: {ticket_id}")

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
        queue_rows=queue_rows,
        receipt_rows=receipt_rows,
        candidate_rows=candidate_rows,
        score_bundle_rows=score_bundle_rows,
        promotion_event_rows=promotion_event_rows,
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


# Candidate rejection reasons that are terminal and not recoverable by any operator
# or worker retry: the loop is done and produced no scoreable result. A loop whose
# only candidates carry one of these (none scored, none still active) must surface as
# a terminal `failed` outcome, not the in-progress-looking `candidate_generation_complete`.
# `stale_parent_rebase_failed`: the live model advanced and the candidate could not be
# re-fit to it (e.g. a legacy image_build with no stored source diff to re-apply).
# `stale_gateway_scoring_retry_limit_exceeded`: scoring exhausted its retries.
_TERMINAL_UNRECOVERABLE_REJECTION_REASONS = frozenset(
    {
        "stale_parent_rebase_failed",
        "stale_parent_rebase_unavailable",
        "stale_gateway_scoring_retry_limit_exceeded",
    }
)


def derive_public_loop_outcome(
    *,
    ticket: Mapping[str, Any],
    queue_rows: Sequence[Mapping[str, Any]],
    receipt_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    score_bundle_rows: Sequence[Mapping[str, Any]],
    promotion_event_rows: Sequence[Mapping[str, Any]],
    improvement_threshold_points: float = 1.0,
) -> PublicLoopOutcome:
    candidate_count = len(candidate_rows)
    status_counts = _status_counts(candidate_rows, "current_candidate_status")
    scored_candidate_count = int(status_counts.get("scored", 0))
    latest_queue = _latest_row(queue_rows, "current_status_at")
    latest_receipt = _latest_row(receipt_rows, "current_status_at")
    latest_candidate = _latest_row(candidate_rows, "current_status_at")
    latest_promotion = _latest_row(promotion_event_rows, "created_at")
    latest_score = _latest_row(score_bundle_rows, "current_status_at")

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
    last_activity_at = _latest_timestamp(
        [ticket],
        queue_rows,
        receipt_rows,
        candidate_rows,
        score_bundle_rows,
        promotion_event_rows,
    )

    queue_status = str(latest_queue.get("current_queue_status") or "") if latest_queue else ""
    queue_reason = str(latest_queue.get("current_reason") or "") if latest_queue else ""

    event_doc = {
        "queue_status": queue_status,
        "queue_reason": queue_reason,
        "receipt_status": str(latest_receipt.get("current_receipt_status") or "") if latest_receipt else "",
        "candidate_status_counts": status_counts,
        "score_bundle_count": len(score_bundle_rows),
        "promotion_event_count": len(promotion_event_rows),
    }

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
    if promotion_status in {"merged", "reward_created"} or promotion_type in {
        "active_version_created",
        "champion_reward_created",
    }:
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
    if promotion_status == "failed" or promotion_type == "promotion_failed":
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
    if queue_status in {"paused", "failed"} and _is_credit_block_reason(queue_reason):
        return _result("blocked_for_credit", "blocked_for_credit", "blocked")
    if _has_candidate_reason(candidate_rows, "stale_parent_needs_rescore"):
        return _result("needs_rescore", "needs_rescore", "blocked")
    if (
        queue_status not in {"queued", "started", "running"}
        and _has_any_candidate_reason(candidate_rows, _TERMINAL_UNRECOVERABLE_REJECTION_REASONS)
        and not _has_active_candidate(status_counts)
        and scored_candidate_count == 0
    ):
        # The loop is over (no run still active) and its only candidates are terminally
        # rejected for an unrecoverable reason. A fresh re-run, if any, would make the
        # latest queue status active and take precedence over this terminal outcome.
        return _result("failed", "failed", "failed")
    if _has_candidate_reason(candidate_rows, "baseline_not_ready") or (
        queue_status == "queued" and queue_reason == "baseline_not_ready"
    ):
        return _result("waiting_for_baseline", "waiting_for_baseline", "pending")

    if status_counts.get("failed", 0) and not _has_active_candidate(status_counts):
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
        if ticket_status == "opened":
            # Emit the canonical lifecycle fields the dashboard consumes
            # (public_status / payment_state) so it renders the "Awaiting payment"
            # stage from the canonical path rather than inferring it from the legacy
            # outcome label. `no_payment` is what maps to Awaiting payment downstream.
            awaiting_event_doc = dict(event_doc)
            awaiting_event_doc["public_status"] = "awaiting_payment"
            awaiting_event_doc["payment_state"] = "no_payment"
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
        "outcome_label": row.get("current_outcome_label") or "submitted",
        "outcome_band": row.get("current_outcome_band") or "pending",
        "candidate_count": int(row.get("current_candidate_count") or 0),
        "scored_candidate_count": int(row.get("current_scored_candidate_count") or 0),
        "best_candidate_public_summary": row.get("current_best_candidate_public_summary") or "",
        "similar_recent_loop_count": int(similar_recent_loop_count),
    }


def public_loop_event_api_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_ref": row.get("event_ref"),
        "event_type": row.get("event_type"),
        "outcome_label": row.get("outcome_label"),
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
        limit=1000,
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


async def fetch_public_loop_detail(ticket_id: str) -> dict[str, Any] | None:
    row = await select_one(
        "research_lab_public_loop_card_current",
        filters=(("ticket_id", ticket_id),),
    )
    if not row:
        return None
    card_id = str(row.get("card_id") or public_loop_card_id(ticket_id))
    events = await select_many(
        "research_lab_public_loop_card_events",
        filters=(("card_id", card_id),),
        order_by=(("seq", False),),
        limit=1000,
    )
    candidate_rows = await select_many(
        "research_lab_candidate_evaluation_current",
        filters=(("ticket_id", ticket_id),),
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


def _has_any_candidate_reason(rows: Sequence[Mapping[str, Any]], reasons: frozenset[str]) -> bool:
    return any(str(row.get("current_reason") or row.get("reason") or "") in reasons for row in rows)


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
