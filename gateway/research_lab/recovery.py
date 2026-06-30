"""Research Lab lifecycle recovery operators.

Operator-safe functions to intentionally continue blocked Research Lab work:

- ``resume_failed_runs_from_checkpoint`` — re-queue terminally ``failed`` hosted
  auto-research runs that have a saved checkpoint, so the hosted worker resumes
  them from that checkpoint. OpenRouter credit-blocked failures are only resumed
  once the miner key reads as funded again.
- ``requeue_baseline_not_ready_candidates`` — wake candidates stuck ``queued``
  with reason ``baseline_not_ready`` once their private baseline is actually ready.
- ``rebase_stale_parent_candidates`` — rebase ``stale_parent_needs_rescore``
  candidates onto the current active parent (reusing the scoring worker's rebase
  path) and queue the derived candidate for scoring.

All operators are append-only (they never UPDATE rows — they append events), default
to ``dry_run=True``, are idempotent (they only act on rows still in the relevant
pre-state and skip work that was already done), and stamp actor/source provenance.
They reuse existing primitives and the scoring worker's tested logic rather than
re-implementing baseline/diff/build machinery.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from .config import ResearchLabGatewayConfig
from .icp_window import RollingIcpWindowUnavailable, fetch_rolling_icp_window
from .key_vault import OpenRouterKeyVaultError, preflight_openrouter_key
from .maintenance import (
    _is_queue_capacity_conflict,
    autoresearch_queue_capacity_doc,
    default_actor_ref,
)
from .promotion import load_active_private_model
from .scoring_worker import CandidateBaselineNotReady, ResearchLabGatewayScoringWorker
from .store import (
    create_candidate_evaluation_event,
    create_queue_event,
    latest_auto_research_checkpoint,
    select_all,
    select_many,
    select_one,
)
from .worker import OpenRouterKeyResolver
from research_lab.eval import PrivateModelArtifactManifest

logger = logging.getLogger(__name__)


# An OpenRouter HTTP 402 / insufficient-credit failure is resumable once the miner
# tops up; it should not be retried in a tight loop but can be re-queued after a
# successful credit re-check. These markers match the failure text the hosted worker
# records on the terminal `failed` queue event.
_OPENROUTER_CREDIT_MARKERS = (
    "insufficient credits",
    "insufficient balance",
    "payment required",
    "more credits",
    "http 402",
    "402:",
    "code\":402",
)


def _is_openrouter_credit_failure(error_text: str) -> bool:
    low = (error_text or "").lower()
    return any(marker in low for marker in _OPENROUTER_CREDIT_MARKERS)


async def _latest_queue_event_error(run_id: str) -> str:
    rows = await select_many(
        "research_loop_run_queue_events",
        columns="event_type,reason,event_doc,seq",
        filters=(("run_id", run_id),),
        order_by=(("seq", True),),
        limit=1,
    )
    if not rows:
        return ""
    doc = rows[0].get("event_doc") if isinstance(rows[0].get("event_doc"), Mapping) else {}
    return str(doc.get("error") or rows[0].get("reason") or "")


async def _openrouter_credit_ready(
    config: ResearchLabGatewayConfig, *, ticket_id: str
) -> tuple[bool, str]:
    """Best-effort credit/key readiness check for the miner that owns ``ticket_id``.

    Read-only (resolves the miner key and calls OpenRouter's key-info endpoint).
    Returns ``(ready, detail)``. ``limit_remaining is None`` is treated as ready
    (unmetered key); any resolution/preflight error is treated as not-ready so a
    credit-blocked run is not re-queued into the same wall.
    """
    ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
    if not ticket:
        return False, "ticket_not_found"
    miner_hotkey = str(ticket.get("miner_hotkey") or "")
    key_ref = str(ticket.get("miner_openrouter_key_ref") or "")
    if not miner_hotkey or not key_ref:
        return False, "missing_miner_openrouter_key_ref"
    try:
        env = await OpenRouterKeyResolver(config).resolve(key_ref, miner_hotkey=miner_hotkey)
        raw_key = str(env.get("OPENROUTER_API_KEY") or "")
        info = await asyncio.to_thread(preflight_openrouter_key, raw_key)
    except OpenRouterKeyVaultError as exc:
        return False, f"credit_preflight_failed:{str(exc)[:120]}"
    except Exception as exc:  # noqa: BLE001 - best-effort; any failure => not ready
        return False, f"credit_check_error:{str(exc)[:120]}"
    remaining = info.get("limit_remaining")
    if remaining is None:
        return True, "limit_remaining_unmetered"
    try:
        return (float(remaining) > 0.0), f"limit_remaining={remaining}"
    except (TypeError, ValueError):
        return True, f"limit_remaining_unparsed={remaining}"


# --------------------------------------------------------------------------- #
# Operator 1: resume failed hosted runs from checkpoint
# --------------------------------------------------------------------------- #
async def resume_failed_runs_from_checkpoint(
    *,
    run_ids: list[str] | None = None,
    dry_run: bool = True,
    require_openrouter_credit: bool = True,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()
    capacity_doc = autoresearch_queue_capacity_doc(config)
    columns = (
        "run_id,ticket_id,current_queue_status,queue_priority,"
        "current_event_hash,current_status_at"
    )

    if run_ids:
        rows: list[Mapping[str, Any]] = []
        for run_id in run_ids:
            row = await select_one(
                "research_loop_run_queue_current", columns=columns, filters=(("run_id", run_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_loop_run_queue_current",
            columns=columns,
            filters=(("current_queue_status", "failed"),),
            max_rows=10000,
        )

    result: dict[str, Any] = {
        "ok": True,
        "action": "resume-failed-runs-from-checkpoint",
        "dry_run": dry_run,
        "found": 0,
        "resumed": 0,
        "not_failed": [],
        "not_resumable_no_checkpoint": [],
        "blocked_for_credit": [],
        "blocked": [],
        "runs": [],
    }

    for row in rows:
        run_id = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if str(row.get("current_queue_status") or "") != "failed":
            result["not_failed"].append(run_id)
            continue
        result["found"] += 1

        checkpoint = await latest_auto_research_checkpoint(run_id)
        if not checkpoint:
            result["not_resumable_no_checkpoint"].append(run_id)
            result["runs"].append({"run_id": run_id, "status": "not_resumable_no_checkpoint"})
            continue
        checkpoint_hash = str(checkpoint.get("checkpoint_hash") or "")

        error_text = await _latest_queue_event_error(run_id)
        credit_failure = _is_openrouter_credit_failure(error_text)
        credit_detail: str | None = None
        if credit_failure and require_openrouter_credit:
            ready, credit_detail = await _openrouter_credit_ready(config, ticket_id=ticket_id)
            if not ready:
                result["blocked_for_credit"].append(run_id)
                result["runs"].append(
                    {"run_id": run_id, "status": "blocked_for_credit", "detail": credit_detail}
                )
                continue

        plan = {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "status": "resumable",
            "checkpoint_hash": checkpoint_hash,
            "credit_failure": credit_failure,
            "credit_detail": credit_detail,
        }
        if dry_run:
            result["runs"].append({**plan, "dry_run": True})
            continue

        try:
            event = await create_queue_event(
                run_id=run_id,
                ticket_id=ticket_id,
                event_type="queued",
                queue_priority=int(row.get("queue_priority") or 0),
                worker_ref=actor,
                reason="operator_resume_from_checkpoint",
                event_doc={
                    "schema_version": "1.0",
                    **capacity_doc,
                    "resume_source": "resume_failed_runs_from_checkpoint",
                    "previous_queue_status": "failed",
                    "previous_event_hash": row.get("current_event_hash"),
                    "previous_status_at": row.get("current_status_at"),
                    "checkpoint_hash": checkpoint_hash,
                    "previous_error": error_text[:300],
                    "credit_failure": credit_failure,
                    "actor_ref": actor,
                },
            )
        except Exception as exc:  # noqa: BLE001
            if _is_queue_capacity_conflict(exc):
                result["blocked"].append(
                    {"run_id": run_id, "error": "queue_capacity_or_hotkey_conflict", "detail": str(exc)[:200]}
                )
                result["runs"].append({**plan, "status": "blocked", "error": str(exc)[:200]})
                continue
            raise

        result["resumed"] += 1
        result["runs"].append(
            {
                **plan,
                "status": "resumed",
                "requeued_event_id": event.get("event_id"),
                "requeued_event_seq": event.get("seq"),
                "requeued_event_hash": event.get("anchored_hash"),
            }
        )

    return result


async def resume_credit_blocked_runs_for_miner(
    miner_hotkey: str,
    *,
    run_ids: list[str] | None = None,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    """Re-queue a miner's credit-blocked (paused/blocked_for_credit) runs after a top-up.

    Re-checks the miner's OpenRouter key; re-queues funded runs with reason
    `credit_topup_resume` (the hosted worker's preflight is the final gate, so a
    still-unfunded key just re-pauses). Never consumes another loop-start payment.
    Used by the miner-facing gateway endpoint and available to operators. Fastapi-free
    so it is unit-testable without the web stack.
    """
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or f"miner:{str(miner_hotkey)[:12]}"
    capacity_doc = autoresearch_queue_capacity_doc(config)
    requested = set(run_ids or [])

    blocked = await select_all(
        "research_loop_run_queue_current",
        columns="run_id,ticket_id,current_queue_status,current_reason,queue_priority",
        filters=(("current_queue_status", "paused"), ("current_reason", "blocked_for_credit")),
        max_rows=10000,
    )
    requeued = 0
    still_blocked = 0
    results: list[dict[str, Any]] = []

    for row in blocked:
        run_id = str(row.get("run_id") or "")
        ticket_id = str(row.get("ticket_id") or "")
        if requested and run_id not in requested:
            continue
        ticket = await select_one("research_loop_ticket_current", filters=(("ticket_id", ticket_id),))
        if not ticket or str(ticket.get("miner_hotkey") or "") != str(miner_hotkey):
            continue  # only this miner's runs

        ready, detail = await _openrouter_credit_ready(config, ticket_id=ticket_id)
        if not ready:
            still_blocked += 1
            results.append({"run_id": run_id, "status": "still_blocked_for_credit", "detail": detail})
            continue
        try:
            await create_queue_event(
                run_id=run_id,
                ticket_id=ticket_id,
                event_type="queued",
                queue_priority=int(row.get("queue_priority") or 0),
                worker_ref=actor,
                reason="credit_topup_resume",
                event_doc={
                    "schema_version": "1.0",
                    **capacity_doc,
                    "resume_source": "miner_credit_topup_resume",
                    "previous_queue_status": "paused",
                    "previous_reason": "blocked_for_credit",
                    "credit_check": detail,
                },
            )
            requeued += 1
            results.append({"run_id": run_id, "status": "requeued", "credit_check": detail})
        except Exception as exc:  # noqa: BLE001
            status = "requeue_capacity_conflict" if _is_queue_capacity_conflict(exc) else "requeue_failed"
            results.append({"run_id": run_id, "status": status, "detail": str(exc)[:160]})

    return {"requeued": requeued, "still_blocked": still_blocked, "results": results}


# --------------------------------------------------------------------------- #
# Operator 2: requeue baseline_not_ready candidates once the baseline is ready
# --------------------------------------------------------------------------- #
async def requeue_baseline_not_ready_candidates(
    *,
    candidate_ids: list[str] | None = None,
    dry_run: bool = True,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()

    if candidate_ids:
        rows: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            row = await select_one(
                "research_lab_candidate_evaluation_current", filters=(("candidate_id", candidate_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            filters=(
                ("current_candidate_status", "queued"),
                ("current_reason", "baseline_not_ready"),
            ),
            max_rows=10000,
        )

    result: dict[str, Any] = {
        "ok": True,
        "action": "requeue-baseline-not-ready",
        "dry_run": dry_run,
        "found": 0,
        "requeued": 0,
        "skipped": [],
        "still_waiting_for_baseline": [],
        "candidates": [],
    }

    worker: ResearchLabGatewayScoringWorker | None = None
    window = None

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        status = str(row.get("current_candidate_status") or "")
        reason = str(row.get("current_reason") or "")
        if status != "queued" or reason != "baseline_not_ready":
            result["skipped"].append({"candidate_id": candidate_id, "status": status, "reason": reason})
            continue
        manifest_doc = row.get("private_model_manifest_doc")
        if not isinstance(manifest_doc, Mapping):
            result["skipped"].append({"candidate_id": candidate_id, "reason": "missing_manifest_doc"})
            continue
        result["found"] += 1

        if worker is None:
            worker = ResearchLabGatewayScoringWorker(config, worker_ref=actor)
        if window is None:
            try:
                window = await fetch_rolling_icp_window(
                    days=config.lab_champion_eval_days,
                    icps_per_day=config.lab_champion_icps_per_day,
                    allow_partial=config.scoring_worker_allow_partial_icp_window,
                )
            except RollingIcpWindowUnavailable as exc:
                result["ok"] = False
                result["error"] = f"rolling_window_unavailable:{str(exc)[:160]}"
                return result

        artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
        try:
            await worker._candidate_private_holdout_gate(artifact=artifact, window_hash=window.window_hash)
        except CandidateBaselineNotReady:
            result["still_waiting_for_baseline"].append(candidate_id)
            result["candidates"].append(
                {"candidate_id": candidate_id, "status": "still_waiting_for_baseline"}
            )
            continue

        plan = {"candidate_id": candidate_id, "status": "baseline_ready"}
        if dry_run:
            result["candidates"].append({**plan, "dry_run": True})
            continue

        event = await create_candidate_evaluation_event(
            candidate_id=candidate_id,
            run_id=str(row["run_id"]),
            ticket_id=str(row["ticket_id"]),
            event_type="queued",
            candidate_status="queued",
            reason="baseline_ready_requeue",
            evaluator_ref=actor,
            event_doc={
                "operator_action": "requeue-baseline-not-ready",
                "resume_source": "requeue_baseline_not_ready_candidates",
                "previous_reason": "baseline_not_ready",
                "rolling_window_hash": window.window_hash,
                "actor_ref": actor,
            },
        )
        result["requeued"] += 1
        result["candidates"].append(
            {
                **plan,
                "status": "requeued",
                "requeued_event_id": event.get("event_id"),
                "requeued_event_seq": event.get("seq"),
            }
        )

    return result


# --------------------------------------------------------------------------- #
# Operator 3: rebase stale-parent candidates onto the current active parent
# --------------------------------------------------------------------------- #
async def rebase_stale_parent_candidates(
    *,
    candidate_ids: list[str] | None = None,
    dry_run: bool = True,
    max_batch_size: int = 25,
    actor_ref: str | None = None,
) -> dict[str, Any]:
    config = ResearchLabGatewayConfig.from_env()
    actor = actor_ref or default_actor_ref()

    if candidate_ids:
        rows: list[Mapping[str, Any]] = []
        for candidate_id in candidate_ids:
            row = await select_one(
                "research_lab_candidate_evaluation_current", filters=(("candidate_id", candidate_id),)
            )
            if row:
                rows.append(row)
    else:
        rows = await select_all(
            "research_lab_candidate_evaluation_current",
            filters=(
                ("current_candidate_status", "rejected"),
                ("current_reason", "stale_parent_needs_rescore"),
            ),
            max_rows=10000,
        )

    if max_batch_size and len(rows) > int(max_batch_size):
        rows = rows[: int(max_batch_size)]

    active = await load_active_private_model(config, register_bootstrap=False)
    active_parent = active.artifact.model_artifact_hash

    result: dict[str, Any] = {
        "ok": True,
        "action": "rebase-stale-parents",
        "dry_run": dry_run,
        "active_parent_artifact_hash": active_parent,
        "found": 0,
        "rebased": 0,
        "skipped": [],
        "already_rebased": [],
        "already_on_current_parent": [],
        "failed": [],
        "rebases": [],
    }

    worker: ResearchLabGatewayScoringWorker | None = None
    evaluation_epoch: int | None = None

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        status = str(row.get("current_candidate_status") or "")
        reason = str(row.get("current_reason") or "")
        if status != "rejected" or reason != "stale_parent_needs_rescore":
            result["skipped"].append({"candidate_id": candidate_id, "status": status, "reason": reason})
            continue
        result["found"] += 1

        existing = await select_many(
            "research_lab_candidate_promotion_events",
            columns="promotion_event_id,event_type,derived_candidate_id",
            filters=(("candidate_id", candidate_id), ("event_type", "rebase_queued")),
            limit=1,
        )
        if existing:
            result["already_rebased"].append(
                {"candidate_id": candidate_id, "derived_candidate_id": existing[0].get("derived_candidate_id")}
            )
            continue

        candidate_parent = str(row.get("parent_artifact_hash") or "")
        if candidate_parent == active_parent:
            result["already_on_current_parent"].append(candidate_id)
            continue

        plan = {
            "candidate_id": candidate_id,
            "candidate_parent_artifact_hash": candidate_parent,
            "active_parent_artifact_hash": active_parent,
            "status": "rebase_eligible",
        }
        if dry_run:
            result["rebases"].append({**plan, "dry_run": True})
            continue

        if worker is None:
            worker = ResearchLabGatewayScoringWorker(config, worker_ref=actor)
            evaluation_epoch = await worker._resolve_evaluation_epoch()

        try:
            rebase = await worker._maybe_rebase_stale_candidate_before_scoring(
                row,
                evaluation_epoch=int(evaluation_epoch or 0),
                elapsed_seconds=lambda: 0.0,
            )
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({"candidate_id": candidate_id, "error": str(exc)[:200]})
            result["rebases"].append({**plan, "status": "failed", "error": str(exc)[:200]})
            continue

        rebase_status = str(rebase.get("status") or "")
        if rebase_status == "stale_parent_rebased_to_current":
            result["rebased"] += 1
            result["rebases"].append(
                {
                    **plan,
                    "status": rebase_status,
                    "derived_candidate_id": rebase.get("derived_candidate_id"),
                    "repair_used": rebase.get("repair_used"),
                }
            )
        else:
            result["rebases"].append({**plan, "status": rebase_status or "no_rebase", "detail": rebase})
            if rebase_status != "current_parent":
                result["failed"].append({"candidate_id": candidate_id, "status": rebase_status})

    if result["failed"]:
        result["ok"] = result["rebased"] > 0 or dry_run
    return result
