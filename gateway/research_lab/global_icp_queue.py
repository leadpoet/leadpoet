"""Global (candidate, icp) scoring queue — persisted, distributed coordination.

Turns the in-memory scaffold ``score_candidates_global_queue`` into a durable
queue that any scoring worker can pull from. The unit of work is one
(candidate, icp) job. Public jobs are enqueued ``queued`` at priority 1;
private jobs are enqueued ``held`` at priority 0 and flipped to ``queued`` only
if the candidate's public score meets the baseline. Workers claim the next
highest-priority job (priority asc, seq asc), so the container pool always
pulls the front of the queue and a passing candidate's private jobs jump ahead
of other candidates' public jobs.

All cross-worker coordination uses compare-and-set updates (update where the
status column still equals the expected value): claiming a job, deciding a
candidate's gate exactly once, and assembling a candidate's result exactly
once. A lost race returns no rows, which the caller treats as "someone else
got it" rather than an error.

This module is pure coordination over the store primitives; the caller injects
scoring and assembly. It does nothing unless the caller invokes it, so it is
inert until the worker wires it in behind the queue flag.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from gateway.research_lab.store import insert_row, select_many, select_one, update_row


def global_icp_queue_enabled() -> bool:
    """Opt-in flag for the persisted global (candidate, icp) queue. Default off
    keeps the per-candidate claim path exactly as it is."""
    raw = str(os.getenv("RESEARCH_LAB_GLOBAL_ICP_QUEUE_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

JOB_TABLE = "research_lab_scoring_job_queue"
CANDIDATE_TABLE = "research_lab_scoring_job_candidate"

PRIORITY_PRIVATE = 0
PRIORITY_PUBLIC = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


async def _cas_update(table: str, values: Mapping[str, Any], *, filters) -> dict[str, Any] | None:
    """Compare-and-set: return the row on success, None if no row matched."""
    try:
        return await update_row(table, dict(values), filters=filters)
    except RuntimeError:
        return None


async def enqueue_candidate(
    *,
    candidate_id: str,
    window_hash: str,
    public_items: Sequence[Mapping[str, Any]],
    private_items: Sequence[Mapping[str, Any]],
    baseline_public_score: float,
    worker_ref: str,
    seq_base: int,
) -> bool:
    """Enqueue one candidate's jobs. Idempotent: a candidate already enqueued
    (its coordination row exists) is skipped. ``seq_base`` orders this
    candidate's jobs after earlier candidates'. Returns True if it enqueued.
    """
    existing = await select_one(CANDIDATE_TABLE, filters=(("candidate_id", candidate_id),))
    if existing is not None:
        return False
    try:
        await insert_row(
            CANDIDATE_TABLE,
            {
                "candidate_id": candidate_id,
                "window_hash": window_hash,
                "public_total": len(public_items),
                "private_total": len(private_items),
                "baseline_public_score": float(baseline_public_score),
                "gate_status": "pending",
                "assembly_status": "pending",
                "enqueued_by": worker_ref,
            },
        )
    except Exception:
        # Another worker created the coordination row first (PK violation) or a
        # transient write error: not ours to fill.
        return False
    now = _iso(_now())
    for offset, item in enumerate(public_items):
        await _insert_job(candidate_id, window_hash, item, offset, "public", PRIORITY_PUBLIC, seq_base + offset, "queued", now)
    # Private jobs are parked as ``held`` (priority 0) until the gate passes.
    for offset, item in enumerate(private_items):
        await _insert_job(candidate_id, window_hash, item, offset, "private", PRIORITY_PRIVATE, seq_base + offset, "held", now)
    return True


async def _insert_job(candidate_id, window_hash, item, item_index, phase, priority, seq, status, now) -> None:
    try:
        await insert_row(
            JOB_TABLE,
            {
                "job_id": str(uuid.uuid4()),
                "candidate_id": candidate_id,
                "window_hash": window_hash,
                "icp_ref": str(item.get("icp_ref") or item.get("item_ref") or f"idx:{item_index}"),
                "item_index": int(item_index),
                "phase": phase,
                "priority": int(priority),
                "seq": int(seq),
                "status": status,
                "created_at": now,
                "updated_at": now,
            },
        )
    except Exception:
        # UNIQUE(candidate_id, phase, icp_ref) — job already enqueued; idempotent.
        pass


async def claim_next_job(*, worker_ref: str, lease_seconds: int, scan_limit: int = 25) -> dict[str, Any] | None:
    """Claim the next queued job by (priority, seq) via compare-and-set. Returns
    the claimed job row, or None if nothing is claimable right now."""
    candidates = await select_many(
        JOB_TABLE,
        filters=(("status", "queued"),),
        order_by=(("priority", False), ("seq", False)),
        limit=scan_limit,
    )
    lease_until = _iso(_now() + timedelta(seconds=int(lease_seconds)))
    for job in candidates:
        won = await _cas_update(
            JOB_TABLE,
            {"status": "claimed", "claimed_by": worker_ref, "lease_expires_at": lease_until, "updated_at": _iso(_now())},
            filters=(("job_id", job["job_id"]), ("status", "queued")),
        )
        if won is not None:
            return won
    return None


async def complete_job(*, job: Mapping[str, Any], result_doc: Mapping[str, Any], failed: bool = False) -> None:
    """Record a claimed job's result (compare-and-set from claimed)."""
    await _cas_update(
        JOB_TABLE,
        {
            "status": "failed" if failed else "done",
            "result_doc": dict(result_doc or {}),
            "updated_at": _iso(_now()),
        },
        filters=(("job_id", job["job_id"]), ("status", "claimed")),
    )


async def public_set_complete(candidate_id: str) -> bool:
    """True when none of the candidate's public jobs are still outstanding."""
    outstanding = await select_many(
        JOB_TABLE,
        filters=(("candidate_id", candidate_id), ("phase", "public")),
        limit=500,
    )
    return all(str(j.get("status")) in ("done", "failed") for j in outstanding) and bool(outstanding)


async def try_decide_gate(*, candidate_id: str, public_score: float) -> str | None:
    """Decide a candidate's gate exactly once. Returns 'passed', 'rejected', or
    None if another worker is deciding / it is already decided.

    On pass, the candidate's held private jobs are flipped to queued (front of
    queue). On miss, held private jobs are marked failed so they never run.
    """
    claimed = await _cas_update(
        CANDIDATE_TABLE,
        {"gate_status": "deciding", "updated_at": _iso(_now())},
        filters=(("candidate_id", candidate_id), ("gate_status", "pending")),
    )
    if claimed is None:
        return None
    baseline = float(claimed.get("baseline_public_score") or 0.0)
    passed = float(public_score) + 1e-9 >= baseline
    now = _iso(_now())
    held = await select_many(
        JOB_TABLE,
        filters=(("candidate_id", candidate_id), ("phase", "private"), ("status", "held")),
        limit=500,
    )
    for job in held:
        await _cas_update(
            JOB_TABLE,
            {"status": "queued" if passed else "failed", "updated_at": now},
            filters=(("job_id", job["job_id"]), ("status", "held")),
        )
    await update_row(
        CANDIDATE_TABLE,
        {"gate_status": "passed" if passed else "rejected", "updated_at": now},
        filters=(("candidate_id", candidate_id),),
    )
    return "passed" if passed else "rejected"


async def candidate_ready_to_assemble(candidate_id: str) -> Mapping[str, Any] | None:
    """Return the coordination row if the candidate is fully scored and gate-
    decided (ready for one-time assembly), else None."""
    row = await select_one(CANDIDATE_TABLE, filters=(("candidate_id", candidate_id),))
    if row is None or str(row.get("gate_status")) not in ("passed", "rejected"):
        return None
    if str(row.get("assembly_status")) == "assembled":
        return None
    jobs = await select_many(JOB_TABLE, filters=(("candidate_id", candidate_id),), limit=1000)
    # Every job must be resolved: done/failed, or (on a rejected gate) held
    # private jobs that were marked failed. No queued/claimed left.
    if any(str(j.get("status")) in ("queued", "claimed", "held") for j in jobs):
        return None
    return row


async def try_claim_assembly(candidate_id: str) -> bool:
    """Compare-and-set the assembly slot so exactly one worker assembles."""
    won = await _cas_update(
        CANDIDATE_TABLE,
        {"assembly_status": "assembling", "updated_at": _iso(_now())},
        filters=(("candidate_id", candidate_id), ("assembly_status", "pending")),
    )
    return won is not None


async def mark_assembled(candidate_id: str) -> None:
    await update_row(
        CANDIDATE_TABLE,
        {"assembly_status": "assembled", "updated_at": _iso(_now())},
        filters=(("candidate_id", candidate_id),),
    )


async def candidate_result_docs(candidate_id: str) -> dict[str, list[dict[str, Any]]]:
    """Collect a candidate's per-ICP result docs, public then private, in item
    order — the shape the scaffold returns for bundle assembly."""
    jobs = await select_many(JOB_TABLE, filters=(("candidate_id", candidate_id),), limit=1000)
    public = sorted((j for j in jobs if j.get("phase") == "public" and j.get("status") == "done"), key=lambda j: int(j.get("item_index") or 0))
    private = sorted((j for j in jobs if j.get("phase") == "private" and j.get("status") == "done"), key=lambda j: int(j.get("item_index") or 0))
    return {
        "public": [dict(j.get("result_doc") or {}) for j in public],
        "private": [dict(j.get("result_doc") or {}) for j in private],
    }


async def run_queue_scoring_pass(
    *,
    worker_ref: str,
    lease_seconds: int,
    score_icp: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
    compute_public_score: Callable[[Sequence[Mapping[str, Any]]], float],
    assemble_candidate: Callable[[str, Mapping[str, list]], Awaitable[None]],
    max_jobs: int | None = None,
) -> dict[str, int]:
    """One worker's pass over the global queue.

    Recovers stale leases, then repeatedly claims the next highest-priority job
    and scores it via the injected ``score_icp``. When a candidate's public set
    finishes it decides the gate (using ``compute_public_score`` over the public
    result docs); when a candidate is fully scored and gate-decided it assembles
    exactly once via ``assemble_candidate``. Returns per-pass counters.

    Injecting scoring and assembly keeps this loop free of the worker's runner
    and bundle machinery, so it is unit-testable in isolation. It is only ever
    invoked when the queue flag is on, so the per-candidate path is untouched.
    """
    counters = {"scored": 0, "failed": 0, "gates_decided": 0, "assembled": 0, "recovered": 0}
    counters["recovered"] = await recover_stale_leases(lease_grace_seconds=lease_seconds)
    while True:
        job = await claim_next_job(worker_ref=worker_ref, lease_seconds=lease_seconds)
        if job is None:
            return counters
        candidate_id = str(job.get("candidate_id") or "")
        try:
            result_doc = await score_icp(job)
            await complete_job(job=job, result_doc=result_doc or {})
            counters["scored"] += 1
        except Exception:
            # A job that errors out is recorded failed, not retried in-loop, so
            # one bad ICP never blocks the queue; stale-lease recovery covers a
            # worker that dies mid-job.
            await complete_job(job=job, result_doc={}, failed=True)
            counters["failed"] += 1
        if str(job.get("phase")) == "public" and await public_set_complete(candidate_id):
            docs = await candidate_result_docs(candidate_id)
            decision = await try_decide_gate(
                candidate_id=candidate_id,
                public_score=float(compute_public_score(docs["public"])),
            )
            if decision is not None:
                counters["gates_decided"] += 1
        ready = await candidate_ready_to_assemble(candidate_id)
        if ready is not None and await try_claim_assembly(candidate_id):
            docs = await candidate_result_docs(candidate_id)
            try:
                await assemble_candidate(candidate_id, docs)
                await mark_assembled(candidate_id)
                counters["assembled"] += 1
            except Exception:
                # Assembly failed: release the slot so another worker retries
                # rather than leaving the candidate wedged in 'assembling'.
                await _cas_update(
                    CANDIDATE_TABLE,
                    {"assembly_status": "pending", "updated_at": _iso(_now())},
                    filters=(("candidate_id", candidate_id),),
                )
        if max_jobs is not None and counters["scored"] + counters["failed"] >= max_jobs:
            return counters


async def recover_stale_leases(*, lease_grace_seconds: int = 0) -> int:
    """Reset jobs whose lease expired back to queued so another worker retries.
    Returns how many were recovered."""
    cutoff = _iso(_now() - timedelta(seconds=int(lease_grace_seconds)))
    stale = await select_many(
        JOB_TABLE,
        # Store filter API: 3-tuple (field, operator, value) for comparisons.
        filters=(("status", "claimed"), ("lease_expires_at", "lte", cutoff)),
        limit=200,
    )
    recovered = 0
    for job in stale:
        won = await _cas_update(
            JOB_TABLE,
            {"status": "queued", "claimed_by": "", "lease_expires_at": None, "updated_at": _iso(_now())},
            filters=(("job_id", job["job_id"]), ("status", "claimed")),
        )
        if won is not None:
            recovered += 1
    return recovered
