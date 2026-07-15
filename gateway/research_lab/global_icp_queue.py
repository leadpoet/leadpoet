"""Global (candidate, icp) scoring queue — persisted, distributed coordination.

Turns the in-memory scaffold ``score_candidates_global_queue`` into a durable
queue that any scoring worker can pull from. The unit of work is one
(candidate, icp) job. Public jobs are enqueued ``queued`` at priority 1;
private and optional conditional jobs are enqueued ``held`` at priority 0.
Private jobs are released only if the candidate's public score meets the
baseline. Conditional jobs are released only if the complete public/private
score clears the frozen preliminary threshold. Workers claim the next
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

import math
import os
import re
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from gateway.research_lab.store import (
    call_rpc,
    insert_row,
    select_many,
    select_one,
    update_row,
)


logger = logging.getLogger(__name__)


def global_icp_queue_enabled() -> bool:
    """Opt-in flag for the persisted global (candidate, icp) queue. Default off
    keeps the per-candidate claim path exactly as it is."""
    raw = str(os.getenv("RESEARCH_LAB_GLOBAL_ICP_QUEUE_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

JOB_TABLE = "research_lab_scoring_job_queue"
CANDIDATE_TABLE = "research_lab_scoring_job_candidate"

PRIORITY_PRIVATE = 0
PRIORITY_CONDITIONAL = 0
PRIORITY_PUBLIC = 1
ATTESTED_RECEIPT_HASHES_FIELD = "_attested_receipt_hashes"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    conditional_items: Sequence[Mapping[str, Any]] = (),
    baseline_public_score: float,
    baseline_preliminary_score: float = 0.0,
    threshold_points: float = 0.0,
    baseline_benchmark_bundle_id: str = "",
    baseline_benchmark_hash: str = "",
    category_assignment_hash: str = "",
    conditional_policy_hash: str = "",
    candidate_artifact_hash: str = "",
    candidate_parent_artifact_hash: str = "",
    scoring_configuration_hash: str = "",
    worker_ref: str,
    seq_base: int,
    scoring_run_id: str = "",
) -> str | None:
    """Create or idempotently refill one active candidate generation.

    ``seq_base`` orders this candidate's jobs after earlier candidates'. The
    generation id is returned for both a new generation and a refill; ``None``
    means no safe active generation could be established.
    """
    if conditional_items:
        _validate_conditional_job_partition(
            public_items=public_items,
            private_items=private_items,
            conditional_items=conditional_items,
        )
        _validate_conditional_generation_values(
            baseline_public_score=baseline_public_score,
            baseline_preliminary_score=baseline_preliminary_score,
            threshold_points=threshold_points,
            baseline_benchmark_bundle_id=baseline_benchmark_bundle_id,
            baseline_benchmark_hash=baseline_benchmark_hash,
            category_assignment_hash=category_assignment_hash,
            conditional_policy_hash=conditional_policy_hash,
            candidate_artifact_hash=candidate_artifact_hash,
            candidate_parent_artifact_hash=candidate_parent_artifact_hash,
            scoring_configuration_hash=scoring_configuration_hash,
        )
    existing = await select_one(
        CANDIDATE_TABLE,
        filters=(
            ("candidate_id", candidate_id),
            ("window_hash", window_hash),
            ("assembly_status", "in", ("pending", "assembling")),
        ),
    )
    if existing is not None:
        if conditional_items:
            _validate_existing_conditional_generation(
                existing,
                public_total=len(public_items),
                private_total=len(private_items),
                conditional_total=len(conditional_items),
                baseline_public_score=baseline_public_score,
                baseline_preliminary_score=baseline_preliminary_score,
                threshold_points=threshold_points,
                baseline_benchmark_bundle_id=baseline_benchmark_bundle_id,
                baseline_benchmark_hash=baseline_benchmark_hash,
                category_assignment_hash=category_assignment_hash,
                conditional_policy_hash=conditional_policy_hash,
                candidate_artifact_hash=candidate_artifact_hash,
                candidate_parent_artifact_hash=candidate_parent_artifact_hash,
                scoring_configuration_hash=scoring_configuration_hash,
            )
        queue_generation_id = str(existing.get("queue_generation_id") or "")
        scoring_run_id = scoring_run_id or str(existing.get("scoring_run_id") or "")
    else:
        queue_generation_id = str(uuid.uuid4())
        try:
            await insert_row(
                CANDIDATE_TABLE,
                {
                    "candidate_id": candidate_id,
                    "queue_generation_id": queue_generation_id,
                    "scoring_run_id": scoring_run_id or None,
                    "window_hash": window_hash,
                    "public_total": len(public_items),
                    "private_total": len(private_items),
                    "conditional_total": len(conditional_items),
                    "baseline_public_score": float(baseline_public_score),
                    "baseline_preliminary_score": float(baseline_preliminary_score),
                    "threshold_points": float(threshold_points),
                    "baseline_benchmark_bundle_id": baseline_benchmark_bundle_id or None,
                    "baseline_benchmark_hash": baseline_benchmark_hash or None,
                    "category_assignment_hash": category_assignment_hash or None,
                    "conditional_policy_hash": conditional_policy_hash or None,
                    "candidate_artifact_hash": candidate_artifact_hash or None,
                    "candidate_parent_artifact_hash": (
                        candidate_parent_artifact_hash or None
                    ),
                    "scoring_configuration_hash": scoring_configuration_hash or None,
                    "gate_status": "pending",
                    "preliminary_gate_status": (
                        "pending" if conditional_items else "not_required"
                    ),
                    "assembly_status": "pending",
                    "enqueued_by": worker_ref,
                },
            )
        except Exception as exc:
            # A concurrent creator may have won the active-generation unique
            # index. Re-read and idempotently fill that generation's jobs.
            existing = await select_one(
                CANDIDATE_TABLE,
                filters=(
                    ("candidate_id", candidate_id),
                    ("window_hash", window_hash),
                    ("assembly_status", "in", ("pending", "assembling")),
                ),
            )
            if existing is None:
                logger.warning(
                    "research_lab_global_queue_candidate_insert_failed candidate_id=%s error=%s",
                    candidate_id[:120],
                    str(exc)[:240],
                )
                return None
            if conditional_items:
                _validate_existing_conditional_generation(
                    existing,
                    public_total=len(public_items),
                    private_total=len(private_items),
                    conditional_total=len(conditional_items),
                    baseline_public_score=baseline_public_score,
                    baseline_preliminary_score=baseline_preliminary_score,
                    threshold_points=threshold_points,
                    baseline_benchmark_bundle_id=baseline_benchmark_bundle_id,
                    baseline_benchmark_hash=baseline_benchmark_hash,
                    category_assignment_hash=category_assignment_hash,
                    conditional_policy_hash=conditional_policy_hash,
                    candidate_artifact_hash=candidate_artifact_hash,
                    candidate_parent_artifact_hash=candidate_parent_artifact_hash,
                    scoring_configuration_hash=scoring_configuration_hash,
                )
            queue_generation_id = str(existing.get("queue_generation_id") or "")
            scoring_run_id = scoring_run_id or str(existing.get("scoring_run_id") or "")
    if not queue_generation_id:
        logger.warning(
            "research_lab_global_queue_generation_missing candidate_id=%s",
            candidate_id[:120],
        )
        return None
    now = _iso(_now())
    for offset, item in enumerate(public_items):
        await _insert_job(queue_generation_id, scoring_run_id, candidate_id, window_hash, item, offset, "public", PRIORITY_PUBLIC, seq_base + offset, "queued", now)
    # Private jobs are parked as ``held`` (priority 0) until the gate passes.
    for offset, item in enumerate(private_items):
        await _insert_job(queue_generation_id, scoring_run_id, candidate_id, window_hash, item, offset, "private", PRIORITY_PRIVATE, seq_base + offset, "held", now)
    for offset, item in enumerate(conditional_items):
        await _insert_job(
            queue_generation_id,
            scoring_run_id,
            candidate_id,
            window_hash,
            item,
            offset,
            "conditional",
            PRIORITY_CONDITIONAL,
            seq_base + len(private_items) + offset,
            "held",
            now,
        )
    return queue_generation_id


def _validate_conditional_job_partition(
    *,
    public_items: Sequence[Mapping[str, Any]],
    private_items: Sequence[Mapping[str, Any]],
    conditional_items: Sequence[Mapping[str, Any]],
) -> None:
    category_refs: dict[str, list[str]] = {
        "public": [_queue_item_ref(item, index) for index, item in enumerate(public_items)],
        "private": [_queue_item_ref(item, index) for index, item in enumerate(private_items)],
        "conditional": [
            _queue_item_ref(item, index) for index, item in enumerate(conditional_items)
        ],
    }
    for category, refs in category_refs.items():
        if len(refs) != len(set(refs)):
            raise ValueError(f"conditional queue has duplicate {category} ICP refs")
    all_refs = [ref for refs in category_refs.values() for ref in refs]
    if len(all_refs) != len(set(all_refs)):
        raise ValueError("conditional queue ICP categories overlap")


def _queue_item_ref(item: Mapping[str, Any], index: int) -> str:
    ref = str(item.get("icp_ref") or item.get("item_ref") or "").strip()
    if not ref:
        raise ValueError(f"conditional queue item at index {index} is missing icp_ref")
    return ref


def _validate_conditional_generation_values(
    *,
    baseline_public_score: float,
    baseline_preliminary_score: float,
    threshold_points: float,
    baseline_benchmark_bundle_id: str,
    baseline_benchmark_hash: str,
    category_assignment_hash: str,
    conditional_policy_hash: str,
    candidate_artifact_hash: str,
    candidate_parent_artifact_hash: str,
    scoring_configuration_hash: str,
) -> None:
    for name, value in (
        ("baseline_public_score", baseline_public_score),
        ("baseline_preliminary_score", baseline_preliminary_score),
        ("threshold_points", threshold_points),
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 100.0:
            raise ValueError(f"{name} must be finite and within 0-100")
    if not str(baseline_benchmark_bundle_id or "").strip():
        raise ValueError("conditional queue requires a baseline benchmark bundle id")
    for name, value in (
        ("baseline_benchmark_hash", baseline_benchmark_hash),
        ("category_assignment_hash", category_assignment_hash),
        ("conditional_policy_hash", conditional_policy_hash),
        ("candidate_artifact_hash", candidate_artifact_hash),
        ("candidate_parent_artifact_hash", candidate_parent_artifact_hash),
        ("scoring_configuration_hash", scoring_configuration_hash),
    ):
        text = str(value or "").strip()
        if not _HASH_RE.fullmatch(text):
            raise ValueError(f"conditional queue requires a valid {name}")


def _validate_existing_conditional_generation(
    existing: Mapping[str, Any],
    **expected: Any,
) -> None:
    for field, wanted in expected.items():
        actual = existing.get(field)
        if field in {
            "baseline_public_score",
            "baseline_preliminary_score",
            "threshold_points",
        }:
            matches = round(float(actual or 0.0), 9) == round(float(wanted), 9)
        elif field in {"public_total", "private_total", "conditional_total"}:
            matches = int(actual or 0) == int(wanted)
        else:
            matches = str(actual or "") == str(wanted or "")
        if not matches:
            raise RuntimeError(
                "research_lab_conditional_queue_commitment_conflict:"
                f"{field}:{actual!r}!={wanted!r}"
            )


async def _insert_job(queue_generation_id, scoring_run_id, candidate_id, window_hash, item, item_index, phase, priority, seq, status, now) -> None:
    try:
        await insert_row(
            JOB_TABLE,
            {
                "job_id": str(uuid.uuid4()),
                "candidate_id": candidate_id,
                "queue_generation_id": queue_generation_id,
                "scoring_run_id": scoring_run_id or None,
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
    except Exception as exc:
        # UNIQUE(queue_generation_id, phase, icp_ref) — idempotent generation fill.
        text = str(exc).lower()
        if "duplicate key" in text or "unique constraint" in text or "23505" in text:
            return
        logger.warning(
            "research_lab_global_queue_job_insert_failed generation=%s icp_ref=%s error=%s",
            str(queue_generation_id)[:64],
            str(item.get("icp_ref") or item.get("item_ref") or "")[:120],
            str(exc)[:240],
        )
        raise


async def _best_effort_callback(tag: str, callback, *args) -> None:
    if callback is None:
        return
    try:
        await callback(*args)
    except Exception:
        logger.warning("%s", tag, exc_info=True)


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
            {
                "status": "claimed",
                "claimed_by": worker_ref,
                "lease_expires_at": lease_until,
                "attempt_count": int(job.get("attempt_count") or 0) + 1,
                "updated_at": _iso(_now()),
            },
            filters=(("job_id", job["job_id"]), ("status", "queued")),
        )
        if won is not None:
            return won
    return None


async def complete_job(*, job: Mapping[str, Any], result_doc: Mapping[str, Any], failed: bool = False) -> bool:
    """Record a claimed job's result (compare-and-set from claimed)."""
    won = await _cas_update(
        JOB_TABLE,
        {
            "status": "failed" if failed else "done",
            "result_doc": dict(result_doc or {}),
            "updated_at": _iso(_now()),
        },
        filters=(
            ("job_id", job["job_id"]),
            ("status", "claimed"),
            ("claimed_by", str(job.get("claimed_by") or "")),
            ("attempt_count", int(job.get("attempt_count") or 0)),
        ),
    )
    return won is not None


async def requeue_job(*, job: Mapping[str, Any], result_doc: Mapping[str, Any]) -> bool:
    """Return one retryable claimed job to the queue without settling it."""

    if str(job.get("phase") or "") == "conditional":
        result = await call_rpc(
            "research_lab_requeue_conditional_scoring_job",
            {
                "target_job_id": str(job["job_id"]),
                "expected_claimed_by": str(job.get("claimed_by") or ""),
                "expected_attempt_count": int(job.get("attempt_count") or 0),
                "target_failure_class": str(
                    result_doc.get("failure_class")
                    or "conditional_validation_retryable_failure"
                ),
            },
        )
        row: Any = result[0] if isinstance(result, list) and result else result
        return bool(isinstance(row, Mapping) and row.get("committed"))

    won = await _cas_update(
        JOB_TABLE,
        {
            "status": "queued",
            "result_doc": dict(result_doc or {}),
            "claimed_by": "",
            "lease_expires_at": None,
            "updated_at": _iso(_now()),
        },
        filters=(
            ("job_id", job["job_id"]),
            ("status", "claimed"),
            ("claimed_by", str(job.get("claimed_by") or "")),
            ("attempt_count", int(job.get("attempt_count") or 0)),
        ),
    )
    return won is not None


async def public_set_complete(queue_generation_id: str) -> bool:
    """True when none of the candidate's public jobs are still outstanding."""
    outstanding = await select_many(
        JOB_TABLE,
        filters=(("queue_generation_id", queue_generation_id), ("phase", "public")),
        limit=500,
    )
    return all(str(j.get("status")) in ("done", "failed") for j in outstanding) and bool(outstanding)


async def phase_set_complete(queue_generation_id: str, phase: str) -> bool:
    if phase not in {"public", "private", "conditional"}:
        raise ValueError("invalid global scoring queue phase")
    outstanding = await select_many(
        JOB_TABLE,
        filters=(("queue_generation_id", queue_generation_id), ("phase", phase)),
        limit=500,
    )
    return bool(outstanding) and all(
        str(job.get("status")) in ("done", "failed") for job in outstanding
    )


async def try_decide_gate(*, queue_generation_id: str, public_score: float) -> str | None:
    """Decide a candidate's gate exactly once. Returns 'passed', 'rejected', or
    None if another worker is deciding / it is already decided.

    On pass, the candidate's held private jobs are flipped to queued (front of
    queue). On miss, held private jobs are marked failed so they never run.
    """
    if not math.isfinite(float(public_score)):
        raise ValueError("public gate score must be finite")
    candidate = await select_one(
        CANDIDATE_TABLE,
        filters=(("queue_generation_id", queue_generation_id),),
    )
    if candidate is None:
        return None
    if int(candidate.get("conditional_total") or 0) > 0:
        return _rpc_gate_decision(
            await call_rpc(
                "research_lab_decide_conditional_public_gate",
                {
                    "target_queue_generation_id": queue_generation_id,
                    "candidate_public_score": float(public_score),
                },
            )
        )

    claimed = await _cas_update(
        CANDIDATE_TABLE,
        {"gate_status": "deciding", "updated_at": _iso(_now())},
        filters=(("queue_generation_id", queue_generation_id), ("gate_status", "pending")),
    )
    if claimed is None:
        return None
    baseline = float(claimed.get("baseline_public_score") or 0.0)
    passed = float(public_score) + 1e-9 >= baseline
    now = _iso(_now())
    held = await select_many(
        JOB_TABLE,
        filters=(("queue_generation_id", queue_generation_id), ("phase", "private"), ("status", "held")),
        limit=500,
    )
    for job in held:
        await _cas_update(
            JOB_TABLE,
            {"status": "queued" if passed else "failed", "updated_at": now},
            filters=(("job_id", job["job_id"]), ("status", "held")),
        )
    if not passed:
        conditional_held = await select_many(
            JOB_TABLE,
            filters=(
                ("queue_generation_id", queue_generation_id),
                ("phase", "conditional"),
                ("status", "held"),
            ),
            limit=500,
        )
        for job in conditional_held:
            await _cas_update(
                JOB_TABLE,
                {"status": "failed", "updated_at": now},
                filters=(("job_id", job["job_id"]), ("status", "held")),
            )
    await update_row(
        CANDIDATE_TABLE,
        {
            "gate_status": "passed" if passed else "rejected",
            **({"preliminary_gate_status": "skipped"} if not passed else {}),
            "updated_at": now,
        },
        filters=(("queue_generation_id", queue_generation_id),),
    )
    return "passed" if passed else "rejected"


async def claim_preliminary_gate(
    *,
    queue_generation_id: str,
    worker_ref: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    """Lease the measured 20-ICP gate so only one worker attests it."""

    value = await call_rpc(
        "research_lab_claim_conditional_preliminary_gate",
        {
            "target_queue_generation_id": queue_generation_id,
            "target_worker_ref": worker_ref,
            "target_lease_seconds": max(30, int(lease_seconds)),
        },
    )
    row: Any = value[0] if isinstance(value, list) and value else value
    if not isinstance(row, Mapping):
        return None
    if str(row.get("decision") or "") != "claimed":
        return None
    claim = row.get("claim")
    return dict(claim) if isinstance(claim, Mapping) else None


async def try_decide_preliminary_gate(
    *,
    queue_generation_id: str,
    preliminary_score: float,
    preliminary_proof: Mapping[str, Any],
    expected_claimed_by: str,
    expected_attempt_count: int,
) -> str | None:
    """Atomically persist the measured gate proof, then release conditional jobs."""

    if not math.isfinite(float(preliminary_score)):
        raise ValueError("preliminary gate score must be finite")
    return _rpc_gate_decision(
        await call_rpc(
            "research_lab_decide_conditional_preliminary_gate",
            {
                "target_queue_generation_id": queue_generation_id,
                "candidate_preliminary_score": float(preliminary_score),
                "target_preliminary_proof": dict(preliminary_proof),
                "expected_claimed_by": expected_claimed_by,
                "expected_attempt_count": int(expected_attempt_count),
            },
        )
    )


async def cancel_preliminary_gate_for_rebase(
    *,
    queue_generation_id: str,
    expected_claimed_by: str,
    expected_attempt_count: int,
    failure_class: str,
) -> bool:
    value = await call_rpc(
        "research_lab_cancel_conditional_generation",
        {
            "target_queue_generation_id": queue_generation_id,
            "expected_claimed_by": expected_claimed_by,
            "expected_attempt_count": int(expected_attempt_count),
            "target_failure_class": str(failure_class or "stale_parent_needs_rescore"),
        },
    )
    row: Any = value[0] if isinstance(value, list) and value else value
    return bool(isinstance(row, Mapping) and row.get("committed") is True)


def _rpc_gate_decision(value: Any) -> str | None:
    row: Any = value
    if isinstance(value, list):
        row = value[0] if value else None
    if not isinstance(row, Mapping):
        return None
    decision = str(row.get("decision") or "")
    if decision in {"passed", "rejected"}:
        return decision
    if decision in {
        "already_decided",
        "busy",
        "claim_changed",
        "not_eligible",
        "not_ready",
        "not_found",
    }:
        return None
    raise RuntimeError(f"unexpected conditional queue gate decision: {decision or '<empty>'}")


async def candidate_ready_to_assemble(queue_generation_id: str) -> Mapping[str, Any] | None:
    """Return the coordination row if the candidate is fully scored and gate-
    decided (ready for one-time assembly), else None."""
    row = await select_one(CANDIDATE_TABLE, filters=(("queue_generation_id", queue_generation_id),))
    if row is None or str(row.get("gate_status")) not in ("passed", "rejected"):
        return None
    if int(row.get("conditional_total") or 0) > 0 and str(
        row.get("preliminary_gate_status") or ""
    ) not in ("passed", "rejected", "skipped"):
        return None
    if str(row.get("assembly_status")) == "assembled":
        return None
    jobs = await select_many(JOB_TABLE, filters=(("queue_generation_id", queue_generation_id),), limit=1000)
    # Every job must be resolved: done/failed, or (on a rejected gate) held
    # private jobs that were marked failed. No queued/claimed left.
    if any(str(j.get("status")) in ("queued", "claimed", "held") for j in jobs):
        return None
    return row


async def try_claim_assembly(queue_generation_id: str) -> bool:
    """Compare-and-set the assembly slot so exactly one worker assembles."""
    won = await _cas_update(
        CANDIDATE_TABLE,
        {"assembly_status": "assembling", "updated_at": _iso(_now())},
        filters=(("queue_generation_id", queue_generation_id), ("assembly_status", "pending")),
    )
    return won is not None


async def mark_assembled(queue_generation_id: str) -> None:
    await update_row(
        CANDIDATE_TABLE,
        {"assembly_status": "assembled", "updated_at": _iso(_now())},
        filters=(("queue_generation_id", queue_generation_id),),
    )


def _split_persisted_result_doc(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    result = dict(value or {})
    raw_hashes = result.pop(ATTESTED_RECEIPT_HASHES_FIELD, [])
    if not isinstance(raw_hashes, list) or any(
        not _HASH_RE.fullmatch(str(item or "")) for item in raw_hashes
    ):
        raise RuntimeError("queue result has invalid attested receipt hashes")
    return result, sorted({str(item) for item in raw_hashes})


async def candidate_result_docs(queue_generation_id: str) -> dict[str, Any]:
    """Collect a candidate's per-ICP result docs, public then private, in item
    order — the shape the scaffold returns for bundle assembly."""
    jobs = await select_many(JOB_TABLE, filters=(("queue_generation_id", queue_generation_id),), limit=1000)
    public = sorted((j for j in jobs if j.get("phase") == "public" and j.get("status") == "done"), key=lambda j: int(j.get("item_index") or 0))
    private = sorted((j for j in jobs if j.get("phase") == "private" and j.get("status") == "done"), key=lambda j: int(j.get("item_index") or 0))
    conditional = sorted((j for j in jobs if j.get("phase") == "conditional" and j.get("status") == "done"), key=lambda j: int(j.get("item_index") or 0))
    phase_docs: dict[str, list[dict[str, Any]]] = {
        "public": [],
        "private": [],
        "conditional": [],
    }
    receipt_hashes: set[str] = set()
    for phase, jobs in (
        ("public", public),
        ("private", private),
        ("conditional", conditional),
    ):
        for job in jobs:
            result_doc, job_receipt_hashes = _split_persisted_result_doc(
                job.get("result_doc")
                if isinstance(job.get("result_doc"), Mapping)
                else None
            )
            phase_docs[phase].append(result_doc)
            receipt_hashes.update(job_receipt_hashes)
    return {
        **phase_docs,
        "attested_receipt_hashes": sorted(receipt_hashes),
    }


async def run_queue_scoring_pass(
    *,
    worker_ref: str,
    lease_seconds: int,
    score_icp: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
    compute_public_score: Callable[[Sequence[Mapping[str, Any]]], float],
    compute_preliminary_score: Callable[[Sequence[Mapping[str, Any]]], float] | None = None,
    preliminary_gate_authorizer: Callable[
        [str, Mapping[str, Any], Mapping[str, Any], float],
        Awaitable[Mapping[str, Any]],
    ] | None = None,
    assemble_candidate: Callable[[str, str, Mapping[str, Any]], Awaitable[None]],
    max_jobs: int | None = None,
    job_completed: Callable[[Mapping[str, Any], Mapping[str, Any], bool, bool, BaseException | None], Awaitable[None]] | None = None,
    stale_job_recovered: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    gate_decided: Callable[[str, str], Awaitable[None]] | None = None,
    preliminary_gate_decided: Callable[[str, str], Awaitable[None]] | None = None,
    preliminary_gate_error: Callable[
        [str, Mapping[str, Any], BaseException], Awaitable[bool]
    ] | None = None,
    retryable_job_error: Callable[[Mapping[str, Any], BaseException], bool] | None = None,
    candidate_assembled: Callable[[str, str], Awaitable[None]] | None = None,
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
    counters = {
        "scored": 0,
        "failed": 0,
        "retried": 0,
        "gates_decided": 0,
        "preliminary_gates_decided": 0,
        "assembled": 0,
        "recovered": 0,
    }
    counters["recovered"] = await recover_stale_leases(
        lease_grace_seconds=lease_seconds,
        stale_job_recovered=stale_job_recovered,
    )

    async def _decide_ready_preliminary_gate(
        queue_generation_id: str,
    ) -> str | None:
        if compute_preliminary_score is None:
            return None
        candidate_row = await select_one(
            CANDIDATE_TABLE,
            filters=(("queue_generation_id", queue_generation_id),),
        )
        if (
            candidate_row is None
            or int(candidate_row.get("conditional_total") or 0) <= 0
            or str(candidate_row.get("gate_status") or "") != "passed"
            or str(candidate_row.get("preliminary_gate_status") or "")
            not in {"pending", "deciding"}
            or not await phase_set_complete(queue_generation_id, "private")
        ):
            return None
        claim = await claim_preliminary_gate(
            queue_generation_id=queue_generation_id,
            worker_ref=worker_ref,
            lease_seconds=lease_seconds,
        )
        if claim is None:
            return None
        try:
            docs = await candidate_result_docs(queue_generation_id)
            preliminary_score = float(
                compute_preliminary_score([*docs["public"], *docs["private"]])
            )
            if not math.isfinite(preliminary_score):
                raise ValueError("preliminary gate score must be finite")
            proof: Mapping[str, Any] = {}
            host_score_passed = (
                preliminary_score
                - float(claim.get("baseline_preliminary_score") or 0.0)
                + 1e-9
                >= float(claim.get("threshold_points") or 0.0)
            )
            if host_score_passed:
                if preliminary_gate_authorizer is None:
                    raise RuntimeError(
                        "conditional preliminary gate authority is unavailable"
                    )
                proof = await preliminary_gate_authorizer(
                    queue_generation_id,
                    claim,
                    docs,
                    preliminary_score,
                )
            decision = await try_decide_preliminary_gate(
                queue_generation_id=queue_generation_id,
                preliminary_score=preliminary_score,
                preliminary_proof=proof,
                expected_claimed_by=worker_ref,
                expected_attempt_count=int(
                    claim.get("preliminary_gate_attempt_count") or 0
                ),
            )
        except Exception as exc:
            handled = False
            if preliminary_gate_error is not None:
                try:
                    handled = bool(
                        await preliminary_gate_error(
                            queue_generation_id,
                            claim,
                            exc,
                        )
                    )
                except Exception:
                    logger.warning(
                        "research_lab_global_queue_preliminary_gate_error_callback_failed",
                        exc_info=True,
                    )
            if not handled:
                logger.warning(
                    "research_lab_global_queue_preliminary_gate_authority_failed generation=%s error=%s",
                    queue_generation_id[:80],
                    str(exc)[:240],
                )
            return None
        if decision is not None:
            counters["preliminary_gates_decided"] += 1
            await _best_effort_callback(
                "research_lab_global_queue_preliminary_gate_callback_failed",
                preliminary_gate_decided,
                queue_generation_id,
                decision,
            )
        return decision

    pending_preliminary = await select_many(
        CANDIDATE_TABLE,
        filters=(
            ("gate_status", "passed"),
            ("preliminary_gate_status", "in", ("pending", "deciding")),
            ("assembly_status", "pending"),
        ),
        order_by=(("updated_at", False),),
        limit=100,
    )
    for pending in pending_preliminary:
        if int(pending.get("conditional_total") or 0) > 0:
            await _decide_ready_preliminary_gate(
                str(pending.get("queue_generation_id") or "")
            )

    while True:
        job = await claim_next_job(worker_ref=worker_ref, lease_seconds=lease_seconds)
        if job is None:
            return counters
        candidate_id = str(job.get("candidate_id") or "")
        queue_generation_id = str(job.get("queue_generation_id") or "")
        try:
            result_doc = await score_icp(job)
            committed = await complete_job(job=job, result_doc=result_doc or {})
            await _best_effort_callback(
                "research_lab_global_queue_job_completed_callback_failed",
                job_completed,
                job,
                result_doc or {},
                False,
                committed,
                None,
            )
            counters["scored"] += 1
        except Exception as exc:
            if retryable_job_error is not None and retryable_job_error(job, exc):
                committed = await requeue_job(
                    job=job,
                    result_doc={"retryable": True, "failure_class": exc.__class__.__name__},
                )
                await _best_effort_callback(
                    "research_lab_global_queue_job_retry_callback_failed",
                    job_completed,
                    job,
                    {},
                    True,
                    committed,
                    exc,
                )
                counters["retried"] += 1
                return counters
            # A job that errors out is recorded failed, not retried in-loop, so
            # one bad ICP never blocks the queue; stale-lease recovery covers a
            # worker that dies mid-job.
            committed = await complete_job(job=job, result_doc={}, failed=True)
            await _best_effort_callback(
                "research_lab_global_queue_job_failed_callback_failed",
                job_completed,
                job,
                {},
                True,
                committed,
                exc,
            )
            counters["failed"] += 1
        if str(job.get("phase")) == "public" and await public_set_complete(queue_generation_id):
            docs = await candidate_result_docs(queue_generation_id)
            decision = await try_decide_gate(
                queue_generation_id=queue_generation_id,
                public_score=float(compute_public_score(docs["public"])),
            )
            if decision is not None:
                counters["gates_decided"] += 1
                await _best_effort_callback(
                    "research_lab_global_queue_gate_callback_failed",
                    gate_decided,
                    queue_generation_id,
                    decision,
                )
        if str(job.get("phase")) == "private":
            await _decide_ready_preliminary_gate(queue_generation_id)
        ready = await candidate_ready_to_assemble(queue_generation_id)
        if ready is not None and await try_claim_assembly(queue_generation_id):
            docs = await candidate_result_docs(queue_generation_id)
            try:
                await assemble_candidate(queue_generation_id, candidate_id, docs)
                await mark_assembled(queue_generation_id)
                counters["assembled"] += 1
                await _best_effort_callback(
                    "research_lab_global_queue_assembled_callback_failed",
                    candidate_assembled,
                    queue_generation_id,
                    candidate_id,
                )
            except Exception:
                # Assembly failed: release the slot so another worker retries
                # rather than leaving the candidate wedged in 'assembling'.
                await _cas_update(
                    CANDIDATE_TABLE,
                    {"assembly_status": "pending", "updated_at": _iso(_now())},
                    filters=(("queue_generation_id", queue_generation_id),),
                )
        if max_jobs is not None and counters["scored"] + counters["failed"] >= max_jobs:
            return counters


async def recover_stale_leases(
    *,
    lease_grace_seconds: int = 0,
    stale_job_recovered: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
) -> int:
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
            {"status": "held", "claimed_by": "", "lease_expires_at": None, "updated_at": _iso(_now())},
            filters=(("job_id", job["job_id"]), ("status", "claimed")),
        )
        if won is not None:
            try:
                if stale_job_recovered is not None:
                    try:
                        await stale_job_recovered(job)
                    except Exception:
                        # Recovery telemetry can never wedge the work queue.
                        pass
            finally:
                await _cas_update(
                    JOB_TABLE,
                    {"status": "queued", "updated_at": _iso(_now())},
                    filters=(("job_id", job["job_id"]), ("status", "held")),
                )
            recovered += 1
    return recovered
