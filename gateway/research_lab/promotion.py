"""Gateway-owned Research Lab promotion and private model lineage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_promotion_event,
    create_champion_reward_obligation,
    create_private_model_version,
    create_private_model_version_event,
    create_private_repo_commit_event,
    select_many,
    select_one,
)
from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.eval import (
    PrivateModelArtifactManifest,
    load_private_artifact_manifest,
    validate_private_model_artifact_manifest,
)


logger = logging.getLogger(__name__)


TRUTHY = {"1", "true", "yes", "on"}

# Fresh-environment escape hatch: bootstrap registration from the configured
# manifest URI is allowed only when the lineage table is genuinely empty AND
# this flag is explicitly enabled. Default false so a transient lineage read
# failure (or an operator manifest drift) can never silently re-activate the
# bootstrap version with a fresh timestamp (bug #2 / Chain A).
ALLOW_BOOTSTRAP_REGISTER_ENV = "RESEARCH_LAB_ALLOW_BOOTSTRAP_REGISTER"

# Bug #29 (auto-commit head-mismatch wedge): default false keeps the current
# hard-fail behavior, which is deliberately the accidental guard against
# noise-merges (§8.3). Enable only after the baseline health gate is live.
AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV = "RESEARCH_LAB_AUTO_COMMIT_HEAD_MISMATCH_RECOVER"

# Historical confirmation re-run support remains for reading old events, but
# automatic promotion is score-only against the stored daily baseline.
PROMOTION_CONFIRMATION_RERUN_ENV = "RESEARCH_LAB_PROMOTION_CONFIRMATION_RERUN"
# Confirmation pass bar; unset/invalid falls back to the promotion threshold.
CONFIRMATION_MIN_DELTA_ENV = "RESEARCH_LAB_CONFIRMATION_MIN_DELTA"

# Confirmation state machine markers. promotion_status/event_type stay within
# the DB CHECK allowlists (scripts/62); these free-text reasons live in
# event_doc.reason and are the queryable confirmation state:
#   held_pending_confirmation      -> candidate cleared every merge gate on the
#                                     first measurement; awaiting confirmation
#   confirmation_rerun_started     -> a scoring worker claimed the measurement
#   confirmation_rerun_recorded    -> measurement doc recorded (side artifact;
#                                     never replaces the day's benchmark)
#   confirmation_rerun_attempt_failed -> infra failure; re-held, bounded by the
#                                     claim-attempt budget
#   rejected_confirmation_failed   -> terminal: confirmation delta below the
#                                     bar (or attempts exhausted); both deltas
#                                     recorded in the rejection event_doc
#   confirmation_rerun_closed      -> a terminal promotion decision landed for
#                                     this candidate+bundle; the worker stops
#                                     re-driving it
CONFIRMATION_HOLD_REASON = "held_pending_confirmation"
CONFIRMATION_STARTED_REASON = "confirmation_rerun_started"
CONFIRMATION_RESULT_REASON = "confirmation_rerun_recorded"
CONFIRMATION_ATTEMPT_FAILED_REASON = "confirmation_rerun_attempt_failed"
CONFIRMATION_CLOSED_REASON = "confirmation_rerun_closed"
CONFIRMATION_REJECTED_REASON = "rejected_confirmation_failed"
_CONFIRMATION_STATE_REASONS = frozenset(
    {
        CONFIRMATION_HOLD_REASON,
        CONFIRMATION_STARTED_REASON,
        CONFIRMATION_RESULT_REASON,
        CONFIRMATION_ATTEMPT_FAILED_REASON,
        CONFIRMATION_CLOSED_REASON,
    }
)

# Promotion outcomes that do NOT settle a held candidate: retryable holds and
# temporarily-disabled promotion keep the confirmation open for a later
# re-drive; anything else closes it.
CONFIRMATION_NON_CLOSING_STATUSES = frozenset(
    {
        "",
        "held_pending_confirmation",
        "held_baseline_doc_unavailable",
        "held_baseline_health_gate_failed",
        "disabled",
        "failed",
    }
)


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY


def promotion_confirmation_rerun_enabled() -> bool:
    """Confirmation reruns are no longer a promotion gate."""
    return False


def confirmation_min_delta(threshold_points: float) -> float:
    raw = os.getenv(CONFIRMATION_MIN_DELTA_ENV, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "research_lab_confirmation_min_delta_invalid value=%r falling back to threshold=%s",
                raw,
                threshold_points,
            )
    return float(threshold_points)


def confirmation_attempt_budget(config: Any) -> int:
    """Confirmation infra-failures re-hold, bounded by the existing
    claim-attempt budget (scoring_worker_max_claim_requeues)."""
    try:
        return max(1, int(getattr(config, "scoring_worker_max_claim_requeues", 3) or 3))
    except (TypeError, ValueError):
        return 3


def _confirmation_event_reason(row: Mapping[str, Any]) -> str:
    doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    return str(doc.get("reason") or "")


def confirmation_doc_from_event(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The recorded measurement doc inside a confirmation_rerun_recorded event."""
    if not isinstance(row, Mapping):
        return {}
    doc = row.get("event_doc") if isinstance(row.get("event_doc"), Mapping) else {}
    confirmation = doc.get("confirmation") if isinstance(doc, Mapping) else None
    return dict(confirmation) if isinstance(confirmation, Mapping) else {}


async def load_confirmation_state(
    *,
    candidate_id: str,
    score_bundle_id: str,
) -> dict[str, Any]:
    """Derive the candidate+bundle confirmation state from promotion events.

    The state machine is event-sourced (no in-memory-only state): every
    transition is a ``promotion_checked``/``checked`` event whose
    ``event_doc.reason`` is one of the confirmation reasons above, scoped to
    the source score bundle. Rows are newest-first; ``latest_reason`` is the
    current phase, ``attempts`` counts started measurement claims.
    """

    rows = await select_many(
        "research_lab_candidate_promotion_events",
        columns=(
            "promotion_event_id,candidate_id,event_type,promotion_status,"
            "source_score_bundle_id,worker_ref,event_doc,created_at"
        ),
        filters=(
            ("candidate_id", candidate_id),
            ("source_score_bundle_id", score_bundle_id),
            ("event_type", "promotion_checked"),
        ),
        order_by=(("created_at", True),),
        limit=200,
    )
    latest_event: dict[str, Any] | None = None
    held_event: dict[str, Any] | None = None
    result_event: dict[str, Any] | None = None
    started_events: list[dict[str, Any]] = []
    attempt_failed_events: list[dict[str, Any]] = []
    open_claim_events: list[dict[str, Any]] = []
    open_claim_run = True
    for row in rows:
        reason = _confirmation_event_reason(row)
        if reason not in _CONFIRMATION_STATE_REASONS:
            continue
        if latest_event is None:
            latest_event = dict(row)
        if reason == CONFIRMATION_STARTED_REASON:
            # The contiguous newest-first run of started events (before any
            # other confirmation event) is the set of competing measurement
            # claims: earlier attempts are separated by their
            # recorded/attempt_failed/hold events.
            if open_claim_run:
                open_claim_events.append(dict(row))
        else:
            open_claim_run = False
        if reason == CONFIRMATION_RESULT_REASON:
            if result_event is None:
                result_event = dict(row)
        elif reason == CONFIRMATION_HOLD_REASON:
            if held_event is None:
                held_event = dict(row)
        elif reason == CONFIRMATION_STARTED_REASON:
            started_events.append(dict(row))
        elif reason == CONFIRMATION_ATTEMPT_FAILED_REASON:
            attempt_failed_events.append(dict(row))
    return {
        "latest_event": latest_event,
        "latest_reason": _confirmation_event_reason(latest_event) if latest_event else "",
        "held_event": held_event,
        "result_event": result_event,
        "started_events": started_events,
        "attempt_failed_events": attempt_failed_events,
        "open_claim_events": open_claim_events,
        "attempts": len(started_events),
    }


async def candidate_already_promoted(candidate_id: str) -> dict[str, Any] | None:
    """The candidate's ``active_version_created`` event, if it ever merged.

    Crash/race guard for confirmation re-drives: a worker crash mid-confirmation
    (or two workers racing a recorded confirmation) must never double-merge —
    the one-active DB trigger is the backstop, not the mechanism.
    """

    rows = await select_many(
        "research_lab_candidate_promotion_events",
        columns="promotion_event_id,event_type,promotion_status,private_model_version_id,created_at",
        filters=(
            ("candidate_id", candidate_id),
            ("event_type", "active_version_created"),
        ),
        order_by=(("created_at", True),),
        limit=1,
    )
    return dict(rows[0]) if rows else None


class PromotionPausedError(RuntimeError):
    """Fail-closed promotion/lineage error: pause and retry, never fall back.

    Raised instead of silently degrading to the bootstrap manifest. Callers
    should treat this as retryable (a paused promotion retries; a bootstrap
    rollback corrupts every in-flight candidate).
    """


class PrivateModelLineageUnavailableError(PromotionPausedError):
    """The private-model lineage could not be read (or its active manifest
    could not be loaded). Retryable; do NOT fall back to bootstrap."""


class NoActivePrivateModelVersionError(PromotionPausedError):
    """Lineage is non-empty but has zero active versions (supersede/create
    crash window, bug #3). Run the ``reconcile-active-lineage`` admin command
    (``reconcile_active_private_model_lineage``) to re-activate the newest
    superseded version."""


class ActiveManifestHashMismatchError(PromotionPausedError):
    """The active lineage row's recorded hashes no longer match the manifest
    at its URI (operator model update mutated the manifest under a live
    lineage row). Requires the explicit ``reregister-active-manifest`` admin
    command; never silently falls through to bootstrap."""

    def __init__(self, *, version_id: str, detail: dict[str, str]):
        self.version_id = version_id
        self.detail = dict(detail)
        super().__init__(
            "mutable_manifest_hash_mismatch: active lineage row "
            f"{_short_ref(version_id)} no longer matches its manifest URI contents; "
            "operator action required — run the reregister-active-manifest admin "
            "command to verify and deliberately re-register the updated manifest"
        )


class RepoHeadMismatchError(RuntimeError):
    """Source branch head does not match the active manifest git sha (bug #29)."""

    def __init__(self, *, head: str, expected_sha: str):
        self.head = str(head)
        self.expected_sha = str(expected_sha)
        super().__init__(
            "repo_head_mismatch: source branch head does not match active model commit "
            f"(head={self.head[:12]}, active_manifest_sha={self.expected_sha[:12]}); "
            "the active manifest records a commit sha the branch head has moved past "
            f"(auto-commit wedge). Set {AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV}=true to "
            "re-resolve against the current head — only after the baseline health gate is live"
        )


@dataclass(frozen=True)
class ActivePrivateModel:
    artifact: PrivateModelArtifactManifest
    version_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromotionImprovementMetric:
    improvement_points: float
    basis: str
    daily_baseline_available: bool
    baseline_aggregate_score: float | None = None
    candidate_total_score: float | None = None
    candidate_delta_vs_daily_baseline: float | None = None
    # Explicit "can't compute" rejection (§0-N3): set on holdout-gated bundles
    # whose stored-baseline basis is unavailable, so a future
    # improvement_threshold_points=0 can never promote unmeasured candidates.
    rejection_status: str | None = None
    # Symmetric ICP exclusion (§5.2-1): ICPs excluded from the candidate totals
    # due to unresolved provider errors, mirrored out of the baseline basis.
    provider_excluded_icp_ids: tuple[str, ...] = ()
    baseline_basis_adjusted: bool = False
    unadjusted_baseline_aggregate_score: float | None = None

    def event_doc(self) -> dict[str, Any]:
        return {
            "improvement_basis": self.basis,
            "daily_baseline_available": self.daily_baseline_available,
            "baseline_aggregate_score": self.baseline_aggregate_score,
            "candidate_total_score": self.candidate_total_score,
            "candidate_delta_vs_daily_baseline": self.candidate_delta_vs_daily_baseline,
            "rejection_status": self.rejection_status,
            "provider_excluded_icp_ids": list(self.provider_excluded_icp_ids),
            "baseline_basis_adjusted": self.baseline_basis_adjusted,
            "unadjusted_baseline_aggregate_score": self.unadjusted_baseline_aggregate_score,
        }


def promotion_improvement_metric(
    score_bundle: Mapping[str, Any],
    *,
    baseline_score_summary_doc: Mapping[str, Any] | None = None,
) -> PromotionImprovementMetric:
    """Return the promotion metric without re-running the active parent model.

    Candidate score bundles emitted by the private-holdout path are judged
    against the stored daily baseline aggregate. Older non-holdout bundles keep
    their legacy paired mean-delta path for compatibility with historical tests
    and tooling, but any bundle that carries a holdout gate must provide the
    stored-baseline final delta to be promotable.

    An unavailable basis is an explicit rejection (``rejection_status`` set,
    §0-N3) rather than a silent 0.0 improvement.

    Provider/runtime health and provider-exclusion audit fields do not change
    the promotion basis. A candidate promotes only when its stored final score
    beats the stored daily baseline aggregate by the configured threshold.
    """

    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    gate = score_bundle.get("private_holdout_gate")
    if isinstance(gate, Mapping):
        decision = str(gate.get("decision") or "")
        baseline_aggregate = _optional_float(gate.get("baseline_aggregate_score"))
        candidate_total = _optional_float(gate.get("candidate_total_score"))
        excluded_icp_ids = _provider_excluded_icp_ids(aggregates)
        daily_delta = _optional_float(gate.get("candidate_delta_vs_daily_baseline"))
        if daily_delta is None and baseline_aggregate is not None and candidate_total is not None:
            daily_delta = candidate_total - baseline_aggregate
        if (
            decision == "private_holdout_approved"
            and bool(gate.get("private_holdout_evaluated"))
            and daily_delta is not None
        ):
            return PromotionImprovementMetric(
                improvement_points=float(daily_delta),
                basis="stored_daily_baseline_total_delta",
                daily_baseline_available=True,
                baseline_aggregate_score=baseline_aggregate,
                candidate_total_score=candidate_total,
                candidate_delta_vs_daily_baseline=float(daily_delta),
                provider_excluded_icp_ids=excluded_icp_ids,
                baseline_basis_adjusted=False,
                unadjusted_baseline_aggregate_score=None,
            )
        unavailable_reason = decision or "missing_decision"
        return PromotionImprovementMetric(
            improvement_points=0.0,
            basis=f"stored_daily_baseline_unavailable:{unavailable_reason}",
            daily_baseline_available=False,
            baseline_aggregate_score=baseline_aggregate,
            candidate_total_score=candidate_total,
            candidate_delta_vs_daily_baseline=daily_delta,
            rejection_status="rejected_basis_unavailable",
            provider_excluded_icp_ids=excluded_icp_ids,
            baseline_basis_adjusted=False,
            unadjusted_baseline_aggregate_score=None,
        )

    legacy_delta = _optional_float(aggregates.get("mean_delta")) or 0.0
    return PromotionImprovementMetric(
        improvement_points=float(legacy_delta),
        basis="legacy_paired_mean_delta_no_holdout_gate",
        daily_baseline_available=False,
    )


def _provider_excluded_icp_ids(aggregates: Mapping[str, Any]) -> tuple[str, ...]:
    value = aggregates.get("provider_excluded_icp_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    seen: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _baseline_per_icp_scores(doc: Mapping[str, Any] | None) -> dict[str, float]:
    """Map icp_ref -> baseline score from the stored daily benchmark doc.

    The benchmark's ``score_summary_doc`` carries ``per_icp_summaries``
    (``sanitize_benchmark_item_summary`` rows with ``icp_ref``/``score``) and a
    ``visibility_split.items`` list with the same fields; prefer the former.
    """

    if not isinstance(doc, Mapping):
        return {}
    scores: dict[str, float] = {}
    summaries = doc.get("per_icp_summaries")
    if isinstance(summaries, list):
        for item in summaries:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or "").strip()
            score = _optional_float(item.get("score"))
            if ref and score is not None:
                scores[ref] = score
    if scores:
        return scores
    split = doc.get("visibility_split")
    items = split.get("items") if isinstance(split, Mapping) else None
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or "").strip()
            score = _optional_float(item.get("score"))
            if ref and score is not None:
                scores[ref] = score
    return scores


def _baseline_aggregate_excluding_icps(
    doc: Mapping[str, Any] | None,
    excluded_icp_ids: Sequence[str],
) -> float | None:
    """Baseline aggregate recomputed without the excluded ICPs, or None.

    Strict: every excluded ICP id must resolve to a baseline per-ICP score, so
    identifier-format drift degrades to an explicit unavailable basis instead
    of silently reintroducing the asymmetry.
    """

    per_icp = _baseline_per_icp_scores(doc)
    if not per_icp:
        return None
    excluded = {str(item) for item in excluded_icp_ids}
    if not excluded.issubset(per_icp.keys()):
        return None
    remaining = [score for ref, score in per_icp.items() if ref not in excluded]
    if not remaining:
        return None
    return float(sum(remaining) / len(remaining))


def _baseline_health_doc(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The day's benchmark ``baseline_health`` dict, or None when absent.

    Shape (written by the parallel-baseline gate):
    ``{"unresolved_provider_errors": n, "gate_passed": bool, ...}``.
    Legacy docs without the field pass (tolerated absence).
    """

    if not isinstance(doc, Mapping):
        return None
    health = doc.get("baseline_health")
    if not isinstance(health, Mapping):
        return None
    return dict(health)


async def load_baseline_summary_doc_for_gate(
    gate: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Load the stored daily benchmark ``score_summary_doc`` a holdout gate used.

    Returns ``(doc, status)`` with status ``loaded`` | ``absent`` |
    ``unavailable``. ``absent`` means there is genuinely no doc to consult
    (legacy bundles pass); ``unavailable`` means the read itself failed and the
    caller should hold (retryable) rather than decide without it.
    """

    if not isinstance(gate, Mapping):
        return None, "absent"
    bundle_id = str(gate.get("baseline_benchmark_bundle_id") or "")
    if not bundle_id:
        return None, "absent"
    try:
        row = await select_one(
            "research_lab_private_model_benchmark_current",
            columns="benchmark_bundle_id,score_summary_doc",
            filters=(("benchmark_bundle_id", bundle_id),),
        )
    except Exception as exc:
        logger.warning(
            "research_lab_promotion_baseline_summary_doc_unavailable: bundle=%s error=%s",
            _short_ref(bundle_id),
            _safe_text(str(exc))[:200],
        )
        return None, "unavailable"
    doc = row.get("score_summary_doc") if isinstance(row, Mapping) else None
    if not isinstance(doc, Mapping):
        return None, "absent"
    return dict(doc), "loaded"


async def load_active_private_model(
    config: ResearchLabGatewayConfig,
    *,
    register_bootstrap: bool = False,
) -> ActivePrivateModel:
    """Load the current active private model, failing CLOSED on lineage doubt.

    The lineage table is authoritative when present. Failure modes are
    distinguished explicitly (bug #2 / Chain A):

    * lineage read error -> ``PrivateModelLineageUnavailableError`` (retryable);
    * active row's manifest fails to load -> ``PrivateModelLineageUnavailableError``;
    * active row's hashes no longer match its manifest URI contents ->
      ``ActiveManifestHashMismatchError`` (operator must run
      ``reregister-active-manifest``);
    * zero active versions but lineage non-empty ->
      ``NoActivePrivateModelVersionError`` (run ``reconcile-active-lineage``);
    * lineage genuinely empty (0 rows) -> the configured manifest URI is used
      as a bootstrap source; registration additionally requires
      ``RESEARCH_LAB_ALLOW_BOOTSTRAP_REGISTER=true`` (fresh environments only).

    None of these ever silently re-activates the bootstrap version.
    """

    try:
        rows = await select_many(
            "research_lab_private_model_version_current",
            filters=(("current_version_status", "active"),),
            order_by=(("current_status_at", True),),
            limit=1,
        )
    except Exception as exc:
        logger.warning("research_lab_active_model_lineage_unavailable: %s", str(exc)[:200])
        raise PrivateModelLineageUnavailableError(
            "private model lineage read failed; promotion paused (retryable), "
            f"not falling back to bootstrap: {_safe_text(str(exc))[:200]}"
        ) from exc

    for row in rows:
        version_id = str(row.get("private_model_version_id") or "")
        try:
            artifact = _load_valid_artifact(str(row["private_model_manifest_uri"]))
        except Exception as exc:
            logger.warning(
                "research_lab_active_model_lineage_row_load_failed: version=%s error=%s",
                _short_ref(version_id),
                _safe_text(str(exc))[:200],
            )
            raise PrivateModelLineageUnavailableError(
                f"active private model manifest load failed for version {_short_ref(version_id)}; "
                "promotion paused (retryable), not falling back to bootstrap: "
                f"{_safe_text(str(exc))[:200]}"
            ) from exc

        row_artifact_hash = str(row["model_artifact_hash"])
        row_manifest_hash = str(row["private_model_manifest_hash"])
        if artifact.model_artifact_hash == row_artifact_hash and artifact.manifest_hash == row_manifest_hash:
            return ActivePrivateModel(artifact=artifact, version_row=row)

        logger.error(
            "research_lab_active_manifest_hash_mismatch_requires_reregister: version=%s "
            "row_artifact=%s loaded_artifact=%s row_manifest=%s loaded_manifest=%s "
            "(operator action: run the reregister-active-manifest admin command)",
            _short_ref(version_id),
            _short_ref(row_artifact_hash),
            _short_ref(artifact.model_artifact_hash),
            _short_ref(row_manifest_hash),
            _short_ref(artifact.manifest_hash),
        )
        raise ActiveManifestHashMismatchError(
            version_id=version_id,
            detail={
                "row_model_artifact_hash": row_artifact_hash,
                "loaded_model_artifact_hash": artifact.model_artifact_hash,
                "row_private_model_manifest_hash": row_manifest_hash,
                "loaded_private_model_manifest_hash": artifact.manifest_hash,
            },
        )

    # No ACTIVE rows. Distinguish "lineage genuinely empty" (fresh environment,
    # bootstrap allowed) from "lineage exists but nothing is active" (bug #3
    # crash window — reconcile, never bootstrap).
    try:
        lineage_rows = await select_many(
            "research_lab_private_model_version_current",
            columns="private_model_version_id,current_version_status,current_status_at",
            filters=(),
            order_by=(("current_status_at", True),),
            limit=1,
        )
    except Exception as exc:
        logger.warning("research_lab_active_model_lineage_unavailable: %s", str(exc)[:200])
        raise PrivateModelLineageUnavailableError(
            "private model lineage emptiness check failed; promotion paused (retryable), "
            f"not falling back to bootstrap: {_safe_text(str(exc))[:200]}"
        ) from exc
    if lineage_rows:
        raise NoActivePrivateModelVersionError(
            "private model lineage is non-empty but has zero active versions "
            "(supersede/create crash window); run the reconcile-active-lineage admin "
            "command to re-activate the newest superseded version"
        )

    artifact = _load_valid_artifact(config.private_model_manifest_uri)
    version_row = None
    if register_bootstrap:
        if not _env_flag(ALLOW_BOOTSTRAP_REGISTER_ENV):
            logger.warning(
                "research_lab_bootstrap_register_suppressed: lineage is empty but %s is not "
                "enabled; returning unregistered bootstrap manifest",
                ALLOW_BOOTSTRAP_REGISTER_ENV,
            )
        else:
            try:
                version_row, _event = await create_private_model_version(
                    artifact_manifest=artifact.to_dict(),
                    manifest_uri=config.private_model_manifest_uri,
                    redacted_version_doc={
                        "source": "bootstrap_private_model_manifest_uri",
                        "model_artifact_hash": artifact.model_artifact_hash,
                        "private_model_manifest_hash": artifact.manifest_hash,
                        "git_commit_sha": artifact.git_commit_sha,
                        "component_registry_version": artifact.component_registry_version,
                        "scoring_adapter_version": artifact.scoring_adapter_version,
                    },
                    version_status="active",
                    reason="bootstrap_private_model_manifest_uri",
                )
            except Exception as exc:
                logger.warning("research_lab_active_model_bootstrap_write_failed: %s", str(exc)[:200])
    return ActivePrivateModel(artifact=artifact, version_row=version_row)


async def reconcile_active_private_model_lineage(
    *,
    actor_ref: str = "maintenance",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Repair the bug #3 crash window: zero active versions, lineage non-empty.

    Re-activates the newest superseded version so the champion lookup recovers
    without a bootstrap rollback. Safe to call at startup or from maintenance;
    a no-op when an active version exists or the lineage is empty.
    """

    active_rows = await select_many(
        "research_lab_private_model_version_current",
        columns="private_model_version_id,current_version_status,current_status_at",
        filters=(("current_version_status", "active"),),
        order_by=(("current_status_at", True),),
        limit=5,
    )
    if active_rows:
        return {
            "ok": True,
            "action": "reconcile-active-lineage",
            "status": "active_version_present",
            "active_version_count": len(active_rows),
            "active_version_ids": [str(row.get("private_model_version_id") or "") for row in active_rows],
        }
    lineage_rows = await select_many(
        "research_lab_private_model_version_current",
        columns="private_model_version_id,current_version_status,current_status_at,model_artifact_hash",
        filters=(),
        order_by=(("current_status_at", True),),
        limit=50,
    )
    if not lineage_rows:
        return {"ok": True, "action": "reconcile-active-lineage", "status": "lineage_empty"}
    superseded_rows = [
        row for row in lineage_rows if str(row.get("current_version_status") or "") == "superseded"
    ]
    if not superseded_rows:
        return {
            "ok": False,
            "action": "reconcile-active-lineage",
            "status": "no_superseded_version_to_reactivate",
            "lineage_statuses": sorted(
                {str(row.get("current_version_status") or "") for row in lineage_rows}
            ),
        }
    newest = superseded_rows[0]
    version_id = str(newest.get("private_model_version_id") or "")
    planned = {
        "private_model_version_id": version_id,
        "model_artifact_hash": str(newest.get("model_artifact_hash") or ""),
        "previous_status_at": str(newest.get("current_status_at") or ""),
    }
    if dry_run:
        return {
            "ok": True,
            "action": "reconcile-active-lineage",
            "status": "would_reactivate_newest_superseded",
            "dry_run": True,
            "planned": planned,
        }
    await create_private_model_version_event(
        private_model_version_id=version_id,
        event_type="active",
        version_status="active",
        reason="reconcile_reactivate_newest_superseded_version",
        event_doc={
            "source": "research_lab_lineage_reconcile",
            "actor_ref": actor_ref,
            "previous_version_status": "superseded",
        },
    )
    logger.warning(
        "research_lab_lineage_reconcile_reactivated: version=%s actor=%s",
        _short_ref(version_id),
        actor_ref,
    )
    return {
        "ok": True,
        "action": "reconcile-active-lineage",
        "status": "reactivated_newest_superseded",
        "dry_run": False,
        "planned": planned,
    }


async def reregister_active_manifest(
    config: ResearchLabGatewayConfig,
    *,
    actor_ref: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Deliberate operator re-register after a mutable-manifest hash mismatch.

    Loads the manifest currently at the active lineage row's URI, verifies it,
    and (on ``--write``) supersedes the mismatched row then registers the
    loaded manifest as the new active version. This is the explicit
    replacement for the old silent fall-through-to-bootstrap path (§0.3).
    The supersede -> create order matches the one-active DB guard
    (scripts/60); a crash in between is repaired by
    ``reconcile_active_private_model_lineage`` (which re-activates the
    superseded row, after which this command can simply be re-run).
    """

    rows = await select_many(
        "research_lab_private_model_version_current",
        filters=(("current_version_status", "active"),),
        order_by=(("current_status_at", True),),
        limit=1,
    )
    if not rows:
        return {
            "ok": False,
            "action": "reregister-active-manifest",
            "status": "no_active_version",
            "guidance": "run reconcile-active-lineage (or bootstrap a fresh environment) first",
        }
    row = rows[0]
    version_id = str(row.get("private_model_version_id") or "")
    manifest_uri = str(row.get("private_model_manifest_uri") or "")
    try:
        artifact = _load_valid_artifact(manifest_uri)
    except Exception as exc:
        return {
            "ok": False,
            "action": "reregister-active-manifest",
            "status": "manifest_load_failed",
            "private_model_version_id": version_id,
            "error": _safe_text(str(exc))[:300],
        }
    row_artifact_hash = str(row.get("model_artifact_hash") or "")
    row_manifest_hash = str(row.get("private_model_manifest_hash") or "")
    mismatch_doc = {
        "row_model_artifact_hash": row_artifact_hash,
        "loaded_model_artifact_hash": artifact.model_artifact_hash,
        "row_private_model_manifest_hash": row_manifest_hash,
        "loaded_private_model_manifest_hash": artifact.manifest_hash,
    }
    if artifact.model_artifact_hash == row_artifact_hash and artifact.manifest_hash == row_manifest_hash:
        return {
            "ok": True,
            "action": "reregister-active-manifest",
            "status": "hashes_match_no_reregister_needed",
            "private_model_version_id": version_id,
        }
    if dry_run:
        return {
            "ok": True,
            "action": "reregister-active-manifest",
            "status": "would_reregister_active_manifest",
            "dry_run": True,
            "private_model_version_id": version_id,
            "mismatch": mismatch_doc,
        }
    await create_private_model_version_event(
        private_model_version_id=version_id,
        event_type="superseded",
        version_status="superseded",
        reason="operator_manifest_reregister",
        event_doc={
            "source": "research_lab_operator_manifest_reregister",
            "actor_ref": actor_ref,
            **mismatch_doc,
        },
    )
    version_row, _event = await create_private_model_version(
        artifact_manifest=artifact.to_dict(),
        manifest_uri=manifest_uri,
        redacted_version_doc={
            "source": "operator_manifest_reregister",
            "model_artifact_hash": artifact.model_artifact_hash,
            "private_model_manifest_hash": artifact.manifest_hash,
            "git_commit_sha": artifact.git_commit_sha,
            "component_registry_version": artifact.component_registry_version,
            "scoring_adapter_version": artifact.scoring_adapter_version,
            "superseded_private_model_version_id": version_id,
        },
        version_status="active",
        reason="operator_manifest_reregister",
    )
    new_version_id = str(version_row.get("private_model_version_id") or "")
    logger.warning(
        "research_lab_operator_manifest_reregistered: old=%s new=%s actor=%s",
        _short_ref(version_id),
        _short_ref(new_version_id),
        actor_ref,
    )
    return {
        "ok": True,
        "action": "reregister-active-manifest",
        "status": "reregistered_active_manifest",
        "dry_run": False,
        "superseded_private_model_version_id": version_id,
        "private_model_version_id": new_version_id,
        "mismatch": mismatch_doc,
    }


async def latest_public_benchmark_summary() -> dict[str, Any]:
    """Return the latest sanitized public benchmark report, or a safe fallback."""

    try:
        rows = await select_many(
            "research_lab_public_benchmark_report_current",
            filters=(("current_report_status", "published"),),
            order_by=(("benchmark_date", True), ("created_at", True)),
            limit=1,
        )
    except Exception as exc:
        logger.warning("research_lab_public_benchmark_summary_unavailable: %s", str(exc)[:200])
        rows = []
    if rows:
        report_doc = rows[0].get("report_doc")
        if isinstance(report_doc, Mapping):
            return dict(report_doc)
    return {
        "schema_version": "1.0",
        "report_type": "research_lab_public_daily_benchmark",
        "status": "unavailable",
        "guidance": "No sanitized daily benchmark report has been published yet.",
    }


class ResearchLabPromotionController:
    """Process scored candidates into active private model versions."""

    def __init__(self, config: ResearchLabGatewayConfig, *, worker_ref: str):
        self.config = config
        self.worker_ref = worker_ref

    async def process_scored_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        bypass_gates: frozenset[str] | set[str] = frozenset(),
    ) -> dict[str, Any]:
        candidate_parent = str(candidate.get("parent_artifact_hash") or score_bundle.get("parent_artifact_hash") or "")
        candidate_kind = str(candidate.get("candidate_kind") or "patch")
        metric = promotion_improvement_metric(score_bundle)
        improvement_points = float(metric.improvement_points)
        delta_lcb = float((score_bundle.get("aggregates") or {}).get("delta_lcb") or 0.0)
        threshold = float(self.config.improvement_threshold_points)
        rolling_window_hash = str(score_bundle.get("icp_set_hash") or "")
        score_bundle_id = str(score_bundle_row.get("score_bundle_id") or "")

        if not self.config.auto_promotion_enabled:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_checked",
                promotion_status="checked",
                active_parent_artifact_hash=candidate_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "delta_lcb": round(delta_lcb, 6),
                    "auto_commit_enabled": self.config.auto_commit_enabled,
                    "candidate_kind": candidate_kind,
                    "auto_promotion_enabled": False,
                    "promotion_metric": metric.event_doc(),
                },
            )
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_disabled",
                promotion_status="disabled",
                active_parent_artifact_hash=candidate_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "auto_promotion_enabled": False,
                    "auto_commit_enabled": self.config.auto_commit_enabled,
                    "candidate_kind": candidate_kind,
                    "delta_lcb": round(delta_lcb, 6),
                    "promotion_metric": metric.event_doc(),
                },
            )
            return {"status": "disabled"}

        active = await load_active_private_model(self.config, register_bootstrap=True)
        active_parent = active.artifact.model_artifact_hash

        # Re-drive idempotency: a candidate that already merged must never
        # merge again, and replay must not insert duplicate promotion_checked
        # events. Checked before the stale-parent branch because a merged
        # candidate's own merge makes its parent stale.
        merged_event = await candidate_already_promoted(str(candidate["candidate_id"]))
        if merged_event is not None:
            logger.info(
                "research_lab_promotion_already_promoted candidate=%s promotion_event=%s",
                _short_ref(candidate["candidate_id"]),
                _short_ref(merged_event.get("promotion_event_id")),
            )
            private_source_status = await self._maybe_finalize_missing_private_source_push(
                candidate=candidate,
                score_bundle_row=score_bundle_row,
                score_bundle=score_bundle,
                active=active,
                candidate_parent=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold=threshold,
            )
            reward_status = await self._maybe_finalize_missing_champion_reward(
                candidate=candidate,
                score_bundle_row=score_bundle_row,
                score_bundle=score_bundle,
                improvement_points=improvement_points,
                threshold=threshold,
            )
            return {
                "status": "already_promoted",
                "promotion_event_id": str(merged_event.get("promotion_event_id") or ""),
                "private_model_version_id": str(merged_event.get("private_model_version_id") or ""),
                "private_source_status": private_source_status,
                **reward_status,
            }

        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "delta_lcb": round(delta_lcb, 6),
                "auto_commit_enabled": self.config.auto_commit_enabled,
                "candidate_kind": candidate_kind,
                "promotion_metric": metric.event_doc(),
            },
        )

        if candidate_kind != "image_build":
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="unsupported_candidate_kind",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "candidate_kind": candidate_kind,
                    "reason": "patch_candidates_are_legacy_read_only",
                    "promotion_metric": metric.event_doc(),
                },
            )
            return {"status": "rejected_legacy_patch_candidate"}

        bypassed_gates: list[str] = []

        if metric.rejection_status:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="below_threshold",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "reason": metric.rejection_status,
                    "promotion_metric": metric.event_doc(),
                },
            )
            return {"status": metric.rejection_status, "bypassed_gates": bypassed_gates}

        if improvement_points < threshold:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="below_threshold",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "mean_delta": round(improvement_points, 6),
                    "delta_lcb": round(delta_lcb, 6),
                    "promotion_metric": metric.event_doc(),
                },
            )
            return {"status": "rejected_below_threshold"}

        if candidate_parent != active_parent:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=score_bundle_id,
                event_type="stale_parent_detected",
                promotion_status="rebase_required",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "candidate_kind": "image_build",
                    "action": "rescore_candidate_image_against_current_parent",
                    "promotion_metric": metric.event_doc(),
                },
            )
            return {"status": "stale_parent_needs_rescore"}

        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=score_bundle_id,
            event_type="promotion_passed",
            promotion_status="passed",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "auto_commit_enabled": self.config.auto_commit_enabled,
                "promotion_metric": metric.event_doc(),
            },
        )

        result = await self._promote_built_image_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            active=active,
            active_parent=active_parent,
            candidate_parent=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        if bypassed_gates:
            result = {**result, "bypassed_gates": bypassed_gates}
        return result

    async def _confirmation_rerun_gate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_id: str,
        metric: PromotionImprovementMetric,
        improvement_points: float,
        threshold: float,
        active_parent: str,
        candidate_parent: str,
        rolling_window_hash: str,
        holdout_gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """§5.2-2 confirmation decision for a candidate that cleared every gate.

        Event-sourced (state derives from promotion events, never memory):

        * no confirmation state -> write ``held_pending_confirmation`` (status
          ``checked`` — retryable hold, candidate stays scored) recording the
          first-pass delta and the benchmark bundle it was measured against;
        * recorded measurement -> promote only if its delta also clears
          ``RESEARCH_LAB_CONFIRMATION_MIN_DELTA``; otherwise a terminal
          ``rejected_confirmation_failed`` recording BOTH deltas;
        * measurement attempts exhausted (infra failures, bounded by the
          claim-attempt budget) -> terminal ``rejected_confirmation_failed``
          with ``failure_mode: confirmation_attempts_exhausted``.
        """

        candidate_id = str(candidate["candidate_id"])
        state = await load_confirmation_state(
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
        )
        min_delta = confirmation_min_delta(threshold)
        budget = confirmation_attempt_budget(self.config)
        result_event = state.get("result_event")
        if result_event is not None:
            confirmation_doc = confirmation_doc_from_event(result_event)
            confirmation_delta = _optional_float(confirmation_doc.get("confirmation_delta"))
            if confirmation_delta is not None and confirmation_delta >= min_delta:
                summary = {
                    "decision": "confirmation_passed",
                    "confirmation_delta": round(confirmation_delta, 6),
                    "confirmation_min_delta": round(min_delta, 6),
                    "first_pass_improvement_points": round(improvement_points, 6),
                    "recorded_promotion_event_id": str(result_event.get("promotion_event_id") or ""),
                    "rolling_window_hash": str(confirmation_doc.get("rolling_window_hash") or ""),
                    "window_match": bool(confirmation_doc.get("window_match", True)),
                    "attempts": int(state.get("attempts") or 0),
                }
                logger.info(
                    "research_lab_promotion_confirmation_passed candidate=%s delta=%.4f min_delta=%.4f first_pass=%.4f",
                    _short_ref(candidate_id),
                    confirmation_delta,
                    min_delta,
                    improvement_points,
                )
                return {"decision": "confirmed", "confirmation_summary": summary}
            failure_mode = (
                "confirmation_delta_below_min"
                if confirmation_delta is not None
                else "confirmation_delta_missing_from_recorded_measurement"
            )
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id,
                event_type="below_threshold",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "reason": CONFIRMATION_REJECTED_REASON,
                    "decision_path": "confirmation_rerun_gate",
                    "failure_mode": failure_mode,
                    "first_pass_improvement_points": round(improvement_points, 6),
                    "confirmation_delta": (
                        round(confirmation_delta, 6) if confirmation_delta is not None else None
                    ),
                    "confirmation_min_delta": round(min_delta, 6),
                    "recorded_promotion_event_id": str(result_event.get("promotion_event_id") or ""),
                    "confirmation": confirmation_doc,
                    "promotion_metric": metric.event_doc(),
                },
            )
            logger.warning(
                "research_lab_promotion_confirmation_failed candidate=%s first_pass=%.4f confirmation=%s min_delta=%.4f",
                _short_ref(candidate_id),
                improvement_points,
                f"{confirmation_delta:.4f}" if confirmation_delta is not None else "missing",
                min_delta,
            )
            return {
                "decision": "rejected",
                "status": CONFIRMATION_REJECTED_REASON,
                "first_pass_improvement_points": round(improvement_points, 6),
                "confirmation_delta": (
                    round(confirmation_delta, 6) if confirmation_delta is not None else None
                ),
                "confirmation_min_delta": round(min_delta, 6),
            }

        attempts = int(state.get("attempts") or 0)
        if attempts >= budget:
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id,
                event_type="below_threshold",
                promotion_status="rejected",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "reason": CONFIRMATION_REJECTED_REASON,
                    "decision_path": "confirmation_rerun_gate",
                    "failure_mode": "confirmation_attempts_exhausted",
                    "first_pass_improvement_points": round(improvement_points, 6),
                    "confirmation_delta": None,
                    "confirmation_min_delta": round(min_delta, 6),
                    "confirmation_attempts": attempts,
                    "confirmation_attempt_budget": budget,
                    "promotion_metric": metric.event_doc(),
                },
            )
            logger.warning(
                "research_lab_promotion_confirmation_attempts_exhausted candidate=%s attempts=%s budget=%s",
                _short_ref(candidate_id),
                attempts,
                budget,
            )
            return {
                "decision": "rejected",
                "status": CONFIRMATION_REJECTED_REASON,
                "failure_mode": "confirmation_attempts_exhausted",
                "first_pass_improvement_points": round(improvement_points, 6),
                "confirmation_delta": None,
                "confirmation_min_delta": round(min_delta, 6),
            }

        if state.get("held_event") is None:
            # Written exactly once per candidate+bundle: the hold event is the
            # worker's discovery marker and never goes away, so re-drives that
            # land back here (e.g. an admin replay while a measurement is in
            # flight) must not spam duplicate holds — and must not reset the
            # state machine under a live started claim.
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_checked",
                promotion_status="checked",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "reason": CONFIRMATION_HOLD_REASON,
                    "decision_path": "confirmation_rerun_hold",
                    "first_pass_improvement_points": round(improvement_points, 6),
                    "confirmation_min_delta": round(min_delta, 6),
                    "confirmation_attempts": attempts,
                    "confirmation_attempt_budget": budget,
                    "baseline_benchmark_bundle_id": str(
                        holdout_gate.get("baseline_benchmark_bundle_id") or ""
                    ),
                    "promotion_metric": metric.event_doc(),
                },
            )
        logger.info(
            "research_lab_promotion_held_pending_confirmation candidate=%s first_pass=%.4f min_delta=%.4f attempts=%s/%s",
            _short_ref(candidate_id),
            improvement_points,
            min_delta,
            attempts,
            budget,
        )
        return {
            "decision": "held",
            "status": "held_pending_confirmation",
            "first_pass_improvement_points": round(improvement_points, 6),
            "confirmation_min_delta": round(min_delta, 6),
        }

    async def _promote_built_image_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        active: ActivePrivateModel,
        active_parent: str,
        candidate_parent: str,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(manifest_doc, Mapping):
            raise RuntimeError("image_build candidate missing candidate_model_manifest_doc")
        new_artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
        errors = validate_private_model_artifact_manifest(new_artifact)
        if errors:
            raise RuntimeError("candidate image manifest failed validation: " + "; ".join(errors))
        if str(score_bundle.get("candidate_artifact_hash") or "") != new_artifact.model_artifact_hash:
            raise RuntimeError("score bundle candidate artifact does not match built image manifest")
        private_repo_result = await self._maybe_push_private_repo_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            active=active,
            new_artifact=new_artifact,
            active_parent=active_parent,
            candidate_parent=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        if private_repo_result.get("status") == "failed":
            return private_repo_result
        # Bug #3 design note: the one-active DB guard (scripts/60) enforces at
        # most one version whose latest event is 'active', so the order here
        # must stay supersede -> create (create-first would conflict with the
        # still-active previous champion). The two writes are adjacent (nothing
        # slow runs between them); a create failure re-activates the previous
        # champion below, and a process kill between the writes is repaired by
        # reconcile_active_private_model_lineage (re-activate newest
        # superseded) instead of the old silent bootstrap rollback.
        if active.version_row:
            await create_private_model_version_event(
                private_model_version_id=str(active.version_row["private_model_version_id"]),
                event_type="superseded",
                version_status="superseded",
                reason="superseded_by_research_lab_image_build_promotion",
                event_doc={"source_candidate_id": str(candidate["candidate_id"])},
            )
        try:
            version_row, _version_event = await create_private_model_version(
                artifact_manifest=new_artifact.to_dict(),
                manifest_uri=new_artifact.manifest_uri,
                source_candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                redacted_version_doc={
                    "source": "gateway_code_edit_image_build",
                    "model_artifact_hash": new_artifact.model_artifact_hash,
                    "private_model_manifest_hash": new_artifact.manifest_hash,
                    "git_commit_sha": new_artifact.git_commit_sha,
                    "component_registry_version": new_artifact.component_registry_version,
                    "scoring_adapter_version": new_artifact.scoring_adapter_version,
                    "candidate_source_diff_hash": candidate.get("candidate_source_diff_hash"),
                },
                version_status="active",
                reason="research_lab_image_build_candidate_promoted",
            )
        except Exception:
            if active.version_row:
                try:
                    await create_private_model_version_event(
                        private_model_version_id=str(active.version_row["private_model_version_id"]),
                        event_type="active",
                        version_status="active",
                        reason="promotion_create_failed_reactivate_previous_champion",
                        event_doc={"source_candidate_id": str(candidate["candidate_id"])},
                    )
                except Exception:
                    logger.exception(
                        "research_lab_promotion_previous_champion_reactivate_failed: version=%s "
                        "(zero active versions until reconcile-active-lineage runs)",
                        _short_ref(active.version_row.get("private_model_version_id")),
                    )
            raise
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            private_model_version_id=str(version_row["private_model_version_id"]),
            event_type="active_version_created",
            promotion_status="merged",
            active_parent_artifact_hash=active_parent,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "new_model_artifact_hash": new_artifact.model_artifact_hash,
                "candidate_kind": "image_build",
            },
        )
        reward_status = await self._maybe_create_champion_reward(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        return {
            "status": "merged",
            "private_model_version_id": str(version_row["private_model_version_id"]),
            **reward_status,
        }

    async def _maybe_push_private_repo_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        active: ActivePrivateModel,
        new_artifact: PrivateModelArtifactManifest,
        active_parent: str,
        candidate_parent: str,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        if not self.config.auto_commit_enabled:
            return {"status": "skipped_auto_commit_disabled"}
        if not self.config.private_repo_url:
            return {"status": "skipped_private_source_repo_not_configured"}

        candidate_id = str(candidate["candidate_id"])
        score_bundle_id = str(score_bundle_row["score_bundle_id"])
        branch_name = str(self.config.private_repo_branch or "main")
        repo_ref_hash = canonical_hash(
            {
                "repo_url": self.config.private_repo_url,
                "branch_name": branch_name,
            }
        )
        existing_attempt_events = await select_many(
            "research_lab_private_repo_commit_events",
            columns="commit_event_id,commit_status,created_at",
            filters=(("candidate_id", candidate_id), ("score_bundle_id", score_bundle_id)),
            order_by=(("created_at", True),),
            limit=100,
        )
        source_push_attempt = len(existing_attempt_events) + 1
        event_base = {
            "source": "research_lab_source_push",
            "source_push_attempt": source_push_attempt,
            "candidate_kind": "image_build",
            "candidate_model_artifact_hash": new_artifact.model_artifact_hash,
            "candidate_source_diff_hash": candidate.get("candidate_source_diff_hash"),
            "active_parent_artifact_hash": active_parent,
            "candidate_parent_artifact_hash": candidate_parent,
        }
        await create_private_repo_commit_event(
            commit_status="started",
            branch_name=branch_name,
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            private_repo_ref_hash=repo_ref_hash,
            event_doc={**event_base, "stage": "started"},
        )
        try:
            result = await asyncio.to_thread(
                _push_candidate_source_diff_to_repo,
                repo_url=self.config.private_repo_url,
                branch_name=branch_name,
                active_git_commit_sha=active.artifact.git_commit_sha,
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                candidate_build_doc=candidate.get("candidate_build_doc"),
                candidate_model_manifest_doc=candidate.get("candidate_model_manifest_doc"),
            )
        except Exception as exc:
            error_hash = canonical_hash({"error": str(exc)})
            # Bug #29 reporting: name the head-mismatch cause explicitly so the
            # wedge is diagnosable from events instead of an opaque error hash.
            failure_detail: dict[str, Any] = {}
            if isinstance(exc, RepoHeadMismatchError):
                failure_detail = {
                    "failure_reason": "repo_head_mismatch",
                    "repo_head_sha_prefix": exc.head[:12],
                    "active_manifest_git_sha_prefix": exc.expected_sha[:12],
                    "recover_flag_env": AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV,
                }
            await create_private_repo_commit_event(
                commit_status="failed",
                branch_name=branch_name,
                candidate_id=candidate_id,
                score_bundle_id=score_bundle_id,
                private_repo_ref_hash=repo_ref_hash,
                event_doc={
                    **event_base,
                    "stage": "failed",
                    "error_hash": error_hash,
                    "error_class": type(exc).__name__,
                    **failure_detail,
                },
            )
            await create_candidate_promotion_event(
                candidate_id=candidate_id,
                source_score_bundle_id=score_bundle_id,
                event_type="promotion_failed",
                promotion_status="failed",
                active_parent_artifact_hash=active_parent,
                candidate_parent_artifact_hash=candidate_parent,
                rolling_window_hash=rolling_window_hash,
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={
                    "reason": "private_source_push_failed",
                    "error_hash": error_hash,
                    "error_class": type(exc).__name__,
                    "candidate_status_preserved": "scored",
                    "source_push_attempt": source_push_attempt,
                    **failure_detail,
                },
            )
            logger.warning(
                "research_lab_private_source_push_failed candidate=%s score_bundle=%s error_hash=%s cause=%s",
                _short_ref(candidate_id),
                _short_ref(score_bundle_id),
                error_hash,
                failure_detail.get("failure_reason") or type(exc).__name__,
            )
            return {
                "status": "failed",
                "reason": "private_source_push_failed",
                "error_hash": error_hash,
                **failure_detail,
            }

        await create_private_repo_commit_event(
            commit_status="pushed" if result.get("status") == "pushed" else "committed",
            branch_name=branch_name,
            candidate_id=candidate_id,
            score_bundle_id=score_bundle_id,
            git_commit_sha=str(result.get("git_commit_sha") or "") or None,
            private_repo_ref_hash=repo_ref_hash,
            event_doc={
                **event_base,
                "stage": str(result.get("status") or "pushed"),
                "target_files": list(result.get("target_files") or []),
                "source_diff_hash": str(result.get("source_diff_hash") or ""),
            },
        )
        return {"status": "private_source_pushed", **result}

    async def _maybe_create_champion_reward(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        uid = await _resolve_miner_uid(str(candidate["miner_hotkey"]))
        if uid is None:
            await create_candidate_promotion_event(
                candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=str(score_bundle_row.get("score_bundle_id") or ""),
                event_type="champion_reward_pending_uid",
                promotion_status="reward_pending_uid",
                improvement_points=improvement_points,
                threshold_points=threshold,
                worker_ref=self.worker_ref,
                event_doc={"miner_hotkey_ref": canonical_hash(str(candidate["miner_hotkey"]))},
            )
            return {"champion_reward_status": "uid_resolution_pending"}
        policy = {
            **self.config.reimbursement_policy_doc(enabled=True),
            "champion_threshold_points": threshold,
            "reward_epochs": self.config.lab_reward_epochs,
        }
        # The reward window starts at reward-creation (merge) time, never at the
        # bundle's scoring epoch: a candidate scored at epoch N but merged at
        # N+15 would otherwise have ~75% of its window expired on creation (the
        # 2026-07-02 incident paid ~2.5h of a ~24h window). The bundle epoch is
        # kept for provenance; start_epoch comes from the live chain epoch.
        evaluation_epoch = int(score_bundle.get("evaluation_epoch") or self.config.evaluation_epoch or 0)
        try:
            current_epoch, _block, _epoch_source = await resolve_research_lab_evaluation_epoch(
                self.config.evaluation_epoch
            )
        except Exception as exc:
            logger.warning(
                "research_lab_champion_reward_epoch_resolution_failed_using_bundle_epoch "
                "candidate=%s error=%s",
                _short_ref(candidate["candidate_id"]),
                str(exc)[:200],
            )
            current_epoch = evaluation_epoch
        obligation_input = {
            "uid": uid,
            "miner_uid": uid,
            "miner_hotkey": str(candidate["miner_hotkey"]),
            "island": str(candidate.get("island") or "generalist"),
            "candidate_id": str(candidate["candidate_id"]),
            "score_bundle_id": str(score_bundle_row["score_bundle_id"]),
            "run_id": str(candidate["run_id"]),
            "evaluation_epoch": evaluation_epoch,
            "start_epoch": max(current_epoch, evaluation_epoch) + 1,
            "improvement_points": improvement_points,
            "threshold_points": threshold,
            "daily_icp_counts": _daily_counts_from_score_bundle(score_bundle),
        }
        obligation = build_champion_reward_obligation(obligation_input, policy)
        if obligation["status"] != "active":
            return {"champion_reward_status": obligation["status"]}
        row, _event = await create_champion_reward_obligation(
            obligation=obligation,
            ticket_id=str(candidate["ticket_id"]),
            obligation_doc={
                "policy_id": str(policy["policy_id"]),
                "source": "gateway_promotion_event",
                "source_score_bundle_hash": str(score_bundle.get("score_bundle_hash") or ""),
            },
        )
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            event_type="champion_reward_created",
            promotion_status="reward_created",
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={"champion_reward_id": str(row["champion_reward_id"])},
        )
        return {"champion_reward_status": "created", "champion_reward_id": str(row["champion_reward_id"])}

    async def _maybe_finalize_missing_private_source_push(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        active: ActivePrivateModel,
        candidate_parent: str,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        candidate_id = str(candidate["candidate_id"])
        score_bundle_id = str(score_bundle_row["score_bundle_id"])
        existing_events = await select_many(
            "research_lab_private_repo_commit_events",
            columns="commit_event_id,commit_status,git_commit_sha,created_at",
            filters=(("candidate_id", candidate_id), ("score_bundle_id", score_bundle_id)),
            order_by=(("created_at", True),),
            limit=10,
        )
        successful_events = [
            event for event in existing_events if str(event.get("commit_status") or "") in {"committed", "pushed"}
        ]
        if successful_events:
            event = successful_events[0]
            return {
                "status": "already_recorded",
                "commit_status": str(event.get("commit_status") or ""),
                "commit_event_id": str(event.get("commit_event_id") or ""),
            }
        if not self.config.auto_commit_enabled:
            return {"status": "skipped_auto_commit_disabled"}
        manifest_doc = candidate.get("candidate_model_manifest_doc")
        if not isinstance(manifest_doc, Mapping):
            return {"status": "skipped_candidate_manifest_missing"}
        try:
            new_artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
            errors = validate_private_model_artifact_manifest(new_artifact)
            if errors:
                return {"status": "skipped_candidate_manifest_invalid", "errors": errors[:5]}
        except Exception as exc:
            return {"status": "skipped_candidate_manifest_invalid", "error": type(exc).__name__}
        return await self._maybe_push_private_repo_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            active=active,
            new_artifact=new_artifact,
            active_parent=candidate_parent,
            candidate_parent=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )

    async def _maybe_finalize_missing_champion_reward(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        created_events = await select_many(
            "research_lab_candidate_promotion_events",
            columns="promotion_event_id,event_type,created_at",
            filters=(("candidate_id", str(candidate["candidate_id"])), ("event_type", "champion_reward_created")),
            order_by=(("created_at", True),),
            limit=1,
        )
        if created_events:
            return {
                "champion_reward_status": "already_created",
                "champion_reward_event_id": str(created_events[0].get("promotion_event_id") or ""),
            }
        return await self._maybe_create_champion_reward(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            improvement_points=improvement_points,
            threshold=threshold,
        )


async def reconcile_pending_champion_rewards(
    config: ResearchLabGatewayConfig,
    *,
    worker_ref: str,
    candidate_ids: Sequence[str] | None = None,
    limit: int = 25,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Finalize champion rewards stuck at ``reward_pending_uid`` (bug #24).

    A transient metagraph/UID resolution failure at promotion time leaves a
    ``champion_reward_pending_uid`` event with no reconciler, permanently
    losing the reward. This re-resolves the UID and creates the obligation +
    ``champion_reward_created`` event via the normal promotion path. Designed
    to be callable from a periodic maintenance pass (follow-up: wire into the
    worker cycle) or the ``reconcile-champion-rewards`` admin command. UIDs
    that still fail to resolve are reported (no event spam) and retried on the
    next run.
    """

    pending_events = await select_many(
        "research_lab_candidate_promotion_events",
        columns=(
            "promotion_event_id,candidate_id,source_score_bundle_id,"
            "improvement_points,threshold_points,created_at"
        ),
        filters=(("event_type", "champion_reward_pending_uid"),),
        order_by=(("created_at", True),),
        limit=200,
    )
    wanted = {str(item) for item in candidate_ids} if candidate_ids else None
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    controller = ResearchLabPromotionController(config, worker_ref=worker_ref)
    for event in pending_events:
        candidate_id = str(event.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        if wanted is not None and candidate_id not in wanted:
            continue
        if len(results) >= max(1, int(limit)):
            break
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "pending_promotion_event_id": str(event.get("promotion_event_id") or ""),
        }
        created_events = await select_many(
            "research_lab_candidate_promotion_events",
            columns="promotion_event_id,event_type,created_at",
            filters=(("candidate_id", candidate_id), ("event_type", "champion_reward_created")),
            order_by=(("created_at", True),),
            limit=1,
        )
        if created_events:
            entry["status"] = "already_created"
            entry["champion_reward_event_id"] = str(created_events[0].get("promotion_event_id") or "")
            results.append(entry)
            continue
        candidate = await select_one(
            "research_lab_candidate_evaluation_current",
            filters=(("candidate_id", candidate_id),),
        )
        if not candidate:
            entry["status"] = "candidate_not_found"
            results.append(entry)
            continue
        score_bundle_id = str(
            event.get("source_score_bundle_id")
            or candidate.get("current_score_bundle_id")
            or ""
        )
        bundle_row = (
            await select_one(
                "research_evaluation_score_bundle_current",
                filters=(("score_bundle_id", score_bundle_id),),
            )
            if score_bundle_id
            else None
        )
        score_bundle = bundle_row.get("score_bundle_doc") if isinstance(bundle_row, Mapping) else None
        if not isinstance(score_bundle, Mapping):
            entry["status"] = "score_bundle_not_found"
            entry["score_bundle_id"] = score_bundle_id
            results.append(entry)
            continue
        uid = await _resolve_miner_uid(str(candidate.get("miner_hotkey") or ""))
        if uid is None:
            entry["status"] = "uid_still_unresolved"
            results.append(entry)
            continue
        entry["resolved_uid"] = int(uid)
        if dry_run:
            entry["status"] = "would_create_champion_reward"
            results.append(entry)
            continue
        reward_status = await controller._maybe_create_champion_reward(
            candidate=candidate,
            score_bundle_row=bundle_row,
            score_bundle=score_bundle,
            improvement_points=float(event.get("improvement_points") or 0.0),
            threshold=float(event.get("threshold_points") or 0.0),
        )
        entry["status"] = str(reward_status.get("champion_reward_status") or "unknown")
        if reward_status.get("champion_reward_id"):
            entry["champion_reward_id"] = str(reward_status["champion_reward_id"])
        results.append(entry)
    finalized = sum(1 for item in results if item.get("status") in {"created", "would_create_champion_reward"})
    return {
        "ok": True,
        "action": "reconcile-champion-rewards",
        "dry_run": dry_run,
        "found_pending": len(results),
        "finalized": finalized,
        "results": results,
    }


def _load_valid_artifact(uri: str) -> PrivateModelArtifactManifest:
    artifact = PrivateModelArtifactManifest.from_mapping(load_private_artifact_manifest(uri))
    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise RuntimeError("private artifact manifest failed validation: " + "; ".join(errors))
    return artifact


async def _resolve_miner_uid(hotkey: str) -> int | None:
    try:
        from gateway.qualification.utils.chain import get_metagraph

        metagraph = await get_metagraph()
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        return hotkeys.index(hotkey) if hotkey in hotkeys else None
    except Exception as exc:
        logger.warning("research_lab_miner_uid_resolution_failed: %s", str(exc)[:200])
        return None


def _daily_counts_from_score_bundle(score_bundle: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    for item in aggregates.get("per_icp_results", []) or []:
        ref = str(item.get("icp_ref") or "")
        match = re.search(r"qualification_private_icp_sets:(\d+):", ref)
        day = match.group(1) if match else ref.split(":")[0]
        if day:
            counts[day] = counts.get(day, 0) + 1
    return counts


def _push_candidate_source_diff_to_repo(
    *,
    repo_url: str,
    branch_name: str,
    active_git_commit_sha: str,
    candidate_id: str,
    score_bundle_id: str,
    candidate_build_doc: Any,
    candidate_model_manifest_doc: Any,
) -> dict[str, Any]:
    if not isinstance(candidate_build_doc, Mapping):
        raise RuntimeError("image-build candidate missing candidate_build_doc")
    if not isinstance(candidate_model_manifest_doc, Mapping):
        raise RuntimeError("image-build candidate missing candidate_model_manifest_doc")
    source_diff_uri = str(candidate_build_doc.get("source_diff_artifact_uri") or "")
    if not source_diff_uri.startswith("s3://"):
        raise RuntimeError("candidate source diff artifact is missing or unsupported")
    source_diff_text = _run_command(
        ["aws", "s3", "cp", source_diff_uri, "-"],
        cwd=None,
        timeout_seconds=30,
        redact=True,
    )
    try:
        source_diff_doc = json.loads(source_diff_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("candidate source diff artifact is not valid JSON") from exc
    unified_diff = str(source_diff_doc.get("unified_diff") or "")
    if not unified_diff.startswith("diff --git "):
        raise RuntimeError("candidate source diff artifact does not contain a git unified diff")
    target_files = _safe_target_files(source_diff_doc.get("target_files"))
    if not target_files:
        raise RuntimeError("candidate source diff artifact has no target files")
    source_diff_hash = str(source_diff_doc.get("source_diff_hash") or candidate_build_doc.get("source_diff_hash") or "")
    candidate_manifest_sha = str(candidate_model_manifest_doc.get("git_commit_sha") or "")

    tmp_dir = Path(tempfile.mkdtemp(prefix="research-lab-private-source-push-"))
    try:
        worktree = tmp_dir / "repo"
        _run_command(
            ["git", "clone", "--branch", branch_name, "--single-branch", repo_url, str(worktree)],
            cwd=None,
            timeout_seconds=120,
            redact=True,
        )
        head = _run_command(["git", "rev-parse", "HEAD"], cwd=worktree, timeout_seconds=10).strip()
        active_sha = str(active_git_commit_sha or "").strip()
        if active_sha and head[: len(active_sha)] != active_sha:
            # Bug #29 wedge: the active manifest's sha comes from a throwaway
            # `git init`, so after the first push the branch head never matches
            # again and every subsequent auto-commit hard-fails here. The
            # hard-fail is deliberately kept as the default (it is currently
            # the accidental guard against noise-merges, §8.3); the recovery
            # flag re-resolves against the current head and must only be
            # enabled once the baseline health gate is live.
            if _env_flag(AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV):
                logger.warning(
                    "research_lab_auto_commit_head_mismatch_recovered: head=%s active_manifest_sha=%s "
                    "(%s enabled; applying candidate diff against current branch head)",
                    head[:12],
                    active_sha[:12],
                    AUTO_COMMIT_HEAD_MISMATCH_RECOVER_ENV,
                )
            else:
                raise RepoHeadMismatchError(head=head, expected_sha=active_sha)

        patch_path = tmp_dir / "candidate.patch"
        patch_text = unified_diff
        patch_normalized = False
        patch_path.write_text(patch_text, encoding="utf-8")
        check = _run_command_result(["git", "apply", "--check", str(patch_path)], cwd=worktree, timeout_seconds=30)
        if check.returncode != 0 and "corrupt patch" in ((check.stderr or "") + (check.stdout or "")):
            normalized = _normalize_unified_diff_hunk_headers(patch_text)
            if normalized != patch_text:
                patch_text = normalized
                patch_normalized = True
                patch_path.write_text(patch_text, encoding="utf-8")
                check = _run_command_result(
                    ["git", "apply", "--check", str(patch_path)],
                    cwd=worktree,
                    timeout_seconds=30,
                )
        if check.returncode != 0:
            reverse = _run_command_result(
                ["git", "apply", "--reverse", "--check", str(patch_path)],
                cwd=worktree,
                timeout_seconds=30,
            )
            if reverse.returncode == 0:
                return {
                    "status": "already_applied",
                    "git_commit_sha": head,
                    "candidate_manifest_git_commit_sha": candidate_manifest_sha,
                    "target_files": target_files,
                    "source_diff_hash": source_diff_hash,
                    "patch_normalized": patch_normalized,
                }
            raise RuntimeError("candidate source diff does not apply to private source branch")

        _run_command(["git", "apply", str(patch_path)], cwd=worktree, timeout_seconds=30)
        status = _run_command(["git", "status", "--porcelain"], cwd=worktree, timeout_seconds=10)
        if not status.strip():
            return {
                "status": "already_applied",
                "git_commit_sha": head,
                "candidate_manifest_git_commit_sha": candidate_manifest_sha,
                "target_files": target_files,
                "source_diff_hash": source_diff_hash,
                "patch_normalized": patch_normalized,
            }
        _run_command(["git", "config", "user.name", os.getenv("RESEARCH_LAB_PRIVATE_REPO_GIT_AUTHOR_NAME", "Leadpoet Research Lab")], cwd=worktree, timeout_seconds=10)
        _run_command(["git", "config", "user.email", os.getenv("RESEARCH_LAB_PRIVATE_REPO_GIT_AUTHOR_EMAIL", "research-lab@leadpoet.ai")], cwd=worktree, timeout_seconds=10)
        _run_command(["git", "add", "--", *target_files], cwd=worktree, timeout_seconds=10)
        short_candidate = _short_ref(candidate_id).replace(":", "-")
        commit_message = (
            f"Promote Research Lab candidate {short_candidate}\n\n"
            f"Candidate: {_short_ref(candidate_id)}\n"
            f"Score bundle: {_short_ref(score_bundle_id)}\n"
            f"Source diff: {_short_ref(source_diff_hash)}\n"
        )
        _run_command(["git", "commit", "-m", commit_message], cwd=worktree, timeout_seconds=30)
        new_head = _run_command(["git", "rev-parse", "HEAD"], cwd=worktree, timeout_seconds=10).strip()
        _run_command(["git", "push", "origin", f"HEAD:{branch_name}"], cwd=worktree, timeout_seconds=120, redact=True)
        return {
            "status": "pushed",
            "git_commit_sha": new_head,
            "candidate_manifest_git_commit_sha": candidate_manifest_sha,
            "target_files": target_files,
            "source_diff_hash": source_diff_hash,
            "patch_normalized": patch_normalized,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _normalize_unified_diff_hunk_headers(unified_diff: str) -> str:
    """Repair incorrect hunk line counts while preserving diff contents."""
    lines = unified_diff.splitlines()
    hunk_header = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@(?P<suffix>.*)$"
    )
    out = list(lines)
    index = 0
    while index < len(lines):
        match = hunk_header.match(lines[index])
        if not match:
            index += 1
            continue
        old_count = 0
        new_count = 0
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].startswith("diff --git ") and not lines[cursor].startswith("@@ "):
            line = lines[cursor]
            if line.startswith("\\"):
                cursor += 1
                continue
            if line.startswith("+"):
                new_count += 1
            elif line.startswith("-"):
                old_count += 1
            else:
                old_count += 1
                new_count += 1
            cursor += 1
        out[index] = (
            f"@@ -{match.group('old_start')},{old_count} "
            f"+{match.group('new_start')},{new_count} @@{match.group('suffix')}"
        )
        index = cursor
    return "\n".join(out) + ("\n" if unified_diff.endswith("\n") else "")


def _safe_target_files(value: Any) -> list[str]:
    files: list[str] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            text = str(item or "").strip()
            if not text or text.startswith("/") or ".." in Path(text).parts:
                continue
            if re.search(r"(^|/)(\.git|\.github|\.env|Dockerfile|requirements[^/]*\.txt|poetry\.lock|uv\.lock)$", text):
                continue
            files.append(text)
    return files[:20]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None,
    timeout_seconds: int,
    redact: bool = False,
) -> str:
    result = _run_command_result(cmd, cwd=cwd, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if redact:
            detail = _safe_text(detail)
        raise RuntimeError(f"command failed: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}: {detail[:500]}")
    return result.stdout


def _run_command_result(
    cmd: Sequence[str],
    *,
    cwd: Path | None,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=max(1, int(timeout_seconds)),
        check=False,
    )


def _safe_text(value: str) -> str:
    text = value or ""
    for marker in ("sk-or-", "service_role", "openrouter_api_key"):
        text = text.replace(marker, "[redacted]")
    return text[:500]


def _short_ref(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 24:
        return text
    return f"{text[:14]}...{text[-6:]}"
