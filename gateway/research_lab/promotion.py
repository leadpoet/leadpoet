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

from gateway.research_lab.bundles import contains_secret_material, sha256_json
from gateway.research_lab.chain import resolve_research_lab_evaluation_epoch
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.public_benchmarks import build_public_benchmark_report
from gateway.research_lab.store import (
    canonical_hash,
    create_candidate_promotion_event,
    create_champion_reward_obligation,
    create_private_model_benchmark_bundle,
    create_private_model_version,
    create_private_model_version_event,
    create_private_repo_commit_event,
    create_public_benchmark_report,
    select_many,
    select_one,
)
from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.eval import (
    PrivateModelArtifactManifest,
    load_private_artifact_manifest,
    sign_digest_with_kms,
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

REPO_HEAD_SYNC_ENABLED_ENV = "RESEARCH_LAB_SYNC_ACTIVE_MODEL_TO_REPO_HEAD"
REPO_HEAD_MANIFEST_WAIT_SECONDS_ENV = "RESEARCH_LAB_REPO_HEAD_MANIFEST_WAIT_SECONDS"
REPO_HEAD_MANIFEST_POLL_SECONDS_ENV = "RESEARCH_LAB_REPO_HEAD_MANIFEST_POLL_SECONDS"
REPO_HEAD_GIT_TIMEOUT_SECONDS_ENV = "RESEARCH_LAB_REPO_HEAD_GIT_TIMEOUT_SECONDS"
DEFAULT_REPO_HEAD_MANIFEST_WAIT_SECONDS = 600
DEFAULT_REPO_HEAD_MANIFEST_POLL_SECONDS = 15
DEFAULT_REPO_HEAD_GIT_TIMEOUT_SECONDS = 20

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


class RepoHeadManifestNotReadyError(PromotionPausedError):
    """GitHub main is ahead of the signed current.json manifest."""

    def __init__(self, *, repo_main_sha: str, current_json_git_sha: str):
        self.repo_main_sha = str(repo_main_sha or "")
        self.current_json_git_sha = str(current_json_git_sha or "")
        super().__init__(
            "repo_head_manifest_not_ready: private repo main is ahead of current.json "
            f"(repo_main={self.repo_main_sha[:12]}, current_json={self.current_json_git_sha[:12]}); "
            "daily benchmark deferred so it cannot run a stale active artifact"
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


def repo_head_sync_enabled() -> bool:
    return _env_flag(REPO_HEAD_SYNC_ENABLED_ENV, "true")


def _repo_head_git_timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv(REPO_HEAD_GIT_TIMEOUT_SECONDS_ENV, DEFAULT_REPO_HEAD_GIT_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_REPO_HEAD_GIT_TIMEOUT_SECONDS


def _repo_head_manifest_wait_seconds(default: int = DEFAULT_REPO_HEAD_MANIFEST_WAIT_SECONDS) -> int:
    try:
        return max(0, int(os.getenv(REPO_HEAD_MANIFEST_WAIT_SECONDS_ENV, default)))
    except ValueError:
        return int(default)


def _repo_head_manifest_poll_seconds() -> int:
    try:
        return max(1, int(os.getenv(REPO_HEAD_MANIFEST_POLL_SECONDS_ENV, DEFAULT_REPO_HEAD_MANIFEST_POLL_SECONDS)))
    except ValueError:
        return DEFAULT_REPO_HEAD_MANIFEST_POLL_SECONDS


def _git_sha_matches(left: Any, right: Any) -> bool:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < len(b):
        return b.startswith(a)
    if len(b) < len(a):
        return a.startswith(b)
    return False


def _resolve_private_repo_head_sha(
    *,
    repo_url: str,
    branch_name: str,
    timeout_seconds: int | None = None,
) -> str:
    """Resolve the exact private repo branch SHA without cloning the repo."""

    output = _run_command(
        ["git", "ls-remote", repo_url, f"refs/heads/{branch_name}"],
        cwd=None,
        timeout_seconds=timeout_seconds or _repo_head_git_timeout_seconds(),
        redact=True,
    ).strip()
    if not output:
        raise RuntimeError(f"private repo branch not found: {branch_name}")
    first = output.splitlines()[0].split()
    sha = first[0] if first else ""
    if not re.match(r"^[0-9a-f]{40}$", sha):
        raise RuntimeError(f"private repo branch returned invalid sha for {branch_name}")
    return sha


async def _active_private_model_version_rows() -> list[dict[str, Any]]:
    rows = await select_many(
        "research_lab_private_model_version_current",
        columns=(
            "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
            "private_model_manifest_uri,git_commit_sha,current_version_status,current_status_at,"
            "source_candidate_id,source_score_bundle_id,source_benchmark_bundle_id,created_at"
        ),
        filters=(("current_version_status", "active"),),
        order_by=(("current_status_at", True),),
        limit=5,
    )
    return [dict(row) for row in rows]


async def private_repo_head_alignment_status(
    config: ResearchLabGatewayConfig,
    *,
    current_artifact: PrivateModelArtifactManifest | None = None,
) -> dict[str, Any]:
    """Read-only status for active lineage vs private repo main/current.json."""

    if not config.private_repo_url:
        return {
            "ok": False,
            "status": "private_repo_not_configured",
            "active_is_repo_head": False,
        }
    branch_name = str(config.private_repo_branch or "main")
    try:
        repo_main_sha = await asyncio.to_thread(
            _resolve_private_repo_head_sha,
            repo_url=config.private_repo_url,
            branch_name=branch_name,
        )
    except Exception as exc:
        return {
            "ok": False,
            "status": "repo_head_unavailable",
            "repo_branch": branch_name,
            "active_is_repo_head": False,
            "error": _safe_text(str(exc))[:200],
        }
    try:
        artifact = current_artifact or await asyncio.to_thread(_load_valid_artifact, config.private_model_manifest_uri)
    except Exception as exc:
        return {
            "ok": False,
            "status": "current_json_unavailable",
            "repo_branch": branch_name,
            "repo_main_sha": repo_main_sha,
            "active_is_repo_head": False,
            "error": _safe_text(str(exc))[:200],
        }
    active_rows = await _active_private_model_version_rows()
    active_row = active_rows[0] if active_rows else {}
    active_git_sha = str(active_row.get("git_commit_sha") or "")
    active_manifest_hash = str(active_row.get("private_model_manifest_hash") or "")
    current_json_git_sha = str(artifact.git_commit_sha or "")
    current_json_matches_repo_head = _git_sha_matches(current_json_git_sha, repo_main_sha)
    active_manifest_matches_current_json = bool(active_manifest_hash and active_manifest_hash == artifact.manifest_hash)
    active_is_repo_head = bool(
        current_json_matches_repo_head
        and active_manifest_matches_current_json
        and _git_sha_matches(active_git_sha, repo_main_sha)
    )
    status = "active_is_repo_head" if active_is_repo_head else "active_not_repo_head"
    if not current_json_matches_repo_head:
        status = "repo_head_manifest_not_ready"
    elif len(active_rows) > 1:
        status = "duplicate_active_versions"
    return {
        "ok": status == "active_is_repo_head",
        "status": status,
        "repo_branch": branch_name,
        "repo_main_sha": repo_main_sha,
        "current_json_git_sha": current_json_git_sha,
        "current_json_manifest_hash": artifact.manifest_hash,
        "current_json_model_artifact_hash": artifact.model_artifact_hash,
        "current_json_pointer_uri": config.private_model_manifest_uri,
        "current_json_manifest_uri": artifact.manifest_uri,
        "current_json_matches_repo_head": current_json_matches_repo_head,
        "active_model_git_sha": active_git_sha,
        "active_model_manifest_hash": active_manifest_hash,
        "active_model_artifact_hash": str(active_row.get("model_artifact_hash") or ""),
        "active_model_version_id": str(active_row.get("private_model_version_id") or ""),
        "active_manifest_uri": str(active_row.get("private_model_manifest_uri") or ""),
        "active_manifest_matches_current_json": active_manifest_matches_current_json,
        "active_is_repo_head": active_is_repo_head,
        "active_version_count": len(active_rows),
        "benchmark_blocked_reason": "" if active_is_repo_head else status,
    }


async def _load_repo_head_current_manifest(
    config: ResearchLabGatewayConfig,
    *,
    repo_main_sha: str,
    wait_for_repo_head: bool,
    wait_timeout_seconds: int,
    poll_seconds: int,
) -> tuple[PrivateModelArtifactManifest, dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + max(0, wait_timeout_seconds)
    attempts = 0
    while True:
        attempts += 1
        artifact = await asyncio.to_thread(_load_valid_artifact, config.private_model_manifest_uri)
        current_json_git_sha = str(artifact.git_commit_sha or "")
        if _git_sha_matches(current_json_git_sha, repo_main_sha):
            return artifact, {
                "status": "current_json_matches_repo_head",
                "attempts": attempts,
                "current_json_git_sha": current_json_git_sha,
            }
        if not wait_for_repo_head or asyncio.get_running_loop().time() >= deadline:
            raise RepoHeadManifestNotReadyError(
                repo_main_sha=repo_main_sha,
                current_json_git_sha=current_json_git_sha,
            )
        logger.warning(
            "research_lab_repo_head_manifest_waiting repo_main=%s current_json=%s attempts=%s poll=%ss",
            repo_main_sha[:12],
            current_json_git_sha[:12],
            attempts,
            poll_seconds,
        )
        await asyncio.sleep(poll_seconds)


async def sync_active_model_to_repo_head(
    config: ResearchLabGatewayConfig,
    *,
    actor_ref: str = "maintenance",
    dry_run: bool = True,
    wait_for_repo_head: bool = False,
    wait_timeout_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> dict[str, Any]:
    """Make the active lineage row point at private repo main's current.json.

    This is the safety gate for daily rebenchmark: GitHub main is the source of
    truth for source, S3 current.json is the source of truth for the immutable
    built/signed image. If they disagree, do not benchmark stale lineage.
    """

    action = "sync-active-model-to-repo-head"
    if not repo_head_sync_enabled():
        return {
            "ok": True,
            "action": action,
            "dry_run": dry_run,
            "status": "disabled",
            "active_is_repo_head": None,
            "disabled_env": REPO_HEAD_SYNC_ENABLED_ENV,
        }
    if not config.private_repo_url:
        return {
            "ok": False,
            "action": action,
            "dry_run": dry_run,
            "status": "private_repo_not_configured",
            "active_is_repo_head": False,
        }
    branch_name = str(config.private_repo_branch or "main")
    try:
        repo_main_sha = await asyncio.to_thread(
            _resolve_private_repo_head_sha,
            repo_url=config.private_repo_url,
            branch_name=branch_name,
        )
    except Exception as exc:
        return {
            "ok": False,
            "action": action,
            "dry_run": dry_run,
            "status": "repo_head_unavailable",
            "repo_branch": branch_name,
            "active_is_repo_head": False,
            "error": _safe_text(str(exc))[:200],
        }
    try:
        current_artifact, manifest_status = await _load_repo_head_current_manifest(
            config,
            repo_main_sha=repo_main_sha,
            wait_for_repo_head=wait_for_repo_head,
            wait_timeout_seconds=(
                _repo_head_manifest_wait_seconds() if wait_timeout_seconds is None else wait_timeout_seconds
            ),
            poll_seconds=poll_seconds or _repo_head_manifest_poll_seconds(),
        )
    except RepoHeadManifestNotReadyError as exc:
        return {
            "ok": False,
            "action": action,
            "dry_run": dry_run,
            "status": "repo_head_manifest_not_ready",
            "repo_branch": branch_name,
            "repo_main_sha": repo_main_sha,
            "current_json_git_sha": exc.current_json_git_sha,
            "active_is_repo_head": False,
            "benchmark_blocked_reason": "repo_head_manifest_not_ready",
            "manifest_uri": config.private_model_manifest_uri,
        }
    except Exception as exc:
        return {
            "ok": False,
            "action": action,
            "dry_run": dry_run,
            "status": "current_json_unavailable",
            "repo_branch": branch_name,
            "repo_main_sha": repo_main_sha,
            "active_is_repo_head": False,
            "error": _safe_text(str(exc))[:200],
        }

    active_rows = await _active_private_model_version_rows()
    if len(active_rows) > 1:
        return {
            "ok": False,
            "action": action,
            "dry_run": dry_run,
            "status": "duplicate_active_versions",
            "repo_branch": branch_name,
            "repo_main_sha": repo_main_sha,
            "current_json_git_sha": current_artifact.git_commit_sha,
            "active_is_repo_head": False,
            "active_version_ids": [
                str(row.get("private_model_version_id") or "") for row in active_rows
            ],
        }
    active_row = active_rows[0] if active_rows else None
    active_model_git_sha = str(active_row.get("git_commit_sha") or "") if active_row else ""
    active_model_manifest_hash = (
        str(active_row.get("private_model_manifest_hash") or "") if active_row else ""
    )
    active_is_repo_head = bool(
        active_row
        and _git_sha_matches(active_model_git_sha, repo_main_sha)
        and active_model_manifest_hash == current_artifact.manifest_hash
    )
    planned = {
        "repo_branch": branch_name,
        "repo_main_sha": repo_main_sha,
        "current_json_git_sha": current_artifact.git_commit_sha,
        "current_json_manifest_hash": current_artifact.manifest_hash,
        "current_json_model_artifact_hash": current_artifact.model_artifact_hash,
        "current_json_image_ref_hash": _image_ref_hash(current_artifact.image_digest),
        "current_json_pointer_uri": config.private_model_manifest_uri,
        "current_json_manifest_uri": current_artifact.manifest_uri,
        "active_model_git_sha": active_model_git_sha,
        "active_model_manifest_hash": active_model_manifest_hash,
        "active_model_artifact_hash": str(active_row.get("model_artifact_hash") or "") if active_row else "",
        "active_model_version_id": str(active_row.get("private_model_version_id") or "") if active_row else "",
        "active_manifest_uri": str(active_row.get("private_model_manifest_uri") or "") if active_row else "",
        "active_is_repo_head": active_is_repo_head,
        "manifest_status": manifest_status,
    }
    if active_is_repo_head:
        return {
            "ok": True,
            "action": action,
            "dry_run": dry_run,
            "status": "active_is_repo_head",
            **planned,
        }
    if dry_run:
        return {
            "ok": True,
            "action": action,
            "dry_run": True,
            "status": "would_sync_active_model_to_repo_head",
            **planned,
        }

    if active_row:
        await create_private_model_version_event(
            private_model_version_id=str(active_row["private_model_version_id"]),
            event_type="superseded",
            version_status="superseded",
            reason="superseded_by_repo_head_sync",
            event_doc={
                "source": "repo_head_sync",
                "actor_ref": actor_ref,
                "repo_branch": branch_name,
                "repo_main_sha": repo_main_sha,
                "current_json_manifest_hash": current_artifact.manifest_hash,
                "current_json_pointer_uri": config.private_model_manifest_uri,
                "current_json_manifest_uri": current_artifact.manifest_uri,
                "previous_active_git_sha": active_model_git_sha,
                "previous_active_manifest_hash": active_model_manifest_hash,
            },
        )
    try:
        version_row, _event = await create_private_model_version(
            artifact_manifest=current_artifact.to_dict(),
            manifest_uri=current_artifact.manifest_uri,
            redacted_version_doc=_private_model_version_doc(
                source="repo_head_sync",
                actor_ref=actor_ref,
                repo_branch=branch_name,
                repo_main_sha=repo_main_sha,
                current_json_pointer_uri=config.private_model_manifest_uri,
                previous_active_model_version_id=(
                    str(active_row.get("private_model_version_id") or "") if active_row else ""
                ),
                previous_active_git_sha=active_model_git_sha,
                previous_active_manifest_hash=active_model_manifest_hash,
                artifact=current_artifact,
            ),
            version_status="active",
            reason="repo_head_sync",
        )
    except Exception:
        if active_row:
            try:
                await create_private_model_version_event(
                    private_model_version_id=str(active_row["private_model_version_id"]),
                    event_type="active",
                    version_status="active",
                    reason="repo_head_sync_create_failed_reactivate_previous",
                    event_doc={
                        "source": "repo_head_sync",
                        "actor_ref": actor_ref,
                        "repo_branch": branch_name,
                        "repo_main_sha": repo_main_sha,
                    },
                )
            except Exception:
                logger.exception(
                    "research_lab_repo_head_sync_previous_reactivate_failed version=%s",
                    _short_ref(active_row.get("private_model_version_id")),
                )
        raise
    logger.warning(
        "research_lab_repo_head_sync_active_model_synced repo_main=%s old_active=%s new_version=%s new_manifest=%s",
        repo_main_sha[:12],
        _short_ref(active_model_manifest_hash),
        _short_ref(version_row.get("private_model_version_id")),
        _short_ref(current_artifact.manifest_hash),
    )
    return {
        "ok": True,
        "action": action,
        "dry_run": False,
        "status": "synced_active_model_to_repo_head",
        "private_model_version_id": str(version_row.get("private_model_version_id") or ""),
        **planned,
        "active_is_repo_head": True,
    }


async def wait_for_current_manifest_git_sha(
    config: ResearchLabGatewayConfig,
    *,
    expected_git_sha: str,
    timeout_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> tuple[PrivateModelArtifactManifest | None, dict[str, Any]]:
    """Wait for configured current.json to point at the expected private repo SHA."""

    expected = str(expected_git_sha or "").strip()
    if not expected:
        return None, {"status": "expected_git_sha_missing"}
    deadline = asyncio.get_running_loop().time() + max(
        0,
        _repo_head_manifest_wait_seconds()
        if timeout_seconds is None
        else int(timeout_seconds),
    )
    poll = poll_seconds or _repo_head_manifest_poll_seconds()
    attempts = 0
    last_status: dict[str, Any] = {}
    while True:
        attempts += 1
        try:
            artifact = await asyncio.to_thread(_load_valid_artifact, config.private_model_manifest_uri)
        except Exception as exc:
            last_status = {
                "status": "current_json_unavailable",
                "attempts": attempts,
                "expected_git_sha": expected,
                "error": _safe_text(str(exc))[:200],
            }
        else:
            current_sha = str(artifact.git_commit_sha or "")
            if _git_sha_matches(current_sha, expected):
                return artifact, {
                    "status": "manifest_ready",
                    "attempts": attempts,
                    "expected_git_sha": expected,
                    "current_json_git_sha": current_sha,
                    "current_json_manifest_hash": artifact.manifest_hash,
                    "current_json_model_artifact_hash": artifact.model_artifact_hash,
                    "current_json_image_ref_hash": _image_ref_hash(artifact.image_digest),
                    "manifest_uri": config.private_model_manifest_uri,
                }
            last_status = {
                "status": "source_pushed_manifest_pending",
                "attempts": attempts,
                "expected_git_sha": expected,
                "current_json_git_sha": current_sha,
                "current_json_manifest_hash": artifact.manifest_hash,
                "current_json_model_artifact_hash": artifact.model_artifact_hash,
                "manifest_uri": config.private_model_manifest_uri,
            }
        if asyncio.get_running_loop().time() >= deadline:
            return None, last_status
        logger.warning(
            "research_lab_private_source_manifest_waiting expected=%s current=%s attempts=%s poll=%ss",
            expected[:12],
            str(last_status.get("current_json_git_sha") or "")[:12],
            attempts,
            poll,
        )
        await asyncio.sleep(poll)


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
            benchmark_bridge_status: dict[str, Any] = {"status": "not_attempted"}
            if not (
                isinstance(private_source_status, Mapping)
                and str(private_source_status.get("status") or "") == "failed"
            ):
                benchmark_bridge_status = await self._maybe_finalize_missing_promoted_benchmark_bridge(
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
                "benchmark_bridge_status": benchmark_bridge_status,
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
        activation_artifact = new_artifact
        activation_manifest_uri = new_artifact.manifest_uri
        manifest_wait_status: dict[str, Any] = {"status": "not_required"}
        source_push_status = str(private_repo_result.get("status") or "")
        if source_push_status in {"pushed", "already_applied", "private_source_pushed"}:
            pushed_commit_sha = str(private_repo_result.get("git_commit_sha") or "")
            activation_artifact, manifest_wait_status = await wait_for_current_manifest_git_sha(
                self.config,
                expected_git_sha=pushed_commit_sha,
            )
            if activation_artifact is None:
                await create_candidate_promotion_event(
                    candidate_id=str(candidate["candidate_id"]),
                    source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                    event_type="promotion_checked",
                    promotion_status="checked",
                    active_parent_artifact_hash=active_parent,
                    candidate_parent_artifact_hash=candidate_parent,
                    rolling_window_hash=rolling_window_hash,
                    improvement_points=improvement_points,
                    threshold_points=threshold,
                    worker_ref=self.worker_ref,
                    event_doc=_db_safe_doc({
                        "reason": "source_pushed_manifest_pending",
                        "candidate_status_preserved": "scored",
                        "private_source_status": private_repo_result,
                        "manifest_wait_status": manifest_wait_status,
                        "action": "leave_previous_active_model_active_until_current_json_matches_pushed_commit",
                    }),
                )
                logger.warning(
                    "research_lab_private_source_manifest_pending candidate=%s expected_git=%s status=%s",
                    _short_ref(candidate["candidate_id"]),
                    pushed_commit_sha[:12],
                    manifest_wait_status.get("status"),
                )
                return {
                    "status": "source_pushed_manifest_pending",
                    "private_source_status": private_repo_result,
                    "manifest_wait_status": manifest_wait_status,
                }
            activation_manifest_uri = activation_artifact.manifest_uri
        benchmark_bridge = await self._create_promoted_candidate_benchmark_bridge(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            new_artifact=activation_artifact,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        source_benchmark_bundle_id = (
            str(benchmark_bridge.get("benchmark_bundle_id") or "")
            if isinstance(benchmark_bridge, Mapping)
            else ""
        ) or None
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
                artifact_manifest=activation_artifact.to_dict(),
                manifest_uri=activation_manifest_uri,
                source_candidate_id=str(candidate["candidate_id"]),
                source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
                source_benchmark_bundle_id=source_benchmark_bundle_id,
                redacted_version_doc=_private_model_version_doc(
                    source="gateway_code_edit_image_build_repo_head_manifest",
                    candidate_source_diff_hash=candidate.get("candidate_source_diff_hash"),
                    scored_candidate_model_artifact_hash=new_artifact.model_artifact_hash,
                    scored_candidate_manifest_hash=new_artifact.manifest_hash,
                    private_source_status=private_repo_result,
                    manifest_wait_status=manifest_wait_status,
                    source_benchmark_bundle_id=source_benchmark_bundle_id,
                    artifact=activation_artifact,
                ),
                version_status="active",
                reason="research_lab_image_build_candidate_repo_head_manifest_promoted",
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
            event_doc=_db_safe_doc({
                "new_model_artifact_hash": activation_artifact.model_artifact_hash,
                "new_private_model_manifest_hash": activation_artifact.manifest_hash,
                "new_git_commit_sha": activation_artifact.git_commit_sha,
                "new_image_ref_hash": _image_ref_hash(activation_artifact.image_digest),
                "scored_candidate_model_artifact_hash": new_artifact.model_artifact_hash,
                "scored_candidate_manifest_hash": new_artifact.manifest_hash,
                "private_source_status": private_repo_result,
                "manifest_wait_status": manifest_wait_status,
                "candidate_kind": "image_build",
                "derived_benchmark_bundle_id": source_benchmark_bundle_id,
                "derived_public_report_id": (
                    str(benchmark_bridge.get("public_report_id") or "")
                    if isinstance(benchmark_bridge, Mapping)
                    else ""
                ),
                "source_score_bundle_id": str(score_bundle_row["score_bundle_id"]),
            }),
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
            "benchmark_bridge_status": benchmark_bridge,
            **reward_status,
        }

    async def _create_promoted_candidate_benchmark_bridge(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        score_bundle: Mapping[str, Any],
        new_artifact: PrivateModelArtifactManifest,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
    ) -> dict[str, Any]:
        gate = (
            score_bundle.get("private_holdout_gate")
            if isinstance(score_bundle.get("private_holdout_gate"), Mapping)
            else None
        )
        if not isinstance(gate, Mapping):
            return {"status": "skipped_no_private_holdout_gate"}
        if str(gate.get("decision") or "") != "private_holdout_approved" or not bool(
            gate.get("private_holdout_evaluated")
        ):
            return {"status": "skipped_private_holdout_not_approved"}
        source_candidate_artifact_hash = str(score_bundle.get("candidate_artifact_hash") or "")
        activation_differs_from_scored_candidate = bool(
            source_candidate_artifact_hash
            and source_candidate_artifact_hash != new_artifact.model_artifact_hash
        )

        baseline_bundle_id = str(gate.get("baseline_benchmark_bundle_id") or "")
        if not baseline_bundle_id:
            raise RuntimeError("promoted benchmark bridge missing source baseline benchmark bundle id")
        baseline_row = await select_one(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,benchmark_date,private_model_artifact_hash,"
                "private_model_manifest_hash,rolling_window_hash,evaluation_epoch,"
                "benchmark_attempt,benchmark_quality,aggregate_score,score_summary_doc,"
                "current_benchmark_status,created_at"
            ),
            filters=(("benchmark_bundle_id", baseline_bundle_id),),
        )
        if not isinstance(baseline_row, Mapping):
            raise RuntimeError(f"promoted benchmark bridge source baseline not found: {baseline_bundle_id}")
        if str(baseline_row.get("current_benchmark_status") or "") != "completed":
            raise RuntimeError(
                "promoted benchmark bridge source baseline is not completed: "
                f"{baseline_bundle_id}"
            )
        baseline_doc = (
            baseline_row.get("score_summary_doc")
            if isinstance(baseline_row.get("score_summary_doc"), Mapping)
            else None
        )
        if not isinstance(baseline_doc, Mapping):
            raise RuntimeError("promoted benchmark bridge source baseline summary missing")
        source_window = str(baseline_row.get("rolling_window_hash") or "")
        if source_window and rolling_window_hash and source_window != rolling_window_hash:
            raise RuntimeError("promoted benchmark bridge rolling window mismatch")

        candidate_total = _optional_float(gate.get("candidate_total_score"))
        if candidate_total is None:
            raise RuntimeError("promoted benchmark bridge missing candidate total score")
        benchmark_date = str(baseline_row.get("benchmark_date") or "")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", benchmark_date):
            raise RuntimeError("promoted benchmark bridge source benchmark date missing")

        benchmark_attempt = 0
        existing_benchmark = await self._existing_promoted_benchmark_row(
            benchmark_date=benchmark_date,
            private_model_manifest_hash=new_artifact.manifest_hash,
            rolling_window_hash=rolling_window_hash,
            benchmark_attempt=benchmark_attempt,
        )
        per_icp_summaries = _candidate_per_icp_summaries_from_score_bundle(
            score_bundle=score_bundle,
            baseline_score_summary_doc=baseline_doc,
        )
        if not per_icp_summaries:
            raise RuntimeError("promoted benchmark bridge score bundle has no per-ICP results")
        visibility_split = _candidate_visibility_split_from_baseline(
            baseline_score_summary_doc=baseline_doc,
            per_icp_summaries=per_icp_summaries,
            rolling_window_hash=rolling_window_hash,
        )
        promotion_metric = promotion_improvement_metric(score_bundle).event_doc()
        source_score_bundle_id = str(score_bundle_row.get("score_bundle_id") or "")
        score_summary_doc = {
            "schema_version": "1.0",
            "benchmark_quality": "passed",
            "benchmark_attempt": benchmark_attempt,
            "rolling_window_hash": rolling_window_hash,
            "per_icp_summaries": per_icp_summaries,
            "visibility_split": visibility_split,
            "aggregate_score": round(float(candidate_total), 6),
            "source": "promoted_candidate_score_bundle",
            "derived_from_candidate_score": True,
            "source_score_bundle_id": source_score_bundle_id,
            "source_candidate_id": str(candidate["candidate_id"]),
            "source_baseline_benchmark_bundle_id": baseline_bundle_id,
            "source_baseline_score_summary_hash": canonical_hash(baseline_doc),
            "source_candidate_artifact_hash": source_candidate_artifact_hash,
            "activation_model_artifact_hash": new_artifact.model_artifact_hash,
            "activation_manifest_hash": new_artifact.manifest_hash,
            "activation_git_commit_sha": new_artifact.git_commit_sha,
            "activation_artifact_differs_from_scored_candidate": (
                activation_differs_from_scored_candidate
            ),
            "reimbursement_preserved": True,
            "supersedes_until_daily_rebenchmark": True,
            "promotion_metric": promotion_metric,
        }

        if existing_benchmark is not None:
            benchmark = existing_benchmark
            benchmark_status = "already_exists"
        else:
            summary_hash = canonical_hash(score_summary_doc)
            signature_ref = await asyncio.to_thread(
                sign_digest_with_kms,
                key_id=self.config.score_bundle_kms_key_id,
                digest_hash=summary_hash,
                signature_uri_prefix=self.config.score_bundle_signature_uri_prefix,
            )
            benchmark, _benchmark_event = await create_private_model_benchmark_bundle(
                benchmark_date=benchmark_date,
                private_model_artifact_hash=new_artifact.model_artifact_hash,
                private_model_manifest_hash=new_artifact.manifest_hash,
                rolling_window_hash=rolling_window_hash,
                evaluation_epoch=int(
                    score_bundle.get("evaluation_epoch")
                    or baseline_row.get("evaluation_epoch")
                    or self.config.evaluation_epoch
                    or 0
                ),
                benchmark_attempt=benchmark_attempt,
                benchmark_quality="passed",
                aggregate_score=float(candidate_total),
                scoring_worker_ref=self.worker_ref,
                proxy_ref_hash=None,
                signature_ref=signature_ref,
                score_summary_doc=score_summary_doc,
            )
            benchmark_status = "created"

        public_report = await self._create_or_reuse_promoted_public_report(
            benchmark_date=benchmark_date,
            benchmark_bundle_id=str(benchmark["benchmark_bundle_id"]),
            new_artifact=new_artifact,
            rolling_window_hash=rolling_window_hash,
            aggregate_score=float(candidate_total),
            benchmark_attempt=benchmark_attempt,
            per_icp_summaries=per_icp_summaries,
            visibility_split=visibility_split,
            baseline_bundle_id=baseline_bundle_id,
            source_score_bundle_id=source_score_bundle_id,
            source_candidate_id=str(candidate["candidate_id"]),
            source_candidate_artifact_hash=source_candidate_artifact_hash,
            activation_model_artifact_hash=new_artifact.model_artifact_hash,
            activation_manifest_hash=new_artifact.manifest_hash,
            activation_git_commit_sha=new_artifact.git_commit_sha,
            activation_artifact_differs_from_scored_candidate=(
                activation_differs_from_scored_candidate
            ),
            promotion_metric=promotion_metric,
        )
        return {
            "status": benchmark_status,
            "benchmark_bundle_id": str(benchmark["benchmark_bundle_id"]),
            "public_report_id": str(public_report.get("report_id") or ""),
            "source_score_bundle_id": source_score_bundle_id,
            "source_baseline_benchmark_bundle_id": baseline_bundle_id,
            "aggregate_score": round(float(candidate_total), 6),
        }

    async def _existing_promoted_benchmark_row(
        self,
        *,
        benchmark_date: str,
        private_model_manifest_hash: str,
        rolling_window_hash: str,
        benchmark_attempt: int,
    ) -> dict[str, Any] | None:
        rows = await select_many(
            "research_lab_private_model_benchmark_current",
            columns=(
                "benchmark_bundle_id,benchmark_date,private_model_manifest_hash,"
                "rolling_window_hash,benchmark_attempt,current_benchmark_status,aggregate_score"
            ),
            filters=(
                ("benchmark_date", benchmark_date),
                ("private_model_manifest_hash", private_model_manifest_hash),
                ("rolling_window_hash", rolling_window_hash),
                ("benchmark_attempt", benchmark_attempt),
            ),
            order_by=(("created_at", True),),
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        if str(row.get("current_benchmark_status") or "") != "completed":
            raise RuntimeError("promoted benchmark bridge found non-completed existing benchmark row")
        return row

    async def _create_or_reuse_promoted_public_report(
        self,
        *,
        benchmark_date: str,
        benchmark_bundle_id: str,
        new_artifact: PrivateModelArtifactManifest,
        rolling_window_hash: str,
        aggregate_score: float,
        benchmark_attempt: int,
        per_icp_summaries: Sequence[Mapping[str, Any]],
        visibility_split: Mapping[str, Any],
        baseline_bundle_id: str,
        source_score_bundle_id: str,
        source_candidate_id: str,
        source_candidate_artifact_hash: str,
        activation_model_artifact_hash: str,
        activation_manifest_hash: str,
        activation_git_commit_sha: str,
        activation_artifact_differs_from_scored_candidate: bool,
        promotion_metric: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing_reports = await select_many(
            "research_lab_public_benchmark_report_current",
            columns=(
                "report_id,benchmark_bundle_id,benchmark_date,private_model_manifest_hash,"
                "rolling_window_hash,benchmark_attempt,current_report_status,aggregate_score"
            ),
            filters=(
                ("benchmark_date", benchmark_date),
                ("private_model_manifest_hash", new_artifact.manifest_hash),
                ("rolling_window_hash", rolling_window_hash),
                ("benchmark_attempt", benchmark_attempt),
            ),
            order_by=(("created_at", True),),
            limit=1,
        )
        if existing_reports:
            row = existing_reports[0]
            if str(row.get("current_report_status") or "") != "published":
                raise RuntimeError("promoted benchmark bridge found non-published existing public report row")
            return row

        source_public_doc: dict[str, Any] | None = None
        source_reports = await select_many(
            "research_lab_public_benchmark_report_current",
            columns="report_id,benchmark_bundle_id,report_doc,current_report_status,created_at",
            filters=(("benchmark_bundle_id", baseline_bundle_id), ("current_report_status", "published")),
            order_by=(("created_at", True),),
            limit=1,
        )
        if source_reports:
            raw_doc = source_reports[0].get("report_doc")
            if isinstance(raw_doc, Mapping):
                source_public_doc = dict(raw_doc)
        report_doc = _build_promoted_public_benchmark_report_doc(
            benchmark_date=benchmark_date,
            rolling_window_hash=rolling_window_hash,
            aggregate_score=aggregate_score,
            per_icp_summaries=per_icp_summaries,
            visibility_split=visibility_split,
            source_public_report_doc=source_public_doc,
            source_score_bundle_id=source_score_bundle_id,
            source_candidate_id=source_candidate_id,
            source_baseline_benchmark_bundle_id=baseline_bundle_id,
            source_candidate_artifact_hash=source_candidate_artifact_hash,
            activation_model_artifact_hash=activation_model_artifact_hash,
            activation_manifest_hash=activation_manifest_hash,
            activation_git_commit_sha=activation_git_commit_sha,
            activation_artifact_differs_from_scored_candidate=(
                activation_artifact_differs_from_scored_candidate
            ),
            promotion_metric=promotion_metric,
        )
        report, _report_event = await create_public_benchmark_report(
            benchmark_date=benchmark_date,
            benchmark_bundle_id=benchmark_bundle_id,
            private_model_artifact_hash=new_artifact.model_artifact_hash,
            private_model_manifest_hash=new_artifact.manifest_hash,
            rolling_window_hash=rolling_window_hash,
            aggregate_score=aggregate_score,
            benchmark_attempt=benchmark_attempt,
            benchmark_quality="passed",
            report_doc=report_doc,
        )
        return report

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

    async def _maybe_finalize_missing_promoted_benchmark_bridge(
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
        bridge = await self._create_promoted_candidate_benchmark_bridge(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
            new_artifact=new_artifact,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
        )
        await self._record_promoted_benchmark_bridge_event(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            active=active,
            candidate_parent=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold=threshold,
            bridge=bridge,
        )
        return bridge

    async def _record_promoted_benchmark_bridge_event(
        self,
        *,
        candidate: Mapping[str, Any],
        score_bundle_row: Mapping[str, Any],
        active: ActivePrivateModel,
        candidate_parent: str,
        rolling_window_hash: str,
        improvement_points: float,
        threshold: float,
        bridge: Mapping[str, Any],
    ) -> None:
        reason = "promoted_candidate_benchmark_bridge_completed"
        existing_events = await select_many(
            "research_lab_candidate_promotion_events",
            columns="promotion_event_id,event_doc,created_at",
            filters=(
                ("candidate_id", str(candidate["candidate_id"])),
                ("source_score_bundle_id", str(score_bundle_row["score_bundle_id"])),
                ("event_type", "promotion_checked"),
            ),
            order_by=(("created_at", True),),
            limit=100,
        )
        for event in existing_events:
            doc = event.get("event_doc") if isinstance(event.get("event_doc"), Mapping) else {}
            if (
                str(doc.get("reason") or "") == reason
                and str(doc.get("derived_benchmark_bundle_id") or "")
                == str(bridge.get("benchmark_bundle_id") or "")
            ):
                return
        await create_candidate_promotion_event(
            candidate_id=str(candidate["candidate_id"]),
            source_score_bundle_id=str(score_bundle_row["score_bundle_id"]),
            event_type="promotion_checked",
            promotion_status="checked",
            active_parent_artifact_hash=active.artifact.model_artifact_hash,
            candidate_parent_artifact_hash=candidate_parent,
            rolling_window_hash=rolling_window_hash,
            improvement_points=improvement_points,
            threshold_points=threshold,
            worker_ref=self.worker_ref,
            event_doc={
                "reason": reason,
                "derived_benchmark_bundle_id": str(bridge.get("benchmark_bundle_id") or ""),
                "derived_public_report_id": str(bridge.get("public_report_id") or ""),
                "source_score_bundle_id": str(score_bundle_row["score_bundle_id"]),
                "source_baseline_benchmark_bundle_id": str(
                    bridge.get("source_baseline_benchmark_bundle_id") or ""
                ),
                "bridge_status": str(bridge.get("status") or ""),
            },
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


def _candidate_per_icp_summaries_from_score_bundle(
    *,
    score_bundle: Mapping[str, Any],
    baseline_score_summary_doc: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregates = score_bundle.get("aggregates") if isinstance(score_bundle.get("aggregates"), Mapping) else {}
    raw_results = aggregates.get("per_icp_results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes, bytearray)):
        return []
    results_by_ref: dict[str, Mapping[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("icp_ref") or item.get("icp_hash") or "").strip()
        if ref and ref not in results_by_ref:
            results_by_ref[ref] = item
    if not results_by_ref:
        return []

    baseline_by_ref = _baseline_summary_by_ref(baseline_score_summary_doc)
    ordered_refs = _baseline_visibility_refs(baseline_score_summary_doc)
    for ref in baseline_by_ref:
        if ref not in ordered_refs:
            ordered_refs.append(ref)
    for ref in results_by_ref:
        if ref not in ordered_refs:
            ordered_refs.append(ref)

    summaries: list[dict[str, Any]] = []
    for ref in ordered_refs:
        result = results_by_ref.get(ref)
        if result is None:
            continue
        baseline = baseline_by_ref.get(ref, {})
        scores = _candidate_score_values(result)
        diagnostics = _candidate_result_diagnostics(result, scores=scores)
        summaries.append(
            {
                "icp_ref": ref,
                "icp_hash": str(result.get("icp_hash") or baseline.get("icp_hash") or ""),
                "score": round(_candidate_icp_score(scores), 6),
                "company_count": len(scores),
                "industry": _summary_bucket_value(baseline, "industry"),
                "sub_industry": _summary_bucket_value(baseline, "sub_industry"),
                "country": _summary_bucket_value(baseline, "country"),
                "geography_bucket": _summary_bucket_value(baseline, "geography_bucket"),
                "company_size_bucket": _summary_bucket_value(baseline, "company_size_bucket"),
                "intent_category_bucket": _summary_bucket_value(baseline, "intent_category_bucket"),
                "diagnostics": diagnostics,
            }
        )
    return summaries


def _baseline_summary_by_ref(doc: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    summaries = doc.get("per_icp_summaries")
    if isinstance(summaries, Sequence) and not isinstance(summaries, (str, bytes, bytearray)):
        for item in summaries:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or "").strip()
            if ref and ref not in rows:
                rows[ref] = item
    return rows


def _baseline_visibility_refs(doc: Mapping[str, Any]) -> list[str]:
    split = doc.get("visibility_split") if isinstance(doc.get("visibility_split"), Mapping) else {}
    items = split.get("items") if isinstance(split.get("items"), Sequence) else []
    refs: list[str] = []
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or "").strip()
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def _summary_bucket_value(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None and key == "country":
        value = summary.get("geography_bucket")
    if value is None and key == "geography_bucket":
        value = summary.get("country")
    text = str(value or "").strip()
    return text or "unspecified"


def _candidate_score_values(result: Mapping[str, Any]) -> list[float]:
    scores = result.get("candidate_company_scores")
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray)):
        return []
    values: list[float] = []
    for item in scores:
        try:
            values.append(float(item or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _candidate_icp_score(scores: Sequence[float]) -> float:
    return float(sum(float(item or 0.0) for item in scores) / len(scores)) if scores else 0.0


def _candidate_result_diagnostics(result: Mapping[str, Any], *, scores: Sequence[float]) -> dict[str, Any]:
    failure_categories = _candidate_failure_categories(
        str(result.get("failure_reason") or ""),
        provider_excluded=bool(result.get("provider_excluded")),
        score_count=len(scores),
    )
    scored = len(scores)
    sourcing_failed = scored <= 0 or bool(result.get("provider_excluded"))
    return {
        "failure_categories": failure_categories,
        "avg_icp_fit": 0.0,
        "avg_intent_signal_final": 0.0,
        "sourcing_failed": sourcing_failed,
        "funnel": {
            "sourced": scored,
            "fit_pass": scored,
            "verified": scored,
            "intent_valid": scored,
            "scored": scored,
        },
        "per_signal": {},
        "evidence_types": {},
        "rejection_reasons": {},
    }


def _candidate_failure_categories(
    reason: str,
    *,
    provider_excluded: bool,
    score_count: int,
) -> list[str]:
    categories: set[str] = set()
    if provider_excluded:
        categories.add("runtime_provider_error")
    for token in [item.strip().lower() for item in str(reason or "").split(";") if item.strip()]:
        if "provider" in token or "http_4" in token or "http_5" in token:
            categories.add("runtime_provider_error")
        elif "timeout" in token:
            categories.add("runtime_timeout")
        elif "zero_scoreable" in token:
            categories.add("zero_scoreable_companies")
        elif "zero_companies" in token:
            categories.add("zero_company_results")
        elif "parse" in token or "json" in token:
            categories.add("parser_error")
        elif "runtime" in token or "adapter" in token:
            categories.add("model_runtime_error")
        else:
            categories.add("other_scoring_failure")
    if score_count <= 0:
        categories.add("zero_company_results")
    return sorted(categories)


def _candidate_visibility_split_from_baseline(
    *,
    baseline_score_summary_doc: Mapping[str, Any],
    per_icp_summaries: Sequence[Mapping[str, Any]],
    rolling_window_hash: str,
) -> dict[str, Any]:
    score_by_ref = {
        str(item.get("icp_ref") or ""): float(item.get("score") or 0.0)
        for item in per_icp_summaries
        if str(item.get("icp_ref") or "").strip()
    }
    split = (
        baseline_score_summary_doc.get("visibility_split")
        if isinstance(baseline_score_summary_doc.get("visibility_split"), Mapping)
        else {}
    )
    raw_items = split.get("items") if isinstance(split.get("items"), Sequence) else []
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes, bytearray)):
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("icp_ref") or "")
            updated = dict(item)
            updated["item_rank"] = int(item.get("item_rank") or index)
            updated["score"] = round(float(score_by_ref.get(ref, item.get("score") or 0.0)), 6)
            items.append(updated)
    if not items:
        for index, summary in enumerate(per_icp_summaries, start=1):
            items.append(
                {
                    "icp_ref": str(summary.get("icp_ref") or ""),
                    "icp_hash": str(summary.get("icp_hash") or ""),
                    "set_id": 0,
                    "day_index": 0,
                    "day_rank": index,
                    "item_rank": index,
                    "score": round(float(summary.get("score") or 0.0), 6),
                    "visibility": "summary_only",
                    "strength_label": "unknown",
                }
            )
    public_count = sum(1 for item in items if str(item.get("visibility") or "") == "public")
    private_count = sum(1 for item in items if str(item.get("visibility") or "") == "private")
    return {
        "schema_version": str(split.get("schema_version") or "1.0"),
        "split_policy": str(split.get("split_policy") or "source_baseline_visibility_split"),
        "rolling_window_hash": rolling_window_hash,
        "public_count": public_count,
        "private_count": private_count,
        "public_strength_counts": dict(split.get("public_strength_counts") or {}),
        "private_strength_counts": dict(split.get("private_strength_counts") or {}),
        "items": items,
    }


def _build_promoted_public_benchmark_report_doc(
    *,
    benchmark_date: str,
    rolling_window_hash: str,
    aggregate_score: float,
    per_icp_summaries: Sequence[Mapping[str, Any]],
    visibility_split: Mapping[str, Any],
    source_public_report_doc: Mapping[str, Any] | None,
    source_score_bundle_id: str,
    source_candidate_id: str,
    source_baseline_benchmark_bundle_id: str,
    source_candidate_artifact_hash: str,
    activation_model_artifact_hash: str,
    activation_manifest_hash: str,
    activation_git_commit_sha: str,
    activation_artifact_differs_from_scored_candidate: bool,
    promotion_metric: Mapping[str, Any],
) -> dict[str, Any]:
    report = build_public_benchmark_report(
        benchmark_date=benchmark_date,
        rolling_window_hash=rolling_window_hash,
        aggregate_score=aggregate_score,
        per_icp_summaries=per_icp_summaries,
        benchmark_items=(),
    )
    summary_by_ref = {
        str(item.get("icp_ref") or ""): item
        for item in per_icp_summaries
        if str(item.get("icp_ref") or "").strip()
    }
    public_icps = _promoted_public_icps_from_source(
        source_public_report_doc=source_public_report_doc,
        summary_by_ref=summary_by_ref,
    )
    public_count = int(visibility_split.get("public_count") or len(public_icps))
    private_count = int(visibility_split.get("private_count") or 0)
    report.update(
        {
            "source": "promoted_candidate_score_bundle",
            "derived_from_candidate_score": True,
            "source_score_bundle_id": source_score_bundle_id,
            "source_candidate_id": source_candidate_id,
            "source_baseline_benchmark_bundle_id": source_baseline_benchmark_bundle_id,
            "source_candidate_artifact_hash": source_candidate_artifact_hash,
            "activation_model_artifact_hash": activation_model_artifact_hash,
            "activation_manifest_hash": activation_manifest_hash,
            "activation_git_commit_sha": activation_git_commit_sha,
            "activation_artifact_differs_from_scored_candidate": (
                activation_artifact_differs_from_scored_candidate
            ),
            "supersedes_until_daily_rebenchmark": True,
            "promotion_metric": dict(promotion_metric),
            "public_icps": public_icps,
            "public_icp_count": public_count,
            "private_holdout_icp_count": private_count,
            "visibility_split": _public_visibility_split_from_private(visibility_split),
            "icp_buckets": _promoted_public_bucket_rows(
                per_icp_summaries=per_icp_summaries,
                visibility_split=visibility_split,
            ),
        }
    )
    report.pop("report_public_hash", None)
    if contains_secret_material(report):
        raise ValueError("promoted candidate public benchmark report contains forbidden material")
    report["report_public_hash"] = sha256_json(report)
    return report


def _promoted_public_icps_from_source(
    *,
    source_public_report_doc: Mapping[str, Any] | None,
    summary_by_ref: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(source_public_report_doc, Mapping):
        return []
    raw_public_icps = source_public_report_doc.get("public_icps")
    if not isinstance(raw_public_icps, Sequence) or isinstance(raw_public_icps, (str, bytes, bytearray)):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_public_icps:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("icp_ref") or "")
        summary = summary_by_ref.get(ref)
        updated = dict(item)
        if summary:
            diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), Mapping) else {}
            updated["score"] = round(float(summary.get("score") or 0.0), 6)
            updated["company_count"] = int(summary.get("company_count") or 0)
            updated["diagnostics"] = {
                **dict(updated.get("diagnostics") or {}),
                "failure_categories": list(diagnostics.get("failure_categories") or []),
                "avg_icp_fit": float(diagnostics.get("avg_icp_fit") or 0.0),
                "avg_intent_signal_final": float(diagnostics.get("avg_intent_signal_final") or 0.0),
            }
        rows.append(updated)
    return sorted(rows, key=lambda row: int(row.get("item_rank") or 0))


def _public_visibility_split_from_private(visibility_split: Mapping[str, Any]) -> dict[str, Any]:
    items = visibility_split.get("items") if isinstance(visibility_split.get("items"), Sequence) else []
    safe_items: list[dict[str, Any]] = []
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            safe_items.append(
                {
                    "item_rank": int(item.get("item_rank") or len(safe_items) + 1),
                    "icp_ref": str(item.get("icp_ref") or ""),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "set_id": int(item.get("set_id") or 0),
                    "day_index": int(item.get("day_index") or 0),
                    "day_rank": int(item.get("day_rank") or 0),
                    "score": round(float(item.get("score") or 0.0), 6),
                    "visibility": str(item.get("visibility") or "summary_only"),
                    "strength_label": str(item.get("strength_label") or "unknown"),
                }
            )
    return {
        "schema_version": str(visibility_split.get("schema_version") or "1.0"),
        "split_policy": str(visibility_split.get("split_policy") or ""),
        "rolling_window_hash": str(visibility_split.get("rolling_window_hash") or ""),
        "public_count": int(visibility_split.get("public_count") or 0),
        "private_count": int(visibility_split.get("private_count") or 0),
        "public_strength_counts": dict(visibility_split.get("public_strength_counts") or {}),
        "private_strength_counts": dict(visibility_split.get("private_strength_counts") or {}),
        "items": safe_items,
    }


def _promoted_public_bucket_rows(
    *,
    per_icp_summaries: Sequence[Mapping[str, Any]],
    visibility_split: Mapping[str, Any],
) -> list[dict[str, Any]]:
    split_by_ref: dict[str, Mapping[str, Any]] = {}
    raw_items = visibility_split.get("items") if isinstance(visibility_split.get("items"), Sequence) else []
    if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes, bytearray)):
        for item in raw_items:
            if isinstance(item, Mapping):
                split_by_ref[str(item.get("icp_ref") or "")] = item
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(per_icp_summaries, start=1):
        ref = str(summary.get("icp_ref") or "")
        split = split_by_ref.get(ref, {})
        diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), Mapping) else {}
        score = float(summary.get("score") or 0.0)
        company_count = int(summary.get("company_count") or 0)
        rows.append(
            {
                "item_rank": int(split.get("item_rank") or index),
                "icp_ref": ref,
                "visibility": str(split.get("visibility") or "summary_only"),
                "strength_label": str(split.get("strength_label") or "unknown"),
                "industry_bucket": _summary_bucket_value(summary, "industry"),
                "sub_industry_bucket": _summary_bucket_value(summary, "sub_industry"),
                "geography_bucket": _summary_bucket_value(summary, "geography_bucket"),
                "company_size_bucket": _summary_bucket_value(summary, "company_size_bucket"),
                "intent_category_bucket": _summary_bucket_value(summary, "intent_category_bucket"),
                "score_band": _score_band(score),
                "company_count_band": _count_band(company_count),
                "failure_categories": list(diagnostics.get("failure_categories") or [])
                if str(split.get("visibility") or "") == "public"
                else [],
            }
        )
    return sorted(rows, key=lambda row: int(row.get("item_rank") or 0))


def _score_band(score: float) -> str:
    if score >= 80:
        return "80_plus"
    if score >= 60:
        return "60_79"
    if score >= 40:
        return "40_59"
    if score > 0:
        return "1_39"
    return "zero"


def _count_band(count: int) -> str:
    if count <= 0:
        return "zero"
    if count <= 2:
        return "1_2"
    if count <= 5:
        return "3_5"
    return "6_plus"


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


def _image_ref_hash(image_ref: Any) -> str:
    """Hash an immutable image ref without storing the ECR URI/digest string."""

    value = str(image_ref or "")
    return canonical_hash({"private_model_image_ref": value}) if value else ""


_DB_DOC_FORBIDDEN_RE = re.compile(
    r"(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|"
    r"private_repo|judge_prompt|hidden_icp|icp_plaintext|intent_signals|"
    r"\.dkr\.ecr\.|image_digest|private_model_manifest_doc|"
    r"candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)",
    re.IGNORECASE,
)


def _db_safe_doc(value: Any) -> Any:
    """Return a JSON value safe for constrained Research Lab audit columns."""

    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, raw_nested in value.items():
            key = _db_safe_key(raw_key)
            if not key:
                continue
            safe[key] = _db_safe_doc(raw_nested)
        return safe
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_db_safe_doc(item) for item in value]
    if isinstance(value, str) and _DB_DOC_FORBIDDEN_RE.search(value):
        return canonical_hash({"value": value})
    return value


def _db_safe_key(value: Any) -> str:
    key = str(value or "")
    # The DB checks reject the literal key substring image_digest. Preserve the
    # lineage meaning using a hash-only image_ref_hash key.
    key = re.sub(r"image[_-]?digest", "image_ref_hash", key, flags=re.IGNORECASE)
    return "" if _DB_DOC_FORBIDDEN_RE.search(key) else key


def _private_model_version_doc(*, artifact: PrivateModelArtifactManifest, **extra: Any) -> dict[str, Any]:
    """Build DB-safe private model version metadata.

    The Supabase CHECK constraints intentionally reject raw ECR image refs and
    keys containing ``image_digest``. The signed private manifest remains the
    source of truth for the immutable image; lineage/event docs store only
    hashes and non-secret version metadata.
    """

    doc = {
        **extra,
        "model_artifact_hash": artifact.model_artifact_hash,
        "private_model_manifest_hash": artifact.manifest_hash,
        "git_commit_sha": artifact.git_commit_sha,
        "component_registry_version": artifact.component_registry_version,
        "scoring_adapter_version": artifact.scoring_adapter_version,
        "image_ref_hash": _image_ref_hash(artifact.image_digest),
    }
    if "image_digest" in doc:
        doc.pop("image_digest", None)
    return _db_safe_doc(doc)


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
